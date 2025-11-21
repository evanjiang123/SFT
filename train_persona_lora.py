#!/usr/bin/env python
"""
Train a persona-specific LoRA adapter on the BluePrint dataset.

Typical usage (run inside your HF-enabled virtual env):

python agent_trainning/train_persona_lora.py \
  --cluster-id 7 \
  --cluster-file /path/to/cluster_7.jsonl \
  --output-dir /scratch/$USER/Qwen/Qwen2.5-7B-Instruct-lora-cluster07
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)  # periodic safety dump

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from accelerate import find_executable_batch_size
import torch

# Disable tokenizer multi-thread spam (HuggingFace warning + weird HPC behavior)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Line-buffered stdout/stderr so SLURM flushes messages promptly
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    # Older Python may not have reconfigure()
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BluePrint cluster LoRA.")
    parser.add_argument(
        "--cluster-id",
        type=int,
        required=True,
        help="Which of the BluePrint clusters to train on (e.g., 0-24 or 0-49).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for adapter checkpoints (one per cluster).",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base causal LM checkpoint on Hugging Face or local path.",
    )
    parser.add_argument(
        "--cluster-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a local cluster JSONL file. "
            "If provided, the script loads data from this file instead of Hugging Face."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument(
        "--packing",
        action="store_true",
        help="(Ignored) Kept for CLI compatibility; packing is not used in this script.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=4,
        help="Number of processes to use when tokenizing the dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    return parser.parse_args()


def load_local_cluster(cluster_id: int, cluster_path: Path) -> Tuple[Dataset, Dataset]:
    """Load a cluster from a local JSONL file with a 'thread' field."""
    LOGGER.info("Loading cluster %s from local file %s", cluster_id, cluster_path)
    rows: List[Dict[str, Any]] = []
    with cluster_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Cluster file {cluster_path} is empty.")

    dataset = Dataset.from_list(rows)
    if "thread" not in dataset.column_names:
        raise ValueError(
            f"Expected column 'thread' in cluster file {cluster_path}, "
            f"found columns {dataset.column_names}"
        )

    LOGGER.info("Local cluster threads: %d", len(dataset))
    split = dataset.train_test_split(test_size=0.01, seed=1234)
    eval_split = split["test"]
    if len(eval_split) > 5000:
        eval_split = eval_split.select(range(5000))
        LOGGER.info("Downsampled eval set to 5000 examples.")
    return split["train"], eval_split


def load_blueprint_cluster(
    cluster_id: int, cluster_file: Path | None = None
) -> Tuple[Dataset, Dataset]:
    """Loads the BluePrint dataset, optionally from a local file."""
    if cluster_file is not None:
        if cluster_file.is_dir():
            raise ValueError("cluster_file should point to a JSONL file, not a folder.")
        return load_local_cluster(cluster_id, cluster_file)

    LOGGER.info("Loading BluePrint dataset from Hugging Face hub...")
    dataset = load_dataset("ComplexDataLab/BluePrint", "25_clusters", split="full")
    if "cluster_id" not in dataset.column_names or "thread" not in dataset.column_names:
        raise ValueError(
            "Expected columns 'cluster_id' and 'thread' in BluePrint dataset; "
            f"found columns {dataset.column_names}"
        )

    LOGGER.info("Loaded %d total threads from BluePrint.", len(dataset))
    persona_rows = dataset.filter(lambda row: row["cluster_id"] == cluster_id)
    if not len(persona_rows):
        raise ValueError(f"No threads found for cluster {cluster_id}")

    LOGGER.info(
        "Cluster %s threads: %d (%.2f%% of corpus)",
        cluster_id,
        len(persona_rows),
        100.0 * len(persona_rows) / len(dataset),
    )
    split = persona_rows.train_test_split(test_size=0.1, seed=1234)
    return split["train"], split["test"]


def format_thread(thread: List[Dict[str, Any]], cluster_id: int) -> str:
    """
    Flattens a BluePrint thread into newline-delimited text:

    [Cluster k]
    00 [userhash] message
    01 [userhash] reply
    ...
    """
    lines: List[str] = [f"[Cluster {cluster_id}]"]
    for idx, message in enumerate(thread):
        text = (message.get("text") or "").strip()
        if not text:
            continue
        user_hash = (message.get("user_id") or "anon")[:8]
        lines.append(f"{idx:02d} [{user_hash}] {text}")
    return "\n".join(lines)


def build_text_dataset(dataset: Dataset, cluster_id: int) -> Dataset:
    """Convert a dataset with 'thread' into a text-only dataset with 'text' column."""
    LOGGER.info("Formatting %d threads into flat text.", len(dataset))
    formatted = dataset.map(
        lambda row: {"text": format_thread(row["thread"], cluster_id)},
        remove_columns=dataset.column_names,
        num_proc=1,  # formatting is cheap
    )
    LOGGER.info("Resulting text dataset has %d examples.", len(formatted))
    return formatted


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving outputs to %s", output_path)

    LOGGER.info("Loading cluster %s...", args.cluster_id)
    train_split, eval_split = load_blueprint_cluster(
        args.cluster_id, cluster_file=args.cluster_file
    )
    LOGGER.info(
        "Train split size: %d | Eval split size: %d",
        len(train_split),
        len(eval_split),
    )

    train_dataset = build_text_dataset(train_split, args.cluster_id)
    eval_dataset = build_text_dataset(eval_split, args.cluster_id)

    LOGGER.info("Loading tokenizer from base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Set pad_token to eos_token for tokenizer.")

    tokenizer.model_max_length = args.max_length

    LOGGER.info("Tokenizing datasets with max_length=%d", args.max_length)

    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        result = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        # For causal LM, labels are just input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=["text"],
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.dataset_num_proc,
        remove_columns=["text"],
    )

    LOGGER.info("Loading base model in bfloat16: %s", args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # you can optionally set cache_dir=os.environ.get("HF_HOME")
    )

    # LoRA config: similar to Aurélien's Qwen config
    if "qwen" in args.base_model.lower():
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        target_modules = ["c_attn", "c_proj"]

    LOGGER.info("Configuring LoRA with targets: %s", target_modules)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    LOGGER.info("Wrapping base model with LoRA adapter...")
    model = get_peft_model(model, lora_config)

    # Match Aurélien's safety tricks
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception as e:
        LOGGER.warning("Could not enable input grads: %s", e)

    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception as e:
        LOGGER.warning("Could not set use_cache=False: %s", e)

    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception as e:
        LOGGER.warning("Could not enable gradient checkpointing: %s", e)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "Trainable params (LoRA): %d / %d total",
        trainable_params,
        all_params,
    )

    data_collator = default_data_collator

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_training_loop(batch_size: int):
        LOGGER.info("Using per_device_train_batch_size=%d", batch_size)
        training_args = TrainingArguments(
            output_dir=str(output_path),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy="no",  # to match older transformers on CC
            save_steps=500,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=True,
            bf16=True,  # A100 supports this
            ddp_find_unused_parameters=False,
            seed=args.seed,
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        LOGGER.info("Starting training for cluster %s...", args.cluster_id)
        trainer.train()

        LOGGER.info("Saving final adapter and tokenizer to %s", output_path)
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path / "tokenizer"))
        LOGGER.info(
            "Finished training cluster %s -> %s",
            args.cluster_id,
            output_path,
        )

    inner_training_loop()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        LOGGER.exception("Fatal error in train_persona_lora", exc_info=e)
        print("\n\n=== PYTHON EXCEPTION CAUGHT IN train_persona_lora ===", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise