#!/usr/bin/env python

"""
Final fixed version — matches Aurelien’s Qwen-LoRA training behavior:

✔ Subsamples to 30k BEFORE any mapping
✔ Caps eval at 5k
✔ No faulthandler timeout
✔ No massive 578k → flat-text → tokenize step
✔ Uses the same LoRA config
✔ Safe for Narval kernel 4.18
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
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

# ----------------------------- Logging -----------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Flush stdout/stderr line-by-line for SLURM
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# ----------------------------- CLI -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a BluePrint cluster LoRA.")
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cluster-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


# ----------------------------- DATA LOADING -----------------------------
MAX_TRAIN = 30000      # same as Aurélien
MAX_EVAL = 5000        # same as Aurélien


def load_local_cluster(cluster_id: int, cluster_path: Path):
    """Load JSONL cluster file containing {"thread": [...] }"""

    LOGGER.info("Loading local cluster: %s", cluster_path)
    rows = []

    with cluster_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Local cluster file is empty: {cluster_path}")

    dataset = Dataset.from_list(rows)
    LOGGER.info("Original local dataset size: %d", len(dataset))

    # ---------- AGGRESSIVE SUBSAMPLING (AURELIEN STYLE) ----------
    if len(dataset) > MAX_TRAIN:
        dataset = dataset.shuffle(seed=1234).select(range(MAX_TRAIN))
        LOGGER.info("Subsampled train to %d", MAX_TRAIN)

    split = dataset.train_test_split(test_size=0.01, seed=1234)
    eval_set = split["test"]

    if len(eval_set) > MAX_EVAL:
        eval_set = eval_set.select(range(MAX_EVAL))
        LOGGER.info("Eval capped to %d", MAX_EVAL)

    return split["train"], eval_set


def load_blueprint_cluster(cluster_id: int, cluster_file: Path | None):
    """Either load from local JSONL or from HF BluePrint."""

    if cluster_file is not None:
        return load_local_cluster(cluster_id, cluster_file)

    LOGGER.info("Loading BluePrint dataset from HF…")
    dataset = load_dataset("ComplexDataLab/BluePrint", "25_clusters", split="full")

    persona_rows = dataset.filter(lambda r: r["cluster_id"] == cluster_id)
    if len(persona_rows) == 0:
        raise ValueError(f"No threads for cluster {cluster_id}")

    LOGGER.info("Full persona rows before subsampling: %d", len(persona_rows))

    if len(persona_rows) > MAX_TRAIN:
        persona_rows = persona_rows.shuffle(seed=1234).select(range(MAX_TRAIN))
        LOGGER.info("Subsampled persona rows to %d", MAX_TRAIN)

    split = persona_rows.train_test_split(test_size=0.1, seed=1234)
    eval_set = split["test"]

    if len(eval_set) > MAX_EVAL:
        eval_set = eval_set.select(range(MAX_EVAL))

    return split["train"], eval_set


# ----------------------------- THREAD → TEXT FORMAT -----------------------------
def format_thread(thread: List[Dict[str, Any]], cluster_id: int):
    """Convert a full thread into newline-based text."""
    lines = [f"[Cluster {cluster_id}]"]
    for idx, item in enumerate(thread):
        txt = (item.get("text") or "").strip()
        if not txt:
            continue
        uid = (item.get("user_id") or "anon")[:8]
        lines.append(f"{idx:02d} [{uid}] {txt}")
    return "\n".join(lines)


def build_text_dataset(dataset: Dataset, cluster_id: int):
    """Flatten BluePrint into text-only dataset."""
    LOGGER.info("Formatting %d threads", len(dataset))
    return dataset.map(
        lambda row: {"text": format_thread(row["thread"], cluster_id)},
        remove_columns=dataset.column_names,
        num_proc=1,
    )


# ----------------------------- MAIN -----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output dir: %s", out_path)

    # ---------- LOAD DATA ----------
    train_split, eval_split = load_blueprint_cluster(
        args.cluster_id, cluster_file=args.cluster_file
    )

    LOGGER.info("Train=%d  Eval=%d", len(train_split), len(eval_split))

    train_text = build_text_dataset(train_split, args.cluster_id)
    eval_text = build_text_dataset(eval_split, args.cluster_id)

    # ---------- TOKENIZER ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = args.max_length

    def tokenize(batch):
        res = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        res["labels"] = res["input_ids"].copy()
        return res

    token_train = train_text.map(
        tokenize, batched=True, num_proc=args.dataset_num_proc, remove_columns=["text"]
    )
    token_eval = eval_text.map(
        tokenize, batched=True, num_proc=args.dataset_num_proc, remove_columns=["text"]
    )

    # ---------- MODEL + LORA ----------
    LOGGER.info("Loading base model on device CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_cfg)

    # Aurelien's required flags
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ---------- TRAINING LOOP ----------
    data_collator = default_data_collator

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def fit(batch_size: int):
        LOGGER.info("Using train batch_size=%d", batch_size)

        targs = TrainingArguments(
            output_dir=str(out_path),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            evaluation_strategy="no",
            save_steps=500,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=50,
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=token_train,
            eval_dataset=token_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(str(out_path))
        tokenizer.save_pretrained(out_path / "tokenizer")

    fit()


if __name__ == "__main__":
    main()
