#!/usr/bin/env python
"""
Train a persona-specific LoRA adapter on the BluePrint dataset.

Typical usage (run inside your HF-enabled virtual env):

python agent-trainning/train_persona_lora.py \
  --cluster-id 7 \
  --dataset-name ComplexDataLab/BluePrint \
  --dataset-config 50_clusters \
  --output-dir /scratch/$USER/Qwen/Qwen2.5-7B-Instruct-lora-cluster07
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import torch

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BluePrint cluster LoRA.")
    parser.add_argument(
        "--cluster-id",
        type=int,
        required=True,
        help="Which of the 50 BluePrint clusters to train on (0-49).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for adapter checkpoints (one per cluster).",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base causal LM checkpoint on Hugging Face.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable SFTTrainer sequence packing (recommended).",
    )
    return parser.parse_args()


def load_blueprint_cluster(cluster_id: int) -> Dataset:
    """Loads the BluePrint 25-cluster dataset and filters to a given cluster."""
    dataset = load_dataset("ComplexDataLab/BluePrint", "25_clusters", split="full")
    if "cluster_id" not in dataset.column_names or "thread" not in dataset.column_names:
        raise ValueError(
            "Expected columns 'cluster_id' and 'thread' in BluePrint dataset."
        )
    LOGGER.info("Loaded %d total threads", len(dataset))
    persona_rows = dataset.filter(lambda row: row["cluster_id"] == cluster_id)
    if not len(persona_rows):
        raise ValueError(f"No threads found for cluster {cluster_id}")
    LOGGER.info(
        "Cluster %s threads: %d (%.2f%% of corpus)",
        cluster_id,
        len(persona_rows),
        100 * len(persona_rows) / len(dataset),
    )
    split = persona_rows.train_test_split(test_size=0.1, seed=1234)
    return split["train"], split["test"]


def format_thread(thread: List[Dict[str, Any]], cluster_id: int) -> str:
    """Flattens a BluePrint thread into newline-delimited text."""
    lines: List[str] = [f"[Cluster {cluster_id}]"]
    for idx, message in enumerate(thread):
        text = (message.get("text") or "").strip()
        if not text:
            continue
        user_hash = message.get("user_id", "anon")[:8]
        lines.append(f"{idx:02d} [{user_hash}] {text}")
    return "\n".join(lines)


def build_text_dataset(dataset: Dataset, cluster_id: int) -> Dataset:
    LOGGER.info("Formatting %d threads into text", len(dataset))
    formatted = dataset.map(
        lambda row: {"text": format_thread(row["thread"], cluster_id)},
        remove_columns=dataset.column_names,
    )
    return formatted


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_split, eval_split = load_blueprint_cluster(args.cluster_id)

    train_dataset = build_text_dataset(train_split, args.cluster_id)
    eval_dataset = build_text_dataset(eval_split, args.cluster_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=device,
            trust_remote_code=True,
        )

    if "qwen" in args.base_model.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["c_attn", "c_proj"]

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        bf16=use_cuda,
        fp16=False,
        weight_decay=0.0,
        max_steps=-1,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    LOGGER.info("Finished training cluster %s -> %s", args.cluster_id, output_path)


if __name__ == "__main__":
    main()
