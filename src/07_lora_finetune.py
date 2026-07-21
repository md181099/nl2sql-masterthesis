#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model
from config import get_param, get_section, load_config
from logging_utils import setup_logging

# Import robust machen (je nachdem wo llm_client.py liegt)
try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient


logger = logging.getLogger(__name__)


def find_response_start(text: str) -> int:
    """
    Locate where the answer starts so prompt tokens can be masked in labels.
    Priority:
    1) Last ```sql block (target starts here in this project)
    2) Assistant marker
    3) SQL: marker
    """
    idx = text.rfind("```sql")
    if idx != -1:
        return idx

    marker = "<|assistant|>"
    idx = text.rfind(marker)
    if idx != -1:
        return idx + len(marker)

    idx = text.rfind("SQL:")
    if idx != -1:
        return idx + len("SQL:")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for NL2SQL")
    p.add_argument(
        "--config",
        default=None,
        help="Optional JSON config path",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--log_format",
        default="text",
        choices=["text", "json"],
        help="Logging format: text (default) or json",
    )
    p.add_argument(
        "--llm",
        default=None,
        help="LLM key from LLMClient registry (e.g., llama32_1b)",
    )
    p.add_argument(
        "--num_train_epochs",
        "--epochs",
        dest="num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    p.add_argument(
        "--learning_rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    p.add_argument(
        "--per_device_train_batch_size",
        "--batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=None,
        help="Per-device train batch size",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        "--grad_accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    p.add_argument(
        "--dataset_path",
        "--train_sft_path",
        dest="dataset_path",
        default=None,
        help="Path to train_sft.jsonl",
    )
    p.add_argument("--output_dir", default=None, help="Adapter output directory")
    p.add_argument("--max_length", type=int, default=None, help="Tokenizer max length")
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit number of loaded training samples (for local smoke tests)",
    )
    p.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
    p.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=None,
        help="LoRA target modules list",
    )
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)
    cfg = load_config(args_cli.config) if args_cli.config else {}

    project_root = Path(__file__).resolve().parents[1]
    llm = get_param(args_cli, cfg, "llm", "llama32_1b")
    # Backward-compatible: allow legacy config key "train_sft_path".
    dataset_cfg = dict(cfg)
    if "dataset_path" not in dataset_cfg and "train_sft_path" in dataset_cfg:
        dataset_cfg["dataset_path"] = dataset_cfg["train_sft_path"]
    dataset_path = get_param(args_cli, dataset_cfg, "dataset_path", "data/train_sft.jsonl")
    default_output_dir = f"adapters/{llm}/lora_sql"
    output_dir = get_param(args_cli, cfg, "output_dir", default_output_dir)
    num_train_epochs = int(get_param(args_cli, cfg, "num_train_epochs", 5))
    learning_rate = float(get_param(args_cli, cfg, "learning_rate", 2e-4))
    per_device_train_batch_size = int(
        get_param(args_cli, cfg, "per_device_train_batch_size", 2)
    )
    gradient_accumulation_steps = int(
        get_param(args_cli, cfg, "gradient_accumulation_steps", 8)
    )
    max_length = int(get_param(args_cli, cfg, "max_length", 512))
    max_train_samples_raw = get_param(args_cli, cfg, "max_train_samples", None)
    max_train_samples = (
        int(max_train_samples_raw) if max_train_samples_raw is not None else None
    )
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be >= 1 or null")
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 5))
    save_steps = int(get_param(args_cli, cfg, "save_steps", 50))
    save_total_limit = int(get_param(args_cli, cfg, "save_total_limit", 2))

    # Support both top-level and nested "lora" config keys.
    lora_section = get_section(cfg, "lora")
    lora_cfg = dict(lora_section)
    if "r" not in lora_cfg and "lora_r" in cfg:
        lora_cfg["r"] = cfg["lora_r"]
    if "lora_alpha" not in lora_cfg and "lora_alpha" in cfg:
        lora_cfg["lora_alpha"] = cfg["lora_alpha"]
    if "lora_dropout" not in lora_cfg and "lora_dropout" in cfg:
        lora_cfg["lora_dropout"] = cfg["lora_dropout"]
    if "target_modules" not in lora_cfg and "lora_target_modules" in cfg:
        lora_cfg["target_modules"] = cfg["lora_target_modules"]

    lora_r = int(get_param(args_cli, lora_cfg, "lora_r", 16, config_name="r"))
    lora_alpha = int(
        get_param(args_cli, lora_cfg, "lora_alpha", 32, config_name="lora_alpha")
    )
    lora_dropout = float(
        get_param(args_cli, lora_cfg, "lora_dropout", 0.05, config_name="lora_dropout")
    )
    lora_target_modules = get_param(
        args_cli,
        lora_cfg,
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        config_name="target_modules",
    )
    if isinstance(lora_target_modules, str):
        lora_target_modules = [lora_target_modules]
    if not isinstance(lora_target_modules, list) or not all(
        isinstance(m, str) for m in lora_target_modules
    ):
        raise ValueError("LoRA target_modules must be a list of strings")

    client = LLMClient(project_root)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(dataset_path)
    if not train_path.is_absolute():
        train_path = project_root / train_path

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path} (run 02_make_sft_dataset.py first)")

    # -----------------------
    # 1) Load training data
    # -----------------------
    rows = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total_samples = len(rows)
    if max_train_samples is not None:
        rows = rows[:max_train_samples]
        logger.info(
            f"Loaded {total_samples} training samples from {train_path}; "
            f"using first {len(rows)} (max_train_samples={max_train_samples})"
        )
    else:
        logger.info(
            f"Loaded {total_samples} training samples from {train_path}; "
            f"using all {len(rows)} samples"
        )

    texts = [r["text"] for r in rows]
    ds = Dataset.from_dict({"text": texts})

    # -----------------------
    # 2) Load tokenizer/model
    # -----------------------
    model_id = client.get_model_id(llm)
    logger.info("Loading tokenizer/model: %s", model_id)

    tokenizer = client.get_tokenizer(llm)

    # Some Llama tokenizers don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = client.get_base_model(llm)

    # -----------------------
    # 3) Attach LoRA adapters
    # -----------------------
    # Target both attention + MLP projections for stronger schema adaptation.
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False
    model.print_trainable_parameters()

    # -----------------------
    # 4) Tokenize dataset
    # -----------------------
    def tokenize(batch):
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for text in batch["text"]:
            enc_full = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            # Compute split point in token space using the text prefix.
            resp_start = find_response_start(text)
            prefix_text = text[:resp_start]
            enc_prefix = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            prefix_len = len(enc_prefix["input_ids"])

            labels = list(enc_full["input_ids"])
            # Ignore prompt tokens in loss.
            for i in range(min(prefix_len, len(labels))):
                labels[i] = -100
            # Ignore padded tokens in loss.
            for i, attn in enumerate(enc_full["attention_mask"]):
                if attn == 0:
                    labels[i] = -100

            input_ids_batch.append(enc_full["input_ids"])
            attention_mask_batch.append(enc_full["attention_mask"])
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    # Labels are already provided by tokenize().
    collator = default_data_collator

    # -----------------------
    # 5) Training configuration
    # -----------------------
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # -----------------------
    # 6) Train
    # -----------------------
    trainer.train()

    # -----------------------
    # 7) Save LoRA adapter
    # -----------------------
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved LoRA adapter to: %s", out_dir)


if __name__ == "__main__":
    main()
