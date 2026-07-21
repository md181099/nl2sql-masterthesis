#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import PrefixTuningConfig, get_peft_model
from config import get_param, get_section, load_config
from logging_utils import setup_logging

# Import robust machen (je nachdem wo llm_client.py liegt)
try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient


logger = logging.getLogger(__name__)
IM_ASSISTANT_START_RE = re.compile(r"<\|im_start\|>assistant(?:\r?\n)?", re.IGNORECASE)
ASSISTANT_TAG_RE = re.compile(r"<\|assistant\|>(?:\r?\n)?", re.IGNORECASE)


def find_response_start(text: str) -> int:
    """
    Locate where the answer starts so prompt tokens can be masked in labels.
    Priority:
    1) Qwen chat assistant marker
    2) Legacy assistant marker
    3) Last ```sql block
    4) SQL: marker
    """
    matches = list(IM_ASSISTANT_START_RE.finditer(text))
    if matches:
        return matches[-1].end()

    matches = list(ASSISTANT_TAG_RE.finditer(text))
    if matches:
        return matches[-1].end()

    idx = text.rfind("```sql")
    if idx != -1:
        return idx

    idx = text.rfind("SQL:")
    if idx != -1:
        return idx + len("SQL:")

    return 0


def parse_bool(value: str) -> bool:
    value_norm = value.strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional JSON config path")
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log_format",
        default="text",
        choices=["text", "json"],
        help="Logging format: text (default) or json",
    )
    parser.add_argument("--llm", default=None, help="LLM key from LLMClient (e.g., llama32_1b)")
    parser.add_argument(
        "--dataset_path",
        "--train_sft_path",
        dest="dataset_path",
        default=None,
        help="Path to train_sft.jsonl",
    )
    parser.add_argument("--output_dir", default=None, help="Adapter output directory")
    parser.add_argument("--num_virtual_tokens", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument(
        "--per_device_train_batch_size",
        "--batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--grad_accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit number of loaded training samples (for local smoke tests)",
    )
    parser.add_argument("--prefix_projection", type=parse_bool, default=None)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=None)
    parser.add_argument("--overwrite_output_dir", type=parse_bool, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)
    cfg = load_config(args_cli.config) if args_cli.config else {}

    llm = get_param(args_cli, cfg, "llm", "llama32_1b")
    # Backward-compatible: allow legacy config key "train_sft_path".
    dataset_cfg = dict(cfg)
    if "dataset_path" not in dataset_cfg and "train_sft_path" in dataset_cfg:
        dataset_cfg["dataset_path"] = dataset_cfg["train_sft_path"]
    dataset_path = get_param(args_cli, dataset_cfg, "dataset_path", "data/train_sft.jsonl")
    default_output_dir = f"adapters/{llm}/prefix_sql"
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
    gradient_checkpointing = bool(get_param(args_cli, cfg, "gradient_checkpointing", False))
    overwrite_output_dir = bool(get_param(args_cli, cfg, "overwrite_output_dir", False))
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 5))
    save_steps = int(get_param(args_cli, cfg, "save_steps", 50))
    save_total_limit = int(get_param(args_cli, cfg, "save_total_limit", 2))

    # Support both top-level and nested "prefix" config keys.
    prefix_section = get_section(cfg, "prefix")
    prefix_cfg = dict(prefix_section)
    if "num_virtual_tokens" not in prefix_cfg and "num_virtual_tokens" in cfg:
        prefix_cfg["num_virtual_tokens"] = cfg["num_virtual_tokens"]
    if "prefix_projection" not in prefix_cfg and "prefix_projection" in cfg:
        prefix_cfg["prefix_projection"] = cfg["prefix_projection"]

    num_virtual_tokens = int(
        get_param(args_cli, prefix_cfg, "num_virtual_tokens", 20, config_name="num_virtual_tokens")
    )
    prefix_projection = bool(
        get_param(args_cli, prefix_cfg, "prefix_projection", True, config_name="prefix_projection")
    )

    client = LLMClient(project_root)

    train_path = Path(dataset_path)
    if not train_path.is_absolute():
        train_path = project_root / train_path
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    if out_dir.exists():
        if out_dir.is_file():
            raise ValueError(f"Output path exists as file, expected directory: {out_dir}")
        has_files = any(out_dir.iterdir())
        if has_files and not overwrite_output_dir:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {out_dir}. "
                "Set overwrite_output_dir=true to allow reuse."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path} (run 02_make_sft_dataset.py first)")

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    logger.info("Prefix-Tuning setup")
    logger.info("dataset_path=%s", train_path)
    logger.info("llm=%s", llm)
    logger.info("output_adapter=%s", out_dir)
    logger.info("max_length=%d", max_length)
    logger.info("per_device_train_batch_size=%d", per_device_train_batch_size)
    logger.info("gradient_accumulation_steps=%d", gradient_accumulation_steps)
    logger.info("effective_batch_size=%d", effective_batch_size)
    logger.info("gradient_checkpointing=%s", gradient_checkpointing)
    logger.info("num_virtual_tokens=%d", num_virtual_tokens)
    logger.info("prefix_projection=%s", prefix_projection)

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
    # 2) Load tokenizer + base model (ONLY ONCE!)
    # -----------------------
    model_id = client.get_model_id(llm)
    logger.info("Loading tokenizer/model: %s", model_id)

    tokenizer = client.get_tokenizer(llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = client.get_base_model(llm)
    if gradient_checkpointing:
        if not hasattr(base_model, "gradient_checkpointing_enable"):
            raise RuntimeError("gradient_checkpointing=true but model does not support gradient_checkpointing_enable()")
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    # -----------------------
    # 3) Attach Prefix Tuning adapter
    # -----------------------
    prefix_cfg = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=prefix_projection,
    )

    model = get_peft_model(base_model, prefix_cfg)
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("trainable_params=%d", trainable_params)
    logger.info("total_params=%d", total_params)
    logger.info("trainable_ratio=%.6f", trainable_params / total_params if total_params else 0.0)

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
    # 7) Save Prefix adapter
    # -----------------------
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved Prefix-Tuning adapter to: %s", out_dir)


if __name__ == "__main__":
    main()
