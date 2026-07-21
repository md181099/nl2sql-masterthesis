#!/usr/bin/env python3
from __future__ import annotations

"""
Separate Prefix-Tuning SFT training pipeline using TRL SFTTrainer with packing.

This script is intentionally separate from:
- src/07_lora_finetune_packing.py: final LoRA packing reference path.
- src/08_prefix_tune.py: older non-packing Prefix-Tuning path.

The goal is a Prefix-Tuning setup that is methodically close to the LoRA packing
pipeline while keeping the existing final LoRA pipeline unchanged.
"""

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PrefixTuningConfig
from transformers import TrainingArguments

from config import get_param, get_section, load_config
from logging_utils import setup_logging

try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient


logger = logging.getLogger(__name__)

TRL_MISSING_MESSAGE = "TRL is required for packing training. Install with: pip install trl"


def parse_bool(value: str) -> bool:
    value_norm = value.strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _supports_kwarg(init_params: dict[str, inspect.Parameter], name: str) -> bool:
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in init_params.values())
    return name in init_params or has_var_kwargs


def _ensure_text_field(dataset_obj: Any, dataset_text_field: str) -> None:
    if dataset_text_field not in dataset_obj.column_names:
        raise ValueError(
            f"Dataset is missing required text field '{dataset_text_field}'. "
            f"Available fields: {dataset_obj.column_names}"
        )


def _count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_params / total_params if total_params else 0.0
    return trainable_params, total_params, trainable_ratio


def _resolve_prefix_config(
    cfg: dict[str, Any],
    args_cli: argparse.Namespace,
) -> tuple[PrefixTuningConfig, dict[str, Any]]:
    prefix_section = get_section(cfg, "prefix")
    prefix_cfg = dict(prefix_section)
    if "num_virtual_tokens" not in prefix_cfg and "num_virtual_tokens" in cfg:
        prefix_cfg["num_virtual_tokens"] = cfg["num_virtual_tokens"]
    if "prefix_projection" not in prefix_cfg and "prefix_projection" in cfg:
        prefix_cfg["prefix_projection"] = cfg["prefix_projection"]

    num_virtual_tokens = int(
        get_param(args_cli, prefix_cfg, "num_virtual_tokens", 48, config_name="num_virtual_tokens")
    )
    prefix_projection = bool(
        get_param(args_cli, prefix_cfg, "prefix_projection", True, config_name="prefix_projection")
    )
    task_type = str(get_param(args_cli, prefix_cfg, "prefix_task_type", "CAUSAL_LM", config_name="task_type"))

    prefix_config = PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=prefix_projection,
    )
    return prefix_config, {
        "task_type": task_type,
        "num_virtual_tokens": num_virtual_tokens,
        "prefix_projection": prefix_projection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate TRL packing Prefix-Tuning SFT trainer")
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
    parser.add_argument("--llm", default=None, help="LLM key from LLMClient registry")
    parser.add_argument("--dataset_path", default=None, help="Path to JSONL SFT dataset")
    parser.add_argument(
        "--dataset_text_field",
        default=None,
        help="Text field name in dataset (default: text)",
    )
    parser.add_argument("--output_dir", default=None, help="Adapter output directory")
    parser.add_argument("--packing", type=parse_bool, default=None, help="Enable packing")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--num_train_epochs", type=float, default=None)
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
    parser.add_argument("--num_virtual_tokens", type=int, default=None)
    parser.add_argument("--prefix_projection", type=parse_bool, default=None)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=None)
    parser.add_argument("--overwrite_output_dir", type=parse_bool, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _build_training_args(
    training_args_cls: Any,
    output_dir: Path,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    dataset_text_field: str,
    packing: bool,
    max_seq_length: int,
) -> tuple[TrainingArguments, dict[str, bool]]:
    init_params = inspect.signature(training_args_cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    applied_sft_fields = {
        "dataset_text_field": False,
        "packing": False,
        "max_seq_length": False,
    }

    def add_if_supported(name: str, value: Any) -> bool:
        if _supports_kwarg(init_params, name):
            kwargs[name] = value
            return True
        return False

    common_args = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": seed,
    }
    for name, value in common_args.items():
        add_if_supported(name, value)

    applied_sft_fields["dataset_text_field"] = add_if_supported("dataset_text_field", dataset_text_field)
    applied_sft_fields["packing"] = add_if_supported("packing", packing)
    if add_if_supported("max_seq_length", max_seq_length):
        applied_sft_fields["max_seq_length"] = True
    elif add_if_supported("max_length", max_seq_length):
        applied_sft_fields["max_seq_length"] = True

    return training_args_cls(**kwargs), applied_sft_fields


def _build_sft_trainer(
    sft_trainer_cls: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    training_args: TrainingArguments,
    prefix_config: PrefixTuningConfig,
    dataset_text_field: str,
    packing: bool,
    max_seq_length: int,
    sft_fields_from_args: dict[str, bool],
) -> Any:
    init_params = inspect.signature(sft_trainer_cls.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": prefix_config,
    }

    def add_if_supported(name: str, value: Any) -> bool:
        if _supports_kwarg(init_params, name):
            kwargs[name] = value
            return True
        return False

    if not add_if_supported("processing_class", tokenizer):
        add_if_supported("tokenizer", tokenizer)

    dataset_text_field_supported = sft_fields_from_args["dataset_text_field"]
    if not dataset_text_field_supported:
        dataset_text_field_supported = add_if_supported("dataset_text_field", dataset_text_field)

    packing_supported = sft_fields_from_args["packing"]
    if not packing_supported:
        packing_supported = add_if_supported("packing", packing)

    max_seq_length_supported = sft_fields_from_args["max_seq_length"]
    if not max_seq_length_supported:
        if add_if_supported("max_seq_length", max_seq_length):
            max_seq_length_supported = True
        elif add_if_supported("max_length", max_seq_length):
            max_seq_length_supported = True

    if not dataset_text_field_supported:
        logger.warning("Installed TRL does not expose dataset_text_field; relying on TRL defaults.")
    if not packing_supported:
        raise RuntimeError("Installed TRL SFTTrainer/SFTConfig does not expose packing; cannot run packing=true.")
    if not max_seq_length_supported:
        raise RuntimeError("Installed TRL SFTTrainer/SFTConfig does not expose max_seq_length/max_length.")

    return sft_trainer_cls(**kwargs)


def main() -> None:
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)

    try:
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except Exception:
            SFTConfig = None
    except Exception as exc:
        raise RuntimeError(TRL_MISSING_MESSAGE) from exc

    cfg = load_config(args_cli.config) if args_cli.config else {}
    project_root = Path(__file__).resolve().parents[1]

    llm = get_param(args_cli, cfg, "llm", "llama32_1b")
    dataset_path = get_param(args_cli, cfg, "dataset_path", "data/train_sft.jsonl")
    dataset_text_field = str(get_param(args_cli, cfg, "dataset_text_field", "text"))
    output_dir = get_param(args_cli, cfg, "output_dir", f"adapters/{llm}/prefix_sql_packing")
    packing = bool(get_param(args_cli, cfg, "packing", True))
    max_seq_length = int(get_param(args_cli, cfg, "max_seq_length", 512))
    num_train_epochs = float(get_param(args_cli, cfg, "num_train_epochs", 1))
    learning_rate = float(get_param(args_cli, cfg, "learning_rate", 5e-4))
    per_device_train_batch_size = int(get_param(args_cli, cfg, "per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(get_param(args_cli, cfg, "gradient_accumulation_steps", 8))
    gradient_checkpointing = bool(get_param(args_cli, cfg, "gradient_checkpointing", False))
    overwrite_output_dir = bool(get_param(args_cli, cfg, "overwrite_output_dir", False))
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 10))
    save_steps = int(get_param(args_cli, cfg, "save_steps", 50))
    save_total_limit = int(get_param(args_cli, cfg, "save_total_limit", 2))
    seed = int(get_param(args_cli, cfg, "seed", 42))
    max_train_samples_raw = get_param(args_cli, cfg, "max_train_samples", None)
    max_train_samples = int(max_train_samples_raw) if max_train_samples_raw is not None else None
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be >= 1 or null")

    train_path = Path(dataset_path)
    if not train_path.is_absolute():
        train_path = project_root / train_path
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")

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

    dataset_dict = load_dataset("json", data_files={"train": str(train_path)})
    train_dataset = dataset_dict["train"]
    _ensure_text_field(train_dataset, dataset_text_field)
    dataset_size_before = len(train_dataset)
    if max_train_samples is not None:
        max_idx = min(max_train_samples, dataset_size_before)
        train_dataset = train_dataset.select(range(max_idx))
    dataset_size = len(train_dataset)

    client = LLMClient(project_root)
    model_id = client.get_model_id(llm)
    tokenizer = client.get_tokenizer(llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = client.get_base_model(llm)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = False
    logger.info("use_cache=False for prefix tuning")
    if gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("gradient_checkpointing=true but model does not support gradient_checkpointing_enable()")
        model.gradient_checkpointing_enable()

    prefix_config, prefix_meta = _resolve_prefix_config(cfg, args_cli)

    training_args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    training_args, sft_fields_from_args = _build_training_args(
        training_args_cls=training_args_cls,
        output_dir=out_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        dataset_text_field=dataset_text_field,
        packing=packing,
        max_seq_length=max_seq_length,
    )

    trainer = _build_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        prefix_config=prefix_config,
        dataset_text_field=dataset_text_field,
        packing=packing,
        max_seq_length=max_seq_length,
        sft_fields_from_args=sft_fields_from_args,
    )

    trainable_params, total_params, trainable_ratio = _count_trainable_parameters(trainer.model)
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    logger.info("Packing Prefix-Tuning SFT setup")
    logger.info("model_id=%s", model_id)
    logger.info("llm=%s", llm)
    logger.info("dataset_path=%s", train_path)
    logger.info("dataset_size=%d", dataset_size)
    logger.info("dataset_text_field=%s", dataset_text_field)
    logger.info("output_dir=%s", out_dir)
    logger.info("max_seq_length=%d", max_seq_length)
    logger.info("packing=%s", packing)
    logger.info("per_device_train_batch_size=%d", per_device_train_batch_size)
    logger.info("gradient_accumulation_steps=%d", gradient_accumulation_steps)
    logger.info("effective_batch_size=%d", effective_batch_size)
    logger.info("gradient_checkpointing=%s", gradient_checkpointing)
    logger.info("prefix=%s", prefix_meta)
    logger.info("trainable_params=%d", trainable_params)
    logger.info("total_params=%d", total_params)
    logger.info("trainable_ratio=%.6f", trainable_ratio)

    trainer.train()
    trainer.model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved packing Prefix-Tuning adapter to: %s", out_dir)


if __name__ == "__main__":
    main()
