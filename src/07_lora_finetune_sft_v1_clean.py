#!/usr/bin/env python3
from __future__ import annotations

"""
Pipeline v1-clean LoRA SFT trainer for NL2SQL.

This is a deliberately clean re-run of the empirically strong v1 setting:
- full ChatML examples in a single text field
- full-chat loss over all non-padding tokens
- real TRL SFTConfig packing with max_length=1024

It reuses the logging, metadata, history, LoRA, packing, and plotting helpers
from the Pipeline v2 trainer without changing the v2 prompt/completion path.
"""

import argparse
import hashlib
import importlib
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from config import get_param, get_section, load_config
from logging_utils import setup_logging

try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient


logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v1_clean_full_chat"
LOSS_MODE = "full_chat_loss"

v2 = importlib.import_module("07_lora_finetune_sft_v2")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline v1-clean full-chat TRL SFTConfig LoRA trainer")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_format", default="text", choices=["text", "json"])
    parser.add_argument("--llm", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--dataset_text_field", default=None)
    parser.add_argument("--eval_dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--continue_from_adapter", default=None)
    parser.add_argument("--packing", type=v2.parse_bool, default=None)
    parser.add_argument("--packing_strategy", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--completion_only_loss", type=v2.parse_bool, default=None)
    parser.add_argument("--assistant_only_loss", type=v2.parse_bool, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", "--batch_size", dest="per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", dest="per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--eval_accumulation_steps", type=int, default=None)
    parser.add_argument("--prediction_loss_only", type=v2.parse_bool, default=None)
    parser.add_argument("--save_strategy", default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--load_best_model_at_end", type=v2.parse_bool, default=None)
    parser.add_argument("--metric_for_best_model", default=None)
    parser.add_argument("--greater_is_better", type=v2.parse_bool, default=None)
    parser.add_argument("--save_best_model", type=v2.parse_bool, default=None)
    parser.add_argument("--auto_resume", type=v2.parse_bool, default=None)
    parser.add_argument("--overwrite_output_dir", type=v2.parse_bool, default=None)
    parser.add_argument("--gradient_checkpointing", type=v2.parse_bool, default=None)
    parser.add_argument("--torch_compile", type=v2.parse_bool, default=None)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=None)
    parser.add_argument("--bf16", type=v2.parse_bool, default=None)
    parser.add_argument("--fp16", type=v2.parse_bool, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fail_on_packing_warning", type=v2.parse_bool, default=None)
    parser.add_argument("--use_flash_attention", type=v2.parse_bool, default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", default=None)
    parser.add_argument("--lora_task_type", default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    return parser.parse_args()


def _ensure_text_dataset(dataset_obj: Any, dataset_text_field: str) -> None:
    if dataset_text_field not in set(dataset_obj.column_names):
        raise ValueError(
            f"Pipeline v1-clean requires text field '{dataset_text_field}'. "
            f"Available fields: {dataset_obj.column_names}"
        )


def _selected_flash_attention_available(
    attn_implementation: str | None,
    availability: dict[str, Any],
) -> bool:
    if attn_implementation == "flash_attention_2":
        return bool(availability.get("flash_attention_2_available"))
    if attn_implementation == "flash_attention_3":
        return bool(availability.get("flash_attention_3_available"))
    if v2._kernel_repo_from_attn_implementation(attn_implementation):
        return bool(availability.get("kernels_installed"))
    return bool(
        availability.get("flash_attention_2_available")
        or availability.get("flash_attention_3_available")
        or availability.get("kernels_installed")
    )


def _verify_full_chat_labels(
    trainer: Any,
    *,
    sample_count: int = 4,
    dataset: Any | None = None,
    dataset_name: str = "train",
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else trainer.train_dataset
    collator = trainer.data_collator
    available = len(dataset)
    if available == 0:
        raise RuntimeError(f"Trainer {dataset_name}_dataset is empty; cannot verify full-chat labels.")

    total_tokens = 0
    trainable_tokens = 0
    masked_tokens = 0
    padding_tokens = 0
    sequence_start_masked_tokens = 0
    checked_samples = min(sample_count, available)
    has_completion_mask = False

    for idx in range(checked_samples):
        example = dataset[idx]
        if "completion_mask" in example:
            has_completion_mask = True
            raise RuntimeError("Full-chat loss path unexpectedly produced completion_mask.")
        batch = collator([example])
        labels = batch["labels"][0].detach().cpu()
        input_ids = batch["input_ids"][0].detach().cpu()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            non_padding = attention_mask[0].detach().cpu().bool()
        else:
            non_padding = torch.ones_like(labels, dtype=torch.bool)
        padding = ~non_padding
        allowed_mask = padding.clone()
        seq_lengths = example.get("seq_lengths")
        if seq_lengths is not None:
            offset = 0
            for seq_len in seq_lengths:
                seq_len_int = int(seq_len)
                if 0 <= offset < labels.numel():
                    allowed_mask[offset] = True
                offset += seq_len_int

        unexpected_mask = (labels == -100) & ~allowed_mask
        if torch.any(unexpected_mask):
            raise RuntimeError(
                "Full-chat labels contain unexpected -100 labels outside padding or packed sequence starts."
            )
        trainable_positions = non_padding & (labels != -100)
        if torch.any(labels[trainable_positions] != input_ids[trainable_positions]):
            raise RuntimeError("Full-chat labels do not match input_ids on non-padding tokens.")
        if torch.any(labels[padding] != -100):
            raise RuntimeError("Padding labels are not masked with -100.")

        total_tokens += int(labels.numel())
        trainable_tokens += int(trainable_positions.sum().item())
        masked_tokens += int((labels == -100).sum().item())
        padding_tokens += int(padding.sum().item())
        sequence_start_masked_tokens += int(((labels == -100) & non_padding).sum().item())

    result = {
        "verified": True,
        "dataset_name": dataset_name,
        "loss_mode": LOSS_MODE,
        "checked_samples": checked_samples,
        "has_completion_mask": has_completion_mask,
        "total_tokens_checked": total_tokens,
        "trainable_tokens_checked": trainable_tokens,
        "masked_tokens_checked": masked_tokens,
        "padding_tokens_checked": padding_tokens,
        "sequence_start_masked_tokens_checked": sequence_start_masked_tokens,
        "trainable_ratio": (trainable_tokens / total_tokens) if total_tokens else 0.0,
        "masked_ratio": (masked_tokens / total_tokens) if total_tokens else 0.0,
    }
    logger.info("Full-chat label audit verified for %s: %s", dataset_name, result)
    return result


class EvalCudaMemoryCleanupCallback(TrainerCallback):
    def _empty_cache_if_evaluating(self, control: Any) -> None:
        if getattr(control, "should_evaluate", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        self._empty_cache_if_evaluating(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._empty_cache_if_evaluating(control)


def main() -> None:
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)

    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError("TRL with SFTConfig/SFTTrainer is required for Pipeline v1-clean.") from exc

    cfg = load_config(args_cli.config) if args_cli.config else {}
    project_root = Path(__file__).resolve().parents[1]

    llm = str(get_param(args_cli, cfg, "llm", "qwen35_9b_base"))
    dataset_text_field = str(get_param(args_cli, cfg, "dataset_text_field", "text"))
    dataset_path_raw = str(
        get_param(
            args_cli,
            cfg,
            "dataset_path",
            "data/sql_create_context/train_sft_qwen35_9b_full_chat_v1_clean_no_spider_dev_overlap.jsonl",
        )
    )
    eval_dataset_path_raw = get_param(args_cli, cfg, "eval_dataset_path", None)
    output_dir_raw = str(
        get_param(
            args_cli,
            cfg,
            "output_dir",
            f"adapters/{llm}/lora_sqlctx_v1_clean_full_chat_packing1024_no_overlap_epochs1",
        )
    )
    continue_from_adapter_raw = get_param(args_cli, cfg, "continue_from_adapter", None)
    packing = bool(get_param(args_cli, cfg, "packing", True))
    packing_strategy = str(get_param(args_cli, cfg, "packing_strategy", "bfd"))
    max_length = int(get_param(args_cli, cfg, "max_length", 1024))
    completion_only_loss = bool(get_param(args_cli, cfg, "completion_only_loss", False))
    assistant_only_loss = bool(get_param(args_cli, cfg, "assistant_only_loss", False))
    num_train_epochs = float(get_param(args_cli, cfg, "num_train_epochs", 1))
    learning_rate = float(get_param(args_cli, cfg, "learning_rate", 1e-4))
    per_device_train_batch_size = int(get_param(args_cli, cfg, "per_device_train_batch_size", 4))
    per_device_eval_batch_size_raw = get_param(args_cli, cfg, "per_device_eval_batch_size", None)
    gradient_accumulation_steps = int(get_param(args_cli, cfg, "gradient_accumulation_steps", 2))
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 10))
    eval_strategy = v2._coerce_eval_strategy(get_param(args_cli, cfg, "eval_strategy", "epoch"))
    eval_steps_raw = get_param(args_cli, cfg, "eval_steps", None)
    eval_steps = int(eval_steps_raw) if eval_steps_raw is not None else None
    eval_accumulation_steps_raw = get_param(args_cli, cfg, "eval_accumulation_steps", None)
    eval_accumulation_steps = (
        int(eval_accumulation_steps_raw) if eval_accumulation_steps_raw is not None else None
    )
    prediction_loss_only_raw = get_param(args_cli, cfg, "prediction_loss_only", None)
    save_strategy = v2._coerce_save_strategy(get_param(args_cli, cfg, "save_strategy", "epoch"))
    save_total_limit_raw = get_param(args_cli, cfg, "save_total_limit", 2)
    save_total_limit = int(save_total_limit_raw) if save_total_limit_raw is not None else None
    load_best_model_at_end = bool(get_param(args_cli, cfg, "load_best_model_at_end", False))
    metric_for_best_model = str(get_param(args_cli, cfg, "metric_for_best_model", "eval_loss"))
    greater_is_better_raw = get_param(args_cli, cfg, "greater_is_better", None)
    greater_is_better = (
        bool(greater_is_better_raw) if greater_is_better_raw is not None else False
    )
    save_best_model = bool(get_param(args_cli, cfg, "save_best_model", True))
    auto_resume = bool(get_param(args_cli, cfg, "auto_resume", True))
    overwrite_output_dir = bool(get_param(args_cli, cfg, "overwrite_output_dir", False))
    gradient_checkpointing = bool(get_param(args_cli, cfg, "gradient_checkpointing", True))
    torch_compile_enabled = bool(get_param(args_cli, cfg, "torch_compile", False))
    torch_empty_cache_steps_raw = get_param(args_cli, cfg, "torch_empty_cache_steps", 4)
    torch_empty_cache_steps = (
        int(torch_empty_cache_steps_raw) if torch_empty_cache_steps_raw is not None else None
    )
    bf16 = bool(get_param(args_cli, cfg, "bf16", False))
    fp16 = bool(get_param(args_cli, cfg, "fp16", True))
    warmup_ratio = float(get_param(args_cli, cfg, "warmup_ratio", 0.03))
    lr_scheduler_type = str(get_param(args_cli, cfg, "lr_scheduler_type", "constant"))
    max_grad_norm = float(get_param(args_cli, cfg, "max_grad_norm", 0.3))
    max_train_samples_raw = get_param(args_cli, cfg, "max_train_samples", None)
    max_train_samples = int(max_train_samples_raw) if max_train_samples_raw is not None else None
    seed = int(get_param(args_cli, cfg, "seed", 42))
    early_stopping_cfg = get_section(cfg, "early_stopping")
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("early_stopping_patience", 1))
    early_stopping_threshold = float(early_stopping_cfg.get("early_stopping_threshold", 0.0))
    early_stopping_metric = str(early_stopping_cfg.get("metric", "eval_loss"))
    test_mode_cfg = get_section(cfg, "test_mode")
    test_mode_enabled = bool(test_mode_cfg.get("enabled", False))
    test_mode_max_train_raw = test_mode_cfg.get("max_train_samples")
    test_mode_max_eval_raw = test_mode_cfg.get("max_eval_samples")
    test_mode_max_train_samples = (
        int(test_mode_max_train_raw) if test_mode_max_train_raw is not None else None
    )
    test_mode_max_eval_samples = (
        int(test_mode_max_eval_raw) if test_mode_max_eval_raw is not None else None
    )
    test_mode_disable_adapter_save = bool(test_mode_cfg.get("disable_adapter_save", False))
    if test_mode_enabled and test_mode_max_train_samples is not None:
        max_train_samples = test_mode_max_train_samples
    fail_on_packing_warning = bool(get_param(args_cli, cfg, "fail_on_packing_warning", True))
    use_flash_attention_raw = get_param(args_cli, cfg, "use_flash_attention", None)
    attn_implementation_raw = get_param(args_cli, cfg, "attn_implementation", None)
    attn_implementation, flash_attention_requested, flash_attention_availability = v2._resolve_attention_settings(
        use_flash_attention_raw=use_flash_attention_raw,
        attn_implementation_raw=attn_implementation_raw,
        packing=packing,
    )
    use_flash_attention = v2._optional_bool(use_flash_attention_raw)
    if use_flash_attention is None:
        use_flash_attention = flash_attention_requested
    flash_attention_available = _selected_flash_attention_available(
        attn_implementation,
        flash_attention_availability,
    )

    if completion_only_loss:
        raise ValueError("Pipeline v1-clean full-chat requires completion_only_loss=false.")
    if assistant_only_loss:
        raise ValueError("Pipeline v1-clean full-chat requires assistant_only_loss=false.")
    if not packing:
        raise ValueError("Pipeline v1-clean requires packing=true.")
    if max_length != 1024:
        logger.warning("Pipeline v1-clean recommended max_length=1024, got %s", max_length)
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be true")
    has_eval_dataset = eval_dataset_path_raw is not None and str(eval_dataset_path_raw).strip() != ""
    per_device_eval_batch_size = (
        int(per_device_eval_batch_size_raw)
        if per_device_eval_batch_size_raw is not None
        else (1 if has_eval_dataset else per_device_train_batch_size)
    )
    prediction_loss_only = (
        bool(prediction_loss_only_raw)
        if prediction_loss_only_raw is not None
        else bool(has_eval_dataset)
    )
    if per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be >= 1.")
    if per_device_eval_batch_size < 1:
        raise ValueError("per_device_eval_batch_size must be >= 1.")
    if eval_accumulation_steps is not None and eval_accumulation_steps < 1:
        raise ValueError("eval_accumulation_steps must be >= 1 when set.")
    if early_stopping_enabled and not has_eval_dataset:
        raise ValueError("early_stopping.enabled=true requires eval_dataset_path.")
    if load_best_model_at_end and not has_eval_dataset:
        raise ValueError("load_best_model_at_end=true requires eval_dataset_path.")
    if has_eval_dataset and eval_strategy == "no":
        raise ValueError("eval_dataset_path requires eval_strategy to be 'epoch' or 'steps'.")
    if load_best_model_at_end and save_strategy != eval_strategy:
        raise ValueError(
            "load_best_model_at_end=true requires save_strategy to match eval_strategy "
            f"(got save_strategy={save_strategy!r}, eval_strategy={eval_strategy!r})."
        )
    if early_stopping_enabled and early_stopping_patience < 1:
        raise ValueError("early_stopping.early_stopping_patience must be >= 1.")
    if early_stopping_enabled and early_stopping_metric != metric_for_best_model:
        logger.warning(
            "early_stopping.metric=%r differs from metric_for_best_model=%r; "
            "Transformers EarlyStoppingCallback follows metric_for_best_model.",
            early_stopping_metric,
            metric_for_best_model,
        )

    dataset_path = Path(dataset_path_raw)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing v1-clean full-chat dataset: {dataset_path}")
    eval_dataset_path = None
    if has_eval_dataset:
        eval_dataset_path = Path(str(eval_dataset_path_raw))
        if not eval_dataset_path.is_absolute():
            eval_dataset_path = project_root / eval_dataset_path
        if not eval_dataset_path.exists():
            raise FileNotFoundError(f"Missing v1-clean eval full-chat dataset: {eval_dataset_path}")

    out_dir = Path(output_dir_raw)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    if test_mode_enabled and "testmode" not in str(out_dir).lower():
        raise ValueError(
            "test_mode.enabled=true requires output_dir to contain 'testmode' to avoid "
            f"accidental final adapter writes: {out_dir}"
        )
    continue_from_adapter_path = None
    if continue_from_adapter_raw:
        continue_from_adapter_path = Path(str(continue_from_adapter_raw))
        if not continue_from_adapter_path.is_absolute():
            continue_from_adapter_path = project_root / continue_from_adapter_path
        if not continue_from_adapter_path.exists():
            raise FileNotFoundError(f"Continued-LoRA start adapter not found: {continue_from_adapter_path}")
        if not continue_from_adapter_path.is_dir():
            raise ValueError(f"Continued-LoRA start adapter is not a directory: {continue_from_adapter_path}")
    checkpoint_dir = out_dir / "checkpoints"
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if out_dir.exists():
        if out_dir.is_file():
            raise ValueError(f"Output path exists as file, expected directory: {out_dir}")
        has_files = any(out_dir.iterdir())
        if has_files and not overwrite_output_dir:
            if auto_resume and last_checkpoint:
                logger.info("Output directory is not empty; resuming from checkpoint: %s", last_checkpoint)
            else:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {out_dir}. "
                    "Set overwrite_output_dir=true or auto_resume=true with an existing checkpoint."
                )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset("json", data_files={"train": str(dataset_path)})
    train_dataset = dataset_dict["train"]
    _ensure_text_dataset(train_dataset, dataset_text_field)
    raw_dataset_len = len(train_dataset)
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, raw_dataset_len)))
    selected_dataset_len = len(train_dataset)
    eval_dataset = None
    raw_eval_dataset_len = None
    selected_eval_dataset_len = None
    if eval_dataset_path is not None:
        eval_dataset_dict = load_dataset("json", data_files={"validation": str(eval_dataset_path)})
        eval_dataset = eval_dataset_dict["validation"]
        _ensure_text_dataset(eval_dataset, dataset_text_field)
        raw_eval_dataset_len = len(eval_dataset)
        if test_mode_enabled and test_mode_max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(test_mode_max_eval_samples, raw_eval_dataset_len)))
        selected_eval_dataset_len = len(eval_dataset)

    client = LLMClient(project_root)
    model_id = client.get_model_id(llm)
    model_revision = client.resolve_model_revision(llm)
    tokenizer = client.get_tokenizer(llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = client.get_base_model(llm, attn_implementation=attn_implementation)
    if continue_from_adapter_path is not None:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("PEFT is required for continued LoRA training.") from exc
        logger.info("Loading continued-LoRA start adapter from: %s", continue_from_adapter_path)
        model = PeftModel.from_pretrained(model, str(continue_from_adapter_path), is_trainable=True)
    effective_attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    model_device = str(next(model.parameters()).device)
    if flash_attention_requested and effective_attn_implementation != attn_implementation:
        logger.warning(
            "Requested attention implementation %r, but model config reports effective implementation %r.",
            attn_implementation,
            effective_attn_implementation,
        )
    model_use_cache_before = getattr(getattr(model, "config", None), "use_cache", None)
    if (
        (gradient_checkpointing or eval_dataset is not None)
        and hasattr(model, "config")
        and hasattr(model.config, "use_cache")
    ):
        model.config.use_cache = False
    model_use_cache_after = getattr(getattr(model, "config", None), "use_cache", None)

    lora_config, lora_meta = v2._resolve_lora_config(cfg, args_cli)
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    expected_steps_per_epoch, expected_total_steps = v2._estimate_steps(
        dataset_len=selected_dataset_len,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    sft_values = {
        "output_dir": str(checkpoint_dir),
        "dataset_text_field": dataset_text_field,
        "packing": packing,
        "packing_strategy": packing_strategy,
        "max_length": max_length,
        "completion_only_loss": completion_only_loss,
        "assistant_only_loss": assistant_only_loss,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "logging_steps": logging_steps,
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": "none",
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "torch_compile": torch_compile_enabled,
    }
    if torch_empty_cache_steps is not None:
        sft_values["torch_empty_cache_steps"] = torch_empty_cache_steps
    if eval_dataset is not None:
        sft_values.update(
            {
                "eval_strategy": eval_strategy,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "prediction_loss_only": prediction_loss_only,
                "load_best_model_at_end": load_best_model_at_end,
                "metric_for_best_model": metric_for_best_model,
                "greater_is_better": greater_is_better,
            }
        )
        if eval_steps is not None:
            sft_values["eval_steps"] = eval_steps
        if eval_accumulation_steps is not None:
            sft_values["eval_accumulation_steps"] = eval_accumulation_steps

    sft_config, applied_sft_fields, effective_sft_kwargs = v2._build_sft_config(SFTConfig, sft_values)
    v2._register_trl_flash_attention_variant(attn_implementation)
    trainer = v2._build_sft_trainer(
        SFTTrainer=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        sft_config=sft_config,
        lora_config=None if continue_from_adapter_path is not None else lora_config,
    )
    processed_train_dataset_len = len(trainer.train_dataset)
    processed_eval_dataset_len = len(trainer.eval_dataset) if eval_dataset is not None else None
    trainer_expected_steps_per_epoch, trainer_expected_total_steps = v2._estimate_steps(
        dataset_len=processed_train_dataset_len,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    label_stats = _verify_full_chat_labels(trainer, dataset_name="train")
    eval_label_stats = None
    if eval_dataset is not None:
        eval_label_stats = _verify_full_chat_labels(
            trainer,
            dataset=trainer.eval_dataset,
            dataset_name="eval",
        )
    packing_result = v2._verify_packing(
        trainer=trainer,
        raw_dataset_len=selected_dataset_len,
        max_length=max_length,
        packing=packing,
        fail_on_warning=fail_on_packing_warning,
    )
    trainable_params, total_params, trainable_param_ratio = v2._count_trainable_parameters(trainer.model)

    history_csv_path, history_jsonl_path = v2.history_paths(out_dir)
    central_history_csv_path, central_history_jsonl_path = v2.central_metric_paths(project_root, out_dir)
    metadata_path = out_dir / "training_metadata.json"
    config_path = None
    if args_cli.config:
        config_path = Path(args_cli.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    entrypoint_path = Path(__file__).resolve()
    native_prompt_module = project_root / "src" / "llama32_native_chat.py"
    training_metadata: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "duration_human_readable": None,
        "trl_version": v2._package_version("trl"),
        "transformers_version": v2._package_version("transformers"),
        "peft_version": v2._package_version("peft"),
        "torch_version": v2._package_version("torch"),
        "datasets_version": v2._package_version("datasets"),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_chat_template_sha256": (
            _sha256_text(str(tokenizer.chat_template))
            if getattr(tokenizer, "chat_template", None)
            else None
        ),
        "tokenizer_special_tokens_map": {
            key: str(value) for key, value in tokenizer.special_tokens_map.items()
        },
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "training_config_path": str(config_path) if config_path is not None else None,
        "training_config_sha256": (
            _sha256_file(config_path) if config_path is not None else None
        ),
        "training_entrypoint": str(entrypoint_path),
        "training_entrypoint_sha256": _sha256_file(entrypoint_path),
        "prompt_chat_implementation": (
            str(native_prompt_module)
            if llm == "llama32_3b_instruct" and native_prompt_module.is_file()
            else None
        ),
        "prompt_chat_implementation_sha256": (
            _sha256_file(native_prompt_module)
            if llm == "llama32_3b_instruct" and native_prompt_module.is_file()
            else None
        ),
        "train_dataset_sha256": _sha256_file(dataset_path),
        "eval_dataset_sha256": (
            _sha256_file(eval_dataset_path) if eval_dataset_path is not None else None
        ),
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "tokenizers_version": v2._package_version("tokenizers"),
        "attn_implementation_requested": attn_implementation,
        "attn_implementation_effective": effective_attn_implementation,
        "use_flash_attention": use_flash_attention,
        "flash_attention_requested": flash_attention_requested,
        "flash_attention_available": flash_attention_available,
        "flash_attention_availability": flash_attention_availability,
        "flash_attention_installed": bool(flash_attention_availability.get("flash_attn_installed")),
        "flash_attn_installed": bool(flash_attention_availability.get("flash_attn_installed")),
        "model_device": model_device,
        "torch_cuda_version": flash_attention_availability.get("torch_cuda_version"),
        "cuda_available": flash_attention_availability.get("cuda_available"),
        "device_count": flash_attention_availability.get("device_count"),
        "device_name_0": flash_attention_availability.get("device_name_0"),
        "capability_0": flash_attention_availability.get("capability_0"),
        "bf16_supported": flash_attention_availability.get("bf16_supported"),
        "llm": llm,
        "adapter": out_dir.name,
        "continued_lora_training": continue_from_adapter_path is not None,
        "continue_from_adapter": str(continue_from_adapter_path) if continue_from_adapter_path is not None else None,
        "additional_epochs": float(cfg.get("additional_epochs", num_train_epochs)),
        "total_effective_epochs": cfg.get("total_effective_epochs"),
        "dataset_path": str(dataset_path),
        "eval_dataset_path": str(eval_dataset_path) if eval_dataset_path is not None else None,
        "dataset_format": "full_chat_text",
        "dataset_text_field": dataset_text_field,
        "loss_mode": LOSS_MODE,
        "completion_only_loss": completion_only_loss,
        "assistant_only_loss": assistant_only_loss,
        "packing": packing,
        "packing_strategy": packing_strategy,
        "max_length": max_length,
        "raw_dataset_len": raw_dataset_len,
        "selected_dataset_len": selected_dataset_len,
        "processed_train_dataset_len": processed_train_dataset_len,
        "raw_eval_dataset_len": raw_eval_dataset_len,
        "selected_eval_dataset_len": selected_eval_dataset_len,
        "processed_eval_dataset_len": processed_eval_dataset_len,
        "label_stats_verified": label_stats["verified"],
        "label_stats": label_stats,
        "eval_label_stats_verified": (
            eval_label_stats["verified"] if eval_label_stats is not None else None
        ),
        "eval_label_stats": eval_label_stats,
        "packing_verified": packing_result["verified"],
        "packing_verification": packing_result,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_param_ratio": trainable_param_ratio,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size if eval_dataset is not None else None,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "expected_steps_per_epoch_before_packing": expected_steps_per_epoch,
        "expected_total_steps_before_packing": expected_total_steps,
        "trainer_expected_steps_per_epoch": trainer_expected_steps_per_epoch,
        "trainer_expected_total_steps": trainer_expected_total_steps,
        "eval_strategy": eval_strategy if eval_dataset is not None else None,
        "eval_steps": eval_steps if eval_dataset is not None else None,
        "eval_accumulation_steps": eval_accumulation_steps if eval_dataset is not None else None,
        "prediction_loss_only": prediction_loss_only if eval_dataset is not None else None,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model if eval_dataset is not None else None,
        "greater_is_better": greater_is_better if eval_dataset is not None else None,
        "save_best_model": save_best_model,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_threshold": early_stopping_threshold,
            "metric": early_stopping_metric,
        },
        "test_mode": {
            "enabled": test_mode_enabled,
            "max_train_samples": test_mode_max_train_samples,
            "max_eval_samples": test_mode_max_eval_samples,
            "disable_adapter_save": test_mode_disable_adapter_save,
        },
        "auto_resume": auto_resume,
        "overwrite_output_dir": overwrite_output_dir,
        "gradient_checkpointing": gradient_checkpointing,
        "model_use_cache_before": model_use_cache_before,
        "model_use_cache_after": model_use_cache_after,
        "eval_cuda_empty_cache_enabled": eval_dataset is not None,
        "torch_compile": torch_compile_enabled,
        "torch_empty_cache_steps": torch_empty_cache_steps,
        "fp16": fp16,
        "bf16": bf16,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "lora": lora_meta,
        "sft_config_applied_fields": applied_sft_fields,
        "effective_sft_config_kwargs": effective_sft_kwargs,
        "sft_config_effective": dict(vars(sft_config)),
        "output_dir": str(out_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "training_history_csv": str(history_csv_path),
        "training_history_jsonl": str(history_jsonl_path),
        "central_training_history_csv": str(central_history_csv_path),
        "central_training_history_jsonl": str(central_history_jsonl_path),
        "final_train_loss": None,
        "latest_eval_loss": None,
        "best_model_checkpoint": None,
        "best_metric": None,
        "best_eval_loss": None,
        "stopped_epoch": None,
        **v2._cuda_metadata(),
    }
    v2._write_training_metadata(metadata_path, training_metadata)

    logger.info("Pipeline v1-clean LoRA SFT setup")
    logger.info("model_id=%s", model_id)
    logger.info("attn_implementation_requested=%s", attn_implementation or "default")
    logger.info("attn_implementation_effective=%s", effective_attn_implementation)
    logger.info("use_flash_attention=%s", use_flash_attention)
    logger.info("flash_attention_requested=%s", flash_attention_requested)
    logger.info("flash_attention_available=%s", flash_attention_available)
    logger.info("flash_attention_availability=%s", flash_attention_availability)
    logger.info("model_device=%s", model_device)
    logger.info("dataset_path=%s", dataset_path)
    if eval_dataset_path is not None:
        logger.info("eval_dataset_path=%s", eval_dataset_path)
    logger.info("loss_mode=%s", LOSS_MODE)
    logger.info("raw_dataset_len=%d selected_dataset_len=%d", raw_dataset_len, selected_dataset_len)
    logger.info("processed_train_dataset_len=%d", processed_train_dataset_len)
    if eval_dataset is not None:
        logger.info(
            "raw_eval_dataset_len=%d selected_eval_dataset_len=%d processed_eval_dataset_len=%d",
            raw_eval_dataset_len,
            selected_eval_dataset_len,
            processed_eval_dataset_len,
        )
        logger.info("per_device_eval_batch_size=%d", per_device_eval_batch_size)
        logger.info("eval_accumulation_steps=%s", eval_accumulation_steps)
        logger.info("prediction_loss_only=%s", prediction_loss_only)
        logger.info("model_use_cache_before=%s model_use_cache_after=%s", model_use_cache_before, model_use_cache_after)
    logger.info("effective SFTConfig kwargs=%s", effective_sft_kwargs)
    logger.info("label_stats=%s", label_stats)
    if eval_label_stats is not None:
        logger.info("eval_label_stats=%s", eval_label_stats)
    logger.info("packing_verification=%s", packing_result)
    logger.info(
        "trainable_params=%d total_params=%d trainable_ratio=%.8f",
        trainable_params,
        total_params,
        trainable_param_ratio,
    )

    trainer.add_callback(v2.TrainingMetadataCallback(metadata_path=metadata_path, metadata=training_metadata))
    trainer.add_callback(v2.TrainingHistoryCallback(adapter_dir=out_dir, project_root=project_root))
    if eval_dataset is not None:
        trainer.add_callback(EvalCudaMemoryCleanupCallback())
    if early_stopping_enabled:
        try:
            from transformers import EarlyStoppingCallback
        except Exception as exc:
            raise RuntimeError("Transformers EarlyStoppingCallback is required for early stopping.") from exc
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        if eval_dataset is not None:
            torch.cuda.empty_cache()
    if auto_resume:
        if last_checkpoint:
            logger.info("Resuming training from checkpoint: %s", last_checkpoint)
        else:
            logger.info("auto_resume=true but no checkpoint found; starting from scratch")

    train_start_dt = datetime.now(timezone.utc)
    train_start_perf = time.perf_counter()
    training_metadata["start_time"] = train_start_dt.isoformat()
    v2._write_training_metadata(metadata_path, training_metadata)
    train_output = trainer.train(resume_from_checkpoint=last_checkpoint if auto_resume else None)
    train_end_dt = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - train_start_perf
    final_train_loss = getattr(train_output, "training_loss", None)
    if final_train_loss is not None:
        final_train_loss = float(final_train_loss)
    training_metadata["end_time"] = train_end_dt.isoformat()
    training_metadata["duration_seconds"] = duration_seconds
    training_metadata["duration_human_readable"] = v2._format_duration_hms(duration_seconds)
    training_metadata["final_train_loss"] = final_train_loss
    training_metadata["best_model_checkpoint"] = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    if best_metric is not None:
        training_metadata["best_metric"] = float(best_metric)
        if metric_for_best_model in {"eval_loss", "loss"}:
            training_metadata["best_eval_loss"] = float(best_metric)
    trainer_epoch = getattr(trainer.state, "epoch", None)
    if (
        early_stopping_enabled
        and trainer_epoch is not None
        and float(trainer_epoch) + 1e-6 < float(num_train_epochs)
    ):
        training_metadata["stopped_epoch"] = float(trainer_epoch)
    training_metadata.update(v2._cuda_metadata())
    v2._write_training_metadata(metadata_path, training_metadata)
    logger.info("Training runtime: %s total (%.2fs)", training_metadata["duration_human_readable"], duration_seconds)
    logger.info("Final train loss: %s", final_train_loss)

    adapter_save_enabled = save_best_model and not test_mode_disable_adapter_save
    training_metadata["adapter_save_enabled"] = adapter_save_enabled
    v2._write_training_metadata(metadata_path, training_metadata)
    if adapter_save_enabled:
        trainer.model.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))
        logger.info("Saved Pipeline v1-clean LoRA adapter to: %s", out_dir)
    else:
        logger.info(
            "Skipping final adapter/tokenizer save because save_best_model=%s and "
            "test_mode.disable_adapter_save=%s.",
            save_best_model,
            test_mode_disable_adapter_save,
        )
    try:
        try:
            from src.plot_training_history import generate_standard_single_run_outputs
        except ModuleNotFoundError:
            from plot_training_history import generate_standard_single_run_outputs

        generated_plots = generate_standard_single_run_outputs(out_dir, project_root=project_root, dpi=300)
        logger.info("Generated %d training history plot(s)", len(generated_plots))
    except Exception as exc:
        logger.warning("Training finished, but training history plotting failed: %s", exc)


if __name__ == "__main__":
    main()
