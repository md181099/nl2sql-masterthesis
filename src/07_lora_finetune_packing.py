#!/usr/bin/env python3
from __future__ import annotations

"""
Separate LoRA SFT training pipeline using TRL SFTTrainer with optional packing.

This script is intentionally separate from src/07_lora_finetune.py and does not
replace it. Keep both pipelines methodically separated in experiments:
- src/07_lora_finetune.py: existing non-packing trainer with prompt-masked loss.
- this script: TRL SFTTrainer path where packing behavior can differ.

As a result, metrics are not always directly comparable and should be reported
as separate experimental settings.
"""

import argparse
import inspect
import json
import logging
import math
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from config import get_param, get_section, load_config
from logging_utils import setup_logging

try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient

try:
    from src.training_history_utils import (
        append_history_row,
        central_metric_paths,
        history_paths,
        load_existing_steps,
        row_from_trainer_log,
    )
except ModuleNotFoundError:
    from training_history_utils import (
        append_history_row,
        central_metric_paths,
        history_paths,
        load_existing_steps,
        row_from_trainer_log,
    )


logger = logging.getLogger(__name__)

TRL_MISSING_MESSAGE = "TRL is required for packing training. Install with: pip install trl"


def parse_bool(value: str) -> bool:
    value_norm = value.strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _coerce_save_strategy(value: str) -> str:
    valid = {"no", "steps", "epoch", "best"}
    normalized = str(value).strip().lower()
    if normalized not in valid:
        raise ValueError(f"Invalid save_strategy '{value}'. Valid values: {', '.join(sorted(valid))}")
    return normalized


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator) if numerator > 0 else 0


def _estimate_steps(
    *,
    dataset_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
) -> tuple[int, int]:
    microbatches_per_epoch = _ceil_div(dataset_len, per_device_train_batch_size)
    steps_per_epoch = _ceil_div(microbatches_per_epoch, gradient_accumulation_steps)
    total_steps = math.ceil(steps_per_epoch * num_train_epochs)
    return steps_per_epoch, total_steps


def _format_duration_hms(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_training_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TrainingMetadataCallback(TrainerCallback):
    def __init__(
        self,
        *,
        metadata_path: Path,
        metadata: dict[str, Any],
        trainer_train_dataset_len: int | None,
    ) -> None:
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.trainer_train_dataset_len = trainer_train_dataset_len

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            "Trainer info: len(train_dataset)=%s, trainer max_steps=%s, trainer num_train_epochs=%s",
            self.trainer_train_dataset_len,
            state.max_steps,
            args.num_train_epochs,
        )
        self.metadata["trainer_train_dataset_len"] = self.trainer_train_dataset_len
        self.metadata["trainer_max_steps"] = state.max_steps
        self.metadata["trainer_num_train_epochs"] = args.num_train_epochs
        self.metadata["train_begin_timestamp"] = datetime.now(timezone.utc).isoformat()
        _write_training_metadata(self.metadata_path, self.metadata)

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            _write_training_metadata(checkpoint_dir / "training_metadata.json", self.metadata)


class TrainingHistoryCallback(TrainerCallback):
    def __init__(
        self,
        *,
        adapter_dir: Path,
        project_root: Path,
    ) -> None:
        self.adapter_csv_path, self.adapter_jsonl_path = history_paths(adapter_dir)
        self.central_csv_path, self.central_jsonl_path = central_metric_paths(project_root, adapter_dir)
        self.start_perf: float | None = None
        try:
            self.seen_steps = load_existing_steps(self.adapter_csv_path, self.adapter_jsonl_path)
        except Exception as exc:
            logger.warning("Could not read existing training history for de-duplication: %s", exc)
            self.seen_steps = set()

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_perf = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if step in self.seen_steps:
            return
        elapsed_seconds = None
        if self.start_perf is not None:
            elapsed_seconds = time.perf_counter() - self.start_perf
        row = row_from_trainer_log(
            dict(logs),
            step=step,
            epoch=getattr(state, "epoch", None),
            elapsed_seconds=elapsed_seconds,
        )
        if row is None:
            return
        try:
            append_history_row(self.adapter_csv_path, self.adapter_jsonl_path, row)
            append_history_row(self.central_csv_path, self.central_jsonl_path, row)
            self.seen_steps.add(step)
        except Exception as exc:
            logger.warning("Failed to write structured training history: %s", exc)


def _ensure_text_field(dataset_obj: Any, dataset_text_field: str) -> None:
    if dataset_text_field not in dataset_obj.column_names:
        raise ValueError(
            f"Dataset is missing required text field '{dataset_text_field}'. "
            f"Available fields: {dataset_obj.column_names}"
        )


def _resolve_lora_config(cfg: dict[str, Any], args_cli: argparse.Namespace) -> tuple[LoraConfig, dict[str, Any]]:
    lora_section = get_section(cfg, "lora")
    lora_cfg = dict(lora_section)

    lora_r = int(get_param(args_cli, lora_cfg, "lora_r", 8, config_name="r"))
    lora_alpha = int(get_param(args_cli, lora_cfg, "lora_alpha", 16, config_name="lora_alpha"))
    lora_dropout = float(
        get_param(args_cli, lora_cfg, "lora_dropout", 0.05, config_name="lora_dropout")
    )
    lora_bias = str(get_param(args_cli, lora_cfg, "lora_bias", "none", config_name="bias"))
    task_type = str(
        get_param(args_cli, lora_cfg, "lora_task_type", "CAUSAL_LM", config_name="task_type")
    )
    use_dora = bool(get_param(args_cli, lora_cfg, "use_dora", False, config_name="use_dora"))
    target_modules = get_param(
        args_cli,
        lora_cfg,
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        config_name="target_modules",
    )
    if isinstance(target_modules, str):
        if target_modules.strip() == "all-linear":
            target_modules = "all-linear"
        else:
            # Backward-compatible: allow a single module name as string.
            target_modules = [target_modules]
    if not (
        target_modules == "all-linear"
        or (
            isinstance(target_modules, list)
            and all(isinstance(module, str) for module in target_modules)
        )
    ):
        raise ValueError("LoRA target_modules must be a list of strings or 'all-linear'")

    lora_kwargs: dict[str, Any] = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": lora_bias,
        "task_type": task_type,
        "target_modules": target_modules,
    }

    supports_use_dora = "use_dora" in inspect.signature(LoraConfig.__init__).parameters
    if use_dora:
        if not supports_use_dora:
            raise RuntimeError(
                "Config requested use_dora=true, but the installed peft version "
                "does not support 'use_dora' in LoraConfig."
            )
        lora_kwargs["use_dora"] = True

    lora_config = LoraConfig(**lora_kwargs)
    return lora_config, {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": lora_bias,
        "task_type": task_type,
        "use_dora": use_dora,
        "target_modules": target_modules,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate TRL packing LoRA SFT trainer")
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
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_strategy", default=None, help="no|steps|epoch|best")
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--auto_resume", type=parse_bool, default=None)
    parser.add_argument("--overwrite_output_dir", type=parse_bool, default=None)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=None)
    parser.add_argument("--torch_compile", type=parse_bool, default=None)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=None)
    parser.add_argument("--bf16", type=parse_bool, default=None)
    parser.add_argument("--fp16", type=parse_bool, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke tests",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", default=None)
    parser.add_argument("--lora_task_type", default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    return parser.parse_args()


def _build_sft_trainer(
    sft_trainer_cls: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    dataset_text_field: str,
    packing: bool,
    max_seq_length: int,
) -> Any:
    init_params = inspect.signature(sft_trainer_cls.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }

    # TRL renamed tokenizer->processing_class in newer versions.
    if "processing_class" in init_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in init_params:
        kwargs["tokenizer"] = tokenizer

    # Some TRL versions take these directly on SFTTrainer.
    if "dataset_text_field" in init_params:
        kwargs["dataset_text_field"] = dataset_text_field
    if "packing" in init_params:
        kwargs["packing"] = packing
    if "max_seq_length" in init_params:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in init_params:
        kwargs["max_length"] = max_seq_length

    return sft_trainer_cls(**kwargs)


def main() -> None:
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)

    try:
        from trl import SFTTrainer
    except Exception as exc:
        raise RuntimeError(TRL_MISSING_MESSAGE) from exc

    cfg = load_config(args_cli.config) if args_cli.config else {}
    project_root = Path(__file__).resolve().parents[1]

    llm = get_param(args_cli, cfg, "llm", "llama32_1b")
    dataset_path = get_param(args_cli, cfg, "dataset_path", "data/train_sft.jsonl")
    dataset_text_field = get_param(args_cli, cfg, "dataset_text_field", "text")
    output_dir = get_param(args_cli, cfg, "output_dir", f"adapters/{llm}/lora_sql_packing")
    packing = bool(get_param(args_cli, cfg, "packing", True))
    max_seq_length = int(get_param(args_cli, cfg, "max_seq_length", 512))
    num_train_epochs = float(get_param(args_cli, cfg, "num_train_epochs", 1))
    learning_rate = float(get_param(args_cli, cfg, "learning_rate", 1e-4))
    per_device_train_batch_size = int(
        get_param(args_cli, cfg, "per_device_train_batch_size", 1)
    )
    gradient_accumulation_steps = int(get_param(args_cli, cfg, "gradient_accumulation_steps", 8))
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 10))
    save_strategy = _coerce_save_strategy(get_param(args_cli, cfg, "save_strategy", "no"))
    save_total_limit_raw = get_param(args_cli, cfg, "save_total_limit", None)
    save_total_limit = int(save_total_limit_raw) if save_total_limit_raw is not None else None
    if save_total_limit is not None and save_total_limit < 1:
        raise ValueError("save_total_limit must be >= 1 or null")
    auto_resume = bool(get_param(args_cli, cfg, "auto_resume", False))
    overwrite_output_dir = bool(get_param(args_cli, cfg, "overwrite_output_dir", False))
    gradient_checkpointing = bool(get_param(args_cli, cfg, "gradient_checkpointing", False))
    torch_compile_enabled = bool(get_param(args_cli, cfg, "torch_compile", False))
    torch_empty_cache_steps_raw = get_param(args_cli, cfg, "torch_empty_cache_steps", None)
    torch_empty_cache_steps = (
        int(torch_empty_cache_steps_raw) if torch_empty_cache_steps_raw is not None else None
    )
    if torch_empty_cache_steps is not None and torch_empty_cache_steps < 1:
        raise ValueError("torch_empty_cache_steps must be >= 1 or null")
    bf16 = bool(get_param(args_cli, cfg, "bf16", False))
    fp16 = bool(get_param(args_cli, cfg, "fp16", torch.cuda.is_available()))
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be true")
    warmup_ratio = float(get_param(args_cli, cfg, "warmup_ratio", 0.0))
    lr_scheduler_type = str(get_param(args_cli, cfg, "lr_scheduler_type", "linear"))
    max_grad_norm = float(get_param(args_cli, cfg, "max_grad_norm", 1.0))
    seed = int(get_param(args_cli, cfg, "seed", 42))
    max_train_samples_raw = get_param(args_cli, cfg, "max_train_samples", None)
    max_train_samples = int(max_train_samples_raw) if max_train_samples_raw is not None else None
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be >= 1 or null")

    if "quantization" in cfg:
        logger.warning("Config field 'quantization' is currently ignored in this packing pipeline.")
    if "deep_speed" in cfg:
        logger.warning("Config field 'deep_speed' is currently ignored in this packing pipeline.")

    train_path = Path(dataset_path)
    if not train_path.is_absolute():
        train_path = project_root / train_path
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    checkpoint_dir = out_dir / "checkpoints"
    last_checkpoint = (
        get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    )

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
                    "Set overwrite_output_dir=true to allow reuse."
                )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset("json", data_files={"train": str(train_path)})
    train_dataset = dataset_dict["train"]
    _ensure_text_field(train_dataset, dataset_text_field)
    dataset_size_before = len(train_dataset)
    dataset_size_after_filters = dataset_size_before
    if max_train_samples is not None:
        max_idx = min(max_train_samples, dataset_size_before)
        train_dataset = train_dataset.select(range(max_idx))
    dataset_size = len(train_dataset)
    dataset_size_after_max_train_samples = dataset_size
    expected_steps_per_epoch, expected_total_steps = _estimate_steps(
        dataset_len=dataset_size,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )
    estimated_example_passes = dataset_size * num_train_epochs
    estimated_input_text_sequences = dataset_size
    estimated_token_sequences_before_trl = None if packing else dataset_size

    client = LLMClient(project_root)
    model_id = client.get_model_id(llm)
    tokenizer = client.get_tokenizer(llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = client.get_base_model(llm)
    if gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("gradient_checkpointing=true but model does not support gradient_checkpointing_enable()")
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora_config, lora_meta = _resolve_lora_config(cfg, args_cli)

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    logger.info("Packing LoRA SFT setup")
    logger.info("model_id=%s", model_id)
    logger.info("dataset_path=%s", train_path)
    logger.info("dataset_loaded_examples=%d", dataset_size_before)
    logger.info("dataset_examples_after_filters=%d", dataset_size_after_filters)
    logger.info("max_train_samples=%s", max_train_samples)
    logger.info("dataset_examples_after_max_train_samples=%d", dataset_size_after_max_train_samples)
    logger.info("training_examples_passed_to_trainer=%d", dataset_size)
    logger.info("Final training examples: %d", dataset_size)
    logger.info("dataset_text_field=%s", dataset_text_field)
    logger.info("seed=%d", seed)
    logger.info("packing=%s", packing)
    logger.info("max_seq_length=%d", max_seq_length)
    logger.info("num_train_epochs=%s", num_train_epochs)
    logger.info("per_device_train_batch_size=%d", per_device_train_batch_size)
    logger.info("gradient_accumulation_steps=%d", gradient_accumulation_steps)
    logger.info("effective_batch_size=%d", effective_batch_size)
    logger.info("expected_steps_per_epoch=%d", expected_steps_per_epoch)
    logger.info("expected_total_steps=%d", expected_total_steps)
    logger.info("estimated_example_passes=%s", estimated_example_passes)
    logger.info("estimated_input_text_sequences=%d", estimated_input_text_sequences)
    logger.info("estimated_token_sequences_before_trl=%s", estimated_token_sequences_before_trl)
    logger.info("output_dir=%s", out_dir)
    logger.info("checkpoint_dir=%s", checkpoint_dir)
    logger.info("save_strategy=%s", save_strategy)
    logger.info("save_total_limit=%s", save_total_limit)
    logger.info("auto_resume=%s", auto_resume)
    logger.info("use_dora=%s", lora_meta["use_dora"])
    logger.info("target_modules=%s", lora_meta["target_modules"])
    logger.info("gradient_checkpointing=%s", gradient_checkpointing)
    logger.info("torch_compile=%s", torch_compile_enabled)
    logger.info("torch_empty_cache_steps=%s", torch_empty_cache_steps)
    logger.info("bf16=%s", bf16)
    logger.info("fp16=%s", fp16)
    logger.info("warmup_ratio=%s", warmup_ratio)
    logger.info("lr_scheduler_type=%s", lr_scheduler_type)
    logger.info("max_grad_norm=%s", max_grad_norm)
    logger.info("lora=%s", lora_meta)

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "fp16": fp16,
        "report_to": "none",
        "seed": seed,
    }
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "bf16" in training_args_params:
        training_args_kwargs["bf16"] = bf16
    elif bf16:
        raise RuntimeError("bf16=true but this transformers version does not support bf16")
    if torch_empty_cache_steps is not None:
        if "torch_empty_cache_steps" in training_args_params:
            training_args_kwargs["torch_empty_cache_steps"] = torch_empty_cache_steps
        else:
            logger.warning(
                "torch_empty_cache_steps=%s ignored because this transformers version "
                "does not support TrainingArguments.torch_empty_cache_steps",
                torch_empty_cache_steps,
            )
    logger.info(
        "TrainingArguments precision/cache: fp16=%s, bf16=%s, torch_empty_cache_steps=%s",
        training_args_kwargs.get("fp16"),
        training_args_kwargs.get("bf16", False),
        training_args_kwargs.get("torch_empty_cache_steps"),
    )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = _build_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        lora_config=lora_config,
        dataset_text_field=str(dataset_text_field),
        packing=packing,
        max_seq_length=max_seq_length,
    )
    try:
        trainer_train_dataset_len = len(trainer.train_dataset)
    except TypeError:
        trainer_train_dataset_len = None
    trainer_expected_steps_per_epoch = None
    trainer_expected_total_steps = None
    if trainer_train_dataset_len is not None:
        trainer_expected_steps_per_epoch, trainer_expected_total_steps = _estimate_steps(
            dataset_len=trainer_train_dataset_len,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
        )
    logger.info("trainer_train_dataset_len=%s", trainer_train_dataset_len)
    logger.info("trainer_expected_steps_per_epoch=%s", trainer_expected_steps_per_epoch)
    logger.info("trainer_expected_total_steps=%s", trainer_expected_total_steps)

    history_csv_path, history_jsonl_path = history_paths(out_dir)
    central_history_csv_path, central_history_jsonl_path = central_metric_paths(project_root, out_dir)
    training_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "duration_human_readable": None,
        "llm": llm,
        "model_id": model_id,
        "dataset_path": str(train_path),
        "dataset_text_field": str(dataset_text_field),
        "dataset_loaded_examples": dataset_size_before,
        "dataset_examples_after_filters": dataset_size_after_filters,
        "max_train_samples": max_train_samples,
        "dataset_examples_after_max_train_samples": dataset_size_after_max_train_samples,
        "training_examples": dataset_size,
        "final_training_examples": dataset_size,
        "epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "seed": seed,
        "max_seq_length": max_seq_length,
        "packing": packing,
        "fp16": fp16,
        "bf16": bf16,
        "torch_empty_cache_steps": torch_empty_cache_steps,
        "expected_steps_per_epoch": expected_steps_per_epoch,
        "expected_total_steps": expected_total_steps,
        "final_train_loss": None,
        "estimated_example_passes": estimated_example_passes,
        "estimated_input_text_sequences": estimated_input_text_sequences,
        "estimated_token_sequences_before_trl": estimated_token_sequences_before_trl,
        "trainer_train_dataset_len": trainer_train_dataset_len,
        "trainer_expected_steps_per_epoch": trainer_expected_steps_per_epoch,
        "trainer_expected_total_steps": trainer_expected_total_steps,
        "output_dir": str(out_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "training_history_csv": str(history_csv_path),
        "training_history_jsonl": str(history_jsonl_path),
        "central_training_history_csv": str(central_history_csv_path),
        "central_training_history_jsonl": str(central_history_jsonl_path),
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "auto_resume": auto_resume,
        "lora": lora_meta,
    }
    metadata_path = out_dir / "training_metadata.json"
    _write_training_metadata(metadata_path, training_metadata)
    logger.info("Training metadata written to: %s", metadata_path)
    trainer.add_callback(
        TrainingMetadataCallback(
            metadata_path=metadata_path,
            metadata=training_metadata,
            trainer_train_dataset_len=trainer_train_dataset_len,
        )
    )
    trainer.add_callback(
        TrainingHistoryCallback(
            adapter_dir=out_dir,
            project_root=project_root,
        )
    )
    if torch_compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch_compile=true but torch.compile is not available in this PyTorch build."
            )
        try:
            trainer.model = torch.compile(trainer.model)
        except Exception as exc:
            raise RuntimeError("torch_compile=true but torch.compile(model) failed.") from exc

    if auto_resume:
        if last_checkpoint:
            logger.info("Resuming training from checkpoint: %s", last_checkpoint)
        else:
            logger.info("auto_resume=true but no checkpoint found; starting from scratch")
    train_start_dt = datetime.now(timezone.utc)
    train_start_perf = time.perf_counter()
    training_metadata["start_time"] = train_start_dt.isoformat()
    _write_training_metadata(metadata_path, training_metadata)
    train_output = trainer.train(resume_from_checkpoint=last_checkpoint if auto_resume else None)
    train_end_dt = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - train_start_perf
    final_train_loss = getattr(train_output, "training_loss", None)
    if final_train_loss is not None:
        final_train_loss = float(final_train_loss)
    training_metadata["end_time"] = train_end_dt.isoformat()
    training_metadata["duration_seconds"] = duration_seconds
    training_metadata["duration_human_readable"] = _format_duration_hms(duration_seconds)
    training_metadata["final_train_loss"] = final_train_loss
    _write_training_metadata(metadata_path, training_metadata)
    logger.info(
        "Training runtime: %s total (%.2fs)",
        training_metadata["duration_human_readable"],
        duration_seconds,
    )
    logger.info("Final train loss: %s", final_train_loss)
    trainer.model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved packing LoRA adapter to: %s", out_dir)
    try:
        try:
            from src.plot_training_history import generate_standard_single_run_outputs
        except ModuleNotFoundError:
            from plot_training_history import generate_standard_single_run_outputs

        generated_plots = generate_standard_single_run_outputs(
            out_dir,
            project_root=project_root,
            dpi=300,
        )
        logger.info("Generated %d training history plot(s)", len(generated_plots))
    except Exception as exc:
        logger.warning("Training finished, but training history plotting failed: %s", exc)


if __name__ == "__main__":
    main()
