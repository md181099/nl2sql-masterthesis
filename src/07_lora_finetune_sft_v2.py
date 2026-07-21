#!/usr/bin/env python3
from __future__ import annotations

"""
Pipeline v2 LoRA SFT trainer for NL2SQL.

This script is intentionally separate from the legacy LoRA trainers. It uses
TRL SFTConfig directly and expects a prompt/completion dataset so that
completion_only_loss can be verified before training starts.
"""

import argparse
import importlib.util
import inspect
import json
import logging
import math
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers.trainer_callback import TrainerCallback
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
        upsert_history_row,
    )
except ModuleNotFoundError:
    from training_history_utils import (
        append_history_row,
        central_metric_paths,
        history_paths,
        load_existing_steps,
        row_from_trainer_log,
        upsert_history_row,
    )


logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v2"
REQUIRED_SFT_FIELDS = {
    "packing",
    "max_length",
    "completion_only_loss",
    "assistant_only_loss",
}
FLASH_ATTN_IMPLEMENTATIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}
DEFAULT_FLASH_ATTN_IMPLEMENTATION = "flash_attention_2"
KERNELS_COMMUNITY_FLASH_ATTN_REPOS = {
    item for item in FLASH_ATTN_IMPLEMENTATIONS if item.startswith("kernels-community/")
}


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
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


def _coerce_eval_strategy(value: str | None) -> str:
    if value is None:
        return "no"
    valid = {"no", "steps", "epoch"}
    normalized = str(value).strip().lower()
    if normalized not in valid:
        raise ValueError(f"Invalid eval_strategy '{value}'. Valid values: {', '.join(sorted(valid))}")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cuda_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "cuda_version": getattr(torch.version, "cuda", None),
            "peak_memory_allocated": None,
            "peak_memory_reserved": None,
        }
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": getattr(torch.version, "cuda", None),
        "peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return parse_bool(str(value))


def _kernel_repo_from_attn_implementation(attn_implementation: str | None) -> str | None:
    if not attn_implementation:
        return None
    text = str(attn_implementation).strip()
    if not text.startswith("kernels-community/"):
        return None
    repo_part = text.split(":", 1)[0]
    return repo_part.split("@", 1)[0]


def _is_flash_attention_implementation(attn_implementation: str | None) -> bool:
    if not attn_implementation:
        return False
    if attn_implementation in FLASH_ATTN_IMPLEMENTATIONS:
        return True
    return _kernel_repo_from_attn_implementation(attn_implementation) in KERNELS_COMMUNITY_FLASH_ATTN_REPOS


def _register_trl_flash_attention_variant(attn_implementation: str | None) -> None:
    if not _is_flash_attention_implementation(attn_implementation):
        return
    if not attn_implementation or not attn_implementation.startswith("kernels-community/"):
        return
    try:
        import trl.trainer.sft_trainer as sft_trainer

        variants = getattr(sft_trainer, "FLASH_ATTENTION_VARIANTS", None)
        if variants is not None and attn_implementation not in variants:
            variants.add(attn_implementation)
            logger.info("Registered pinned TRL flash attention variant: %s", attn_implementation)
    except Exception as exc:
        logger.warning("Could not register pinned TRL flash attention variant %r: %s", attn_implementation, exc)


def _flash_attention_availability() -> dict[str, Any]:
    try:
        from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available
    except Exception:
        is_flash_attn_2_available = None
        is_flash_attn_3_available = None

    cuda_available = torch.cuda.is_available()
    device_name = None
    capability = None
    bf16_supported = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            capability = None

    return {
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name_0": device_name,
        "capability_0": capability,
        "bf16_supported": bf16_supported,
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "flash_attn_interface_installed": importlib.util.find_spec("flash_attn_interface") is not None,
        "kernels_installed": importlib.util.find_spec("kernels") is not None,
        "flash_attention_2_available": (
            bool(is_flash_attn_2_available()) if is_flash_attn_2_available is not None else False
        ),
        "flash_attention_3_available": (
            bool(is_flash_attn_3_available()) if is_flash_attn_3_available is not None else False
        ),
    }


def _resolve_attention_settings(
    *,
    use_flash_attention_raw: Any,
    attn_implementation_raw: Any,
    packing: bool,
) -> tuple[str | None, bool, dict[str, Any]]:
    use_flash_attention = _optional_bool(use_flash_attention_raw)
    attn_implementation = _optional_str(attn_implementation_raw)
    if attn_implementation is not None and attn_implementation.lower() in {"none", "null", "default"}:
        attn_implementation = None

    if use_flash_attention is True and attn_implementation is None:
        attn_implementation = DEFAULT_FLASH_ATTN_IMPLEMENTATION
    if _is_flash_attention_implementation(attn_implementation) and use_flash_attention is None:
        use_flash_attention = True
    if use_flash_attention is False and _is_flash_attention_implementation(attn_implementation):
        raise ValueError(
            "Config is inconsistent: use_flash_attention=false but attn_implementation "
            f"requests {attn_implementation!r}."
        )

    flash_requested = bool(use_flash_attention or _is_flash_attention_implementation(attn_implementation))
    availability = _flash_attention_availability()

    if flash_requested:
        if not _is_flash_attention_implementation(attn_implementation):
            allowed = ", ".join(sorted(FLASH_ATTN_IMPLEMENTATIONS))
            raise ValueError(
                "use_flash_attention=true requires attn_implementation to be one of: "
                f"{allowed}, optionally with @revision for kernels-community entries. "
                f"Got: {attn_implementation!r}"
            )
        if attn_implementation == "flash_attention_2" and not availability["flash_attention_2_available"]:
            raise RuntimeError(
                "Flash Attention 2 was explicitly requested, but it is not available. "
                "Install a compatible flash-attn package in the project environment and run on CUDA. "
                f"Availability: {availability}"
            )
        if attn_implementation == "flash_attention_3" and not availability["flash_attention_3_available"]:
            raise RuntimeError(
                "Flash Attention 3 was explicitly requested, but it is not available. "
                "Install a compatible flash-attn-3/flash_attn_interface package and run on CUDA. "
                f"Availability: {availability}"
            )
        if _kernel_repo_from_attn_implementation(attn_implementation) and not availability["kernels_installed"]:
            raise RuntimeError(
                f"{attn_implementation!r} was explicitly requested, but the Hugging Face kernels "
                f"package is not installed. Availability: {availability}"
            )
    elif packing:
        logger.warning(
            "packing=true but no Flash Attention implementation was requested. "
            "TRL may warn about padding-free/packing attention compatibility."
        )

    return attn_implementation, flash_requested, availability


def _count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = trainable_params / total_params if total_params else 0.0
    return trainable_params, total_params, ratio


def _percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (len(sorted_values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return int(sorted_values[lower])
    low_value = sorted_values[lower]
    high_value = sorted_values[upper]
    return int(round(low_value + (high_value - low_value) * (rank - lower)))


def _sequence_length_stats(dataset: Any) -> dict[str, Any]:
    lengths: list[int] = []
    for example in dataset:
        input_ids = example.get("input_ids")
        if input_ids is None:
            continue
        lengths.append(len(input_ids))
    lengths_sorted = sorted(lengths)
    return {
        "count": len(lengths_sorted),
        "avg": (sum(lengths_sorted) / len(lengths_sorted)) if lengths_sorted else None,
        "p50": _percentile(lengths_sorted, 0.50),
        "p90": _percentile(lengths_sorted, 0.90),
        "p95": _percentile(lengths_sorted, 0.95),
        "p99": _percentile(lengths_sorted, 0.99),
        "max": max(lengths_sorted) if lengths_sorted else None,
    }


class TrainingMetadataCallback(TrainerCallback):
    def __init__(self, *, metadata_path: Path, metadata: dict[str, Any]) -> None:
        self.metadata_path = metadata_path
        self.metadata = metadata

    def on_train_begin(self, args, state, control, **kwargs):
        self.metadata["trainer_max_steps"] = state.max_steps
        self.metadata["trainer_num_train_epochs"] = args.num_train_epochs
        self.metadata["train_begin_timestamp"] = datetime.now(timezone.utc).isoformat()
        _write_training_metadata(self.metadata_path, self.metadata)

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            _write_training_metadata(checkpoint_dir / "training_metadata.json", self.metadata)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.metadata["latest_eval_loss"] = float(metrics["eval_loss"])
        self.metadata["best_model_checkpoint"] = getattr(state, "best_model_checkpoint", None)
        best_metric = getattr(state, "best_metric", None)
        if best_metric is not None:
            self.metadata["best_metric"] = float(best_metric)
            metric_name = str(getattr(args, "metric_for_best_model", "") or "")
            if metric_name in {"eval_loss", "loss"}:
                self.metadata["best_eval_loss"] = float(best_metric)
        _write_training_metadata(self.metadata_path, self.metadata)

    def on_train_end(self, args, state, control, **kwargs):
        self.metadata["best_model_checkpoint"] = getattr(state, "best_model_checkpoint", None)
        best_metric = getattr(state, "best_metric", None)
        if best_metric is not None:
            self.metadata["best_metric"] = float(best_metric)
            metric_name = str(getattr(args, "metric_for_best_model", "") or "")
            if metric_name in {"eval_loss", "loss"}:
                self.metadata["best_eval_loss"] = float(best_metric)
        epoch = getattr(state, "epoch", None)
        target_epochs = getattr(args, "num_train_epochs", None)
        if epoch is not None and target_epochs is not None and float(epoch) + 1e-6 < float(target_epochs):
            self.metadata["stopped_epoch"] = float(epoch)
        _write_training_metadata(self.metadata_path, self.metadata)


class TrainingHistoryCallback(TrainerCallback):
    def __init__(self, *, adapter_dir: Path, project_root: Path) -> None:
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
            upsert_history_row(self.adapter_csv_path, self.adapter_jsonl_path, row)
            upsert_history_row(self.central_csv_path, self.central_jsonl_path, row)
            self.seen_steps.add(step)
        except Exception as exc:
            logger.warning("Failed to write structured training history: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline v2 TRL SFTConfig LoRA trainer")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_format", default="text", choices=["text", "json"])
    parser.add_argument("--llm", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--packing", type=parse_bool, default=None)
    parser.add_argument("--packing_strategy", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--completion_only_loss", type=parse_bool, default=None)
    parser.add_argument("--assistant_only_loss", type=parse_bool, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", "--batch_size", dest="per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_strategy", default=None)
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
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fail_on_packing_warning", type=parse_bool, default=None)
    parser.add_argument("--allow_unpacked_ablation", type=parse_bool, default=None)
    parser.add_argument("--use_flash_attention", type=parse_bool, default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", default=None)
    parser.add_argument("--lora_task_type", default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    return parser.parse_args()


def _ensure_prompt_completion_dataset(dataset_obj: Any) -> None:
    required = {"prompt", "completion"}
    column_names = set(dataset_obj.column_names)
    missing = sorted(required - column_names)
    if missing:
        raise ValueError(
            f"Pipeline v2 requires prompt/completion columns. Missing: {missing}. "
            f"Available fields: {dataset_obj.column_names}"
        )


def _resolve_lora_config(cfg: dict[str, Any], args_cli: argparse.Namespace) -> tuple[LoraConfig, dict[str, Any]]:
    lora_section = get_section(cfg, "lora")
    lora_cfg = dict(lora_section)

    lora_r = int(get_param(args_cli, lora_cfg, "lora_r", 8, config_name="r"))
    lora_alpha = int(get_param(args_cli, lora_cfg, "lora_alpha", 16, config_name="lora_alpha"))
    lora_dropout = float(get_param(args_cli, lora_cfg, "lora_dropout", 0.05, config_name="lora_dropout"))
    lora_bias = str(get_param(args_cli, lora_cfg, "lora_bias", "none", config_name="bias"))
    task_type = str(get_param(args_cli, lora_cfg, "lora_task_type", "CAUSAL_LM", config_name="task_type"))
    use_dora = bool(get_param(args_cli, lora_cfg, "use_dora", False, config_name="use_dora"))
    target_modules = get_param(
        args_cli,
        lora_cfg,
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        config_name="target_modules",
    )
    if isinstance(target_modules, str):
        target_modules = "all-linear" if target_modules.strip() == "all-linear" else [target_modules]
    if not (
        target_modules == "all-linear"
        or (isinstance(target_modules, list) and all(isinstance(module, str) for module in target_modules))
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
            raise RuntimeError("Config requested use_dora=true, but installed peft does not support it.")
        lora_kwargs["use_dora"] = True

    return LoraConfig(**lora_kwargs), {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": lora_bias,
        "task_type": task_type,
        "use_dora": use_dora,
        "target_modules": target_modules,
    }


def _add_sft_arg(
    *,
    kwargs: dict[str, Any],
    signature_params: dict[str, Any],
    applied: dict[str, bool],
    name: str,
    value: Any,
) -> None:
    if name in signature_params:
        kwargs[name] = value
        applied[name] = True
    else:
        applied[name] = False


def _build_sft_config(SFTConfig: Any, values: dict[str, Any]) -> tuple[Any, dict[str, bool], dict[str, Any]]:
    signature_params = inspect.signature(SFTConfig.__init__).parameters
    if (
        "eval_strategy" in values
        and "eval_strategy" not in signature_params
        and "evaluation_strategy" in signature_params
    ):
        values = dict(values)
        values["evaluation_strategy"] = values.pop("eval_strategy")
    kwargs: dict[str, Any] = {}
    applied: dict[str, bool] = {}
    for name, value in values.items():
        _add_sft_arg(
            kwargs=kwargs,
            signature_params=signature_params,
            applied=applied,
            name=name,
            value=value,
        )

    missing_required = sorted(field for field in REQUIRED_SFT_FIELDS if not applied.get(field, False))
    if missing_required:
        raise RuntimeError(
            "Installed TRL SFTConfig does not expose required Pipeline v2 field(s): "
            + ", ".join(missing_required)
        )
    return SFTConfig(**kwargs), applied, kwargs


def _build_sft_trainer(
    *,
    SFTTrainer: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    sft_config: Any,
    lora_config: LoraConfig | None,
) -> Any:
    init_params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
    }
    if lora_config is not None:
        kwargs["peft_config"] = lora_config
    if eval_dataset is not None:
        if "eval_dataset" not in init_params:
            raise RuntimeError("Installed TRL SFTTrainer does not support eval_dataset.")
        kwargs["eval_dataset"] = eval_dataset
    if "processing_class" in init_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in init_params:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def _verify_label_masking(trainer: Any, *, sample_count: int = 4) -> dict[str, Any]:
    dataset = trainer.train_dataset
    collator = trainer.data_collator
    available = len(dataset)
    if available == 0:
        raise RuntimeError("Trainer train_dataset is empty; cannot verify label masking.")

    total_tokens = 0
    total_masked = 0
    total_completion = 0
    total_prompt = 0
    checked_samples = min(sample_count, available)
    for idx in range(checked_samples):
        example = dataset[idx]
        if "completion_mask" not in example:
            raise RuntimeError(
                "completion_only_loss=True did not produce completion_mask in the processed dataset."
            )
        batch = collator([example])
        labels = batch["labels"][0].detach().cpu()
        input_ids = batch["input_ids"][0].detach().cpu()
        completion_mask = torch.tensor(example["completion_mask"], dtype=torch.bool)
        if labels.numel() != completion_mask.numel():
            raise RuntimeError(
                "Label/completion_mask length mismatch: "
                f"labels={labels.numel()} completion_mask={completion_mask.numel()}"
            )
        prompt_positions = ~completion_mask
        completion_positions = completion_mask
        if int(completion_positions.sum().item()) == 0:
            raise RuntimeError("Processed sample has no completion tokens.")
        if torch.any(labels[prompt_positions] != -100):
            raise RuntimeError("Prompt tokens are not fully masked with -100.")
        if torch.any(labels[completion_positions] == -100):
            raise RuntimeError("Completion tokens contain -100 labels.")
        if torch.any(labels[completion_positions] != input_ids[completion_positions]):
            raise RuntimeError("Completion labels do not match input_ids.")
        total_tokens += int(labels.numel())
        total_masked += int((labels == -100).sum().item())
        total_completion += int(completion_positions.sum().item())
        total_prompt += int(prompt_positions.sum().item())

    masked_ratio = total_masked / total_tokens if total_tokens else 0.0
    result = {
        "verified": True,
        "checked_samples": checked_samples,
        "total_tokens_checked": total_tokens,
        "prompt_tokens_checked": total_prompt,
        "completion_tokens_checked": total_completion,
        "masked_tokens_checked": total_masked,
        "masked_ratio": masked_ratio,
    }
    logger.info("Label masking verified: %s", result)
    return result


def _verify_packing(
    *,
    trainer: Any,
    raw_dataset_len: int,
    max_length: int,
    packing: bool,
    fail_on_warning: bool,
) -> dict[str, Any]:
    processed_len = len(trainer.train_dataset)
    stats = _sequence_length_stats(trainer.train_dataset)
    avg_len = stats["avg"] or 0.0
    efficiency = avg_len / max_length if max_length else None
    has_seq_lengths = "seq_lengths" in set(trainer.train_dataset.column_names or [])
    expected_upper_bound = max(1, int(math.ceil(raw_dataset_len * 0.98)))
    plausible_reduction = (not packing) or processed_len < expected_upper_bound
    verified = (not packing) or (has_seq_lengths and plausible_reduction)
    result = {
        "verified": bool(verified),
        "raw_dataset_len": raw_dataset_len,
        "processed_train_dataset_len": processed_len,
        "has_seq_lengths": has_seq_lengths,
        "sequence_length_stats": stats,
        "packing_efficiency": efficiency,
        "plausible_reduction": plausible_reduction,
        "expected_processed_len_upper_bound": expected_upper_bound,
    }
    if packing and not verified:
        message = (
            "Packing verification failed or is implausible: "
            f"raw_len={raw_dataset_len}, processed_len={processed_len}, "
            f"has_seq_lengths={has_seq_lengths}"
        )
        if fail_on_warning:
            raise RuntimeError(message)
        logger.warning(message)
    else:
        logger.info("Packing verified: %s", result)
    return result


def main() -> None:
    args_cli = parse_args()
    setup_logging(args_cli.log_level, args_cli.log_format)

    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError("TRL with SFTConfig/SFTTrainer is required for Pipeline v2.") from exc

    cfg = load_config(args_cli.config) if args_cli.config else {}
    project_root = Path(__file__).resolve().parents[1]

    llm = str(get_param(args_cli, cfg, "llm", "qwen35_9b_base"))
    dataset_path_raw = str(
        get_param(
            args_cli,
            cfg,
            "dataset_path",
            "data/sql_create_context/train_sft_qwen35_9b_prompt_completion_v2_no_spider_dev_overlap.jsonl",
        )
    )
    output_dir_raw = str(
        get_param(
            args_cli,
            cfg,
            "output_dir",
            f"adapters/{llm}/lora_sqlctx_v2_completion_only_packing1024_no_overlap_epochs1",
        )
    )
    packing = bool(get_param(args_cli, cfg, "packing", True))
    packing_strategy = str(get_param(args_cli, cfg, "packing_strategy", "bfd"))
    max_length = int(get_param(args_cli, cfg, "max_length", 1024))
    completion_only_loss = bool(get_param(args_cli, cfg, "completion_only_loss", True))
    assistant_only_loss = bool(get_param(args_cli, cfg, "assistant_only_loss", False))
    num_train_epochs = float(get_param(args_cli, cfg, "num_train_epochs", 1))
    learning_rate = float(get_param(args_cli, cfg, "learning_rate", 1e-4))
    per_device_train_batch_size = int(get_param(args_cli, cfg, "per_device_train_batch_size", 4))
    gradient_accumulation_steps = int(get_param(args_cli, cfg, "gradient_accumulation_steps", 2))
    logging_steps = int(get_param(args_cli, cfg, "logging_steps", 10))
    save_strategy = _coerce_save_strategy(get_param(args_cli, cfg, "save_strategy", "epoch"))
    save_total_limit_raw = get_param(args_cli, cfg, "save_total_limit", 2)
    save_total_limit = int(save_total_limit_raw) if save_total_limit_raw is not None else None
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
    fail_on_packing_warning = bool(get_param(args_cli, cfg, "fail_on_packing_warning", True))
    allow_unpacked_ablation = bool(get_param(args_cli, cfg, "allow_unpacked_ablation", False))
    use_flash_attention_raw = get_param(args_cli, cfg, "use_flash_attention", None)
    attn_implementation_raw = get_param(args_cli, cfg, "attn_implementation", None)
    attn_implementation, flash_attention_requested, flash_attention_availability = _resolve_attention_settings(
        use_flash_attention_raw=use_flash_attention_raw,
        attn_implementation_raw=attn_implementation_raw,
        packing=packing,
    )

    if not completion_only_loss:
        raise ValueError("Pipeline v2 requires completion_only_loss=true.")
    if assistant_only_loss:
        raise ValueError("Pipeline v2 prompt/completion path requires assistant_only_loss=false.")
    if not packing and not allow_unpacked_ablation:
        raise ValueError(
            "Pipeline v2 final training requires packing=true. "
            "Set allow_unpacked_ablation=true only for an explicit no-packing ablation."
        )
    if not packing and allow_unpacked_ablation:
        logger.warning("Running explicit Pipeline v2 no-packing ablation.")
    if max_length != 1024:
        logger.warning("Pipeline v2 recommended max_length=1024, got %s", max_length)
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be true")

    dataset_path = Path(dataset_path_raw)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing v2 prompt/completion dataset: {dataset_path}")

    out_dir = Path(output_dir_raw)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
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
    _ensure_prompt_completion_dataset(train_dataset)
    raw_dataset_len = len(train_dataset)
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, raw_dataset_len)))
    selected_dataset_len = len(train_dataset)

    client = LLMClient(project_root)
    model_id = client.get_model_id(llm)
    tokenizer = client.get_tokenizer(llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = client.get_base_model(llm, attn_implementation=attn_implementation)
    effective_attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if gradient_checkpointing and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config, lora_meta = _resolve_lora_config(cfg, args_cli)
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    expected_steps_per_epoch, expected_total_steps = _estimate_steps(
        dataset_len=selected_dataset_len,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    sft_values = {
        "output_dir": str(checkpoint_dir),
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

    sft_config, applied_sft_fields, effective_sft_kwargs = _build_sft_config(SFTConfig, sft_values)
    _register_trl_flash_attention_variant(attn_implementation)
    trainer = _build_sft_trainer(
        SFTTrainer=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        sft_config=sft_config,
        lora_config=lora_config,
    )
    processed_train_dataset_len = len(trainer.train_dataset)
    trainer_expected_steps_per_epoch, trainer_expected_total_steps = _estimate_steps(
        dataset_len=processed_train_dataset_len,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    label_masking_result = _verify_label_masking(trainer)
    packing_result = _verify_packing(
        trainer=trainer,
        raw_dataset_len=selected_dataset_len,
        max_length=max_length,
        packing=packing,
        fail_on_warning=fail_on_packing_warning,
    )
    trainable_params, total_params, trainable_param_ratio = _count_trainable_parameters(trainer.model)

    history_csv_path, history_jsonl_path = history_paths(out_dir)
    central_history_csv_path, central_history_jsonl_path = central_metric_paths(project_root, out_dir)
    metadata_path = out_dir / "training_metadata.json"
    training_metadata: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "duration_human_readable": None,
        "trl_version": _package_version("trl"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "torch_version": _package_version("torch"),
        "datasets_version": _package_version("datasets"),
        "model_id": model_id,
        "attn_implementation_requested": attn_implementation,
        "attn_implementation_effective": effective_attn_implementation,
        "flash_attention_requested": flash_attention_requested,
        "flash_attention_availability": flash_attention_availability,
        "llm": llm,
        "adapter": out_dir.name,
        "dataset_path": str(dataset_path),
        "dataset_format": "prompt_completion",
        "completion_only_loss": completion_only_loss,
        "assistant_only_loss": assistant_only_loss,
        "packing": packing,
        "allow_unpacked_ablation": allow_unpacked_ablation,
        "packing_strategy": packing_strategy,
        "max_length": max_length,
        "raw_dataset_len": raw_dataset_len,
        "selected_dataset_len": selected_dataset_len,
        "processed_train_dataset_len": processed_train_dataset_len,
        "label_masking_verified": label_masking_result["verified"],
        "label_masking": label_masking_result,
        "packing_verified": packing_result["verified"],
        "packing_verification": packing_result,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_param_ratio": trainable_param_ratio,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "expected_steps_per_epoch_before_packing": expected_steps_per_epoch,
        "expected_total_steps_before_packing": expected_total_steps,
        "trainer_expected_steps_per_epoch": trainer_expected_steps_per_epoch,
        "trainer_expected_total_steps": trainer_expected_total_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "auto_resume": auto_resume,
        "overwrite_output_dir": overwrite_output_dir,
        "gradient_checkpointing": gradient_checkpointing,
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
        **_cuda_metadata(),
    }
    _write_training_metadata(metadata_path, training_metadata)

    logger.info("Pipeline v2 LoRA SFT setup")
    logger.info("model_id=%s", model_id)
    logger.info("attn_implementation_requested=%s", attn_implementation or "default")
    logger.info("attn_implementation_effective=%s", effective_attn_implementation)
    logger.info("flash_attention_requested=%s", flash_attention_requested)
    logger.info("flash_attention_availability=%s", flash_attention_availability)
    logger.info("packing=%s", packing)
    logger.info("dataset_path=%s", dataset_path)
    logger.info("raw_dataset_len=%d selected_dataset_len=%d", raw_dataset_len, selected_dataset_len)
    logger.info("processed_train_dataset_len=%d", processed_train_dataset_len)
    logger.info("effective SFTConfig kwargs=%s", effective_sft_kwargs)
    logger.info("label_masking=%s", label_masking_result)
    logger.info("packing_verification=%s", packing_result)
    logger.info(
        "trainable_params=%d total_params=%d trainable_ratio=%.8f",
        trainable_params,
        total_params,
        trainable_param_ratio,
    )

    trainer.add_callback(TrainingMetadataCallback(metadata_path=metadata_path, metadata=training_metadata))
    trainer.add_callback(TrainingHistoryCallback(adapter_dir=out_dir, project_root=project_root))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
    training_metadata.update(_cuda_metadata())
    _write_training_metadata(metadata_path, training_metadata)
    logger.info("Training runtime: %s total (%.2fs)", training_metadata["duration_human_readable"], duration_seconds)
    logger.info("Final train loss: %s", final_train_loss)

    trainer.model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved Pipeline v2 LoRA adapter to: %s", out_dir)
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
