#!/usr/bin/env python3
"""CPU-only tokenizer, packing, collator, and config preflight for Qwen v2.

No model or adapter classes are imported or loaded. The script mirrors the
tokenization, seed-42 shuffling, BFD packing, and full-chat collator behavior of
the current TRL SFT pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json",
    "configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json",
)
V1_VALIDATION = (
    "data/sql_create_context/val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42.jsonl"
)
MODEL_REGISTRY = {
    "qwen35_2b_base": {
        "model_id": "Qwen/Qwen3.5-2B-Base",
        "revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    },
    "qwen35_9b_base": {
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
    },
}
SUPPORTED_TOP_LEVEL_KEYS = {
    "llm", "dataset_path", "eval_dataset_path", "dataset_text_field", "output_dir",
    "continue_from_adapter", "packing", "packing_strategy", "max_length",
    "completion_only_loss", "assistant_only_loss", "use_flash_attention",
    "attn_implementation", "num_train_epochs", "learning_rate",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "gradient_checkpointing", "torch_compile",
    "torch_empty_cache_steps", "bf16", "fp16", "warmup_ratio", "lr_scheduler_type",
    "max_grad_norm", "logging_steps", "eval_strategy", "eval_steps",
    "eval_accumulation_steps", "prediction_loss_only", "save_strategy",
    "save_total_limit", "load_best_model_at_end", "metric_for_best_model",
    "greater_is_better", "save_best_model", "auto_resume", "overwrite_output_dir",
    "fail_on_packing_warning", "seed", "max_train_samples", "additional_epochs",
    "total_effective_epochs", "early_stopping", "test_mode", "lora",
}
SUPPORTED_EARLY_STOPPING_KEYS = {
    "enabled", "early_stopping_patience", "early_stopping_threshold", "metric"
}
SUPPORTED_LORA_KEYS = {
    "r", "lora_alpha", "lora_dropout", "bias", "task_type", "use_dora", "target_modules"
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_text_rows(path: Path) -> list[str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            text = row.get("text")
            require(isinstance(text, str) and text, f"Missing text at {path}:{line_number}")
            rows.append(text)
    return rows


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def length_stats(lengths: list[int], max_length: int) -> dict[str, Any]:
    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "median": quantile(lengths, 0.5),
        "p95": quantile(lengths, 0.95),
        "p99": quantile(lengths, 0.99),
        "max": max(lengths),
        "exactly_max_length": sum(length == max_length for length in lengths),
        "over_max_length": sum(length > max_length for length in lengths),
    }


def tokenize_texts(texts: list[str], tokenizer: Any) -> tuple[list[list[int]], list[int]]:
    encoded_rows: list[list[int]] = []
    lengths: list[int] = []
    eos = tokenizer.eos_token
    require(eos is not None, "Tokenizer lacks EOS token")
    for text in texts:
        tokenized_text = text if text.endswith(eos) else text + eos
        ids = list(tokenizer(tokenized_text, add_special_tokens=True, truncation=False)["input_ids"])
        encoded_rows.append(ids)
        lengths.append(len(ids))
    return encoded_rows, lengths


def pack_hash(dataset: Any) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(struct.pack("<I", len(row["input_ids"])))
        digest.update(struct.pack(f"<{len(row['input_ids'])}I", *row["input_ids"]))
        digest.update(struct.pack("<I", len(row["seq_lengths"])))
        digest.update(struct.pack(f"<{len(row['seq_lengths'])}I", *row["seq_lengths"]))
    return digest.hexdigest()


def pack_rows(encoded_rows: list[list[int]], seed: int, max_length: int, strategy: str) -> Any:
    from datasets import Dataset
    from trl import pack_dataset

    dataset = Dataset.from_dict({"input_ids": encoded_rows})
    dataset = dataset.shuffle(seed=seed)
    packed = pack_dataset(dataset, seq_length=max_length, strategy=strategy)
    return packed.shuffle(seed=seed)


def collator_audit(packed: Any, tokenizer: Any, raw_examples: int, batch_size: int) -> dict[str, Any]:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=2048,
        completion_only_loss=False,
        padding_free=True,
    )
    total_input = 0
    total_labels = 0
    total_masked = 0
    total_boundaries = 0
    batches = 0
    for start in range(0, len(packed), batch_size):
        examples = [packed[index] for index in range(start, min(start + batch_size, len(packed)))]
        require(all("completion_mask" not in row and "assistant_masks" not in row for row in examples), "Unexpected completion/assistant mask")
        batch = collator(examples)
        require("position_ids" in batch and "attention_mask" not in batch, "Padding-free collator mismatch")
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        positions = batch["position_ids"]
        require(input_ids.shape == labels.shape == positions.shape, "Collator tensor shape mismatch")
        masked = labels == -100
        require(bool((masked == (positions == 0)).all().item()), "Masked labels differ from document starts")
        trainable = ~masked
        require(bool((labels[trainable] == input_ids[trainable]).all().item()), "Full-chat labels differ from input IDs")
        total_input += int(input_ids.numel())
        total_labels += int(trainable.sum().item())
        total_masked += int(masked.sum().item())
        total_boundaries += sum(len(row["seq_lengths"]) for row in examples)
        batches += 1
    require(total_masked == total_boundaries == raw_examples, "Document-boundary masking count mismatch")
    return {
        "verified": True,
        "batches": batches,
        "padding_free": True,
        "completion_only_loss": False,
        "assistant_only_loss": False,
        "input_tokens": total_input,
        "trainable_labels": total_labels,
        "masked_document_starts": total_masked,
        "document_boundaries": total_boundaries,
        "padding_tokens": 0,
    }


def config_audit(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(config) - SUPPORTED_TOP_LEVEL_KEYS)
    unknown_early = sorted(set(config.get("early_stopping", {})) - SUPPORTED_EARLY_STOPPING_KEYS)
    unknown_lora = sorted(set(config.get("lora", {})) - SUPPORTED_LORA_KEYS)
    require(not unknown and not unknown_early and not unknown_lora, f"Unknown config keys: {unknown}, {unknown_early}, {unknown_lora}")
    llm = config["llm"]
    require(llm in MODEL_REGISTRY, f"Unknown model registry key: {llm}")
    expected = {
        "packing": True,
        "packing_strategy": "bfd",
        "max_length": 2048,
        "completion_only_loss": False,
        "assistant_only_loss": False,
        "num_train_epochs": 5,
        "learning_rate": 0.0001,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "bf16": False,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.03,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": 42,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"Unexpected {key} in {config_path}: {config.get(key)!r}")
    early = config["early_stopping"]
    require(early == {"enabled": True, "early_stopping_patience": 2, "early_stopping_threshold": 0.001, "metric": "eval_loss"}, "Early-stopping mismatch")
    lora = config["lora"]
    require(lora == {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM", "use_dora": False, "target_modules": "all-linear"}, "LoRA mismatch")
    train_path = resolve(config["dataset_path"])
    eval_path = resolve(config["eval_dataset_path"])
    output_path = resolve(config["output_dir"])
    require(train_path.is_file(), f"Training dataset missing: {train_path}")
    require(eval_path.is_file(), f"Validation missing: {eval_path}")
    require("full_chat_v2_mixed" in eval_path.name and "schemaheaderfix" in eval_path.name, "Wrong validation version")
    require("sqlcc_only" not in eval_path.name, "SQLCC-only validation referenced")
    require(not output_path.exists(), f"Output path collision: {output_path}")
    return {
        "config_path": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "unknown_keys": [],
        "registry_model_id": MODEL_REGISTRY[llm]["model_id"],
        "registry_revision": MODEL_REGISTRY[llm]["revision"],
        "train_path": relative(train_path),
        "train_sha256": sha256_file(train_path),
        "validation_path": relative(eval_path),
        "validation_sha256": sha256_file(eval_path),
        "output_path": relative(output_path),
        "output_collision": False,
        "effective_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
        "checkpoint_retention": {
            "max_epochs": 5,
            "save_total_limit": 5,
            "all_completed_epoch_checkpoints_retained": True,
            "best_checkpoint_retained": True,
            "last_checkpoint_retained": True,
            "root_export_source": "load_best_model_at_end restores best weights before root save",
        },
    }


def run_model_preflight(config_path: Path, v1_texts: list[str]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = load_json(config_path)
    config_result = config_audit(config_path, config)
    registry = MODEL_REGISTRY[config["llm"]]
    tokenizer = AutoTokenizer.from_pretrained(
        registry["model_id"], revision=registry["revision"], local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_texts = load_text_rows(resolve(config["dataset_path"]))
    eval_texts = load_text_rows(resolve(config["eval_dataset_path"]))
    require(len(train_texts) == 25000, "Training row count is not 25000")
    require(len(eval_texts) == 2500, "Validation row count is not 2500")
    require(len(v1_texts) == len(eval_texts), "v1/v2 validation size mismatch")
    train_encoded, train_lengths = tokenize_texts(train_texts, tokenizer)
    v1_encoded, v1_lengths = tokenize_texts(v1_texts, tokenizer)
    eval_encoded, eval_lengths = tokenize_texts(eval_texts, tokenizer)
    deltas = [new - old for old, new in zip(v1_lengths, eval_lengths)]
    require(set(deltas) == {-4}, f"Header-fix token delta is not -4 for every row: {Counter(deltas)}")
    require(max(train_lengths) <= 2048 and max(eval_lengths) <= 2048, "Raw example exceeds max_length")
    train_packed = pack_rows(train_encoded, config["seed"], config["max_length"], config["packing_strategy"])
    eval_packed = pack_rows(eval_encoded, config["seed"], config["max_length"], config["packing_strategy"])
    train_hash = pack_hash(train_packed)
    eval_hash = pack_hash(eval_packed)
    train_repeat = pack_rows(train_encoded, config["seed"], config["max_length"], config["packing_strategy"])
    eval_repeat = pack_rows(eval_encoded, config["seed"], config["max_length"], config["packing_strategy"])
    require(pack_hash(train_repeat) == train_hash, "Training packing is not reproducible")
    require(pack_hash(eval_repeat) == eval_hash, "Validation packing is not reproducible")
    del train_repeat, eval_repeat
    train_examples = sum(len(row["seq_lengths"]) for row in train_packed)
    eval_examples = sum(len(row["seq_lengths"]) for row in eval_packed)
    require(train_examples == 25000 and eval_examples == 2500, "Packing changed example counts")
    train_tokens = sum(len(row["input_ids"]) for row in train_packed)
    eval_tokens = sum(len(row["input_ids"]) for row in eval_packed)
    require(train_tokens == sum(train_lengths), "Training packing changed token count")
    require(eval_tokens == sum(eval_lengths), "Validation packing changed token count")
    train_collator = collator_audit(train_packed, tokenizer, 25000, config["per_device_train_batch_size"])
    eval_collator = collator_audit(eval_packed, tokenizer, 2500, config["per_device_eval_batch_size"])
    steps_per_epoch = math.ceil(
        len(train_packed) /
        (config["per_device_train_batch_size"] * config["gradient_accumulation_steps"])
    )
    checkpoints = [steps_per_epoch * epoch for epoch in range(1, config["num_train_epochs"] + 1)]
    return {
        "config": config_result,
        "tokenizer": {
            "id": registry["model_id"],
            "revision": registry["revision"],
            "class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "raw_lengths": {
            "train": length_stats(train_lengths, 2048),
            "validation_v1": length_stats(v1_lengths, 2048),
            "validation_v2": length_stats(eval_lengths, 2048),
            "v2_minus_v1_distribution": dict(Counter(deltas)),
            "truncated_examples": 0,
        },
        "training_packing": {
            "raw_examples": 25000,
            "raw_tokens_with_eos": sum(train_lengths),
            "packed_sequences": len(train_packed),
            "packed_examples": train_examples,
            "packed_tokens": train_tokens,
            "max_pack_length": max(len(row["input_ids"]) for row in train_packed),
            "empty_packs": sum(not row["input_ids"] for row in train_packed),
            "pack_sha256": train_hash,
            "reproducible_seed_42": True,
            "collator": train_collator,
        },
        "validation_packing": {
            "raw_examples": 2500,
            "raw_tokens_with_eos": sum(eval_lengths),
            "packed_sequences": len(eval_packed),
            "packed_examples": eval_examples,
            "packed_tokens": eval_tokens,
            "max_pack_length": max(len(row["input_ids"]) for row in eval_packed),
            "empty_packs": sum(not row["input_ids"] for row in eval_packed),
            "pack_sha256": eval_hash,
            "reproducible_seed_42": True,
            "collator": eval_collator,
        },
        "steps": {
            "packs_per_epoch": len(train_packed),
            "effective_batch_size": config_result["effective_batch_size"],
            "optimizer_steps_per_epoch": steps_per_epoch,
            "maximum_optimizer_steps": steps_per_epoch * config["num_train_epochs"],
            "expected_epoch_checkpoints": checkpoints,
            "expected_checkpoint_names": [f"checkpoint-{step}" for step in checkpoints],
        },
        "model_loaded": False,
        "adapter_loaded": False,
        "cuda_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=2, default=list(DEFAULT_CONFIGS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [resolve(value).resolve() for value in args.configs]
    for path in config_paths:
        require(path.is_file(), f"Config missing: {path}")
    v1_path = resolve(V1_VALIDATION)
    require(v1_path.is_file(), f"v1 validation missing: {v1_path}")
    v1_texts = load_text_rows(v1_path)
    results = [run_model_preflight(path, v1_texts) for path in config_paths]
    require(results[0]["config"]["train_sha256"] == results[1]["config"]["train_sha256"], "old25k aliases are not byte-identical")
    require(results[0]["config"]["validation_sha256"] == results[1]["config"]["validation_sha256"], "Validation differs between models")
    require(results[0]["raw_lengths"] == results[1]["raw_lengths"], "2B/9B token lengths differ")
    require(results[0]["training_packing"]["pack_sha256"] == results[1]["training_packing"]["pack_sha256"], "2B/9B training packing differs")
    require(results[0]["validation_packing"]["pack_sha256"] == results[1]["validation_packing"]["pack_sha256"], "2B/9B validation packing differs")
    summary = {
        "status": "PASS",
        "preflight_script": relative(Path(__file__)),
        "preflight_script_sha256": sha256_file(Path(__file__)),
        "v1_validation": relative(v1_path),
        "v1_validation_sha256": sha256_file(v1_path),
        "models": results,
        "cross_model": {
            "old25k_byte_identical": True,
            "validation_identical": True,
            "token_lengths_identical": True,
            "training_packing_identical": True,
            "validation_packing_identical": True,
            "effective_batch_size_identical": True,
            "methodology_identical": True,
            "only_allowed_differences": ["llm registry key", "model/tokenizer revision", "byte-identical old25k filename alias", "output path"],
        },
        "model_loaded": False,
        "adapter_loaded": False,
        "cuda_used": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
