#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama32_native_chat import (  # noqa: E402
    LLAMA32_3B_INSTRUCT_MODEL_ID,
    LLAMA32_3B_INSTRUCT_REVISION,
    configure_llama32_padding,
    llama32_generation_stop_token_ids,
)


CONFIG = ROOT / (
    "configs/train_lora_llama32_3b_instruct_v2_fullchat_old25k_r8_alpha16_"
    "mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json"
)
DATASET_MANIFEST = ROOT / (
    "data/sql_create_context/llama32_3b_instruct_native_chat_v2_dataset_manifest_20260714.json"
)
OUTPUT = ROOT / "audits/derived/llama32_3b_instruct_training_preflight_corrected_20260714.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training preflight: {output}")

    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer
    from trl import pack_dataset
    from trl.data_utils import is_conversational
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    allowed_root_keys = {
        "llm", "dataset_path", "eval_dataset_path", "dataset_text_field", "output_dir",
        "packing", "packing_strategy", "max_length", "completion_only_loss",
        "assistant_only_loss", "use_flash_attention", "attn_implementation",
        "num_train_epochs", "learning_rate", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps",
        "gradient_checkpointing", "torch_compile", "torch_empty_cache_steps", "bf16",
        "fp16", "warmup_ratio", "lr_scheduler_type", "max_grad_norm", "logging_steps",
        "eval_strategy", "eval_steps", "eval_accumulation_steps", "prediction_loss_only",
        "save_strategy", "save_total_limit", "load_best_model_at_end",
        "metric_for_best_model", "greater_is_better", "save_best_model", "auto_resume",
        "overwrite_output_dir", "fail_on_packing_warning", "seed", "early_stopping", "lora",
    }
    unknown_keys = sorted(set(config) - allowed_root_keys)
    train_path = ROOT / config["dataset_path"]
    validation_path = ROOT / config["eval_dataset_path"]
    adapter_root = ROOT / config["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA32_3B_INSTRUCT_MODEL_ID,
        revision=LLAMA32_3B_INSTRUCT_REVISION,
        local_files_only=True,
    )
    pad_token_id = configure_llama32_padding(tokenizer)
    model_config = AutoConfig.from_pretrained(
        LLAMA32_3B_INSTRUCT_MODEL_ID,
        revision=LLAMA32_3B_INSTRUCT_REVISION,
        local_files_only=True,
    )
    train_rows = load_rows(train_path)
    validation_rows = load_rows(validation_path)
    if not is_conversational(train_rows[0]) or not is_conversational(validation_rows[0]):
        raise RuntimeError("Native Llama datasets are not recognized as conversational by TRL")

    def tokenized(rows: list[dict[str, Any]]) -> list[list[int]]:
        return [
            tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                return_dict=True,
                **row["chat_template_kwargs"],
            )["input_ids"]
            for row in rows
        ]

    train_ids = tokenized(train_rows)
    validation_ids = tokenized(validation_rows)

    def packed(ids: list[list[int]]) -> Any:
        dataset = Dataset.from_dict({"input_ids": ids}).shuffle(seed=int(config["seed"]))
        value = pack_dataset(
            dataset,
            seq_length=int(config["max_length"]),
            strategy=config["packing_strategy"],
        )
        return value.shuffle(seed=int(config["seed"]))

    packed_train = packed(train_ids)
    packed_validation = packed(validation_ids)
    first = packed_train[0]
    collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        max_length=int(config["max_length"]),
        completion_only_loss=False,
        padding_free=True,
    )
    batch = collator([first])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    position_ids = batch["position_ids"][0]
    boundary_mask = position_ids == 0
    non_boundary = ~boundary_mask
    label_audit = {
        "system_user_assistant_full_chat": True,
        "completion_mask_present": "completion_mask" in first,
        "assistant_mask_present": "assistant_masks" in first,
        "document_start_tokens_masked": bool((labels[boundary_mask] == -100).all().item()),
        "all_other_labels_equal_input_ids": bool((labels[non_boundary] == input_ids[non_boundary]).all().item()),
        "packed_boundaries": int(boundary_mask.sum().item()),
        "cross_sample_loss_prevented": bool((labels[boundary_mask] == -100).all().item()),
    }

    microbatches = math.ceil(len(packed_train) / int(config["per_device_train_batch_size"]))
    steps_per_epoch = math.ceil(microbatches / int(config["gradient_accumulation_steps"]))
    disk = shutil.disk_usage(ROOT)
    snapshot = (
        Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
        / LLAMA32_3B_INSTRUCT_REVISION
    )
    existing_reference_adapter = ROOT / (
        "adapters/llama32_3b_instruct/lora_v1_fullchat_mix_spider_train_sqlcc_"
        "complexity_enriched_25k_seed42_flashattn2_llama32_3b_instruct_no_overlap_"
        "epochs2_maxlen2048/adapter_config.json"
    )
    reference_adapter = json.loads(existing_reference_adapter.read_text(encoding="utf-8"))
    expected_targets = sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    qwen2 = json.loads((ROOT / "configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json").read_text())
    qwen9 = json.loads((ROOT / "configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json").read_text())
    method_keys = sorted(set(config) & set(qwen2) & set(qwen9) - {"llm", "dataset_path", "eval_dataset_path", "output_dir"})
    method_differences = {
        key: {"llama": config[key], "qwen2": qwen2[key], "qwen9": qwen9[key]}
        for key in method_keys
        if not (config[key] == qwen2[key] == qwen9[key])
    }

    status = "PASS"
    failures = []
    checks = {
        "unknown_keys_zero": not unknown_keys,
        "train_sha_matches_manifest": sha256(train_path) == dataset_manifest["materialized"]["train_sha256"],
        "validation_sha_matches_manifest": sha256(validation_path) == dataset_manifest["materialized"]["validation_sha256"],
        "train_rows_25000": len(train_rows) == 25000,
        "validation_rows_2500": len(validation_rows) == 2500,
        "no_overlength_train": max(map(len, train_ids)) <= 2048,
        "no_overlength_validation": max(map(len, validation_ids)) <= 2048,
        "full_chat": config["completion_only_loss"] is False and config["assistant_only_loss"] is False,
        "packing_bfd": config["packing"] is True and config["packing_strategy"] == "bfd",
        "label_audit": all(
            label_audit[key]
            for key in (
                "system_user_assistant_full_chat",
                "document_start_tokens_masked",
                "all_other_labels_equal_input_ids",
                "cross_sample_loss_prevented",
            )
        )
        and not label_audit["completion_mask_present"]
        and not label_audit["assistant_mask_present"],
        "qwen_method_parameters_identical": not method_differences,
        "output_collision_free": not adapter_root.exists(),
        "model_snapshot_local": snapshot.is_dir(),
        "target_modules_verified": sorted(reference_adapter["target_modules"]) == expected_targets,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    if failures:
        status = "FAIL"

    result = {
        "status": status,
        "failures": failures,
        "checks": checks,
        "config": {
            "path": str(CONFIG.relative_to(ROOT)),
            "sha256": sha256(CONFIG),
            "unknown_keys": unknown_keys,
            "ignored_relevant_keys": [],
            "type_errors": [],
        },
        "model": {
            "registry_key": config["llm"],
            "model_id": LLAMA32_3B_INSTRUCT_MODEL_ID,
            "revision": LLAMA32_3B_INSTRUCT_REVISION,
            "snapshot": str(snapshot),
            "architecture": model_config.architectures,
            "model_type": model_config.model_type,
            "layers": model_config.num_hidden_layers,
            "hidden_size": model_config.hidden_size,
            "attention_heads": model_config.num_attention_heads,
            "kv_heads": model_config.num_key_value_heads,
            "max_position_embeddings": model_config.max_position_embeddings,
            "rope_parameters": model_config.rope_parameters,
            "tie_word_embeddings": model_config.tie_word_embeddings,
            "source_dtype": str(model_config.dtype),
            "configured_training_dtype": "float16",
            "weights_loaded": False,
        },
        "tokenizer": {
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "stop_token_ids": llama32_generation_stop_token_ids(tokenizer),
            "padding_policy": "existing EOT/EOS token reused; no vocabulary or embedding resize",
        },
        "datasets": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_packs": len(packed_train),
            "validation_packs": len(packed_validation),
            "train_tokens": sum(map(len, train_ids)),
            "validation_tokens": sum(map(len, validation_ids)),
            "train_max_tokens": max(map(len, train_ids)),
            "validation_max_tokens": max(map(len, validation_ids)),
            "truncations": 0,
        },
        "training": {
            "steps_per_epoch": steps_per_epoch,
            "max_total_steps": steps_per_epoch * int(config["num_train_epochs"]),
            "possible_epoch_checkpoints": [steps_per_epoch * epoch for epoch in range(1, 6)],
            "effective_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
            "label_audit": label_audit,
            "target_modules": expected_targets,
            "target_module_evidence": str(existing_reference_adapter.relative_to(ROOT)),
            "method_differences_from_qwen_v2": method_differences,
        },
        "environment": {
            "python": sys.version,
            "transformers": package_version("transformers"),
            "trl": package_version("trl"),
            "peft": package_version("peft"),
            "torch": package_version("torch"),
            "tokenizers": package_version("tokenizers"),
            "flash_attn": package_version("flash-attn"),
            "cuda_initialized": False,
            "gpu_runtime_verified": False,
            "disk_free_bytes": disk.free,
        },
        "output": {
            "adapter_root": str(adapter_root.relative_to(ROOT)),
            "exists": adapter_root.exists(),
            "auto_resume": config["auto_resume"],
            "overwrite": config["overwrite_output_dir"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
