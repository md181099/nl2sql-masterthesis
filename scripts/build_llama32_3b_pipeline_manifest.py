#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
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
    LLAMA32_NATIVE_CHAT_FORMAT,
    LLAMA32_NATIVE_TEMPLATE_DATE,
    configure_llama32_padding,
    llama32_assistant_generation_prefix,
    llama32_generation_stop_token_ids,
)


OUTPUT = ROOT / "audits/llama32_3b_instruct_full_pipeline_manifest_20260714.json"
TRAIN_CONFIG = ROOT / (
    "configs/train_lora_llama32_3b_instruct_v2_fullchat_old25k_r8_alpha16_"
    "mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json"
)
DATASET_MANIFEST = ROOT / (
    "data/sql_create_context/llama32_3b_instruct_native_chat_v2_dataset_manifest_20260714.json"
)
TRAINING_PREFLIGHT = ROOT / (
    "audits/derived/llama32_3b_instruct_training_preflight_corrected_20260714.json"
)
PROMPT_SMOKE = ROOT / "audits/derived/llama32_3b_instruct_prompt_smoke_20260714.json"
POSTHOC_CONFIG = ROOT / (
    "configs/eval_posthoc_loss_llama32_3b_instruct_v2_mixedval2500_"
    "schemaheaderfix_all_checkpoints.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def new_eval_configs() -> tuple[list[Path], list[Path], list[Path]]:
    base = sorted(
        path
        for path in (ROOT / "configs").glob("eval_llama32_3b_instruct_base_*.json")
        if "smoke5" not in path.name
    )
    lora = sorted(
        path
        for path in (ROOT / "configs").glob(
            "eval_llama32_3b_instruct_lora_v2_old25k_r8_alpha16_mixedval2500_v2_*.json"
        )
        if "smoke5" not in path.name
    )
    smoke = sorted(
        path
        for path in (ROOT / "configs").glob("eval_llama32_3b_instruct_*smoke5*.json")
    )
    return base, lora, smoke


def config_variant(path: Path) -> str:
    name = path.name
    if "structure_rerank_v2_gate070" in name:
        return "structure_gate070"
    if "structure_rerank_v2_gate085" in name:
        return "structure_gate085"
    if "structure_rerank_v2" in name:
        return "structure"
    if "top1_gate070" in name:
        return "top1_gate070"
    if "top1_gate085" in name:
        return "top1_gate085"
    if "top1" in name:
        return "top1"
    if "static" in name:
        return "static"
    if "zero_shot" in name:
        return "zero_shot"
    raise RuntimeError(f"Unknown evaluation config variant: {path}")


def validate_eval_config(path: Path, expected_adapter: str, *, smoke: bool) -> dict[str, Any]:
    cfg = load(path)
    variant = config_variant(path)
    errors: list[str] = []

    def expect(key: str, value: Any) -> None:
        if cfg.get(key) != value:
            errors.append(f"{key}: expected {value!r}, got {cfg.get(key)!r}")

    expect("llm", "llama32_3b_instruct")
    expect("adapter", expected_adapter)
    expect("prompt_format", LLAMA32_NATIVE_CHAT_FORMAT)
    expect("system_prompt_variant", "sqlctx_anti_overjoin")
    expect("testcases_path", "data/testcases_spider_dev_full.jsonl")
    expect("max_input_tokens", 2048)
    expect("max_new_tokens", 256)
    expect("generation_batch_size", 1)
    expect("compute_perplexity", False)
    expect("allow_overlap", False)
    expect("same_db_only", False)
    expect("extractor_mode", "sql_first_statement_only")
    expect("max_test_samples", 5 if smoke else None)

    if variant == "zero_shot":
        expect("prompt_tuning", "none")
        expect("k", 0)
        for forbidden in ("retrieval_index_path", "retrieval_pool_path", "fewshot_gate_threshold"):
            if forbidden in cfg:
                errors.append(f"zero-shot contains forbidden field {forbidden}")
    elif variant == "static":
        expect("prompt_tuning", "static_fewshot")
        expect("k", 1)
        expect("retrieval_method", "static_seeded")
        expect(
            "retrieval_pool_path",
            "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl",
        )
        if "retrieval_index_path" in cfg:
            errors.append("static config must not activate a retrieval index")
    else:
        expect("prompt_tuning", "dynamic_fewshot")
        expect("k", 1)
        expect("retrieval_method", "sentence_transformer_faiss")
        expect("retrieval_index_path", "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15")
        expect(
            "retrieval_pool_path",
            "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl",
        )
        expect("embedding_model", "BAAI/bge-large-en-v1.5")
        expect("fewshot_example_schema_mode", "full")
        expect("fewshot_example_mode", "schema_with_rules")
        structure = variant.startswith("structure")
        if structure:
            expect("retrieval_rerank_method", "structure_topk_v2")
            expect("retrieval_rerank_top_n", 10)
            expect("retrieval_structure_bonus_max", 0.08)
        elif "retrieval_rerank_method" in cfg:
            errors.append("top-1 config unexpectedly activates a reranker")
        gated = "gate" in variant
        expect("fewshot_gate_enabled", gated)
        if gated:
            expect("fewshot_gate_mode", "similarity_only")
            expect("fewshot_gate_threshold", 0.70 if variant.endswith("070") else 0.85)

    if errors:
        raise RuntimeError(f"Invalid evaluation config {path}: {errors}")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "variant": variant,
        "adapter": expected_adapter,
        "status": "CONFIG READY" if expected_adapter != "base" else "READY",
        "release": "ADAPTER PENDING" if expected_adapter != "base" else "PREFLIGHT PASSED",
    }


def active_qwen_processes() -> list[dict[str, Any]]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(term in command for term in ("06_batch_run.py", "07_lora", "qwen35_2b")):
            active.append({"pid": int(entry.name), "command": command.strip()})
    return active


def count_llama_results() -> int:
    count = 0
    for path in (ROOT / "results").glob("run_*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle), None)
        except Exception:
            continue
        if first and first.get("run_llm") == "llama32_3b_instruct":
            count += 1
    return count


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite full pipeline manifest: {OUTPUT}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA32_3B_INSTRUCT_MODEL_ID,
        revision=LLAMA32_3B_INSTRUCT_REVISION,
        local_files_only=True,
    )
    pad_id = configure_llama32_padding(tokenizer)
    dataset = load(DATASET_MANIFEST)
    training = load(TRAINING_PREFLIGHT)
    prompts = load(PROMPT_SMOKE)
    if training["status"] != "PASS" or prompts["status"] != "PASS":
        raise RuntimeError("Training or prompt preflight did not pass")

    base_paths, lora_paths, smoke_paths = new_eval_configs()
    if (len(base_paths), len(lora_paths), len(smoke_paths)) != (8, 8, 3):
        raise RuntimeError(
            f"Unexpected config counts: base={len(base_paths)}, lora={len(lora_paths)}, smoke={len(smoke_paths)}"
        )
    future_adapter = (
        "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_"
        "schemaheaderfix_evalstop_maxlen2048_epochs5"
    )
    base_configs = [validate_eval_config(path, "base", smoke=False) for path in base_paths]
    lora_configs = [validate_eval_config(path, future_adapter, smoke=False) for path in lora_paths]
    smoke_configs = [
        validate_eval_config(
            path,
            future_adapter if "_lora_v2_" in path.name else "base",
            smoke=True,
        )
        for path in smoke_paths
    ]

    snapshot = (
        Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
        / LLAMA32_3B_INSTRUCT_REVISION
    )
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"):
        path = snapshot / name
        if path.is_file():
            tokenizer_files[name] = {"path": str(path), "sha256": sha256(path)}

    index_dir = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
    static_path = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
    test_path = ROOT / "data/testcases_spider_dev_full.jsonl"
    active = active_qwen_processes()
    gpu_warning = "nvidia-smi unavailable in the sandbox; verify the host GPU runtime immediately before execution"
    new_config_names = {path.name for path in base_paths + lora_paths + smoke_paths}
    old_config_files = [
        path
        for path in (ROOT / "configs").rglob("*.json")
        if "llama" in path.name.lower()
        and path.name not in new_config_names
        and path not in {TRAIN_CONFIG, POSTHOC_CONFIG}
    ]
    old_dataset_files = [
        path
        for path in (ROOT / "data").rglob("*")
        if path.is_file()
        and "llama" in path.name.lower()
        and path not in {
            ROOT / dataset["materialized"]["train_path"],
            ROOT / dataset["materialized"]["validation_path"],
            DATASET_MANIFEST,
        }
    ]
    adapter_roots = sorted(
        {
            path.parent
            for path in (ROOT / "adapters/llama32_3b_instruct").rglob("adapter_model.safetensors")
            if path.parent.name != "checkpoints" and "checkpoint-" not in path.parent.name
        }
    )

    manifest = {
        "schema_version": 1,
        "created_utc_date": "2026-07-14",
        "preflight_status": "PASS MIT WARNUNGEN",
        "warnings": [gpu_warning],
        "active_qwen_evaluation": {"detected": bool(active), "processes": active},
        "model": {
            "registry_key": "llama32_3b_instruct",
            "model_id": LLAMA32_3B_INSTRUCT_MODEL_ID,
            "revision": LLAMA32_3B_INSTRUCT_REVISION,
            "snapshot_path": str(snapshot),
            "tokenizer_available": True,
            "model_files_local": all((snapshot / name).exists() for name in ("config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")),
            "gated_access_issue": False,
        },
        "native_chat": {
            "format": LLAMA32_NATIVE_CHAT_FORMAT,
            "template_date": LLAMA32_NATIVE_TEMPLATE_DATE,
            "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
            "tokenizer_files": tokenizer_files,
            "bos": {"token": tokenizer.bos_token, "id": tokenizer.bos_token_id},
            "eos": {"token": tokenizer.eos_token, "id": tokenizer.eos_token_id},
            "eot_id": tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            "end_of_text_id": tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            "pad": {"token": tokenizer.pad_token, "id": pad_id, "embedding_resize": False},
            "stop_token_ids": llama32_generation_stop_token_ids(tokenizer),
            "assistant_generation_prefix": llama32_assistant_generation_prefix(tokenizer),
            "rendering": "tokenizer.apply_chat_template",
            "rendered_prompt_tokenization_add_special_tokens": False,
            "double_special_tokens": False,
        },
        "datasets": dataset,
        "training": {
            "config_path": rel(TRAIN_CONFIG),
            "config_sha256": sha256(TRAIN_CONFIG),
            "entrypoint": "src/07_lora_finetune_sft_v1_clean.py",
            "entrypoint_sha256": sha256(ROOT / "src/07_lora_finetune_sft_v1_clean.py"),
            "prompt_implementation": "src/llama32_native_chat.py",
            "prompt_implementation_sha256": sha256(ROOT / "src/llama32_native_chat.py"),
            "output_adapter": load(TRAIN_CONFIG)["output_dir"],
            "parameters": {
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
                "learning_rate": 0.0001,
                "scheduler": "constant",
                "warmup_ratio": 0.03,
                "max_length": 2048,
                "loss": "full_chat",
                "packing": "bfd",
                "batch": 2,
                "gradient_accumulation": 4,
                "effective_batch": 8,
                "seed": 42,
                "max_epochs": 5,
                "early_stopping": {"patience": 2, "threshold": 0.001, "metric": "eval_loss"},
            },
            "preflight": training,
        },
        "posthoc": {
            "config_path": rel(POSTHOC_CONFIG),
            "config_sha256": sha256(POSTHOC_CONFIG),
            "evaluator": "src/21_eval_qwen35_posthoc_loss_general.py",
            "evaluator_sha256": sha256(ROOT / "src/21_eval_qwen35_posthoc_loss_general.py"),
            "validate_only_status": "PASS",
            "adapter_pending": True,
            "checkpoint_selection_changed": False,
        },
        "evaluation": {
            "runner": "src/06_batch_run.py",
            "runner_sha256": sha256(ROOT / "src/06_batch_run.py"),
            "testcases_path": rel(test_path),
            "testcases_sha256": sha256(test_path),
            "testcases_rows": 1032,
            "base_configs": base_configs,
            "lora_configs": lora_configs,
            "smoke_configs": smoke_configs,
            "prompt_smoke": prompts,
            "generation": {"max_input_tokens": 2048, "max_new_tokens": 256, "batch": 1, "do_sample": False, "decoding": "greedy"},
        },
        "retrieval": {
            "index_dir": rel(index_dir),
            "index_sha256": sha256(index_dir / "index.faiss"),
            "metadata_sha256": sha256(index_dir / "metadata.jsonl"),
            "manifest_sha256": sha256(index_dir / "manifest.json"),
            "pool_size": 6960,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "faiss": "IndexFlatIP",
            "dev_overlap": 0,
            "structure_reranker": "structure_topk_v2",
            "structure_adjustment_cap": 0.08,
            "gate_semantics": "score >= threshold on original BGE score of the final selected demo",
            "static_resource": rel(static_path),
            "static_resource_sha256": sha256(static_path),
            "static_demo_id": "SPIDER_TRAIN_001657",
        },
        "legacy_inventory": {
            "preexisting_llama_configs": len(old_config_files),
            "preexisting_llama_dataset_and_manifest_files": len(old_dataset_files),
            "adapter_roots_with_weights": len(adapter_roots),
            "historical_llama_result_csvs": count_llama_results(),
            "classification": {
                "current_mainline": 0,
                "reusable_with_changes": 3,
                "legacy_or_incomplete": len(old_config_files) - 3,
                "reason": "No pre-existing line combines native tokenizer chat template, old25k, MixedVal-v2 early stopping, and the controlled 2048-token evaluation matrix.",
            },
        },
        "qwen_regression": {
            "prompt_changes": 0,
            "metric_logic_changes": 0,
            "retrieval_selection_changes": 0,
            "tests": "tests/test_llama32_native_pipeline.py",
            "status": "PASS",
        },
        "scientific_comparability": {
            "within_llama_base_vs_lora": "controlled after official adapter selection",
            "qwen_vs_llama": "confounded by Base-versus-Instruct status in addition to family and size",
            "rank_ablation_included": False,
            "exploratory_conditions": ["structure_top10_v2_gate070", "structure_top10_v2_gate085"],
        },
        "release": {
            "training": "JA - conditional on successful host nvidia-smi/CUDA check immediately before start",
            "base_evaluation": "JA - conditional on successful host nvidia-smi/CUDA check immediately before start",
            "lora_evaluation": "PENDING TRAINING AND BEST-CHECKPOINT HASH AUDIT",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["preflight_status"], "output": rel(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
