#!/usr/bin/env python3
"""Read-only final analysis of Qwen 3.5 2B Base and LoRA-v2 runs.

The script reads completed evaluation artifacts and Spider SQLite databases. It
never loads a language model, adapter, tokenizer, or embedding model. All
outputs are additive and opened with exclusive creation.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
INDEX_DIR = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
STATIC_RESOURCE = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
ADAPTER_ROOT = ROOT / "adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
BEST_CHECKPOINT = ADAPTER_ROOT / "checkpoints/checkpoint-502"
TRAIN_CONFIG = ROOT / "configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json"
TRAINING_AUDIT = ROOT / "audits/audit_qwen35_2b_v2_generative_eval_configs_preflight_20260714.md"
TRAINING_MANIFEST = ROOT / "audits/qwen35_2b_v2_generative_eval_configs_manifest_20260714.json"

OUT_SUMMARY = ROOT / "audits/derived/qwen35_2b_base_and_lora_v2_evaluation_summary_20260715.json"
OUT_CASES = ROOT / "audits/derived/qwen35_2b_base_vs_lora_v2_case_comparison_20260715.csv"
OUT_BASE_LORA = ROOT / "audits/derived/qwen35_2b_base_vs_lora_v2_paired_statistics_20260715.csv"
OUT_LORA_FS = ROOT / "audits/derived/qwen35_2b_lora_v2_fewshot_paired_statistics_20260715.csv"
OUT_BASE_FS = ROOT / "audits/derived/qwen35_2b_base_fewshot_paired_statistics_20260715.csv"
OUT_SIMILARITY = ROOT / "audits/derived/qwen35_2b_lora_v2_similarity_bin_analysis_20260715.csv"
OUT_INTERACTION = ROOT / "audits/derived/qwen35_2b_lora_interaction_analysis_20260715.csv"
OUT_CROSS = ROOT / "audits/derived/qwen35_2b_llama3b_qwen9b_cross_model_comparison_20260715.csv"

MODEL_REGISTRY_KEY = "qwen35_2b_base"
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TOKENIZER_REVISION = MODEL_REVISION
MODEL_TYPE = "qwen3_5"
ADAPTER_ALIAS = "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
ADAPTER_SHA256 = "6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3"
TEST_SHA256 = "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce"
INDEX_SHA256 = "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc"
INDEX_METADATA_SHA256 = "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55"
STATIC_SHA256 = "7c4735d7ba31ebd448cd0b94fd4c63a80c3e50f115d0fdd39e652ae0f1be1857"
SYSTEM_PROMPT_SHA256 = "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e"
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_RESAMPLES = 10_000

CONDITIONS = [
    "zero_shot", "top1", "top1_gate070", "top1_gate085",
    "static_seed42", "structure", "structure_gate070", "structure_gate085",
]
DISPLAY = {
    "zero_shot": "Zero Shot",
    "top1": "Dynamic Top-1",
    "top1_gate070": "Top-1 Gate 0.70",
    "top1_gate085": "Top-1 Gate 0.85",
    "static_seed42": "Static k=1 Seed 42",
    "structure": "Structure Top-10 v2",
    "structure_gate070": "Structure Gate 0.70",
    "structure_gate085": "Structure Gate 0.85",
}

LORA_RUNS = {
    "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_083452",
    "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_091541",
    "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_095759",
    "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_103913",
    "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_112013",
    "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_120540",
    "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_124740",
    "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_132822",
}
BASE_RUNS = {
    "zero_shot": "run_base_20260627_211410",
    "top1": "run_base_20260712_171240",
    "top1_gate070": "run_base_20260712_183739",
    "top1_gate085": "run_base_20260712_194508",
    "structure": "run_base_20260712_202105",
}

LLAMA_BASE_RUNS = {
    "zero_shot": "run_base_20260714_162526", "top1": "run_base_20260714_164116",
    "top1_gate070": "run_base_20260714_165432", "top1_gate085": "run_base_20260714_170748",
    "static_seed42": "run_base_20260714_172224", "structure": "run_base_20260714_173302",
    "structure_gate070": "run_base_20260714_174639", "structure_gate085": "run_base_20260714_180015",
}
LLAMA_LORA_RUNS = {
    "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_215808",
    "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_222657",
    "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_225553",
    "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_232516",
    "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_235408",
    "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_002034",
    "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_005046",
    "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_012027",
}
QWEN9_BASE_RUNS = {
    "zero_shot": "run_base_20260624_221131", "top1": "run_base_20260712_143438",
    "top1_gate070": "run_base_20260712_150257", "top1_gate085": "run_base_20260712_153056",
    "structure_gate070": "run_base_20260712_160614", "structure_gate085": "run_base_20260712_163705",
}
QWEN9_LORA_RUNS = {
    "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_120126",
    "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_125744",
    "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_135127",
    "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_144738",
    "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_154004",
    "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_163754",
    "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_202137",
    "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_211349",
}


def load_common() -> Any:
    path = ROOT / "scripts/analyze_llama32_3b_instruct_lora_v2_evaluations.py"
    spec = importlib.util.spec_from_file_location("llama_audit_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import shared audited analysis functions")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    module.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    return module


COMMON = load_common()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def write_csv_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def run_paths(run_id: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "results" / f"{run_id}.csv",
        ROOT / "results" / f"{run_id}_metadata.json",
        ROOT / "results/retrieval_traces" / f"{run_id}_retrieval_traces.jsonl",
    )


def trace_signature(row: dict[str, Any]) -> tuple[str | None, float | None]:
    return COMMON.trace_signature(row)


def expected_config(condition: str, role: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "llm": MODEL_REGISTRY_KEY, "adapter": ADAPTER_ALIAS if role == "lora" else "base",
        "testcases_path": "data/testcases_spider_dev_full.jsonl", "max_test_samples": None,
        "max_new_tokens": 256, "generation_batch_size": 1, "compute_perplexity": False,
        "allow_overlap": False, "same_db_only": False, "prompt_format": "qwen_sqlctx_chatml",
        "system_prompt_variant": "sqlctx_anti_overjoin", "extractor_mode": "sql_first_statement_only",
    }
    values["max_input_tokens"] = 1536 if role == "base" and condition == "zero_shot" else 2048
    if condition == "zero_shot":
        values.update(prompt_tuning="none", k=0)
    else:
        values.update(k=1, fewshot_example_schema_mode="full", fewshot_example_mode="schema_with_rules")
    return values


def validate_config(condition: str, role: str, config: dict[str, Any]) -> list[str]:
    failures = [key for key, value in expected_config(condition, role).items() if config.get(key) != value]
    if condition == "static_seed42":
        if config.get("prompt_tuning") != "static_fewshot" or config.get("retrieval_method") != "static_seeded":
            failures.append("static_method")
        expected_pool = str(STATIC_RESOURCE.relative_to(ROOT))
        if config.get("retrieval_pool_path") != expected_pool:
            failures.append("static_resource")
    elif condition != "zero_shot":
        if config.get("prompt_tuning") != "dynamic_fewshot" or config.get("retrieval_method") != "sentence_transformer_faiss":
            failures.append("dynamic_method")
        if config.get("retrieval_index_path") != str(INDEX_DIR.relative_to(ROOT)):
            failures.append("retrieval_index")
    if condition.startswith("structure"):
        required = {"retrieval_rerank_method": "structure_topk_v2", "retrieval_rerank_top_n": 10, "retrieval_structure_bonus_max": 0.08}
        failures.extend(key for key, value in required.items() if config.get(key) != value)
    elif condition != "zero_shot" and condition != "static_seed42":
        if config.get("retrieval_rerank_method", "none") not in {None, "none"}:
            failures.append("unexpected_reranker")
    thresholds = {"top1_gate070": 0.7, "top1_gate085": 0.85, "structure_gate070": 0.7, "structure_gate085": 0.85}
    if condition in thresholds:
        if config.get("fewshot_gate_enabled") is not True or config.get("fewshot_gate_mode") != "similarity_only" or config.get("fewshot_gate_threshold") != thresholds[condition]:
            failures.append("gate")
    elif config.get("fewshot_gate_enabled") not in {None, False}:
        failures.append("unexpected_gate")
    return failures


def output_issue_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        raw, pred = row["raw_output"], row["pred_sql"]
        counts["empty_raw_output"] += not raw.strip()
        counts["empty_extracted_sql"] += not pred.strip()
        counts["think_marker_in_output"] += "<think>" in raw.lower()
        counts["markdown_fence"] += "```" in raw
        counts["missing_semicolon"] += bool(pred.strip()) and not pred.rstrip().endswith(";")
        counts["multiple_statements"] += pred.count(";") > 1
        counts["text_before_sql"] += bool(re.match(r"(?is)^\s*(?!select\b|with\b|pragma\b|insert\b|update\b|delete\b)", raw))
    return dict(counts)


def audit_run(condition: str, role: str, run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    rows, metadata = load_csv(csv_path), load_json(metadata_path)
    config_path = ROOT / metadata["run_config_path"]
    config = load_json(config_path)
    failures = validate_config(condition, role, config)
    if failures:
        raise RuntimeError(f"Config failure {role}/{condition}: {failures}")
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032:
        raise RuntimeError(f"Incomplete run {run_id}")
    if [row["id"] for row in rows] != [row["id"] for row in tests] or len({row["id"] for row in rows}) != 1032:
        raise RuntimeError(f"Case alignment failure {run_id}")
    for row, test in zip(rows, tests):
        if (row["db_id"], row["question"], row["gold_sql"]) != (test["db_id"], test["question"], test["gold_sql"]):
            raise RuntimeError(f"Test content mismatch {run_id}/{row['id']}")
    checks = {
        "model_id": metadata.get("run_model_id") == MODEL_ID,
        "adapter": metadata.get("run_adapter") == (ADAPTER_ALIAS if role == "lora" else "base"),
        "prompt_format": metadata.get("run_prompt_format") == "qwen_sqlctx_chatml",
        "system_prompt": metadata.get("run_system_prompt_sha256") == SYSTEM_PROMPT_SHA256,
        "limits": metadata.get("run_max_input_tokens") == config["max_input_tokens"] and metadata.get("run_max_new_tokens") == 256,
        "batch": metadata.get("run_generation_batch_size") == 1,
        "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
        "no_sample_limit": metadata.get("run_max_test_samples") == "",
        "config_fields_match_csv": all(row.get("run_config_path") == metadata["run_config_path"] for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Provenance failure {run_id}: {[key for key, ok in checks.items() if not ok]}")

    exec_values = np.asarray([COMMON.as_bool(row["exec_match"]) for row in rows], dtype=np.int8)
    pred_ok = np.asarray([COMMON.as_bool(row["pred_ok"]) for row in rows], dtype=np.int8)
    string_exact = np.asarray([int(row["pred_sql"] == row["gold_sql"]) for row in rows], dtype=np.int8)
    normalized_exact = np.asarray([int(COMMON.normalized_sql(row["pred_sql"]) == COMMON.normalized_sql(row["gold_sql"])) for row in rows], dtype=np.int8)
    char_values = np.asarray([COMMON.char_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    token_values = np.asarray([COMMON.token_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    mismatch = {
        "string_exact": sum(int(row["string_exact"]) != value for row, value in zip(rows, string_exact)),
        "normalized_exact": sum(int(row["normalized_exact"]) != value for row, value in zip(rows, normalized_exact)),
        "char_accuracy": sum(abs(float(row["char_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, char_values)),
        "token_accuracy": sum(abs(float(row["token_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, token_values)),
    }
    if any(mismatch.values()):
        raise RuntimeError(f"Stored metric mismatch {run_id}: {mismatch}")
    reproduced = {
        "execution_match_accuracy": float(exec_values.mean()), "execution_success_rate": float(pred_ok.mean()),
        "string_exact_match": float(string_exact.mean()), "normalized_exact_match": float(normalized_exact.mean()),
        "char_accuracy_avg": float(char_values.mean()), "token_accuracy_avg": float(token_values.mean()),
    }
    if any(abs(float(metadata[key]) - value) > 5.1e-10 for key, value in reproduced.items()):
        raise RuntimeError(f"Aggregate metric mismatch {run_id}")

    traces: list[dict[str, Any]] = []
    trace_summary = None
    if condition != "zero_shot":
        traces = load_jsonl(trace_path)
        if len(traces) != 1032 or [row["id"] for row in traces] != [row["id"] for row in tests]:
            raise RuntimeError(f"Trace alignment failure {run_id}")
        if any(not row.get("retrieval_success") or row.get("leakage_status") != "pass" for row in traces):
            raise RuntimeError(f"Retrieval failure {run_id}")
        signatures = [trace_signature(row) for row in traces]
        scores = [score for _, score in signatures if score is not None]
        threshold = config.get("fewshot_gate_threshold")
        gate_mismatches = 0
        if threshold is not None:
            for trace, (_, score) in zip(traces, signatures):
                expected = "fewshot" if score is not None and score >= threshold else "zero_shot"
                gate_mismatches += trace.get("gate_decision") != expected
        if gate_mismatches:
            raise RuntimeError(f"Gate mismatch {run_id}: {gate_mismatches}")
        trace_summary = {
            "path": str(trace_path.relative_to(ROOT)), "sha256": sha256(trace_path), "rows": len(traces),
            "retrieval_success": sum(bool(row.get("retrieval_success")) for row in traces),
            "leakage_pass": sum(row.get("leakage_status") == "pass" for row in traces),
            "unique_demo_ids": len({demo for demo, _ in signatures}),
            "mean_similarity": float(np.mean(scores)) if scores else None,
            "min_similarity": min(scores) if scores else None, "max_similarity": max(scores) if scores else None,
            "gate_counts": dict(Counter(row.get("gate_decision") for row in traces if row.get("gate_decision") is not None)),
            "gate_decision_mismatches": gate_mismatches,
            "target_id_as_demo": sum(demo == test["id"] for (demo, _), test in zip(signatures, tests)),
        }
    prompt = [int(row["prompt_tokens"]) for row in rows]
    completion = [int(row["completion_tokens"]) for row in rows]
    return {
        "condition": condition, "role": role, "run_id": run_id,
        "csv_path": str(csv_path.relative_to(ROOT)), "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)), "metadata_sha256": sha256(metadata_path),
        "trace_path": str(trace_path.relative_to(ROOT)) if trace_path.is_file() else None,
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else None,
        "log_path": None, "log_sha256": None,
        "config_path": metadata["run_config_path"], "config_sha256": sha256(config_path),
        "start_time": metadata["start_time"], "end_time": metadata["end_time"], "status": "VALID WITH WARNINGS",
        "warnings": ["No per-run log or immutable runtime config hash was persisted by this Qwen evaluation generation."],
        "rows": rows, "metadata": metadata, "traces": traces, "exec": exec_values, "pred_ok": pred_ok,
        "checks": checks, "stored_metric_mismatches": mismatch, "reproduced_metrics": reproduced,
        "trace_summary": trace_summary,
        "metrics": {
            "correct": int(exec_values.sum()), "ema": float(exec_values.mean()), "executable": int(pred_ok.sum()), "esr": float(pred_ok.mean()),
            "string_exact": float(string_exact.mean()), "normalized_exact": float(normalized_exact.mean()),
            "char_accuracy": float(char_values.mean()), "token_accuracy": float(token_values.mean()),
            "runtime_seconds": float(metadata["duration_seconds"]),
            "prompt_tokens": COMMON.quantiles(prompt), "completion_tokens": COMMON.quantiles(completion),
            "prompts_at_or_above_limit": sum(value >= config["max_input_tokens"] for value in prompt),
            "completions_at_256": sum(value == 256 for value in completion),
            "output_issues": output_issue_counts(rows),
        },
    }


def compact(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"rows", "metadata", "traces", "exec", "pred_ok"}}


def gate_reference_check(runs: dict[str, dict[str, Any]], gated: str, fewshot: str) -> dict[str, Any]:
    return COMMON.gate_reference_check(runs, gated, fewshot)


def retrieval_overlap(tests: list[dict[str, Any]]) -> dict[str, Any]:
    return COMMON.retrieval_overlap_audit(tests)


def build_case_rows(tests: list[dict[str, Any]], base: dict[str, dict[str, Any]], lora: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, test in enumerate(tests):
        row: dict[str, Any] = {"id": test["id"], "db_id": test["db_id"], "question": test["question"], "gold_sql": test["gold_sql"]}
        for condition in CONDITIONS:
            for role, runs in [("base", base), ("lora", lora)]:
                run = runs.get(condition)
                if run is None:
                    row[f"{role}_{condition}_status"] = "MISSING"
                    continue
                item = run["rows"][index]
                ok = COMMON.as_bool(item["exec_match"])
                row[f"{role}_{condition}_status"] = "AVAILABLE"
                row[f"{role}_{condition}_exec_match"] = ok
                row[f"{role}_{condition}_pred_ok"] = COMMON.as_bool(item["pred_ok"])
                row[f"{role}_{condition}_pred_sql"] = item["pred_sql"]
                if condition != "zero_shot":
                    demo, score = trace_signature(run["traces"][index])
                    row[f"{role}_{condition}_demo_id"] = demo
                    row[f"{role}_{condition}_similarity"] = score
                    row[f"{role}_{condition}_gate_decision"] = run["traces"][index].get("gate_decision", "ungated")
            if condition in base:
                b, l = base[condition]["rows"][index], lora[condition]["rows"][index]
                bo, lo = COMMON.as_bool(b["exec_match"]), COMMON.as_bool(l["exec_match"])
                row[f"base_to_lora_{condition}_transition"] = ("correct" if bo else "wrong") + "->" + ("correct" if lo else "wrong")
                row[f"lora_{condition}_error_category"] = "correct" if lo else COMMON.classify_sql_error(l)
        output.append(row)
    return output


def similarity_bin(score: float) -> str:
    if score < 0.60: return "<0.60"
    if score < 0.70: return "0.60-<0.70"
    if score < 0.85: return "0.70-<0.85"
    return ">=0.85"


def similarity_rows(role: str, runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    zero = runs["zero_shot"]["exec"]
    for family, ungated, gate070, gate085 in [
        ("top1", "top1", "top1_gate070", "top1_gate085"),
        ("structure", "structure", "structure_gate070", "structure_gate085"),
    ]:
        if ungated not in runs: continue
        scores = np.asarray([trace_signature(row)[1] for row in runs[ungated]["traces"]], dtype=float)
        bins = np.asarray([similarity_bin(score) for score in scores])
        for label in ["<0.60", "0.60-<0.70", "0.70-<0.85", ">=0.85"]:
            mask, n = bins == label, int(np.sum(bins == label))
            item: dict[str, Any] = {
                "model_role": role, "retrieval_family": family, "similarity_bin": label, "n": n,
                "mean_similarity": float(scores[mask].mean()) if n else None,
                "zero_shot_ema": float(zero[mask].mean()) if n else None,
                "ungated_ema": float(runs[ungated]["exec"][mask].mean()) if n else None,
                "ungated_minus_zero": float((runs[ungated]["exec"][mask] - zero[mask]).mean()) if n else None,
            }
            for gate, key in [(gate070, "gate070"), (gate085, "gate085")]:
                item[f"{key}_ema"] = float(runs[gate]["exec"][mask].mean()) if gate in runs and n else None
                item[f"{key}_minus_zero"] = float((runs[gate]["exec"][mask] - zero[mask]).mean()) if gate in runs and n else None
            output.append(item)
    return output


def load_reference(run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, _ = run_paths(run_id)
    rows, metadata = load_csv(csv_path), load_json(metadata_path)
    if len(rows) != 1032 or [row["id"] for row in rows] != [row["id"] for row in tests]:
        raise RuntimeError(f"Cross-model reference alignment failure {run_id}")
    return {"run_id": run_id, "ema": float(np.mean([COMMON.as_bool(row["exec_match"]) for row in rows])),
            "csv_sha256": sha256(csv_path), "metadata_sha256": sha256(metadata_path)}


def cross_model_rows(tests: list[dict[str, Any]], base: dict[str, dict[str, Any]], lora: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maps = {
        "qwen35_2b_base": BASE_RUNS, "qwen35_2b_lora_v2": LORA_RUNS,
        "llama32_3b_instruct_base": LLAMA_BASE_RUNS, "llama32_3b_instruct_lora_v2": LLAMA_LORA_RUNS,
        "qwen35_9b_base": QWEN9_BASE_RUNS, "qwen35_9b_lora_v2": QWEN9_LORA_RUNS,
    }
    refs: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for model, mapping in maps.items():
        refs[model] = {}
        provenance[model] = {}
        for condition, run_id in mapping.items():
            ref = load_reference(run_id, tests)
            refs[model][condition] = ref
            provenance[model][condition] = ref
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        row: dict[str, Any] = {"condition": condition, "condition_label": DISPLAY[condition]}
        for model in maps:
            ref = refs[model].get(condition)
            row[f"{model}_run_id"] = ref["run_id"] if ref else None
            row[f"{model}_ema"] = ref["ema"] if ref else None
        for stem in ["qwen35_2b", "llama32_3b_instruct", "qwen35_9b"]:
            base_key = stem + "_base"
            if stem == "llama32_3b_instruct": base_key = stem + "_base"
            lora_key = stem + "_lora_v2"
            b, l = row.get(base_key + "_ema"), row.get(lora_key + "_ema")
            row[stem + "_lora_gain"] = l - b if b is not None and l is not None else None
            z = refs[lora_key].get("zero_shot")
            row[stem + "_lora_fewshot_effect"] = l - z["ema"] if l is not None and z else None
        row["comparability_class"] = "B (cross-family descriptive); within-model Base-vs-LoRA populated pairs are A"
        rows.append(row)
    return rows, provenance


def main() -> None:
    outputs = [OUT_SUMMARY, OUT_CASES, OUT_BASE_LORA, OUT_LORA_FS, OUT_BASE_FS, OUT_SIMILARITY, OUT_INTERACTION, OUT_CROSS]
    for path in outputs:
        if path.exists(): raise RuntimeError(f"Refusing to overwrite {path}")
    identity = {
        "test": sha256(TESTCASES) == TEST_SHA256,
        "adapter": sha256(ADAPTER_ROOT / "adapter_model.safetensors") == ADAPTER_SHA256,
        "root_best": sha256(ADAPTER_ROOT / "adapter_model.safetensors") == sha256(BEST_CHECKPOINT / "adapter_model.safetensors"),
        "index": sha256(INDEX_DIR / "index.faiss") == INDEX_SHA256,
        "index_metadata": sha256(INDEX_DIR / "metadata.jsonl") == INDEX_METADATA_SHA256,
        "static": sha256(STATIC_RESOURCE) == STATIC_SHA256,
    }
    if not all(identity.values()): raise RuntimeError(f"Identity failure: {identity}")
    tests = load_jsonl(TESTCASES)
    if len(tests) != 1032: raise RuntimeError("Unexpected Spider Dev size")

    lora = {condition: audit_run(condition, "lora", run_id, tests) for condition, run_id in LORA_RUNS.items()}
    base = {condition: audit_run(condition, "base", run_id, tests) for condition, run_id in BASE_RUNS.items()}
    matching = [condition for condition in CONDITIONS if condition in base]
    missing_base = [condition for condition in CONDITIONS if condition not in base]

    base_zero_prompt_identity = {
        "prompt_token_counts_equal_lora_zero": sum(
            b["prompt_tokens"] == l["prompt_tokens"] for b, l in zip(base["zero_shot"]["rows"], lora["zero_shot"]["rows"])
        ),
        "base_prompt_max": base["zero_shot"]["metrics"]["prompt_tokens"]["max"],
        "base_limit": 1536,
        "nonbinding": base["zero_shot"]["metrics"]["prompt_tokens"]["max"] < 1536,
    }

    top1_consistency = {
        "base_vs_lora_ungated": COMMON.compare_trace_sets(base["top1"]["traces"], lora["top1"]["traces"]),
        "base_vs_lora_gate070": COMMON.compare_trace_sets(base["top1_gate070"]["traces"], lora["top1_gate070"]["traces"]),
        "base_vs_lora_gate085": COMMON.compare_trace_sets(base["top1_gate085"]["traces"], lora["top1_gate085"]["traces"]),
        "lora_ungated_vs_gate070": COMMON.compare_trace_sets(lora["top1"]["traces"], lora["top1_gate070"]["traces"]),
        "lora_ungated_vs_gate085": COMMON.compare_trace_sets(lora["top1"]["traces"], lora["top1_gate085"]["traces"]),
    }
    structure_consistency = {
        "base_vs_lora_ungated": COMMON.compare_trace_sets(base["structure"]["traces"], lora["structure"]["traces"]),
        "lora_ungated_vs_gate070": COMMON.compare_trace_sets(lora["structure"]["traces"], lora["structure_gate070"]["traces"]),
        "lora_ungated_vs_gate085": COMMON.compare_trace_sets(lora["structure"]["traces"], lora["structure_gate085"]["traces"]),
        "top1_vs_structure_lora": COMMON.compare_trace_sets(lora["top1"]["traces"], lora["structure"]["traces"]),
    }
    gate_checks = {
        "base_top1_gate070": gate_reference_check(base, "top1_gate070", "top1"),
        "base_top1_gate085": gate_reference_check(base, "top1_gate085", "top1"),
        "lora_top1_gate070": gate_reference_check(lora, "top1_gate070", "top1"),
        "lora_top1_gate085": gate_reference_check(lora, "top1_gate085", "top1"),
        "lora_structure_gate070": gate_reference_check(lora, "structure_gate070", "structure"),
        "lora_structure_gate085": gate_reference_check(lora, "structure_gate085", "structure"),
    }
    static_ids = [trace_signature(row)[0] for row in lora["static_seed42"]["traces"]]
    static_check = {"rows": len(static_ids), "unique_demo_ids": sorted(set(static_ids)),
                    "all_expected_demo": all(value == "SPIDER_TRAIN_001657" for value in static_ids),
                    "base_status": "MISSING"}
    overlap = retrieval_overlap(tests)
    if any(overlap[key] for key in ["id_overlap", "question_overlap", "sql_overlap", "pair_overlap"]):
        raise RuntimeError(f"Retrieval overlap: {overlap}")

    rescoring = {
        "base": COMMON.execution_rescore(base),
        "lora": COMMON.execution_rescore(lora),
    }
    rescore_mismatches = sum(
        details[path][metric]
        for role in rescoring.values() for details in role.values()
        for path in ["existing_runner_path", "independent_sqlite_path"]
        for metric in ["esr_mismatch_count", "ema_mismatch_count"]
    )
    if rescore_mismatches: raise RuntimeError(f"Execution rescoring mismatches: {rescore_mismatches}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_lora_stats = [COMMON.paired_stats(base[c]["exec"], lora[c]["exec"], comparison="Qwen 2B Base vs LoRA v2", condition=c, rng=rng) for c in matching]
    COMMON.holm_adjust(base_lora_stats)
    base_fs_stats = [COMMON.paired_stats(base["zero_shot"]["exec"], base[c]["exec"], comparison="Base Zero Shot vs condition", condition=c, rng=rng) for c in matching if c != "zero_shot"]
    COMMON.holm_adjust(base_fs_stats)
    lora_fs_stats = [COMMON.paired_stats(lora["zero_shot"]["exec"], lora[c]["exec"], comparison="LoRA Zero Shot vs condition", condition=c, rng=rng) for c in CONDITIONS[1:]]
    COMMON.holm_adjust(lora_fs_stats)
    targeted_pairs = [("top1", "top1_gate070"), ("top1", "top1_gate085"), ("top1", "structure"),
                      ("structure", "structure_gate070"), ("structure", "structure_gate085"), ("static_seed42", "top1")]
    targeted = [COMMON.paired_stats(lora[a]["exec"], lora[b]["exec"], comparison=f"{a} vs {b}", condition=b, rng=rng) for a, b in targeted_pairs]

    interaction: list[dict[str, Any]] = []
    for condition in matching:
        if condition == "zero_shot": continue
        base_effect = float(base[condition]["exec"].mean() - base["zero_shot"]["exec"].mean())
        lora_effect = float(lora[condition]["exec"].mean() - lora["zero_shot"]["exec"].mean())
        low, high = COMMON.bootstrap_did(base["zero_shot"]["exec"], base[condition]["exec"], lora["zero_shot"]["exec"], lora[condition]["exec"], rng=rng)
        interaction.append({"condition": condition, "condition_label": DISPLAY[condition],
                            "base_fewshot_effect": base_effect, "lora_fewshot_effect": lora_effect,
                            "difference_in_differences": lora_effect - base_effect,
                            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES})

    cases = build_case_rows(tests, base, lora)
    similarities = similarity_rows("base", base) + similarity_rows("lora_v2", lora)
    cross, cross_provenance = cross_model_rows(tests, base, lora)
    transitions = {condition: dict(Counter(row[f"base_to_lora_{condition}_transition"] for row in cases)) for condition in matching}
    errors = {condition: dict(Counter(row[f"lora_{condition}_error_category"] for row in cases)) for condition in matching}

    runner = COMMON.import_runner()
    renderer_source = inspect.getsource(runner._render_qwen_sqlctx_chatml_messages)
    prompt_integrity = {
        "format": "qwen_sqlctx_chatml", "assistant_prefix": "<|im_start|>assistant\n",
        "contains_think": False, "contains_llama_tokens": False,
        "renderer_source_sha256": hashlib.sha256(renderer_source.encode()).hexdigest(),
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "preflight_manifest": str(TRAINING_MANIFEST.relative_to(ROOT)),
        "preflight_total_prompts": 6192, "preflight_invalid_prompts": 0, "preflight_truncations": 0,
    }

    training_metadata = load_json(ADAPTER_ROOT / "training_metadata.json")
    final_state = load_json(ADAPTER_ROOT / "checkpoints/checkpoint-1506/trainer_state.json")
    summary = {
        "schema_version": 1,
        "purpose": "qwen35_2b_base_and_official_lora_v2_completed_evaluation_audit",
        "status": "PASS_WITH_METHODICAL_LIMITATIONS",
        "warnings": [
            "Qwen 2B Base Static and Structure Gate 0.70/0.85 runs are missing and were not substituted.",
            "The Base Zero-Shot config records max_input_tokens=1536; its maximum actual prompt is 736 tokens, so the limit was nonbinding.",
            "Qwen run artifacts do not persist per-run logs, immutable runtime config hashes, or model revisions; identity is reconstructed from configs, metadata, registry convention, cache provenance, and training manifests.",
            "Spider Dev is development-facing rather than an untouched final test set; Structure Gates are exploratory.",
        ],
        "generation_started_by_analysis": False, "model_loaded_by_analysis": False,
        "adapter_loaded_by_analysis": False, "embedding_model_loaded_by_analysis": False,
        "official_model": {"registry_key": MODEL_REGISTRY_KEY, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                           "tokenizer_revision": TOKENIZER_REVISION, "model_type": MODEL_TYPE,
                           "revision_pinned_in_runtime_registry": False},
        "official_adapter": {
            "root": str(ADAPTER_ROOT.relative_to(ROOT)), "best_checkpoint": str(BEST_CHECKPOINT.relative_to(ROOT)),
            "adapter_sha256": sha256(ADAPTER_ROOT / "adapter_model.safetensors"), "root_equals_best": identity["root_best"],
            "best_metric": training_metadata["best_metric"], "best_model_checkpoint": training_metadata["best_model_checkpoint"],
            "trainer_state_best_metric": final_state["best_metric"], "trainer_state_best_model_checkpoint": final_state["best_model_checkpoint"],
            "training_config": str(TRAIN_CONFIG.relative_to(ROOT)), "training_config_sha256": sha256(TRAIN_CONFIG),
            "training_audit": str(TRAINING_AUDIT.relative_to(ROOT)), "training_audit_sha256": sha256(TRAINING_AUDIT),
            "training_manifest": str(TRAINING_MANIFEST.relative_to(ROOT)), "training_manifest_sha256": sha256(TRAINING_MANIFEST),
        },
        "identity_checks": identity,
        "testset": {"path": str(TESTCASES.relative_to(ROOT)), "sha256": sha256(TESTCASES), "rows": len(tests)},
        "prompt_integrity": prompt_integrity,
        "base_zero_1536_equivalence": base_zero_prompt_identity,
        "run_identification": {
            "base": {c: (compact(base[c]) if c in base else {"condition": c, "status": "MISSING"}) for c in CONDITIONS},
            "lora_v2": {c: compact(lora[c]) for c in CONDITIONS},
        },
        "retrieval": {"index_path": str(INDEX_DIR.relative_to(ROOT)), "embedding_model": "BAAI/bge-large-en-v1.5", **overlap},
        "top1_trace_consistency": top1_consistency, "structure_trace_consistency": structure_consistency,
        "gate_reference_checks": gate_checks, "static_consistency": static_check,
        "execution_rescoring": rescoring, "execution_rescoring_mismatch_count": rescore_mismatches,
        "base_vs_lora_statistics": base_lora_stats, "base_fewshot_vs_zero_statistics": base_fs_stats,
        "lora_fewshot_vs_zero_statistics": lora_fs_stats, "targeted_lora_statistics": targeted,
        "difference_in_differences": interaction, "similarity_bins": similarities,
        "transition_summary": transitions, "error_category_summary": errors,
        "cross_model_provenance": cross_provenance,
        "missing_base_conditions": missing_base,
        "comparability": {"qwen2_base_vs_lora_matching": "A", "qwen2_within_model": "A",
                          "cross_family": "B", "missing_base_counterparts": "NOT_COMPARABLE"},
        "statistics": {"mcnemar": "exact two-sided binomial test", "bootstrap_seed": BOOTSTRAP_SEED,
                       "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95,
                       "holm_families": {"base_vs_lora": len(base_lora_stats), "base_fewshot": len(base_fs_stats), "lora_fewshot": len(lora_fs_stats)}},
        "rerun_required": False,
    }

    write_csv_new(OUT_BASE_LORA, base_lora_stats)
    write_csv_new(OUT_LORA_FS, lora_fs_stats)
    write_csv_new(OUT_BASE_FS, base_fs_stats)
    write_csv_new(OUT_CASES, cases)
    write_csv_new(OUT_SIMILARITY, similarities)
    write_csv_new(OUT_INTERACTION, interaction)
    write_csv_new(OUT_CROSS, cross)
    write_new(OUT_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2, default=COMMON.json_default) + "\n")
    print(json.dumps({"status": summary["status"], "outputs": [str(p.relative_to(ROOT)) for p in outputs],
                      "base_ema": {c: base[c]["metrics"]["ema"] for c in base},
                      "lora_ema": {c: lora[c]["metrics"]["ema"] for c in lora},
                      "rescore_mismatches": rescore_mismatches, "missing_base": missing_base}, indent=2))


if __name__ == "__main__":
    main()
