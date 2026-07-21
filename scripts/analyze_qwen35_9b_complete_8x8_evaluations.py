#!/usr/bin/env python3
"""Read-only final analysis of the complete Qwen 3.5 9B Base/LoRA-v2 8x8 matrix.

The script reads completed evaluation artifacts and Spider SQLite databases.
It never loads a model, adapter, tokenizer, or embedding model. Derived files
are opened with exclusive creation, so an existing audit artifact is never
replaced.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = ROOT / "scripts/analyze_qwen35_2b_base_and_lora_v2_evaluations.py"
SHARED_SCRIPT = ROOT / "scripts/analyze_llama32_3b_instruct_lora_v2_evaluations.py"
PRIOR_INVENTORY = ROOT / "audits/qwen35_9b_base_lora_v2_run_inventory_manifest_20260716.json"
NEW_BASE_MANIFEST = ROOT / "audits/qwen35_9b_new_base_static_structure_runs_and_504_comparability_manifest_20260716.json"

OUT_SUMMARY = ROOT / "audits/derived/qwen35_9b_complete_8x8_evaluation_summary_20260716.json"
OUT_CASES = ROOT / "audits/derived/qwen35_9b_complete_8x8_base_vs_lora_case_comparison_20260716.csv"
OUT_BASE_LORA = ROOT / "audits/derived/qwen35_9b_complete_8x8_base_vs_lora_statistics_20260716.csv"
OUT_BASE_FS = ROOT / "audits/derived/qwen35_9b_complete_8x8_base_fewshot_statistics_20260716.csv"
OUT_LORA_FS = ROOT / "audits/derived/qwen35_9b_complete_8x8_lora_fewshot_statistics_20260716.csv"
OUT_INTERACTION = ROOT / "audits/derived/qwen35_9b_complete_8x8_interaction_analysis_20260716.csv"
OUT_GATE = ROOT / "audits/derived/qwen35_9b_complete_8x8_gate_analysis_20260716.csv"
OUT_COMPLETION = ROOT / "audits/derived/qwen35_9b_complete_8x8_completion_diagnostics_20260716.csv"

TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
INDEX_DIR = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
STATIC_RESOURCE = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
ADAPTER_ROOT = ROOT / "adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
BEST_CHECKPOINT = ADAPTER_ROOT / "checkpoints/checkpoint-502"

MODEL_REGISTRY_KEY = "qwen35_9b_base"
MODEL_ID = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
ADAPTER_ALIAS = "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
ADAPTER_SHA256 = "dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0"
TEST_SHA256 = "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce"
INDEX_SHA256 = "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc"
INDEX_METADATA_SHA256 = "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55"
STATIC_SHA256 = "7c4735d7ba31ebd448cd0b94fd4c63a80c3e50f115d0fdd39e652ae0f1be1857"
SYSTEM_PROMPT_SHA256 = "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e"
BOOTSTRAP_SEED = 20260716
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
BASE_RUNS = {
    "zero_shot": "run_base_20260624_221131",
    "top1": "run_base_20260712_143438",
    "top1_gate070": "run_base_20260712_150257",
    "top1_gate085": "run_base_20260712_153056",
    "static_seed42": "run_base_20260716_084140",
    "structure": "run_base_20260716_090811",
    "structure_gate070": "run_base_20260712_160614",
    "structure_gate085": "run_base_20260712_163705",
}
LORA_RUNS = {
    "zero_shot": f"run_{ADAPTER_ALIAS}_20260713_120126",
    "top1": f"run_{ADAPTER_ALIAS}_20260713_125744",
    "top1_gate070": f"run_{ADAPTER_ALIAS}_20260713_135127",
    "top1_gate085": f"run_{ADAPTER_ALIAS}_20260713_144738",
    "static_seed42": f"run_{ADAPTER_ALIAS}_20260713_154004",
    "structure": f"run_{ADAPTER_ALIAS}_20260713_163754",
    "structure_gate070": f"run_{ADAPTER_ALIAS}_20260713_202137",
    "structure_gate085": f"run_{ADAPTER_ALIAS}_20260713_211349",
}


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("qwen9_audit_helper", HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import audited Qwen helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    module.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    module.COMMON.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    module.COMMON.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    return module


Q = import_helper()
C = Q.COMMON


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


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=C.json_default)
        handle.write("\n")


def write_csv_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def inventory_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    old = load_json(PRIOR_INVENTORY)
    by_run = {row["run_id"]: row for row in old["run_inventory"] if row.get("run_id")}
    new = load_json(NEW_BASE_MANIFEST)
    for condition, item in new["runs"].items():
        by_run[item["run_id"]] = {
            "role": "base", "condition": condition, "run_id": item["run_id"],
            "recorded_config_path": item["config"]["path"],
            "physical_config_path": item["config"]["path"],
            "config_sha256": item["config"]["sha256"],
            "csv_path": item["csv"]["path"], "csv_sha256": item["csv"]["sha256"],
            "metadata_path": item["metadata"]["path"], "metadata_sha256": item["metadata"]["sha256"],
            "trace_path": item["trace"]["path"], "trace_sha256": item["trace"]["sha256"],
            "log_path": item["log"]["path"], "log_sha256": item["log"]["sha256"],
            "status": "COMPLETE_WITH_WARNINGS" if condition == "static" else "COMPLETE_AND_MATCHING",
            "warnings": ["documented_nonblocking_http_504"] if condition == "static" else [],
        }
    return by_run, {row["condition"]: row for row in old["run_inventory"] if row["role"] == "base"}


RUN_INVENTORY, BASE_CONDITION_INVENTORY = inventory_maps()


def expected_config(condition: str, role: str) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "llm": MODEL_REGISTRY_KEY,
        "adapter": ADAPTER_ALIAS if role == "lora_v2" else "base",
        "testcases_path": "data/testcases_spider_dev_full.jsonl",
        "max_test_samples": None,
        "max_input_tokens": 1536 if role == "base" and condition == "zero_shot" else 2048,
        "max_new_tokens": 256,
        "generation_batch_size": 1,
        "compute_perplexity": False,
        "allow_overlap": False,
        "same_db_only": False,
        "prompt_format": "qwen_sqlctx_chatml",
        "system_prompt_variant": "sqlctx_anti_overjoin",
        "extractor_mode": "sql_first_statement_only",
    }
    if condition == "zero_shot":
        expected.update(prompt_tuning="none", k=0)
    else:
        expected.update(k=1, fewshot_example_schema_mode="full", fewshot_example_mode="schema_with_rules")
    return expected


def validate_config(condition: str, role: str, config: dict[str, Any]) -> list[str]:
    failures = [key for key, value in expected_config(condition, role).items() if config.get(key) != value]
    if condition == "static_seed42":
        if config.get("prompt_tuning") != "static_fewshot" or config.get("retrieval_method") != "static_seeded":
            failures.append("static_method")
        if config.get("retrieval_pool_path") != str(STATIC_RESOURCE.relative_to(ROOT)):
            failures.append("static_resource")
    elif condition != "zero_shot":
        if config.get("prompt_tuning") != "dynamic_fewshot" or config.get("retrieval_method") != "sentence_transformer_faiss":
            failures.append("dynamic_method")
        if config.get("retrieval_index_path") != str(INDEX_DIR.relative_to(ROOT)):
            failures.append("retrieval_index")
    if condition.startswith("structure"):
        required = {
            "retrieval_rerank_method": "structure_topk_v2",
            "retrieval_rerank_top_n": 10,
            "retrieval_structure_bonus_max": 0.08,
        }
        failures.extend(key for key, value in required.items() if config.get(key) != value)
    elif condition not in {"zero_shot", "static_seed42"} and config.get("retrieval_rerank_method", "none") not in {None, "none"}:
        failures.append("unexpected_reranker")
    thresholds = {"top1_gate070": 0.70, "top1_gate085": 0.85, "structure_gate070": 0.70, "structure_gate085": 0.85}
    if condition in thresholds:
        gate = {
            "fewshot_gate_enabled": True,
            "fewshot_gate_mode": "similarity_only",
            "fewshot_gate_threshold": thresholds[condition],
        }
        failures.extend(key for key, value in gate.items() if config.get(key) != value)
    elif config.get("fewshot_gate_enabled") not in {None, False}:
        failures.append("unexpected_gate")
    return failures


def output_diagnostics(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        raw, pred = row["raw_output"], row["pred_sql"]
        tokens = re.findall(r"\w+|[^\w\s]", raw.lower())
        grams = [tuple(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))]
        repeated_ratio = 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)
        first_sql = re.search(r"(?is)\b(select|with)\b", raw)
        counts["empty_raw_output"] += not raw.strip()
        counts["empty_extracted_sql"] += not pred.strip()
        counts["think_marker"] += "<think>" in raw.lower()
        counts["llama_marker"] += any(token in raw for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"])
        counts["markdown_fence"] += "```" in raw
        counts["missing_semicolon"] += bool(pred.strip()) and not pred.rstrip().endswith(";")
        counts["multiple_extracted_statements"] += pred.count(";") > 1
        counts["reason_text_before_sql"] += bool(first_sql and raw[:first_sql.start()].strip())
        counts["no_sql_prefix"] += first_sql is None
        counts["repetitive_output"] += repeated_ratio >= 0.20
        counts["limit_and_no_semicolon"] += int(row["completion_tokens"]) == 256 and ";" not in raw
    return dict(counts)


def audit_run(condition: str, role: str, run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = RUN_INVENTORY[run_id]
    csv_path = ROOT / inventory["csv_path"]
    metadata_path = ROOT / inventory["metadata_path"]
    trace_path = ROOT / inventory["trace_path"] if inventory.get("trace_path") else None
    config_path = ROOT / inventory["physical_config_path"]
    rows, metadata, config = load_csv(csv_path), load_json(metadata_path), load_json(config_path)
    failures = validate_config(condition, role, config)
    if failures:
        raise RuntimeError(f"Config failure {role}/{condition}: {sorted(set(failures))}")
    if sha256(config_path) != inventory["config_sha256"]:
        raise RuntimeError(f"Config hash mismatch {run_id}")
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032 or len({row["id"] for row in rows}) != 1032:
        raise RuntimeError(f"Incomplete or duplicate cases {run_id}")
    if [row["id"] for row in rows] != [row["id"] for row in tests]:
        raise RuntimeError(f"Case order mismatch {run_id}")
    for row, test in zip(rows, tests):
        if (row["db_id"], row["question"], row["gold_sql"]) != (test["db_id"], test["question"], test["gold_sql"]):
            raise RuntimeError(f"Test content mismatch {run_id}/{row['id']}")
    expected_adapter = ADAPTER_ALIAS if role == "lora_v2" else "base"
    checks = {
        "model": metadata.get("run_model_id") == MODEL_ID,
        "registry": metadata.get("run_llm") == MODEL_REGISTRY_KEY,
        "adapter": metadata.get("run_adapter") == expected_adapter,
        "prompt": metadata.get("run_prompt_format") == "qwen_sqlctx_chatml",
        "system_prompt": metadata.get("run_system_prompt_sha256") == SYSTEM_PROMPT_SHA256,
        "limits": metadata.get("run_max_input_tokens") == config["max_input_tokens"] and metadata.get("run_max_new_tokens") == 256,
        "batch": metadata.get("run_generation_batch_size") == 1,
        "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
        "full_test": metadata.get("run_max_test_samples") == "",
        "config_record": metadata.get("run_config_path") == inventory["recorded_config_path"],
        "row_config_record": all(row.get("run_config_path") == inventory["recorded_config_path"] for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Provenance failure {run_id}: {[key for key, ok in checks.items() if not ok]}")

    exec_values = np.asarray([C.as_bool(row["exec_match"]) for row in rows], dtype=np.int8)
    pred_ok = np.asarray([C.as_bool(row["pred_ok"]) for row in rows], dtype=np.int8)
    string_exact = np.asarray([int(row["pred_sql"] == row["gold_sql"]) for row in rows], dtype=np.int8)
    normalized_exact = np.asarray([int(C.normalized_sql(row["pred_sql"]) == C.normalized_sql(row["gold_sql"])) for row in rows], dtype=np.int8)
    char_values = np.asarray([C.char_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    token_values = np.asarray([C.token_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    mismatch = {
        "string_exact": sum(int(row["string_exact"]) != value for row, value in zip(rows, string_exact)),
        "normalized_exact": sum(int(row["normalized_exact"]) != value for row, value in zip(rows, normalized_exact)),
        "char_accuracy": sum(abs(float(row["char_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, char_values)),
        "token_accuracy": sum(abs(float(row["token_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, token_values)),
    }
    if any(mismatch.values()):
        raise RuntimeError(f"Stored metric mismatch {run_id}: {mismatch}")
    reproduced = {
        "execution_match_accuracy": float(exec_values.mean()),
        "execution_success_rate": float(pred_ok.mean()),
        "string_exact_match": float(string_exact.mean()),
        "normalized_exact_match": float(normalized_exact.mean()),
        "char_accuracy_avg": float(char_values.mean()),
        "token_accuracy_avg": float(token_values.mean()),
    }
    if any(abs(float(metadata[key]) - value) > 5.1e-10 for key, value in reproduced.items()):
        raise RuntimeError(f"Aggregate metric mismatch {run_id}")

    traces: list[dict[str, Any]] = []
    trace_summary = None
    if condition != "zero_shot":
        if trace_path is None or not trace_path.is_file():
            raise RuntimeError(f"Missing trace {run_id}")
        traces = load_jsonl(trace_path)
        if len(traces) != 1032 or [row["id"] for row in traces] != [row["id"] for row in tests]:
            raise RuntimeError(f"Trace alignment failure {run_id}")
        if any(not row.get("retrieval_success") or row.get("leakage_status") != "pass" for row in traces):
            raise RuntimeError(f"Retrieval failure {run_id}")
        signatures = [C.trace_signature(row) for row in traces]
        threshold = config.get("fewshot_gate_threshold")
        gate_mismatches = 0
        if threshold is not None:
            for trace, (_, score) in zip(traces, signatures):
                expected = "fewshot" if score is not None and score >= threshold else "zero_shot"
                gate_mismatches += trace.get("gate_decision") != expected
        if gate_mismatches:
            raise RuntimeError(f"Gate decision mismatch {run_id}: {gate_mismatches}")
        trace_summary = {
            "rows": len(traces),
            "unique_demo_ids": len({demo for demo, _ in signatures}),
            "gate_counts": dict(Counter(row.get("gate_decision") for row in traces if row.get("gate_decision") is not None)),
            "gate_decision_mismatches": gate_mismatches,
            "selected_target_as_demo": sum(demo == test["id"] for (demo, _), test in zip(signatures, tests)),
        }

    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    completion_tokens = [int(row["completion_tokens"]) for row in rows]
    warnings = list(inventory.get("warnings") or [])
    if inventory.get("log_path") is None:
        warnings.append("per_run_log_missing")
    warnings.append("model_revision_not_persisted_in_run_metadata")
    if role == "base" and condition == "zero_shot":
        warnings.append("max_input_1536_nonbinding_prompt_max_736")
    return {
        "condition": condition, "role": role, "run_id": run_id,
        "csv_path": str(csv_path.relative_to(ROOT)), "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)), "metadata_sha256": sha256(metadata_path),
        "trace_path": str(trace_path.relative_to(ROOT)) if trace_path else None,
        "trace_sha256": sha256(trace_path) if trace_path else None,
        "log_path": inventory.get("log_path"), "log_sha256": inventory.get("log_sha256"),
        "recorded_config_path": inventory["recorded_config_path"],
        "physical_config_path": inventory["physical_config_path"], "config_sha256": sha256(config_path),
        "start_time": metadata["start_time"], "end_time": metadata["end_time"],
        "status": "COMPLETE_WITH_WARNINGS" if warnings else "COMPLETE_AND_MATCHING",
        "warnings": sorted(set(warnings)), "rows": rows, "metadata": metadata, "traces": traces,
        "exec": exec_values, "pred_ok": pred_ok, "checks": checks,
        "stored_metric_mismatches": mismatch, "reproduced_metrics": reproduced,
        "trace_summary": trace_summary,
        "metrics": {
            "correct": int(exec_values.sum()), "ema": float(exec_values.mean()),
            "executable": int(pred_ok.sum()), "esr": float(pred_ok.mean()),
            "string_exact": float(string_exact.mean()), "normalized_exact": float(normalized_exact.mean()),
            "char_accuracy": float(char_values.mean()), "token_accuracy": float(token_values.mean()),
            "runtime_seconds": float(metadata["duration_seconds"]),
            "prompt_tokens": C.quantiles(prompt_tokens), "completion_tokens": C.quantiles(completion_tokens),
            "prompt_truncations": sum(value >= config["max_input_tokens"] for value in prompt_tokens),
            "completions_at_256": sum(value == 256 for value in completion_tokens),
            "output_diagnostics": output_diagnostics(rows),
        },
    }


def compact(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"rows", "metadata", "traces", "exec", "pred_ok"}}


def trace_demo_ids(run: dict[str, Any]) -> list[str | None]:
    return [C.trace_signature(row)[0] for row in run["traces"]]


def gate_analysis(role: str, runs: dict[str, dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    output = []
    for family, ungated, gated in [
        ("top1", "top1", "top1_gate070"), ("top1", "top1", "top1_gate085"),
        ("structure", "structure", "structure_gate070"), ("structure", "structure", "structure_gate085"),
    ]:
        traces = runs[gated]["traces"]
        accepted = np.asarray([row.get("gate_decision") == "fewshot" for row in traces])
        rejected = ~accepted
        stats = C.paired_stats(
            runs[ungated]["exec"], runs[gated]["exec"],
            comparison=f"{role} {ungated} vs {gated}", condition=gated, rng=rng,
        )
        stats.update({
            "model_role": role, "retrieval_family": family,
            "exploratory": family == "structure",
            "accepted_count": int(accepted.sum()), "fallback_count": int(rejected.sum()),
            "accepted_ungated_ema": float(runs[ungated]["exec"][accepted].mean()),
            "accepted_zero_ema": float(runs["zero_shot"]["exec"][accepted].mean()),
            "fallback_ungated_ema": float(runs[ungated]["exec"][rejected].mean()),
            "fallback_zero_ema": float(runs["zero_shot"]["exec"][rejected].mean()),
            "accepted_gate_matches_ungated_raw": sum(
                gate["raw_output"] == few["raw_output"]
                for gate, few, keep in zip(runs[gated]["rows"], runs[ungated]["rows"], accepted) if keep
            ),
            "fallback_gate_matches_zero_raw": sum(
                gate["raw_output"] == zero["raw_output"]
                for gate, zero, keep in zip(runs[gated]["rows"], runs["zero_shot"]["rows"], rejected) if keep
            ),
        })
        output.append(stats)
    return output


def build_case_rows(tests: list[dict[str, Any]], base: dict[str, dict[str, Any]], lora: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, test in enumerate(tests):
        row: dict[str, Any] = {"id": test["id"], "db_id": test["db_id"], "question": test["question"], "gold_sql": test["gold_sql"]}
        for condition in CONDITIONS:
            base_row, lora_row = base[condition]["rows"][index], lora[condition]["rows"][index]
            base_ok, lora_ok = C.as_bool(base_row["exec_match"]), C.as_bool(lora_row["exec_match"])
            base_exec, lora_exec = C.as_bool(base_row["pred_ok"]), C.as_bool(lora_row["pred_ok"])
            row.update({
                f"base_{condition}_exec_match": base_ok,
                f"lora_{condition}_exec_match": lora_ok,
                f"base_{condition}_pred_ok": base_exec,
                f"lora_{condition}_pred_ok": lora_exec,
                f"base_{condition}_pred_sql": base_row["pred_sql"],
                f"lora_{condition}_pred_sql": lora_row["pred_sql"],
                f"base_to_lora_{condition}_transition": ("correct" if base_ok else "wrong") + "->" + ("correct" if lora_ok else "wrong"),
                f"base_to_lora_{condition}_execution_transition": ("executable" if base_exec else "not_executable") + "->" + ("executable" if lora_exec else "not_executable"),
            })
        output.append(row)
    return output


def completion_rows(base: dict[str, dict[str, Any]], lora: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for role, runs in [("base", base), ("lora_v2", lora)]:
        for condition in CONDITIONS:
            metrics = runs[condition]["metrics"]
            output.append({
                "model_role": role, "condition": condition, "condition_label": DISPLAY[condition],
                "run_id": runs[condition]["run_id"], "prompt_limit": runs[condition]["metadata"]["run_max_input_tokens"],
                "prompt_tokens_mean": metrics["prompt_tokens"]["mean"], "prompt_tokens_max": metrics["prompt_tokens"]["max"],
                "prompt_truncations": metrics["prompt_truncations"],
                "completion_tokens_mean": metrics["completion_tokens"]["mean"],
                "completion_tokens_max": metrics["completion_tokens"]["max"],
                "completion_limit_cases": metrics["completions_at_256"],
                **metrics["output_diagnostics"],
            })
    return output


def cross_model_context() -> dict[str, Any]:
    q2 = load_json(ROOT / "audits/derived/qwen35_2b_complete_8x8_evaluation_summary_20260715.json")
    llama = load_json(ROOT / "audits/derived/llama32_3b_instruct_lora_v2_evaluation_summary_20260715.json")
    llama_pairs = {row["condition"]: row for row in llama["base_vs_lora_statistics"]}
    return {
        "classification": "descriptive_cross_family_context_only",
        "qwen35_2b": {
            "base_ema": {key: value["metrics"]["ema"] for key, value in q2["base_runs"].items()},
            "lora_v2_ema": {key: value["metrics"]["ema"] for key, value in q2["lora_v2_runs"].items()},
        },
        "llama32_3b_instruct": {
            "base_ema": {key: value["a_ema"] for key, value in llama_pairs.items()},
            "lora_v2_ema": {key: value["b_ema"] for key, value in llama_pairs.items()},
        },
        "limitations": ["different_model_families", "descriptive_only", "no_cross_family_causal_claims"],
    }


def main() -> None:
    outputs = [OUT_SUMMARY, OUT_CASES, OUT_BASE_LORA, OUT_BASE_FS, OUT_LORA_FS, OUT_INTERACTION, OUT_GATE, OUT_COMPLETION]
    for path in outputs:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
    identity = {
        "testset": sha256(TESTCASES) == TEST_SHA256,
        "adapter": sha256(ADAPTER_ROOT / "adapter_model.safetensors") == ADAPTER_SHA256,
        "root_equals_best": sha256(ADAPTER_ROOT / "adapter_model.safetensors") == sha256(BEST_CHECKPOINT / "adapter_model.safetensors"),
        "retrieval_index": sha256(INDEX_DIR / "index.faiss") == INDEX_SHA256,
        "retrieval_metadata": sha256(INDEX_DIR / "metadata.jsonl") == INDEX_METADATA_SHA256,
        "static_resource": sha256(STATIC_RESOURCE) == STATIC_SHA256,
    }
    if not all(identity.values()):
        raise RuntimeError(f"Identity failure: {identity}")
    tests = load_jsonl(TESTCASES)
    if len(tests) != 1032:
        raise RuntimeError("Unexpected Spider Dev size")

    base = {condition: audit_run(condition, "base", run_id, tests) for condition, run_id in BASE_RUNS.items()}
    lora = {condition: audit_run(condition, "lora_v2", run_id, tests) for condition, run_id in LORA_RUNS.items()}

    base_zero_prompt_identity = {
        "base_limit": 1536, "lora_limit": 2048,
        "base_prompt_max": base["zero_shot"]["metrics"]["prompt_tokens"]["max"],
        "prompt_token_counts_equal": sum(
            left["prompt_tokens"] == right["prompt_tokens"]
            for left, right in zip(base["zero_shot"]["rows"], lora["zero_shot"]["rows"])
        ),
        "nonbinding": base["zero_shot"]["metrics"]["prompt_tokens"]["max"] < 1536,
    }
    if not base_zero_prompt_identity["nonbinding"] or base_zero_prompt_identity["prompt_token_counts_equal"] != 1032:
        raise RuntimeError(f"Base Zero input-limit equivalence failure: {base_zero_prompt_identity}")

    trace_identity = {}
    for condition in CONDITIONS[1:]:
        trace_identity[condition] = C.compare_trace_sets(base[condition]["traces"], lora[condition]["traces"])
        if trace_identity[condition]["different_demo_ids"] or trace_identity[condition]["different_scores"]:
            raise RuntimeError(f"Base/LoRA trace mismatch {condition}: {trace_identity[condition]}")
    static_ids = {"base": trace_demo_ids(base["static_seed42"]), "lora_v2": trace_demo_ids(lora["static_seed42"])}
    static_check = {
        "demo_id": "SPIDER_TRAIN_001657",
        "base_all": set(static_ids["base"]) == {"SPIDER_TRAIN_001657"},
        "lora_all": set(static_ids["lora_v2"]) == {"SPIDER_TRAIN_001657"},
        "base_lora_same_cases": sum(a == b for a, b in zip(static_ids["base"], static_ids["lora_v2"])),
    }
    if not static_check["base_all"] or not static_check["lora_all"] or static_check["base_lora_same_cases"] != 1032:
        raise RuntimeError(f"Static identity failure: {static_check}")

    gate_checks = {}
    for role, runs in [("base", base), ("lora_v2", lora)]:
        for gated, ungated in [
            ("top1_gate070", "top1"), ("top1_gate085", "top1"),
            ("structure_gate070", "structure"), ("structure_gate085", "structure"),
        ]:
            key = f"{role}_{gated}"
            gate_checks[key] = C.gate_reference_check(runs, gated, ungated)
            check = gate_checks[key]
            if min(check["prompt_token_matches_selected_reference"], check["raw_output_matches_selected_reference"], check["pred_sql_matches_selected_reference"]) != 1032:
                raise RuntimeError(f"Gate reference mismatch: {key}")
    expected_splits = {"top1_gate070": (634, 398), "top1_gate085": (57, 975), "structure_gate070": (613, 419), "structure_gate085": (57, 975)}
    for role, runs in [("base", base), ("lora_v2", lora)]:
        for condition, expected in expected_splits.items():
            actual = runs[condition]["trace_summary"]["gate_counts"]
            if (actual.get("fewshot", 0), actual.get("zero_shot", 0)) != expected:
                raise RuntimeError(f"Gate split mismatch {role}/{condition}: {actual}")

    overlap = C.retrieval_overlap_audit(tests)
    if any(overlap[key] for key in ["id_overlap", "question_overlap", "sql_overlap", "pair_overlap"]):
        raise RuntimeError(f"Retrieval leakage: {overlap}")

    rescoring = {"base": C.execution_rescore(base), "lora_v2": C.execution_rescore(lora)}
    rescore_mismatches = sum(
        details[path][metric]
        for role in rescoring.values() for details in role.values()
        for path in ["existing_runner_path", "independent_sqlite_path"]
        for metric in ["esr_mismatch_count", "ema_mismatch_count"]
    )
    path_disagreements = sum(details["path_disagreement_count"] for role in rescoring.values() for details in role.values())
    if rescore_mismatches or path_disagreements:
        raise RuntimeError(f"Execution rescoring mismatch={rescore_mismatches}, path disagreement={path_disagreements}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_lora_stats = [C.paired_stats(base[c]["exec"], lora[c]["exec"], comparison="Qwen 9B Base vs LoRA v2", condition=c, rng=rng) for c in CONDITIONS]
    C.holm_adjust(base_lora_stats)
    base_fs_stats = [C.paired_stats(base["zero_shot"]["exec"], base[c]["exec"], comparison="Base Zero Shot vs condition", condition=c, rng=rng) for c in CONDITIONS[1:]]
    C.holm_adjust(base_fs_stats)
    lora_fs_stats = [C.paired_stats(lora["zero_shot"]["exec"], lora[c]["exec"], comparison="LoRA v2 Zero Shot vs condition", condition=c, rng=rng) for c in CONDITIONS[1:]]
    C.holm_adjust(lora_fs_stats)

    interaction = []
    for condition in CONDITIONS[1:]:
        base_effect = float(base[condition]["exec"].mean() - base["zero_shot"]["exec"].mean())
        lora_effect = float(lora[condition]["exec"].mean() - lora["zero_shot"]["exec"].mean())
        low, high = C.bootstrap_did(base["zero_shot"]["exec"], base[condition]["exec"], lora["zero_shot"]["exec"], lora[condition]["exec"], rng=rng)
        interaction.append({
            "condition": condition, "condition_label": DISPLAY[condition],
            "base_fewshot_effect": base_effect, "lora_fewshot_effect": lora_effect,
            "difference_in_differences": lora_effect - base_effect,
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "bootstrap_ci_excludes_zero": low > 0 or high < 0,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "causal_interpretation_allowed": False,
        })
    gate_rows = gate_analysis("base", base, rng) + gate_analysis("lora_v2", lora, rng)
    case_rows = build_case_rows(tests, base, lora)
    completion = completion_rows(base, lora)

    transition_summary = {}
    for condition in CONDITIONS:
        transitions = Counter(row[f"base_to_lora_{condition}_transition"] for row in case_rows)
        execution_transitions = Counter(row[f"base_to_lora_{condition}_execution_transition"] for row in case_rows)
        n01_dbs = sorted({row["db_id"] for row in case_rows if row[f"base_to_lora_{condition}_transition"] == "wrong->correct"})
        n10_dbs = sorted({row["db_id"] for row in case_rows if row[f"base_to_lora_{condition}_transition"] == "correct->wrong"})
        transition_summary[condition] = {
            "execution_match": dict(transitions), "execution_success": dict(execution_transitions),
            "n01_unique_databases": len(n01_dbs), "n10_unique_databases": len(n10_dbs),
            "n01_database_ids": n01_dbs, "n10_database_ids": n10_dbs,
        }

    summary = {
        "schema_version": 1,
        "purpose": "complete_qwen35_9b_base_vs_official_lora_v2_8x8_final_evaluation_audit",
        "status": "PASS_WITH_METHODICAL_LIMITATIONS",
        "analysis_mode": "strict_read_only_existing_results_plus_additive_derived_outputs",
        "generation_started": False, "model_loaded": False, "adapter_loaded": False, "bge_loaded": False,
        "analysis_code_provenance": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "helper_script": str(HELPER_SCRIPT.relative_to(ROOT)), "helper_script_sha256": sha256(HELPER_SCRIPT),
            "shared_script": str(SHARED_SCRIPT.relative_to(ROOT)), "shared_script_sha256": sha256(SHARED_SCRIPT),
        },
        "official_model": {"registry_key": MODEL_REGISTRY_KEY, "model_id": MODEL_ID, "snapshot": MODEL_REVISION, "tokenizer_snapshot": MODEL_REVISION},
        "official_adapter": {
            "root": str(ADAPTER_ROOT.relative_to(ROOT)), "best_checkpoint": str(BEST_CHECKPOINT.relative_to(ROOT)),
            "adapter_sha256": sha256(ADAPTER_ROOT / "adapter_model.safetensors"), "root_equals_best": identity["root_equals_best"],
            "selection_metric": "Trainer Full-Chat eval_loss on MixedVal2500-v2", "best_metric": 0.4077516198158264,
        },
        "identity_checks": identity,
        "testset": {"path": str(TESTCASES.relative_to(ROOT)), "sha256": sha256(TESTCASES), "rows": len(tests), "difficulty_labels_available": False},
        "run_identification": {"base": {c: compact(base[c]) for c in CONDITIONS}, "lora_v2": {c: compact(lora[c]) for c in CONDITIONS}},
        "base_zero_1536_nonbinding_equivalence": base_zero_prompt_identity,
        "retrieval": {"index_path": str(INDEX_DIR.relative_to(ROOT)), "embedding_model": "BAAI/bge-large-en-v1.5", **overlap},
        "base_lora_trace_identity": trace_identity, "static_identity": static_check, "gate_reference_identity": gate_checks,
        "execution_rescoring": rescoring, "execution_rescoring_mismatch_count": rescore_mismatches, "rescore_path_disagreement_count": path_disagreements,
        "base_vs_lora_statistics": base_lora_stats,
        "base_fewshot_vs_zero_statistics": base_fs_stats,
        "lora_fewshot_vs_zero_statistics": lora_fs_stats,
        "difference_in_differences": interaction,
        "gate_analysis": gate_rows,
        "completion_diagnostics": completion,
        "transition_summary": transition_summary,
        "cross_model_context": cross_model_context(),
        "difficulty_analysis": {"status": "DIFFICULTY ANALYSIS NOT AVAILABLE", "reason": "Spider testcase artifacts do not contain authoritative difficulty labels."},
        "statistics": {
            "mcnemar": "exact two-sided binomial test", "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95,
            "holm_families": {"base_vs_lora": 8, "base_fewshot": 7, "lora_fewshot": 7},
        },
        "comparability": {
            "within_qwen9_base_vs_lora": "A",
            "base_zero_1536_vs_lora_2048": "A_with_documented_nonbinding_limit_exception",
            "structure_gates": "exploratory",
            "cross_family_context": "descriptive_only",
        },
        "warnings": [
            "Historical per-run terminal logs and immutable runtime runner hashes are absent for 14 runs.",
            "Model revision is reconstructed from the single local snapshot and prior provenance audits rather than persisted per run.",
            "Base Zero Shot used max_input_tokens=1536; maximum prompt length was 736 and all 1032 prompt-token counts equal the LoRA 2048 run.",
            "The Static Base run had a documented nonblocking HTTP 504 HEAD retry with unchanged cached model payloads.",
            "Structure Gate conditions are exploratory and Spider Dev is development-facing.",
            "Difficulty-stratified analysis was omitted because no authoritative difficulty labels are present.",
        ],
        "rerun_required": False,
    }

    write_csv_new(OUT_CASES, case_rows)
    write_csv_new(OUT_BASE_LORA, base_lora_stats)
    write_csv_new(OUT_BASE_FS, base_fs_stats)
    write_csv_new(OUT_LORA_FS, lora_fs_stats)
    write_csv_new(OUT_INTERACTION, interaction)
    write_csv_new(OUT_GATE, gate_rows)
    write_csv_new(OUT_COMPLETION, completion)
    write_json_new(OUT_SUMMARY, summary)
    print(json.dumps({
        "status": summary["status"], "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "base_ema": {c: base[c]["metrics"]["ema"] for c in CONDITIONS},
        "lora_ema": {c: lora[c]["metrics"]["ema"] for c in CONDITIONS},
        "rescore_mismatches": rescore_mismatches,
    }, indent=2))


if __name__ == "__main__":
    main()
