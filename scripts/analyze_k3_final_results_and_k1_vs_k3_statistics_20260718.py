#!/usr/bin/env python3
"""Authoritative read-only k3 synthesis and paired k1-vs-k3 statistics.

The script reads frozen prediction, metadata, trace, config, and audit artifacts.
It neither imports model/retrieval libraries nor executes SQL. Outputs are additive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N = 1032
SEED = 20260716
BOOTSTRAP_RESAMPLES = 10_000
HOLM_FAMILY = "K1_VS_K3_DEMONSTRATION_COUNT_FAMILY"
TESTSET = ROOT / "data/testcases_spider_dev_full.jsonl"
TESTSET_SHA256 = "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce"
INDEX = "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BASELINE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
INVENTORY = ROOT / "audits/derived/k3_all_runs_completion_inventory_after_repair_20260718.csv"
EQUIVALENCE = ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json"

OUT_METRICS = ROOT / "audits/derived/k3_authoritative_run_metrics_20260718.csv"
OUT_MAPPING = ROOT / "audits/derived/k1_k3_authoritative_pair_mapping_20260718.csv"
OUT_STATS = ROOT / "audits/derived/k1_vs_k3_paired_statistics_20260718.csv"
OUT_EFFICIENCY = ROOT / "audits/derived/k1_vs_k3_efficiency_comparison_20260718.csv"
OUT_GATE = ROOT / "audits/derived/k3_gate_and_fallback_distribution_20260718.csv"
OUT_DESCRIPTIVE = ROOT / "audits/derived/k3_descriptive_comparisons_20260718.csv"
OUT_STATE = ROOT / "audits/derived/k3_final_quantitative_state_20260718.json"

MODEL_ORDER = ("qwen2b", "llama3b", "qwen9b")
ROLE_ORDER = ("base", "lora_v2")
K3_ORDER = (
    "top3",
    "top3_gate070",
    "top3_gate085",
    "structure_top3",
    "structure_top3_gate070",
    "structure_top3_gate085",
)
K1_FOR_K3 = {
    "top3": "top1",
    "top3_gate070": "top1_gate070",
    "top3_gate085": "top1_gate085",
    "structure_top3": "structure",
    "structure_top3_gate070": "structure_gate070",
    "structure_top3_gate085": "structure_gate085",
}
UNGATED_FOR_GATE = {
    "top3_gate070": "top3",
    "top3_gate085": "top3",
    "structure_top3_gate070": "structure_top3",
    "structure_top3_gate085": "structure_top3",
}
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B",
    "llama3b": "Llama 3.2 3B Instruct",
    "qwen9b": "Qwen 3.5 9B",
}
MODEL_KEYS = {value: key for key, value in MODEL_LABELS.items()}
ROLE_LABELS = {"base": "Base", "lora_v2": "LoRA v2"}
CONDITION_LABELS = {
    "top3": "Dynamic Top-3",
    "top3_gate070": "Dynamic Top-3 Gate 0.70",
    "top3_gate085": "Dynamic Top-3 Gate 0.85",
    "structure_top3": "Structure Top-3",
    "structure_top3_gate070": "Structure Top-3 Gate 0.70",
    "structure_top3_gate085": "Structure Top-3 Gate 0.85",
}
CORE_EQUAL_FIELDS = (
    "llm", "adapter", "testcases_path", "prompt_format", "system_prompt_variant",
    "fewshot_example_schema_mode", "fewshot_example_mode", "retrieval_index_path",
    "retrieval_pool_path", "embedding_model", "prompt_tuning", "allow_overlap",
    "same_db_only", "extractor_mode", "max_new_tokens", "generation_batch_size",
    "compute_perplexity", "max_test_samples", "retrieval_method",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bool_int(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def exact_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for offset in range(0, BOOTSTRAP_RESAMPLES, 250):
        size = min(250, BOOTSTRAP_RESAMPLES - offset)
        indices = rng.integers(0, len(diff), size=(size, len(diff)))
        values[offset : offset + size] = diff[indices].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["mcnemar_p_raw"]))
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * float(row["mcnemar_p_raw"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["significant_unadjusted_0_05"] = float(row["mcnemar_p_raw"]) < 0.05
        rows[index]["significant_holm_0_05"] = running < 0.05


def metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    prompt = [int(float(row["prompt_tokens"])) for row in rows]
    completion = [int(float(row["completion_tokens"])) for row in rows]
    total = [int(float(row["total_tokens"])) for row in rows]
    raw_nonempty = sum(bool(row.get("raw_output", "").strip()) for row in rows)
    sql_nonempty = sum(bool(row.get("pred_sql", "").strip()) for row in rows)
    return {
        "cases": len(rows),
        "ema_correct": sum(bool_int(row["exec_match"]) for row in rows),
        "ema": sum(bool_int(row["exec_match"]) for row in rows) / len(rows),
        "esr_executable": sum(bool_int(row["pred_ok"]) for row in rows),
        "esr": sum(bool_int(row["pred_ok"]) for row in rows) / len(rows),
        "string_exact": float(np.mean(floats(rows, "string_exact"))),
        "normalized_exact": float(np.mean(floats(rows, "normalized_exact"))),
        "char_accuracy": float(np.mean(floats(rows, "char_accuracy"))),
        "token_accuracy": float(np.mean(floats(rows, "token_accuracy"))),
        "generation_time_mean": float(np.mean(floats(rows, "generation_time_seconds"))),
        "prompt_tokens_mean": float(np.mean(prompt)),
        "prompt_tokens_max": max(prompt),
        "completion_tokens_mean": float(np.mean(completion)),
        "completion_tokens_max": max(completion),
        "total_tokens_mean": float(np.mean(total)),
        "raw_output_empty": len(rows) - raw_nonempty,
        "sql_extraction_empty": len(rows) - sql_nonempty,
        "completion_limit_cases": sum(value == int(float(rows[0]["run_max_new_tokens"])) for value in completion),
        "prediction_timeout_cases": sum("timeout" in row.get("pred_error", "").lower() for row in rows),
        "gold_timeout_cases": sum("timeout" in row.get("gold_error", "").lower() for row in rows),
    }


def score_summary(rows: list[dict[str, str]]) -> dict[str, float | int]:
    scores: list[float] = []
    set_min: list[float] = []
    first: list[float] = []
    second: list[float] = []
    third: list[float] = []
    for row in rows:
        values = [float(value) for value in json.loads(row.get("retrieved_scores") or "[]")]
        scores.extend(values)
        if values:
            set_min.append(min(values)); first.append(values[0])
        if len(values) > 1: second.append(values[1])
        if len(values) > 2: third.append(values[2])
    return {
        "retrieval_similarity_selected_slot_mean": float(np.mean(scores)),
        "retrieval_similarity_min": min(scores),
        "retrieval_similarity_max": max(scores),
        "retrieval_set_min_mean": float(np.mean(set_min)),
        "retrieval_rank1_mean": float(np.mean(first)),
        "retrieval_rank2_mean": float(np.mean(second)) if second else float("nan"),
        "retrieval_rank3_mean": float(np.mean(third)) if third else float("nan"),
    }


def semantic_pair_check(k1_cfg: dict[str, Any], k3_cfg: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for field in CORE_EQUAL_FIELDS:
        if k1_cfg.get(field) != k3_cfg.get(field):
            warnings.append(f"unexpected:{field}:{k1_cfg.get(field)!r}!={k3_cfg.get(field)!r}")
    if k1_cfg.get("k") != 1 or k3_cfg.get("k") != 3:
        warnings.append("invalid_k_pair")
    if k1_cfg.get("max_input_tokens") != 2048 or k3_cfg.get("max_input_tokens") != 4352:
        warnings.append("unexpected_input_limits")
    if bool(k1_cfg.get("fewshot_gate_enabled")) != bool(k3_cfg.get("fewshot_gate_enabled")):
        warnings.append("gate_enabled_mismatch")
    for field in ("fewshot_gate_threshold", "retrieval_rerank_method", "retrieval_rerank_top_n", "retrieval_structure_bonus_max"):
        if k1_cfg.get(field) != k3_cfg.get(field):
            warnings.append(f"method:{field}:{k1_cfg.get(field)!r}!={k3_cfg.get(field)!r}")
    fatal = [item for item in warnings if item.startswith("unexpected:") or item in {"invalid_k_pair", "unexpected_input_limits", "gate_enabled_mismatch"}]
    return ("NOT_COMPARABLE" if fatal else "MATCHED_WITH_LIMITATIONS"), warnings


def descriptive_comparisons(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model_key"], row["model_role"], row["condition"]): row for row in metrics_rows}
    output: list[dict[str, Any]] = []
    def add(kind: str, model: str, role: str, condition: str, a: dict[str, Any], b: dict[str, Any]) -> None:
        output.append({
            "comparison_type": kind, "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role,
            "condition": condition, "reference_run_id": a["run_id"], "target_run_id": b["run_id"],
            "reference_label": f"{a['model_role']}:{a['condition']}", "target_label": f"{b['model_role']}:{b['condition']}",
            "reference_ema": a["ema"], "target_ema": b["ema"], "delta_pp": 100 * (float(b["ema"]) - float(a["ema"])),
            "reference_esr": a["esr"], "target_esr": b["esr"], "esr_delta_pp": 100 * (float(b["esr"]) - float(a["esr"])),
            "inference_status": "DESCRIPTIVE_ONLY_NO_NEW_SIGNIFICANCE_TEST",
        })
    for model in MODEL_ORDER:
        for condition in K3_ORDER:
            add("BASE_VS_LORA", model, "base_vs_lora", condition, lookup[(model, "base", condition)], lookup[(model, "lora_v2", condition)])
        for role in ROLE_ORDER:
            for top, structure in (("top3", "structure_top3"), ("top3_gate070", "structure_top3_gate070"), ("top3_gate085", "structure_top3_gate085")):
                add("TOP3_VS_STRUCTURE_TOP3", model, role, top, lookup[(model, role, top)], lookup[(model, role, structure)])
            for ungated, gate in (("top3", "top3_gate070"), ("top3", "top3_gate085"), ("structure_top3", "structure_top3_gate070"), ("structure_top3", "structure_top3_gate085")):
                add("UNGATED_VS_GATE", model, role, gate, lookup[(model, role, ungated)], lookup[(model, role, gate)])
            best = max((lookup[(model, role, condition)] for condition in K3_ORDER), key=lambda row: float(row["ema"]))
            output.append({"comparison_type": "BEST_K3_BY_MODEL_ROLE", "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role, "condition": best["condition"], "reference_run_id": "", "target_run_id": best["run_id"], "reference_label": "", "target_label": f"{role}:{best['condition']}", "reference_ema": "", "target_ema": best["ema"], "delta_pp": "", "reference_esr": "", "target_esr": best["esr"], "esr_delta_pp": "", "inference_status": "DESCRIPTIVE_ONLY_NO_NEW_SIGNIFICANCE_TEST"})
        best_line = max((lookup[(model, role, condition)] for role in ROLE_ORDER for condition in K3_ORDER), key=lambda row: float(row["ema"]))
        output.append({"comparison_type": "BEST_K3_BY_MODEL_LINE", "model_key": model, "model_line": MODEL_LABELS[model], "model_role": best_line["model_role"], "condition": best_line["condition"], "reference_run_id": "", "target_run_id": best_line["run_id"], "reference_label": "", "target_label": f"{best_line['model_role']}:{best_line['condition']}", "reference_ema": "", "target_ema": best_line["ema"], "delta_pp": "", "reference_esr": "", "target_esr": best_line["esr"], "esr_delta_pp": "", "inference_status": "DESCRIPTIVE_ONLY_NO_NEW_SIGNIFICANCE_TEST"})
    return output


def main() -> None:
    if "--finalize-descriptive" in sys.argv[1:]:
        if OUT_DESCRIPTIVE.exists() or not OUT_METRICS.exists():
            raise RuntimeError("Descriptive finalization requires existing metrics and a free target")
        rows = descriptive_comparisons(read_csv(OUT_METRICS))
        write_csv_new(OUT_DESCRIPTIVE, rows)
        print(json.dumps({"status": "PASS", "descriptive_rows": len(rows)}, indent=2))
        return
    for path in (OUT_METRICS, OUT_MAPPING, OUT_STATS, OUT_EFFICIENCY, OUT_GATE, OUT_DESCRIPTIVE, OUT_STATE):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite: {path}")
    if sha256(TESTSET) != TESTSET_SHA256:
        raise RuntimeError("Authoritative testset hash mismatch")
    tests = read_jsonl(TESTSET)
    case_ids = [row["id"] for row in tests]
    if len(case_ids) != N or len(set(case_ids)) != N:
        raise RuntimeError("Invalid authoritative testset")
    equivalence = read_json(EQUIVALENCE)
    if equivalence.get("status") != "PASS" or not equivalence.get("k1_vs_k3_comparison_permitted"):
        raise RuntimeError("K1 prompt-equivalence prerequisite is not PASS")

    baseline_rows = read_csv(BASELINE)
    inventory_rows = read_csv(INVENTORY)
    if len(baseline_rows) != 48 or len(inventory_rows) != 36:
        raise RuntimeError("Expected authoritative 48+36 run registers")
    baseline = {(row["model_key"], row["role"], row["condition"]): row for row in baseline_rows}
    inventory: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in inventory_rows:
        key = (MODEL_KEYS[row["model_line"]], row["model_role"], row["condition"])
        if key in inventory:
            raise RuntimeError(f"Duplicate k3 inventory key: {key}")
        if row["final_run_status"] not in {"COMPLETE_VALID", "COMPLETE_WITH_WARNING"}:
            raise RuntimeError(f"Non-authoritative k3 status: {key}")
        inventory[key] = row
    expected = {(model, role, condition) for model in MODEL_ORDER for role in ROLE_ORDER for condition in K3_ORDER}
    if set(inventory) != expected:
        raise RuntimeError("K3 inventory does not match the 36-run design")

    k3_data: dict[tuple[str, str, str], dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []
    summary_mismatches: list[str] = []
    for key in sorted(expected, key=lambda value: (MODEL_ORDER.index(value[0]), ROLE_ORDER.index(value[1]), K3_ORDER.index(value[2]))):
        rec = inventory[key]
        paths = {
            "config": ROOT / rec["config_path"], "csv": ROOT / rec["output_path"],
            "metadata": ROOT / rec["metadata_path"], "trace": ROOT / rec["trace_path"],
        }
        for name, field in (("config", "config_sha256"), ("csv", "output_sha256"), ("metadata", "metadata_sha256"), ("trace", "trace_sha256")):
            if sha256(paths[name]) != rec[field]:
                raise RuntimeError(f"K3 artifact hash mismatch: {key}:{name}")
        rows = read_csv(paths["csv"]); traces = read_jsonl(paths["trace"]); metadata = read_json(paths["metadata"])
        if len(rows) != N or len(traces) != N or [row["id"] for row in rows] != case_ids or [row["id"] for row in traces] != case_ids:
            raise RuntimeError(f"K3 case integrity failure: {key}")
        if len({row["id"] for row in rows}) != N:
            raise RuntimeError(f"K3 duplicate IDs: {key}")
        metric = metric_summary(rows); score = score_summary(rows)
        checks = {
            "ema": (metric["ema"], float(metadata["execution_match_accuracy"])),
            "esr": (metric["esr"], float(metadata["execution_success_rate"])),
            "string": (metric["string_exact"], float(metadata["string_exact_match"])),
            "normalized": (metric["normalized_exact"], float(metadata["normalized_exact_match"])),
            "char": (metric["char_accuracy"], float(metadata["char_accuracy_avg"])),
            "token": (metric["token_accuracy"], float(metadata["token_accuracy_avg"])),
            "generation_time": (metric["generation_time_mean"], float(metadata["avg_generation_time_seconds"])),
            "prompt_tokens": (metric["prompt_tokens_mean"], float(metadata["avg_prompt_tokens"])),
            "completion_tokens": (metric["completion_tokens_mean"], float(metadata["avg_completion_tokens"])),
            "total_tokens": (metric["total_tokens_mean"], float(metadata["avg_total_tokens"])),
        }
        local_mismatch = [name for name, (a, b) in checks.items() if abs(a - b) > 1e-7]
        if local_mismatch:
            summary_mismatches.append(f"{key}:{','.join(local_mismatch)}")
        decisions = Counter(row.get("gate_decision") or "ungated" for row in rows)
        effective_k3 = decisions["fewshot"] if key[2] in UNGATED_FOR_GATE else N
        effective_k0 = decisions["zero_shot"] if key[2] in UNGATED_FOR_GATE else 0
        unexpected = N - effective_k3 - effective_k0
        if unexpected:
            raise RuntimeError(f"Unexpected actual_k: {key}")
        k3_data[key] = {"rows": rows, "by_id": {row["id"]: row for row in rows}, "traces": traces, "metadata": metadata, "record": rec}
        metrics_rows.append({
            "model_key": key[0], "model_line": MODEL_LABELS[key[0]], "model_role": key[1], "role_label": ROLE_LABELS[key[1]],
            "condition": key[2], "condition_label": CONDITION_LABELS[key[2]], "run_id": rec["run_id"], "cases": N,
            **metric, **score, "effective_k3_cases": effective_k3, "zero_shot_fallback_cases": effective_k0,
            "actual_k_distribution": json.dumps({"3": effective_k3, "0": effective_k0}, sort_keys=True),
            "max_input_tokens": int(rows[0]["run_max_input_tokens"]), "max_new_tokens": int(rows[0]["run_max_new_tokens"]),
            "timeout_policy": "900s_per_gold_and_prediction_statement" if key[0] == "qwen9b" else "no_explicit_statement_timeout",
            "config_path": rec["config_path"], "config_sha256": rec["config_sha256"],
            "result_path": rec["output_path"], "result_sha256": rec["output_sha256"],
            "metadata_path": rec["metadata_path"], "metadata_sha256": rec["metadata_sha256"],
            "trace_path": rec["trace_path"], "trace_sha256": rec["trace_sha256"],
            "summary_reproduction_status": "PASS" if not local_mismatch else "FAIL",
            "run_status": rec["final_run_status"],
        })
    if summary_mismatches:
        raise RuntimeError(f"K3 summary mismatches: {summary_mismatches}")

    baseline_data: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rec in baseline.items():
        csv_path = ROOT / rec["csv_path"]
        if sha256(csv_path) != rec["csv_sha256"]:
            raise RuntimeError(f"Baseline CSV hash mismatch: {key}")
        if sha256(ROOT / rec["config_path"]) != rec["config_sha256"] or sha256(ROOT / rec["metadata_path"]) != rec["metadata_sha256"]:
            raise RuntimeError(f"Baseline provenance hash mismatch: {key}")
        if rec.get("trace_path") and sha256(ROOT / rec["trace_path"]) != rec["trace_sha256"]:
            raise RuntimeError(f"Baseline trace hash mismatch: {key}")
        rows = read_csv(csv_path)
        if len(rows) != N or [row["id"] for row in rows] != case_ids:
            raise RuntimeError(f"Baseline case integrity failure: {key}")
        baseline_data[key] = {"rows": rows, "by_id": {row["id"]: row for row in rows}, "record": rec}

    mapping_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for model in MODEL_ORDER:
        for role in ROLE_ORDER:
            for condition in K3_ORDER:
                k3_key = (model, role, condition); k1_condition = K1_FOR_K3[condition]; k1_key = (model, role, k1_condition)
                k3_rec = inventory[k3_key]; k1_rec = baseline[k1_key]
                k1_cfg = read_json(ROOT / k1_rec["config_path"]); k3_cfg = read_json(ROOT / k3_rec["config_path"])
                comparison_status, differences = semantic_pair_check(k1_cfg, k3_cfg)
                if comparison_status != "MATCHED_WITH_LIMITATIONS":
                    raise RuntimeError(f"Pair is not comparable: {k3_key}: {differences}")
                k1_rows = baseline_data[k1_key]["rows"]; k3_rows = k3_data[k3_key]["rows"]
                if [row["id"] for row in k1_rows] != [row["id"] for row in k3_rows]:
                    raise RuntimeError(f"Pair order mismatch: {k3_key}")
                k1_metric = metric_summary(k1_rows); k3_metric = metric_summary(k3_rows)
                k1_score = score_summary(k1_rows); k3_score = score_summary(k3_rows)
                warning = "max_input_tokens_2048_vs_4352"
                if model == "qwen9b": warning += ";k3_900s_statement_timeout_vs_k1_no_explicit_statement_timeout"
                mapping_rows.append({
                    "pair_id": f"{model}:{role}:{k1_condition}_vs_{condition}", "model_key": model, "model_line": MODEL_LABELS[model],
                    "model_role": role, "role_label": ROLE_LABELS[role], "semantic_condition": condition,
                    "k1_condition": k1_condition, "k3_condition": condition, "k1_run_id": k1_rec["run_id"], "k3_run_id": k3_rec["run_id"],
                    "k1_config_path": k1_rec["config_path"], "k1_config_sha256": k1_rec["config_sha256"],
                    "k3_config_path": k3_rec["config_path"], "k3_config_sha256": k3_rec["config_sha256"],
                    "k1_result_path": k1_rec["csv_path"], "k1_result_sha256": k1_rec["csv_sha256"],
                    "k3_result_path": k3_rec["output_path"], "k3_result_sha256": k3_rec["output_sha256"],
                    "k1_metadata_path": k1_rec["metadata_path"], "k3_metadata_path": k3_rec["metadata_path"],
                    "k1_trace_path": k1_rec["trace_path"], "k3_trace_path": k3_rec["trace_path"],
                    "cases": N, "case_order_identical": True, "model_adapter_match": True, "testset_match": True,
                    "prompt_system_schema_match": True, "retrieval_method_gate_structure_match": True,
                    "extractor_decoding_output_limit_batch_match": True, "k1_k": 1, "k3_k": 3,
                    "k1_max_input_tokens": 2048, "k3_max_input_tokens": 4352,
                    "k1_timeout_policy": "no_explicit_statement_timeout", "k3_timeout_policy": "900s_per_statement" if model == "qwen9b" else "no_explicit_statement_timeout",
                    "expected_differences": "k=1_vs_3;prompt_length;max_input_tokens;output_identity" + (";qwen9b_k3_timeout_policy" if model == "qwen9b" else ""),
                    "config_method_notes": json.dumps(differences, ensure_ascii=False), "comparability_status": comparison_status,
                    "comparability_warning": warning,
                })
                efficiency_rows.append({
                    "pair_id": mapping_rows[-1]["pair_id"], "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role,
                    "condition": condition, "k1_prompt_tokens_mean": k1_metric["prompt_tokens_mean"], "k3_prompt_tokens_mean": k3_metric["prompt_tokens_mean"],
                    "prompt_tokens_delta": k3_metric["prompt_tokens_mean"] - k1_metric["prompt_tokens_mean"],
                    "k1_completion_tokens_mean": k1_metric["completion_tokens_mean"], "k3_completion_tokens_mean": k3_metric["completion_tokens_mean"],
                    "completion_tokens_delta": k3_metric["completion_tokens_mean"] - k1_metric["completion_tokens_mean"],
                    "k1_total_tokens_mean": k1_metric["total_tokens_mean"], "k3_total_tokens_mean": k3_metric["total_tokens_mean"],
                    "total_tokens_delta": k3_metric["total_tokens_mean"] - k1_metric["total_tokens_mean"],
                    "k1_generation_time_mean_seconds": k1_metric["generation_time_mean"], "k3_generation_time_mean_seconds": k3_metric["generation_time_mean"],
                    "generation_time_delta_seconds": k3_metric["generation_time_mean"] - k1_metric["generation_time_mean"],
                    "generation_time_relative_change": (k3_metric["generation_time_mean"] / k1_metric["generation_time_mean"] - 1.0) if k1_metric["generation_time_mean"] else None,
                    "k1_retrieval_similarity_selected_slot_mean": k1_score["retrieval_similarity_selected_slot_mean"],
                    "k3_retrieval_similarity_selected_slot_mean": k3_score["retrieval_similarity_selected_slot_mean"],
                    "retrieval_similarity_delta": k3_score["retrieval_similarity_selected_slot_mean"] - k1_score["retrieval_similarity_selected_slot_mean"],
                    "k1_esr": k1_metric["esr"], "k3_esr": k3_metric["esr"], "esr_delta_pp": 100 * (k3_metric["esr"] - k1_metric["esr"]),
                    "k1_ema": k1_metric["ema"], "k3_ema": k3_metric["ema"], "ema_delta_pp": 100 * (k3_metric["ema"] - k1_metric["ema"]),
                    "tokenizer_comparability_note": "within_model_pair_only;cross_model_token_counts_not_tokenizer_independent",
                    "comparability_warning": warning,
                })
                a = np.array([bool_int(row["exec_match"]) for row in k1_rows], dtype=np.int8)
                b = np.array([bool_int(row["exec_match"]) for row in k3_rows], dtype=np.int8)
                diff = b.astype(float) - a.astype(float)
                n01 = int(np.sum((a == 0) & (b == 1))); n10 = int(np.sum((a == 1) & (b == 0)))
                ci_low, ci_high = bootstrap_ci(diff, rng)
                delta = float(diff.mean())
                stats_rows.append({
                    "family": HOLM_FAMILY, "family_size": 36, "pair_id": mapping_rows[-1]["pair_id"], "model_key": model,
                    "model_line": MODEL_LABELS[model], "model_role": role, "role_label": ROLE_LABELS[role], "condition": condition,
                    "cases": N, "k1_correct": int(a.sum()), "k3_correct": int(b.sum()), "k1_ema": float(a.mean()), "k3_ema": float(b.mean()),
                    "delta_ema": delta, "delta_pp": 100 * delta, "k1_wrong_k3_correct_n01": n01, "k1_correct_k3_wrong_n10": n10,
                    "both_correct": int(np.sum((a == 1) & (b == 1))), "both_wrong": int(np.sum((a == 0) & (b == 0))),
                    "mcnemar_p_raw": exact_mcnemar_p(n01, n10), "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
                    "bootstrap_ci_low_pp": 100 * ci_low, "bootstrap_ci_high_pp": 100 * ci_high,
                    "bootstrap_seed": SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "direction": "K3_BETTER" if delta > 0 else "K3_WORSE" if delta < 0 else "IDENTICAL_POINT_ESTIMATE",
                    "comparability_status": comparison_status, "comparability_warning": warning,
                })
    if len(mapping_rows) != 36 or len(stats_rows) != 36:
        raise RuntimeError("Expected 36 authoritative pairs")
    holm_adjust(stats_rows)
    for row in stats_rows:
        if row["significant_holm_0_05"]:
            row["significance_status"] = "HOLM_SIGNIFICANT_K3_ADVANTAGE" if row["delta_pp"] > 0 else "HOLM_SIGNIFICANT_K3_DISADVANTAGE"
        else:
            row["significance_status"] = "NOT_HOLM_SIGNIFICANT"

    gate_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for role in ROLE_ORDER:
            zero = baseline_data[(model, role, "zero_shot")]["by_id"]
            other_role = "lora_v2" if role == "base" else "base"
            for condition in K3_ORDER:
                data = k3_data[(model, role, condition)]
                traces = data["traces"]; rows = data["rows"]
                all_scores: list[float] = []; set_mins: list[float] = []
                unique_three = 0; missing_retrieval = 0; unexpected_k = 0
                for trace, row in zip(traces, rows):
                    ids = [str(value) for value in trace.get("retrieved_ids", [])]
                    scores = [float(value) for value in trace.get("retrieved_scores", [])]
                    all_scores.extend(scores)
                    if scores: set_mins.append(min(scores))
                    if len(ids) == 3 and len(set(ids)) == 3 and len(scores) == 3: unique_three += 1
                    else: missing_retrieval += 1
                    decision = row.get("gate_decision") or "ungated"
                    if decision not in ({"fewshot", "zero_shot"} if condition in UNGATED_FOR_GATE else {"ungated"}): unexpected_k += 1
                decisions = Counter(row.get("gate_decision") or "ungated" for row in rows)
                accepted = decisions["fewshot"] if condition in UNGATED_FOR_GATE else N
                fallback = decisions["zero_shot"] if condition in UNGATED_FOR_GATE else 0
                base_lora_other = k3_data[(model, other_role, condition)]["traces"]
                role_identity = sum(
                    left.get("retrieved_ids") == right.get("retrieved_ids") and left.get("retrieved_scores") == right.get("retrieved_scores")
                    for left, right in zip(traces, base_lora_other)
                )
                gate_selection_identity = N
                fallback_identity = accepted_identity = None
                if condition in UNGATED_FOR_GATE:
                    ungated = k3_data[(model, role, UNGATED_FOR_GATE[condition])]
                    gate_selection_identity = sum(
                        left.get("retrieved_ids") == right.get("retrieved_ids") and left.get("retrieved_scores") == right.get("retrieved_scores")
                        for left, right in zip(traces, ungated["traces"])
                    )
                    fallback_identity = sum(
                        row["raw_output"] == zero[row["id"]]["raw_output"]
                        and row["pred_sql"] == zero[row["id"]]["pred_sql"]
                        and bool_int(row["pred_ok"]) == bool_int(zero[row["id"]]["pred_ok"])
                        and bool_int(row["exec_match"]) == bool_int(zero[row["id"]]["exec_match"])
                        for row in rows if row.get("gate_decision") == "zero_shot"
                    )
                    ungated_by_id = ungated["by_id"]
                    accepted_identity = sum(
                        row["raw_output"] == ungated_by_id[row["id"]]["raw_output"]
                        and row["pred_sql"] == ungated_by_id[row["id"]]["pred_sql"]
                        and bool_int(row["pred_ok"]) == bool_int(ungated_by_id[row["id"]]["pred_ok"])
                        and bool_int(row["exec_match"]) == bool_int(ungated_by_id[row["id"]]["exec_match"])
                        for row in rows if row.get("gate_decision") == "fewshot"
                    )
                gate_rows.append({
                    "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role, "condition": condition,
                    "cases": N, "effective_k3_cases": accepted, "zero_shot_fallback_cases": fallback,
                    "effective_k3_rate": accepted / N, "zero_shot_fallback_rate": fallback / N,
                    "retrieval_slots": len(all_scores), "retrieval_score_mean": float(np.mean(all_scores)), "retrieval_score_min": min(all_scores), "retrieval_score_max": max(all_scores),
                    "set_min_score_mean": float(np.mean(set_mins)), "three_distinct_demos_and_scores_cases": unique_three,
                    "missing_or_invalid_retrieval_cases": missing_retrieval, "unexpected_actual_k_cases": unexpected_k,
                    "base_lora_demo_score_identity_cases": role_identity, "gate_vs_ungated_selection_identity_cases": gate_selection_identity,
                    "fallback_output_identity_cases": fallback_identity, "accepted_output_identity_cases": accepted_identity,
                    "fallback_identity_status": "PASS" if fallback_identity in {None, fallback} else "FAIL",
                    "accepted_identity_status": "PASS" if accepted_identity in {None, accepted} else "FAIL",
                    "similarity_interpretation": "uncalibrated_similarity_not_probability",
                })
    if any(row["three_distinct_demos_and_scores_cases"] != N or row["base_lora_demo_score_identity_cases"] != N or row["gate_vs_ungated_selection_identity_cases"] != N or row["fallback_identity_status"] != "PASS" or row["accepted_identity_status"] != "PASS" for row in gate_rows):
        raise RuntimeError("Retrieval, gate, or output-reference identity failed")

    deltas = [float(row["delta_pp"]) for row in stats_rows]
    positive = sum(value > 0 for value in deltas); negative = sum(value < 0 for value in deltas); identical = sum(value == 0 for value in deltas)
    sig_adv = sum(bool(row["significant_holm_0_05"]) and float(row["delta_pp"]) > 0 for row in stats_rows)
    sig_harm = sum(bool(row["significant_holm_0_05"]) and float(row["delta_pp"]) < 0 for row in stats_rows)
    aggregate = {
        "positive_differences": positive, "negative_differences": negative, "identical_differences": identical,
        "holm_significant_k3_advantages": sig_adv, "holm_significant_k3_disadvantages": sig_harm,
        "not_holm_significant": 36 - sig_adv - sig_harm, "mean_delta_pp": float(np.mean(deltas)), "median_delta_pp": float(np.median(deltas)),
        "min_delta_pp": min(deltas), "max_delta_pp": max(deltas),
        "largest_advantage_pair": max(stats_rows, key=lambda row: float(row["delta_pp"]))["pair_id"],
        "largest_disadvantage_pair": min(stats_rows, key=lambda row: float(row["delta_pp"]))["pair_id"],
        "mean_prompt_token_delta": float(np.mean([float(row["prompt_tokens_delta"]) for row in efficiency_rows])),
        "mean_generation_time_delta_seconds": float(np.mean([float(row["generation_time_delta_seconds"]) for row in efficiency_rows])),
        "mean_generation_time_relative_change": float(np.mean([float(row["generation_time_relative_change"]) for row in efficiency_rows])),
    }
    best_by_role = []
    for model in MODEL_ORDER:
        for role in ROLE_ORDER:
            candidates = [row for row in metrics_rows if row["model_key"] == model and row["model_role"] == role]
            best = max(candidates, key=lambda row: float(row["ema"]))
            best_by_role.append({"model_key": model, "model_role": role, "condition": best["condition"], "ema": best["ema"], "run_id": best["run_id"]})

    write_csv_new(OUT_METRICS, metrics_rows)
    write_csv_new(OUT_MAPPING, mapping_rows)
    write_csv_new(OUT_STATS, stats_rows)
    write_csv_new(OUT_EFFICIENCY, efficiency_rows)
    write_csv_new(OUT_GATE, gate_rows)
    descriptive_rows = descriptive_comparisons(metrics_rows)
    write_csv_new(OUT_DESCRIPTIVE, descriptive_rows)
    state = {
        "status": "PASS_WITH_WARNINGS", "k3_runs": 36, "pre_k3_runs": 48, "pairs": 36, "cases_per_run": N,
        "testset": {"path": str(TESTSET.relative_to(ROOT)), "sha256": TESTSET_SHA256},
        "retrieval": {"index": INDEX, "embedding_model": EMBEDDING_MODEL, "pool_cases": 6960},
        "bootstrap": {"seed": SEED, "resamples": BOOTSTRAP_RESAMPLES, "ci": 0.95, "method": "paired_percentile"},
        "holm_family": {"name": HOLM_FAMILY, "size": 36}, "aggregate": aggregate, "best_k3_by_model_role": best_by_role,
        "gate_distribution_per_role": {
            "top3_gate070": {"k3": 480, "fallback": 552}, "top3_gate085": {"k3": 7, "fallback": 1025},
            "structure_top3_gate070": {"k3": 450, "fallback": 582}, "structure_top3_gate085": {"k3": 6, "fallback": 1026},
        },
        "summary_reproduction_mismatches": summary_mismatches,
        "input_limit_limitation": "k1=2048; k3=4352; prompt equivalence PASS; not an isolated one-factor demonstration-count effect",
        "timeout_limitation": "24 k3 runs without explicit statement timeout; 12 Qwen9B v3 runs with 900s per statement; effective in two Qwen9B Base runs",
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (BASELINE, INVENTORY, EQUIVALENCE, TESTSET)},
        "generated": {str(path.relative_to(ROOT)): sha256(path) for path in (OUT_METRICS, OUT_MAPPING, OUT_STATS, OUT_EFFICIENCY, OUT_GATE, OUT_DESCRIPTIVE)},
    }
    write_json_new(OUT_STATE, state)
    print(json.dumps({"status": state["status"], "runs": 36, "pairs": 36, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
