#!/usr/bin/env python3
"""Complete read-only analysis of the 24 frozen LoRA-v2 evaluation runs.

The script reads only frozen configs, run CSVs, metadata, retrieval traces,
SQLite databases, and authoritative manifests.  It never imports model,
adapter, tokenizer, or retrieval libraries.  Every output is additive and the
script refuses to run when any target already exists.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen35_matplotlib_cache_complete24")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_project_completion_and_remaining_gaps import build_gap_rows, environment_snapshot


DATE = "20260716"
N = 1032
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260716
MODEL_ORDER = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B LoRA v2",
    "llama3b": "Llama 3.2 3B Instruct LoRA v2",
    "qwen9b": "Qwen 3.5 9B LoRA v2",
}
CONDITIONS = [
    "zero_shot", "top1", "top1_gate070", "top1_gate085",
    "static_seed42", "structure", "structure_gate070", "structure_gate085",
]
COND_LABELS = {
    "zero_shot": "Zero Shot", "top1": "Top-1", "top1_gate070": "Top-1 Gate 0.70",
    "top1_gate085": "Top-1 Gate 0.85", "static_seed42": "Static",
    "structure": "Structure", "structure_gate070": "Structure Gate 0.70",
    "structure_gate085": "Structure Gate 0.85",
}

CROSS_MANIFEST = ROOT / "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json"
EXPECTED = {
    "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json": "24b4dec07d2d4981b42ce22e1295d27b0ccd9cbcc10666a422118b267fd14e37",
    "scripts/analyze_cross_model_complete_8x8_synthesis.py": "95dbda1932ec957bba9fa54c3ddbb8963ded1d7d974ced492e5199d3fd6b6475",
    "audits/audit_conservative_final_error_analysis_synthesis_20260716.md": "144266173d194c05763867662b9233105c38dfe0ea0d952c9115875ea2f0e2b0",
    "audits/conservative_final_error_analysis_synthesis_manifest_20260716.json": "8a1cec37f6a4cf79a5ee666ba0a5456c0901bfdf1c82df9ea8fb298713c39a0d",
    "scripts/build_conservative_final_error_analysis_synthesis.py": "a703a69b46e4cc103ef66dff916368db1f5e1b8c3e55491b66c1cac5d18ff8f8",
}

OUT = {
    "audit": ROOT / f"audits/audit_complete_24_lora_run_error_analysis_and_project_gap_review_{DATE}.md",
    "manifest": ROOT / f"audits/complete_24_lora_run_error_analysis_and_project_gap_review_manifest_{DATE}.json",
    "profiles": ROOT / f"audits/derived/complete_24_lora_run_results_and_error_profiles_{DATE}.csv",
    "labels": ROOT / f"audits/derived/complete_24_lora_run_error_labels_long_{DATE}.csv",
    "transitions": ROOT / f"audits/derived/lora_cross_prompt_zero_shot_transitions_{DATE}.csv",
    "family_transitions": ROOT / f"audits/derived/lora_cross_prompt_error_family_transitions_{DATE}.csv",
    "robustness": ROOT / f"audits/derived/lora_cross_prompt_case_robustness_{DATE}.csv",
    "overlap": ROOT / f"audits/derived/lora_cross_prompt_error_set_overlap_{DATE}.csv",
    "gate": ROOT / f"audits/derived/lora_gate_partition_error_analysis_{DATE}.csv",
    "ungated_gated": ROOT / f"audits/derived/lora_ungated_vs_gated_error_transitions_{DATE}.csv",
    "top1_structure": ROOT / f"audits/derived/lora_top1_vs_structure_error_analysis_{DATE}.csv",
    "static": ROOT / f"audits/derived/lora_static_vs_dynamic_error_analysis_{DATE}.csv",
    "output": ROOT / f"audits/derived/complete_24_lora_run_output_termination_diagnostics_{DATE}.csv",
    "esr_ema": ROOT / f"audits/derived/complete_24_lora_run_esr_ema_gap_{DATE}.csv",
    "failure": ROOT / f"audits/derived/lora_cross_run_empirical_failure_frequency_{DATE}.csv",
    "examples": ROOT / f"audits/derived/complete_24_lora_run_qualitative_examples_{DATE}.csv",
    "thesis_tables": ROOT / f"audits/derived/complete_24_lora_run_error_analysis_thesis_ready_tables_{DATE}.md",
    "thesis_text": ROOT / f"audits/derived/complete_24_lora_run_error_analysis_thesis_ready_text_{DATE}.md",
    "supported": ROOT / f"audits/derived/complete_24_lora_run_supported_statements_{DATE}.md",
    "limitations": ROOT / f"audits/derived/complete_24_lora_run_limitations_{DATE}.md",
    "gap": ROOT / f"audits/derived/project_completion_gap_matrix_{DATE}.csv",
    "checklist": ROOT / f"audits/derived/project_completion_action_checklist_{DATE}.md",
    "rq": ROOT / f"audits/derived/final_research_questions_results_matrix_{DATE}.md",
}
PLOTS = {
    "results": ROOT / f"audits/plots/complete_24_lora_results_and_residual_errors_{DATE}",
    "transitions": ROOT / f"audits/plots/complete_24_lora_prompt_benefit_harm_{DATE}",
    "families": ROOT / f"audits/plots/complete_24_lora_broad_error_families_{DATE}",
    "robustness": ROOT / f"audits/plots/complete_24_lora_prompt_robustness_{DATE}",
    "jaccard": ROOT / f"audits/plots/complete_24_lora_incorrect_set_jaccard_{DATE}",
    "gate": ROOT / f"audits/plots/complete_24_lora_gate_partitions_{DATE}",
    "gap": ROOT / f"audits/plots/complete_24_lora_esr_ema_gap_{DATE}",
    "frequency": ROOT / f"audits/plots/complete_24_lora_failure_frequency_{DATE}",
    "completion": ROOT / f"audits/plots/complete_24_lora_project_completion_status_{DATE}",
}


def load_taxonomy() -> Any:
    path = ROOT / "scripts/analyze_cross_model_zero_shot_error_taxonomy.py"
    spec = importlib.util.spec_from_file_location("frozen_zero_shot_taxonomy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import taxonomy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZE = load_taxonomy()

BROAD = {
    "A_OUTPUT_CONTROL": set("EMPTY_RAW_OUTPUT EMPTY_SQL_EXTRACTION NO_TERMINATING_SEMICOLON MULTIPLE_STATEMENTS MARKDOWN_OR_EXPLANATION THINK_MARKER REPETITIVE_GENERATION COMPLETION_LIMIT_REACHED TRUNCATED_SQL EXTRACTOR_FAILURE CHAT_TEMPLATE_ARTIFACT LONGER_OR_UNSTABLE_OUTPUT".split()),
    "B_SYNTAX_EXECUTION": set("SQL_PARSE_ERROR SQL_SYNTAX_ERROR SQLITE_EXECUTION_ERROR UNKNOWN_TABLE UNKNOWN_COLUMN AMBIGUOUS_COLUMN TYPE_OR_FUNCTION_ERROR OTHER_EXECUTION_ERROR".split()),
    "C_SCHEMA_PROJECTION": set("WRONG_TABLE MISSING_TABLE EXTRA_TABLE WRONG_COLUMN MISSING_COLUMN EXTRA_COLUMN WRONG_SCHEMA_LINK WRONG_SELECT_EXPRESSION MISSING_SELECT_EXPRESSION EXTRA_SELECT_EXPRESSION WRONG_SELECT_CARDINALITY WRONG_DISTINCT MISSING_DISTINCT EXTRA_DISTINCT".split()),
    "D_QUERY_STRUCTURE_LOGIC": set("MISSING_JOIN EXTRA_JOIN WRONG_JOIN_TABLE WRONG_JOIN_PATH WRONG_JOIN_CONDITION CARTESIAN_PRODUCT OVER_JOIN MISSING_WHERE_CONDITION EXTRA_WHERE_CONDITION WRONG_WHERE_COLUMN WRONG_OPERATOR WRONG_LITERAL_VALUE WRONG_LOGICAL_CONNECTOR WRONG_NEGATION WRONG_NULL_HANDLING WRONG_BETWEEN_OR_RANGE WRONG_LIKE_PATTERN WRONG_AGGREGATION MISSING_AGGREGATION EXTRA_AGGREGATION WRONG_COUNT_TARGET WRONG_MIN_MAX WRONG_SUM_AVG MISSING_GROUP_BY EXTRA_GROUP_BY WRONG_GROUP_BY MISSING_HAVING EXTRA_HAVING WRONG_HAVING MISSING_ORDER_BY EXTRA_ORDER_BY WRONG_ORDER_COLUMN WRONG_SORT_DIRECTION MISSING_LIMIT EXTRA_LIMIT WRONG_LIMIT_VALUE WRONG_ORDER_BY MISSING_SUBQUERY EXTRA_SUBQUERY WRONG_SUBQUERY WRONG_NESTING_LEVEL WRONG_CORRELATION MISSING_SET_OPERATION EXTRA_SET_OPERATION WRONG_SET_OPERATION WRONG_UNION WRONG_INTERSECT WRONG_EXCEPT DEMO_LITERAL_COPY DEMO_STRUCTURE_COPY".split()),
    "E_RESULT_DEVIATION": set("WRONG_RESULT_CARDINALITY DUPLICATE_ROWS MISSING_ROWS EXTRA_ROWS WRONG_SCALAR_VS_LIST ORDER_ONLY_MISMATCH".split()),
    "F_UNCLEAR_HEURISTIC": set("COMPLEX_MULTI_COMPONENT_ERROR HEURISTIC_ONLY MANUAL_REVIEW_REQUIRED UNCLASSIFIED PARSER_DISAGREEMENT PARSER_UNAVAILABLE".split()),
}
BROAD_LABEL = {
    "A_OUTPUT_CONTROL": "Output/Kontrolle", "B_SYNTAX_EXECUTION": "Syntax/Ausführung",
    "C_SCHEMA_PROJECTION": "Schema/Projektion", "D_QUERY_STRUCTURE_LOGIC": "Querylogik",
    "E_RESULT_DEVIATION": "Ergebnisabweichung", "F_UNCLEAR_HEURISTIC": "Unklar/heuristisch",
}
LABEL_BROAD = {label: family for family, labels in BROAD.items() for label in labels}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def save_plot(base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        target = base.with_suffix(suffix)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def exact_mcnemar(n01: int, n10: int) -> float:
    return ZE.exact_mcnemar(n01, n10)


def bootstrap_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        size = min(250, BOOTSTRAP_RESAMPLES - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        values[start:start + size] = diff[idx].mean(axis=1) * 100.0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "mcnemar_p") -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    running = 0.0
    m = len(rows)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[idx][p_key]))
        rows[idx]["holm_7_p"] = running
        rows[idx]["holm_significant_0_05"] = running < 0.05


def case_order_hash(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def metric_values(rows: list[dict[str, str]]) -> dict[str, float | int]:
    return {
        "correct": sum(truth(row["exec_match"]) for row in rows),
        "ema": sum(truth(row["exec_match"]) for row in rows) / len(rows),
        "executable": sum(truth(row["pred_ok"]) for row in rows),
        "esr": sum(truth(row["pred_ok"]) for row in rows) / len(rows),
        "string_exact": sum(truth(row["string_exact"]) for row in rows) / len(rows),
        "normalized_exact": sum(truth(row["normalized_exact"]) for row in rows) / len(rows),
        "char_accuracy": sum(float(row["char_accuracy"]) for row in rows) / len(rows),
        "token_accuracy": sum(float(row["token_accuracy"]) for row in rows) / len(rows),
    }


def top_level_statement_count(sql: str) -> int:
    return len([part for part in ZE.split_top(ZE.tokenize(sql), ";") if part])


def label_sets_for(
    labels: dict[tuple[str, str, str], dict[str, dict[str, str]]],
    model: str, condition: str, case_id: str,
) -> set[str]:
    return set(labels[(model, condition, case_id)])


def broad_sets_for(
    labels: dict[tuple[str, str, str], dict[str, dict[str, str]]],
    model: str, condition: str, case_id: str,
) -> set[str]:
    return {LABEL_BROAD.get(label, "F_UNCLEAR_HEURISTIC") for label in label_sets_for(labels, model, condition, case_id)}


def transition_counts(left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, int]:
    lc = np.array([truth(row["exec_match"]) for row in left], dtype=bool)
    rc = np.array([truth(row["exec_match"]) for row in right], dtype=bool)
    return {
        "right_repairs_left": int((~lc & rc).sum()),
        "right_harms_left": int((lc & ~rc).sum()),
        "both_wrong": int((~lc & ~rc).sum()),
        "both_correct": int((lc & rc).sum()),
    }


def family_delta_json(
    labels: dict[tuple[str, str, str], dict[str, dict[str, str]]], model: str,
    left_condition: str, right_condition: str, ids: list[str],
) -> tuple[str, str]:
    removed: dict[str, int] = {}
    introduced: dict[str, int] = {}
    for family in BROAD:
        left = {cid for cid in ids if family in broad_sets_for(labels, model, left_condition, cid)}
        right = {cid for cid in ids if family in broad_sets_for(labels, model, right_condition, cid)}
        removed[family] = len(left - right)
        introduced[family] = len(right - left)
    return json.dumps(removed, sort_keys=True), json.dumps(introduced, sort_keys=True)


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    all_targets = list(OUT.values()) + [base.with_suffix(ext) for base in PLOTS.values() for ext in (".png", ".pdf")]
    existing = [path for path in all_targets if path.exists()]
    if existing:
        raise FileExistsError("Additive targets already exist: " + ", ".join(map(str, existing)))

    source_checks: list[dict[str, Any]] = []
    for path_str, expected in EXPECTED.items():
        path = ROOT / path_str
        actual = sha256(path)
        source_checks.append({"path": path_str, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
        if actual != expected:
            raise RuntimeError(f"Authoritative source hash mismatch: {path_str}")

    with CROSS_MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    lora_records = [record for record in manifest["runs"] if record["role"] == "lora_v2"]
    if len(lora_records) != 24 or len({(record["model"], record["condition"]) for record in lora_records}) != 24:
        raise RuntimeError("Expected exactly 24 unique LoRA-v2 model-condition records")
    record_by = {(record["model"], record["condition"]): record for record in lora_records}
    if set(record_by) != {(model, condition) for model in MODEL_ORDER for condition in CONDITIONS}:
        raise RuntimeError("LoRA-v2 run matrix is not the required 3 x 8 matrix")

    requested_testset = ROOT / manifest["testset"]["path"]
    resolved_testset = requested_testset
    if not requested_testset.is_file() or sha256(requested_testset) != manifest["testset"]["sha256"]:
        candidate = ROOT / "data/testcases_spider_dev_full.jsonl"
        if not candidate.is_file() or sha256(candidate) != manifest["testset"]["sha256"]:
            raise RuntimeError("Frozen 1,032-row Spider Dev testset cannot be resolved by hash")
        resolved_testset = candidate
    testcase_rows = read_jsonl(resolved_testset)
    if len(testcase_rows) != N:
        raise RuntimeError("Resolved testset does not contain 1,032 cases")
    testcase_by_id = {row["id"]: row for row in testcase_rows}

    rows_by: dict[tuple[str, str], list[dict[str, str]]] = {}
    row_by: dict[tuple[str, str, str], dict[str, str]] = {}
    traces_by: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    metric_reproduction: list[dict[str, Any]] = []
    artifact_hashes: list[dict[str, Any]] = []
    common_order: list[str] | None = None

    for model in MODEL_ORDER:
        provenance = manifest["model_and_adapter_provenance"][model]
        for condition in CONDITIONS:
            record = record_by[(model, condition)]
            for kind in ("config", "csv", "metadata", "trace", "log"):
                path_key, hash_key = f"{kind}_path", f"{kind}_sha256"
                if record.get(path_key) is None:
                    continue
                path = ROOT / record[path_key]
                actual = sha256(path)
                expected = record[hash_key]
                artifact_hashes.append({"model": model, "condition": condition, "kind": kind, "path": record[path_key], "sha256": actual, "expected_sha256": expected, "match": actual == expected})
                if actual != expected:
                    raise RuntimeError(f"Hash mismatch: {record[path_key]}")
            rows = read_csv(ROOT / record["csv_path"])
            ids = [row["id"] for row in rows]
            if len(rows) != N or len(set(ids)) != N or case_order_hash(rows) != record["case_ids_sha256"]:
                raise RuntimeError(f"Case integrity failed: {model}/{condition}")
            if common_order is None:
                common_order = ids
            if ids != common_order:
                raise RuntimeError(f"Case order differs: {model}/{condition}")
            if ids != [row["id"] for row in testcase_rows]:
                raise RuntimeError(f"Run/testset case order differs: {model}/{condition}")
            if not all(record["technical_checks"].values()):
                raise RuntimeError(f"Frozen technical check failed: {model}/{condition}")
            config = json.loads((ROOT / record["config_path"]).read_text(encoding="utf-8"))
            metadata = json.loads((ROOT / record["metadata_path"]).read_text(encoding="utf-8"))
            if metadata["run_model_id"] != provenance["model_id"] or metadata["run_adapter"] in {"", "base", None}:
                raise RuntimeError(f"Model/adapter provenance failed: {model}/{condition}")
            if config.get("max_new_tokens") != 256 or metadata.get("run_max_new_tokens") != 256:
                raise RuntimeError(f"Generation budget mismatch: {model}/{condition}")
            if metadata.get("run_generation_batch_size") != 1 or metadata.get("run_extractor_mode") != "sql_first_statement_only":
                raise RuntimeError(f"Evaluation method mismatch: {model}/{condition}")
            values = metric_values(rows)
            for metric in ("correct", "ema", "executable", "esr", "string_exact", "normalized_exact", "char_accuracy", "token_accuracy"):
                reported = record["metrics"][metric]
                reproduced = values[metric]
                delta = abs(float(reported) - float(reproduced))
                metric_reproduction.append({"model": model, "condition": condition, "metric": metric, "reported": reported, "reproduced": reproduced, "absolute_difference": delta, "status": "PASS" if delta <= 1e-12 else "FAIL"})
                if delta > 1e-12:
                    raise RuntimeError(f"Metric reproduction mismatch: {model}/{condition}/{metric}")
            rows_by[(model, condition)] = rows
            for row in rows:
                row_by[(model, condition, row["id"])] = row
            if record.get("trace_path"):
                trace_rows = read_jsonl(ROOT / record["trace_path"])
                if len(trace_rows) != N or [row["id"] for row in trace_rows] != ids:
                    raise RuntimeError(f"Trace integrity failed: {model}/{condition}")
                if any(row.get("leakage_status") != "pass" for row in trace_rows):
                    raise RuntimeError(f"Trace leakage failed: {model}/{condition}")
                traces_by[(model, condition)] = {row["id"]: row for row in trace_rows}

    assert common_order is not None
    if hashlib.sha256("\n".join(common_order).encode()).hexdigest() != manifest["testset"]["case_order_sha256"]:
        raise RuntimeError("Common case-order hash differs from authoritative manifest")

    # Cross-role retrieval identity is a frozen prerequisite and is rechecked for LoRA roles here.
    for condition in CONDITIONS[1:]:
        reference = traces_by[(MODEL_ORDER[0], condition)]
        for model in MODEL_ORDER[1:]:
            candidate = traces_by[(model, condition)]
            for cid in common_order:
                if reference[cid].get("retrieved_ids") != candidate[cid].get("retrieved_ids"):
                    raise RuntimeError(f"Retrieval demo identity failed: {model}/{condition}/{cid}")
                if reference[cid].get("retrieved_scores") != candidate[cid].get("retrieved_scores"):
                    raise RuntimeError(f"Retrieval score identity failed: {model}/{condition}/{cid}")

    # Apply the frozen deterministic multi-label taxonomy to every incorrect prediction.
    sqlite_cache: dict[Any, Any] = {}
    feature_cache: dict[str, dict[str, Any]] = {}
    labels: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    label_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for condition in CONDITIONS:
            record = record_by[(model, condition)]
            for row in rows_by[(model, condition)]:
                gold = row["gold_sql"]
                pred = row["pred_sql"]
                if gold not in feature_cache:
                    feature_cache[gold] = ZE.sql_features(gold)
                if pred not in feature_cache:
                    feature_cache[pred] = ZE.sql_features(pred)
                found = ZE.classify(row, feature_cache[gold], feature_cache[pred], sqlite_cache)
                labels[(model, condition, row["id"])] = found
                for label, detail in sorted(found.items()):
                    broad = LABEL_BROAD.get(label, "F_UNCLEAR_HEURISTIC")
                    label_rows.append({
                        "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                        "condition_label": COND_LABELS[condition], "run_id": record["run_id"],
                        "case_id": row["id"], "db_id": row["db_id"], "question": row["question"],
                        "error_label": label, "broad_family": broad, "broad_family_label": BROAD_LABEL[broad],
                        **detail, "execution_success": truth(row["pred_ok"]), "execution_match": False,
                        "taxonomy_version": ZE.TAXONOMY_VERSION, "parser": ZE.PARSER_NAME,
                        "parser_version": ZE.PARSER_VERSION, "analysis_class": "EXPLORATIVE MULTI-LABEL ERROR DIAGNOSTIC",
                    })

    # Per-run result and residual-error profiles.
    profiles: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    esr_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for condition in CONDITIONS:
            rows = rows_by[(model, condition)]
            record = record_by[(model, condition)]
            values = metric_values(rows)
            wrong = [row for row in rows if not truth(row["exec_match"])]
            completion = np.array([int(float(row["completion_tokens"] or 0)) for row in rows])
            family_unique = Counter()
            family_assignments = Counter()
            evidence_assignments = Counter()
            for row in wrong:
                details = labels[(model, condition, row["id"])]
                families = {LABEL_BROAD.get(label, "F_UNCLEAR_HEURISTIC") for label in details}
                family_unique.update(families)
                family_assignments.update(LABEL_BROAD.get(label, "F_UNCLEAR_HEURISTIC") for label in details)
                evidence_assignments.update(item["evidence_level"] for item in details.values())
            profile: dict[str, Any] = {
                "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                "condition_label": COND_LABELS[condition], "run_id": record["run_id"],
                "correct": values["correct"], "incorrect": N - int(values["correct"]),
                "execution_success": values["executable"], "execution_failure": N - int(values["executable"]),
                "executable_but_wrong": int(values["executable"]) - int(values["correct"]),
                "ema": values["ema"], "esr": values["esr"], "string_exact": values["string_exact"],
                "normalized_exact": values["normalized_exact"], "char_accuracy": values["char_accuracy"],
                "token_accuracy": values["token_accuracy"],
                "empty_sql": sum(not row["pred_sql"].strip() for row in rows),
                "completion_limit": sum(int(float(row["completion_tokens"] or 0)) == int(float(row["run_max_new_tokens"] or 256)) for row in rows),
                "repetitive_generation": sum(ZE.repeated_generation(row["raw_output"]) for row in rows),
                "E1_label_assignments": evidence_assignments["E1"], "E2_label_assignments": evidence_assignments["E2"],
                "E3_label_assignments": evidence_assignments["E3"], "E4_label_assignments": evidence_assignments["E4"],
                "e1_e2_label_assignments": evidence_assignments["E1"] + evidence_assignments["E2"],
                "e3_e4_label_assignments": evidence_assignments["E3"] + evidence_assignments["E4"],
                "prompt_tokens_mean": record["metrics"]["prompt_tokens_mean"], "prompt_tokens_max": record["metrics"]["prompt_tokens_max"],
                "prompt_truncations": record["metrics"]["prompt_truncations"],
            }
            for family in BROAD:
                profile[f"{family}_unique_cases"] = family_unique[family]
                profile[f"{family}_label_assignments"] = family_assignments[family]
                profile[f"{family}_rate_per_1032"] = family_unique[family] / N
                profile[f"{family}_share_of_incorrect"] = family_unique[family] / len(wrong) if wrong else 0.0
            profiles.append(profile)

            out = {
                "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                "completion_tokens_mean": float(completion.mean()), "completion_tokens_median": float(np.median(completion)),
                "completion_tokens_p95": float(np.quantile(completion, 0.95)), "completion_tokens_max": int(completion.max()),
                "completion_limit_count": profile["completion_limit"],
                "empty_raw_count": sum(not row["raw_output"].strip() for row in rows),
                "empty_sql_count": profile["empty_sql"], "repetition_count": profile["repetitive_generation"],
                "multiple_statement_count": sum(top_level_statement_count(row["pred_sql"]) > 1 for row in rows),
                "extraction_failure_count": sum(bool(row["raw_output"].strip()) and not bool(row["pred_sql"].strip()) for row in rows),
                "missing_semicolon_count": sum(bool(row["pred_sql"].strip()) and not row["pred_sql"].rstrip().endswith(";") for row in rows),
                "markdown_or_explanation_count": sum("```" in row["raw_output"] or "here is" in row["raw_output"].lower() for row in rows),
                "think_marker_count": sum("<think>" in row["raw_output"].lower() or "</think>" in row["raw_output"].lower() for row in rows),
                "non_executable": profile["execution_failure"],
            }
            output_rows.append(out)
            esr_rows.append({
                "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                "esr": values["esr"], "ema": values["ema"], "esr_minus_ema": float(values["esr"]) - float(values["ema"]),
                "executable_but_wrong": profile["executable_but_wrong"],
                "normalized_em_false_ema_true": sum(not truth(row["normalized_exact"]) and truth(row["exec_match"]) for row in rows),
                "normalized_em_true_ema_false": sum(truth(row["normalized_exact"]) and not truth(row["exec_match"]) for row in rows),
                "different_sql_but_execution_match": sum(truth(row["exec_match"]) and ZE.norm_sql(row["pred_sql"]) != ZE.norm_sql(row["gold_sql"]) for row in rows),
            })

    # Zero-shot-centered paired outcome transitions and exploratory inference.
    transitions: list[dict[str, Any]] = []
    for mi, model in enumerate(MODEL_ORDER):
        model_rows: list[dict[str, Any]] = []
        zero = rows_by[(model, "zero_shot")]
        z = np.array([truth(row["exec_match"]) for row in zero], dtype=bool)
        for ci, condition in enumerate(CONDITIONS[1:]):
            current = rows_by[(model, condition)]
            c = np.array([truth(row["exec_match"]) for row in current], dtype=bool)
            benefit = int((~z & c).sum()); harm = int((z & ~c).sum())
            lo, hi = bootstrap_ci(c.astype(float) - z.astype(float), BOOTSTRAP_SEED + mi * 100 + ci)
            item = {
                "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                "condition_label": COND_LABELS[condition], "benefit": benefit, "harm": harm, "net": benefit - harm,
                "persistent_wrong": int((~z & ~c).sum()), "stable_correct": int((z & c).sum()),
                "ema_zero": float(z.mean()), "ema_condition": float(c.mean()), "ema_delta_pp": float((c.mean() - z.mean()) * 100),
                "mcnemar_p": exact_mcnemar(benefit, harm), "bootstrap_95_ci_low_pp": lo,
                "bootstrap_95_ci_high_pp": hi, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED, "inference_class": "EXPLORATIVE LORA CROSS-PROMPT ERROR INFERENCE",
            }
            if item["net"] != int(sum(c) - sum(z)):
                raise RuntimeError(f"Transition/EMA count identity failed: {model}/{condition}")
            model_rows.append(item)
        holm_adjust(model_rows)
        transitions.extend(model_rows)

    # Broad-family and individual-label transitions, with evidence bands.
    family_transitions: list[dict[str, Any]] = []
    all_labels = sorted(set().union(*(set(details) for details in labels.values())))
    for model in MODEL_ORDER:
        for condition in CONDITIONS[1:]:
            for granularity, names in (("broad_family", list(BROAD)), ("error_label", all_labels)):
                for name in names:
                    def relevant(cid: str, cond: str) -> dict[str, dict[str, str]]:
                        details = labels[(model, cond, cid)]
                        if granularity == "error_label":
                            return {key: value for key, value in details.items() if key == name}
                        return {key: value for key, value in details.items() if LABEL_BROAD.get(key, "F_UNCLEAR_HEURISTIC") == name}
                    zero_set = {cid for cid in common_order if relevant(cid, "zero_shot")}
                    cond_set = {cid for cid in common_order if relevant(cid, condition)}
                    if not zero_set and not cond_set:
                        continue
                    row = {
                        "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
                        "granularity": granularity, "error_group": name,
                        "group_label": BROAD_LABEL.get(name, name), "zero_shot_count": len(zero_set),
                        "condition_count": len(cond_set), "repaired": len(zero_set - cond_set),
                        "introduced": len(cond_set - zero_set), "persistent": len(zero_set & cond_set),
                        "net_condition_minus_zero": len(cond_set) - len(zero_set),
                    }
                    for band, levels in (("e1_e2", {"E1", "E2"}), ("e3_e4", {"E3", "E4"})):
                        for cond_name, prefix in (("zero_shot", "zero"), (condition, "condition")):
                            row[f"{band}_{prefix}_count"] = sum(any(item["evidence_level"] in levels for item in relevant(cid, cond_name).values()) for cid in common_order)
                    family_transitions.append(row)

    # Case-level robustness and pairwise incorrect-set overlap.
    robustness_rows: list[dict[str, Any]] = []
    robustness_summary: dict[str, Counter[str]] = {}
    overlap_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        summary = Counter()
        for cid in common_order:
            run_rows = [row_by[(model, condition, cid)] for condition in CONDITIONS]
            correct = sum(truth(row["exec_match"]) for row in run_rows)
            status = "stable_unsolved_0_of_8" if correct == 0 else "stable_solved_8_of_8" if correct == 8 else "prompt_sensitive_1_to_7_of_8"
            summary[status] += 1
            robustness_rows.append({
                "model_key": model, "model_line": MODEL_LABELS[model], "case_id": cid,
                "db_id": run_rows[0]["db_id"], "question": run_rows[0]["question"],
                "number_correct_out_of_8": correct, "number_executable_out_of_8": sum(truth(row["pred_ok"]) for row in run_rows),
                "number_execution_failures_out_of_8": sum(not truth(row["pred_ok"]) for row in run_rows),
                "number_empty_sql_out_of_8": sum(not row["pred_sql"].strip() for row in run_rows),
                "number_completion_limits_out_of_8": sum(int(float(row["completion_tokens"] or 0)) == 256 for row in run_rows),
                "number_unique_pred_sql": len({row["pred_sql"] for row in run_rows}),
                "number_unique_normalized_sql": len({ZE.norm_sql(row["pred_sql"]) for row in run_rows}),
                "robustness_class": status,
            })
        robustness_summary[model] = summary
        wrong_sets = {condition: {cid for cid in common_order if not truth(row_by[(model, condition, cid)]["exec_match"])} for condition in CONDITIONS}
        all_wrong = set.intersection(*(wrong_sets[condition] for condition in CONDITIONS))
        only_one_wrong = sum(sum(cid in wrong_sets[condition] for condition in CONDITIONS) == 1 for cid in common_order)
        only_one_correct = sum(sum(cid not in wrong_sets[condition] for condition in CONDITIONS) == 1 for cid in common_order)
        for left in CONDITIONS:
            for right in CONDITIONS:
                inter = wrong_sets[left] & wrong_sets[right]
                union = wrong_sets[left] | wrong_sets[right]
                overlap_rows.append({
                    "row_type": "pairwise_jaccard", "model_key": model, "model_line": MODEL_LABELS[model],
                    "condition_a": left, "condition_b": right, "intersection": len(inter), "union": len(union),
                    "jaccard": len(inter) / len(union) if union else 1.0,
                    "only_a_incorrect": len(wrong_sets[left] - wrong_sets[right]),
                    "only_b_incorrect": len(wrong_sets[right] - wrong_sets[left]),
                    "all_eight_incorrect": len(all_wrong), "only_one_condition_incorrect": only_one_wrong,
                    "only_one_condition_correct": only_one_correct,
                })

    # Empirical cross-run failure frequency; this is explicitly not Spider Difficulty.
    failure_rows: list[dict[str, Any]] = []
    for cid in common_order:
        failures = {(model, condition) for model in MODEL_ORDER for condition in CONDITIONS if not truth(row_by[(model, condition, cid)]["exec_match"])}
        count = len(failures)
        group = "24_of_24" if count == 24 else "20_to_23" if count >= 20 else "12_to_19" if count >= 12 else "1_to_11" if count else "0_of_24"
        family_counts = Counter()
        for model, condition in failures:
            family_counts.update(broad_sets_for(labels, model, condition, cid))
        tc = testcase_by_id[cid]
        failure_rows.append({
            "case_id": cid, "db_id": tc["db_id"], "question": tc["question"],
            "cross_run_empirical_failure_frequency": count, "frequency_group": group,
            "qwen2b_failures_out_of_8": sum(("qwen2b", condition) in failures for condition in CONDITIONS),
            "llama3b_failures_out_of_8": sum(("llama3b", condition) in failures for condition in CONDITIONS),
            "qwen9b_failures_out_of_8": sum(("qwen9b", condition) in failures for condition in CONDITIONS),
            "dominant_broad_families": ";".join(family for family, _ in family_counts.most_common()),
            "broad_family_counts_json": json.dumps(family_counts, sort_keys=True),
            "official_spider_difficulty": "NOT AVAILABLE",
        })

    # Gate partitions and exact reference-output identity.
    gate_specs = {
        "top1_gate070": ("top1", 634, 398, 0.70),
        "top1_gate085": ("top1", 57, 975, 0.85),
        "structure_gate070": ("structure", 613, 419, 0.70),
        "structure_gate085": ("structure", 57, 975, 0.85),
    }
    gate_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for gate_condition, (ungated, expected_few, expected_zero, threshold) in gate_specs.items():
            trace = traces_by[(model, gate_condition)]
            # Gate traces retain the selected pre-gate candidate in
            # num_fewshot_examples/retrieved_ids.  gate_decision is the
            # authoritative record of whether that candidate entered the prompt.
            accepted = [cid for cid in common_order if trace[cid].get("gate_decision") == "fewshot"]
            fallback = [cid for cid in common_order if trace[cid].get("gate_decision") == "zero_shot"]
            if (len(accepted), len(fallback)) != (expected_few, expected_zero):
                raise RuntimeError(f"Gate partition mismatch: {model}/{gate_condition}")
            if any(abs(float(trace[cid].get("gate_threshold")) - threshold) > 1e-12 for cid in common_order):
                raise RuntimeError(f"Gate threshold mismatch: {model}/{gate_condition}")
            accepted_raw = sum(row_by[(model, gate_condition, cid)]["raw_output"] == row_by[(model, ungated, cid)]["raw_output"] for cid in accepted)
            accepted_sql = sum(row_by[(model, gate_condition, cid)]["pred_sql"] == row_by[(model, ungated, cid)]["pred_sql"] for cid in accepted)
            fallback_raw = sum(row_by[(model, gate_condition, cid)]["raw_output"] == row_by[(model, "zero_shot", cid)]["raw_output"] for cid in fallback)
            fallback_sql = sum(row_by[(model, gate_condition, cid)]["pred_sql"] == row_by[(model, "zero_shot", cid)]["pred_sql"] for cid in fallback)
            if accepted_raw != len(accepted) or accepted_sql != len(accepted) or fallback_raw != len(fallback) or fallback_sql != len(fallback):
                raise RuntimeError(f"Gate reference-output identity failed: {model}/{gate_condition}")
            for partition, ids, reference, raw_identity, sql_identity in (
                ("fewshot_accepted", accepted, ungated, accepted_raw, accepted_sql),
                ("zero_shot_fallback", fallback, "zero_shot", fallback_raw, fallback_sql),
            ):
                selected = [row_by[(model, gate_condition, cid)] for cid in ids]
                ref = [row_by[(model, "zero_shot", cid)] for cid in ids]
                correct = sum(truth(row["exec_match"]) for row in selected)
                executable = sum(truth(row["pred_ok"]) for row in selected)
                zc = np.array([truth(row["exec_match"]) for row in ref], dtype=bool)
                gc = np.array([truth(row["exec_match"]) for row in selected], dtype=bool)
                families = Counter()
                for cid in ids:
                    families.update(broad_sets_for(labels, model, gate_condition, cid))
                gate_rows.append({
                    "model_key": model, "model_line": MODEL_LABELS[model], "gate_condition": gate_condition,
                    "ungated_reference": ungated, "threshold": threshold, "partition": partition,
                    "cases": len(ids), "correct": correct, "ema": correct / len(ids) if ids else 0.0,
                    "non_executable": len(ids) - executable, "executable_but_wrong": executable - correct,
                    "prompt_benefit_vs_zero": int((~zc & gc).sum()), "prompt_harm_vs_zero": int((zc & ~gc).sum()),
                    "raw_output_reference_identity": raw_identity, "pred_sql_reference_identity": sql_identity,
                    "reference_identity_expected": len(ids), "reference_identity_pass": raw_identity == len(ids) and sql_identity == len(ids),
                    "original_bge_score_gate": True, "broad_family_unique_case_counts_json": json.dumps(families, sort_keys=True),
                })

    # Paired ungated/gated comparisons.
    ungated_gated_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for left, right in (("top1", "top1_gate070"), ("top1", "top1_gate085"), ("structure", "structure_gate070"), ("structure", "structure_gate085")):
            counts = transition_counts(rows_by[(model, left)], rows_by[(model, right)])
            removed, introduced = family_delta_json(labels, model, left, right, common_order)
            ungated_gated_rows.append({
                "model_key": model, "model_line": MODEL_LABELS[model], "ungated_condition": left,
                "gated_condition": right, "gate_repairs_ungated": counts["right_repairs_left"],
                "gate_harms_ungated": counts["right_harms_left"],
                "net_gate_minus_ungated": counts["right_repairs_left"] - counts["right_harms_left"],
                "both_wrong": counts["both_wrong"], "both_correct": counts["both_correct"],
                "error_families_removed_json": removed, "error_families_introduced_json": introduced,
                "interpretation": "exploratory paired gate comparison; not an optimality test",
            })

    # Top-1 versus Structure, including corresponding gate pairs.
    top1_structure_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for top_condition, structure_condition, comparison in (
            ("top1", "structure", "ungated"),
            ("top1_gate070", "structure_gate070", "gate070_exploratory"),
            ("top1_gate085", "structure_gate085", "gate085_exploratory"),
        ):
            top_rows = rows_by[(model, top_condition)]
            structure_rows = rows_by[(model, structure_condition)]
            counts = transition_counts(top_rows, structure_rows)
            top_trace = traces_by[(model, top_condition)]
            structure_trace = traces_by[(model, structure_condition)]
            same_demo = sum(top_trace[cid].get("retrieved_ids") == structure_trace[cid].get("retrieved_ids") for cid in common_order)
            def means(condition: str) -> tuple[float, float, float, float]:
                features = [feature_cache[row_by[(model, condition, cid)]["pred_sql"]] for cid in common_order]
                return (
                    float(np.mean([len(item["tables"]) for item in features])),
                    float(np.mean([item["joins"] for item in features])),
                    float(np.mean([sum(item["aggs"].values()) for item in features])),
                    float(np.mean([item["subqueries"] for item in features])),
                )
            top_means, structure_means = means(top_condition), means(structure_condition)
            removed, introduced = family_delta_json(labels, model, top_condition, structure_condition, common_order)
            top1_structure_rows.append({
                "model_key": model, "model_line": MODEL_LABELS[model], "comparison": comparison,
                "top1_condition": top_condition, "structure_condition": structure_condition,
                "top1_only_correct": counts["right_harms_left"], "structure_only_correct": counts["right_repairs_left"],
                "both_correct": counts["both_correct"], "both_wrong": counts["both_wrong"],
                "same_demo_id": same_demo, "different_demo_id": N - same_demo,
                "top1_mean_table_count": top_means[0], "structure_mean_table_count": structure_means[0],
                "top1_mean_join_count": top_means[1], "structure_mean_join_count": structure_means[1],
                "top1_mean_aggregation_count": top_means[2], "structure_mean_aggregation_count": structure_means[2],
                "top1_mean_subquery_count": top_means[3], "structure_mean_subquery_count": structure_means[3],
                "structure_removed_families_json": removed, "structure_introduced_families_json": introduced,
                "causal_superiority_claim": False,
            })

    # Static compared with Zero Shot and the two ungated dynamic methods.
    static_resource = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
    static_rows_resource = read_jsonl(static_resource)
    if len(static_rows_resource) != 1 or static_rows_resource[0].get("id") != "SPIDER_TRAIN_001657":
        raise RuntimeError("Static resource identity failed")
    demo_sql = static_rows_resource[0].get("gold_sql", "")
    demo_tokens = set(ZE.tokenize(demo_sql.lower()))
    static_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        static_trace = traces_by[(model, "static_seed42")]
        if any(trace.get("retrieved_ids") != ["SPIDER_TRAIN_001657"] for trace in static_trace.values()):
            raise RuntimeError(f"Static demo identity failed: {model}")
        for reference in ("zero_shot", "top1", "structure"):
            counts = transition_counts(rows_by[(model, reference)], rows_by[(model, "static_seed42")])
            removed, introduced = family_delta_json(labels, model, reference, "static_seed42", common_order)
            similarities = []
            harmed_similarities = []
            for cid in common_order:
                pred_tokens = set(ZE.tokenize(row_by[(model, "static_seed42", cid)]["pred_sql"].lower()))
                sim = len(pred_tokens & demo_tokens) / len(pred_tokens | demo_tokens) if pred_tokens | demo_tokens else 0.0
                similarities.append(sim)
                if truth(row_by[(model, reference, cid)]["exec_match"]) and not truth(row_by[(model, "static_seed42", cid)]["exec_match"]):
                    harmed_similarities.append(sim)
            static_rows.append({
                "model_key": model, "model_line": MODEL_LABELS[model], "reference_condition": reference,
                "static_demo_id": "SPIDER_TRAIN_001657", "static_demo_same_1032_of_1032": True,
                "static_repairs_reference": counts["right_repairs_left"], "static_harms_reference": counts["right_harms_left"],
                "net_static_minus_reference": counts["right_repairs_left"] - counts["right_harms_left"],
                "both_wrong": counts["both_wrong"], "both_correct": counts["both_correct"],
                "error_families_removed_json": removed, "error_families_introduced_json": introduced,
                "mean_prediction_demo_token_jaccard": float(np.mean(similarities)),
                "mean_harmed_prediction_demo_token_jaccard": float(np.mean(harmed_similarities)) if harmed_similarities else 0.0,
                "demo_similarity_evidence": "E3 heuristic; no causal interpretation",
            })

    # Qualitative examples: deterministic, balanced, and capped at 24.
    examples: list[dict[str, Any]] = []
    candidate_groups: dict[str, list[dict[str, Any]]] = {key: [] for key in ("prompt_benefit", "prompt_harm", "stable_unresolved", "alternative_valid")}
    for model in MODEL_ORDER:
        for condition in CONDITIONS[1:]:
            for cid in common_order:
                zero = row_by[(model, "zero_shot", cid)]
                current = row_by[(model, condition, cid)]
                zc, cc = truth(zero["exec_match"]), truth(current["exec_match"])
                group = "prompt_benefit" if not zc and cc else "prompt_harm" if zc and not cc else None
                if group:
                    failing_condition = "zero_shot" if group == "prompt_benefit" else condition
                    details = labels[(model, failing_condition, cid)]
                    concrete = [label for label, item in details.items() if item["evidence_level"] in {"E1", "E2"} and LABEL_BROAD.get(label) != "F_UNCLEAR_HEURISTIC"]
                    if concrete:
                        candidate_groups[group].append({"model": model, "condition": condition, "cid": cid, "labels": concrete})
                if zc and cc and zero["pred_sql"] != current["pred_sql"] and ZE.norm_sql(zero["pred_sql"]) != ZE.norm_sql(current["pred_sql"]):
                    candidate_groups["alternative_valid"].append({"model": model, "condition": condition, "cid": cid, "labels": ["ALTERNATIVE_VALID_FORMULATION"]})
        for cid in common_order:
            if all(not truth(row_by[(model, condition, cid)]["exec_match"]) for condition in CONDITIONS):
                details = labels[(model, "zero_shot", cid)]
                concrete = [label for label, item in details.items() if item["evidence_level"] in {"E1", "E2"} and LABEL_BROAD.get(label) != "F_UNCLEAR_HEURISTIC"]
                if concrete:
                    candidate_groups["stable_unresolved"].append({"model": model, "condition": "structure", "cid": cid, "labels": concrete})

    limits = {"prompt_benefit": 8, "prompt_harm": 8, "stable_unresolved": 4, "alternative_valid": 4}
    for group, limit in limits.items():
        candidates = sorted(candidate_groups[group], key=lambda item: (MODEL_ORDER.index(item["model"]), CONDITIONS.index(item["condition"]), item["cid"]))
        selected: list[dict[str, Any]] = []
        # Round-robin by model to avoid letting one line dominate the examples.
        for offset in range(limit * 4):
            model = MODEL_ORDER[offset % len(MODEL_ORDER)]
            candidate = next((item for item in candidates if item["model"] == model and item not in selected), None)
            if candidate is not None:
                selected.append(candidate)
            if len(selected) == limit:
                break
        for item in selected:
            model, condition, cid = item["model"], item["condition"], item["cid"]
            zero = row_by[(model, "zero_shot", cid)]
            current = row_by[(model, condition, cid)]
            tc = testcase_by_id[cid]
            interpretation = {
                "prompt_benefit": "The prompted condition execution-matches where Zero Shot does not.",
                "prompt_harm": "Zero Shot execution-matches while the prompted condition does not.",
                "stable_unresolved": "The case remains incorrect in all eight prompt conditions.",
                "alternative_valid": "Both outputs execution-match despite structurally different SQL.",
            }[group]
            examples.append({
                "example_type": group, "model_key": model, "model_line": MODEL_LABELS[model],
                "condition": condition, "case_id": cid, "db_id": tc["db_id"], "question": tc["question"],
                "schema_excerpt": tc.get("schema_prompt", "")[:1800], "gold_sql": tc["gold_sql"],
                "zero_shot_sql": zero["pred_sql"], "condition_sql": current["pred_sql"],
                "execution_status": f"zero_success={truth(zero['pred_ok'])};zero_match={truth(zero['exec_match'])};condition_success={truth(current['pred_ok'])};condition_match={truth(current['exec_match'])}",
                "outcome_transition": group, "explorative_error_labels": ";".join(item["labels"]),
                "short_interpretation": interpretation,
                "methodological_note": "Illustrative deterministic example; no prevalence or causal claim.",
            })

    if Counter(row["example_type"] for row in examples) != Counter(limits):
        raise RuntimeError("Could not construct the required balanced 24-example set")

    gap_rows = build_gap_rows()

    # Scientific figures. All are additive PNG/PDF pairs at 300 dpi.
    colors = {"qwen2b": "#276FBF", "llama3b": "#F08A24", "qwen9b": "#2E8B57"}
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        current = [next(row for row in profiles if row["model_key"] == model and row["condition"] == condition) for condition in CONDITIONS]
        ax.bar(x, [100 * row["ema"] for row in current], color=colors[model], alpha=0.86, label="EMA (%)")
        ax2 = ax.twinx(); ax2.plot(x, [row["incorrect"] for row in current], color="#333333", marker="o", label="Incorrect")
        ax.set_ylabel("EMA (%)"); ax2.set_ylabel("Incorrect cases"); ax.set_title(MODEL_LABELS[model]); ax.set_ylim(0, 100)
    axes[-1].set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], rotation=35, ha="right")
    fig.suptitle("Execution accuracy and residual errors across 24 LoRA-v2 runs")
    fig.text(0.01, 0.005, "Source: frozen 1,032-case Spider Dev run CSVs; Execution Match is authoritative.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); save_plot(PLOTS["results"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        current = [row for row in transitions if row["model_key"] == model]
        xx = np.arange(len(current)); width = 0.38
        ax.bar(xx - width / 2, [row["benefit"] for row in current], width, label="Benefit", color="#2E8B57")
        ax.bar(xx + width / 2, [row["harm"] for row in current], width, label="Harm", color="#B33A3A")
        ax.axhline(0, color="#333333", linewidth=0.7); ax.set_ylabel("Cases"); ax.set_title(MODEL_LABELS[model]); ax.legend()
    axes[-1].set_xticks(np.arange(7), [COND_LABELS[c] for c in CONDITIONS[1:]], rotation=35, ha="right")
    fig.suptitle("Prompt benefit and harm relative to LoRA Zero Shot")
    fig.text(0.01, 0.005, "Paired case transitions; descriptive/exploratory inference.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); save_plot(PLOTS["transitions"])

    family_order = list(BROAD)[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        matrix = np.array([[next(row for row in profiles if row["model_key"] == model and row["condition"] == condition)[f"{family}_unique_cases"] for condition in CONDITIONS] for family in family_order])
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(family_order)), [BROAD_LABEL[f] for f in family_order]); ax.set_title(MODEL_LABELS[model])
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]): ax.text(ix, iy, int(matrix[iy, ix]), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01, label="Unique incorrect cases")
    axes[-1].set_xticks(range(8), [COND_LABELS[c] for c in CONDITIONS], rotation=35, ha="right")
    fig.suptitle("Broad residual-error families across prompt conditions")
    fig.text(0.01, 0.005, "Multi-label counts can overlap; meta-family F is excluded from the technical ranking.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); save_plot(PLOTS["families"])

    fig, ax = plt.subplots(figsize=(10, 5.5)); xx = np.arange(3); width = 0.24
    classes = [("stable_unsolved_0_of_8", "0/8 correct", "#B33A3A"), ("prompt_sensitive_1_to_7_of_8", "1-7/8 correct", "#F0A202"), ("stable_solved_8_of_8", "8/8 correct", "#2E8B57")]
    for offset, (key, label, color) in enumerate(classes):
        ax.bar(xx + (offset - 1) * width, [robustness_summary[m][key] for m in MODEL_ORDER], width, label=label, color=color)
    ax.set_xticks(xx, [MODEL_LABELS[m] for m in MODEL_ORDER]); ax.set_ylabel("Cases"); ax.set_title("Prompt robustness across eight LoRA conditions"); ax.legend()
    fig.text(0.01, 0.005, "Source: paired Execution Match outcomes for all 1,032 cases per model line.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save_plot(PLOTS["robustness"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, model in zip(axes, MODEL_ORDER):
        matrix = np.zeros((8, 8))
        for row in overlap_rows:
            if row["model_key"] == model:
                matrix[CONDITIONS.index(row["condition_a"]), CONDITIONS.index(row["condition_b"])] = row["jaccard"]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(8), [COND_LABELS[c] for c in CONDITIONS], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(8), [COND_LABELS[c] for c in CONDITIONS], fontsize=7); ax.set_title(MODEL_LABELS[model])
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Jaccard")
    fig.suptitle("Jaccard overlap of incorrect case sets")
    fig.text(0.01, 0.005, "Same incorrect outcome does not imply the same SQL error.", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.28, top=0.88, wspace=0.35); save_plot(PLOTS["jaccard"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    gate_order = list(gate_specs)
    for ax, model in zip(axes, MODEL_ORDER):
        few = [next(row for row in gate_rows if row["model_key"] == model and row["gate_condition"] == gate and row["partition"] == "fewshot_accepted") for gate in gate_order]
        fallback = [next(row for row in gate_rows if row["model_key"] == model and row["gate_condition"] == gate and row["partition"] == "zero_shot_fallback") for gate in gate_order]
        xx = np.arange(4); width = 0.38
        ax.bar(xx - width / 2, [100 * row["ema"] for row in few], width, label="Accepted demo", color="#276FBF")
        ax.bar(xx + width / 2, [100 * row["ema"] for row in fallback], width, label="Zero-shot fallback", color="#9AA0A6")
        ax.set_ylim(0, 100); ax.set_ylabel("EMA (%)"); ax.set_title(MODEL_LABELS[model]); ax.legend()
    axes[-1].set_xticks(range(4), [COND_LABELS[c] for c in gate_order], rotation=30, ha="right")
    fig.suptitle("Gate partitions: accepted demonstrations versus fallback")
    fig.text(0.01, 0.005, "Fallback outputs are required to match the corresponding Zero-Shot output exactly.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); save_plot(PLOTS["gate"])

    fig, ax = plt.subplots(figsize=(13, 6)); xx = np.arange(24)
    ordered_esr = [next(row for row in esr_rows if row["model_key"] == model and row["condition"] == condition) for model in MODEL_ORDER for condition in CONDITIONS]
    ax.bar(xx, [100 * row["esr_minus_ema"] for row in ordered_esr], color=[colors[row["model_key"]] for row in ordered_esr])
    ax.set_xticks(xx, [f"{MODEL_LABELS[row['model_key']].split()[0]} {COND_LABELS[row['condition']]}" for row in ordered_esr], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("ESR - EMA (percentage points)"); ax.set_title("Executable but semantically wrong result gap")
    fig.text(0.01, 0.005, "EMA remains the primary metric; the gap counts executable but non-matching SQL.", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save_plot(PLOTS["gap"])

    fig, ax = plt.subplots(figsize=(10, 5.5)); freq_counter = Counter(row["cross_run_empirical_failure_frequency"] for row in failure_rows)
    ax.bar(range(25), [freq_counter[i] for i in range(25)], color="#5B5F97")
    ax.set_xlabel("Cross-run empirical failure frequency (0-24)"); ax.set_ylabel("Spider Dev cases"); ax.set_title("Empirical failure frequency across 24 LoRA-v2 runs")
    fig.text(0.01, 0.005, "This empirical frequency is not official Spider Difficulty.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save_plot(PLOTS["frequency"])

    fig, ax = plt.subplots(figsize=(10, 5.5)); status_order = ["ABGESCHLOSSEN", "ZWINGEND OFFEN", "EMPFOHLEN OFFEN", "OPTIONAL", "NICHT MEHR ERFORDERLICH"]
    status_counts = Counter(row["status"] for row in gap_rows)
    ax.bar(status_order, [status_counts[s] for s in status_order], color=["#2E8B57", "#B33A3A", "#F0A202", "#6C8EBF", "#8D8D8D"])
    ax.set_ylabel("Audit items"); ax.set_title("Project completion and remaining-gap status"); ax.tick_params(axis="x", rotation=25)
    fig.text(0.01, 0.005, "Open items concern reproducibility freeze, thesis integration, and formal submission checks, not new experiments.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save_plot(PLOTS["completion"])

    # Persist all machine-readable derived tables before rendering the narrative audit.
    write_csv_new(OUT["profiles"], profiles)
    write_csv_new(OUT["labels"], label_rows)
    write_csv_new(OUT["transitions"], transitions)
    write_csv_new(OUT["family_transitions"], family_transitions)
    write_csv_new(OUT["robustness"], robustness_rows)
    write_csv_new(OUT["overlap"], overlap_rows)
    write_csv_new(OUT["gate"], gate_rows)
    write_csv_new(OUT["ungated_gated"], ungated_gated_rows)
    write_csv_new(OUT["top1_structure"], top1_structure_rows)
    write_csv_new(OUT["static"], static_rows)
    write_csv_new(OUT["output"], output_rows)
    write_csv_new(OUT["esr_ema"], esr_rows)
    write_csv_new(OUT["failure"], failure_rows)
    write_csv_new(OUT["examples"], examples)
    write_csv_new(OUT["gap"], gap_rows)

    profile_by = {(row["model_key"], row["condition"]): row for row in profiles}
    transition_by = {(row["model_key"], row["condition"]): row for row in transitions}
    best = {model: max(CONDITIONS, key=lambda condition: profile_by[(model, condition)]["ema"]) for model in MODEL_ORDER}
    worst = {model: min(CONDITIONS, key=lambda condition: profile_by[(model, condition)]["ema"]) for model in MODEL_ORDER}
    dominant = {}
    for model in MODEL_ORDER:
        totals = Counter({family: sum(profile_by[(model, condition)][f"{family}_unique_cases"] for condition in CONDITIONS) for family in BROAD if family != "F_UNCLEAR_HEURISTIC"})
        dominant[model] = totals.most_common(3)
    largest_gap = max(esr_rows, key=lambda row: row["esr_minus_ema"])
    smallest_gap = min(esr_rows, key=lambda row: row["esr_minus_ema"])
    freq_counts = Counter(row["frequency_group"] for row in failure_rows)
    solved_24 = freq_counts["0_of_24"]
    failed_24 = freq_counts["24_of_24"]
    failed_20 = freq_counts["24_of_24"] + freq_counts["20_to_23"]

    supported = [
        "Execution Match is the authoritative correctness criterion; text metrics are complementary diagnostics.",
        f"All 24 LoRA-v2 runs are complete on the same {N:,} Spider Dev cases with identical case order.",
        "Prompt conditions change both correct-case totals and the identity of solved cases; equal or similar EMA does not imply case identity.",
        "Gate 0.85 mostly reproduces Zero Shot because only 57 of 1,032 cases accept a demonstration.",
        "Every gated fallback output exactly reproduces its model-line Zero-Shot output in the frozen runs.",
        "Static, Top-1, and Structure produce model-specific benefit/harm balances; no prompt strategy is universally superior.",
        f"Across all model lines, {failed_24} cases fail in all 24 LoRA runs and {solved_24} are solved in all 24 runs.",
        "Residual error families are multi-label diagnostics and their counts can overlap.",
        "Qwen 2B versus Qwen 9B is a B+ descriptive comparison; Qwen versus Llama remains class B.",
        "The experimental and analytical run program is complete; remaining mandatory work is reproducibility freeze and thesis completion.",
    ]
    unsupported = [
        "The automatic taxonomy has been human validated.", "Error families are disjoint causal explanations.",
        "A demonstration causally produced each observed prompt error.", "Gate 0.85 is an optimal threshold.",
        "Structure is generally superior or inferior to Top-1.", "Zero Shot is universally the best NL2SQL strategy.",
        "Qwen 9B differences are caused exclusively by model size.", "Llama differences are caused by architecture.",
        "Cross-run empirical failure frequency is official Spider Difficulty.", "Non-significant differences prove equality.",
    ]
    limitations = [
        "The taxonomy is deterministic and rule-based but not manually validated.",
        "No local SQL AST parser was available; the frozen clause fallback 1.0.0 was used.",
        "Multi-label families overlap and do not identify a unique causal primary error.",
        "SQLite execution equivalence on the available databases is not universal semantic equivalence.",
        "Spider Dev serves as the frozen evaluation set; repeated experimental design decisions can still induce indirect adaptation risk.",
        "Gate and prompt-transition tests are exploratory and Holm-adjusted separately per model line.",
        "Cross-model comparisons mix Qwen Base and Llama Instruct starting points and are descriptive class B/B+.",
        "The qualitative examples are illustrative rather than prevalence estimates.",
        "Empirical failure frequency is not official benchmark difficulty.",
        "The historical testset manifest path drift requires transparent hash-based resolution to the byte-identical 1,032-row copy.",
    ]

    if sum(row["completion_limit_count"] for row in output_rows if row["model_key"] in {"qwen2b", "qwen9b"}) != 0:
        raise RuntimeError("Known Qwen LoRA completion-limit invariant failed")
    llama_limits = [row for row in output_rows if row["model_key"] == "llama3b" and row["completion_limit_count"]]
    if len(llama_limits) != 1 or llama_limits[0]["condition"] != "static_seed42" or llama_limits[0]["completion_limit_count"] != 1:
        raise RuntimeError("Known isolated Llama Static completion-limit invariant failed")

    table_a = md_table(
        ["Modelllinie", "Bedingung", "EMA", "ESR", "falsch", "nicht ausführbar", "ausführbar falsch"],
        ([row["model_line"], row["condition_label"], pct(row["ema"]), pct(row["esr"]), row["incorrect"], row["execution_failure"], row["executable_but_wrong"]] for row in profiles),
    )
    table_b = md_table(
        ["Modelllinie", "Bedingung", "Nutzen", "Schaden", "Netto", "persistent falsch", "stabil korrekt"],
        ([row["model_line"], row["condition_label"], row["benefit"], row["harm"], row["net"], row["persistent_wrong"], row["stable_correct"]] for row in transitions),
    )
    table_c = md_table(
        ["Modelllinie", "Bedingung", "Output/Kontrolle", "Syntax/Ausführung", "Schema/Projektion", "Querylogik", "Ergebnisabweichung"],
        ([row["model_line"], row["condition_label"], *[row[f"{family}_unique_cases"] for family in list(BROAD)[:-1]]] for row in profiles),
    )
    table_d = md_table(
        ["Modelllinie", "0/8 korrekt", "1-7/8 korrekt", "8/8 korrekt"],
        ([MODEL_LABELS[m], robustness_summary[m]["stable_unsolved_0_of_8"], robustness_summary[m]["prompt_sensitive_1_to_7_of_8"], robustness_summary[m]["stable_solved_8_of_8"]] for m in MODEL_ORDER),
    )
    table_e = md_table(
        ["Modelllinie", "Gate", "Few Shot", "Fallback", "EMA akzeptiert", "EMA Fallback"],
        ([MODEL_LABELS[m], COND_LABELS[g], next(row["cases"] for row in gate_rows if row["model_key"] == m and row["gate_condition"] == g and row["partition"] == "fewshot_accepted"), next(row["cases"] for row in gate_rows if row["model_key"] == m and row["gate_condition"] == g and row["partition"] == "zero_shot_fallback"), pct(next(row["ema"] for row in gate_rows if row["model_key"] == m and row["gate_condition"] == g and row["partition"] == "fewshot_accepted")), pct(next(row["ema"] for row in gate_rows if row["model_key"] == m and row["gate_condition"] == g and row["partition"] == "zero_shot_fallback"))] for m in MODEL_ORDER for g in gate_specs),
    )
    table_f = md_table(
        ["Modelllinie", "Vergleich", "Gate repariert", "Gate schadet", "Netto"],
        ([row["model_line"], f"{row['ungated_condition']} -> {row['gated_condition']}", row["gate_repairs_ungated"], row["gate_harms_ungated"], row["net_gate_minus_ungated"]] for row in ungated_gated_rows),
    )
    table_g = md_table(
        ["Modelllinie", "Vergleich", "Top-1-only korrekt", "Structure-only korrekt", "beide korrekt", "beide falsch"],
        ([row["model_line"], row["comparison"], row["top1_only_correct"], row["structure_only_correct"], row["both_correct"], row["both_wrong"]] for row in top1_structure_rows),
    )
    table_h = md_table(
        ["Modelllinie", "Bedingung", "ESR", "EMA", "Lücke"],
        ([row["model_line"], COND_LABELS[row["condition"]], pct(row["esr"]), pct(row["ema"]), f"{100 * row['esr_minus_ema']:.2f} pp"] for row in esr_rows),
    )
    table_i = md_table(
        ["Failure-Frequency-Gruppe", "Fälle"],
        ([group, freq_counts[group]] for group in ("24_of_24", "20_to_23", "12_to_19", "1_to_11", "0_of_24")),
    )
    table_j = md_table(
        ["Bereich", "Punkt", "Status", "Priorität", "Nächster Schritt"],
        ([row["area"], row["requirement"], row["status"], row["priority"], row["concrete_next_step"]] for row in gap_rows),
    )
    thesis_tables = "\n\n".join([
        "# Thesis-ready Tabellen: vollständige 24-LoRA-Run-Fehleranalyse",
        "## Tabelle A: alle 24 LoRA-Runs\n\n" + table_a,
        "## Tabelle B: Zero-Shot-zentrierte Promptübergänge\n\n" + table_b,
        "## Tabelle C: LoRA-Restfehlerfamilien\n\n" + table_c + "\n\nMulti-Label-Zählungen können sich überlappen.",
        "## Tabelle D: Promptrobustheit\n\n" + table_d,
        "## Tabelle E: Gatepartitionen\n\n" + table_e,
        "## Tabelle F: ungated versus gated\n\n" + table_f,
        "## Tabelle G: Top-1 versus Structure\n\n" + table_g,
        "## Tabelle H: ESR-EMA-Lücke\n\n" + table_h,
        "## Tabelle I: empirische Cross-Run-Fehlerhäufigkeit\n\n" + table_i + "\n\nNicht als offizielle Spider Difficulty interpretieren.",
        "## Tabelle J: Projektabschluss-Gap-Matrix\n\n" + table_j,
    ])
    write_text_new(OUT["thesis_tables"], thesis_tables)

    rq_answers = {
        "LQ1": f"Inkorrekte Outputs zeigen überlappende technische, Schema-/Projektions-, Querylogik- und Ergebnisabweichungen. Die dominanten technischen Familien sind modell- und bedingungsspezifisch; Meta-Labels werden nicht als Top-Fehler gewertet.",
        "LQ2": "; ".join(f"{MODEL_LABELS[m]}: {robustness_summary[m]['stable_solved_8_of_8']} stabil gelöst, {robustness_summary[m]['stable_unsolved_0_of_8']} stabil ungelöst, {robustness_summary[m]['prompt_sensitive_1_to_7_of_8']} promptsensitiv" for m in MODEL_ORDER) + ".",
        "LQ3": "Alle sieben Promptbedingungen reparieren und beschädigen jeweils Fälle relativ zu Zero Shot; Nutzen und Schaden sowie Holm-7-Tests sind vollständig in der Übergangstabelle ausgewiesen.",
        "LQ4": "Zero Shot, Static, Top-1 und Structure besitzen unterschiedliche, überlappende Restfehlermengen und breite Multi-Label-Profile; daraus folgt keine kausale Promptwirkung.",
        "LQ5": "Gates filtern einen Teil der Demonstrationen und verändern damit die Nutzen-/Schadensbilanz. Die akzeptierten Fälle tragen den eigentlichen Demonstrationseffekt; Fallbacks sind exakte Zero-Shot-Reproduktionen.",
        "LQ6": "Ja, deskriptiv: Gate 0.85 akzeptiert nur 57/1.032 Demonstrationen und reproduziert bei 975/1.032 Fällen exakt Zero Shot.",
        "LQ7": f"{failed_24} Fälle bleiben in allen 24 LoRA-Runs falsch; die modelllinienbezogenen 0/8-Gruppen sind in der Robustheitstabelle ausgewiesen.",
        "LQ8": "Promptsensitive Fälle sind jene mit 1-7 korrekten Bedingungen; ihre Anzahl und SQL-Variabilität stehen fallweise in der Robustheitstabelle.",
        "LQ9": "Die Linien unterscheiden sich deskriptiv in EMA, Ausführbarkeit und Restfehlerprofilen. Qwen 2B versus Qwen 9B ist B+, Qwen versus Llama B; kausale Größen- oder Architekturfolgerungen sind unzulässig.",
        "LQ10": "Die 24-Run-Analyse zeigt zusätzlich Promptrobustheit, Fehler-Set-Überlappung, Gatepartitionen, Static/Structure-Unterschiede und promptübergreifend persistente Fälle, die eine reine Zero-Shot-Analyse nicht abbildet.",
    }
    rq_text = ["# Finale Forschungsfragen-Ergebnis-Matrix", "", "| Frage | Ergebnis | Evidenz | Einordnung |", "|---|---|---|---|"]
    for key, answer in rq_answers.items():
        rq_text.append(f"| {key} | {answer} | 24 eingefrorene LoRA-v2-Runs; abgeleitete Tabellen dieses Audits | explorativ/deskriptiv, EMA autoritativ |")
    write_text_new(OUT["rq"], "\n".join(rq_text))

    best_text = ", ".join(
        f"{MODEL_LABELS[model]} {COND_LABELS[best[model]]} ({pct(profile_by[(model, best[model])]['ema'])})"
        for model in MODEL_ORDER
    )
    thesis_text = [
        "# Thesis-ready Text: vollständige LoRA-Cross-Prompt-Fehleranalyse", "",
        "## Methode", "",
        f"Die Analyse umfasst 24 eingefrorene LoRA-v2-Evaluationsläufe ({len(MODEL_ORDER)} Modelllinien × {len(CONDITIONS)} Promptbedingungen) mit jeweils {N:,} Spider-Dev-Fällen, insgesamt 24.768 Vorhersagen. Execution Match wurde als autoritatives Korrektheitskriterium verwendet. Inkorrekte Vorhersagen wurden explorativ mit der deterministischen, regelbasierten Multi-Label-Taxonomie `project-local-sqlite-clause-fallback` Version 1.0.0 klassifiziert. Ein lokaler SQL-AST-Parser und eine vollständige menschliche Validierung lagen nicht vor. E1/E2-Aussagen beruhen auf deterministischen beziehungsweise hochsicheren Regeln; E3/E4 bleiben heuristisch oder prüfbedürftig.", "",
        "## Ergebnisse", "",
        f"Über alle Promptbedingungen waren die jeweils höchsten EMA-Werte: {best_text}. Diese deskriptiven Maxima begründen keine universelle Promptüberlegenheit. Die fallweise Robustheitsanalyse identifizierte {failed_24} Fälle, die in allen 24 LoRA-Läufen falsch blieben, und {solved_24} Fälle, die in allen 24 korrekt waren. Gate 0.85 akzeptierte modellunabhängig 57 Demonstrationen und fiel in 975 Fällen exakt auf den jeweiligen Zero-Shot-Output zurück.", "",
        "## Interpretation", "",
        "Promptverfahren verschoben sowohl die Gesamtzahl korrekter Fälle als auch deren Identität. Gates sind daher als Mischbedingungen aus akzeptiertem Few Shot und deterministischem Zero-Shot-Fallback zu interpretieren. Ähnliche EMA-Werte bedeuten nicht, dass dieselben Fälle gelöst wurden. Die breiten Restfehlerfamilien überlappen und dürfen nicht als disjunkte Ursachen gelesen werden. Cross-Model-Unterschiede bleiben aufgrund unterschiedlicher Ausgangsmodelltypen deskriptiv (B/B+).", "",
        "## Abschluss", "",
        "Die experimentelle und analytische Run-Matrix ist abgeschlossen. Die offiziellen Ergebnisse werden nicht durch weitere Modellläufe, Gate-Schwellen oder Tokenlimits ergänzt. Offen bleiben der technische Reproduzierbarkeits-Freeze, die Integration in die Thesis und die formale Abschlusskontrolle.",
    ]
    write_text_new(OUT["thesis_text"], "\n".join(thesis_text))
    write_text_new(OUT["supported"], "# Gestützte Aussagen\n\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(supported, 1)))
    write_text_new(OUT["limitations"], "# Methodische Limitationen\n\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(limitations, 1)) + "\n\n## Nicht gestützte Aussagen\n\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(unsupported, 1)))

    mandatory = [row for row in gap_rows if row["status"] == "ZWINGEND OFFEN"]
    recommended = [row for row in gap_rows if row["status"] == "EMPFOHLEN OFFEN"]
    optional = [row for row in gap_rows if row["status"] == "OPTIONAL"]
    no_longer = [row for row in gap_rows if row["status"] == "NICHT MEHR ERFORDERLICH"]
    checklist = [
        "# Projektabschluss-Checkliste", "", "## Verbindliche Entscheidungen", "",
        "- Fehlende Modelltrainings: NEIN", "- Fehlende Evaluationen: NEIN", "- Fehlende notwendige Error-Analysis-Blöcke: NEIN",
        "- Rerun erforderlich: NEIN", "- Experimenteller Teil abgeschlossen: JA", "- Analytischer Teil abgeschlossen: JA",
        "- Technischer Reproduzierbarkeits-Freeze offen: JA", "- Schreiben der Thesis offen: JA", "",
        "## Nächste drei zwingende Schritte", "",
        "1. Finalen Reproduzierbarkeits-Freeze mit Commit/Tag, sauberem Status, Umgebungsmanifest und externer Sicherung erstellen.",
        "2. Die eingefrorenen Tabellen, Abbildungen, Methodenhinweise und qualifizierten Aussagen in die Thesis integrieren.",
        "3. Zahlen-, Quellen-, Format- und PDF-Sichtprüfung durchführen und die finale Abgabeversion sichern.", "",
        "## Keine weiteren Experimente", "",
    ] + [f"- {row['requirement']}" for row in no_longer]
    write_text_new(OUT["checklist"], "\n".join(checklist))

    audit_lines = [
        "# Vollständige 24-LoRA-Run-Fehleranalyse und Projektabschluss-Gap-Audit", "",
        "**COMPLETE-24-LORA-RUN-ERROR-ANALYSIS: PASS MIT METHODISCHEN EINSCHRÄNKUNGEN**", "",
        "**PROJECT-COMPLETION-GAP-AUDIT: PASS MIT WARNUNGEN**", "",
        "## Executive Summary", "",
        f"Alle 24 autoritativen LoRA-v2-Runs wurden eindeutig aus dem gehashten Cross-Model-Manifest zugeordnet. Jeder Run enthält {N:,} eindeutige Fälle in derselben Reihenfolge; insgesamt wurden 24.768 Vorhersagen geprüft. Sämtliche Config-, CSV-, Metadaten-, Trace- und vorhandenen Loghashes stimmen mit dem Manifest überein. Die Metrikreproduktion ergab null Abweichungen.", "",
        f"Die vollständige automatische Fehleranalyse klassifizierte {len(label_rows):,} Multi-Label-Zuweisungen auf inkorrekten Predictions. Sie ist explorativ, deterministisch und nicht menschlich validiert. `{ZE.PARSER_NAME}` `{ZE.PARSER_VERSION}` wurde unverändert wiederverwendet; Execution Match wurde nie durch ein Fehlerlabel überschrieben.", "",
        "Die 48-Run-Evaluation, alle drei Trainingslinien und die nun vollständige Error Analysis sind abgeschlossen. Es fehlen keine Modelltrainings, Evaluationen oder notwendigen Analyseblöcke. Verbindlich offen bleiben der technische Reproduzierbarkeits-Freeze, das Schreiben beziehungsweise Integrieren der Thesis und die formale Abschlusskontrolle.", "",
        "## 1. Scope und Read-only-Bestätigung", "",
        "Es wurden keine Modelle, Adapter, Tokenizer oder BGE-Modelle geladen, keine Generationen oder Trainings gestartet und keine bestehenden Artefakte verändert. SQLite-Diagnosen nutzten ausschließlich `mode=ro` und `PRAGMA query_only=ON`. Alle Dateien dieses Audits sind additiv.", "",
        "## 2. Quellintegrität", "",
        f"- LoRA-Runs: 24/24 eindeutig und valide", f"- Fälle: {N:,} pro Run; 24.768 Predictions", f"- Case-Order-SHA256: `{manifest['testset']['case_order_sha256']}`",
        f"- Testset-Inhalt: `{rel(resolved_testset)}` / `{sha256(resolved_testset)}`", f"- Manifest-Pfaddrift: `{manifest['testset']['path']}` verweist aktuell auf einen anderen 200-Zeilen-Inhalt; verwendet wurde die byteidentische 1.032-Zeilen-Kopie mit dem autoritativen Hash.",
        f"- Retrievalindex: `{manifest['retrieval']['index']}` / `{manifest['retrieval']['index_sha256']}`", "- Prompttruncationen: 0", "- Leakage: PASS", "- LORA-24-RUN-SOURCE-INTEGRITY: PASS MIT DOKUMENTATIONSWARNUNG (Testset-Pfaddrift)", "",
        "## 3. Vollständige Run-Matrix", "", table_a, "",
        "## 4. Fehlerklassifikation", "",
        f"Taxonomie: `{ZE.TAXONOMY_VERSION}`; Parser/Fallback: `{ZE.PARSER_NAME}` `{ZE.PARSER_VERSION}`. Breite Familien A-E sind technische/inhaltliche Multi-Label-Diagnosen. Familie F enthält Meta-/Heuristiklabels und wird nicht als fachliche Top-Fehlerfamilie gerankt.", "",
        "## 5. Restfehlerprofile", "", table_c, "",
        "Die Zeilen zählen eindeutige inkorrekte Fälle je Familie; ein Fall kann mehreren Familien angehören. E1/E2- und E3/E4-Zuweisungen stehen je Run in der maschinenlesbaren Profiltabelle.", "",
        "## 6. Zero-Shot-zentrierte Promptübergänge", "", table_b, "",
        "Nutzen minus Schaden entspricht in allen 21 Vergleichen exakt der EMA-Zählerdifferenz. McNemar-, Bootstrap- und Holm-7-Werte sind explorativ und separat je Modelllinie berechnet.", "",
        "## 7. Fehlerfamilienübergänge", "",
        "Die Long-Tabelle weist für jedes Label und jede breite Familie reparierte, eingeführte und persistente Fälle aus und trennt E1/E2 von E3/E4. Diese Übergänge zeigen Assoziationen mit Promptbedingungen, keine kausalen Wirkungen der Demonstration.", "",
        "## 8. Promptrobustheit", "", table_d, "",
        "## 9. Fehlerüberlappung und empirische Häufigkeit", "",
        f"Gemeinsam über alle 24 LoRA-Runs: {failed_24} Fälle falsch in 24/24, {failed_20} Fälle falsch in mindestens 20/24 und {solved_24} Fälle korrekt in 24/24. Die 8×8-Jaccard-Matrizen werden je Modelllinie vollständig berichtet. Diese Häufigkeit ist ausdrücklich keine offizielle Spider Difficulty.", "", table_i, "",
        "## 10. Gateanalyse", "", table_e, "",
        "Alle vier Gatebedingungen reproduzieren in ihren Fallbackpartitionen raw output, extrahiertes SQL und Outcomes exakt aus Zero Shot. Akzeptierte Fälle reproduzieren exakt den jeweiligen ungated Promptlauf. Gateeffekte entstehen daher ausschließlich durch die akzeptierte Teilmenge; Gate 0.85 ist überwiegend eine Zero-Shot-Reproduktion, aber nicht als optimale Schwelle interpretierbar.", "",
        "## 11. Ungated versus gated", "", table_f, "",
        "## 12. Top-1 versus Structure", "", table_g, "",
        "Keine Modelllinie stützt eine allgemeine kausale Überlegenheit von Structure oder Top-1. Demoidentität und SQL-Strukturdiagnosen sind fallweise beziehungsweise aggregiert in der Derived-Tabelle dokumentiert.", "",
        "## 13. Static versus dynamische Verfahren", "",
        "Alle sechs Rollen des Cross-Model-Designs und alle drei LoRA-Linien verwenden dieselbe materialisierte Full-Schema-Demo `SPIDER_TRAIN_001657` in 1.032/1.032 Fällen. Die Demo-Nähe ist lediglich eine E3-Heuristik.", "",
        "## 14. Output und Terminierung", "",
        "Qwen 2B LoRA und Qwen 9B LoRA besitzen über alle Bedingungen null Completionlimitfälle. Llama LoRA besitzt genau einen isolierten Llama-Static-Limitfall mit Repetitionsdiagnose. Dies ist Modellverhalten, kein Pipelinefehler.", "",
        "## 15. ESR versus EMA und Textmetriken", "", table_h, "",
        f"Die größte ESR-EMA-Lücke liegt bei {largest_gap['model_line']} / {COND_LABELS[largest_gap['condition']]} ({100*largest_gap['esr_minus_ema']:.2f} pp), die kleinste bei {smallest_gap['model_line']} / {COND_LABELS[smallest_gap['condition']]} ({100*smallest_gap['esr_minus_ema']:.2f} pp). Ausführbarkeit ist damit nicht mit Ergebnisrichtigkeit gleichzusetzen. Textmetriken bleiben ergänzend und ersetzen EMA nicht.", "",
        "## 16. Deskriptiver Cross-Model-Vergleich", "",
        "Qwen 2B versus Qwen 9B wird als B+ deskriptiv, Qwen versus Llama als B eingestuft. Aussagen über Modellgröße oder Architektur sind aus diesen Daten nicht kausal ableitbar.", "",
        "## 17. Qualitative Beispiele", "",
        "Die additive Beispieltabelle enthält exakt 8 Nutzen-, 8 Schadens-, 4 stabil ungelöste und 4 alternative valide Formulierungen. Sie ist illustrativ und trägt keine Prävalenzaussage.", "",
        "## 18. Forschungsfragen", "",
    ]
    for key, answer in rq_answers.items():
        audit_lines += [f"### {key}", "", answer, ""]
    audit_lines += [
        "## 19. Gestützte Aussagen", "",
        *[f"{i}. {text}" for i, text in enumerate(supported, 1)], "",
        "## 20. Nicht gestützte Aussagen", "",
        *[f"{i}. {text}" for i, text in enumerate(unsupported, 1)], "",
        "## 21. Methodische Limitationen", "",
        *[f"{i}. {text}" for i, text in enumerate(limitations, 1)], "",
        "## 22. Projektabschluss- und Gap-Audit", "", table_j, "",
        "### Abschlussentscheidungen", "",
        "- Fehlen noch Modelltrainings? **NEIN**", "- Fehlen noch Evaluationen? **NEIN**", "- Fehlen noch notwendige Error-Analysis-Blöcke? **NEIN**",
        "- Ist ein Rerun erforderlich? **NEIN**", "- Ist der experimentelle Teil abgeschlossen? **JA**", "- Ist der analytische Teil abgeschlossen? **JA**",
        "- Ist ein technischer Reproduzierbarkeits-Freeze noch offen? **JA**", "- Ist das Schreiben der Thesis noch offen? **JA**", "",
        "### Nächste drei zwingende Schritte", "",
        "1. Finalen Reproduzierbarkeits-Freeze mit Commit/Tag, Umgebungsmanifest und externer Sicherung erstellen.",
        "2. Thesis-ready Tabellen, Abbildungen und qualifizierte Aussagen in die Thesis integrieren.",
        "3. Zahlen-, Quellen-, Format- und PDF-Sichtprüfung abschließen und die Abgabeversion sichern.", "",
        "## 23. Read-only-Bestätigung", "",
        "Keine bestehende Config, CSV, Metadatei, Trace-, Log-, Audit-, Plot-, Modell-, Adapter- oder Datendatei wurde verändert. Kein Training und keine Evaluation wurde gestartet.",
    ]
    write_text_new(OUT["audit"], "\n".join(audit_lines))

    # The manifest is written last and hashes every newly produced artifact except itself.
    generated = [path for key, path in OUT.items() if key != "manifest"] + [base.with_suffix(ext) for base in PLOTS.values() for ext in (".png", ".pdf")]
    new_hashes = {rel(path): sha256(path) for path in generated}
    manifest_out = {
        "audit_status": "PASS MIT METHODISCHEN EINSCHRANKUNGEN",
        "project_gap_audit_status": "PASS MIT WARNUNGEN", "date": DATE,
        "scope": "complete read-only error analysis of 24 frozen LoRA-v2 runs and project completion gap review",
        "read_only": True, "existing_files_modified": False, "training_started": False,
        "evaluation_started": False, "model_or_adapter_loaded": False, "network_used": False,
        "authoritative_cross_manifest": {"path": rel(CROSS_MANIFEST), "sha256": sha256(CROSS_MANIFEST)},
        "source_checks": source_checks, "artifact_hash_checks": artifact_hashes,
        "run_ids": [record["run_id"] for record in lora_records],
        "runs": lora_records, "models": manifest["model_and_adapter_provenance"],
        "testset": {"manifest_path": manifest["testset"]["path"], "resolved_path": rel(resolved_testset), "sha256": sha256(resolved_testset), "rows": N, "case_order_sha256": manifest["testset"]["case_order_sha256"], "path_drift_warning": resolved_testset != requested_testset},
        "retrieval": manifest["retrieval"], "prompt_conditions": CONDITIONS, "predictions": 24 * N,
        "metric_validation": {"checks": len(metric_reproduction), "mismatches": sum(row["status"] != "PASS" for row in metric_reproduction), "rows": metric_reproduction},
        "error_codebook": {"path": "audits/error_analysis_codebook_20260716.md", "sha256": sha256(ROOT / "audits/error_analysis_codebook_20260716.md"), "taxonomy_version": ZE.TAXONOMY_VERSION},
        "clause_fallback": {"name": ZE.PARSER_NAME, "version": ZE.PARSER_VERSION, "external_ast_parser_used": False, "sqlite_mode": "read-only mode=ro + query_only"},
        "evidence_levels": ["E1", "E2", "E3", "E4"], "broad_families": {key: sorted(value) for key, value in BROAD.items()},
        "run_profiles": profiles, "zero_shot_prompt_transitions": transitions,
        "prompt_robustness_summary": {model: dict(robustness_summary[model]) for model in MODEL_ORDER},
        "jaccard_matrices_path": rel(OUT["overlap"]), "empirical_failure_frequency_summary": dict(freq_counts),
        "gate_analysis": {"expected_partitions": manifest["retrieval"]["gate_counts"], "rows": gate_rows, "fallback_identity": "PASS", "accepted_identity": "PASS"},
        "ungated_vs_gated": ungated_gated_rows, "top1_vs_structure": top1_structure_rows,
        "static_vs_dynamic": static_rows, "output_diagnostics": output_rows, "esr_ema_gap": esr_rows,
        "qualitative_examples": {"count": len(examples), "groups": dict(Counter(row["example_type"] for row in examples))},
        "cross_model_comparability": {"qwen2b_vs_qwen9b": "B+ descriptive", "qwen_vs_llama": "B descriptive"},
        "project_gap_audit": {"rows": gap_rows, "mandatory_open": [row["requirement"] for row in mandatory], "recommended_open": [row["requirement"] for row in recommended], "optional": [row["requirement"] for row in optional], "no_longer_required": [row["requirement"] for row in no_longer]},
        "decisions": {"missing_model_training": False, "missing_evaluation": False, "missing_necessary_error_analysis": False, "rerun_required": False, "experimental_part_complete": True, "analytical_part_complete": True, "technical_reproducibility_freeze_open": True, "thesis_writing_open": True},
        "environment_read_only_snapshot": environment_snapshot(),
        "supported_statements": supported, "unsupported_statements": unsupported, "limitations": limitations,
        "new_files": new_hashes, "manifest_self_hash": None,
        "manifest_self_hash_note": "The manifest cannot contain its own stable byte hash; hash it externally after creation.",
    }
    write_text_new(OUT["manifest"], json.dumps(manifest_out, ensure_ascii=False, indent=2, sort_keys=True))

    print(json.dumps({
        "status": "PASS MIT METHODISCHEN EINSCHRANKUNGEN", "valid_runs": 24,
        "predictions": 24 * N, "label_assignments": len(label_rows),
        "failed_24_of_24": failed_24, "failed_at_least_20_of_24": failed_20,
        "solved_24_of_24": solved_24, "audit": rel(OUT["audit"]),
        "manifest": rel(OUT["manifest"]), "new_file_count": len(generated) + 1,
    }, indent=2))


if __name__ == "__main__":
    main()
