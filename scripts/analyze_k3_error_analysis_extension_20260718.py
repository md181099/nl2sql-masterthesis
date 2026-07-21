#!/usr/bin/env python3
"""Additive k3 error-analysis extension using the frozen 20260716 taxonomy.

Model inference and retrieval are never invoked. The imported taxonomy opens
Spider SQLite databases read-only only for its established result diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import analyze_cross_model_zero_shot_error_taxonomy as taxonomy


ROOT = Path(__file__).resolve().parents[1]
N = 1032
DATE = "20260718"
ANALYSIS_CLASS = "K3 ERROR ANALYSIS EXTENSION"
BASELINE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
INVENTORY = ROOT / "audits/derived/k3_all_runs_completion_inventory_after_repair_20260718.csv"
MAPPING = ROOT / "audits/derived/k1_k3_authoritative_pair_mapping_20260718.csv"
METRICS = ROOT / "audits/derived/k3_authoritative_run_metrics_20260718.csv"
STATS = ROOT / "audits/derived/k1_vs_k3_paired_statistics_20260718.csv"
EFFICIENCY = ROOT / "audits/derived/k1_vs_k3_efficiency_comparison_20260718.csv"
GATES = ROOT / "audits/derived/k3_gate_and_fallback_distribution_20260718.csv"
DESCRIPTIVE = ROOT / "audits/derived/k3_descriptive_comparisons_20260718.csv"
STATE = ROOT / "audits/derived/k3_final_quantitative_state_20260718.json"
TESTSET = ROOT / "data/testcases_spider_dev_full.jsonl"
DEMO_POOL = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
INDEX_DIR = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
CODEBOOK = ROOT / "audits/error_analysis_codebook_20260716.md"
METHOD_AUDIT = ROOT / "docs/literature/error_validity_followup_20260718/AUDIT_ERROR_ANALYSIS_METHOD_20260718.md"
CODEBOOK_RECONSTRUCTION = ROOT / "docs/literature/error_validity_followup_20260718/ERROR_ANALYSIS_CODEBOOK_RECONSTRUCTION_20260718.md"

OUT_PROFILES = ROOT / "audits/derived/k3_error_profiles_20260718.csv"
OUT_TRANSITIONS = ROOT / "audits/derived/k1_k3_transition_counts_20260718.csv"
OUT_LABELS = ROOT / "audits/derived/k3_error_labels_long_20260718.csv"
OUT_EXAMPLES = ROOT / "audits/derived/k3_qualitative_examples_20260718.csv"
OUT_TEX = ROOT / "audits/derived/k3_thesis_ready_tables_20260718.tex"
OUT_AUDIT = ROOT / "audits/audit_k3_final_results_statistics_and_error_extension_20260718.md"
OUT_MANIFEST = ROOT / "audits/derived/k3_final_analysis_manifest_20260718.json"

MODEL_ORDER = ("qwen2b", "llama3b", "qwen9b")
ROLE_ORDER = ("base", "lora_v2")
K3_ORDER = ("top3", "top3_gate070", "top3_gate085", "structure_top3", "structure_top3_gate070", "structure_top3_gate085")
MODEL_LABELS = {"qwen2b": "Qwen 3.5 2B", "llama3b": "Llama 3.2 3B Instruct", "qwen9b": "Qwen 3.5 9B"}
MODEL_KEYS = {value: key for key, value in MODEL_LABELS.items()}
COARSE_FAMILIES = {
    "OUTPUT_CONTROL": "A_OUTPUT_CONTROL",
    "EXECUTION_SYNTAX": "B_SYNTAX_EXECUTION",
    "SCHEMA_LINKING": "C_SCHEMA_PROJECTION", "PROJECTION": "C_SCHEMA_PROJECTION",
    "AGGREGATION": "D_QUERY_STRUCTURE_LOGIC", "JOIN": "D_QUERY_STRUCTURE_LOGIC", "FILTER": "D_QUERY_STRUCTURE_LOGIC",
    "GROUPING": "D_QUERY_STRUCTURE_LOGIC", "ORDER_LIMIT": "D_QUERY_STRUCTURE_LOGIC", "SUBQUERY_SET": "D_QUERY_STRUCTURE_LOGIC",
    "RESULT_CARDINALITY": "E_RESULT_DEVIATION", "UNCLASSIFIED_REVIEW": "F_UNCLEAR_HEURISTIC",
}


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


def csv_text(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def verify_existing_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        existing = handle.read()
    if existing == csv_text(rows):
        return
    # The imported 20260716 taxonomy stringifies Python sets in these two
    # explanatory display fields. Their item order is process-dependent, while
    # labels, evidence levels, rules, case assignment, and counts are stable.
    if path == OUT_LABELS:
        existing_rows = read_csv(path)
        if len(existing_rows) != len(rows):
            raise RuntimeError(f"Existing additive label row count differs: {path}")
        ignored = {"gold_component", "pred_component"}
        for index, (old, new) in enumerate(zip(existing_rows, rows)):
            for key in old:
                if key not in ignored and old[key] != str(new.get(key, "")):
                    raise RuntimeError(f"Existing additive label output differs at row {index}, field {key}: {path}")
        return
    raise RuntimeError(f"Existing additive output differs from recomputation: {path}")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    write_text_new(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def demo_ids(row: dict[str, str]) -> list[str]:
    try:
        return [str(value) for value in json.loads(row.get("retrieved_ids") or "[]")]
    except json.JSONDecodeError:
        return []


def is_effective_fewshot(row: dict[str, str]) -> bool:
    decision = row.get("gate_decision", "")
    return decision != "zero_shot" and bool(demo_ids(row))


def add_fewshot_labels(
    labels: dict[str, dict[str, str]],
    row: dict[str, str],
    gold_features: dict[str, Any],
    pred_features: dict[str, Any],
    zero_row: dict[str, str],
    demos: dict[str, dict[str, Any]],
) -> None:
    if pred_features["order"] != gold_features["order"]:
        taxonomy.add_label(labels, "WRONG_ORDER_BY", "E2", "fewshot_order_difference", "few-shot ORDER BY differs from gold")
    if int(float(row.get("completion_tokens") or 0)) > int(float(zero_row.get("completion_tokens") or 0)) * 2 and int(float(row.get("completion_tokens") or 0)) >= 80:
        taxonomy.add_label(labels, "LONGER_OR_UNSTABLE_OUTPUT", "E2", "fewshot_length_ratio", "few-shot completion is over twice zero-shot length and at least 80 tokens")
    if not is_effective_fewshot(row):
        return
    pred_literals = set(pred_features["literals"]); gold_literals = set(gold_features["literals"])
    structural = lambda features: np.array([
        len(features["tables"]), features["joins"], sum(features["aggs"].values()), int(features["distinct"]),
        features["subqueries"], len(features["group"]), len(features["order"]), sum(features["setops"].values()),
    ], dtype=float)
    pred_vector = structural(pred_features); gold_vector = structural(gold_features)
    literal_sources: list[str] = []; structural_sources: list[str] = []
    for did in demo_ids(row):
        demo = demos.get(did)
        if not demo:
            continue
        features = taxonomy.sql_features(str(demo.get("gold_sql") or demo.get("sql") or ""))
        copied = set(features["literals"]) & pred_literals - gold_literals
        if copied:
            literal_sources.append(f"{did}:{sorted(copied)}")
        demo_vector = structural(features)
        adopted = np.any((pred_vector == demo_vector) & (gold_vector != demo_vector))
        if adopted and np.abs(pred_vector - demo_vector).sum() < np.abs(gold_vector - demo_vector).sum():
            structural_sources.append(did)
    if literal_sources:
        taxonomy.add_label(labels, "DEMO_LITERAL_COPY", "E3", "demo_literal_overlap", "prediction-only literal(s) occur in included demo(s): " + " | ".join(literal_sources))
    if structural_sources:
        taxonomy.add_label(labels, "DEMO_STRUCTURE_COPY", "E3", "demo_structure_distance", "prediction is structurally closer to included demo(s): " + ",".join(structural_sources))


def main() -> None:
    finalize_existing = "--finalize-existing" in sys.argv[1:]
    refresh_manifest = "--refresh-manifest" in sys.argv[1:]
    early_outputs = (OUT_PROFILES, OUT_TRANSITIONS, OUT_LABELS, OUT_EXAMPLES, OUT_AUDIT)
    if refresh_manifest:
        if any(not path.exists() for path in (*early_outputs, OUT_TEX, OUT_MANIFEST, DESCRIPTIVE)):
            raise RuntimeError("Manifest refresh requires all previously verified additive outputs")
    elif finalize_existing:
        if any(not path.exists() for path in early_outputs) or OUT_TEX.exists() or OUT_MANIFEST.exists():
            raise RuntimeError("Finalize mode requires the verified early outputs and free final targets")
    else:
        for path in (*early_outputs, OUT_TEX, OUT_MANIFEST):
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite: {path}")
    state = read_json(STATE)
    if state.get("k3_runs") != 36 or state.get("pairs") != 36:
        raise RuntimeError("Quantitative prerequisite is not complete")
    baseline_rows = read_csv(BASELINE); inventory_rows = read_csv(INVENTORY); mapping_rows = read_csv(MAPPING)
    metrics_rows = read_csv(METRICS); stats_rows = read_csv(STATS); efficiency_rows = read_csv(EFFICIENCY); gate_rows = read_csv(GATES); descriptive_rows = read_csv(DESCRIPTIVE)
    if not (len(baseline_rows) == 48 and len(inventory_rows) == len(mapping_rows) == len(metrics_rows) == len(stats_rows) == 36):
        raise RuntimeError("Unexpected authoritative register sizes")
    tests = read_jsonl(TESTSET); test_by_id = {row["id"]: row for row in tests}
    demos = {row["id"]: row for row in read_jsonl(DEMO_POOL)}
    if len(tests) != N or len(demos) != 6960:
        raise RuntimeError("Unexpected test or demo pool size")
    baseline_lookup = {(row["model_key"], row["role"], row["condition"]): row for row in baseline_rows}
    inventory_lookup = {(MODEL_KEYS[row["model_line"]], row["model_role"], row["condition"]): row for row in inventory_rows}

    run_cache: dict[tuple[str, str], list[dict[str, str]]] = {}
    def load_run(path: str, expected_hash: str) -> list[dict[str, str]]:
        key = (path, expected_hash)
        if key not in run_cache:
            target = ROOT / path
            if sha256(target) != expected_hash:
                raise RuntimeError(f"Prediction hash mismatch: {path}")
            rows = read_csv(target)
            if len(rows) != N or len({row["id"] for row in rows}) != N:
                raise RuntimeError(f"Prediction integrity failure: {path}")
            run_cache[key] = rows
        return run_cache[key]

    label_cache: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    sqlite_cache: dict[tuple[str, str], tuple[bool, list[tuple[Any, ...]] | None, str]] = {}
    labels_long: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_cases: list[dict[str, Any]] = []

    def classify_prediction(
        model: str, role: str, condition: str, side: str, run_id: str,
        row: dict[str, str], zero_row: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        cache_key = (run_id, row["id"], side, condition)
        if cache_key in label_cache:
            return label_cache[cache_key]
        if as_bool(row["exec_match"]):
            label_cache[cache_key] = {}
            return {}
        gold_features = taxonomy.sql_features(row["gold_sql"])
        pred_features = taxonomy.sql_features(row.get("pred_sql", ""))
        labels = taxonomy.classify(row, gold_features, pred_features, sqlite_cache)
        add_fewshot_labels(labels, row, gold_features, pred_features, zero_row, demos)
        label_cache[cache_key] = labels
        for label, detail in sorted(labels.items()):
            family = taxonomy.LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW")
            labels_long.append({
                "analysis_class": ANALYSIS_CLASS, "prediction_side": side, "model_key": model, "model_line": MODEL_LABELS[model],
                "model_role": role, "condition": condition, "run_id": run_id, "case_id": row["id"], "db_id": row["db_id"],
                "execution_success": as_bool(row["pred_ok"]), "execution_match": as_bool(row["exec_match"]),
                "error_label": label, "error_family": family, "coarse_family": COARSE_FAMILIES.get(family, "F_UNCLEAR_HEURISTIC"),
                "evidence_level": detail["evidence_level"], "automatic_rule": detail["automatic_rule"], "evidence_text": detail["evidence_text"],
                "gold_component": detail["gold_component"], "pred_component": detail["pred_component"],
                "demo_ids": json.dumps(demo_ids(row)), "effective_fewshot": is_effective_fewshot(row),
                "review_required": detail["evidence_level"] == "E4" or label == "MANUAL_REVIEW_REQUIRED",
                "causal_interpretation_permitted": False,
            })
        return labels

    for pair in mapping_rows:
        model = pair["model_key"]; role = pair["model_role"]; condition = pair["k3_condition"]
        k1_record = baseline_lookup[(model, role, pair["k1_condition"])]
        k3_record = inventory_lookup[(model, role, condition)]
        zero_record = baseline_lookup[(model, role, "zero_shot")]
        k1_rows = load_run(k1_record["csv_path"], k1_record["csv_sha256"])
        k3_rows = load_run(k3_record["output_path"], k3_record["output_sha256"])
        zero_rows = load_run(zero_record["csv_path"], zero_record["csv_sha256"])
        zero_by_id = {row["id"]: row for row in zero_rows}
        counts = Counter(); subtype = Counter(); benefit_families = Counter(); harm_families = Counter()
        for k1, k3 in zip(k1_rows, k3_rows):
            if k1["id"] != k3["id"]:
                raise RuntimeError(f"Pair order mismatch: {pair['pair_id']}")
            k1_ok = as_bool(k1["exec_match"]); k3_ok = as_bool(k3["exec_match"])
            transition = "STABLE_CORRECT" if k1_ok and k3_ok else "K3_BENEFIT" if not k1_ok and k3_ok else "K3_HARM" if k1_ok and not k3_ok else "PERSISTENT_WRONG"
            counts[transition] += 1
            if transition == "K3_BENEFIT": subtype["benefit_k1_nonexec"] += not as_bool(k1["pred_ok"]); subtype["benefit_k1_exec_wrong"] += as_bool(k1["pred_ok"])
            if transition == "K3_HARM": subtype["harm_k3_nonexec"] += not as_bool(k3["pred_ok"]); subtype["harm_k3_exec_wrong"] += as_bool(k3["pred_ok"])
            k1_labels = classify_prediction(model, role, pair["k1_condition"], "K1_REFERENCE", k1_record["run_id"], k1, zero_by_id[k1["id"]]) if not k1_ok else {}
            k3_labels = classify_prediction(model, role, condition, "K3_CURRENT", k3_record["run_id"], k3, zero_by_id[k3["id"]]) if not k3_ok else {}
            if transition == "K3_BENEFIT": benefit_families.update(set(taxonomy.LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW") for label in k1_labels))
            if transition == "K3_HARM": harm_families.update(set(taxonomy.LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW") for label in k3_labels))
            alternative = k3_ok and taxonomy.norm_sql(k3.get("pred_sql", "")) != taxonomy.norm_sql(k3.get("gold_sql", ""))
            transition_cases.append({
                "pair_id": pair["pair_id"], "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role,
                "condition": condition, "case_id": k3["id"], "db_id": k3["db_id"], "transition": transition,
                "k1_execution_success": as_bool(k1["pred_ok"]), "k3_execution_success": as_bool(k3["pred_ok"]),
                "k1_execution_match": k1_ok, "k3_execution_match": k3_ok,
                "k1_error_labels": json.dumps(sorted(k1_labels)), "k3_error_labels": json.dumps(sorted(k3_labels)),
                "k1_error_families": json.dumps(sorted(set(taxonomy.LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW") for label in k1_labels))),
                "k3_error_families": json.dumps(sorted(set(taxonomy.LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW") for label in k3_labels))),
                "alternative_valid_k3_formulation": alternative,
                "question": k3.get("question", ""), "gold_sql": k3.get("gold_sql", ""), "k1_pred_sql": k1.get("pred_sql", ""), "k3_pred_sql": k3.get("pred_sql", ""),
                "k1_raw_output": k1.get("raw_output", ""), "k3_raw_output": k3.get("raw_output", ""), "k3_demo_ids": json.dumps(demo_ids(k3)),
            })
        transition_rows.append({
            "pair_id": pair["pair_id"], "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role, "condition": condition,
            "cases": N, "k3_benefit": counts["K3_BENEFIT"], "k3_harm": counts["K3_HARM"],
            "persistent_wrong": counts["PERSISTENT_WRONG"], "stable_correct": counts["STABLE_CORRECT"],
            "net_benefit_minus_harm": counts["K3_BENEFIT"] - counts["K3_HARM"],
            "benefit_k1_nonexecutable": subtype["benefit_k1_nonexec"], "benefit_k1_executable_wrong": subtype["benefit_k1_exec_wrong"],
            "harm_k3_nonexecutable": subtype["harm_k3_nonexec"], "harm_k3_executable_wrong": subtype["harm_k3_exec_wrong"],
            "benefit_k1_error_families_json": json.dumps(benefit_families, sort_keys=True),
            "harm_k3_error_families_json": json.dumps(harm_families, sort_keys=True),
            "terminology_note": "descriptive paired transition; no causal attribution",
        })

    # Profiles are based only on the current k3 predictions; k1 labels remain available for transition interpretation.
    current_labels = [row for row in labels_long if row["prediction_side"] == "K3_CURRENT"]
    profiles: list[dict[str, Any]] = []
    incorrect_by_run = Counter((row["model_key"], row["model_role"], row["condition"]) for row in transition_cases if not row["k3_execution_match"])
    for model in MODEL_ORDER:
        for role in ROLE_ORDER:
            for condition in K3_ORDER:
                subset = [row for row in current_labels if row["model_key"] == model and row["model_role"] == role and row["condition"] == condition]
                run_key = (model, role, condition); incorrect = incorrect_by_run[run_key]
                for level, field in (("DETAIL_LABEL", "error_label"), ("DETAIL_FAMILY", "error_family"), ("COARSE_FAMILY", "coarse_family")):
                    values = sorted(set(row[field] for row in subset))
                    for value in values:
                        selected = [row for row in subset if row[field] == value]
                        cases = {row["case_id"] for row in selected}
                        evidence_cases = {evidence: len({row["case_id"] for row in selected if row["evidence_level"] == evidence}) for evidence in ("E1", "E2", "E3", "E4")}
                        profiles.append({
                            "analysis_class": ANALYSIS_CLASS, "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role,
                            "condition": condition, "aggregation_level": level, "category": value, "incorrect_predictions": incorrect,
                            "unique_case_count": len(cases), "label_assignment_count": len(selected), "share_of_incorrect": len(cases) / incorrect if incorrect else 0.0,
                            "E1_case_count": evidence_cases["E1"], "E2_case_count": evidence_cases["E2"], "E3_case_count": evidence_cases["E3"], "E4_case_count": evidence_cases["E4"],
                            "multi_label_note": "case counts may overlap across labels/families; labels are not independent error cases",
                        })

    # Deterministic 24-case illustration: one Base and one LoRA case per model and requested example class.
    examples: list[dict[str, Any]] = []
    example_classes = ("K3_BENEFIT", "K3_HARM", "PERSISTENT_WRONG", "ALTERNATIVE_VALID_FORMULATION")
    used: set[tuple[str, str, str, str]] = set()
    for class_index, example_class in enumerate(example_classes):
        for model_index, model in enumerate(MODEL_ORDER):
            for role_index, role in enumerate(ROLE_ORDER):
                candidates = [row for row in transition_cases if row["model_key"] == model and row["model_role"] == role]
                if example_class == "ALTERNATIVE_VALID_FORMULATION":
                    candidates = [row for row in candidates if row["alternative_valid_k3_formulation"]]
                else:
                    candidates = [row for row in candidates if row["transition"] == example_class]
                target_condition = K3_ORDER[(class_index + model_index + role_index) % len(K3_ORDER)]
                candidates.sort(key=lambda row: (row["condition"] != target_condition, K3_ORDER.index(row["condition"]), row["case_id"]))
                chosen = next((row for row in candidates if (example_class, model, role, row["case_id"]) not in used), None)
                if chosen is None:
                    raise RuntimeError(f"No qualitative candidate for {example_class}:{model}:{role}")
                used.add((example_class, model, role, chosen["case_id"]))
                examples.append({
                    "example_id": f"K3-{example_class}-{model}-{role}", "selection_class": example_class,
                    "model_key": model, "model_line": MODEL_LABELS[model], "model_role": role, "condition": chosen["condition"],
                    "case_id": chosen["case_id"], "db_id": chosen["db_id"], "question": chosen["question"], "gold_sql": chosen["gold_sql"],
                    "k1_pred_sql": chosen["k1_pred_sql"], "k3_pred_sql": chosen["k3_pred_sql"],
                    "k1_execution_match": chosen["k1_execution_match"], "k3_execution_match": chosen["k3_execution_match"],
                    "k1_error_labels": chosen["k1_error_labels"], "k3_error_labels": chosen["k3_error_labels"], "k3_demo_ids": chosen["k3_demo_ids"],
                    "selection_rule": "deterministic model-role stratum; rotating preferred condition; lowest case ID after condition ordering",
                    "validation_status": "AUTOMATIC_UNREVIEWED_ILLUSTRATION", "representative": False, "human_validated": False,
                    "causal_interpretation_permitted": False,
                })

    if finalize_existing or refresh_manifest:
        verify_existing_csv(OUT_PROFILES, profiles)
        verify_existing_csv(OUT_TRANSITIONS, transition_rows)
        verify_existing_csv(OUT_LABELS, labels_long)
        verify_existing_csv(OUT_EXAMPLES, examples)
    else:
        write_csv_new(OUT_PROFILES, profiles)
        write_csv_new(OUT_TRANSITIONS, transition_rows)
        write_csv_new(OUT_LABELS, labels_long)
        write_csv_new(OUT_EXAMPLES, examples)

    aggregate = state["aggregate"]
    transition_total = Counter()
    for row in transition_rows:
        transition_total["benefit"] += int(row["k3_benefit"]); transition_total["harm"] += int(row["k3_harm"])
        transition_total["persistent_wrong"] += int(row["persistent_wrong"]); transition_total["stable_correct"] += int(row["stable_correct"])
    evidence_total = Counter(row["evidence_level"] for row in current_labels)
    current_incorrect_cases = len({(row["model_key"], row["model_role"], row["condition"], row["case_id"]) for row in current_labels})
    alt_count = sum(row["alternative_valid_k3_formulation"] for row in transition_cases)

    result_lines = ["| Model | Role | Condition | EMA | ESR | Correct | Prompt tokens | Completion tokens | Timeout |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows:
        result_lines.append(f"| {row['model_line']} | {row['role_label']} | {row['condition_label']} | {100*float(row['ema']):.2f}% | {100*float(row['esr']):.2f}% | {row['ema_correct']}/1032 | {float(row['prompt_tokens_mean']):.1f} | {float(row['completion_tokens_mean']):.1f} | {row['prediction_timeout_cases']} |")
    pair_lines = ["| Model | Role | Comparison | Delta pp | Benefit | Harm | p | Holm-36 p | 95% CI pp |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats_rows:
        pair_lines.append(f"| {row['model_line']} | {row['role_label']} | {row['condition']} | {float(row['delta_pp']):.2f} | {row['k1_wrong_k3_correct_n01']} | {row['k1_correct_k3_wrong_n10']} | {float(row['mcnemar_p_raw']):.4g} | {float(row['holm_adjusted_p']):.4g} | [{float(row['bootstrap_ci_low_pp']):.2f}, {float(row['bootstrap_ci_high_pp']):.2f}] |")
    gate_unique: dict[str, tuple[int, int]] = {}
    for row in gate_rows:
        gate_unique[row["condition"]] = (int(row["effective_k3_cases"]), int(row["zero_shot_fallback_cases"]))
    gate_lines = ["| Condition | k=3 | Zero-shot fallback |", "|---|---:|---:|"] + [f"| {condition} | {values[0]} | {values[1]} |" for condition, values in gate_unique.items()]
    profile_overall = Counter((row["error_family"], row["evidence_level"]) for row in current_labels)
    profile_lines = ["| Detail family | E1 assignments | E2 | E3 | E4 |", "|---|---:|---:|---:|---:|"]
    for family in taxonomy.FAMILY_LABELS:
        profile_lines.append(f"| {family} | {profile_overall[(family,'E1')]} | {profile_overall[(family,'E2')]} | {profile_overall[(family,'E3')]} | {profile_overall[(family,'E4')]} |")

    audit_text = f"""# Authoritative k3 final results, k1-vs-k3 statistics, and error extension

## 1. Executive Summary

```text
K3-FINAL-QUANTITATIVE-ANALYSIS: PASS MIT WARNUNGEN
K1-VS-K3-STATISTICS: PASS MIT WARNUNGEN
K3-ERROR-ANALYSIS: PASS MIT WARNUNGEN
K3-THESIS-ANALYSIS: READY_WITH_LIMITATIONS
```

All 36 k3 runs and all 36 semantic k1-k3 pairs passed artifact-hash, 1,032-case,
case-order, metric-reproduction, retrieval, and summary checks. The frozen 48-run analysis remains
the `PRE-K3 AUTHORITATIVE BASELINE`; the new material is a separate 36-run extension, not a homogeneous
84-run matrix.

Across the 36 paired EMA comparisons, {aggregate['positive_differences']} point estimates favored k3,
{aggregate['negative_differences']} favored k1, and {aggregate['identical_differences']} were identical.
No comparison remained significant after the new Holm-36 correction. The mean delta was
{aggregate['mean_delta_pp']:.3f} pp and the median was {aggregate['median_delta_pp']:.3f} pp.

## 2. Authoritative Run Registers

- Pre-k3: 48/48 runs from `{BASELINE.relative_to(ROOT)}`; unchanged.
- K3: 36/36 runs from `{INVENTORY.relative_to(ROOT)}`; 24 `COMPLETE_VALID`, 12 Qwen-9B `COMPLETE_WITH_WARNING`.
- Historical Qwen-9B Base Top-3 partial runs are excluded in the manifest.
- Testset: `{TESTSET.relative_to(ROOT)}`, SHA256 `{sha256(TESTSET)}`.
- Retrieval: `{INDEX_DIR.relative_to(ROOT)}`, 6,960 examples, `BAAI/bge-large-en-v1.5`.

## 3. K1-K3 Pair Mapping

All 36 pairs are `MATCHED_WITH_LIMITATIONS`. The matching dimensions are model/adapter, testset/order,
system prompt, model-native prompt format, full-schema representation, retrieval pool/model, gate and
structure mode, extractor, greedy decoding, batch size 1, and 256 output tokens. Intended differences are
one versus three demonstrations, prompt length, and the 2,048 versus 4,352 input ceilings. For Qwen 9B,
k3 additionally uses a 900-second per-statement guard.

## 4. Comparability

The prior prompt-equivalence audit found no k1 truncation and no token-ID or prompt-hash change when the
k1 ceiling was hypothetically raised. This supports pairing but does not turn the comparison into an isolated
one-factor causal estimate: k3 was evaluated under a larger permitted context window. The mixed timeout policy
is inactive in 34 runs and active for `SPIDER_DEV_000484` in two Qwen-9B Base runs.

## 5. Descriptive K3 Results

{chr(10).join(result_lines)}

The complete machine-readable table also contains String EM, normalized EM, Char/Token Accuracy, generation
time, all token means, limit cases, empty outputs/extractions, retrieval similarity, effective-k distribution,
artifact paths, hashes, and summary-reproduction status. Base-LoRA, Top3-Structure, ungated-gate, model-line,
and best-condition comparisons are descriptive unless covered by the explicit k1-k3 family.

## 6. Retrieval and Gate Analysis

{chr(10).join(gate_lines)}

All runs contain three distinct persisted candidate IDs and scores in 1,032/1,032 cases. Base-LoRA candidate
identity, gate-to-ungated selection identity, fallback-to-zero output identity, and accepted-to-ungated output
identity are all 1,032/1,032 within their relevant subsets. Gate 0.85 is predominantly Zero Shot and must not
be described as continuously three-shot. Similarity scores are uncalibrated similarities, not probabilities.

## 7. Efficiency Analysis

Across the 36 within-model pairs, k3 increased the mean prompt length by
{aggregate['mean_prompt_token_delta']:.2f} tokens. The mean generation-time delta was
{aggregate['mean_generation_time_delta_seconds']:.6f} seconds per case ({100*aggregate['mean_generation_time_relative_change']:.3f}% relative).
Completion and total-token changes are reported per pair. Cross-model token totals are not treated as
tokenizer-independent measurements.

## 8. Paired K1-K3 Statistics

{chr(10).join(pair_lines)}

All tests use the exact two-sided McNemar test and a paired percentile bootstrap with 10,000 resamples,
seed 20260716. `EMA_k3 - EMA_k1` is the target statistic. The 36 comparisons form the single new family
`K1_VS_K3_DEMONSTRATION_COUNT_FAMILY`; no pre-k3 Holm family was changed. Non-significance is not evidence
of equality. Although several unadjusted intervals exclude zero, no comparison survives Holm-36.

## 9. Aggregate Patterns

- Positive/negative/identical deltas: {aggregate['positive_differences']}/{aggregate['negative_differences']}/{aggregate['identical_differences']}.
- Holm-significant k3 advantages/disadvantages: {aggregate['holm_significant_k3_advantages']}/{aggregate['holm_significant_k3_disadvantages']}.
- Largest advantage: `{aggregate['largest_advantage_pair']}`, {aggregate['max_delta_pp']:.3f} pp.
- Largest disadvantage: `{aggregate['largest_disadvantage_pair']}`, {aggregate['min_delta_pp']:.3f} pp.
- Mean/median delta: {aggregate['mean_delta_pp']:.3f}/{aggregate['median_delta_pp']:.3f} pp.

These 36 point estimates are descriptively aggregated; they are not treated as independent observations for
another significance test.

## 10. K3 Error Analysis Extension

The frozen taxonomy `{taxonomy.TAXONOMY_VERSION}` and parser `{taxonomy.PARSER_NAME}` `{taxonomy.PARSER_VERSION}`
were imported unchanged. The block contains {len(current_labels):,} deduplicated label assignments over
{current_incorrect_cases:,} incorrect k3 run-case observations. E1/E2/E3/E4 assignment counts are
{evidence_total['E1']:,}/{evidence_total['E2']:,}/{evidence_total['E3']:,}/{evidence_total['E4']:,}.
Labels are multi-label: assignment totals are not counts of independent error cases.

{chr(10).join(profile_lines)}

Pair transitions total {transition_total['benefit']:,} k3 Benefits, {transition_total['harm']:,} k3 Harms,
{transition_total['persistent_wrong']:,} Persistent-Wrong, and {transition_total['stable_correct']:,} Stable-Correct
observations. These are descriptive outcomes, not causal mechanisms. `DEMO_LITERAL_COPY` and
`DEMO_STRUCTURE_COPY` remain E3 associations. The taxonomy used read-only SQLite diagnostics with the original
progress and 10,000-row caps; it did not alter EMA or any source artifact.

## 11. Qualitative Examples

The 24 examples include six each for Benefit, Harm, Persistent Wrong, and alternative execution-matching
formulations, with Base and LoRA represented for every model line. Selection is deterministic and rule-bound.
The examples are automatic, unreviewed, illustrative, non-representative, and unsuitable for prevalence claims.

## 12. Methodological Limitations

1. K1 uses 2,048 and k3 4,352 permitted input tokens; the pairing estimates the combined configured comparison,
   not a pure causal demonstration-count effect.
2. The k3 timeout policy is mixed: 24 runs have no explicit statement timeout, twelve Qwen-9B v3 runs use 900 s.
3. The taxonomy is project-local, rule-based, heuristic, multi-label, and not human-validated.
4. Read-only result diagnostics describe the frozen Spider SQLite instances and do not prove universal SQL equivalence.
5. Gate 0.85 overwhelmingly reproduces Zero Shot; similarity is not a calibrated probability.

## 13. Thesis Release

The k3 quantitative, efficiency, retrieval, transition, and explicitly exploratory error results are ready with
the stated limitations. The 48-run main study remains authoritative and unchanged. Report the project as a
48-run main investigation plus a separate 36-run extension, or as 84 evaluated runs, never as a homogeneous
84-run matrix.

## 14. Generated Artifacts and Hashes

Hashes and complete source provenance are recorded in `{OUT_MANIFEST.relative_to(ROOT)}`. The manifest excludes
itself from its checksum map. No source prediction, config, metadata, trace, pre-k3 statistic, or audit was modified.

## 15. Chapter Recommendations

- **5.4.3:** Define k3 selection, binary gate fallback, 4,352-token ceiling, and prompt-equivalence evidence.
- **5.4.4:** Define exactly the new Holm-36 family, McNemar direction, bootstrap seed/resamples, and non-equivalence caveat.
- **5.5.4:** Present the k3 error block separately, retain E1-E4, multi-label counting, E3 demo-copy limits, and unreviewed examples.
- **Chapter 7:** Report all 36 descriptive outcomes and paired tests, while separating them from the 48-run conclusions.

## Final Status

```text
K3-FINAL-QUANTITATIVE-ANALYSIS: PASS MIT WARNUNGEN
K1-VS-K3-STATISTICS: PASS MIT WARNUNGEN
K3-ERROR-ANALYSIS: PASS MIT WARNUNGEN
K3-THESIS-ANALYSIS: READY_WITH_LIMITATIONS
```
"""
    if finalize_existing or refresh_manifest:
        if OUT_AUDIT.read_text(encoding="utf-8") != audit_text:
            raise RuntimeError("Existing additive audit differs from recomputation")
    else:
        write_text_new(OUT_AUDIT, audit_text)

    tex = r"""%% Additive tables only; interpretation remains in the audit.
\begin{table}[htbp]
\centering
\caption{Aggregated paired k1--k3 EMA pattern.}
\begin{tabular}{lr}
\hline
Positive k3 differences & %d \\
Negative k3 differences & %d \\
Identical point estimates & %d \\
Holm-significant k3 advantages & %d \\
Holm-significant k3 disadvantages & %d \\
Mean difference (pp) & %.3f \\
Median difference (pp) & %.3f \\
\hline
\end{tabular}
\end{table}
""" % (aggregate["positive_differences"], aggregate["negative_differences"], aggregate["identical_differences"], aggregate["holm_significant_k3_advantages"], aggregate["holm_significant_k3_disadvantages"], aggregate["mean_delta_pp"], aggregate["median_delta_pp"])
    if refresh_manifest:
        if OUT_TEX.read_text(encoding="utf-8") != tex:
            raise RuntimeError("Existing LaTeX table differs from recomputation")
    else:
        write_text_new(OUT_TEX, tex)

    # Register all authoritative source artifacts and explicitly excluded partials.
    pre_k3_register = []
    for row in baseline_rows:
        pre_k3_register.append({
            "run_id": row["run_id"], "model_key": row["model_key"], "role": row["role"], "condition": row["condition"],
            "config": {"path": row["config_path"], "sha256": row["config_sha256"]},
            "result": {"path": row["csv_path"], "sha256": row["csv_sha256"]},
            "metadata": {"path": row["metadata_path"], "sha256": row["metadata_sha256"]},
            "trace": {"path": row["trace_path"], "sha256": row["trace_sha256"]} if row["trace_path"] else None,
        })
    k3_register = []
    for row in inventory_rows:
        k3_register.append({
            "run_id": row["run_id"], "model_line": row["model_line"], "role": row["model_role"], "condition": row["condition"],
            "status": row["final_run_status"], "timeout_policy": "900s_per_statement" if row["model_line"] == "Qwen 3.5 9B" else "no_explicit_statement_timeout",
            "config": {"path": row["config_path"], "sha256": row["config_sha256"]},
            "result": {"path": row["output_path"], "sha256": row["output_sha256"]},
            "metadata": {"path": row["metadata_path"], "sha256": row["metadata_sha256"]},
            "trace": {"path": row["trace_path"], "sha256": row["trace_sha256"]},
            "log": {"path": row["log_path"], "sha256": row["log_sha256"]},
        })
    exclusions = []
    for path in (
        ROOT / "results/k3_extension_20260717/run_k3_qwen9b_base_top3_maxin4352_20260718_011245.csv",
        ROOT / "results/k3_extension_20260717/run_k3_qwen9b_base_top3_maxin4352_sqltimeout900_20260718_085006.csv",
    ):
        exclusions.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "reason": "historical partial Qwen9B Base Top3; excluded"})
    generated_paths = [
        METRICS, MAPPING, STATS, EFFICIENCY, GATES, DESCRIPTIVE, STATE,
        OUT_PROFILES, OUT_TRANSITIONS, OUT_LABELS, OUT_EXAMPLES, OUT_TEX, OUT_AUDIT,
        ROOT / "scripts/analyze_k3_final_results_and_k1_vs_k3_statistics_20260718.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "status": "READY_WITH_LIMITATIONS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scopes": ["PRE-K3 AUTHORITATIVE BASELINE", "36-RUN K3 EXTENSION", "PAIRED K1-VS-K3", "K3 ERROR ANALYSIS EXTENSION"],
        "pre_k3_runs": pre_k3_register, "k3_runs": k3_register, "pairs": mapping_rows,
        "testset": {"path": str(TESTSET.relative_to(ROOT)), "sha256": sha256(TESTSET), "cases": N},
        "retrieval": {
            "index_path": str(INDEX_DIR.relative_to(ROOT)), "pool_cases": 6960, "embedding_model": "BAAI/bge-large-en-v1.5",
            "files": {str(path.relative_to(ROOT)): sha256(path) for path in (INDEX_DIR / "index.faiss", INDEX_DIR / "metadata.jsonl", INDEX_DIR / "manifest.json")},
        },
        "statistics": {"family": "K1_VS_K3_DEMONSTRATION_COUNT_FAMILY", "family_size": 36, "mcnemar": "exact_two_sided", "holm": True, "bootstrap_seed": 20260716, "bootstrap_resamples": 10000, "bootstrap_ci": "paired_percentile_95"},
        "error_analysis": {
            "class": ANALYSIS_CLASS, "taxonomy_version": taxonomy.TAXONOMY_VERSION, "taxonomy_script": {"path": str(Path(taxonomy.__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(taxonomy.__file__).resolve())},
            "codebook": {"path": str(CODEBOOK.relative_to(ROOT)), "sha256": sha256(CODEBOOK)}, "evidence_levels": ["E1", "E2", "E3", "E4"],
            "read_only_sql_diagnostics_required": True, "sqlite_unique_query_cache_entries": len(sqlite_cache), "result_cap": 10000,
            "manual_validation": False, "qualitative_examples": len(examples), "qualitative_status": "AUTOMATIC_UNREVIEWED_ILLUSTRATION",
        },
        "method_sources": {str(path.relative_to(ROOT)): sha256(path) for path in (METHOD_AUDIT, CODEBOOK_RECONSTRUCTION)},
        "limitations": [
            "k1 max_input_tokens=2048; k3=4352; prompt equivalence PASS but demonstration count is not an isolated causal factor",
            "24 k3 runs without explicit statement timeout; 12 Qwen9B v3 runs with 900 seconds per statement",
            "timeout effective for SPIDER_DEV_000484 in two Qwen9B Base runs",
            "project-local automatic multi-label taxonomy is not human-validated",
            "Gate 0.85 predominantly falls back to Zero Shot",
        ],
        "excluded_historical_runs": exclusions,
        "pre_k3_holm_families_modified": False, "pre_k3_artifacts_modified": False,
        "generated_files": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "size": path.stat().st_size} for path in generated_paths},
        "manifest_self_hash_note": "Self-hash is intentionally external because embedding it would be recursive.",
    }
    if refresh_manifest:
        OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        write_json_new(OUT_MANIFEST, manifest)
    print(json.dumps({
        "status": "PASS_WITH_WARNINGS", "k3_label_assignments": len(current_labels), "k3_incorrect_observations": current_incorrect_cases,
        "transitions": dict(transition_total), "qualitative_examples": len(examples), "read_only_sql_cache_entries": len(sqlite_cache),
    }, indent=2))


if __name__ == "__main__":
    main()
