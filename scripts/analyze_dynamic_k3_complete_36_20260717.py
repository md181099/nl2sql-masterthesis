#!/usr/bin/env python3
"""Read-only analysis of the completed 36-run dynamic few-shot k=3 extension."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N = 1032
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260717
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
K1_REFERENCE = {
    "top3": "top1",
    "top3_gate070": "top1_gate070",
    "top3_gate085": "top1_gate085",
    "structure_top3": "structure",
    "structure_top3_gate070": "structure_gate070",
    "structure_top3_gate085": "structure_gate085",
}
APPROACH_UNGATED = {
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
ROLE_LABELS = {"base": "Base/Ausgangsmodell", "lora_v2": "LoRA v2"}
CONDITION_LABELS = {
    "top3": "Dynamic Top-3",
    "top3_gate070": "Top-3 Gate 0.70",
    "top3_gate085": "Top-3 Gate 0.85",
    "structure_top3": "Structure Top-3",
    "structure_top3_gate070": "Structure Top-3 Gate 0.70",
    "structure_top3_gate085": "Structure Top-3 Gate 0.85",
}
BASELINE_RESULTS = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
K1_EQUIVALENCE = ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json"
OUTPUTS = {
    "audit": ROOT / "audits/audit_dynamic_fewshot_k3_complete_36_run_evaluation_20260717.md",
    "manifest": ROOT / "audits/dynamic_fewshot_k3_complete_36_run_evaluation_manifest_20260717.json",
    "results": ROOT / "audits/derived/dynamic_k3_complete_results_20260717.csv",
    "stats": ROOT / "audits/derived/dynamic_k1_vs_k3_paired_statistics_20260717.csv",
    "gate": ROOT / "audits/derived/dynamic_k3_gate_analysis_20260717.csv",
    "retrieval": ROOT / "audits/derived/dynamic_k3_retrieval_profiles_20260717.csv",
    "error": ROOT / "audits/derived/dynamic_k3_error_transitions_20260717.csv",
    "error_summary": ROOT / "audits/derived/dynamic_k3_error_transition_summary_20260717.csv",
    "tables": ROOT / "audits/derived/dynamic_k3_thesis_ready_tables_20260717.md",
    "text": ROOT / "audits/derived/dynamic_k3_thesis_ready_text_20260717.md",
}
DOCS = ROOT / "docs/final_project_documentation_20260717_k3_extension_v3_84run"


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


def as_bool(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def exact_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def bootstrap_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    offset = 0
    while offset < BOOTSTRAP_RESAMPLES:
        size = min(250, BOOTSTRAP_RESAMPLES - offset)
        indices = rng.integers(0, len(diff), size=(size, len(diff)))
        values[offset : offset + size] = diff[indices].mean(axis=1)
        offset += size
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stable_seed(label: str) -> int:
    return BOOTSTRAP_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def paired_row(
    family: str,
    model_key: str,
    role: str,
    comparison: str,
    a_label: str,
    b_label: str,
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, Any]:
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    both_correct = int(np.sum((a == 1) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))
    ci_low, ci_high = bootstrap_ci(b.astype(float) - a.astype(float), stable_seed(comparison))
    a_ema = float(a.mean())
    b_ema = float(b.mean())
    error_a = 1.0 - a_ema
    return {
        "family": family,
        "model_key": model_key,
        "model_line": MODEL_LABELS[model_key],
        "role": role,
        "role_label": ROLE_LABELS.get(role, role),
        "comparison": comparison,
        "reference": a_label,
        "target": b_label,
        "reference_correct": int(a.sum()),
        "target_correct": int(b.sum()),
        "reference_ema": a_ema,
        "target_ema": b_ema,
        "delta": b_ema - a_ema,
        "delta_pp": 100.0 * (b_ema - a_ema),
        "repairs_n01": n01,
        "harms_n10": n10,
        "net_repairs": n01 - n10,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "relative_error_reduction": (a_ema - b_ema) * -1.0 / error_a if error_a else None,
        "mcnemar_p": exact_mcnemar_p(n01, n10),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_seed": stable_seed(comparison),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def holm_by_family(rows: list[dict[str, Any]]) -> None:
    for family in sorted({str(row["family"]) for row in rows}):
        positions = [index for index, row in enumerate(rows) if row["family"] == family]
        ordered = sorted(positions, key=lambda index: float(rows[index]["mcnemar_p"]))
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(rows[index]["mcnemar_p"]))
            running = max(running, adjusted)
            rows[index]["holm_family_size"] = count
            rows[index]["holm_p"] = running
            rows[index]["significant_raw_0_05"] = float(rows[index]["mcnemar_p"]) < 0.05
            rows[index]["significant_holm_0_05"] = running < 0.05


def classify_error(row: dict[str, str]) -> str:
    if as_bool(row["exec_match"]):
        return "correct"
    pred = row.get("pred_sql", "").strip()
    error = row.get("pred_error", "").lower()
    raw = row.get("raw_output", "")
    if not pred:
        return "empty_sql"
    if not as_bool(row["pred_ok"]):
        for needle, label in (
            ("syntax error", "syntax_error"),
            ("no such table", "missing_table"),
            ("no such column", "missing_column"),
            ("ambiguous", "ambiguous_column"),
            ("misuse of aggregate", "aggregate_execution_error"),
            ("timeout", "timeout"),
        ):
            if needle in error:
                return label
        return "other_execution_error"
    words = re.findall(r"\w+|\S", raw.lower())
    repeated_five = 0.0
    if len(words) >= 5:
        grams = [tuple(words[index : index + 5]) for index in range(len(words) - 4)]
        repeated_five = 1.0 - len(set(grams)) / len(grams)
    if int(float(row.get("completion_tokens") or 0)) == 256 and repeated_five >= 0.2:
        return "completion_limit_repetition"
    if int(float(row.get("completion_tokens") or 0)) == 256:
        return "completion_limit_semantic_error"
    if repeated_five >= 0.2:
        return "repetitive_executable_error"
    return "executable_semantic_mismatch"


def load_sources() -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, str]],
    list[dict[str, Any]],
]:
    equivalence = read_json(K1_EQUIVALENCE)
    if equivalence.get("status") != "PASS" or not equivalence.get("k1_vs_k3_comparison_permitted"):
        raise RuntimeError("K1 prompt equivalence is not PASS")
    matrix_rows = read_csv(MATRIX)
    if len(matrix_rows) != 36:
        raise RuntimeError("Expected 36 k3 config rows")
    matrix = {(row["model_key"], row["role"], row["condition"]): row for row in matrix_rows}
    baseline_rows = read_csv(BASELINE_RESULTS)
    if len(baseline_rows) != 48:
        raise RuntimeError("Expected frozen 48-run baseline")
    baseline = {(row["model_key"], row["role"], row["condition"]): row for row in baseline_rows}
    validations: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        for role in ROLE_ORDER:
            path = ROOT / f"audits/derived/dynamic_k3_group_validation_{model_key}_{role}_v2_20260717.json"
            payload = read_json(path)
            if payload.get("status") != "PASS" or payload.get("runs_complete") != 6:
                raise RuntimeError(f"Group validation not PASS: {path}")
            validations.extend(payload["runs"])
    if len(validations) != 36:
        raise RuntimeError("Expected 36 validated runs")
    return matrix, baseline, validations


def main() -> None:
    for path in (*OUTPUTS.values(), DOCS):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite: {path}")
    matrix, baseline, validations = load_sources()
    k3_data: dict[tuple[str, str, str], dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for validation in validations:
        key = (validation["model_key"], validation["role"], validation["condition"])
        csv_path = ROOT / validation["csv_path"]
        metadata_path = ROOT / validation["metadata_path"]
        trace_path = ROOT / validation["trace_path"]
        rows = read_csv(csv_path)
        traces = read_jsonl(trace_path)
        metadata = read_json(metadata_path)
        provenance = metadata.get("provenance") or {}
        adapter_provenance = provenance.get("adapter_provenance") or {}
        if len(rows) != N or len(traces) != N:
            raise RuntimeError(f"Incomplete run: {key}")
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != N:
            raise RuntimeError(f"Duplicate cases: {key}")
        k3_data[key] = {
            "rows": rows,
            "by_id": by_id,
            "traces": traces,
            "validation": validation,
            "metadata": metadata,
        }
        gate_accept = int(validation["actual_k3"])
        result_rows.append({
            "model_key": key[0],
            "model_line": MODEL_LABELS[key[0]],
            "role": key[1],
            "role_label": ROLE_LABELS[key[1]],
            "condition": key[2],
            "condition_label": CONDITION_LABELS[key[2]],
            "run_id": validation["run_id"],
            "cases": N,
            "ema": validation["metrics"]["ema"],
            "correct": validation["metrics"]["ema_correct"],
            "esr": validation["metrics"]["esr"],
            "executable": validation["metrics"]["esr_executable"],
            "string_exact": validation["metrics"]["string_exact"],
            "normalized_exact": validation["metrics"]["normalized_exact"],
            "char_accuracy": validation["metrics"]["char_accuracy"],
            "token_accuracy": validation["metrics"]["token_accuracy"],
            "prompt_tokens_mean": validation["prompt_tokens_mean"],
            "prompt_tokens_max": validation["prompt_tokens_max"],
            "completion_tokens_mean": validation["completion_tokens_mean"],
            "completion_tokens_max": validation["completion_tokens_max"],
            "completion_limit_cases": validation["completion_limit_cases"],
            "duration_seconds": validation["duration_seconds"],
            "seconds_per_case": float(validation["duration_seconds"]) / N,
            "actual_k3": gate_accept,
            "actual_k0": int(validation["actual_k0"]),
            "gate_acceptance": gate_accept / N,
            "prompt_truncations": validation["prompt_truncations"],
            "retrieval_leakage": validation["leakage_rows"],
            "unexpected_actual_k": validation["unexpected_actual_k"],
            "config_path": validation["config_path"],
            "config_sha256": validation["config_sha256"],
            "csv_path": validation["csv_path"],
            "csv_sha256": validation["csv_sha256"],
            "metadata_path": validation["metadata_path"],
            "metadata_sha256": validation["metadata_sha256"],
            "trace_path": validation["trace_path"],
            "trace_sha256": validation["trace_sha256"],
            "log_path": validation["log_path"],
            "log_sha256": validation["log_sha256"],
            "interpreter": provenance.get("sys_executable"),
            "python_version": provenance.get("python"),
            "base_model_id": provenance.get("base_model_id"),
            "base_model_revision": provenance.get("base_model_revision"),
            "adapter_path": adapter_provenance.get("adapter_path"),
            "adapter_model_sha256": adapter_provenance.get("adapter_model_sha256"),
            "testcases_sha256": provenance.get("testcases_sha256"),
            "retrieval_artifact_sha256": json.dumps(provenance.get("retrieval_artifact_sha256"), sort_keys=True),
            "code_sha256": json.dumps(provenance.get("code_sha256"), sort_keys=True),
            "torch": provenance.get("torch"),
            "transformers": provenance.get("transformers"),
            "peft": provenance.get("peft"),
            "flash_attn": provenance.get("flash_attn"),
            "cuda_compiled_version": provenance.get("cuda_compiled_version"),
            "gpu": provenance.get("gpu"),
        })
    result_rows.sort(key=lambda row: (MODEL_ORDER.index(row["model_key"]), ROLE_ORDER.index(row["role"]), K3_ORDER.index(row["condition"])))

    baseline_data: dict[tuple[str, str, str], dict[str, Any]] = {}
    canonical_case_ids = [row["id"] for row in k3_data[("qwen2b", "base", "top3")]["rows"]]
    for key, info in baseline.items():
        rows = read_csv(ROOT / info["csv_path"])
        if len(rows) != N or [row["id"] for row in rows] != canonical_case_ids:
            raise RuntimeError(f"Invalid baseline run: {key}")
        baseline_data[key] = {"rows": rows, "by_id": {row["id"]: row for row in rows}, "info": info}

    stats_rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        for role in ROLE_ORDER:
            zero = np.array([as_bool(row["exec_match"]) for row in baseline_data[(model_key, role, "zero_shot")]["rows"]])
            for condition in K3_ORDER:
                target = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, role, condition)]["rows"]])
                stats_rows.append(paired_row(
                    "k3_vs_zero", model_key, role,
                    f"{model_key}:{role}:{condition}:vs_zero", "zero_shot", condition, zero, target,
                ))
                ref_condition = K1_REFERENCE[condition]
                k1 = np.array([as_bool(row["exec_match"]) for row in baseline_data[(model_key, role, ref_condition)]["rows"]])
                stats_rows.append(paired_row(
                    "k3_vs_k1", model_key, role,
                    f"{model_key}:{role}:{condition}:vs_{ref_condition}", ref_condition, condition, k1, target,
                ))
            for top, structure in (
                ("top3", "structure_top3"),
                ("top3_gate070", "structure_top3_gate070"),
                ("top3_gate085", "structure_top3_gate085"),
            ):
                a = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, role, top)]["rows"]])
                b = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, role, structure)]["rows"]])
                stats_rows.append(paired_row(
                    "top3_vs_structure_top3", model_key, role,
                    f"{model_key}:{role}:{structure}:vs_{top}", top, structure, a, b,
                ))
            for gate, ungated in APPROACH_UNGATED.items():
                a = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, role, ungated)]["rows"]])
                b = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, role, gate)]["rows"]])
                stats_rows.append(paired_row(
                    "ungated_vs_gate", model_key, role,
                    f"{model_key}:{role}:{gate}:vs_{ungated}", ungated, gate, a, b,
                ))
        for condition in K3_ORDER:
            a = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, "base", condition)]["rows"]])
            b = np.array([as_bool(row["exec_match"]) for row in k3_data[(model_key, "lora_v2", condition)]["rows"]])
            stats_rows.append(paired_row(
                "base_vs_lora_k3", model_key, "base_vs_lora",
                f"{model_key}:{condition}:base_vs_lora", f"base:{condition}", f"lora_v2:{condition}", a, b,
            ))
    holm_by_family(stats_rows)

    gate_rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        for role in ROLE_ORDER:
            zero = baseline_data[(model_key, role, "zero_shot")]["by_id"]
            for gate, ungated in APPROACH_UNGATED.items():
                gate_data = k3_data[(model_key, role, gate)]
                ungated_rows = k3_data[(model_key, role, ungated)]["by_id"]
                fallback = accepted = 0
                fallback_raw = fallback_sql = fallback_exec = 0
                accepted_raw = accepted_sql = accepted_exec = 0
                scores: list[float] = []
                for row in gate_data["rows"]:
                    scores.append(float(row["gate_score"]))
                    if row["gate_decision"] == "zero_shot":
                        fallback += 1
                        reference = zero[row["id"]]
                        fallback_raw += row["raw_output"] == reference["raw_output"]
                        fallback_sql += row["pred_sql"] == reference["pred_sql"]
                        fallback_exec += (
                            as_bool(row["pred_ok"]) == as_bool(reference["pred_ok"])
                            and as_bool(row["exec_match"]) == as_bool(reference["exec_match"])
                        )
                    else:
                        accepted += 1
                        reference = ungated_rows[row["id"]]
                        accepted_raw += row["raw_output"] == reference["raw_output"]
                        accepted_sql += row["pred_sql"] == reference["pred_sql"]
                        accepted_exec += (
                            as_bool(row["pred_ok"]) == as_bool(reference["pred_ok"])
                            and as_bool(row["exec_match"]) == as_bool(reference["exec_match"])
                        )
                identity = (
                    fallback_raw == fallback_sql == fallback_exec == fallback
                    and accepted_raw == accepted_sql == accepted_exec == accepted
                )
                gate_rows.append({
                    "model_key": model_key, "model_line": MODEL_LABELS[model_key], "role": role,
                    "condition": gate, "accepted_k3": accepted, "fallback_k0": fallback,
                    "gate_acceptance": accepted / N, "gate_score_min": min(scores),
                    "gate_score_mean": sum(scores) / N, "gate_score_max": max(scores),
                    "fallback_raw_matches_zero": fallback_raw, "fallback_sql_matches_zero": fallback_sql,
                    "fallback_execution_matches_zero": fallback_exec,
                    "accepted_raw_matches_ungated": accepted_raw, "accepted_sql_matches_ungated": accepted_sql,
                    "accepted_execution_matches_ungated": accepted_exec,
                    "reference_identity_status": "PASS" if identity else "FAIL",
                })
    if any(row["reference_identity_status"] != "PASS" for row in gate_rows):
        raise RuntimeError("Gate fallback or accepted-reference identity failed")

    retrieval_rows: list[dict[str, Any]] = []
    for key, data in k3_data.items():
        all_ids: list[str] = []
        all_scores: list[float] = []
        demo_dbs: list[str] = []
        same_db_demos = 0
        selected_sets: dict[str, set[str]] = {}
        for trace in data["traces"]:
            ids = [str(value) for value in trace["retrieved_ids"]]
            scores = [float(value) for value in trace["retrieved_scores"]]
            dbs = [str(value) for value in trace["retrieved_db_ids"]]
            selected_sets[str(trace["id"])] = set(ids)
            all_ids.extend(ids)
            all_scores.extend(scores)
            demo_dbs.extend(dbs)
            same_db_demos += sum(db_id == str(trace["db_id"]) for db_id in dbs)
        overlap_exact = overlap_mean = None
        if key[2].startswith("structure"):
            top_condition = key[2].replace("structure_top3", "top3")
            other = {
                str(trace["id"]): set(str(value) for value in trace["retrieved_ids"])
                for trace in k3_data[(key[0], key[1], top_condition)]["traces"]
            }
            intersections = [len(selected_sets[case_id] & other[case_id]) for case_id in selected_sets]
            overlap_exact = sum(value == 3 for value in intersections)
            overlap_mean = sum(intersections) / N
        retrieval_rows.append({
            "model_key": key[0], "model_line": MODEL_LABELS[key[0]], "role": key[1], "condition": key[2],
            "cases": N, "selected_demo_slots": len(all_ids), "unique_demo_ids": len(set(all_ids)),
            "unique_demo_databases": len(set(demo_dbs)), "similarity_min": min(all_scores),
            "similarity_mean": sum(all_scores) / len(all_scores), "similarity_max": max(all_scores),
            "same_db_demo_slots": same_db_demos, "same_db_demo_slot_rate": same_db_demos / len(all_ids),
            "most_common_demo_id": Counter(all_ids).most_common(1)[0][0],
            "most_common_demo_count": Counter(all_ids).most_common(1)[0][1],
            "top3_structure_exact_set_overlap_cases": overlap_exact,
            "top3_structure_mean_shared_demos": overlap_mean,
        })

    error_rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        for role in ROLE_ORDER:
            for condition in K3_ORDER:
                target_by_id = k3_data[(model_key, role, condition)]["by_id"]
                for reference_type, reference_condition in (("zero", "zero_shot"), ("k1", K1_REFERENCE[condition])):
                    reference_by_id = baseline_data[(model_key, role, reference_condition)]["by_id"]
                    for case_id, target in target_by_id.items():
                        reference = reference_by_id[case_id]
                        a_ok, b_ok = as_bool(reference["exec_match"]), as_bool(target["exec_match"])
                        transition = (
                            "persistent_correct" if a_ok and b_ok else
                            "repair" if not a_ok and b_ok else
                            "harm" if a_ok and not b_ok else "persistent_wrong"
                        )
                        error_rows.append({
                            "model_key": model_key, "model_line": MODEL_LABELS[model_key], "role": role,
                            "condition": condition, "reference_type": reference_type,
                            "reference_condition": reference_condition, "case_id": case_id,
                            "db_id": target["db_id"], "transition": transition,
                            "reference_execution_success": as_bool(reference["pred_ok"]),
                            "target_execution_success": as_bool(target["pred_ok"]),
                            "reference_execution_match": a_ok, "target_execution_match": b_ok,
                            "reference_error_category": classify_error(reference),
                            "target_error_category": classify_error(target),
                            "reference_completion_limit": int(float(reference["completion_tokens"])) == 256,
                            "target_completion_limit": int(float(target["completion_tokens"])) == 256,
                            "added_select_count": target["pred_sql"].lower().count("select") - reference["pred_sql"].lower().count("select"),
                            "added_join_count": target["pred_sql"].lower().count(" join ") - reference["pred_sql"].lower().count(" join "),
                        })
    summary_groups: dict[tuple[str, str, str, str], Counter[str]] = {}
    for row in error_rows:
        key = (row["model_key"], row["role"], row["condition"], row["reference_type"])
        summary_groups.setdefault(key, Counter())[row["transition"]] += 1
    error_summary = [{
        "model_key": key[0], "model_line": MODEL_LABELS[key[0]], "role": key[1],
        "condition": key[2], "reference_type": key[3],
        "persistent_correct": counts["persistent_correct"], "repairs": counts["repair"],
        "harms": counts["harm"], "persistent_wrong": counts["persistent_wrong"],
        "net_repairs": counts["repair"] - counts["harm"],
    } for key, counts in sorted(summary_groups.items())]

    write_csv_new(OUTPUTS["results"], result_rows)
    write_csv_new(OUTPUTS["stats"], stats_rows)
    write_csv_new(OUTPUTS["gate"], gate_rows)
    write_csv_new(OUTPUTS["retrieval"], retrieval_rows)
    write_csv_new(OUTPUTS["error"], error_rows)
    write_csv_new(OUTPUTS["error_summary"], error_summary)

    matrix_lines = ["| Model | Role | Condition | EMA | ESR | Correct | Limit cases |", "|---|---|---|---:|---:|---:|---:|"]
    for row in result_rows:
        matrix_lines.append(
            f"| {row['model_line']} | {row['role_label']} | {row['condition_label']} | "
            f"{100*float(row['ema']):.2f}% | {100*float(row['esr']):.2f}% | {row['correct']}/{N} | {row['completion_limit_cases']} |"
        )
    stats_lines = ["| Family | Comparison | Delta pp | Repairs | Harms | p | Holm p | 95% CI pp |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats_rows:
        stats_lines.append(
            f"| {row['family']} | {row['comparison']} | {row['delta_pp']:.2f} | {row['repairs_n01']} | "
            f"{row['harms_n10']} | {row['mcnemar_p']:.6g} | {row['holm_p']:.6g} | "
            f"[{100*row['bootstrap_ci_low']:.2f}, {100*row['bootstrap_ci_high']:.2f}] |"
        )
    tables_text = "# Dynamic k=3 thesis-ready tables\n\n## Complete 36-run matrix\n\n" + "\n".join(matrix_lines) + "\n\n## Paired comparisons\n\n" + "\n".join(stats_lines) + "\n"
    write_text_new(OUTPUTS["tables"], tables_text)

    significant = [row for row in stats_rows if row["significant_holm_0_05"]]
    text = (
        "# Dynamic k=3 thesis-ready text\n\n"
        "Die kontrollierte k=3-Erweiterung umfasst 36 additive Spider-Dev-Vollruns. Die historische "
        "48-Run-Auswertung bleibt die PRE-K3 AUTHORITATIVE BASELINE. Fuer k=3 wurde das Eingabelimit "
        "auf 4.352 Tokens gesetzt; die direkte k1-vs-k3-Interpretation bleibt zulaessig, weil der "
        "vorgeschaltete Aequivalenzcheck fuer alle historischen k1-Prompts weder Truncation noch Token- "
        "oder Hashunterschiede fand. Das unterschiedliche zulaessige Limit bleibt dennoch als "
        "methodische Einschraenkung sichtbar.\n\n"
        f"Von {len(stats_rows)} vorab definierten gepaarten Vergleichen bleiben {len(significant)} nach "
        "familienweiser Holm-Korrektur bei alpha=0,05 signifikant. Nichtsignifikanz wird nicht als "
        "Gleichheit interpretiert. Fehleruebergaenge sind regelbasiert-deskriptiv und erlauben keine "
        "kausale Zuschreibung an einzelne Demonstrationen.\n"
    )
    write_text_new(OUTPUTS["text"], text)

    matrix_table = "\n".join(matrix_lines)
    audit = f"""# Audit: Dynamic Few-Shot k=3 complete 36-run evaluation

## Status

```text
DYNAMIC-FEWSHOT-K3-COMPLETE-36-RUN-EVALUATION: PASS MIT METHODISCHEN EINSCHRANKUNGEN
K1-2048-VS-4352-PROMPT-EQUIVALENCE: PASS
FULL RUNS: 36/36
CASES PER RUN: 1032
PROMPT TRUNCATIONS: 0
RETRIEVAL LEAKAGE: 0
UNEXPECTED ACTUAL_K: 0
FALLBACK IDENTITY: PASS
EXISTING PRE-K3 FILES MODIFIED: NEIN
RERUN REQUIRED: NEIN
```

## Scope and provenance

The extension used only `.venv_flash`, batch size 1, greedy decoding, 4,352 input tokens,
256 new tokens, and the frozen 6,960-example retrieval index. All six model/role groups
passed the additive validator-v2. The first validator report for Qwen 2B Base failed only
because a 1e-12 floating-point aggregation tolerance rejected Char/Token Accuracy differences
below 2e-8. Validator-v2 uses an explicit absolute tolerance of 1e-7; no generation was repeated.

## Methodological limitation

K3 uses a higher allowed input ceiling than the historical k1 runs. The comparison is technically
controlled because {read_json(K1_EQUIVALENCE)['prompt_rows']} k1 prompt materializations had zero
truncations and zero token-ID/hash differences between ceilings. The ceiling difference is still
reported and the original 48 runs remain the authoritative pre-k3 baseline.

## Complete results

{matrix_table}

## Statistical families

Holm correction was performed separately for k3-vs-zero (36), k3-vs-k1 (36),
Base-vs-LoRA-k3 (18), Top3-vs-Structure-Top3 (18), and ungated-vs-gates (24).
Every comparison uses the exact two-sided McNemar test and 10,000 paired bootstrap resamples.

## Gate identity

All fallback outputs match their frozen Zero-Shot references, and all accepted outputs match the
corresponding ungated k3 references, for raw output, extracted SQL, execution status, and EMA.

## Error and robustness analysis

The case-level transition file reports repairs, harms, persistent-correct and persistent-wrong
cases against both Zero Shot and matching k1. Error labels are deterministic execution/output
categories; they are descriptive and not causal.
"""
    write_text_new(OUTPUTS["audit"], audit)

    DOCS.mkdir(parents=True, exist_ok=False)
    docs_files: dict[str, str] = {
        "README.md": "# 84-run extended analysis\n\nAdditive k=3 documentation. The frozen 48-run documentation remains authoritative as the pre-k3 baseline.\n",
        "MASTERARBEIT_COMPLETE_PROJECT_DOCUMENTATION_K3_EXTENSION_V3.md": "# Masterarbeit project documentation: k=3 extension v3\n\n" + text + "\nSee the registered audit and thesis-ready tables for all values.\n",
        "EXECUTIVE_PROJECT_SUMMARY_K3_EXTENSION_V3.md": "# Executive summary\n\n36 k=3 runs were added to the 48-run baseline, producing an 84-run extended analysis. Technical validation passed; comparisons retain the documented input-ceiling limitation.\n",
        "DYNAMIC_K1_VS_K3_METHODS_AND_RESULTS.md": "# Dynamic k1 versus k3\n\nK1 prompts are token- and hash-identical under 2,048 and 4,352 ceilings. K3 uses three full-schema demonstrations or binary zero-shot fallback.\n\n" + "\n".join(matrix_lines),
        "UPDATED_RESEARCH_QUESTIONS_RESULTS_MATRIX_K3_V3.md": "# Updated research questions\n\nK3 adds controlled evidence on demonstration count, gating, structure reranking, and Base/LoRA interaction. Statistical evidence is in the paired-comparison register.\n",
        "UPDATED_THESIS_CHAPTER_WRITING_BLUEPRINT_K3_V3.md": "# Thesis writing blueprint\n\n1. Preserve the 48-run baseline.\n2. Introduce k3 as an additive extension.\n3. Report all 36 runs.\n4. Separate confirmatory-looking paired tests from exploratory error analysis.\n5. State the input-ceiling limitation.\n",
        "K3_EXTENSION_TIMELINE_AND_DECISION_ADDENDUM.md": "# Timeline addendum\n\nThe initial 2,048-token preflight blocked on prompt truncation. A documented decision raised only the k3 input ceiling to 4,352. A full preflight then passed before generation.\n",
        "K3_EXTENSION_LIMITATIONS_AND_COMPARABILITY.md": "# Limitations and comparability\n\nK3 permits a larger input ceiling. Historical k1 prompts do not change under that ceiling, as established by the 37,152-row equivalence audit. Cross-family prompt templates remain model-native and therefore class-B comparisons across Qwen and Llama.\n",
    }
    for name, content in docs_files.items():
        write_text_new(DOCS / name, content)
    updated_results: list[dict[str, Any]] = []
    for row in read_csv(BASELINE_RESULTS):
        updated_results.append({"analysis_generation": "PRE-K3 AUTHORITATIVE BASELINE", **row})
    for row in result_rows:
        updated_results.append({"analysis_generation": "K3 EXTENSION", **row})
    write_csv_new(DOCS / "UPDATED_AUTHORITATIVE_RESULTS_REGISTRY_K3_V3.csv", updated_results)
    artifact_rows = [{"artifact": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in OUTPUTS.values() if path.exists() and path != OUTPUTS["manifest"]]
    write_csv_new(DOCS / "UPDATED_AUTHORITATIVE_ARTIFACT_REGISTRY_K3_V3.csv", artifact_rows)
    write_csv_new(DOCS / "UPDATED_CLAIMS_EVIDENCE_SOURCE_MATRIX_K3_V3.csv", [
        {"claim": "K1 prompt equivalence", "status": "PASS", "evidence": str(K1_EQUIVALENCE.relative_to(ROOT))},
        {"claim": "36 k3 runs complete", "status": "PASS", "evidence": str(OUTPUTS["results"].relative_to(ROOT))},
        {"claim": "Gate fallback identity", "status": "PASS", "evidence": str(OUTPUTS["gate"].relative_to(ROOT))},
    ])
    write_csv_new(DOCS / "UPDATED_THESIS_TABLE_AND_FIGURE_REGISTER_K3_V3.csv", [
        {"type": "table", "title": "Complete k3 results", "source": str(OUTPUTS["results"].relative_to(ROOT))},
        {"type": "table", "title": "Paired k1/k3 statistics", "source": str(OUTPUTS["stats"].relative_to(ROOT))},
        {"type": "table", "title": "Gate analysis", "source": str(OUTPUTS["gate"].relative_to(ROOT))},
        {"type": "table", "title": "Error transitions", "source": str(OUTPUTS["error_summary"].relative_to(ROOT))},
    ])

    generated = [path for key, path in OUTPUTS.items() if key != "manifest"]
    generated.extend(path for path in DOCS.iterdir() if path.is_file())
    manifest = {
        "status": "PASS_WITH_METHODOLOGICAL_LIMITATIONS",
        "analysis_type": "84-RUN EXTENDED ANALYSIS",
        "pre_k3_baseline": "48 runs unchanged",
        "authoritative_environment": str(ROOT / ".venv_flash/bin/python"),
        "runs": 36,
        "cases_per_run": N,
        "max_input_tokens": 4352,
        "max_new_tokens": 256,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "holm_families": {family: sum(row["family"] == family for row in stats_rows) for family in sorted({row["family"] for row in stats_rows})},
        "k1_prompt_equivalence": read_json(K1_EQUIVALENCE),
        "run_ids": [row["run_id"] for row in result_rows],
        "runs_detail": result_rows,
        "fallback_identity": gate_rows,
        "rerun_required": False,
        "generated_files": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(generated)},
        "analysis_script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve())},
    }
    write_text_new(OUTPUTS["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    doc_manifest = {
        "status": "COMPLETE",
        "analysis": "84-RUN EXTENDED ANALYSIS",
        "pre_k3_baseline_preserved": True,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(DOCS.iterdir()) if path.is_file()},
        "technical_manifest": str(OUTPUTS["manifest"].relative_to(ROOT)),
    }
    write_text_new(DOCS / "PROJECT_DOCUMENTATION_K3_EXTENSION_MANIFEST.json", json.dumps(doc_manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"], "runs": len(result_rows), "statistics": len(stats_rows),
        "gate_identity": "PASS", "docs": str(DOCS.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
