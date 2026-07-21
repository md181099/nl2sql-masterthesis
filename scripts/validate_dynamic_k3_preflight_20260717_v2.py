#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
PROMPTS = ROOT / "audits/derived/dynamic_k3_prompt_preflight_20260717.csv"
RETRIEVAL = ROOT / "audits/derived/dynamic_k3_retrieval_selection_validation_20260717.csv"
SUMMARY = ROOT / "audits/derived/dynamic_k3_prompt_preflight_summary_20260717.json"
CONTEXT = ROOT / "audits/derived/dynamic_k3_model_context_capacity_20260717.json"
CONFIG_CHANGES = ROOT / "audits/derived/dynamic_k3_maxin4352_config_changes_20260717.csv"
BASELINE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
OUT_IDENTITY = ROOT / "audits/derived/dynamic_k3_k1_reference_identity_20260717_v2.csv"
OUT_VALIDATION = ROOT / "audits/derived/dynamic_k3_preflight_validation_20260717_v2.json"
SOURCE_CONDITIONS = {
    "top1",
    "top1_gate070",
    "top1_gate085",
    "structure",
    "structure_gate070",
    "structure_gate085",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_scoped(path: Path, text: str) -> None:
    require(path in {OUT_IDENTITY, OUT_VALIDATION}, f"Refusing out-of-scope write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".maxin4352.tmp")
    require(not temporary.exists(), f"Temporary output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    matrix = read_csv(MATRIX)
    prompts = read_csv(PROMPTS)
    retrieval = read_csv(RETRIEVAL)
    baseline = [row for row in read_csv(BASELINE) if row["condition"] in SOURCE_CONDITIONS]
    require(len(matrix) == 36, "Config matrix is not 36 rows")
    require(len(prompts) == 36 * 1032, "Prompt preflight row count mismatch")
    require(len(retrieval) == 1032, "Retrieval validation row count mismatch")
    require(len(baseline) == 36, "Baseline reference count mismatch")

    retrieval_by_id = {row["case_id"]: row for row in retrieval}
    identity_rows: list[dict[str, Any]] = []
    for reference in baseline:
        trace_path = ROOT / reference["trace_path"]
        traces = read_jsonl(trace_path)
        require(len(traces) == 1032, f"Reference trace incomplete: {trace_path}")
        id_matches = 0
        score_matches = 0
        case_order_matches = 0
        mismatched_ids: list[str] = []
        mismatched_scores: list[str] = []
        is_structure = reference["condition"].startswith("structure")
        id_key = "structure_top3_demo_ids" if is_structure else "top3_demo_ids"
        score_key = "structure_original_bge_scores" if is_structure else "top3_scores"
        for index, trace in enumerate(traces):
            case_id = str(trace["id"])
            expected = retrieval_by_id[case_id]
            expected_id = str(json.loads(expected[id_key])[0])
            expected_score = float(json.loads(expected[score_key])[0])
            trace_ids = trace.get("retrieved_ids", [])
            trace_scores = trace.get("retrieved_scores", [])
            actual_id = str(trace_ids[0]) if trace_ids else ""
            actual_score = float(trace_scores[0]) if trace_scores else float("nan")
            if case_id == retrieval[index]["case_id"]:
                case_order_matches += 1
            if actual_id == expected_id:
                id_matches += 1
            else:
                mismatched_ids.append(case_id)
            if abs(actual_score - expected_score) <= 1e-5:
                score_matches += 1
            else:
                mismatched_scores.append(case_id)
        status = (
            "PASS"
            if id_matches == score_matches == case_order_matches == 1032
            else "FAIL"
        )
        identity_rows.append(
            {
                "model_key": reference["model_key"],
                "role": reference["role"],
                "source_condition": reference["condition"],
                "run_id": reference["run_id"],
                "trace_path": reference["trace_path"],
                "trace_sha256": sha256(trace_path),
                "cases": len(traces),
                "case_order_matches": case_order_matches,
                "first_demo_id_matches": id_matches,
                "first_original_bge_score_matches_tolerance_1e-5": score_matches,
                "mismatched_demo_case_ids": json.dumps(mismatched_ids),
                "mismatched_score_case_ids": json.dumps(mismatched_scores),
                "status": status,
            }
        )

    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(identity_rows[0]))
    writer.writeheader()
    writer.writerows(identity_rows)
    write_scoped(OUT_IDENTITY, output.getvalue())

    matrix_hash_failures = 0
    diff_failures = 0
    allowed_diffs = {
        "k",
        "expected_model_revision",
        "results_dir",
        "run_output_prefix",
        "fewshot_gate_mode",
        "fewshot_gate_debug",
        "max_input_tokens",
    }
    for row in matrix:
        config_path = ROOT / row["new_k3_config"]
        matrix_hash_failures += int(sha256(config_path) != row["config_sha256"])
        diff_failures += int(not set(json.loads(row["changed_fields"])) <= allowed_diffs)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        diff_failures += int("maxin4352" not in config_path.name)
        diff_failures += int(config.get("max_input_tokens") != 4352)
        diff_failures += int(config.get("max_new_tokens") != 256)
        diff_failures += int("maxin4352" not in str(config.get("run_output_prefix", "")))

    actual_k_counts = Counter(row["actual_k"] for row in prompts)
    unexpected_actual_k = sum(row["actual_k"] not in {"0", "3"} for row in prompts)
    invalid_prompts = sum(row["status"] != "PASS" for row in prompts)
    prompt_truncations = sum(int(row["would_truncate"]) for row in prompts)
    maximum_prompt_tokens = max(int(row["prompt_tokens"]) for row in prompts)
    input_limit_failures = sum(int(row["max_input_tokens"] != "4352") for row in prompts)
    duplicate_demo_rows = 0
    gate_math_failures = 0
    for row in prompts:
        ids = json.loads(row["demo_ids"])
        duplicate_demo_rows += int(bool(ids) and len(ids) != len(set(ids)))
        if row["gate_threshold"]:
            scores = [float(value) for value in json.loads(row["similarities"])]
            gate_score = float(row["gate_set_score"])
            threshold = float(row["gate_threshold"])
            expected_fallback = min(scores) < threshold
            gate_math_failures += int(abs(gate_score - min(scores)) > 1e-12)
            gate_math_failures += int(bool(int(row["fallback"])) != expected_fallback)

    role_prompt_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    fallback_prompt_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in prompts:
        role_prompt_groups[(row["model_key"], row["condition"], row["case_id"])].add(
            row["prompt_sha256"]
        )
        if row["fallback"] == "1":
            fallback_prompt_groups[(row["model_key"], row["role"], row["case_id"])].add(
                row["prompt_sha256"]
            )
    base_lora_prompt_mismatches = sum(len(values) != 1 for values in role_prompt_groups.values())
    fallback_prompt_mismatches = sum(len(values) != 1 for values in fallback_prompt_groups.values())
    retrieval_leakage = sum(int(row["target_or_dev_leakage"]) for row in retrieval)
    retrieval_distinct_failures = sum(
        row["top3_distinct"] != "1" or row["structure_distinct"] != "1"
        for row in retrieval
    )
    identity_failures = sum(row["status"] != "PASS" for row in identity_rows)
    expected_gates = {
        "top3_gate070": {"3": 480, "0": 552},
        "top3_gate085": {"3": 7, "0": 1025},
        "structure_top3_gate070": {"3": 450, "0": 582},
        "structure_top3_gate085": {"3": 6, "0": 1026},
    }
    gate_distribution_failures = 0
    gate_distributions: list[dict[str, Any]] = []
    grouped_gate_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in prompts:
        if row["condition"] in expected_gates:
            grouped_gate_counts[(row["model_key"], row["role"], row["condition"])][
                row["actual_k"]
            ] += 1
    for (model_key, role, condition), counts in sorted(grouped_gate_counts.items()):
        expected = expected_gates[condition]
        status = "PASS" if dict(counts) == expected else "FAIL"
        gate_distribution_failures += int(status == "FAIL")
        gate_distributions.append(
            {
                "model_key": model_key,
                "role": role,
                "condition": condition,
                "actual_k3": counts.get("3", 0),
                "actual_k0": counts.get("0", 0),
                "expected_k3": expected["3"],
                "expected_k0": expected["0"],
                "status": status,
            }
        )
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    context = json.loads(CONTEXT.read_text(encoding="utf-8"))
    prior_selection_identity_failure = int(
        summary["retrieval"]["prior_maxin2048_selection_identity"]["status"] != "PASS"
    )
    prior_prompt_identity_failure = int(
        summary["prior_maxin2048_prompt_identity"]["status"] != "PASS"
    )
    context_capacity_failure = int(context["status"] != "PASS")
    hard_failures = {
        "matrix_hash_failures": matrix_hash_failures,
        "disallowed_config_diff_rows": diff_failures,
        "unexpected_actual_k_rows": unexpected_actual_k,
        "invalid_prompt_rows": invalid_prompts,
        "duplicate_demo_rows": duplicate_demo_rows,
        "gate_math_failures": gate_math_failures,
        "base_lora_prompt_hash_mismatches": base_lora_prompt_mismatches,
        "fallback_prompt_hash_mismatches": fallback_prompt_mismatches,
        "retrieval_leakage_rows": retrieval_leakage,
        "retrieval_distinct_failures": retrieval_distinct_failures,
        "k1_reference_identity_failures": identity_failures,
        "max_input_token_rows_not_4352": input_limit_failures,
        "gate_distribution_failures": gate_distribution_failures,
        "prior_k3_retrieval_selection_identity_failures": prior_selection_identity_failure,
        "prior_k3_prompt_identity_failures": prior_prompt_identity_failure,
        "model_context_capacity_failures": context_capacity_failure,
    }
    non_truncation_failures = sum(hard_failures.values())
    status = (
        "BLOCKED-BY-PROMPT-TRUNCATION"
        if non_truncation_failures == 0 and prompt_truncations > 0
        else "FAIL"
        if non_truncation_failures
        else "PASS"
    )
    validation = {
        "status": status,
        "full_runs_released": status == "PASS",
        "full_runs_started": False,
        "configs": len(matrix),
        "prompt_rows": len(prompts),
        "retrieval_rows": len(retrieval),
        "prompt_truncations": prompt_truncations,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "configured_max_input_tokens": 4352,
        "configured_max_new_tokens": 256,
        "actual_k_counts": dict(actual_k_counts),
        "gate_distributions": gate_distributions,
        "model_context_capacity": context,
        "hard_failures_excluding_truncation": hard_failures,
        "k1_reference_identity": {
            "references": len(identity_rows),
            "passing": len(identity_rows) - identity_failures,
            "score_tolerance": 1e-5,
            "csv_path": str(OUT_IDENTITY.relative_to(ROOT)),
            "csv_sha256": sha256(OUT_IDENTITY),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (MATRIX, PROMPTS, RETRIEVAL, SUMMARY, CONTEXT, CONFIG_CHANGES, BASELINE)
        },
    }
    write_scoped(OUT_VALIDATION, json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
