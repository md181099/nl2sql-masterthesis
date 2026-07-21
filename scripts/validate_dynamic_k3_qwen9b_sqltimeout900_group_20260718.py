#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    ROOT
    / "audits/derived/dynamic_k3_qwen9b_remaining_sqltimeout900_config_matrix_20260718.csv"
)
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
RESULTS = ROOT / "results/k3_extension_20260717"
LOGS = ROOT / "logs/k3_extension_20260718_sqltimeout900"
EXPECTED_INTERPRETER = str(ROOT / ".venv_flash/bin/python")
EXPECTED_INDEX = str((ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15").resolve())
EXPECTED_GATES = {
    "top3_gate070": (480, 552),
    "top3_gate085": (7, 1025),
    "structure_top3_gate070": (450, 582),
    "structure_top3_gate085": (6, 1026),
}
METRIC_ABS_TOLERANCE = 1e-7
CONDITION_ORDER = {
    "top3": 0,
    "top3_gate070": 1,
    "top3_gate085": 2,
    "structure_top3": 3,
    "structure_top3_gate070": 4,
    "structure_top3_gate085": 5,
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
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_json_list(value: str) -> list[Any]:
    parsed = json.loads(value)
    require(isinstance(parsed, list), f"Expected list, found {type(parsed).__name__}")
    return parsed


def find_run(prefix: str) -> tuple[Path, Path, Path, Path]:
    csvs = sorted(RESULTS.glob(f"{prefix}_*.csv"))
    require(len(csvs) == 1, f"Expected exactly one CSV for {prefix}, found {len(csvs)}")
    csv_path = csvs[0]
    stem = csv_path.stem
    metadata = RESULTS / f"{stem}_metadata.json"
    trace = RESULTS / "retrieval_traces" / f"{stem}_retrieval_traces.jsonl"
    log = LOGS / f"{prefix}.log"
    for path in (metadata, trace, log):
        require(path.is_file(), f"Missing run artifact: {path}")
    return csv_path, metadata, trace, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("base", "lora_v2"))
    args = parser.parse_args()
    output = ROOT / f"audits/derived/dynamic_k3_qwen9b_{args.role}_sqltimeout900_group_validation_20260718.json"
    require(not output.exists(), f"Refusing to overwrite group validation: {output}")
    matrix = [
        row
        for row in read_csv(MATRIX)
        if row["model_key"] == "qwen9b" and row["role"] == args.role
    ]
    matrix.sort(key=lambda row: CONDITION_ORDER[row["condition"]])
    require(len(matrix) == 6, f"Expected six configs, found {len(matrix)}")
    testcases = read_jsonl(TESTCASES)
    testcase_ids = [str(row["id"]) for row in testcases]
    require(len(testcase_ids) == len(set(testcase_ids)) == 1032, "Testcase integrity failure")

    run_rows: list[dict[str, Any]] = []
    group_failures: list[str] = []
    for matrix_row in matrix:
        condition = matrix_row["condition"]
        config_path = ROOT / matrix_row["timeout_config"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prefix = str(config["run_output_prefix"])
        csv_path, metadata_path, trace_path, log_path = find_run(prefix)
        rows = read_csv(csv_path)
        traces = read_jsonl(trace_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        if sha256(config_path) != matrix_row["timeout_config_sha256"]:
            failures.append("config_hash")
        ids = [row["id"] for row in rows]
        trace_ids = [str(row["id"]) for row in traces]
        if len(rows) != 1032 or ids != testcase_ids:
            failures.append("csv_cases_or_order")
        if len(traces) != 1032 or trace_ids != testcase_ids:
            failures.append("trace_cases_or_order")
        if len(set(ids)) != 1032:
            failures.append("duplicate_case_ids")
        prompt_truncations = sum(int(float(row["prompt_tokens"])) > 4352 for row in rows)
        if prompt_truncations:
            failures.append("prompt_truncation")
        if any(row.get("run_max_input_tokens") != "4352" for row in rows):
            failures.append("csv_max_input")
        if any(row.get("run_max_new_tokens") != "256" for row in rows):
            failures.append("csv_max_new")
        if any(row.get("run_generation_batch_size") != "1" for row in rows):
            failures.append("csv_batch")
        if any(row.get("run_config_path") != matrix_row["timeout_config"] for row in rows):
            failures.append("csv_config_path")
        if any(row.get("run_execution_timeout_seconds") != "900.0" for row in rows):
            failures.append("csv_execution_timeout")
        duplicate_demo_rows = 0
        leakage_rows = 0
        actual_k_counts: Counter[int] = Counter()
        for row, trace in zip(rows, traces):
            retrieved_ids = [str(value) for value in parse_json_list(row["retrieved_ids"])]
            duplicate_demo_rows += int(len(retrieved_ids) != 3 or len(set(retrieved_ids)) != 3)
            leakage_rows += int(str(trace.get("leakage_status", "")).lower() != "pass")
            if config.get("fewshot_gate_enabled"):
                decision = row.get("gate_decision")
                if decision == "fewshot":
                    actual_k_counts[3] += 1
                elif decision == "zero_shot":
                    actual_k_counts[0] += 1
                else:
                    actual_k_counts[-1] += 1
            else:
                actual_k_counts[3] += 1
        if duplicate_demo_rows:
            failures.append("duplicate_or_missing_demos")
        if leakage_rows:
            failures.append("retrieval_leakage")
        unexpected_actual_k = sum(value for key, value in actual_k_counts.items() if key not in {0, 3})
        if unexpected_actual_k:
            failures.append("unexpected_actual_k")
        if condition in EXPECTED_GATES:
            expected_k3, expected_k0 = EXPECTED_GATES[condition]
            if (actual_k_counts[3], actual_k_counts[0]) != (expected_k3, expected_k0):
                failures.append("gate_distribution")
        elif (actual_k_counts[3], actual_k_counts[0]) != (1032, 0):
            failures.append("ungated_actual_k")
        provenance = metadata.get("provenance") or {}
        if provenance.get("sys_executable") != EXPECTED_INTERPRETER:
            failures.append("interpreter")
        if provenance.get("cuda_available") is not True or provenance.get("gpu") != "NVIDIA L40S":
            failures.append("gpu")
        for package in (
            "torch",
            "transformers",
            "peft",
            "trl",
            "accelerate",
            "datasets",
            "flash_attn",
            "sentence_transformers",
            "faiss",
        ):
            if not provenance.get(package):
                failures.append(f"package_{package}")
        if provenance.get("config_sha256") != matrix_row["timeout_config_sha256"]:
            failures.append("metadata_config_hash")
        if provenance.get("testcases_sha256") != sha256(TESTCASES):
            failures.append("metadata_testset_hash")
        retrieval_hashes = provenance.get("retrieval_artifact_sha256") or {}
        if set(retrieval_hashes) != {"index.faiss", "metadata.jsonl", "manifest.json"}:
            failures.append("retrieval_hashes")
        if metadata.get("run_max_input_tokens") != 4352 or metadata.get("run_max_new_tokens") != 256:
            failures.append("metadata_token_limits")
        if metadata.get("run_execution_timeout_seconds") != 900.0:
            failures.append("metadata_execution_timeout")
        if metadata.get("execution_timeout_seconds") != 900.0:
            failures.append("summary_execution_timeout")
        if provenance.get("runner_variant") != "dynamic_k3_sqltimeout_v2":
            failures.append("runner_variant")
        if metadata.get("total_testcases") != 1032:
            failures.append("metadata_cases")
        if Path(str(metadata.get("retrieval_index_path", ""))).resolve() != Path(EXPECTED_INDEX):
            failures.append("metadata_retrieval_index")
        metric_keys = (
            "execution_success_rate",
            "execution_match_accuracy",
            "string_exact_match",
            "normalized_exact_match",
            "char_accuracy_avg",
            "token_accuracy_avg",
        )
        if any(not isinstance(metadata.get(key), (int, float)) or not math.isfinite(float(metadata[key])) for key in metric_keys):
            failures.append("metadata_metrics")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if re.search(
            r"Traceback|CUDA out of memory|\bKilled\b|^ERROR|\| ERROR \|",
            log_text,
            flags=re.MULTILINE,
        ):
            failures.append("log_error")
        if "Run metadata written to:" not in log_text:
            failures.append("missing_completion_log")

        ema_correct = sum(parse_bool(row["exec_match"]) for row in rows)
        esr_executable = sum(parse_bool(row["pred_ok"]) for row in rows)
        metrics = {
            "ema": ema_correct / 1032,
            "ema_correct": ema_correct,
            "esr": esr_executable / 1032,
            "esr_executable": esr_executable,
            "string_exact": sum(float(row["string_exact"]) for row in rows) / 1032,
            "normalized_exact": sum(float(row["normalized_exact"]) for row in rows) / 1032,
            "char_accuracy": sum(float(row["char_accuracy"]) for row in rows) / 1032,
            "token_accuracy": sum(float(row["token_accuracy"]) for row in rows) / 1032,
        }
        metadata_metric_pairs = {
            "ema": "execution_match_accuracy",
            "esr": "execution_success_rate",
            "string_exact": "string_exact_match",
            "normalized_exact": "normalized_exact_match",
            "char_accuracy": "char_accuracy_avg",
            "token_accuracy": "token_accuracy_avg",
        }
        metric_mismatches = sum(
            abs(metrics[key] - float(metadata[metadata_key])) > METRIC_ABS_TOLERANCE
            for key, metadata_key in metadata_metric_pairs.items()
        )
        if metric_mismatches:
            failures.append("metric_reproduction")
        completion_limit_cases = sum(int(float(row["completion_tokens"])) == 256 for row in rows)
        gold_timeout_rows = sum(
            row.get("gold_error", "").startswith("SQLExecutionTimeout(") for row in rows
        )
        pred_timeout_rows = sum(
            row.get("pred_error", "").startswith("SQLExecutionTimeout(") for row in rows
        )
        if gold_timeout_rows:
            failures.append("gold_sql_timeout")
        if metadata.get("gold_sql_timeout_total") != gold_timeout_rows:
            failures.append("gold_timeout_count")
        if metadata.get("pred_sql_timeout_total") != pred_timeout_rows:
            failures.append("pred_timeout_count")
        run_id = csv_path.stem
        run_rows.append(
            {
                "model_key": "qwen9b",
                "role": args.role,
                "condition": condition,
                "run_id": run_id,
                "config_path": matrix_row["timeout_config"],
                "config_sha256": matrix_row["timeout_config_sha256"],
                "csv_path": str(csv_path.relative_to(ROOT)),
                "csv_sha256": sha256(csv_path),
                "metadata_path": str(metadata_path.relative_to(ROOT)),
                "metadata_sha256": sha256(metadata_path),
                "trace_path": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha256(trace_path),
                "log_path": str(log_path.relative_to(ROOT)),
                "log_sha256": sha256(log_path),
                "cases": len(rows),
                "prompt_tokens_mean": sum(float(row["prompt_tokens"]) for row in rows) / 1032,
                "prompt_tokens_max": max(int(float(row["prompt_tokens"])) for row in rows),
                "completion_tokens_mean": sum(float(row["completion_tokens"]) for row in rows) / 1032,
                "completion_tokens_max": max(int(float(row["completion_tokens"])) for row in rows),
                "completion_limit_cases": completion_limit_cases,
                "prompt_truncations": prompt_truncations,
                "leakage_rows": leakage_rows,
                "duplicate_demo_rows": duplicate_demo_rows,
                "unexpected_actual_k": unexpected_actual_k,
                "actual_k3": actual_k_counts[3],
                "actual_k0": actual_k_counts[0],
                "metric_mismatches": metric_mismatches,
                "execution_timeout_seconds": metadata.get("execution_timeout_seconds"),
                "gold_sql_timeout_total": metadata.get("gold_sql_timeout_total"),
                "pred_sql_timeout_total": metadata.get("pred_sql_timeout_total"),
                "metrics": metrics,
                "duration_seconds": metadata.get("duration_seconds"),
                "failures": failures,
                "status": "PASS" if not failures else "FAIL",
            }
        )
        group_failures.extend(f"{condition}:{failure}" for failure in failures)

    result = {
        "status": "PASS" if not group_failures else "FAIL",
        "model_key": "qwen9b",
        "role": args.role,
        "runs_complete": sum(row["status"] == "PASS" for row in run_rows),
        "expected_runs": 6,
        "cases_per_run": 1032,
        "group_failures": group_failures,
        "runs": run_rows,
        "source": {
            "config_matrix": str(MATRIX.relative_to(ROOT)),
            "config_matrix_sha256": sha256(MATRIX),
            "testcases": str(TESTCASES.relative_to(ROOT)),
            "testcases_sha256": sha256(TESTCASES),
            "validator_sha256": sha256(Path(__file__).resolve()),
            "metric_absolute_tolerance": METRIC_ABS_TOLERANCE,
            "runner_variant": "dynamic_k3_sqltimeout_v2",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "model_key": "qwen9b",
        "role": args.role,
        "runs_complete": result["runs_complete"],
        "failures": group_failures,
        "output": str(output.relative_to(ROOT)),
    }, indent=2))
    if group_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
