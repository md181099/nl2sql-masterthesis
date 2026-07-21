#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits/audit_dynamic_fewshot_k3_execution_blocked_by_unbounded_sql_20260718.md"
STATUS = ROOT / "audits/derived/dynamic_k3_execution_status_20260718.csv"
ADDENDUM = ROOT / "docs/final_project_documentation_20260717_k3_extension_v2/K3_EXECUTION_BLOCKER_ADDENDUM_20260718.md"
INVENTORY = ROOT / "audits/derived/dynamic_k3_completed_and_partial_run_inventory_20260718.csv"
MANIFEST = ROOT / "audits/dynamic_fewshot_k3_execution_blocked_by_unbounded_sql_manifest_20260718.json"
GROUPS = (
    ("qwen2b", "base"),
    ("qwen2b", "lora_v2"),
    ("llama3b", "base"),
    ("llama3b", "lora_v2"),
)
PARTIAL_CSV = ROOT / "results/k3_extension_20260717/run_k3_qwen9b_base_top3_maxin4352_20260718_011245.csv"
PARTIAL_TRACE = ROOT / "results/k3_extension_20260717/retrieval_traces/run_k3_qwen9b_base_top3_maxin4352_20260718_011245_retrieval_traces.jsonl"
PARTIAL_LOG = ROOT / "logs/k3_extension_20260717/run_k3_qwen9b_base_top3_maxin4352.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if INVENTORY.exists() or MANIFEST.exists():
        raise RuntimeError("Refusing to overwrite blocker inventory or manifest")
    complete: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    validation_sources: list[dict[str, Any]] = []
    for model_key, role in GROUPS:
        validation_path = ROOT / f"audits/derived/dynamic_k3_group_validation_{model_key}_{role}_v2_20260717.json"
        payload = read_json(validation_path)
        if payload.get("status") != "PASS" or payload.get("runs_complete") != 6:
            raise RuntimeError(f"Validation not PASS: {validation_path}")
        validation_sources.append({
            "path": str(validation_path.relative_to(ROOT)),
            "sha256": sha256(validation_path),
        })
        for run in payload["runs"]:
            record = {
                "state": "COMPLETE_AND_GROUP_VALIDATED",
                "model_key": run["model_key"],
                "role": run["role"],
                "condition": run["condition"],
                "run_id": run["run_id"],
                "cases": run["cases"],
                "config_path": run["config_path"],
                "config_sha256": run["config_sha256"],
                "csv_path": run["csv_path"],
                "csv_sha256": run["csv_sha256"],
                "metadata_path": run["metadata_path"],
                "metadata_sha256": run["metadata_sha256"],
                "trace_path": run["trace_path"],
                "trace_sha256": run["trace_sha256"],
                "log_path": run["log_path"],
                "log_sha256": run["log_sha256"],
                "ema": run["metrics"]["ema"],
                "esr": run["metrics"]["esr"],
                "prompt_truncations": run["prompt_truncations"],
                "retrieval_leakage": run["leakage_rows"],
                "unexpected_actual_k": run["unexpected_actual_k"],
            }
            inventory.append(record)
            complete.append(record)
    partial_rows = count_csv(PARTIAL_CSV)
    partial_trace_rows = sum(1 for line in PARTIAL_TRACE.open(encoding="utf-8") if line.strip())
    inventory.append({
        "state": "PARTIAL_BLOCKED_NOT_FOR_ANALYSIS",
        "model_key": "qwen9b",
        "role": "base",
        "condition": "top3",
        "run_id": PARTIAL_CSV.stem,
        "cases": partial_rows,
        "config_path": "configs/eval_qwen35_9b_base_dynamic_bge_large_top3_k3_full_schema_maxin4352_full_aliasnames.json",
        "config_sha256": sha256(ROOT / "configs/eval_qwen35_9b_base_dynamic_bge_large_top3_k3_full_schema_maxin4352_full_aliasnames.json"),
        "csv_path": str(PARTIAL_CSV.relative_to(ROOT)),
        "csv_sha256": sha256(PARTIAL_CSV),
        "metadata_path": "",
        "metadata_sha256": "",
        "trace_path": str(PARTIAL_TRACE.relative_to(ROOT)),
        "trace_sha256": sha256(PARTIAL_TRACE),
        "log_path": str(PARTIAL_LOG.relative_to(ROOT)),
        "log_sha256": sha256(PARTIAL_LOG),
        "ema": "",
        "esr": "",
        "prompt_truncations": "not_released",
        "retrieval_leakage": "not_released",
        "unexpected_actual_k": "not_released",
    })
    write_csv_new(INVENTORY, inventory)
    generated = (AUDIT, STATUS, ADDENDUM, INVENTORY, Path(__file__).resolve())
    manifest = {
        "status": "BLOCKED_BY_UNBOUNDED_SQL",
        "requested_runs": 36,
        "complete_group_validated_runs": 24,
        "partial_runs": 1,
        "not_started_runs": 11,
        "groups_passed": 4,
        "groups_expected": 6,
        "k1_prompt_equivalence": {
            "path": "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json",
            "sha256": sha256(ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json"),
            "status": "PASS",
        },
        "preflight": {
            "path": "audits/audit_dynamic_fewshot_k3_implementation_and_preflight_20260717.md",
            "sha256": sha256(ROOT / "audits/audit_dynamic_fewshot_k3_implementation_and_preflight_20260717.md"),
        },
        "config_matrix": {
            "path": "audits/derived/dynamic_k3_config_matrix_20260717.csv",
            "sha256": sha256(ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"),
        },
        "group_validations": validation_sources,
        "complete_runs": complete,
        "partial_run": inventory[-1],
        "blocker": {
            "case_id": "SPIDER_DEV_000481",
            "db_id": "wta_1",
            "question": "Find the number of matches happened in each year.",
            "symptom": "More than two hours at 100% CPU, 0% GPU, no new result row",
            "root_cause_class": "generated SQLite query without progress-handler deadline",
            "termination": "SIGINT ineffective during native SQLite call; SIGTERM used",
            "orchestrator_exit": 1,
            "child_exit": -15,
        },
        "methodological_decision_required": True,
        "decision_options": [
            "uniform new timeout/row-cap policy and full 36-run k3 rerun",
            "retain unbounded policy and accept non-termination risk",
        ],
        "complete_analysis_released": False,
        "statistics_released": False,
        "error_analysis_released": False,
        "84_run_documentation_released": False,
        "existing_pre_k3_files_modified": False,
        "generated_files": {str(path.relative_to(ROOT)): sha256(path) for path in generated},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "complete_runs": len(complete),
        "partial_rows": partial_rows,
        "partial_trace_rows": partial_trace_rows,
        "inventory": str(INVENTORY.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
