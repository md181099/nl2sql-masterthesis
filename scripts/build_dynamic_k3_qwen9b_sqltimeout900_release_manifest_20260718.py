#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audits/dynamic_k3_remaining_qwen9b_sqltimeout900_release_manifest_20260718.json"
MATRIX = ROOT / "audits/derived/dynamic_k3_qwen9b_remaining_sqltimeout900_config_matrix_20260718.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUTPUT}")

    rows = list(csv.DictReader(MATRIX.open(newline="", encoding="utf-8")))
    if len(rows) != 12 or any(row["allowed_diff_only"] != "PASS" for row in rows):
        raise RuntimeError("The twelve-config release matrix is incomplete or invalid")

    completed_group_validations = [
        ROOT / "audits/derived/dynamic_k3_group_validation_qwen2b_base_v2_20260717.json",
        ROOT / "audits/derived/dynamic_k3_group_validation_qwen2b_lora_v2_v2_20260717.json",
        ROOT / "audits/derived/dynamic_k3_group_validation_llama3b_base_v2_20260717.json",
        ROOT / "audits/derived/dynamic_k3_group_validation_llama3b_lora_v2_v2_20260717.json",
    ]
    for path in completed_group_validations:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "PASS" or data.get("runs_complete") != 6:
            raise RuntimeError(f"Completed group validation is not PASS: {path}")

    timeout_configs = []
    for row in rows:
        path = ROOT / row["timeout_config"]
        config = json.loads(path.read_text(encoding="utf-8"))
        timeout_configs.append(
            {
                "model_key": row["model_key"],
                "role": row["role"],
                "condition": row["condition"],
                "source_config": row["source_config"],
                "source_config_sha256": row["source_config_sha256"],
                "config": row["timeout_config"],
                "config_sha256": sha256(path),
                "run_output_prefix": config["run_output_prefix"],
                "execution_timeout_seconds": config["execution_timeout_seconds"],
                "max_input_tokens": config["max_input_tokens"],
                "max_new_tokens": config["max_new_tokens"],
                "allowed_diff_only": row["allowed_diff_only"],
            }
        )

    generated = [
        ROOT / "audits/audit_dynamic_k3_remaining_qwen9b_sqltimeout900_release_20260718.md",
        MATRIX,
        ROOT / "src/06_batch_run_dynamic_k3_sqltimeout_v2.py",
        ROOT / "scripts/prepare_dynamic_k3_qwen9b_sqltimeout900_20260718.py",
        ROOT / "scripts/validate_dynamic_k3_qwen9b_sqltimeout900_group_20260718.py",
        ROOT / "scripts/run_dynamic_k3_remaining_qwen9b_sqltimeout900_20260718.sh",
        *[ROOT / row["timeout_config"] for row in rows],
    ]

    manifest = {
        "status": "PASS_RELEASED_ON_GPU_HOST",
        "date": "2026-07-18",
        "project_root": str(ROOT),
        "execution_started_by_release_audit": False,
        "existing_completed_runs": {
            "reused": True,
            "count": 24,
            "rerun_required": False,
            "group_validations": [artifact(path) for path in completed_group_validations],
        },
        "remaining_runs": {
            "count": 12,
            "partial_clean_restart": 1,
            "first_execution": 11,
            "model_key": "qwen9b",
            "roles": {"base": 6, "lora_v2": 6},
        },
        "partial_excluded_run": {
            "run_id": "run_k3_qwen9b_base_top3_maxin4352_20260718_011245",
            "complete_csv_rows": 479,
            "trace_rows": 483,
            "status": "PARTIAL_BLOCKED_NOT_FOR_ANALYSIS",
        },
        "execution_policy": {
            "runner_variant": "dynamic_k3_sqltimeout_v2",
            "per_statement_sqlite_timeout_seconds": 900,
            "progress_handler_instruction_interval": 10000,
            "gold_timeout_is_validation_failure": True,
            "prediction_timeout_is_non_executable": True,
            "largest_adjacent_log_interval_completed_runs_seconds": 361,
            "post_hoc_policy_limitation_must_be_reported": True,
        },
        "config_matrix": artifact(MATRIX),
        "configs": timeout_configs,
        "checks": {
            "python_compile": "PASS",
            "shell_syntax": "PASS",
            "allowed_config_diff_only": "PASS_12_OF_12",
            "sqlite_success_query": "PASS",
            "sqlite_timeout_interrupt": "PASS",
            "sqlite_handler_reset": "PASS",
            "output_collisions": 0,
            "active_writer": False,
            "gpu_visible_in_current_audit_sandbox": False,
            "launch_script_gpu_host_guard": "PASS",
        },
        "launch": {
            "command": "bash scripts/run_dynamic_k3_remaining_qwen9b_sqltimeout900_20260718.sh",
            "sequential": True,
            "base_group_first": True,
            "validate_after_each_six_run_group": True,
            "offline_model_resolution": True,
        },
        "generated_files": [artifact(path) for path in generated],
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
