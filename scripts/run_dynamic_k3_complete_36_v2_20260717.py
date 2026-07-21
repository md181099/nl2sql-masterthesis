#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv_flash/bin/python"
RUNNER = ROOT / "src/06_batch_run_dynamic_k3_v1.py"
VALIDATOR = ROOT / "scripts/validate_dynamic_k3_run_group_v2_20260717.py"
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
K1_EQUIVALENCE = ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json"
RESULTS = ROOT / "results/k3_extension_20260717"
LOGS = ROOT / "logs/k3_extension_20260717"
GROUPS = (
    ("qwen2b", "base"),
    ("qwen2b", "lora_v2"),
    ("llama3b", "base"),
    ("llama3b", "lora_v2"),
    ("qwen9b", "base"),
    ("qwen9b", "lora_v2"),
)
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


def read_matrix() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_run_state(prefix: str) -> str:
    csvs = sorted(RESULTS.glob(f"{prefix}_*.csv")) if RESULTS.exists() else []
    metadata = sorted(RESULTS.glob(f"{prefix}_*_metadata.json")) if RESULTS.exists() else []
    traces = (
        sorted((RESULTS / "retrieval_traces").glob(f"{prefix}_*_retrieval_traces.jsonl"))
        if (RESULTS / "retrieval_traces").exists()
        else []
    )
    log = LOGS / f"{prefix}.log"
    if not csvs and not metadata and not traces and not log.exists():
        return "missing"
    if len(csvs) == len(metadata) == len(traces) == 1 and log.is_file():
        try:
            meta = json.loads(metadata[0].read_text(encoding="utf-8"))
            if meta.get("total_testcases") == 1032 and "Run metadata written to:" in log.read_text(
                encoding="utf-8", errors="replace"
            ):
                return "complete"
        except Exception:
            pass
    return "partial_or_ambiguous"


def main() -> None:
    require(Path(sys.executable).absolute() == PYTHON, f"Use authoritative interpreter: {PYTHON}")
    equivalence = json.loads(K1_EQUIVALENCE.read_text(encoding="utf-8"))
    require(equivalence.get("status") == "PASS", "K1 prompt equivalence did not pass")
    require(equivalence.get("k1_vs_k3_comparison_permitted") is True, "K1-vs-K3 comparison not released")
    matrix = read_matrix()
    require(len(matrix) == 36, "Expected 36 k3 configs")
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for group_index, (model_key, role) in enumerate(GROUPS, start=1):
        group_rows = [
            row for row in matrix if row["model_key"] == model_key and row["role"] == role
        ]
        group_rows.sort(key=lambda row: CONDITION_ORDER[row["condition"]])
        require(len(group_rows) == 6, f"Expected six configs for {model_key}/{role}")
        print(f"=== GROUP {group_index}/6 START {model_key}/{role} ===", flush=True)
        validation_path = ROOT / f"audits/derived/dynamic_k3_group_validation_{model_key}_{role}_v2_20260717.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            require(validation.get("status") == "PASS", f"Existing group validation is not PASS: {validation_path}")
            print(f"=== GROUP {group_index}/6 ALREADY VALIDATED {model_key}/{role} ===", flush=True)
            continue
        for run_index, row in enumerate(group_rows, start=1):
            config_path = ROOT / row["new_k3_config"]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            prefix = str(config["run_output_prefix"])
            state = existing_run_state(prefix)
            require(state != "partial_or_ambiguous", f"Partial or ambiguous artifacts for {prefix}")
            if state == "complete":
                print(f"[{group_index}/6 {run_index}/6] COMPLETE, SKIP {prefix}", flush=True)
                continue
            log_path = LOGS / f"{prefix}.log"
            print(f"[{group_index}/6 {run_index}/6] START {row['new_k3_config']}", flush=True)
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.run(
                    [str(PYTHON), str(RUNNER), "--config", row["new_k3_config"]],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            require(process.returncode == 0, f"Evaluation failed with exit {process.returncode}: {log_path}")
            print(f"[{group_index}/6 {run_index}/6] FINISH {prefix}", flush=True)
        subprocess.run(
            [
                str(PYTHON),
                str(VALIDATOR),
                "--model-key",
                model_key,
                "--role",
                role,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        print(f"=== GROUP {group_index}/6 VALIDATED {model_key}/{role} ===", flush=True)
    print("=== ALL 36 K3 RUNS COMPLETE AND GROUP-VALIDATED ===", flush=True)


if __name__ == "__main__":
    main()
