#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
OUTPUT_MATRIX = (
    ROOT
    / "audits/derived/dynamic_k3_qwen9b_remaining_sqltimeout900_v3_config_matrix_20260718.csv"
)
TIMEOUT_SECONDS = 900


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            differences[key] = {"source": before.get(key), "new": after.get(key)}
    return differences


def main() -> None:
    if OUTPUT_MATRIX.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUTPUT_MATRIX}")

    with SOURCE_MATRIX.open(newline="", encoding="utf-8") as handle:
        source_rows = [
            row for row in csv.DictReader(handle) if row["model_key"] == "qwen9b"
        ]
    if len(source_rows) != 12:
        raise RuntimeError(f"Expected 12 Qwen-9B configs, found {len(source_rows)}")

    output_rows: list[dict[str, Any]] = []
    for row in source_rows:
        source_path = ROOT / row["new_k3_config"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        new = dict(source)
        new["run_output_prefix"] = f"{source['run_output_prefix']}_sqltimeout900v3"
        new["execution_timeout_seconds"] = TIMEOUT_SECONDS

        target_name = f"{source_path.stem}_sqltimeout900v3.json"
        target_path = source_path.with_name(target_name)
        if target_path.exists():
            raise RuntimeError(f"Refusing to overwrite: {target_path}")
        target_path.write_text(
            json.dumps(new, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        differences = semantic_diff(source, new)
        expected_fields = {"execution_timeout_seconds", "run_output_prefix"}
        status = "PASS" if set(differences) == expected_fields else "FAIL"
        output_rows.append(
            {
                "model_key": row["model_key"],
                "model_line": row["model_line"],
                "role": row["role"],
                "condition": row["condition"],
                "source_config": str(source_path.relative_to(ROOT)),
                "source_config_sha256": sha256(source_path),
                "timeout_config": str(target_path.relative_to(ROOT)),
                "timeout_config_sha256": sha256(target_path),
                "source_run_output_prefix": source["run_output_prefix"],
                "timeout_run_output_prefix": new["run_output_prefix"],
                "execution_timeout_seconds": TIMEOUT_SECONDS,
                "changed_fields": json.dumps(differences, ensure_ascii=False, sort_keys=True),
                "allowed_diff_only": status,
            }
        )

    fieldnames = list(output_rows[0])
    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MATRIX.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        json.dumps(
            {
                "status": "PASS",
                "configs_created": len(output_rows),
                "execution_timeout_seconds": TIMEOUT_SECONDS,
                "matrix": str(OUTPUT_MATRIX.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
