#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
CHANGE_REPORT = ROOT / "audits/derived/dynamic_k3_maxin4352_config_changes_20260717.csv"
OLD_NAME_TOKEN = "maxinput2048"
NEW_NAME_TOKEN = "maxin4352"
NEW_MAX_INPUT = 4352


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_fields(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"reference": left.get(key), "new": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def csv_text(rows: list[dict[str, Any]]) -> str:
    require(bool(rows), "Cannot serialize an empty CSV")
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_replace_scoped(path: Path, text: str) -> None:
    allowed = {
        MATRIX.resolve(),
        CHANGE_REPORT.resolve(),
    }
    require(path.resolve() in allowed, f"Refusing out-of-scope write: {path}")
    if path == CHANGE_REPORT:
        require(not path.exists(), f"Refusing to overwrite change report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".maxin4352.tmp")
    require(not temporary.exists(), f"Temporary output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    require(MATRIX.is_file(), f"Missing k3 matrix: {MATRIX}")
    require(not CHANGE_REPORT.exists(), f"Change report already exists: {CHANGE_REPORT}")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 36, f"Expected 36 k3 configs, found {len(rows)}")

    plans: list[dict[str, Any]] = []
    targets: set[Path] = set()
    allowed_reference_diffs = {
        "k",
        "expected_model_revision",
        "results_dir",
        "run_output_prefix",
        "fewshot_gate_mode",
        "fewshot_gate_debug",
        "max_input_tokens",
    }
    for row in rows:
        old_path = (ROOT / row["new_k3_config"]).resolve()
        require(old_path.is_file(), f"Missing k3 config: {old_path}")
        require(old_path.parent == (ROOT / "configs").resolve(), f"Out-of-scope config: {old_path}")
        require(old_path.name.count(OLD_NAME_TOKEN) == 1, f"Unexpected old name: {old_path.name}")
        new_path = old_path.with_name(old_path.name.replace(OLD_NAME_TOKEN, NEW_NAME_TOKEN))
        require(new_path != old_path, f"Config name did not change: {old_path}")
        require(not new_path.exists(), f"Target config already exists: {new_path}")
        require(new_path not in targets, f"Duplicate target config: {new_path}")
        targets.add(new_path)

        old_config = json.loads(old_path.read_text(encoding="utf-8"))
        require(sha256(old_path) == row["config_sha256"], f"Matrix hash drift: {old_path}")
        require(old_config.get("k") == 3, f"Not a k3 config: {old_path}")
        require(old_config.get("max_input_tokens") == 2048, f"Unexpected old input limit: {old_path}")
        require(old_config.get("max_new_tokens") == 256, f"Unexpected output limit: {old_path}")
        require(old_config.get("results_dir") == "results/k3_extension_20260717", f"Unexpected result directory: {old_path}")

        new_config = dict(old_config)
        new_config["max_input_tokens"] = NEW_MAX_INPUT
        old_prefix = str(old_config.get("run_output_prefix", ""))
        require("k3" in old_prefix and NEW_NAME_TOKEN not in old_prefix, f"Unexpected output prefix: {old_prefix}")
        new_config["run_output_prefix"] = f"{old_prefix}_{NEW_NAME_TOKEN}"
        migration_diff = changed_fields(old_config, new_config)
        require(
            set(migration_diff) == {"max_input_tokens", "run_output_prefix"},
            f"Migration is not limited to the approved fields: {new_path}",
        )

        reference_path = (ROOT / row["reference_k1_config"]).resolve()
        require(reference_path.is_file(), f"Missing k1 reference: {reference_path}")
        require(sha256(reference_path) == row["reference_config_sha256"], f"Reference hash drift: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_diff = changed_fields(reference, new_config)
        require(
            set(reference_diff) <= allowed_reference_diffs,
            f"Disallowed reference differences for {new_path}: {sorted(set(reference_diff) - allowed_reference_diffs)}",
        )
        plans.append(
            {
                "row": row,
                "old_path": old_path,
                "new_path": new_path,
                "old_config": old_config,
                "new_config": new_config,
                "old_sha256": sha256(old_path),
                "migration_diff": migration_diff,
                "reference_diff": reference_diff,
                "unchanged_fields": sorted(
                    key
                    for key in reference
                    if key in new_config and reference[key] == new_config[key]
                ),
            }
        )

    matrix_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for plan in plans:
        old_path = plan["old_path"]
        new_path = plan["new_path"]
        old_path.rename(new_path)
        new_path.write_text(
            json.dumps(plan["new_config"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        new_hash = sha256(new_path)
        row = dict(plan["row"])
        row["new_k3_config"] = str(new_path.relative_to(ROOT))
        row["changed_fields"] = json.dumps(plan["reference_diff"], ensure_ascii=False, sort_keys=True)
        row["unchanged_fields"] = json.dumps(plan["unchanged_fields"], ensure_ascii=False)
        row["config_sha256"] = new_hash
        row["one_factor_extension_status"] = "PASS_MAXIN4352"
        row["previous_max_input_tokens"] = 2048
        row["configured_max_input_tokens"] = NEW_MAX_INPUT
        row["pre_migration_config_sha256"] = plan["old_sha256"]
        row["maxin4352_migration_changed_fields"] = json.dumps(
            plan["migration_diff"], ensure_ascii=False, sort_keys=True
        )
        row["maxin4352_migration_status"] = "PASS"
        matrix_rows.append(row)
        change_rows.append(
            {
                "config_path": str(new_path.relative_to(ROOT)),
                "model_key": row["model_key"],
                "role": row["role"],
                "condition": row["condition"],
                "previous_max_input_tokens": 2048,
                "configured_max_input_tokens": NEW_MAX_INPUT,
                "max_new_tokens": plan["new_config"]["max_new_tokens"],
                "pre_migration_config_sha256": plan["old_sha256"],
                "config_sha256": new_hash,
                "changed_fields": json.dumps(plan["migration_diff"], ensure_ascii=False, sort_keys=True),
                "unintended_changed_fields": "[]",
                "status": "PASS",
            }
        )

    write_replace_scoped(MATRIX, csv_text(matrix_rows))
    write_replace_scoped(CHANGE_REPORT, csv_text(change_rows))
    print(
        json.dumps(
            {
                "status": "PASS",
                "configs_renamed": len(plans),
                "configs_updated": len(plans),
                "max_input_tokens": NEW_MAX_INPUT,
                "max_new_tokens": 256,
                "config_matrix": str(MATRIX.relative_to(ROOT)),
                "change_report": str(CHANGE_REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
