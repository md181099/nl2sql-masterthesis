#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMPARE_FIELDS = ["raw_output", "pred_sql", "pred_ok", "exec_match"]


def load_by_id(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV is missing required 'id' column: {path}")
        for row in reader:
            row_id = row.get("id", "")
            if not row_id:
                raise ValueError(f"CSV contains a row without id: {path}")
            if row_id in rows:
                raise ValueError(f"CSV contains duplicate id={row_id!r}: {path}")
            rows[row_id] = row
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two NL2SQL evaluation CSVs for exact output equality.")
    parser.add_argument("--csv_a", required=True, help="First evaluation CSV.")
    parser.add_argument("--csv_b", required=True, help="Second evaluation CSV.")
    parser.add_argument("--max_diffs", type=int, default=20, help="Maximum differing ids to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    rows_a = load_by_id(csv_a)
    rows_b = load_by_id(csv_b)

    ids_a = set(rows_a)
    ids_b = set(rows_b)
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)

    diffs: list[tuple[str, list[str]]] = []
    for row_id in sorted(ids_a & ids_b):
        changed_fields = [
            field
            for field in COMPARE_FIELDS
            if rows_a[row_id].get(field, "") != rows_b[row_id].get(field, "")
        ]
        if changed_fields:
            diffs.append((row_id, changed_fields))

    print(f"csv_a={csv_a}")
    print(f"csv_b={csv_b}")
    print(f"rows_a={len(rows_a)}")
    print(f"rows_b={len(rows_b)}")
    print(f"missing_in_b={len(only_a)}")
    print(f"missing_in_a={len(only_b)}")
    print(f"rows_with_differences={len(diffs)}")

    if only_a:
        print("first_missing_in_b=" + ", ".join(only_a[: args.max_diffs]))
    if only_b:
        print("first_missing_in_a=" + ", ".join(only_b[: args.max_diffs]))

    for row_id, changed_fields in diffs[: args.max_diffs]:
        print(f"diff id={row_id} fields={','.join(changed_fields)}")

    return 1 if only_a or only_b or diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
