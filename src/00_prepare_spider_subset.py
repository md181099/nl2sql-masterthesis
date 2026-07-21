#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "question",
    "gold_sql",
    "db_id",
    "db_path",
    "schema_prompt",
    "source_split",
    "source_idx",
)


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at index {idx} in {path}")
    return data


def deduplicate_split(items: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    dropped = 0

    for source_idx, row in enumerate(items):
        question = str(row.get("question", "")).strip()
        query = str(row.get("query", "")).strip()
        db_id = str(row.get("db_id", "")).strip()

        if not question or not query or not db_id:
            raise ValueError(
                f"Invalid row in split={split_name} at index={source_idx}: "
                "required keys are question/query/db_id"
            )

        key = (db_id, question, query)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(
            {
                "question": question,
                "query": query,
                "db_id": db_id,
                "source_idx": source_idx,
                "source_split": split_name,
            }
        )

    return out, dropped


def sample_rows(rows: list[dict[str, Any]], n: int, rng: random.Random, label: str) -> list[dict[str, Any]]:
    if n < 1:
        raise ValueError(f"Requested sample size for {label} must be >= 1")
    if n > len(rows):
        raise ValueError(f"Requested {n} examples for {label}, but only {len(rows)} available")
    selected_idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in selected_idx]


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def build_schema_prompt(db_file: Path) -> str:
    if not db_file.exists():
        raise FileNotFoundError(f"Missing SQLite DB file: {db_file}")

    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name;"
            ).fetchall()
        ]

        lines: list[str] = ["Database schema:"]
        for idx, table_name in enumerate(tables):
            pragma_table_sql = f"PRAGMA table_info({_quote_ident(table_name)});"
            table_info_rows = cur.execute(pragma_table_sql).fetchall()
            cols = [str(col[1]) for col in table_info_rows]
            pk_cols = [
                str(col[1])
                for col in sorted(table_info_rows, key=lambda x: int(x[5]) if x[5] is not None else 0)
                if int(col[5]) > 0
            ]

            pragma_fk_sql = f"PRAGMA foreign_key_list({_quote_ident(table_name)});"
            fk_rows = cur.execute(pragma_fk_sql).fetchall()

            lines.append(f"Table: {table_name}")
            lines.append(f"Columns: {', '.join(cols)}")
            if pk_cols:
                lines.append(f"Primary key: {', '.join(pk_cols)}")
            else:
                lines.append("Primary key: none")

            if fk_rows:
                lines.append("Foreign keys:")
                for fk in fk_rows:
                    # PRAGMA foreign_key_list columns: id, seq, table, from, to, on_update, on_delete, match
                    ref_table = str(fk[2])
                    from_col = str(fk[3])
                    to_col = str(fk[4])
                    lines.append(f"- {table_name}.{from_col} -> {ref_table}.{to_col}")

            if idx != len(tables) - 1:
                lines.append("")
    finally:
        conn.close()

    schema = "\n".join(lines).strip()
    if not schema:
        raise RuntimeError(f"Could not build schema prompt from DB: {db_file}")
    return schema


def canonical_example_key(db_id: str, question: str, sql: str) -> tuple[str, str, str]:
    return (db_id.strip(), question.strip(), ensure_semicolon(sql))


def normalize_question_for_overlap(question: str) -> str:
    return " ".join(question.strip().lower().split())


def normalize_sql_for_overlap(sql: str) -> str:
    return " ".join(ensure_semicolon(sql).strip().lower().split())


def make_record(
    item: dict[str, Any],
    split_name: str,
    spider_dir: Path,
    schema_cache: dict[str, str],
) -> dict[str, Any]:
    db_id = str(item["db_id"]).strip()
    question = str(item["question"]).strip()
    gold_sql = ensure_semicolon(str(item["query"]))
    source_idx = int(item["source_idx"])

    db_abs = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if db_id not in schema_cache:
        schema_cache[db_id] = build_schema_prompt(db_abs)

    db_rel = Path("data") / "spider" / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
    id_prefix = "TRAIN" if split_name == "train_spider" else "DEV"
    record_id = f"SPIDER_{id_prefix}_{source_idx:06d}"

    return {
        "id": record_id,
        "question": question,
        "gold_sql": gold_sql,
        "db_id": db_id,
        "db_path": str(db_rel).replace("\\", "/"),
        "schema_prompt": schema_cache[db_id],
        "source_split": split_name,
        "source_idx": source_idx,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_split_rows(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_split: str,
    expected_id_prefix: str,
    project_root: Path,
) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Validation failed for {expected_split}: expected {expected_count}, found {len(rows)}"
        )

    ids: set[str] = set()
    seen_keys: set[tuple[str, str, str]] = set()
    for idx, row in enumerate(rows):
        missing = [k for k in REQUIRED_FIELDS if k not in row]
        if missing:
            raise RuntimeError(
                f"Validation failed for {expected_split} row={idx}: missing keys {missing}"
            )

        for key in ("id", "question", "gold_sql", "db_id", "db_path", "schema_prompt", "source_split"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise RuntimeError(
                    f"Validation failed for {expected_split} row={idx}: key '{key}' is empty"
                )

        if not isinstance(row["source_idx"], int) or row["source_idx"] < 0:
            raise RuntimeError(
                f"Validation failed for {expected_split} row={idx}: invalid source_idx={row['source_idx']}"
            )

        if row["source_split"] != expected_split:
            raise RuntimeError(
                f"Validation failed for {expected_split} row={idx}: source_split={row['source_split']}"
            )

        if not row["id"].startswith(expected_id_prefix):
            raise RuntimeError(
                f"Validation failed for {expected_split} row={idx}: id prefix mismatch ({row['id']})"
            )

        if row["id"] in ids:
            raise RuntimeError(
                f"Validation failed for {expected_split}: duplicate id detected ({row['id']})"
            )
        ids.add(row["id"])

        db_path = Path(row["db_path"])
        db_abs = db_path if db_path.is_absolute() else project_root / db_path
        if not db_abs.exists():
            raise RuntimeError(
                f"Validation failed for {expected_split} row={idx}: db_path does not exist ({db_abs})"
            )

        key = canonical_example_key(row["db_id"], row["question"], row["gold_sql"])
        if key in seen_keys:
            raise RuntimeError(
                f"Validation failed for {expected_split}: duplicate example key detected ({key})"
            )
        seen_keys.add(key)


def validate_cross_split_overlap(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> None:
    train_keys = {
        canonical_example_key(r["db_id"], r["question"], r["gold_sql"])
        for r in train_rows
    }
    dev_keys = {
        canonical_example_key(r["db_id"], r["question"], r["gold_sql"])
        for r in dev_rows
    }
    overlap = train_keys & dev_keys
    if overlap:
        raise RuntimeError(
            f"Validation failed: found {len(overlap)} identical examples across train/dev"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible Spider subset and map to project JSONL format."
    )
    parser.add_argument(
        "--spider_dir",
        type=str,
        default="data/spider/spider_data",
        help="Path to Spider dataset root (contains train_spider.json, dev.json, tables.json, database/).",
    )
    parser.add_argument("--out_train", type=str, default="data/traincases.jsonl")
    parser.add_argument("--out_test", type=str, default="data/testcases.jsonl")
    parser.add_argument(
        "--out_test_full",
        type=str,
        default=None,
        help=(
            "Optional path for full non-overlapping Spider dev set JSONL. "
            "If unset, no additional full dev file is written."
        ),
    )
    parser.add_argument("--manifest", type=str, default="data/spider/subset_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_dev", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    spider_dir = resolve_path(project_root, args.spider_dir)
    out_train = resolve_path(project_root, args.out_train)
    out_test = resolve_path(project_root, args.out_test)
    out_test_full = (
        resolve_path(project_root, args.out_test_full) if args.out_test_full else None
    )
    manifest_path = resolve_path(project_root, args.manifest)

    train_json = spider_dir / "train_spider.json"
    dev_json = spider_dir / "dev.json"
    tables_json = spider_dir / "tables.json"
    database_dir = spider_dir / "database"

    if not spider_dir.exists():
        raise FileNotFoundError(f"Spider directory not found: {spider_dir}")
    if not tables_json.exists():
        raise FileNotFoundError(f"Missing tables.json: {tables_json}")
    if not database_dir.exists():
        raise FileNotFoundError(f"Missing database directory: {database_dir}")

    train_raw = load_json_array(train_json)
    dev_raw = load_json_array(dev_json)

    train_dedup, train_dropped = deduplicate_split(train_raw, "train_spider")
    dev_dedup, dev_dropped = deduplicate_split(dev_raw, "dev")

    rng = random.Random(args.seed)
    train_selected = sample_rows(train_dedup, args.n_train, rng, "train_spider")

    train_key_set = {
        canonical_example_key(x["db_id"], x["question"], x["query"])
        for x in train_selected
    }
    train_question_set = {
        normalize_question_for_overlap(x["question"])
        for x in train_selected
    }
    train_sql_set = {
        normalize_sql_for_overlap(x["query"])
        for x in train_selected
    }
    dev_pool_after_exact_filter = [
        x
        for x in dev_dedup
        if canonical_example_key(x["db_id"], x["question"], x["query"]) not in train_key_set
    ]
    dev_pool_after_question_filter = [
        x
        for x in dev_pool_after_exact_filter
        if normalize_question_for_overlap(x["question"]) not in train_question_set
    ]
    dev_non_overlap_pool = [
        x
        for x in dev_pool_after_question_filter
        if normalize_sql_for_overlap(x["query"]) not in train_sql_set
    ]
    dev_selected = sample_rows(dev_non_overlap_pool, args.n_dev, rng, "dev")

    schema_cache: dict[str, str] = {}
    train_rows = [
        make_record(item, "train_spider", spider_dir, schema_cache)
        for item in train_selected
    ]
    dev_rows = [
        make_record(item, "dev", spider_dir, schema_cache)
        for item in dev_selected
    ]
    dev_full_rows: list[dict[str, Any]] = []
    if out_test_full is not None:
        dev_full_rows = [
            make_record(item, "dev", spider_dir, schema_cache)
            for item in dev_non_overlap_pool
        ]

    validate_split_rows(
        train_rows,
        expected_count=args.n_train,
        expected_split="train_spider",
        expected_id_prefix="SPIDER_TRAIN_",
        project_root=project_root,
    )
    validate_split_rows(
        dev_rows,
        expected_count=args.n_dev,
        expected_split="dev",
        expected_id_prefix="SPIDER_DEV_",
        project_root=project_root,
    )
    validate_cross_split_overlap(train_rows, dev_rows)
    if out_test_full is not None:
        validate_split_rows(
            dev_full_rows,
            expected_count=len(dev_non_overlap_pool),
            expected_split="dev",
            expected_id_prefix="SPIDER_DEV_",
            project_root=project_root,
        )
        validate_cross_split_overlap(train_rows, dev_full_rows)

    write_jsonl(out_train, train_rows)
    write_jsonl(out_test, dev_rows)
    if out_test_full is not None:
        write_jsonl(out_test_full, dev_full_rows)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_counts": {
            "train_spider": args.n_train,
            "dev": args.n_dev,
        },
        "source_paths": {
            "spider_dir": str(spider_dir),
            "train_spider_json": str(train_json),
            "dev_json": str(dev_json),
            "tables_json": str(tables_json),
            "database_dir": str(database_dir),
        },
        "deduplication": {
            "train_spider": {
                "before": len(train_raw),
                "after": len(train_dedup),
                "removed": train_dropped,
            },
            "dev": {
                "before": len(dev_raw),
                "after": len(dev_dedup),
                "removed": dev_dropped,
            },
        },
        "sampling": {
            "train_spider_selected": len(train_selected),
            "dev_pool_after_exact_overlap_filter": len(dev_pool_after_exact_filter),
            "dev_pool_after_question_overlap_filter": len(dev_pool_after_question_filter),
            "dev_pool_after_train_overlap_filter": len(dev_non_overlap_pool),
            "dev_selected": len(dev_selected),
            "dev_full_selected": len(dev_full_rows) if out_test_full is not None else None,
        },
        "output_paths": {
            "traincases_jsonl": str(out_train),
            "testcases_jsonl": str(out_test),
            "testcases_spider_dev_full_jsonl": str(out_test_full) if out_test_full else None,
            "subset_manifest_json": str(manifest_path),
        },
        "validation": {
            "train_count_ok": len(train_rows) == args.n_train,
            "dev_count_ok": len(dev_rows) == args.n_dev,
            "dev_full_count_ok": (
                len(dev_full_rows) == len(dev_non_overlap_pool) if out_test_full is not None else None
            ),
            "required_fields_ok": True,
            "db_paths_exist_ok": True,
            "cross_split_overlap_ok": True,
            "all_passed": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prepared Spider subset successfully.")
    print(
        f"train_spider: raw={len(train_raw)}, dedup={len(train_dedup)}, "
        f"sampled={len(train_rows)}"
    )
    print(
        f"dev: raw={len(dev_raw)}, dedup={len(dev_dedup)}, "
        f"pool_non_overlap={len(dev_non_overlap_pool)}, sampled={len(dev_rows)}"
    )
    print(f"Wrote traincases: {out_train}")
    print(f"Wrote testcases: {out_test}")
    if out_test_full is not None:
        print(f"Wrote full dev testcases: {out_test_full}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
