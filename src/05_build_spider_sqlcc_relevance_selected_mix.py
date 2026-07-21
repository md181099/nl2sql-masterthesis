#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sqlite3
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_NAME = "spider_train_sqlcc_relevance_selected_8k"
DEFAULT_OUTPUT = (
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_relevance_selected_8k_seed42_no_dev_overlap.jsonl"
)
DEFAULT_MANIFEST = (
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_relevance_selected_8k_seed42_no_dev_overlap_manifest.json"
)
TARGET_BUCKETS: OrderedDict[str, int] = OrderedDict(
    [
        ("hard_cases", 1800),
        ("join_decision", 1300),
        ("difficult_filters", 1700),
        ("schema_grounding", 2000),
        ("other_spider_relevant", 400),
        ("robustness_fill", 800),
    ]
)
BUCKET_LABELS = {
    "hard_cases": "Hard Cases",
    "join_decision": "Join Decision",
    "difficult_filters": "Schwierige Filter",
    "schema_grounding": "Schema Grounding",
    "other_spider_relevant": "Sonstige Spider-relevante Faelle",
    "robustness_fill": "Robustness-Fill",
}
BUCKET_OUTPUT_NAMES = {
    "hard_cases": "sqlcc_relevance_hard_cases",
    "join_decision": "sqlcc_relevance_join_decision",
    "difficult_filters": "sqlcc_relevance_difficult_filters",
    "schema_grounding": "sqlcc_relevance_schema_grounding",
    "other_spider_relevant": "sqlcc_relevance_other_spider_relevant",
    "robustness_fill": "sqlcc_relevance_robustness_fill",
}
RESERVED_WORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "on",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "like",
    "between",
    "group",
    "by",
    "having",
    "order",
    "limit",
    "asc",
    "desc",
    "distinct",
    "union",
    "intersect",
    "except",
    "count",
    "avg",
    "sum",
    "min",
    "max",
    "case",
    "when",
    "then",
    "else",
    "end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Spider Train + 8k SQLCC relevance-selected mix in "
            "Spider-like schema format."
        )
    )
    parser.add_argument("--sqlcc_path", default="data/sql_create_context/train.jsonl")
    parser.add_argument("--spider_dir", default="data/spider/spider_data")
    parser.add_argument("--spider_train_path", default="data/spider/spider_data/train_spider.json")
    parser.add_argument("--dev_reference_path", default="data/testcases_spider_dev_full.jsonl")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--sqlcc_target_size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mix_helpers(project_root: Path):
    module_path = project_root / "src" / "04_build_spider_sqlcc_complexity_mix.py"
    spec = importlib.util.spec_from_file_location("complexity_mix_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", sql or "")


def sql_without_trailing_semicolon(sql: str) -> str:
    return re.sub(r";+\s*$", "", str(sql).strip())


def starts_read_query(sql: str) -> bool:
    return bool(re.match(r"(?is)^\s*(select|with)\b", sql or ""))


def sqlite_schema_ok(context: str, sql: str) -> tuple[bool, str | None]:
    if not starts_read_query(sql):
        return False, "sql_not_select_or_with"
    query = sql_without_trailing_semicolon(sql)
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(context)
        conn.execute("EXPLAIN QUERY PLAN " + query)
        conn.close()
        return True, None
    except Exception as exc:  # SQLite gives precise no such table/column errors.
        try:
            conn.close()
        except Exception:
            pass
        return False, str(exc)


def extract_where_clause(sql: str) -> str:
    low = strip_sql_literals(sql).lower()
    match = re.search(r"\bwhere\b(.+)", low, flags=re.DOTALL)
    if match is None:
        return ""
    tail = match.group(1)
    tail = re.split(
        r"\b(group\s+by|having|order\s+by|limit|union|intersect|except)\b",
        tail,
        maxsplit=1,
        flags=re.DOTALL,
    )[0]
    return tail.strip()


def condition_count(where_clause: str) -> int:
    if not where_clause:
        return 0
    return 1 + len(re.findall(r"\b(and|or)\b", where_clause))


def is_projection_query(sql: str) -> bool:
    low = strip_sql_literals(sql).lower()
    match = re.search(r"\bselect\b(.+?)\bfrom\b", low, flags=re.DOTALL)
    if match is None:
        return False
    selected = match.group(1).strip()
    return bool(selected and selected != "*")


def clean_sql_identifier(value: str) -> str:
    value = value.strip().rstrip(",;)")
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("`") and value.endswith("`"))
        or (value.startswith("[") and value.endswith("]"))
    ):
        value = value[1:-1]
    if "." in value:
        value = value.split(".")[-1]
    return value.strip()


def query_table_names(sql: str) -> set[str]:
    low = strip_sql_literals(sql)
    names: set[str] = set()
    for match in re.finditer(
        r"(?is)\b(?:from|join)\s+("
        r"\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*"
        r")",
        low,
    ):
        name = clean_sql_identifier(match.group(1))
        if name and name.lower() not in RESERVED_WORDS:
            names.add(name.lower())
    return names


def count_only_simple(sql: str, features: dict[str, Any]) -> bool:
    low = strip_sql_literals(sql).lower()
    return bool(
        features.get("count_agg", 0)
        and not features.get("group_by")
        and not features.get("having")
        and not features.get("nested_select")
        and not features.get("compound")
        and not features.get("order_by")
        and not features.get("limit")
        and not extract_where_clause(low)
    )


def make_selection_features(row: dict[str, Any]) -> dict[str, Any]:
    sql = str(row.get("gold_sql", ""))
    low = strip_sql_literals(sql).lower()
    features = row["sql_features"]
    join_count = int(features.get("join_count", 0))
    schema_table_count = int(row.get("schema_table_count", 0))
    query_table_count = len(query_table_names(sql))
    column_count = int(row.get("schema_column_count", 0))
    where = extract_where_clause(sql)
    cond_count = condition_count(where)
    has_and = bool(re.search(r"\band\b", where))
    has_or = bool(re.search(r"\bor\b", where))
    has_range = bool(re.search(r"<=|>=|<>|!=|<|>|\bbetween\b", where))
    has_like = bool(re.search(r"\blike\b", where))
    has_in = bool(re.search(r"\bin\b", where))
    set_operation = bool(
        features.get("union") or features.get("intersect") or features.get("except")
    )
    order_limit = bool(features.get("order_by") and features.get("limit"))
    difficult_filter = bool(
        where
        and (
            cond_count >= 2
            or has_and
            or has_or
            or has_range
            or has_like
            or has_in
        )
    )
    schema_grounding = bool(
        join_count == 0
        and query_table_count <= 1
        and not set_operation
        and not features.get("nested_select")
        and column_count >= 3
        and (where or is_projection_query(sql))
    )
    no_join_negative = bool(schema_grounding and (where or column_count >= 4))
    join_positive = bool(join_count > 0)
    hard_case = bool(
        join_count > 0
        or query_table_count > 1
        or features.get("group_by")
        or features.get("having")
        or features.get("nested_select")
        or set_operation
        or order_limit
        or features.get("distinct")
        or (join_count > 0 and features.get("any_aggregation"))
        or (join_count > 0 and features.get("group_by"))
        or (join_count > 0 and features.get("having"))
    )
    aggregation_no_group = bool(features.get("any_aggregation") and not features.get("group_by"))
    filtered_min_max = bool(
        where and (int(features.get("min_agg", 0)) > 0 or int(features.get("max_agg", 0)) > 0)
    )
    filtered_count = bool(where and int(features.get("count_agg", 0)) > 0)
    other_spider_relevant = bool(
        (features.get("order_by") and not features.get("limit"))
        or aggregation_no_group
        or order_limit
        or features.get("distinct")
        or filtered_min_max
        or filtered_count
    )
    flags = {
        "hard_cases": hard_case,
        "join_decision": bool(join_positive or no_join_negative),
        "difficult_filters": difficult_filter,
        "schema_grounding": schema_grounding,
        "other_spider_relevant": other_spider_relevant,
        "robustness_fill": True,
    }
    return {
        "flags": flags,
        "join_count": join_count,
        "query_table_count": query_table_count,
        "schema_table_count": schema_table_count,
        "column_count": column_count,
        "where": bool(where),
        "condition_count": cond_count,
        "has_and": has_and,
        "has_or": has_or,
        "has_range": has_range,
        "has_like": has_like,
        "has_in": has_in,
        "set_operation": set_operation,
        "order_by_limit": order_limit,
        "aggregation_no_group": aggregation_no_group,
        "filtered_min_max": filtered_min_max,
        "filtered_count": filtered_count,
        "no_join_negative": no_join_negative,
        "join_positive": join_positive,
        "count_only_simple": count_only_simple(sql, features),
    }


def bucket_score(row: dict[str, Any], bucket: str) -> int:
    f = row["selection_features"]
    sqlf = row["sql_features"]
    if bucket == "hard_cases":
        return (
            12 * min(int(f["join_count"]), 2)
            + 8 * bool(f["query_table_count"] > 1)
            + 8 * bool(sqlf.get("group_by"))
            + 10 * bool(sqlf.get("having"))
            + 8 * bool(sqlf.get("nested_select"))
            + 8 * bool(f["set_operation"])
            + 6 * bool(f["order_by_limit"])
            + 4 * bool(sqlf.get("distinct"))
            + 5 * bool(sqlf.get("any_aggregation") and f["join_count"] > 0)
            + 3 * bool(f["condition_count"] >= 2)
            + min(int(f["column_count"]), 8)
        )
    if bucket == "join_decision":
        return (
            14 * bool(f["join_positive"])
            + 8 * bool(f["no_join_negative"])
            + 5 * min(int(f["join_count"]), 2)
            + 4 * bool(sqlf.get("any_aggregation") and f["join_count"] > 0)
            + 4 * bool(sqlf.get("group_by") and f["join_count"] > 0)
            + 5 * bool(sqlf.get("having") and f["join_count"] > 0)
            + 3 * bool(f["where"])
            + min(int(f["column_count"]), 10)
        )
    if bucket == "difficult_filters":
        return (
            6 * min(int(f["condition_count"]), 4)
            + 5 * bool(f["has_range"])
            + 5 * bool(f["has_or"])
            + 3 * bool(f["has_and"])
            + 3 * bool(f["filtered_min_max"])
            + 3 * bool(f["filtered_count"])
            + min(int(f["column_count"]), 8)
        )
    if bucket == "schema_grounding":
        return (
            5 * bool(f["where"])
            + 4 * min(int(f["column_count"]), 8)
            + 2 * bool(f["condition_count"] >= 2)
            + 2 * bool(is_projection_query(str(row.get("gold_sql", ""))))
        )
    if bucket == "other_spider_relevant":
        return (
            6 * bool(sqlf.get("order_by") and not sqlf.get("limit"))
            + 5 * bool(f["aggregation_no_group"])
            + 4 * bool(f["filtered_min_max"])
            + 4 * bool(f["filtered_count"])
            + 4 * bool(sqlf.get("distinct"))
            + 3 * bool(f["order_by_limit"])
            + min(int(f["column_count"]), 6)
        )
    if bucket == "robustness_fill":
        return (
            3 * bool(f["where"])
            + 2 * min(int(f["column_count"]), 8)
            + 2 * bool(f["condition_count"] >= 2)
            - 10 * bool(f["count_only_simple"])
        )
    raise ValueError(f"Unknown bucket: {bucket}")


def category_flags_for_manifest(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        flags = row.get("selection_features", {}).get("flags", {})
        for key, value in flags.items():
            if value:
                counts[key] += 1
    return {BUCKET_LABELS.get(key, key): counts[key] for key in TARGET_BUCKETS}


def rich_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for row in rows:
        features = row.get("sql_features") or {}
        join_count = int(features.get("join_count", 0))
        lengths.append(len(str(row.get("gold_sql", ""))))
        counts["joins"] += join_count > 0
        counts["one_join"] += join_count == 1
        counts["two_plus_joins"] += join_count >= 2
        counts["aggregations"] += bool(features.get("any_aggregation"))
        counts["group_by"] += bool(features.get("group_by"))
        counts["having"] += bool(features.get("having"))
        counts["order_by"] += bool(features.get("order_by"))
        counts["limit"] += bool(features.get("limit"))
        counts["order_by_limit"] += bool(features.get("order_by") and features.get("limit"))
        counts["distinct"] += bool(features.get("distinct"))
        counts["nested_select"] += bool(features.get("nested_select"))
        counts["set_operations"] += bool(
            features.get("union") or features.get("intersect") or features.get("except")
        )
    return {
        "rows": len(rows),
        "joins": counts["joins"],
        "one_join": counts["one_join"],
        "two_plus_joins": counts["two_plus_joins"],
        "aggregations": counts["aggregations"],
        "group_by": counts["group_by"],
        "having": counts["having"],
        "order_by": counts["order_by"],
        "limit": counts["limit"],
        "order_by_limit": counts["order_by_limit"],
        "distinct": counts["distinct"],
        "nested_select": counts["nested_select"],
        "set_operations": counts["set_operations"],
        "avg_sql_length": (sum(lengths) / len(lengths)) if lengths else None,
    }


def build_sqlcc_pool(
    *,
    helpers,
    sqlcc_path: Path,
    dev_sets: tuple[set[str], set[str], set[tuple[str, str]]],
    spider_question_set: set[str],
    spider_sql_set: set[str],
    spider_pair_set: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    dev_question_set, dev_sql_set, dev_pair_set = dev_sets
    rows: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    seen_questions: set[str] = set()
    seen_sqls: set[str] = set()
    conversion_examples: list[dict[str, str]] = []

    with sqlcc_path.open("r", encoding="utf-8") as handle:
        for source_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            sql = str(item.get("gold_sql") or item.get("answer") or "").strip()
            context = str(item.get("schema_prompt") or item.get("context") or "").strip()
            row_id = str(item.get("id") or f"SCC_TRAIN_{source_idx + 1:06d}")
            if not question or not sql or not context:
                removed["missing_required_field"] += 1
                continue

            q_norm = helpers.normalize_question(question)
            s_norm = helpers.normalize_sql(sql)
            if q_norm in dev_question_set:
                removed["dev_question_overlap"] += 1
            if s_norm in dev_sql_set:
                removed["dev_sql_overlap"] += 1
            if (q_norm, s_norm) in dev_pair_set:
                removed["dev_question_sql_pair_overlap"] += 1
            if q_norm in dev_question_set or s_norm in dev_sql_set or (q_norm, s_norm) in dev_pair_set:
                removed["dev_question_or_sql_or_pair_overlap"] += 1
                continue

            if q_norm in spider_question_set:
                removed["spider_train_question_overlap_retained"] += 1
            if s_norm in spider_sql_set:
                removed["spider_train_sql_overlap_retained"] += 1
            if (q_norm, s_norm) in spider_pair_set:
                removed["spider_train_question_sql_pair_overlap_removed"] += 1
                continue

            if q_norm in seen_questions:
                removed["duplicate_question"] += 1
                continue
            if s_norm in seen_sqls:
                removed["duplicate_sql"] += 1
                continue

            try:
                schema_prompt, table_count, column_count = helpers.convert_create_context_to_spider_schema(context)
            except Exception:
                removed["nonparseable_schema"] += 1
                continue

            ok, error = sqlite_schema_ok(context, sql)
            if not ok:
                removed["schema_or_sql_not_sqlite_valid"] += 1
                if error:
                    key = error.split(":", 1)[0].strip().lower()
                    removed[f"schema_or_sql_error::{key[:80]}"] += 1
                continue

            if "CREATE TABLE" in schema_prompt.upper():
                removed["create_table_in_harmonized_schema"] += 1
                continue

            seen_questions.add(q_norm)
            seen_sqls.add(s_norm)
            features = helpers.sql_features(sql)
            row = {
                "id": row_id,
                "source_dataset": "sql_create_context",
                "split": str(item.get("split") or "train"),
                "source_path": "data/sql_create_context/train.jsonl",
                "source_idx": source_idx,
                "question": question,
                "context": schema_prompt,
                "schema_prompt": schema_prompt,
                "gold_sql": sql,
                "schema_format": "spider_schema_harmonized_table_columns_empty_pk_fk",
                "schema_table_count": table_count,
                "schema_column_count": column_count,
                "sql_features": features,
                "selection_bucket": "unselected",
                "selection_primary_class": None,
                "source_row_sha256": helpers.sha256_text(
                    json.dumps(
                        {
                            "question": question,
                            "context": context,
                            "gold_sql": sql,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            }
            row["selection_features"] = make_selection_features(row)
            rows.append(row)
            if len(conversion_examples) < 10:
                conversion_examples.append(
                    {
                        "id": row_id,
                        "original": context,
                        "converted": schema_prompt,
                    }
                )

    stats = {
        "input_rows": sum(1 for _ in sqlcc_path.open("r", encoding="utf-8")),
        "available_after_filters": len(rows),
        "removed": dict(sorted(removed.items())),
    }
    return rows, stats, conversion_examples


def candidate_sort_key(row: dict[str, Any], bucket: str, rng_values: dict[str, float]) -> tuple[int, int, float]:
    return (
        bucket_score(row, bucket),
        int(row.get("schema_column_count", 0)),
        rng_values[str(row["id"])],
    )


def select_from_bucket(
    *,
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    bucket: str,
    target_count: int,
    rng_values: dict[str, float],
) -> tuple[list[dict[str, Any]], int]:
    candidates = [
        row
        for row in rows
        if row["id"] not in selected_ids
        and row["selection_features"]["flags"].get(bucket, False)
    ]
    candidates.sort(key=lambda row: candidate_sort_key(row, bucket, rng_values), reverse=True)
    take = candidates[:target_count]
    for row in take:
        row["selection_bucket"] = BUCKET_OUTPUT_NAMES[bucket]
        row["selection_primary_class"] = BUCKET_LABELS[bucket]
        row["selection_score"] = bucket_score(row, bucket)
        selected_ids.add(str(row["id"]))
    gap = max(0, target_count - len(take))
    return take, gap


def select_sqlcc_rows(
    *,
    rows: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_size != sum(TARGET_BUCKETS.values()):
        raise ValueError(
            f"sqlcc_target_size={target_size} does not match configured bucket sum "
            f"{sum(TARGET_BUCKETS.values())}"
        )
    rng_values = {str(row["id"]): rng.random() for row in rows}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    gaps: dict[str, int] = {}

    for bucket, target_count in TARGET_BUCKETS.items():
        bucket_rows, gap = select_from_bucket(
            rows=rows,
            selected_ids=selected_ids,
            bucket=bucket,
            target_count=target_count,
            rng_values=rng_values,
        )
        selected.extend(bucket_rows)
        if gap:
            gaps[bucket] = gap

    if len(selected) < target_size:
        remaining = [
            row
            for row in rows
            if row["id"] not in selected_ids
            and any(row["selection_features"]["flags"].get(bucket, False) for bucket in TARGET_BUCKETS)
        ]
        remaining.sort(
            key=lambda row: (
                sum(bucket_score(row, bucket) for bucket in TARGET_BUCKETS),
                int(row.get("schema_column_count", 0)),
                rng_values[str(row["id"])],
            ),
            reverse=True,
        )
        for row in remaining[: target_size - len(selected)]:
            row["selection_bucket"] = "sqlcc_relevance_fallback_fill"
            row["selection_primary_class"] = "Fallback aus naechstbester geeigneter Kategorie"
            row["selection_score"] = sum(bucket_score(row, bucket) for bucket in TARGET_BUCKETS)
            selected_ids.add(str(row["id"]))
        selected.extend(remaining[: target_size - len(selected)])

    if len(selected) != target_size:
        raise RuntimeError(f"Could not select requested SQLCC rows: {len(selected)} != {target_size}")

    selected_bucket_sizes = Counter(str(row["selection_bucket"]) for row in selected)
    stats = {
        "target_size": target_size,
        "target_bucket_sizes": {BUCKET_LABELS[k]: v for k, v in TARGET_BUCKETS.items()},
        "priority_order": [BUCKET_LABELS[k] for k in TARGET_BUCKETS],
        "available_category_flags": category_flags_for_manifest(rows),
        "selected_category_flags": category_flags_for_manifest(selected),
        "selected_bucket_sizes": dict(sorted(selected_bucket_sizes.items())),
        "bucket_gaps": {BUCKET_LABELS.get(k, k): v for k, v in gaps.items()},
        "selected_distribution": rich_distribution(selected),
    }
    return selected, stats


def leakage_counts(
    helpers,
    rows: list[dict[str, Any]],
    dev_pairs: tuple[set[str], set[str], set[tuple[str, str]]],
) -> dict[str, int]:
    dev_q, dev_s, dev_pair = dev_pairs
    q_overlap = q_norm_overlap = 0
    s_overlap = s_norm_overlap = 0
    pair_overlap = pair_norm_overlap = 0
    for row in rows:
        q = str(row.get("question", "")).strip()
        s = str(row.get("gold_sql", "")).strip()
        q_norm = helpers.normalize_question(q)
        s_norm = helpers.normalize_sql(s)
        q_overlap += q in dev_q
        q_norm_overlap += q_norm in dev_q
        s_overlap += s in dev_s
        s_norm_overlap += s_norm in dev_s
        pair_overlap += (q, s) in dev_pair
        pair_norm_overlap += (q_norm, s_norm) in dev_pair
    return {
        "question_overlap": q_overlap,
        "normalized_question_overlap": q_norm_overlap,
        "sql_overlap": s_overlap,
        "normalized_sql_overlap": s_norm_overlap,
        "question_sql_pair_overlap": pair_overlap,
        "normalized_question_sql_pair_overlap": pair_norm_overlap,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    helpers = load_mix_helpers(project_root)

    sqlcc_path = resolve_path(project_root, args.sqlcc_path)
    spider_dir = resolve_path(project_root, args.spider_dir)
    spider_train_path = resolve_path(project_root, args.spider_train_path)
    dev_reference_path = resolve_path(project_root, args.dev_reference_path)
    output_path = resolve_path(project_root, args.output_path)
    manifest_path = resolve_path(project_root, args.manifest_path)

    for input_path in (sqlcc_path, spider_train_path, dev_reference_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
    for output in (output_path, manifest_path):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    rng = random.Random(args.seed)
    build_schema_prompt = helpers.load_spider_schema_builder(project_root)
    dev_sets = helpers.load_dev_overlap_sets(dev_reference_path)
    dev_question_set, dev_sql_set, _dev_pair_set = dev_sets

    spider_rows_raw = helpers.load_json_array(spider_train_path)
    spider_rows, spider_stats = helpers.build_spider_rows(
        spider_rows_raw=spider_rows_raw,
        spider_dir=spider_dir,
        build_schema_prompt=build_schema_prompt,
        dev_question_set=dev_question_set,
        dev_sql_set=dev_sql_set,
    )
    if len(spider_rows) != 6960:
        raise RuntimeError(f"Expected 6960 strict Spider rows, got {len(spider_rows)}")

    spider_question_set = {helpers.normalize_question(str(row["question"])) for row in spider_rows}
    spider_sql_set = {helpers.normalize_sql(str(row["gold_sql"])) for row in spider_rows}
    spider_pair_set = {
        (helpers.normalize_question(str(row["question"])), helpers.normalize_sql(str(row["gold_sql"])))
        for row in spider_rows
    }

    sqlcc_pool, sqlcc_pool_stats, conversion_examples = build_sqlcc_pool(
        helpers=helpers,
        sqlcc_path=sqlcc_path,
        dev_sets=dev_sets,
        spider_question_set=spider_question_set,
        spider_sql_set=spider_sql_set,
        spider_pair_set=spider_pair_set,
    )
    sqlcc_rows, sqlcc_selection_stats = select_sqlcc_rows(
        rows=sqlcc_pool,
        target_size=args.sqlcc_target_size,
        rng=rng,
    )

    final_rows = spider_rows + sqlcc_rows
    rng.shuffle(final_rows)
    expected_total = len(spider_rows) + args.sqlcc_target_size

    validation = {
        "target_size_ok": len(final_rows) == expected_total,
        "source_counts": dict(Counter(str(row.get("source_dataset", "unknown")) for row in final_rows)),
        "dev_leakage": leakage_counts(helpers, final_rows, dev_sets),
        "duplicates": helpers.duplicate_counts(final_rows),
        "schema_validation": {
            "create_table_count": helpers.dataset_distribution(final_rows)["schema_create_table_count"],
            "missing_required_schema_labels": helpers.dataset_distribution(final_rows)[
                "missing_required_schema_labels"
            ],
            "missing_foreign_keys_label": helpers.dataset_distribution(final_rows)[
                "missing_foreign_keys_label"
            ],
        },
    }
    validation["all_passed"] = (
        validation["target_size_ok"]
        and validation["source_counts"].get("spider_train") == 6960
        and validation["source_counts"].get("sql_create_context") == args.sqlcc_target_size
        and validation["dev_leakage"]["normalized_question_overlap"] == 0
        and validation["dev_leakage"]["normalized_sql_overlap"] == 0
        and validation["dev_leakage"]["normalized_question_sql_pair_overlap"] == 0
        and validation["duplicates"]["duplicate_id"] == 0
        and validation["duplicates"]["duplicate_question_sql"] == 0
        and validation["schema_validation"]["create_table_count"] == 0
        and validation["schema_validation"]["missing_required_schema_labels"] == 0
    )
    if not validation["all_passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    helpers.write_jsonl(output_path, final_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/05_build_spider_sqlcc_relevance_selected_mix.py",
        "target_name": TARGET_NAME,
        "seed": args.seed,
        "spider_target_size": 6960,
        "sqlcc_target_size": args.sqlcc_target_size,
        "target_size": expected_total,
        "actual_size": len(final_rows),
        "source_paths": {
            "sql_create_context": str(sqlcc_path.relative_to(project_root)),
            "spider_train": str(spider_train_path.relative_to(project_root)),
            "spider_dir": str(spider_dir.relative_to(project_root)),
            "dev_reference": str(dev_reference_path.relative_to(project_root)),
        },
        "output_path": str(output_path.relative_to(project_root)),
        "sampling_strategy": (
            "Include all strict Spider-Dev-clean Spider-Train examples, then select exactly "
            "8000 SQLCC examples with prioritized error-driven buckets: Hard Cases, Join "
            "Decision, Difficult Filters, Schema Grounding, Other Spider-relevant, "
            "Robustness-Fill. Every SQLCC candidate must be Spider-schema harmonizable, "
            "SQLite schema-valid, dev-overlap-clean, deduplicated, and renderable without "
            "CREATE TABLE in the final schema prompt."
        ),
        "selection_targets": {BUCKET_LABELS[k]: v for k, v in TARGET_BUCKETS.items()},
        "priority_order": [BUCKET_LABELS[k] for k in TARGET_BUCKETS],
        "quality_filters": {
            "schema_ok": "SQLite executes original CREATE TABLE context and EXPLAIN QUERY PLAN for SQL",
            "spider_schema_harmonizable": True,
            "final_sft_compatible_precheck": True,
            "removed_if_question_matches_dev": True,
            "removed_if_sql_matches_dev": True,
            "removed_if_pair_matches_dev": True,
            "removed_if_question_sql_pair_matches_selected_spider_train": True,
            "sqlcc_internal_question_dedupe": True,
            "sqlcc_internal_sql_dedupe": True,
            "no_create_table_in_harmonized_schema": True,
        },
        "schema_harmonization": {
            "target_format": "Spider-like Table/Columns/Primary key/Foreign keys",
            "spider_schema_builder": "src/00_prepare_spider_subset.py::build_schema_prompt",
            "sqlcc_converter": "permissive CREATE TABLE parser; no PK/FK invented",
            "sqlcc_primary_key_when_absent": "empty Primary key: section",
            "sqlcc_foreign_keys_when_absent": "empty Foreign keys: section",
        },
        "spider_train": {
            **spider_stats,
            "distribution": helpers.dataset_distribution(spider_rows),
            "rich_distribution": rich_distribution(spider_rows),
        },
        "sql_create_context": {
            "pool": {
                **sqlcc_pool_stats,
                "distribution": helpers.dataset_distribution(sqlcc_pool),
                "rich_distribution": rich_distribution(sqlcc_pool),
                "available_category_flags": category_flags_for_manifest(sqlcc_pool),
            },
            "selection": sqlcc_selection_stats,
        },
        "final_distribution": helpers.dataset_distribution(final_rows),
        "final_rich_distribution": rich_distribution(final_rows),
        "validation": validation,
        "conversion_examples": conversion_examples,
        "input_sha256": {
            "sql_create_context": helpers.sha256_file(sqlcc_path),
            "spider_train": helpers.sha256_file(spider_train_path),
            "dev_reference": helpers.sha256_file(dev_reference_path),
        },
    }
    manifest["output_sha256"] = helpers.sha256_file(output_path)
    helpers.write_json(manifest_path, manifest)

    print(f"Wrote mix dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(manifest["sql_create_context"]["selection"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["final_rich_distribution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
