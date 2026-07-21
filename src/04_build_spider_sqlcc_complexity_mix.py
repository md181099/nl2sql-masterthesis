#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


AGG_RE = re.compile(r"\b(count|avg|sum|min|max)\s*\(", re.IGNORECASE)
SQL_START_RE = re.compile(r"(?is)\b(select|with)\b")
CREATE_TABLE_RE = re.compile(
    r"(?is)^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?P<name>\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*)"
    r"\s*\((?P<body>.*)\)\s*$"
)
CONSTRAINT_START_RE = re.compile(
    r"(?is)^(?:constraint|primary|foreign|unique|check|key)\b"
)
FK_RE = re.compile(
    r"(?is)foreign\s+key\s*\((?P<cols>[^)]+)\)\s*references\s+"
    r"(?P<table>\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*)"
    r"(?:\s*\((?P<refcols>[^)]+)\))?"
)
PK_RE = re.compile(r"(?is)primary\s+key\s*\((?P<cols>[^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 25k Spider-Train + SQL-Create-Context complexity-enriched "
            "mix in Spider-like schema format."
        )
    )
    parser.add_argument("--sqlcc_path", default="data/sql_create_context/train.jsonl")
    parser.add_argument("--spider_dir", default="data/spider/spider_data")
    parser.add_argument("--spider_train_path", default="data/spider/spider_data/train_spider.json")
    parser.add_argument("--dev_reference_path", default="data/testcases_spider_dev_full.jsonl")
    parser.add_argument(
        "--output_path",
        default=(
            "data/sql_create_context/"
            "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
            "25k_seed42_no_dev_overlap.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest_path",
        default=(
            "data/sql_create_context/"
            "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
            "25k_seed42_no_dev_overlap_manifest.json"
        ),
    )
    parser.add_argument("--target_size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sqlcc_aggregation_target_rate",
        type=float,
        default=0.58,
        help=(
            "Target any-aggregation share inside selected SQLCC rows after all rare "
            "complexity rows are included."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def normalize_question(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_sql(value: str) -> str:
    value = str(value).strip().casefold()
    value = re.sub(r";+\s*$", "", value)
    return " ".join(value.split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_spider_schema_builder(project_root: Path) -> Callable[[Path], str]:
    module_path = project_root / "src" / "00_prepare_spider_subset.py"
    spec = importlib.util.spec_from_file_location("prepare_spider_subset_for_mix", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Spider schema builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_schema_prompt = getattr(module, "build_schema_prompt", None)
    if build_schema_prompt is None:
        raise RuntimeError("00_prepare_spider_subset.py has no build_schema_prompt")
    return build_schema_prompt


def split_top_level(value: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False
    in_bracket = False
    for idx, ch in enumerate(value):
        if ch == "'" and not (in_double or in_backtick or in_bracket):
            in_single = not in_single
        elif ch == '"' and not (in_single or in_backtick or in_bracket):
            in_double = not in_double
        elif ch == "`" and not (in_single or in_double or in_bracket):
            in_backtick = not in_backtick
        elif ch == "[" and not (in_single or in_double or in_backtick):
            in_bracket = True
        elif ch == "]" and in_bracket:
            in_bracket = False
        elif not (in_single or in_double or in_backtick or in_bracket):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == delimiter and depth == 0:
                parts.append(value[start:idx].strip())
                start = idx + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_create_table_statements(context: str) -> list[str]:
    statements: list[str] = []
    start = 0
    depth = 0
    in_single = False
    in_double = False
    for idx, ch in enumerate(context):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not (in_single or in_double):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ";" and depth == 0:
                stmt = context[start:idx].strip()
                if stmt:
                    statements.append(stmt)
                start = idx + 1
    tail = context[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def clean_identifier(value: str) -> str:
    value = value.strip()
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("`") and value.endswith("`"))
        or (value.startswith("[") and value.endswith("]"))
    ):
        value = value[1:-1]
    if "." in value:
        value = value.split(".")[-1]
    return value.strip()


def parse_column_identifier(definition: str) -> str | None:
    definition = definition.strip()
    if not definition or CONSTRAINT_START_RE.match(definition):
        return None
    if definition[0] in {'"', "`", "["}:
        closing = '"' if definition[0] == '"' else ("`" if definition[0] == "`" else "]")
        end_idx = definition.find(closing, 1)
        if end_idx > 0:
            return clean_identifier(definition[: end_idx + 1])
    return clean_identifier(definition.split(None, 1)[0])


def parse_identifier_list(value: str) -> list[str]:
    return [clean_identifier(part) for part in split_top_level(value) if clean_identifier(part)]


def convert_create_context_to_spider_schema(context: str) -> tuple[str, int, int]:
    statements = split_create_table_statements(context.strip())
    if not statements:
        raise ValueError("schema context has no CREATE TABLE statements")

    lines: list[str] = ["Database schema:"]
    total_columns = 0
    table_count = 0
    for stmt_idx, statement in enumerate(statements):
        match = CREATE_TABLE_RE.match(statement)
        if match is None:
            raise ValueError(f"could not parse CREATE TABLE statement: {statement[:120]}")

        table_name = clean_identifier(match.group("name"))
        body = match.group("body")
        columns: list[str] = []
        primary_keys: list[str] = []
        foreign_keys: list[str] = []

        for part in split_top_level(body):
            pk_match = PK_RE.search(part)
            if pk_match is not None:
                primary_keys.extend(parse_identifier_list(pk_match.group("cols")))
                continue

            fk_match = FK_RE.search(part)
            if fk_match is not None:
                from_cols = parse_identifier_list(fk_match.group("cols"))
                ref_table = clean_identifier(fk_match.group("table"))
                ref_cols = parse_identifier_list(fk_match.group("refcols") or "")
                for idx, from_col in enumerate(from_cols):
                    ref_col = ref_cols[idx] if idx < len(ref_cols) else ""
                    target = f"{ref_table}.{ref_col}" if ref_col else ref_table
                    foreign_keys.append(f"- {table_name}.{from_col} -> {target}")
                continue

            column_name = parse_column_identifier(part)
            if column_name:
                columns.append(column_name)

        if not columns:
            raise ValueError(f"no columns parsed for table: {table_name}")

        if stmt_idx:
            lines.append("")
        lines.append(f"Table: {table_name}")
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append(f"Primary key: {', '.join(primary_keys)}" if primary_keys else "Primary key:")
        if foreign_keys:
            lines.append("Foreign keys:")
            lines.extend(foreign_keys)
        else:
            lines.append("Foreign keys:")

        total_columns += len(columns)
        table_count += 1

    return "\n".join(lines).strip(), table_count, total_columns


def sql_features(sql: str) -> dict[str, Any]:
    low = sql.lower()
    join_count = len(re.findall(r"\bjoin\b", low))
    agg_counts = {
        name: len(re.findall(rf"\b{name}\s*\(", low))
        for name in ("count", "avg", "sum", "min", "max")
    }
    return {
        "join_count": join_count,
        "group_by": bool(re.search(r"\bgroup\s+by\b", low)),
        "having": bool(re.search(r"\bhaving\b", low)),
        "nested_select": len(re.findall(r"\bselect\b", low)) > 1,
        "order_by": bool(re.search(r"\border\s+by\b", low)),
        "limit": bool(re.search(r"\blimit\b", low)),
        "distinct": bool(re.search(r"\bdistinct\b", low)),
        "union": bool(re.search(r"\bunion\b", low)),
        "intersect": bool(re.search(r"\bintersect\b", low)),
        "except": bool(re.search(r"\bexcept\b", low)),
        "compound": bool(re.search(r"\b(union|intersect|except)\b", low)),
        "any_aggregation": bool(AGG_RE.search(sql)),
        **{f"{name}_agg": count for name, count in agg_counts.items()},
    }


def rare_complexity(features: dict[str, Any]) -> bool:
    return bool(
        features["join_count"] > 0
        or features["group_by"]
        or features["having"]
        or features["nested_select"]
        or features["order_by"]
        or features["limit"]
        or features["distinct"]
        or features["compound"]
    )


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, max(0, round((len(values_sorted) - 1) * p)))
    return values_sorted[idx]


def dataset_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    join_bins: Counter[str] = Counter()
    agg_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    db_counts: Counter[str] = Counter()
    schema_table_counts: Counter[str] = Counter()
    sql_lengths: list[int] = []
    group_by = order_by = having = nested = union = intersect = except_count = distinct = limit = 0
    schema_with_fk = 0
    create_table_count = 0
    missing_schema_labels = 0
    missing_foreign_keys_label = 0

    for row in rows:
        source_counts[str(row.get("source_dataset", "unknown"))] += 1
        if row.get("db_id"):
            db_counts[str(row["db_id"])] += 1
        sql = str(row.get("gold_sql", ""))
        features = row.get("sql_features") or sql_features(sql)
        join_count = int(features["join_count"])
        if join_count >= 3:
            join_bins["3plus_join"] += 1
        else:
            join_bins[f"{join_count}_join"] += 1
        for name in ("count", "avg", "sum", "min", "max"):
            if int(features.get(f"{name}_agg", 0)) > 0:
                agg_counts[name] += 1
        if features["any_aggregation"]:
            agg_counts["any_aggregation"] += 1
        group_by += bool(features["group_by"])
        order_by += bool(features["order_by"])
        having += bool(features["having"])
        nested += bool(features["nested_select"])
        union += bool(features["union"])
        intersect += bool(features["intersect"])
        except_count += bool(features["except"])
        distinct += bool(features["distinct"])
        limit += bool(features["limit"])
        sql_lengths.append(len(sql))

        schema = str(row.get("schema_prompt") or row.get("context") or "")
        create_table_count += "CREATE TABLE" in schema.upper()
        has_core_labels = all(
            label in schema
            for label in ("Database schema:", "Table:", "Columns:", "Primary key:")
        )
        missing_schema_labels += not has_core_labels
        missing_foreign_keys_label += "Foreign keys:" not in schema
        schema_with_fk += " -> " in schema
        table_count = len(re.findall(r"(?m)^Table:\s+", schema))
        schema_table_counts[str(table_count)] += 1

    return {
        "rows": len(rows),
        "source_distribution": dict(sorted(source_counts.items())),
        "join_count_distribution": {
            "0_join": join_bins["0_join"],
            "1_join": join_bins["1_join"],
            "2_join": join_bins["2_join"],
            "3plus_join": join_bins["3plus_join"],
        },
        "aggregation_distribution": {
            "count": agg_counts["count"],
            "avg": agg_counts["avg"],
            "sum": agg_counts["sum"],
            "min": agg_counts["min"],
            "max": agg_counts["max"],
            "any_aggregation": agg_counts["any_aggregation"],
        },
        "group_by_count": group_by,
        "order_by_count": order_by,
        "having_count": having,
        "nested_select_count": nested,
        "union_count": union,
        "intersect_count": intersect,
        "except_count": except_count,
        "distinct_count": distinct,
        "limit_count": limit,
        "sql_length_distribution": {
            "min": min(sql_lengths) if sql_lengths else None,
            "mean": sum(sql_lengths) / len(sql_lengths) if sql_lengths else None,
            "p50": percentile(sql_lengths, 0.50),
            "p90": percentile(sql_lengths, 0.90),
            "p95": percentile(sql_lengths, 0.95),
            "p99": percentile(sql_lengths, 0.99),
            "max": max(sql_lengths) if sql_lengths else None,
        },
        "schema_table_count_distribution": dict(sorted(schema_table_counts.items())),
        "schema_with_foreign_key_edges": schema_with_fk,
        "schema_create_table_count": create_table_count,
        "missing_required_schema_labels": missing_schema_labels,
        "missing_foreign_keys_label": missing_foreign_keys_label,
        "db_id_distribution_spider_examples": dict(sorted(db_counts.items())),
    }


def load_dev_overlap_sets(dev_reference_path: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    dev_rows = load_jsonl(dev_reference_path)
    questions = {normalize_question(str(row.get("question", ""))) for row in dev_rows}
    sqls = {normalize_sql(str(row.get("gold_sql", ""))) for row in dev_rows}
    pairs = {
        (normalize_question(str(row.get("question", ""))), normalize_sql(str(row.get("gold_sql", ""))))
        for row in dev_rows
    }
    return questions, sqls, pairs


def build_spider_rows(
    *,
    spider_rows_raw: list[dict[str, Any]],
    spider_dir: Path,
    build_schema_prompt: Callable[[Path], str],
    dev_question_set: set[str],
    dev_sql_set: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    removed_question = 0
    removed_sql = 0
    removed_any = 0
    duplicate_question_sql = 0
    seen_question_sql: set[tuple[str, str]] = set()

    for source_idx, item in enumerate(spider_rows_raw):
        question = str(item.get("question", "")).strip()
        sql = ensure_semicolon(str(item.get("query", "")))
        db_id = str(item.get("db_id", "")).strip()
        if not question or not sql or not db_id:
            continue

        q_norm = normalize_question(question)
        s_norm = normalize_sql(sql)
        q_overlap = q_norm in dev_question_set
        s_overlap = s_norm in dev_sql_set
        if q_overlap:
            removed_question += 1
        if s_overlap:
            removed_sql += 1
        if q_overlap or s_overlap:
            removed_any += 1
            continue

        pair_key = (q_norm, s_norm)
        if pair_key in seen_question_sql:
            duplicate_question_sql += 1
            continue
        seen_question_sql.add(pair_key)

        if db_id not in schema_cache:
            db_file = spider_dir / "database" / db_id / f"{db_id}.sqlite"
            schema_cache[db_id] = build_schema_prompt(db_file)
        db_rel = Path("data") / "spider" / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
        features = sql_features(sql)
        rows.append(
            {
                "id": f"SPIDER_TRAIN_{source_idx:06d}",
                "source_dataset": "spider_train",
                "split": "train_spider",
                "source_path": "data/spider/spider_data/train_spider.json",
                "source_idx": source_idx,
                "question": question,
                "context": schema_cache[db_id],
                "schema_prompt": schema_cache[db_id],
                "gold_sql": sql,
                "db_id": db_id,
                "db_path": str(db_rel).replace("\\", "/"),
                "schema_format": "spider_table_columns_pk_fk_from_sqlite",
                "sql_features": features,
                "selection_bucket": "spider_train_strict_dev_clean",
            }
        )

    stats = {
        "input_rows": len(spider_rows_raw),
        "removed_dev_question_overlap": removed_question,
        "removed_dev_sql_overlap": removed_sql,
        "removed_dev_question_or_sql_overlap": removed_any,
        "removed_duplicate_question_sql": duplicate_question_sql,
        "written_rows": len(rows),
    }
    return rows, stats


def build_sqlcc_pool(
    *,
    sqlcc_path: Path,
    dev_question_set: set[str],
    dev_sql_set: set[str],
    spider_question_set: set[str],
    spider_sql_set: set[str],
    spider_pair_set: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    removed = Counter()
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

            q_norm = normalize_question(question)
            s_norm = normalize_sql(sql)
            if q_norm in dev_question_set:
                removed["dev_question_overlap"] += 1
            if s_norm in dev_sql_set:
                removed["dev_sql_overlap"] += 1
            if q_norm in dev_question_set or s_norm in dev_sql_set:
                removed["dev_question_or_sql_overlap"] += 1
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
                schema_prompt, table_count, column_count = convert_create_context_to_spider_schema(context)
            except Exception:
                removed["nonparseable_schema"] += 1
                continue

            seen_questions.add(q_norm)
            seen_sqls.add(s_norm)
            features = sql_features(sql)
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
                "source_row_sha256": sha256_text(
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


def select_sqlcc_rows(
    *,
    rows: list[dict[str, Any]],
    target_size: int,
    aggregation_target_rate: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rare_rows = [row for row in rows if rare_complexity(row["sql_features"])]
    agg_only_rows = [
        row
        for row in rows
        if not rare_complexity(row["sql_features"]) and row["sql_features"]["any_aggregation"]
    ]
    simple_rows = [
        row
        for row in rows
        if not rare_complexity(row["sql_features"]) and not row["sql_features"]["any_aggregation"]
    ]

    if len(rare_rows) > target_size:
        raise RuntimeError(
            f"Rare SQLCC pool ({len(rare_rows)}) exceeds SQLCC target ({target_size}); "
            "reduce rare bucket rules or increase target size."
        )

    rng.shuffle(rare_rows)
    rng.shuffle(agg_only_rows)
    rng.shuffle(simple_rows)

    selected = list(rare_rows)
    for row in selected:
        row["selection_bucket"] = "sqlcc_rare_complexity"

    desired_agg_count = round(target_size * aggregation_target_rate)
    current_agg_count = sum(bool(row["sql_features"]["any_aggregation"]) for row in selected)
    agg_needed = max(0, desired_agg_count - current_agg_count)
    agg_take = min(agg_needed, len(agg_only_rows), target_size - len(selected))
    for row in agg_only_rows[:agg_take]:
        row["selection_bucket"] = "sqlcc_aggregation_fill"
    selected.extend(agg_only_rows[:agg_take])

    simple_needed = target_size - len(selected)
    if simple_needed > len(simple_rows):
        for row in simple_rows:
            row["selection_bucket"] = "sqlcc_simple_fill"
        selected.extend(simple_rows)
        remaining = target_size - len(selected)
        if remaining > 0:
            extra_agg = agg_only_rows[agg_take : agg_take + remaining]
            for row in extra_agg:
                row["selection_bucket"] = "sqlcc_extra_aggregation_fill"
            selected.extend(extra_agg)
    else:
        for row in simple_rows[:simple_needed]:
            row["selection_bucket"] = "sqlcc_simple_fill"
        selected.extend(simple_rows[:simple_needed])

    if len(selected) != target_size:
        raise RuntimeError(f"Could not select requested SQLCC rows: {len(selected)} != {target_size}")

    stats = {
        "target_size": target_size,
        "aggregation_target_rate": aggregation_target_rate,
        "desired_sqlcc_aggregation_count": desired_agg_count,
        "available_bucket_sizes": {
            "rare_complexity": len(rare_rows),
            "aggregation_only": len(agg_only_rows),
            "simple": len(simple_rows),
        },
        "selected_bucket_sizes": dict(Counter(row["selection_bucket"] for row in selected)),
        "selected_distribution": dataset_distribution(selected),
    }
    return selected, stats


def leakage_counts(rows: list[dict[str, Any]], dev_pairs: tuple[set[str], set[str], set[tuple[str, str]]]) -> dict[str, int]:
    dev_q, dev_s, dev_pair = dev_pairs
    q_overlap = 0
    s_overlap = 0
    pair_overlap = 0
    for row in rows:
        q_norm = normalize_question(str(row.get("question", "")))
        s_norm = normalize_sql(str(row.get("gold_sql", "")))
        q_overlap += q_norm in dev_q
        s_overlap += s_norm in dev_s
        pair_overlap += (q_norm, s_norm) in dev_pair
    return {
        "question_overlap": q_overlap,
        "sql_overlap": s_overlap,
        "question_sql_pair_overlap": pair_overlap,
    }


def duplicate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids = [str(row.get("id", "")) for row in rows]
    questions = [normalize_question(str(row.get("question", ""))) for row in rows]
    sqls = [normalize_sql(str(row.get("gold_sql", ""))) for row in rows]
    pairs = list(zip(questions, sqls))
    return {
        "duplicate_id": len(ids) - len(set(ids)),
        "duplicate_question": len(questions) - len(set(questions)),
        "duplicate_sql": len(sqls) - len(set(sqls)),
        "duplicate_question_sql": len(pairs) - len(set(pairs)),
    }


def main() -> None:
    args = parse_args()
    if args.target_size < 1:
        raise ValueError("target_size must be >= 1")
    if not 0.0 <= args.sqlcc_aggregation_target_rate <= 1.0:
        raise ValueError("sqlcc_aggregation_target_rate must be between 0 and 1")

    project_root = Path(__file__).resolve().parents[1]
    sqlcc_path = resolve_path(project_root, args.sqlcc_path)
    spider_dir = resolve_path(project_root, args.spider_dir)
    spider_train_path = resolve_path(project_root, args.spider_train_path)
    dev_reference_path = resolve_path(project_root, args.dev_reference_path)
    output_path = resolve_path(project_root, args.output_path)
    manifest_path = resolve_path(project_root, args.manifest_path)

    for input_path in (sqlcc_path, spider_train_path, dev_reference_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {manifest_path}")

    rng = random.Random(args.seed)
    build_schema_prompt = load_spider_schema_builder(project_root)
    dev_sets = load_dev_overlap_sets(dev_reference_path)
    dev_question_set, dev_sql_set, _dev_pair_set = dev_sets

    spider_rows_raw = load_json_array(spider_train_path)
    spider_rows, spider_stats = build_spider_rows(
        spider_rows_raw=spider_rows_raw,
        spider_dir=spider_dir,
        build_schema_prompt=build_schema_prompt,
        dev_question_set=dev_question_set,
        dev_sql_set=dev_sql_set,
    )

    if len(spider_rows) >= args.target_size:
        raise RuntimeError(
            f"Spider rows alone ({len(spider_rows)}) exceed or fill target_size={args.target_size}"
        )

    spider_question_set = {normalize_question(str(row["question"])) for row in spider_rows}
    spider_sql_set = {normalize_sql(str(row["gold_sql"])) for row in spider_rows}
    spider_pair_set = {
        (normalize_question(str(row["question"])), normalize_sql(str(row["gold_sql"])))
        for row in spider_rows
    }
    sqlcc_pool, sqlcc_pool_stats, conversion_examples = build_sqlcc_pool(
        sqlcc_path=sqlcc_path,
        dev_question_set=dev_question_set,
        dev_sql_set=dev_sql_set,
        spider_question_set=spider_question_set,
        spider_sql_set=spider_sql_set,
        spider_pair_set=spider_pair_set,
    )

    sqlcc_target_size = args.target_size - len(spider_rows)
    sqlcc_rows, sqlcc_selection_stats = select_sqlcc_rows(
        rows=sqlcc_pool,
        target_size=sqlcc_target_size,
        aggregation_target_rate=args.sqlcc_aggregation_target_rate,
        rng=rng,
    )

    final_rows = spider_rows + sqlcc_rows
    rng.shuffle(final_rows)

    validation = {
        "target_size_ok": len(final_rows) == args.target_size,
        "dev_leakage": leakage_counts(final_rows, dev_sets),
        "duplicates": duplicate_counts(final_rows),
        "schema_validation": {
            "create_table_count": dataset_distribution(final_rows)["schema_create_table_count"],
            "missing_required_schema_labels": dataset_distribution(final_rows)["missing_required_schema_labels"],
        },
    }
    validation["all_passed"] = (
        validation["target_size_ok"]
        and validation["dev_leakage"]["question_overlap"] == 0
        and validation["dev_leakage"]["sql_overlap"] == 0
        and validation["dev_leakage"]["question_sql_pair_overlap"] == 0
        and validation["duplicates"]["duplicate_id"] == 0
        and validation["duplicates"]["duplicate_question_sql"] == 0
        and validation["schema_validation"]["create_table_count"] == 0
        and validation["schema_validation"]["missing_required_schema_labels"] == 0
    )
    if not validation["all_passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    write_jsonl(output_path, final_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/04_build_spider_sqlcc_complexity_mix.py",
        "seed": args.seed,
        "target_size": args.target_size,
        "actual_size": len(final_rows),
        "source_paths": {
            "sql_create_context": str(sqlcc_path.relative_to(project_root)),
            "spider_train": str(spider_train_path.relative_to(project_root)),
            "spider_dir": str(spider_dir.relative_to(project_root)),
            "dev_reference": str(dev_reference_path.relative_to(project_root)),
        },
        "output_path": str(output_path.relative_to(project_root)),
        "sampling_strategy": (
            "Include all strict Spider-Dev-clean Spider-Train examples, then fill to target "
            "with SQLCC rows in Spider-like schema format: all rare complexity rows first "
            "(joins/group/having/nested/order/limit/distinct/compound), aggregation-only "
            "rows up to target rate, simple rows last."
        ),
        "leakage_policy": {
            "dev_reference": "Spider Dev full JSONL",
            "removed_if_question_matches_dev": True,
            "removed_if_sql_matches_dev": True,
            "removed_if_pair_matches_dev": True,
            "sqlcc_removed_if_question_matches_selected_spider_train": False,
            "sqlcc_removed_if_sql_matches_selected_spider_train": False,
            "sqlcc_removed_if_question_sql_pair_matches_selected_spider_train": True,
            "sqlcc_question_or_sql_only_overlap_with_spider_train": "counted but retained; single-field cross-source overlap is not leakage",
            "spider_train_internal_sql_only_dedupe": False,
            "sqlcc_internal_question_dedupe": True,
            "sqlcc_internal_sql_dedupe": True,
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
            "distribution": dataset_distribution(spider_rows),
        },
        "sql_create_context": {
            "pool": {
                **sqlcc_pool_stats,
                "distribution": dataset_distribution(sqlcc_pool),
            },
            "selection": sqlcc_selection_stats,
        },
        "final_distribution": dataset_distribution(final_rows),
        "validation": validation,
        "conversion_examples": conversion_examples,
        "input_sha256": {
            "sql_create_context": sha256_file(sqlcc_path),
            "spider_train": sha256_file(spider_train_path),
            "dev_reference": sha256_file(dev_reference_path),
        },
    }
    manifest["output_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)

    print(f"Wrote mix dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(manifest["final_distribution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
