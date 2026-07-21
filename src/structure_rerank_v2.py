#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata
from typing import Any


METHOD_NAME = "structure_topk_v2"
MAX_ADJUSTMENT = 0.08
MIN_ADJUSTMENT = -0.04

_STOPWORDS = {
    "a", "all", "an", "and", "are", "by", "each", "for", "from", "has",
    "have", "how", "in", "is", "list", "many", "of", "on", "or", "show",
    "that", "the", "their", "there", "to", "what", "which", "who", "with",
}


def _normalize_words(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", str(sql))


def question_structure_hints(question: str) -> set[str]:
    q = _normalize_words(question)
    hints: set[str] = set()
    if re.search(r"\b(how many|number of|count|counts|total number)\b", q):
        hints.add("count")
    if re.search(r"\b(average|avg|mean)\b", q):
        hints.add("avg")
    if re.search(r"\b(sum|summed)\b", q) or (
        re.search(r"\btotal\b", q) and not re.search(r"\btotal number\b", q)
    ):
        hints.add("sum")
    if re.search(r"\b(maximum|max|highest|largest|greatest)\b", q):
        hints.update({"max", "order_extreme"})
    if re.search(r"\b(minimum|min|lowest|smallest)\b", q):
        hints.update({"min", "order_extreme"})
    if re.search(r"\b(most|least|fewest)\b", q):
        hints.add("order_extreme")
    if re.search(r"\b(for each|per|each|grouped by|by every)\b", q):
        hints.add("group_by")
    if re.search(r"\b(distinct|different|unique)\b", q):
        hints.add("distinct")
    if re.search(r"\b(without|never|not|no |except)\b", q):
        hints.update({"negation", "nested_select"})
    if re.search(r"\b(more than the average|less than the average|above average|below average)\b", q):
        hints.add("nested_select")
    return hints


def candidate_sql_features(sql: str) -> dict[str, Any]:
    low = _strip_sql_literals(sql).casefold()
    features: set[str] = set()
    for func in ("count", "sum", "avg", "min", "max"):
        if re.search(rf"\b{func}\s*\(", low):
            features.add(func)
    for label, pattern in (
        ("group_by", r"\bgroup\s+by\b"),
        ("having", r"\bhaving\b"),
        ("order_by", r"\border\s+by\b"),
        ("limit", r"\blimit\b"),
        ("distinct", r"\bdistinct\b"),
        ("exists", r"\bexists\s*\("),
        ("not_in", r"\bnot\s+in\s*\("),
    ):
        if re.search(pattern, low):
            features.add(label)
    if {"order_by", "limit"} <= features:
        features.add("order_by_limit")
    join_count = len(re.findall(r"\bjoin\b", low))
    select_count = len(re.findall(r"\bselect\b", low))
    if join_count:
        features.add("join")
    if join_count >= 2:
        features.add("multi_join")
    if select_count >= 2:
        features.add("nested_select")
    for operation in ("union", "intersect", "except"):
        if re.search(rf"\b{operation}\b", low):
            features.add(operation)
    return {
        "features": features,
        "join_count": join_count,
        "join_bucket": min(join_count, 2),
        "select_count": select_count,
    }


def _schema_entities(schema: str) -> tuple[dict[str, set[str]], int]:
    tables: dict[str, set[str]] = {}
    current: str | None = None
    foreign_keys = 0
    for raw_line in str(schema).splitlines():
        line = raw_line.strip()
        low = line.casefold()
        if low.startswith("table:"):
            current = _normalize_words(line.split(":", 1)[1].replace("_", " "))
            if current:
                tables.setdefault(current, set())
        elif current and low.startswith("columns:"):
            for value in line.split(":", 1)[1].split(","):
                column = _normalize_words(value.replace("_", " "))
                if column:
                    tables[current].add(column)
        elif line.startswith("-") and "->" in line:
            foreign_keys += 1
    return tables, foreign_keys


def estimate_target_join_bucket(question: str, schema: str) -> tuple[int | None, dict[str, Any]]:
    q = f" {_normalize_words(question)} "
    tables, foreign_keys = _schema_entities(schema)
    matched_tables: set[str] = set()
    table_matches: list[str] = []
    column_matches: list[dict[str, str]] = []

    for table, columns in tables.items():
        table_variants = {table}
        if " " not in table:
            table_variants.add(table + "s")
            if table.endswith("y") and len(table) > 1:
                table_variants.add(table[:-1] + "ies")
            if table.endswith(("s", "x", "z", "ch", "sh")):
                table_variants.add(table + "es")
        if len(table) >= 3 and any(f" {variant} " in q for variant in table_variants):
            matched_tables.add(table)
            table_matches.append(table)
        for column in columns:
            words = column.split()
            informative = (
                len(column) >= 5
                and column not in _STOPWORDS
                and (len(words) > 1 or words[0] not in {"id", "name", "type", "date", "year"})
            )
            if informative and f" {column} " in q:
                matched_tables.add(table)
                column_matches.append({"table": table, "column": column})

    if len(matched_tables) >= 3:
        bucket: int | None = 2
    elif len(matched_tables) == 2:
        bucket = 1
    else:
        bucket = None
    return bucket, {
        "target_schema_table_count": len(tables),
        "target_schema_foreign_key_count": foreign_keys,
        "matched_tables": sorted(matched_tables),
        "table_matches": sorted(table_matches),
        "column_matches": column_matches,
    }


def structure_rerank_adjustment(
    *,
    question: str,
    target_schema: str,
    candidate_sql: str,
    candidate_schema: str,
    max_adjustment: float = MAX_ADJUSTMENT,
) -> tuple[float, dict[str, Any]]:
    hints = question_structure_hints(question)
    candidate = candidate_sql_features(candidate_sql)
    candidate_features = candidate["features"]
    target_join_bucket, target_details = estimate_target_join_bucket(question, target_schema)
    candidate_tables, candidate_foreign_keys = _schema_entities(candidate_schema)
    components: list[dict[str, Any]] = []
    raw = 0.0

    def add(label: str, condition: bool, value: float) -> None:
        nonlocal raw
        if condition:
            raw += value
            components.append({"feature": label, "adjustment": value})

    for feature in ("count", "sum", "avg", "min", "max"):
        required = feature in hints
        add(feature + "_match", required and feature in candidate_features, 0.016)
        add(feature + "_missing", required and feature not in candidate_features, -0.018)
    add("group_by_match", "group_by" in hints and "group_by" in candidate_features, 0.014)
    add("group_by_missing", "group_by" in hints and "group_by" not in candidate_features, -0.014)
    add("distinct_match", "distinct" in hints and "distinct" in candidate_features, 0.010)
    add("distinct_missing", "distinct" in hints and "distinct" not in candidate_features, -0.010)
    add("extreme_match", "order_extreme" in hints and "order_by_limit" in candidate_features, 0.014)
    add("extreme_missing", "order_extreme" in hints and "order_by_limit" not in candidate_features, -0.014)
    add(
        "negation_match",
        "negation" in hints and bool({"not_in", "exists", "except", "nested_select"} & candidate_features),
        0.012,
    )
    add(
        "negation_missing",
        "negation" in hints and not bool({"not_in", "exists", "except", "nested_select"} & candidate_features),
        -0.012,
    )
    add("nested_match", "nested_select" in hints and "nested_select" in candidate_features, 0.010)

    if target_join_bucket is not None:
        distance = abs(int(candidate["join_bucket"]) - target_join_bucket)
        add("join_bucket_match", distance == 0, 0.026)
        add("join_bucket_off_by_one", distance == 1, -0.014)
        add("join_bucket_off_by_two", distance >= 2, -0.026)

    candidate_schema_chars = len(str(candidate_schema))
    add("very_long_demo_schema", candidate_schema_chars > 6000, -0.008)

    cap = max(0.0, min(float(max_adjustment), MAX_ADJUSTMENT))
    adjustment = max(MIN_ADJUSTMENT, min(raw, cap))
    return adjustment, {
        "method": METHOD_NAME,
        "question_hints": sorted(hints),
        "target_join_bucket": target_join_bucket,
        "target_details": target_details,
        "candidate_sql_features": sorted(candidate_features),
        "candidate_join_count": candidate["join_count"],
        "candidate_join_bucket": candidate["join_bucket"],
        "candidate_schema_table_count": len(candidate_tables),
        "candidate_schema_foreign_key_count": candidate_foreign_keys,
        "candidate_schema_chars": candidate_schema_chars,
        "components": components,
        "raw_adjustment": raw,
        "final_adjustment": adjustment,
        "max_adjustment": cap,
        "min_adjustment": MIN_ADJUSTMENT,
    }
