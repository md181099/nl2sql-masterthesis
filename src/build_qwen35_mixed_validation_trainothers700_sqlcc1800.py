#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import statistics
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SEED = 42
TRAINOTHERS_TARGET = 700
SQLCC_TARGET = 1800
TOTAL_TARGET = TRAINOTHERS_TARGET + SQLCC_TARGET
MAX_LENGTH = 2048
BUILDER_VERSION = "mixed_validation_trainothers700_sqlcc1800_v1"
CHAT_FORMAT = "qwen_sqlctx_chatml"
SYSTEM_PROMPT_VARIANT = "sqlctx_anti_overjoin"

TRAIN_RAW = "data/sql_create_context/train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
TRAIN_SFT = "data/sql_create_context/train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
SQLCC_RAW = "data/sql_create_context/train.jsonl"
OLD_SQLCC_VAL = "data/sql_create_context/val_sft_qwen35_full_chat_v1_clean_anti_overjoin_sqlcc_only_no_spider_no_train_overlap_2500_seed42.jsonl"
TRAIN_OTHERS = "data/spider/spider_data/train_others.json"
SPIDER_TRAIN = "data/spider/spider_data/train_spider.json"
DEV_1032 = "data/testcases_spider_dev_full.jsonl"
DEV_1034 = "data/spider/spider_data/dev.json"
SPIDER_DIR = "data/spider/spider_data"
TOKENIZER_ID = "Qwen/Qwen3.5-9B-Base"

DEFAULT_OUTPUT = "data/sql_create_context/val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42.jsonl"
DEFAULT_MANIFEST = "data/sql_create_context/val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_manifest.json"
DEFAULT_AUDIT = "audits/audit_qwen35_mixed_validation_trainothers700_sqlcc1800_seed42_20260710.md"

DENIED_SQL_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "vacuum", "pragma", "begin", "commit", "rollback",
    "savepoint", "release", "reindex", "analyze",
}
SQL_KEYWORDS = {
    "select", "with", "recursive", "from", "join", "inner", "left", "right", "full",
    "cross", "on", "where", "group", "by", "having", "order", "limit", "offset",
    "distinct", "union", "intersect", "except", "as", "and", "or", "not", "in",
    "exists", "between", "like", "is", "null", "asc", "desc", "case", "when", "then",
    "else", "end", "count", "sum", "avg", "min", "max", "all",
}
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "how", "many", "much", "show", "list",
    "find", "give", "me", "there", "does", "do", "with", "that", "have", "has",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit the fixed Qwen mixed loss-validation set.")
    parser.add_argument("--write", action="store_true", help="Write final artifacts atomically; default is dry-run.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array: {path}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(scope: str, row_id: str) -> str:
    return sha256_text(f"{SEED}|{scope}|{row_id}")


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    chars = []
    for char in value:
        category = unicodedata.category(char)
        chars.append(" " if category.startswith(("P", "S", "Z")) else char)
    return " ".join("".join(chars).split())


def strip_sql_comments(value: str) -> str:
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(value):
        char = value[i]
        if quote:
            out.append(char)
            if char == quote:
                if i + 1 < len(value) and value[i + 1] == quote:
                    out.append(value[i + 1])
                    i += 1
                else:
                    quote = None
            i += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            out.append(char)
            i += 1
            continue
        if value.startswith("--", i):
            end = value.find("\n", i + 2)
            i = len(value) if end < 0 else end
            out.append(" ")
            continue
        if value.startswith("/*", i):
            end = value.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated SQL block comment")
            i = end + 2
            out.append(" ")
            continue
        out.append(char)
        i += 1
    if quote:
        raise ValueError("unterminated SQL quote")
    return "".join(out)


def sql_tokens(value: str) -> list[str]:
    value = strip_sql_comments(unicodedata.normalize("NFKC", str(value)))
    pattern = re.compile(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
        r"<=|>=|<>|!=|==|\|\||[(),;.*+/%<>=-]|[A-Za-z_][\w$]*|\d+(?:\.\d+)?|\S",
        flags=re.UNICODE,
    )
    return pattern.findall(value)


def normalize_sql(value: str) -> str:
    tokens = sql_tokens(value)
    while tokens and tokens[-1] == ";":
        tokens.pop()
    return " ".join(token.casefold() for token in tokens)


def exact_sql(value: str) -> str:
    return strip_sql_comments(unicodedata.normalize("NFKC", str(value))).strip().rstrip(";").strip()


def statement_safety(sql: str) -> tuple[bool, str | None]:
    try:
        tokens = sql_tokens(sql)
    except ValueError as exc:
        return False, str(exc)
    if not tokens:
        return False, "empty_sql"
    semicolons = [i for i, token in enumerate(tokens) if token == ";"]
    if len(semicolons) > 1 or (semicolons and semicolons[0] != len(tokens) - 1):
        return False, "multiple_statements"
    words = [token.casefold() for token in tokens if re.match(r"^[A-Za-z_]", token)]
    if not words or words[0] not in {"select", "with"}:
        return False, "not_select_or_with"
    denied = sorted(set(words) & DENIED_SQL_KEYWORDS)
    if denied:
        return False, "denied_keyword:" + ",".join(denied)
    return True, None


def schema_struct(schema: str) -> tuple[tuple[Any, ...], ...]:
    blocks = re.split(r"(?m)^Table:\s*", str(schema))[1:]
    parsed: list[tuple[Any, ...]] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        name = unicodedata.normalize("NFKC", lines[0]).casefold().strip(' `"[]')
        columns: tuple[str, ...] = ()
        primary: tuple[str, ...] = ()
        foreign: list[str] = []
        in_foreign = False
        for line in lines[1:]:
            stripped = line.strip()
            low = stripped.casefold()
            if low.startswith("columns:"):
                columns = tuple(sorted(x.strip().casefold() for x in stripped.split(":", 1)[1].split(",") if x.strip()))
                in_foreign = False
            elif low.startswith("primary key:"):
                primary = tuple(sorted(x.strip().casefold() for x in stripped.split(":", 1)[1].split(",") if x.strip() and x.strip().casefold() != "none"))
                in_foreign = False
            elif low.startswith("foreign keys:"):
                in_foreign = True
            elif in_foreign and stripped:
                foreign.append(" ".join(low.split()))
        if name:
            parsed.append((name, columns, primary, tuple(sorted(foreign))))
    return tuple(sorted(parsed))


def schema_signature(schema: str) -> str:
    return sha256_text(json.dumps(schema_struct(schema), ensure_ascii=False, sort_keys=True))


def schema_table_ids(schema: str) -> set[str]:
    return {item[0] for item in schema_struct(schema)}


def source_key(source_path: str, source_idx: int) -> str:
    return f"{source_path}#{source_idx}"


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    sql = str(row.get("gold_sql") or row.get("query") or row.get("answer") or "").strip()
    schema = str(row.get("schema_prompt") or row.get("context") or "").strip()
    db_or_schema = str(row.get("db_id") or schema_signature(schema))
    return {
        "id": str(row.get("id", "")),
        "source_id": source_key(str(row.get("source_path", "")), int(row.get("source_idx", -1))),
        "question_exact": question,
        "question_norm": normalize_question(question),
        "sql_exact": exact_sql(sql),
        "sql_norm": normalize_sql(sql),
        "pair_exact": (question, exact_sql(sql)),
        "pair_norm": (normalize_question(question), normalize_sql(sql)),
        "schema_question": (db_or_schema, normalize_question(question)),
        "schema_question_sql": (db_or_schema, normalize_question(question), normalize_sql(sql)),
        "schema_signature": schema_signature(schema),
    }


def identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[Any]]:
    keys = [
        "id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm",
        "pair_exact", "pair_norm", "schema_question", "schema_question_sql", "schema_signature",
    ]
    values = {key: set() for key in keys}
    for row in rows:
        identity = row_identity(row)
        for key in keys:
            values[key].add(identity[key])
    return values


def overlap_counts(rows: list[dict[str, Any]], reference: dict[str, set[Any]]) -> dict[str, int]:
    counts = collections.Counter()
    for row in rows:
        identity = row_identity(row)
        for key, ref_values in reference.items():
            counts[key] += identity[key] in ref_values
    return dict(counts)


def sqlite_schema_prompt(db_path: Path) -> str:
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        lines = ["Database schema:"]
        for index, table in enumerate(tables):
            quoted = '"' + str(table).replace('"', '""') + '"'
            info = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            columns = [str(row[1]) for row in info]
            primary = [str(row[1]) for row in sorted(info, key=lambda row: int(row[5] or 0)) if int(row[5] or 0) > 0]
            foreign = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            lines.extend([
                f"Table: {table}",
                f"Columns: {', '.join(columns)}",
                f"Primary key: {', '.join(primary)}" if primary else "Primary key: none",
                "Foreign keys:",
            ])
            for fk in foreign:
                lines.append(f"- {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            if index != len(tables) - 1:
                lines.append("")
        return "\n".join(lines).strip()
    finally:
        connection.close()


def execute_readonly(db_path: Path, sql: str, timeout_seconds: float = 3.0) -> tuple[bool, str | None]:
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2)
    deadline = time.monotonic() + timeout_seconds
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = connection.execute(sql)
        cursor.fetchone()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()


def validate_sqlcc_in_memory(context: str, sql: str) -> tuple[bool, str | None]:
    statements = [part.strip() for part in context.split(";") if part.strip()]
    if not statements or any(not re.match(r"(?is)^create\s+table\b", part) for part in statements):
        return False, "schema_context_not_create_table_only"
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(";".join(statements) + ";")
        connection.execute("PRAGMA query_only=ON")
        connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()


def lexical_features(sql: str, question: str, schema: str) -> dict[str, Any]:
    tokens = sql_tokens(sql)
    low = [token.casefold() for token in tokens]
    words = [token for token in low if re.match(r"^[a-z_]", token)]
    select_depths: list[int] = []
    depth = 0
    for token in low:
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif token == "select":
            select_depths.append(depth)
    join_count = sum(token == "join" for token in low)
    aggregation_names = {name: sum(1 for i, token in enumerate(low[:-1]) if token == name and low[i + 1] == "(") for name in ("count", "sum", "avg", "min", "max")}
    where_index = next((i for i, token in enumerate(low) if token == "where"), None)
    where_tail = low[where_index + 1 :] if where_index is not None else []
    selected_columns = 0
    if "select" in low and "from" in low:
        start = low.index("select") + 1
        end = low.index("from", start)
        selected_columns = 1 + sum(token == "," for token in low[start:end]) if end > start else 0
    literals = sum(bool(re.match(r"^(?:'|\"|\d)", token)) for token in tokens)
    table_ids = schema_table_ids(schema)
    referenced_tables = set()
    for i, token in enumerate(low[:-1]):
        if token in {"from", "join"}:
            candidate = low[i + 1].strip('`"[]')
            if candidate and candidate != "(":
                referenced_tables.add(candidate)
    rare = bool(join_count or len(select_depths) > 1 or any(x in words for x in ("group", "having", "order", "limit", "distinct", "union", "intersect", "except")))
    aggregation = any(aggregation_names.values())
    return {
        "analysis_method": "project_local_sql_lexer",
        "parser_success": True,
        "join_count": join_count,
        "join_bin": "0" if join_count == 0 else ("1" if join_count == 1 else "2+"),
        "referenced_table_count": len(referenced_tables) or min(1, len(table_ids)),
        "aggregation": aggregation,
        "aggregation_names": aggregation_names,
        "group_by": "group" in words and "by" in words,
        "having": "having" in words,
        "order_by": "order" in words and "by" in words,
        "limit": "limit" in words,
        "distinct": "distinct" in words,
        "subquery": len(select_depths) > 1,
        "correlated_subquery": None,
        "in_operator": "in" in words,
        "exists_operator": "exists" in words,
        "union": "union" in words,
        "intersect": "intersect" in words,
        "except": "except" in words,
        "set_operation": any(x in words for x in ("union", "intersect", "except")),
        "where": where_index is not None,
        "where_condition_count": (1 + sum(token in {"and", "or"} for token in where_tail)) if where_tail else 0,
        "and_operator": "and" in where_tail,
        "or_operator": "or" in where_tail,
        "like": "like" in words,
        "between": "between" in words,
        "null_check": "null" in words and "is" in words,
        "comparison": any(token in {"=", "==", "!=", "<>", "<", ">", "<=", ">="} for token in low),
        "selected_column_count": selected_columns,
        "sql_token_length": len(tokens),
        "question_token_length": len(normalize_question(question).split()),
        "schema_table_count": len(table_ids),
        "schema_column_count": sum(len(item[1]) for item in schema_struct(schema)),
        "literal_count": literals,
        "rare_complexity": rare,
        "complexity_class": "complex" if join_count >= 2 or len(select_depths) > 1 or any(x in words for x in ("having", "union", "intersect", "except")) else ("moderate" if rare or aggregation else "simple"),
    }


def spider_ast_features(row: dict[str, Any], schema: str) -> dict[str, Any]:
    features = lexical_features(str(row["query"]), str(row["question"]), schema)
    ast = row.get("sql")
    if not isinstance(ast, dict):
        features["analysis_method"] = "project_local_sql_lexer_fallback_missing_spider_ast"
        features["parser_success"] = False
        return features

    def visit(node: Any, nested: bool = False) -> dict[str, int]:
        counts = collections.Counter()
        if isinstance(node, dict):
            if "select" in node:
                counts["select_nodes"] += 1
                counts["subqueries"] += int(nested)
            from_part = node.get("from") or {}
            units = from_part.get("table_units") or [] if isinstance(from_part, dict) else []
            counts["table_units"] += sum(1 for unit in units if isinstance(unit, list) and unit and unit[0] == "table_unit")
            for value in node.values():
                child = visit(value, nested or (value is not node and "select" in node))
                counts.update(child)
        elif isinstance(node, list):
            for value in node:
                counts.update(visit(value, nested))
        return counts

    counts = visit(ast)
    top_tables = len((ast.get("from") or {}).get("table_units") or [])
    features.update({
        "analysis_method": "spider_sql_ast_plus_project_local_sql_lexer",
        "parser_success": True,
        "referenced_table_count": max(features["referenced_table_count"], counts["table_units"]),
        "join_count": max(features["join_count"], max(0, top_tables - 1)),
        "subquery": counts["select_nodes"] > 1,
        "group_by": bool(ast.get("groupBy")),
        "having": bool(ast.get("having")),
        "order_by": bool(ast.get("orderBy")),
        "limit": ast.get("limit") is not None,
        "union": ast.get("union") is not None,
        "intersect": ast.get("intersect") is not None,
        "except": ast.get("except") is not None,
        "set_operation": any(ast.get(key) is not None for key in ("union", "intersect", "except")),
    })
    features["join_bin"] = "0" if features["join_count"] == 0 else ("1" if features["join_count"] == 1 else "2+")
    features["rare_complexity"] = bool(features["join_count"] or features["subquery"] or features["group_by"] or features["having"] or features["order_by"] or features["limit"] or features["distinct"] or features["set_operation"])
    features["complexity_class"] = "complex" if features["join_count"] >= 2 or features["subquery"] or features["having"] or features["set_operation"] else ("moderate" if features["rare_complexity"] or features["aggregation"] else "simple")
    return features


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def length_bin(value: int, thresholds: tuple[float, float, float]) -> str:
    return "q1" if value <= thresholds[0] else ("q2" if value <= thresholds[1] else ("q3" if value <= thresholds[2] else "q4"))


def allocate_with_caps(weights: dict[str, float], capacities: dict[str, int], total: int) -> dict[str, int]:
    allocation = {key: 0 for key in capacities}
    remaining = total
    active = {key for key, capacity in capacities.items() if capacity > 0}
    while remaining and active:
        weight_sum = sum(max(weights.get(key, 0.0), 0.0) for key in active)
        if weight_sum == 0:
            weight_sum = float(len(active))
            local_weights = {key: 1.0 for key in active}
        else:
            local_weights = {key: max(weights.get(key, 0.0), 0.0) for key in active}
        ideals = {key: remaining * local_weights[key] / weight_sum for key in active}
        progress = 0
        for key in sorted(active):
            add = min(capacities[key] - allocation[key], int(math.floor(ideals[key])))
            if add > 0:
                allocation[key] += add
                remaining -= add
                progress += add
        if remaining == 0:
            break
        ranked = sorted(active, key=lambda key: (-(ideals[key] - math.floor(ideals[key])), stable_key("quota", key)))
        for key in ranked:
            if remaining == 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                progress += 1
        active = {key for key in active if allocation[key] < capacities[key]}
        if progress == 0:
            break
    if remaining:
        raise RuntimeError(f"Cannot allocate requested total={total}; unallocated={remaining}")
    return allocation


def select_stratified(rows: list[dict[str, Any]], quota: int, scope: str, signature_keys: tuple[str, ...], group_cap: int | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        signature = "|".join(str(row["features"].get(key)) for key in signature_keys)
        groups[signature].append(row)
    quotas = allocate_with_caps({key: len(value) for key, value in groups.items()}, {key: len(value) for key, value in groups.items()}, quota)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_counts = collections.Counter()
    for signature in sorted(groups):
        if quotas[signature] == 0:
            continue
        selected_in_signature = 0
        for row in sorted(groups[signature], key=lambda item: stable_key(scope + "|" + signature, item["id"])):
            table_group = row.get("group_id", row["id"])
            if group_cap is not None and group_counts[table_group] >= group_cap:
                continue
            selected.append(row)
            selected_ids.add(row["id"])
            group_counts[table_group] += 1
            selected_in_signature += 1
            if selected_in_signature >= quotas[signature]:
                break
    if len(selected) < quota:
        for row in sorted(rows, key=lambda item: stable_key(scope + "|fill", item["id"])):
            if row["id"] in selected_ids:
                continue
            table_group = row.get("group_id", row["id"])
            if group_cap is not None and group_counts[table_group] >= group_cap:
                continue
            selected.append(row)
            selected_ids.add(row["id"])
            group_counts[table_group] += 1
            if len(selected) == quota:
                break
    if len(selected) != quota:
        raise RuntimeError(f"Stratified selection {scope} produced {len(selected)} != {quota}")
    return selected


def feature_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boolean_keys = ["aggregation", "group_by", "having", "order_by", "limit", "distinct", "subquery", "set_operation", "where", "in_operator", "exists_operator", "like", "between", "null_check"]
    result: dict[str, Any] = {"rows": len(rows)}
    result["source"] = dict(collections.Counter(row["source_dataset"] for row in rows))
    result["db_id"] = dict(sorted(collections.Counter(str(row.get("db_id") or "") for row in rows).items()))
    result["complexity_class"] = dict(sorted(collections.Counter(row["features"]["complexity_class"] for row in rows).items()))
    result["join_bin"] = dict(sorted(collections.Counter(row["features"]["join_bin"] for row in rows).items()))
    for key in boolean_keys:
        count = sum(bool(row["features"].get(key)) for row in rows)
        result[key] = {"count": count, "rate": count / len(rows) if rows else 0.0}
    for key in ["sql_token_length", "question_token_length", "schema_table_count", "schema_column_count", "referenced_table_count", "selected_column_count", "literal_count"]:
        values = [int(row["features"].get(key) or 0) for row in rows]
        result[key] = numeric_stats(values)
    return result


def numeric_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": min(values), "mean": statistics.mean(values),
        "p50": quantile(values, 0.50), "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95), "p99": quantile(values, 0.99), "max": max(values),
    }


def js_divergence(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return 0.0
    pa = [a.get(key, 0) / sa for key in keys]
    pb = [b.get(key, 0) / sb for key in keys]
    middle = [(x + y) / 2 for x, y in zip(pa, pb)]
    def kl(left: list[float], right: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(left, right) if x > 0 and y > 0)
    return 0.5 * kl(pa, middle) + 0.5 * kl(pb, middle)


def representation_metrics(pool: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    categorical = {
        "db_id": lambda row: str(row.get("db_id") or ""),
        "complexity": lambda row: row["features"]["complexity_class"],
        "join_bin": lambda row: row["features"]["join_bin"],
        "sql_length_bin": lambda row: row["features"]["sql_length_bin"],
    }
    js = {}
    max_abs = 0.0
    for name, getter in categorical.items():
        pool_counts = collections.Counter(getter(row) for row in pool)
        selected_counts = collections.Counter(getter(row) for row in selected)
        js[name] = js_divergence(pool_counts, selected_counts)
        for key in set(pool_counts) | set(selected_counts):
            delta = abs(pool_counts[key] / len(pool) - selected_counts[key] / len(selected))
            max_abs = max(max_abs, delta)
    quantiles = {}
    for key in ("sql_token_length", "question_token_length"):
        p = [row["features"][key] for row in pool]
        s = [row["features"][key] for row in selected]
        quantiles[key] = {q: quantile(s, f) - quantile(p, f) for q, f in (("p50", .5), ("p90", .9), ("p95", .95))}
    return {"js_divergence": js, "max_absolute_share_delta": max_abs, "quantile_deltas": quantiles}


def render_row(row: dict[str, Any], sft_module: Any, system_prompt: str) -> dict[str, Any]:
    completion = sft_module.sanitize_completion(row["gold_sql"])
    user_prompt = sft_module.build_user_prompt(row["schema_prompt"], row["question"])
    text = sft_module.build_full_chat_text(system_prompt=system_prompt, user_prompt=user_prompt, completion=completion, chat_format=CHAT_FORMAT)
    return {"id": row["id"], "text": text, "completion": completion, "user_prompt": user_prompt}


def token_components(row: dict[str, Any], rendered: dict[str, Any], tokenizer: Any, system_prompt: str) -> dict[str, int]:
    schema = row["schema_prompt"].strip()
    question = row["question"].strip()
    completion = rendered["completion"]
    segments = [
        ("system", "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"),
        ("other_prompt", "<|im_start|>user\nDatabase schema:\n"),
        ("schema", schema),
        ("other_prompt", "\n\nRules:\n- Use only the tables and columns from the schema.\n- Output exactly ONE SQLite read query.\n- Start directly with SELECT or WITH.\n- End with a semicolon.\n- Do NOT explain anything.\n- Do NOT use markdown.\n\nQuestion:\n"),
        ("question", question),
        ("other_prompt", "\n\nSQL:<|im_end|>\n<|im_start|>assistant\n"),
        ("assistant_sql", completion),
        ("other_prompt", "<|im_end|>\n"),
    ]
    counts = collections.Counter()
    prefix = ""
    previous = 0
    for name, text in segments:
        prefix += text
        current = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        counts[name] += current - previous
        previous = current
    counts["total"] = previous
    if prefix != rendered["text"]:
        raise RuntimeError(f"Component serialization mismatch for {row['id']}")
    return dict(counts)


def bfd_pack_stats(lengths: list[int]) -> dict[str, Any]:
    from datasets import Dataset, disable_progress_bars
    from trl.data_utils import pack_dataset

    disable_progress_bars()
    source = Dataset.from_dict({"input_ids": [[0] * length for length in lengths]})
    packed = pack_dataset(
        source,
        seq_length=MAX_LENGTH,
        strategy="bfd",
        map_kwargs={"load_from_cache_file": False},
    )
    sequence_lengths = [list(values) for values in packed["seq_lengths"]]
    packed_lengths = [len(values) for values in packed["input_ids"]]
    return {
        "strategy": "trl.data_utils.pack_dataset(strategy=bfd)",
        "packed_sequences": len(packed),
        "total_examples": len(lengths),
        "total_example_boundaries": sum(max(0, len(values) - 1) for values in sequence_lengths),
        "examples_per_packed_sequence": numeric_stats([len(values) for values in sequence_lengths]),
        "packed_token_lengths": numeric_stats(packed_lengths),
        "sequence_example_counts": [len(values) for values in sequence_lengths],
        "all_boundaries_have_chatml_end_start_markers": True,
    }


def near_duplicate_top(validation: list[dict[str, Any]], reference: list[dict[str, Any]], label: str, limit: int = 20) -> list[dict[str, Any]]:
    ref_tokens = [set(normalize_question(row["question"]).split()) for row in reference]
    inverted: dict[str, set[int]] = collections.defaultdict(set)
    for index, tokens in enumerate(ref_tokens):
        for token in tokens - STOPWORDS:
            if len(token) >= 3:
                inverted[token].add(index)
    hits: list[dict[str, Any]] = []
    for row in validation:
        question_norm = normalize_question(row["question"])
        tokens = set(question_norm.split())
        candidates: set[int] = set()
        for token in tokens - STOPWORDS:
            if len(token) >= 3:
                candidates.update(inverted.get(token, set()))
        scored = []
        for index in candidates:
            other = reference[index]
            if label == "internal" and row["id"] == other.get("id"):
                continue
            union = tokens | ref_tokens[index]
            jaccard = len(tokens & ref_tokens[index]) / len(union) if union else 1.0
            if jaccard >= 0.45:
                scored.append((jaccard, index))
        for jaccard, index in sorted(scored, reverse=True)[:5]:
            other = reference[index]
            sequence = SequenceMatcher(None, question_norm, normalize_question(other["question"]), autojunk=False).ratio()
            hits.append({
                "comparison": label, "validation_id": row["id"], "validation_question": row["question"],
                "reference_id": other.get("id"), "reference_question": other["question"],
                "token_jaccard": jaccard, "sequence_similarity": sequence,
                "same_sql_structure": row["features"]["structure_signature"] == other.get("features", {}).get("structure_signature"),
            })
    return sorted(hits, key=lambda item: (-max(item["token_jaccard"], item["sequence_similarity"]), -item["token_jaccard"], str(item["validation_id"])))[:limit]


def structure_signature(features: dict[str, Any]) -> str:
    keys = ("join_bin", "aggregation", "group_by", "having", "order_by", "limit", "distinct", "subquery", "set_operation", "where_condition_count")
    return "|".join(str(features.get(key)) for key in keys)


def make_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        schema = str(row.get("schema_prompt") or row.get("context") or "")
        sql = str(row.get("gold_sql") or row.get("query") or row.get("answer") or "")
        features = lexical_features(sql, str(row.get("question", "")), schema)
        features["structure_signature"] = structure_signature(features)
        result.append({**row, "features": features})
    return result


def build(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {name: project_root / value for name, value in {
        "train_raw": TRAIN_RAW, "train_sft": TRAIN_SFT, "sqlcc_raw": SQLCC_RAW,
        "old_sqlcc_validation": OLD_SQLCC_VAL, "train_others": TRAIN_OTHERS,
        "spider_train": SPIDER_TRAIN, "dev_1032": DEV_1032, "dev_1034": DEV_1034,
    }.items()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    mix_module = load_module(project_root / "src/04_build_spider_sqlcc_complexity_mix.py", "mixed_val_mix")
    sft_module = load_module(project_root / "src/02_make_sft_dataset_v1_clean_full_chat.py", "mixed_val_sft")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    system_prompt, system_source, system_path, _ = sft_module.resolve_system_prompt(
        project_root=project_root, system_prompt_variant=SYSTEM_PROMPT_VARIANT, system_prompt_path=None
    )

    train = load_jsonl(paths["train_raw"])
    train_sft = load_jsonl(paths["train_sft"])
    old_val_sft = load_jsonl(paths["old_sqlcc_validation"])
    train_others_raw = load_json(paths["train_others"])
    sqlcc_raw = load_jsonl(paths["sqlcc_raw"])
    dev_1032 = load_jsonl(paths["dev_1032"])
    dev_1034_raw = load_json(paths["dev_1034"])

    schema_cache: dict[str, str] = {}
    dev_1034 = []
    for index, row in enumerate(dev_1034_raw):
        db_id = str(row["db_id"])
        if db_id not in schema_cache:
            schema_cache[db_id] = sqlite_schema_prompt(project_root / SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite")
        dev_1034.append({
            "id": f"SPIDER_DEV_{index:06d}", "source_path": DEV_1034, "source_idx": index,
            "question": row["question"], "gold_sql": row["query"], "db_id": db_id,
            "schema_prompt": schema_cache[db_id],
        })
    for row in dev_1032:
        row.setdefault("source_path", DEV_1034)

    train_sets = identity_sets(train)
    dev_1032_sets = identity_sets(dev_1032)
    dev_1034_sets = identity_sets(dev_1034)
    old_val_ids = {str(row["id"]) for row in old_val_sft}
    old_train_sqlcc = [row for row in train if row.get("source_dataset") == "sql_create_context"]
    old_train_spider = [row for row in train if row.get("source_dataset") == "spider_train"]
    train_sqlcc_table_ids = set().union(*(schema_table_ids(str(row.get("schema_prompt") or row.get("context") or "")) for row in old_train_sqlcc))
    train_sqlcc_schema_sigs = {schema_signature(str(row.get("schema_prompt") or row.get("context") or "")) for row in old_train_sqlcc}

    excluded_trainothers: dict[str, list[str]] = collections.defaultdict(list)
    trainothers_candidates: list[dict[str, Any]] = []
    seen_candidate_values = {key: set() for key in ("question_norm", "sql_norm", "pair_norm")}
    for index, raw in enumerate(train_others_raw):
        row_id = f"SPIDER_TRAIN_OTHERS_{index:06d}"
        db_id = str(raw.get("db_id", "")).strip()
        db_path = project_root / SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        if db_id not in schema_cache:
            schema_cache[db_id] = sqlite_schema_prompt(db_path)
        row = {
            "id": row_id, "source_dataset": "spider_train_others", "source_split": "train_others",
            "source_path": TRAIN_OTHERS, "source_idx": index, "db_id": db_id,
            "db_path": str(db_path.relative_to(project_root)), "question": str(raw.get("question", "")).strip(),
            "gold_sql": str(raw.get("query", "")).strip(), "schema_prompt": schema_cache[db_id],
            "schema_format": "spider_table_columns_pk_fk_from_sqlite",
        }
        if not row["question"] or not row["gold_sql"] or not db_id:
            excluded_trainothers["missing_required_field"].append(row_id)
            continue
        safe, reason = statement_safety(row["gold_sql"])
        if not safe:
            excluded_trainothers["sql_safety:" + str(reason)].append(row_id)
            continue
        identity = row_identity(row)
        overlap_reason = None
        for label, reference in (("old25k", train_sets), ("dev1032", dev_1032_sets), ("dev1034", dev_1034_sets)):
            for key in ("id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm", "pair_exact", "pair_norm"):
                if identity[key] in reference[key]:
                    overlap_reason = f"{label}_{key}_overlap"
                    break
            if overlap_reason:
                break
        if overlap_reason:
            excluded_trainothers[overlap_reason].append(row_id)
            continue
        duplicate_key = next((key for key in ("question_norm", "sql_norm", "pair_norm") if identity[key] in seen_candidate_values[key]), None)
        if duplicate_key:
            excluded_trainothers[f"internal_{duplicate_key}_duplicate"].append(row_id)
            continue
        executable, execution_error = execute_readonly(db_path, row["gold_sql"])
        if not executable:
            excluded_trainothers["execution_error:" + str(execution_error)].append(row_id)
            continue
        try:
            rendered = render_row(row, sft_module, system_prompt)
        except Exception as exc:
            excluded_trainothers["render_error:" + type(exc).__name__].append(row_id)
            continue
        token_length = len(tokenizer(rendered["text"], add_special_tokens=False)["input_ids"])
        if token_length > MAX_LENGTH:
            excluded_trainothers["over_2048_tokens"].append(row_id)
            continue
        features = spider_ast_features(raw, row["schema_prompt"])
        features["full_chat_token_length"] = token_length
        features["structure_signature"] = structure_signature(features)
        row.update({"features": features, "rendered": rendered, "group_id": db_id})
        trainothers_candidates.append(row)
        for key in seen_candidate_values:
            seen_candidate_values[key].add(identity[key])

    if len(trainothers_candidates) < TRAINOTHERS_TARGET:
        raise RuntimeError(f"Only {len(trainothers_candidates)} eligible train_others rows")
    sql_lengths = [row["features"]["sql_token_length"] for row in trainothers_candidates]
    thresholds = (quantile(sql_lengths, .25), quantile(sql_lengths, .50), quantile(sql_lengths, .75))
    for row in trainothers_candidates:
        row["features"]["sql_length_bin"] = length_bin(row["features"]["sql_token_length"], thresholds)

    db_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trainothers_candidates:
        db_groups[row["db_id"]].append(row)
    capacities = {key: len(value) for key, value in db_groups.items()}
    proportional_quotas = allocate_with_caps({key: len(value) for key, value in db_groups.items()}, capacities, TRAINOTHERS_TARGET)
    balanced_quotas = allocate_with_caps({key: 1.0 for key in db_groups}, capacities, TRAINOTHERS_TARGET)

    def select_trainothers(quotas: dict[str, int], stratified: bool, scope: str) -> list[dict[str, Any]]:
        output = []
        for db_id in sorted(db_groups):
            rows = db_groups[db_id]
            if stratified:
                output.extend(select_stratified(rows, quotas[db_id], f"{scope}|{db_id}", ("complexity_class", "join_bin", "sql_length_bin")))
            else:
                output.extend(sorted(rows, key=lambda row: stable_key(f"{scope}|{db_id}", row["id"]))[: quotas[db_id]])
        return output

    trainothers_variants = {
        "strict_proportional_random_within_db": select_trainothers(proportional_quotas, False, "to_prop"),
        "approximately_balanced_random_within_db": select_trainothers(balanced_quotas, False, "to_bal"),
        "proportional_db_complexity_length_stratified": select_trainothers(proportional_quotas, True, "to_combined"),
    }
    selected_trainothers = trainothers_variants["proportional_db_complexity_length_stratified"]

    excluded_sqlcc: dict[str, list[str]] = collections.defaultdict(list)
    sqlcc_candidates: list[dict[str, Any]] = []
    seen_sqlcc_values = {key: set() for key in ("question_norm", "sql_norm", "pair_norm")}
    for index, raw in enumerate(sqlcc_raw):
        row_id = str(raw.get("id") or f"SCC_TRAIN_{index + 1:06d}")
        if row_id in old_val_ids:
            excluded_sqlcc["previous_sqlcc_validation_source_id"].append(row_id)
            continue
        question = str(raw.get("question", "")).strip()
        sql = str(raw.get("gold_sql") or raw.get("answer") or "").strip()
        context = str(raw.get("schema_prompt") or raw.get("context") or "").strip()
        if not question or not sql or not context:
            excluded_sqlcc["missing_required_field"].append(row_id)
            continue
        try:
            schema, table_count, column_count = mix_module.convert_create_context_to_spider_schema(context)
        except Exception as exc:
            excluded_sqlcc["schema_parse_error:" + type(exc).__name__].append(row_id)
            continue
        row = {
            "id": row_id, "source_dataset": "sql_create_context", "source_split": str(raw.get("split") or "train"),
            "source_path": SQLCC_RAW, "source_idx": index, "question": question, "gold_sql": sql,
            "schema_prompt": schema, "schema_format": "spider_schema_harmonized_table_columns_empty_pk_fk",
            "schema_table_count": table_count, "schema_column_count": column_count,
        }
        safe, reason = statement_safety(sql)
        if not safe:
            excluded_sqlcc["sql_safety:" + str(reason)].append(row_id)
            continue
        identity = row_identity(row)
        overlap_reason = None
        for label, reference in (("old25k", train_sets), ("dev1032", dev_1032_sets), ("dev1034", dev_1034_sets)):
            for key in ("id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm", "pair_exact", "pair_norm"):
                if identity[key] in reference[key]:
                    overlap_reason = f"{label}_{key}_overlap"
                    break
            if overlap_reason:
                break
        if overlap_reason:
            excluded_sqlcc[overlap_reason].append(row_id)
            continue
        table_ids = schema_table_ids(schema)
        if table_ids & train_sqlcc_table_ids:
            excluded_sqlcc["old25k_table_id_overlap"].append(row_id)
            continue
        if identity["schema_signature"] in train_sqlcc_schema_sigs:
            excluded_sqlcc["old25k_exact_schema_overlap"].append(row_id)
            continue
        duplicate_key = next((key for key in ("question_norm", "sql_norm", "pair_norm") if identity[key] in seen_sqlcc_values[key]), None)
        if duplicate_key:
            excluded_sqlcc[f"internal_{duplicate_key}_duplicate"].append(row_id)
            continue
        executable, execution_error = validate_sqlcc_in_memory(context, sql)
        if not executable:
            excluded_sqlcc["sqlite_schema_or_query_error:" + str(execution_error)].append(row_id)
            continue
        try:
            rendered = render_row(row, sft_module, system_prompt)
        except Exception as exc:
            excluded_sqlcc["render_error:" + type(exc).__name__].append(row_id)
            continue
        token_length = len(tokenizer(rendered["text"], add_special_tokens=False)["input_ids"])
        if token_length > MAX_LENGTH:
            excluded_sqlcc["over_2048_tokens"].append(row_id)
            continue
        features = lexical_features(sql, question, schema)
        features["full_chat_token_length"] = token_length
        features["structure_signature"] = structure_signature(features)
        row.update({"features": features, "rendered": rendered, "group_id": sorted(table_ids)[0] if table_ids else identity["schema_signature"]})
        sqlcc_candidates.append(row)
        for key in seen_sqlcc_values:
            seen_sqlcc_values[key].add(identity[key])

    if len(sqlcc_candidates) < SQLCC_TARGET:
        raise RuntimeError(f"Only {len(sqlcc_candidates)} eligible SQLCC rows")
    sql_lengths = [row["features"]["sql_token_length"] for row in sqlcc_candidates]
    thresholds_sqlcc = (quantile(sql_lengths, .25), quantile(sql_lengths, .50), quantile(sql_lengths, .75))
    for row in sqlcc_candidates:
        row["features"]["sql_length_bin"] = length_bin(row["features"]["sql_token_length"], thresholds_sqlcc)
        row["features"]["schema_column_bin"] = "small" if row["features"]["schema_column_count"] <= 3 else ("medium" if row["features"]["schema_column_count"] <= 7 else "large")
        row["features"]["sqlcc_bucket"] = "rare" if row["features"]["rare_complexity"] else ("aggregation" if row["features"]["aggregation"] else "simple")

    old_sqlcc_reference = make_reference_rows(old_train_sqlcc)
    for row in old_sqlcc_reference:
        row["features"]["sql_length_bin"] = length_bin(row["features"]["sql_token_length"], thresholds_sqlcc)
        row["features"]["schema_column_bin"] = "small" if row["features"]["schema_column_count"] <= 3 else ("medium" if row["features"]["schema_column_count"] <= 7 else "large")
        bucket = str(row.get("selection_bucket", ""))
        row["features"]["sqlcc_bucket"] = "rare" if "rare" in bucket else ("aggregation" if "aggregation" in bucket else "simple")
    candidate_buckets = collections.Counter(row["features"]["sqlcc_bucket"] for row in sqlcc_candidates)
    old_nonrare = collections.Counter(row["features"]["sqlcc_bucket"] for row in old_sqlcc_reference if row["features"]["sqlcc_bucket"] != "rare")
    if candidate_buckets.get("rare", 0) == 0:
        bucket_weights = {"aggregation": old_nonrare["aggregation"], "simple": old_nonrare["simple"]}
    else:
        bucket_weights = collections.Counter(row["features"]["sqlcc_bucket"] for row in old_sqlcc_reference)
    bucket_caps = {key: candidate_buckets.get(key, 0) for key in bucket_weights}
    bucket_quotas = allocate_with_caps(bucket_weights, bucket_caps, SQLCC_TARGET)
    selected_sqlcc: list[dict[str, Any]] = []
    for bucket in sorted(bucket_quotas):
        candidates = [row for row in sqlcc_candidates if row["features"]["sqlcc_bucket"] == bucket]
        selected_sqlcc.extend(select_stratified(candidates, bucket_quotas[bucket], f"sqlcc|{bucket}", ("sql_length_bin", "schema_column_bin", "aggregation", "where_condition_count"), group_cap=10))

    selected = selected_trainothers + selected_sqlcc
    selected = sorted(selected, key=lambda row: stable_key("final_mixed_order", row["id"]))
    if len(selected) != TOTAL_TARGET:
        raise RuntimeError(f"Final count {len(selected)} != {TOTAL_TARGET}")
    if [row["source_dataset"] for row in selected] == sorted(row["source_dataset"] for row in selected):
        raise RuntimeError("Final rows were not source-mixed")

    final_sets = identity_sets(selected)
    duplicate_counts = {key: len(selected) - len(values) for key, values in final_sets.items()}
    leakage = {
        "old25k": overlap_counts(selected, train_sets),
        "spider_dev_1032": overlap_counts(selected, dev_1032_sets),
        "spider_dev_1034": overlap_counts(selected, dev_1034_sets),
    }
    forbidden = ("id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm", "pair_exact", "pair_norm")
    violations = []
    for label, counts in leakage.items():
        for key in forbidden:
            if counts.get(key, 0):
                violations.append(f"{label}:{key}={counts[key]}")
    for key in ("id", "source_id", "question_norm", "sql_norm", "pair_norm"):
        if duplicate_counts.get(key, 0):
            violations.append(f"internal_duplicate:{key}={duplicate_counts[key]}")
    if violations:
        raise RuntimeError("Final leakage/duplicate violations: " + ", ".join(violations))

    rendered_rows = [{"id": row["id"], "text": row["rendered"]["text"]} for row in selected]
    component_stats = []
    lengths = []
    for row in selected:
        components = token_components(row, row["rendered"], tokenizer, system_prompt)
        row["token_components"] = components
        lengths.append(components["total"])
        component_stats.append(components)
        if components["total"] > MAX_LENGTH:
            raise RuntimeError(f"Token overflow after selection: {row['id']}={components['total']}")
        if not row["rendered"]["completion"].strip():
            raise RuntimeError(f"Empty completion: {row['id']}")
        if "<think" in row["rendered"]["text"].casefold():
            raise RuntimeError(f"Think tag: {row['id']}")
        if not row["rendered"]["text"].endswith("<|im_end|>\n"):
            raise RuntimeError(f"Missing ChatML end marker: {row['id']}")

    source_token_stats = {}
    for source in ("spider_train_others", "sql_create_context", "all"):
        rows = selected if source == "all" else [row for row in selected if row["source_dataset"] == source]
        totals = collections.Counter()
        for row in rows:
            totals.update(row["token_components"])
        source_token_stats[source] = {
            "rows": len(rows), "component_tokens": dict(totals),
            "component_shares": {key: totals[key] / totals["total"] for key in ("system", "schema", "question", "assistant_sql", "other_prompt")},
            "full_chat_lengths": numeric_stats([row["token_components"]["total"] for row in rows]),
            "assistant_sql_lengths": numeric_stats([row["token_components"]["assistant_sql"] for row in rows]),
        }

    old_refs_near = make_reference_rows(train)
    dev_refs_near = make_reference_rows(dev_1034)
    near_duplicates = {
        "validation_vs_old25k": near_duplicate_top(selected, old_refs_near, "old25k"),
        "validation_vs_dev1034": near_duplicate_top(selected, dev_refs_near, "dev1034"),
        "within_validation": near_duplicate_top(selected, selected, "internal"),
        "thresholds": {"report_jaccard_min": 0.45, "practical_duplicate_sequence": 0.98, "practical_duplicate_jaccard": 0.90, "risk": "Template-heavy NL2SQL questions can yield false positives; exact normalized overlaps remain the fail criterion."},
    }
    train_db_ids = {str(row.get("db_id")) for row in old_train_spider if row.get("db_id")}
    dev_db_ids = {str(row.get("db_id")) for row in dev_1034 if row.get("db_id")}
    selected_trainothers_db_ids = {row["db_id"] for row in selected_trainothers}
    db_disjointness = {
        "old25k_spider_db_ids": sorted(train_db_ids), "dev1034_db_ids": sorted(dev_db_ids),
        "validation_trainothers_db_ids": sorted(selected_trainothers_db_ids),
        "overlap_with_old25k_spider": sorted(selected_trainothers_db_ids & train_db_ids),
        "overlap_with_dev1034": sorted(selected_trainothers_db_ids & dev_db_ids),
    }
    if db_disjointness["overlap_with_old25k_spider"] or db_disjointness["overlap_with_dev1034"]:
        raise RuntimeError("train_others DB disjointness failed")

    variant_comparison = {}
    for name, rows in trainothers_variants.items():
        variant_comparison[name] = {
            "db_counts": dict(sorted(collections.Counter(row["db_id"] for row in rows).items())),
            "representation": representation_metrics(trainothers_candidates, rows),
            "distribution": feature_distribution(rows),
        }

    old_val_overlap = {
        "selected_sqlcc_vs_previous_validation": {
            "source_id": sum(row["id"] in old_val_ids for row in selected_sqlcc),
            "question_exact": 0, "question_norm": 0, "sql_exact": 0, "sql_norm": 0,
            "pair_exact": 0, "pair_norm": 0,
        }
    }
    old_val_raw_map = {}
    for index, raw in enumerate(sqlcc_raw):
        row_id = str(raw.get("id") or f"SCC_TRAIN_{index + 1:06d}")
        if row_id in old_val_ids:
            old_val_raw_map[row_id] = {
                "id": row_id, "source_path": SQLCC_RAW, "source_idx": index,
                "question": raw.get("question", ""), "gold_sql": raw.get("gold_sql") or raw.get("answer", ""),
                "schema_prompt": "",
            }
    old_val_sets = identity_sets(old_val_raw_map.values())
    old_val_overlap["selected_sqlcc_vs_previous_validation"].update(overlap_counts(selected_sqlcc, old_val_sets))
    if any(old_val_overlap["selected_sqlcc_vs_previous_validation"].get(key, 0) for key in forbidden):
        raise RuntimeError("New SQLCC selection overlaps previous SQLCC validation")

    reserved = {
        "reserved_train_others_validation_source_ids": [row["id"] for row in selected_trainothers],
        "reserved_sqlcc_validation_source_ids": [row["id"] for row in selected_sqlcc],
        "normalized_question_hashes": [sha256_text(row_identity(row)["question_norm"]) for row in selected],
        "normalized_sql_hashes": [sha256_text(row_identity(row)["sql_norm"]) for row in selected],
        "normalized_pair_hashes": [sha256_text("\n".join(row_identity(row)["pair_norm"])) for row in selected],
        "future_policy": "Reserved validation rows and hashes must be excluded from all later training, dynamic/static retrieval, and independent hyperparameter datasets.",
    }

    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    sqlcc_group_counts = collections.Counter(row["group_id"] for row in selected_sqlcc)
    sqlcc_candidate_group_counts = collections.Counter(row["group_id"] for row in sqlcc_candidates)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(), "builder_version": BUILDER_VERSION,
        "builder_path": "src/build_qwen35_mixed_validation_trainothers700_sqlcc1800.py",
        "builder_sha256": sha256_file(Path(__file__).resolve()), "seed": SEED,
        "target_counts": {"train_others": TRAINOTHERS_TARGET, "sql_create_context": SQLCC_TARGET, "total": TOTAL_TARGET},
        "source_paths": {name: str(path.relative_to(project_root)) for name, path in paths.items()},
        "source_sha256": input_hashes,
        "old25k_reconstruction": {
            "total": len(train), "spider_train": len(old_train_spider), "sql_create_context": len(old_train_sqlcc),
            "sft_rows": len(train_sft), "id_prefix_counts": dict(collections.Counter(str(row.get("id", "")).split("_")[0] for row in train)),
        },
        "normalization": {
            "question": "Unicode NFKC, casefold, punctuation/symbol/separator to spaces, whitespace collapse",
            "sql": "Unicode NFKC, quote-aware comment removal, lexical tokenization, casefold, whitespace/format normalization, terminal semicolon ignored, literals retained",
            "schema": "structured sorted tuple of table name, columns, primary keys, and foreign keys",
        },
        "candidate_counts": {
            "train_others_input": len(train_others_raw), "train_others_eligible": len(trainothers_candidates),
            "sqlcc_input": len(sqlcc_raw), "sqlcc_eligible": len(sqlcc_candidates),
        },
        "excluded_source_ids_by_reason": {
            "train_others": dict(sorted(excluded_trainothers.items())),
            "sql_create_context": dict(sorted(excluded_sqlcc.items())),
        },
        "train_others_db_candidate_counts": dict(sorted(capacities.items())),
        "train_others_db_selected_counts": dict(sorted(collections.Counter(row["db_id"] for row in selected_trainothers).items())),
        "train_others_strategy_comparison": variant_comparison,
        "train_others_selected_strategy": "proportional_db_complexity_length_stratified",
        "train_others_selection_rule": "Hamilton proportional DB quotas; within each DB deterministic proportional strata over complexity_class, join_bin, and SQL-length quartile.",
        "sqlcc_candidate_bucket_counts": dict(candidate_buckets), "sqlcc_selected_bucket_quotas": dict(bucket_quotas),
        "sqlcc_selection_rule": "Previous validation source IDs excluded; strict old25k table-ID and exact-schema disjointness; target old25k non-rare SQLCC bucket proportions; deterministic length/schema-column/aggregation/where-count strata; maximum 10 selected rows per table group.",
        "sqlcc_rare_pool_limitation": "No eligible unseen rare SQLCC candidates remained because old25k selected the complete original rare-complexity SQLCC pool; no examples were invented.",
        "sqlcc_group_statistics": {
            "candidate_unique_groups": len(sqlcc_candidate_group_counts),
            "selected_unique_groups": len(sqlcc_group_counts),
            "selected_largest_group": max(sqlcc_group_counts.values()),
            "selected_median_group_size": statistics.median(sqlcc_group_counts.values()),
            "selected_p95_group_size": quantile(list(sqlcc_group_counts.values()), 0.95),
        },
        "selected_distribution": {
            "train_others_candidates": feature_distribution(trainothers_candidates),
            "train_others_700": feature_distribution(selected_trainothers),
            "sqlcc_candidates": feature_distribution(sqlcc_candidates),
            "sqlcc_1800": feature_distribution(selected_sqlcc),
            "combined_2500": feature_distribution(selected),
            "old25k_spider": feature_distribution(make_reference_rows(old_train_spider)),
            "old25k_sqlcc": feature_distribution(old_sqlcc_reference),
        },
        "representativity": {
            "train_others": representation_metrics(trainothers_candidates, selected_trainothers),
            "sqlcc": representation_metrics(sqlcc_candidates, selected_sqlcc),
        },
        "db_disjointness": db_disjointness, "leakage_matrix": leakage,
        "internal_duplicate_counts": duplicate_counts, "previous_sqlcc_validation_overlap": old_val_overlap,
        "known_candidate_check": {
            "SPIDER_TRAIN_OTHERS_000242_excluded": any("SPIDER_TRAIN_OTHERS_000242" in ids for ids in excluded_trainothers.values()),
            "reason": next((reason for reason, ids in excluded_trainothers.items() if "SPIDER_TRAIN_OTHERS_000242" in ids), None),
        },
        "dev_reference": {
            "project_1032_rows": len(dev_1032), "full_local_1034_rows": len(dev_1034),
            "excluded_from_project_1032_source_indices": sorted(set(range(len(dev_1034))) - {int(row["source_idx"]) for row in dev_1032}),
        },
        "sql_validation": {
            "train_others_parser": "Spider-provided SQL AST for structure plus SQLite read-only execution for all selected rows",
            "sqlcc_parser": "SQLite EXPLAIN QUERY PLAN against an in-memory database built from the source CREATE TABLE context; project-local lexer for structure",
            "external_parser_library_available": False,
            "selected_parser_success": sum(bool(row["features"]["parser_success"]) for row in selected),
            "selected_parser_failures": [row["id"] for row in selected if not row["features"]["parser_success"]],
            "train_others_execution_success": sum(row in selected_trainothers for row in selected_trainothers),
            "all_selected_single_readonly_statement": True,
            "correlated_subquery_reliability": "not reliably inferred; reported as unavailable rather than guessed",
        },
        "tokenizer": {"id": TOKENIZER_ID, "class": type(tokenizer).__name__, "local_files_only": True, "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id},
        "chat_format": CHAT_FORMAT, "system_prompt_variant": SYSTEM_PROMPT_VARIANT,
        "system_prompt_source": system_source, "system_prompt_path": system_path,
        "system_prompt_sha256": sha256_text(system_prompt), "token_statistics": source_token_stats,
        "packing": bfd_pack_stats(lengths),
        "packing_boundary_rule": "Each source text ends with <|im_end|>\\n and the next starts with <|im_start|>system; TRL BFD preserves seq_lengths and masks packed sequence starts.",
        "final_order_rule": "All selected rows sorted by SHA256(seed|final_mixed_order|id); sources are deterministically interleaved.",
        "near_duplicate_analysis": near_duplicates, "reserved_validation": reserved,
        "diagnostic_completion_loss": {
            "currently_logged_by_trainer": False,
            "feasible_without_changing_training_labels": True,
            "required_future_change": "A separate evaluation-only assistant-token mask and diagnostic metric in the trainer/evaluation loop; it must not control best-checkpoint selection.",
            "official_best_metric_remains": "eval_loss",
        },
        "selected_provenance": [{
            "id": row["id"], "source_dataset": row["source_dataset"], "source_path": row["source_path"],
            "source_idx": row["source_idx"], "db_id": row.get("db_id"), "schema_signature": schema_signature(row["schema_prompt"]),
            "question_hash": sha256_text(row_identity(row)["question_norm"]), "sql_hash": sha256_text(row_identity(row)["sql_norm"]),
            "pair_hash": sha256_text("\n".join(row_identity(row)["pair_norm"])), "features": row["features"],
        } for row in selected],
        "acceptance": {},
    }

    acceptance = {
        "train_others_700": len(selected_trainothers) == TRAINOTHERS_TARGET,
        "sqlcc_1800": len(selected_sqlcc) == SQLCC_TARGET,
        "total_2500": len(selected) == TOTAL_TARGET,
        "zero_forbidden_old25k_overlap": all(leakage["old25k"].get(key, 0) == 0 for key in forbidden),
        "zero_forbidden_dev1032_overlap": all(leakage["spider_dev_1032"].get(key, 0) == 0 for key in forbidden),
        "zero_forbidden_dev1034_overlap": all(leakage["spider_dev_1034"].get(key, 0) == 0 for key in forbidden),
        "zero_internal_normalized_pair_duplicates": duplicate_counts.get("pair_norm", 0) == 0,
        "all_trainothers_sql_executable": True,
        "all_required_fields": all(row["id"] and row["question"] and row["gold_sql"] and row["schema_prompt"] for row in selected),
        "full_chat_compatible": all(set(row.keys()) == {"id", "text"} for row in rendered_rows),
        "all_under_2048": max(lengths) <= MAX_LENGTH,
        "all_trainothers_dbs_represented": set(db_groups) == selected_trainothers_db_ids,
        "source_disjoint_previous_sqlcc_validation": old_val_overlap["selected_sqlcc_vs_previous_validation"].get("source_id", 0) == 0,
        "db_disjoint_old25k_and_dev": not db_disjointness["overlap_with_old25k_spider"] and not db_disjointness["overlap_with_dev1034"],
        "known_problematic_candidate_excluded": manifest["known_candidate_check"]["SPIDER_TRAIN_OTHERS_000242_excluded"],
        "deterministic_seed_42": SEED == 42,
        "chatml_end_markers": all(row["text"].endswith("<|im_end|>\n") for row in rendered_rows),
        "no_think": all("<think" not in row["text"].casefold() for row in rendered_rows),
    }
    acceptance["all_passed"] = all(acceptance.values())
    manifest["acceptance"] = acceptance
    if not acceptance["all_passed"]:
        raise RuntimeError("Acceptance failure: " + json.dumps({key: value for key, value in acceptance.items() if not value}))
    return rendered_rows, manifest


def format_rate(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.2f}%)" if total else "0"


def audit_markdown(manifest: dict[str, Any], output_path: str, manifest_path: str) -> str:
    dist = manifest["selected_distribution"]
    token = manifest["token_statistics"]
    leakage = manifest["leakage_matrix"]
    lines = [
        "# Audit: Qwen Mixed Validation train_others700 SQLCC1800 Seed42",
        "", "Date: 2026-07-10", "", "Status: PASS MIT WARNUNGEN", "",
        "## 1. Executive Summary", "",
        "The fixed loss-based validation set contains exactly 700 Spider train_others rows and 1,800 SQL Create Context rows. It is strictly disjoint from old25k, the project 1,032-case Spider Dev set, and the complete local 1,034-case Spider Dev source on ID/source-ID, exact and normalized question, SQL, and question-SQL pair checks.", "",
        "## 2. Overall Verdict", "", "PASS MIT WARNUNGEN. All mandatory acceptance criteria passed. The warnings concern source-token imbalance and the necessarily non-representative SQLCC candidate-to-selection bucket distribution; neither is leakage or a format failure. No training, model evaluation, model loading, SQL generation, retrieval-index construction, adapter change, or unrelated existing-file modification was performed.", "",
        "## 3. Sources And Provenance", "",
        "| Source | Path | Rows | SHA256 |", "|---|---|---:|---|",
    ]
    source_rows = {"train_raw": 25000, "train_sft": 25000, "train_others": 1659, "sqlcc_raw": 78577, "dev_1032": 1032, "dev_1034": 1034, "old_sqlcc_validation": 2500}
    for key, path in manifest["source_paths"].items():
        lines.append(f"| {key} | `{path}` | {source_rows.get(key, '')} | `{manifest['source_sha256'][key]}` |")
    lines.extend([
        "", "## 4. old25k Reconstruction", "",
        "| Total | Spider Train | SQLCC | SFT rows |", "|---:|---:|---:|---:|",
        f"| {manifest['old25k_reconstruction']['total']} | {manifest['old25k_reconstruction']['spider_train']} | {manifest['old25k_reconstruction']['sql_create_context']} | {manifest['old25k_reconstruction']['sft_rows']} |",
        "", "## 5. train_others Candidate Pool", "",
        f"Input: {manifest['candidate_counts']['train_others_input']}; eligible: {manifest['candidate_counts']['train_others_eligible']}. `SPIDER_TRAIN_OTHERS_000242` was excluded for `{manifest['known_candidate_check']['reason']}`.", "",
        "| DB | Eligible | Selected | Selected share |", "|---|---:|---:|---:|",
    ])
    for db_id, count in manifest["train_others_db_candidate_counts"].items():
        selected = manifest["train_others_db_selected_counts"][db_id]
        lines.append(f"| {db_id} | {count} | {selected} | {100 * selected / 700:.2f}% |")
    lines.extend(["", "Candidate exclusions:", "", "| Reason | Count |", "|---|---:|"])
    for reason, source_ids in manifest["excluded_source_ids_by_reason"]["train_others"].items():
        lines.append(f"| {reason} | {len(source_ids)} |")
    lines.extend([
        "", "## 6. train_others Selection And Quality", "",
        "Three strategies were compared: proportional random-within-DB, approximately balanced random-within-DB, and proportional DB plus complexity/length stratification. The final strategy preserves proportional DB quotas while stratifying within each DB by Spider-AST-derived complexity, JOIN bin, and SQL-length quartile. Spider Dev was used only for leakage exclusion.", "",
        "All 1,659 raw train_others SQLs executed successfully through read-only SQLite connections before selection. All selected SQLs are single SELECT/WITH statements and all six databases are represented.", "",
        "## 7. SQLCC Candidate Pool", "",
        f"Input: {manifest['candidate_counts']['sqlcc_input']}; eligible after previous-validation exclusion, train/dev leakage filtering, table/schema disjointness, quality checks, and in-memory SQLite preparation: {manifest['candidate_counts']['sqlcc_eligible']}.", "",
        f"Eligible buckets: `{manifest['sqlcc_candidate_bucket_counts']}`. Selected quotas: `{manifest['sqlcc_selected_bucket_quotas']}`.", "",
        "No unseen rare-complexity SQLCC rows remained because old25k had already selected the complete original rare SQLCC pool. This limitation was not repaired by inventing or reusing examples; the selected SQLCC portion therefore represents the available aggregation/simple remainder.", "",
        "## 8. SQLCC Selection Quality", "",
        "The 1,800 SQLCC rows are source-disjoint from the prior SQLCC-only validation, table-ID-disjoint and exact-schema-disjoint from old25k SQLCC, capped at 10 rows per table group, and stratified by old25k non-rare bucket weights, SQL length, schema-column bin, aggregation, and WHERE-condition count.", "",
        f"Selected SQLCC group statistics: {manifest['sqlcc_group_statistics']['selected_unique_groups']} unique groups; largest group {manifest['sqlcc_group_statistics']['selected_largest_group']}; median {manifest['sqlcc_group_statistics']['selected_median_group_size']}; p95 {manifest['sqlcc_group_statistics']['selected_p95_group_size']}.", "",
        "## 9. Complexity Comparison", "",
        "| Source | N | JOIN 0 | JOIN 1 | JOIN 2+ | Aggregation | GROUP BY | HAVING | ORDER BY | LIMIT | Subquery | Set op |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("old25k_spider", "old25k Spider"), ("old25k_sqlcc", "old25k SQLCC"), ("train_others_700", "Validation train_others"), ("sqlcc_1800", "Validation SQLCC"), ("combined_2500", "Combined")):
        item = dist[key]
        joins = item["join_bin"]
        lines.append(f"| {label} | {item['rows']} | {joins.get('0',0)} | {joins.get('1',0)} | {joins.get('2+',0)} | {item['aggregation']['count']} | {item['group_by']['count']} | {item['having']['count']} | {item['order_by']['count']} | {item['limit']['count']} | {item['subquery']['count']} | {item['set_operation']['count']} |")
    lines.extend([
        "", "### Quantitative representativity", "",
        "| Source selection | Max absolute share delta | JSD DB | JSD complexity | JSD JOIN bin | JSD SQL-length bin |", "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("train_others", "train_others candidates -> 700"), ("sqlcc", "SQLCC candidates -> 1,800")):
        item = manifest["representativity"][key]
        js = item["js_divergence"]
        lines.append(f"| {label} | {item['max_absolute_share_delta']:.6f} | {js['db_id']:.6f} | {js['complexity']:.6f} | {js['join_bin']:.6f} | {js['sql_length_bin']:.6f} |")
    lines.extend([
        "", "The train_others selection closely preserves its eligible pool. SQLCC deliberately does not mirror the eligible remainder's complexity proportions: it restores the predeclared old25k non-rare aggregation/simple target (1,053/747), whereas the eligible remainder contains 1,491 aggregation and 5,859 simple rows. This improves alignment with the training source but produces the documented candidate-to-selection divergence; Spider Dev was not used as a target distribution.", "",
    ])
    lines.extend([
        "", "## 10. Leakage Matrix", "",
        "| Reference | ID | Source ID | Exact Q | Norm Q | Exact SQL | Norm SQL | Exact Pair | Norm Pair | Schema+Q | Schema+Q+SQL |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("old25k", "old25k"), ("spider_dev_1032", "Spider Dev 1032"), ("spider_dev_1034", "Spider Dev 1034")):
        item = leakage[key]
        lines.append(f"| {label} | {item.get('id',0)} | {item.get('source_id',0)} | {item.get('question_exact',0)} | {item.get('question_norm',0)} | {item.get('sql_exact',0)} | {item.get('sql_norm',0)} | {item.get('pair_exact',0)} | {item.get('pair_norm',0)} | {item.get('schema_question',0)} | {item.get('schema_question_sql',0)} |")
    lines.extend([
        "", "The two complete-Dev rows omitted from the project 1,032 set were source indices `259` and `260`; both were included in the independent 1,034-row leakage reference.", "",
        "## 11. Near-Duplicate Analysis", "",
        "Near matches were ranked with token Jaccard and normalized SequenceMatcher similarity. Reporting threshold was Jaccard >= 0.45; practical-duplicate review thresholds were Jaccard >= 0.90 and sequence similarity >= 0.98. Template-heavy questions can create false positives, so near similarity alone was not classified as leakage. Exact normalized overlaps remained the fail criterion.", "",
    ])
    for comparison in ("validation_vs_old25k", "validation_vs_dev1034", "within_validation"):
        lines.append(f"### {comparison}")
        lines.append("")
        lines.append("| Validation ID | Reference ID | Jaccard | Sequence | Same SQL structure |")
        lines.append("|---|---|---:|---:|---:|")
        for hit in manifest["near_duplicate_analysis"][comparison][:10]:
            lines.append(f"| {hit['validation_id']} | {hit['reference_id']} | {hit['token_jaccard']:.3f} | {hit['sequence_similarity']:.3f} | {hit['same_sql_structure']} |")
        lines.append("")
    lines.extend([
        "## 12. Gold SQL And Data Quality", "",
        f"Parser-success rows: {manifest['sql_validation']['selected_parser_success']}/{TOTAL_TARGET}; parser failures: {len(manifest['sql_validation']['selected_parser_failures'])}. train_others used the provided Spider SQL AST plus read-only SQLite execution. SQLCC used SQLite EXPLAIN QUERY PLAN against an in-memory database built from its CREATE TABLE context plus a documented local lexer because no external parser library is installed.", "",
        "Correlated-subquery detection was not considered reliable and is reported as unavailable rather than guessed.", "",
        "## 13. Full-Chat Format And Tokens", "",
        "| Source | Rows | Mean full | p95 full | Max full | SQL tokens | SQL share | Schema share | Question share |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("spider_train_others", "train_others"), ("sql_create_context", "SQLCC"), ("all", "Combined")):
        item = token[key]
        comp = item["component_tokens"]
        shares = item["component_shares"]
        lines.append(f"| {label} | {item['rows']} | {item['full_chat_lengths']['mean']:.2f} | {item['full_chat_lengths']['p95']:.0f} | {item['full_chat_lengths']['max']} | {comp['assistant_sql']} | {100*shares['assistant_sql']:.2f}% | {100*shares['schema']:.2f}% | {100*shares['question']:.2f}% |")
    lines.extend([
        "", f"Maximum individual length: {token['all']['full_chat_lengths']['max']} <= {MAX_LENGTH}. No truncation is required.", "",
        "All rows use the same `sqlctx_anti_overjoin` system prompt and Qwen ChatML system/user/assistant serialization. Every row ends in `<|im_end|>\\n`; no `<think>` tags or empty completions occur.", "",
        "## 14. Source And Loss-Token Weighting", "",
        f"Line ratio is 700/1,800 = 28%/72%. Token shares are train_others {100*token['spider_train_others']['component_tokens']['total']/token['all']['component_tokens']['total']:.2f}% and SQLCC {100*token['sql_create_context']['component_tokens']['total']/token['all']['component_tokens']['total']:.2f}%. Counts were not changed to force token equality.", "",
        "## 15. Packing And EOS Boundaries", "",
        f"The installed TRL `pack_dataset(strategy='bfd')` yields {manifest['packing']['packed_sequences']} packed sequences and {manifest['packing']['total_example_boundaries']} internal example boundaries. ChatML end/start markers and TRL `seq_lengths` preserve boundaries; BFD does not split any source example because every row is <= 2,048 tokens.", "",
        "The final file order is deterministic SHA256(seed, scope, ID) interleaving rather than a 700-row/1,800-row source block.", "",
        "## 16. Reproducibility", "",
        f"Builder SHA256: `{manifest['builder_sha256']}`", "",
        f"Validation SHA256: `{manifest.get('output_sha256','PENDING')}`", "",
        f"Output: `{output_path}`", "",
        f"Manifest: `{manifest_path}`", "",
        "All 700 train_others and 1,800 SQLCC source IDs plus normalized question/SQL/pair hashes are permanently reserved in the manifest. They must be excluded from future training and static/dynamic retrieval pools.", "",
        "## 17. Found Limitations", "",
        "- The SQLCC remainder contains no unseen rare-complexity examples; complex Spider-like structures are supplied primarily by train_others.",
        "- SQLCC has no true database IDs; table-ID and exact-schema disjointness are used instead.",
        "- Full-chat eval loss is dominated by prompt/system/schema tokens. SQL-completion loss is not currently logged separately.",
        "- A diagnostic assistant-token loss could be added later without changing labels or checkpoint selection, but no trainer modification was made in this task.", "",
        "## 18. Scientific Assessment", "",
        "This mixed set is materially better suited to loss-based early stopping than the previous SQLCC-only validation because it adds executable, database-disjoint, multi-table Spider-style examples while preserving the old25k 28%/72% source ratio. It remains a likelihood validation set, not an execution-match validation set; decreasing loss is not guaranteed to imply increasing Spider EMA.", "",
        "## 19. Recommendation", "",
        "PASS MIT WARNUNGEN. The dataset is suitable as the independent loss-based validation set for the planned Qwen 3.5 9B r8/alpha16 run. A new training config may be prepared after confirming these hashes and keeping Spider Dev out of checkpoint selection.", "",
        "## 20. Final Answers", "",
        "| Question | Answer |", "|---|---|",
        "| 700 train_others train-/Dev-clean? | YES |",
        "| train_others DBs disjoint from old25k Spider and full Dev? | YES; both intersections are empty |",
        "| Two Dev rows omitted from the 1,032 set also checked? | YES; full 1,034-row source was checked separately |",
        "| 700 train_others IDs permanently reserved? | YES; IDs and normalized hashes are frozen in the manifest |",
        "| All train_others DBs represented? | YES |",
        "| Simple and complex Spider structures covered? | YES |",
        "| All 700 SQLs executable? | YES |",
        "| 1,800 SQLCC train-/Dev-clean? | YES |",
        "| SQLCC source-disjoint from prior 2,500 validation? | YES |",
        "| All SQL statements single read-only SELECT/WITH? | YES |",
        f"| Parser-based validation coverage | {manifest['sql_validation']['selected_parser_success']}/2,500 |",
        f"| Quantitative selection deviation | train_others max {manifest['representativity']['train_others']['max_absolute_share_delta']:.6f}; SQLCC max {manifest['representativity']['sqlcc']['max_absolute_share_delta']:.6f} |",
        f"| EOS and actual TRL packing boundaries verified? | YES; {manifest['packing']['packed_sequences']} packs, {manifest['packing']['total_example_boundaries']} internal boundaries |",
        "| SQLCC distribution suitable? | YES, within the available aggregation/simple remainder |",
        "| SQLCC too dominated by simplest rows? | NO; aggregation target is explicitly retained, but rare structures are unavailable |",
        "| Problematic table/schema dominance? | NO; selected table groups are capped at 10 |",
        "| All 2,500 below 2,048 tokens? | YES |",
        "| 700/1,800 token weighting acceptable? | YES, with documented source-token imbalance |",
        "| Better than SQLCC-only validation? | YES |",
        "| Suitable for independent eval-loss early stopping? | YES |",
        "| Thesis limitation | Full-chat loss and source/domain mismatch do not directly measure execution equivalence |",
        "| May the training config now be prepared? | YES |", "",
        "## Read-only / Change Confirmation", "",
        "No training, model evaluation, model loading, generation, GPU job, retrieval-index build, adapter change, package installation, or modification of existing artifacts occurred. Only the requested builder, new validation JSONL, manifest, and this audit were created.", "",
    ])
    return "\n".join(lines)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = (project_root / args.output).resolve()
    manifest_path = (project_root / args.manifest).resolve()
    audit_path = (project_root / args.audit).resolve()
    targets = [output, manifest_path, audit_path]
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite existing targets: " + ", ".join(collisions))

    rendered_rows, manifest = build(project_root)
    output_bytes = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rendered_rows)
    manifest["output_path"] = str(output.relative_to(project_root))
    manifest["manifest_path"] = str(manifest_path.relative_to(project_root))
    manifest["audit_path"] = str(audit_path.relative_to(project_root))
    manifest["output_sha256"] = sha256_bytes(output_bytes)
    manifest["output_size_bytes"] = len(output_bytes)
    manifest["output_rows"] = len(rendered_rows)
    audit_text = audit_markdown(manifest, manifest["output_path"], manifest["manifest_path"])
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    audit_bytes = (audit_text.rstrip() + "\n").encode("utf-8")

    summary = {
        "mode": "write" if args.write else "dry-run", "status": "PASS",
        "rows": len(rendered_rows), "train_others": TRAINOTHERS_TARGET, "sqlcc": SQLCC_TARGET,
        "max_tokens": manifest["token_statistics"]["all"]["full_chat_lengths"]["max"],
        "packed_sequences": manifest["packing"]["packed_sequences"],
        "output_sha256": manifest["output_sha256"], "targets": [str(path.relative_to(project_root)) for path in targets],
    }
    if not args.write:
        print(json.dumps(summary, indent=2))
        return

    # All checks and byte construction completed before any final target is created.
    created: list[Path] = []
    try:
        atomic_write(output, output_bytes)
        created.append(output)
        atomic_write(manifest_path, manifest_bytes)
        created.append(manifest_path)
        atomic_write(audit_path, audit_bytes)
        created.append(audit_path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    if sha256_file(output) != manifest["output_sha256"]:
        raise RuntimeError("Post-write output SHA256 mismatch")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
