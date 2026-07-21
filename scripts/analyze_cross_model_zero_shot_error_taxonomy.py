#!/usr/bin/env python3
"""Deterministic read-only cross-model zero-shot SQL error analysis.

The script reads frozen evaluation artifacts, opens Spider databases in SQLite
read-only mode, and writes only the additive artifacts dated 20260716. It does
not import or load models, adapters, tokenizers, retrievers, or network clients.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260716"
SEED = 20260716
N = 1032
BOOTSTRAP_RESAMPLES = 10_000
TAXONOMY_VERSION = "cross-model-sql-error-taxonomy-v1.0-20260716"
PARSER_NAME = "project-local-sqlite-clause-fallback"
PARSER_VERSION = "1.0.0"

CROSS_AUDIT = "audits/audit_cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_20260716.md"
CROSS_MANIFEST = "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json"
CROSS_TABLE = "audits/derived/cross_model_complete_48_run_results_20260716.csv"
CROSS_SCRIPT = "scripts/analyze_cross_model_complete_8x8_synthesis.py"
EXPECTED = {
    CROSS_AUDIT: "4304813c8b5fc6a87c62291b2c6c4ff90b747d43dfb217bb07fe4db6d2513b74",
    CROSS_MANIFEST: "24b4dec07d2d4981b42ce22e1295d27b0ccd9cbcc10666a422118b267fd14e37",
    CROSS_TABLE: "f051867b5d7ce599d4e8a3a9ffb45448c9d9e715a106fee0696199ea94c7f7d2",
    CROSS_SCRIPT: "95dbda1932ec957bba9fa54c3ddbb8963ded1d7d974ced492e5199d3fd6b6475",
}

MODEL_ORDER = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B",
    "llama3b": "Llama 3.2 3B Instruct",
    "qwen9b": "Qwen 3.5 9B",
}
MODEL_INFO = {
    "qwen2b": {
        "model_id": "Qwen/Qwen3.5-2B-Base",
        "snapshot": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
        "adapter_root": "adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "adapter_sha256": "6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3",
    },
    "llama3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "snapshot": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "adapter_root": "adapters/llama32_3b_instruct/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "adapter_sha256": "fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d",
    },
    "qwen9b": {
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "snapshot": "68c46c4b3498877f3ef123c856ecfde50c39f404",
        "adapter_root": "adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "adapter_sha256": "dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0",
    },
}

OUT = {
    "audit": ROOT / "audits/audit_cross_model_zero_shot_error_analysis_20260716.md",
    "manifest": ROOT / "audits/cross_model_zero_shot_error_analysis_manifest_20260716.json",
    "cases": ROOT / "audits/derived/cross_model_zero_shot_error_analysis_cases_20260716.csv",
    "labels": ROOT / "audits/derived/cross_model_zero_shot_error_labels_long_20260716.csv",
    "transitions": ROOT / "audits/derived/cross_model_zero_shot_error_transition_summary_20260716.csv",
    "categories": ROOT / "audits/derived/cross_model_zero_shot_error_category_summary_20260716.csv",
    "statistics": ROOT / "audits/derived/cross_model_zero_shot_error_category_statistics_20260716.csv",
    "alternative": ROOT / "audits/derived/cross_model_zero_shot_alternative_valid_sql_20260716.csv",
    "review": ROOT / "audits/derived/cross_model_zero_shot_error_manual_review_sample_20260716.csv",
    "blind": ROOT / "audits/derived/cross_model_zero_shot_error_double_coding_blinded_20260716.csv",
    "key": ROOT / "audits/derived/cross_model_zero_shot_error_double_coding_key_20260716.csv",
    "fewshot": ROOT / "audits/derived/cross_model_fewshot_harm_error_analysis_20260716.csv",
    "thesis": ROOT / "audits/derived/cross_model_error_analysis_thesis_ready_tables_20260716.md",
    "codebook": ROOT / "audits/error_analysis_codebook_20260716.md",
}
PLOTS = {
    "transitions": ROOT / "audits/plots/cross_model_error_transitions_20260716",
    "families": ROOT / "audits/plots/cross_model_error_families_base_vs_lora_20260716",
    "changes": ROOT / "audits/plots/cross_model_errors_repaired_vs_introduced_20260716",
    "coarse": ROOT / "audits/plots/cross_model_technical_vs_semantic_errors_20260716",
    "fewshot": ROOT / "audits/plots/cross_model_fewshot_harm_errors_20260716",
}

FAMILY_LABELS = {
    "OUTPUT_CONTROL": ["EMPTY_RAW_OUTPUT", "EMPTY_SQL_EXTRACTION", "NO_TERMINATING_SEMICOLON", "MULTIPLE_STATEMENTS", "MARKDOWN_OR_EXPLANATION", "THINK_MARKER", "REPETITIVE_GENERATION", "COMPLETION_LIMIT_REACHED", "TRUNCATED_SQL", "EXTRACTOR_FAILURE", "CHAT_TEMPLATE_ARTIFACT", "LONGER_OR_UNSTABLE_OUTPUT"],
    "EXECUTION_SYNTAX": ["SQL_PARSE_ERROR", "SQL_SYNTAX_ERROR", "SQLITE_EXECUTION_ERROR", "UNKNOWN_TABLE", "UNKNOWN_COLUMN", "AMBIGUOUS_COLUMN", "TYPE_OR_FUNCTION_ERROR", "OTHER_EXECUTION_ERROR"],
    "SCHEMA_LINKING": ["WRONG_TABLE", "MISSING_TABLE", "EXTRA_TABLE", "WRONG_COLUMN", "MISSING_COLUMN", "EXTRA_COLUMN", "WRONG_SCHEMA_LINK"],
    "PROJECTION": ["WRONG_SELECT_EXPRESSION", "MISSING_SELECT_EXPRESSION", "EXTRA_SELECT_EXPRESSION", "WRONG_SELECT_CARDINALITY", "WRONG_DISTINCT", "MISSING_DISTINCT", "EXTRA_DISTINCT"],
    "AGGREGATION": ["WRONG_AGGREGATION", "MISSING_AGGREGATION", "EXTRA_AGGREGATION", "WRONG_COUNT_TARGET", "WRONG_MIN_MAX", "WRONG_SUM_AVG"],
    "JOIN": ["MISSING_JOIN", "EXTRA_JOIN", "WRONG_JOIN_TABLE", "WRONG_JOIN_PATH", "WRONG_JOIN_CONDITION", "CARTESIAN_PRODUCT", "OVER_JOIN"],
    "FILTER": ["MISSING_WHERE_CONDITION", "EXTRA_WHERE_CONDITION", "WRONG_WHERE_COLUMN", "WRONG_OPERATOR", "WRONG_LITERAL_VALUE", "WRONG_LOGICAL_CONNECTOR", "WRONG_NEGATION", "WRONG_NULL_HANDLING", "WRONG_BETWEEN_OR_RANGE", "WRONG_LIKE_PATTERN", "DEMO_LITERAL_COPY"],
    "GROUPING": ["MISSING_GROUP_BY", "EXTRA_GROUP_BY", "WRONG_GROUP_BY", "MISSING_HAVING", "EXTRA_HAVING", "WRONG_HAVING"],
    "ORDER_LIMIT": ["MISSING_ORDER_BY", "EXTRA_ORDER_BY", "WRONG_ORDER_COLUMN", "WRONG_SORT_DIRECTION", "MISSING_LIMIT", "EXTRA_LIMIT", "WRONG_LIMIT_VALUE", "WRONG_ORDER_BY"],
    "SUBQUERY_SET": ["MISSING_SUBQUERY", "EXTRA_SUBQUERY", "WRONG_SUBQUERY", "WRONG_NESTING_LEVEL", "WRONG_CORRELATION", "MISSING_SET_OPERATION", "EXTRA_SET_OPERATION", "WRONG_SET_OPERATION", "WRONG_UNION", "WRONG_INTERSECT", "WRONG_EXCEPT", "DEMO_STRUCTURE_COPY"],
    "RESULT_CARDINALITY": ["WRONG_RESULT_CARDINALITY", "DUPLICATE_ROWS", "MISSING_ROWS", "EXTRA_ROWS", "WRONG_SCALAR_VS_LIST", "ORDER_ONLY_MISMATCH"],
    "UNCLASSIFIED_REVIEW": ["COMPLEX_MULTI_COMPONENT_ERROR", "PARSER_UNAVAILABLE", "PARSER_DISAGREEMENT", "HEURISTIC_ONLY", "MANUAL_REVIEW_REQUIRED", "UNCLASSIFIED"],
}
LABEL_FAMILY = {label: family for family, labels in FAMILY_LABELS.items() for label in labels}
LABEL_FAMILY["ALTERNATIVE_VALID_FORMULATION"] = "ALTERNATIVE_VALID"
COARSE = {
    "OUTPUT_CONTROL": "Output/Control",
    "EXECUTION_SYNTAX": "Syntax/Execution",
    "SCHEMA_LINKING": "Schema/Structure",
    "JOIN": "Schema/Structure",
    "PROJECTION": "Semantic Query Logic",
    "AGGREGATION": "Semantic Query Logic",
    "FILTER": "Semantic Query Logic",
    "GROUPING": "Semantic Query Logic",
    "ORDER_LIMIT": "Semantic Query Logic",
    "SUBQUERY_SET": "Semantic Query Logic",
    "RESULT_CARDINALITY": "Semantic Query Logic",
    "UNCLASSIFIED_REVIEW": "Unclassified/Review",
}
PRIMARY_FAMILIES = [x for x in FAMILY_LABELS if x != "UNCLASSIFIED_REVIEW"]

TOKEN_RE = re.compile(
    r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\]|"
    r"<=|>=|<>|!=|==|\|\||[-+*/%<>=]|[A-Za-z_][A-Za-z0-9_$]*|\d+(?:\.\d+)?|[(),.;]",
    re.S,
)
RESERVED = set("select from where join inner left right full outer cross on as group by having order limit offset union intersect except distinct all and or not null is in like between exists case when then else end asc desc with recursive collate cast over partition rows range current row preceding following true false".split())
AGGS = {"count", "sum", "avg", "min", "max", "total", "group_concat"}
SETOPS = {"union", "intersect", "except"}


def sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with (ROOT / path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def norm_sql(sql: str) -> str:
    tokens = tokenize(sql)
    return " ".join(t.lower() for t in tokens if t != ";")


def tokenize(sql: str) -> list[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(sql or "") if not m.group(0).startswith(("--", "/*"))]


def split_top(tokens: list[str], delimiter: str = ",") -> list[list[str]]:
    parts: list[list[str]] = [[]]
    depth = 0
    for token in tokens:
        if token == "(": depth += 1
        elif token == ")": depth -= 1
        if token == delimiter and depth == 0:
            parts.append([])
        else:
            parts[-1].append(token)
    return [p for p in parts if p]


def clause(tokens: list[str], start: tuple[str, ...], stops: set[str]) -> list[str]:
    low = [x.lower() for x in tokens]
    depth = 0
    pos = None
    for i, token in enumerate(low):
        if tokens[i] == "(": depth += 1
        elif tokens[i] == ")": depth -= 1
        if depth == 0 and tuple(low[i:i + len(start)]) == start:
            pos = i + len(start)
            break
    if pos is None: return []
    out: list[str] = []
    depth = 0
    for token in tokens[pos:]:
        if token == "(": depth += 1
        elif token == ")": depth -= 1
        if depth == 0 and token.lower() in stops:
            break
        out.append(token)
    return out


def ident(token: str) -> str:
    return token.strip('`"[]').lower()


def sql_features(sql: str) -> dict[str, Any]:
    tokens = tokenize(sql)
    low = [x.lower() for x in tokens]
    balanced = True
    depth = 0
    for token in tokens:
        if token == "(": depth += 1
        elif token == ")": depth -= 1
        if depth < 0: balanced = False
    balanced = balanced and depth == 0
    select_part = clause(tokens, ("select",), {"from"})
    from_part = clause(tokens, ("from",), {"where", "group", "having", "order", "limit", "union", "intersect", "except"})
    where = clause(tokens, ("where",), {"group", "having", "order", "limit", "union", "intersect", "except"})
    group = clause(tokens, ("group", "by"), {"having", "order", "limit", "union", "intersect", "except"})
    having = clause(tokens, ("having",), {"order", "limit", "union", "intersect", "except"})
    order = clause(tokens, ("order", "by"), {"limit", "union", "intersect", "except"})
    limit = clause(tokens, ("limit",), {"offset", "union", "intersect", "except"})
    tables: list[str] = []
    aliases: dict[str, str] = {}
    i = 0
    expect_table = bool(from_part)
    while i < len(from_part):
        t = from_part[i]
        tl = t.lower()
        if tl in {"join", ","}: expect_table = True; i += 1; continue
        if tl == "on":
            expect_table = False; i += 1
            while i < len(from_part) and from_part[i].lower() not in {"join", ","}: i += 1
            continue
        if expect_table and t != "(":
            table = ident(t)
            if table not in RESERVED:
                tables.append(table)
                j = i + 1
                if j < len(from_part) and from_part[j].lower() == "as": j += 1
                if j < len(from_part) and re.match(r"^[A-Za-z_]", from_part[j]) and from_part[j].lower() not in RESERVED:
                    aliases[ident(from_part[j])] = table
                expect_table = False
        i += 1
    columns: list[str] = []
    for i, t in enumerate(tokens):
        tl = ident(t)
        if not re.match(r"^[A-Za-z_]", t) or tl in RESERVED or tl in AGGS or tl in tables or tl in aliases:
            continue
        if i + 1 < len(tokens) and tokens[i + 1] == "(":
            continue
        if i > 0 and tokens[i - 1] == ".":
            prefix = aliases.get(ident(tokens[i - 2]), ident(tokens[i - 2])) if i > 1 else ""
            columns.append(f"{prefix}.{tl}" if prefix else tl)
        elif i + 1 < len(tokens) and tokens[i + 1] == ".":
            continue
        elif tl not in {"sqlite", "date", "year", "month", "day"}:
            columns.append(tl)
    select_exprs = [norm_sql(" ".join(x)) for x in split_top(select_part)]
    group_exprs = [norm_sql(" ".join(x)) for x in split_top(group)]
    order_exprs = [norm_sql(" ".join(x)) for x in split_top(order)]
    aggs = Counter(t.lower() for t in tokens if t.lower() in AGGS)
    operators = [t.lower() for t in where + having if t.lower() in {"=", "==", "!=", "<>", ">", "<", ">=", "<=", "like", "in", "between", "is"}]
    literals = [t.lower() for t in where + having if t.startswith("'") or re.fullmatch(r"\d+(?:\.\d+)?", t)]
    depth = 0
    subqueries = 0
    for t in tokens:
        if t == "(": depth += 1
        elif t == ")": depth -= 1
        elif t.lower() == "select" and depth > 0: subqueries += 1
    return {
        "tokens": tokens, "parse_success": bool(tokens and "select" in low and balanced), "balanced": balanced,
        "tables": set(tables), "columns": set(columns), "select": select_exprs, "select_count": len(select_exprs),
        "aggs": aggs, "distinct": "distinct" in low, "joins": low.count("join"),
        "join_on": norm_sql(" ".join(clause(tokens, ("on",), {"join", "where", "group", "having", "order", "limit"}))),
        "where": norm_sql(" ".join(where)), "operators": operators, "literals": literals,
        "and_count": low.count("and"), "or_count": low.count("or"), "not_count": low.count("not"),
        "null": "null" in low, "between": "between" in low, "like": "like" in low,
        "group": group_exprs, "having": norm_sql(" ".join(having)), "order": order_exprs,
        "order_dir": [x for x in low if x in {"asc", "desc"}], "limit": norm_sql(" ".join(limit)),
        "subqueries": subqueries, "setops": Counter(x for x in low if x in SETOPS),
        "normalized": norm_sql(sql),
    }


def add_label(labels: dict[str, dict[str, str]], label: str, evidence: str, rule: str, text: str, gold: Any = "", pred: Any = "") -> None:
    current = labels.get(label)
    rank = {"E1": 4, "E2": 3, "E3": 2, "E4": 1}
    item = {"evidence_level": evidence, "automatic_rule": rule, "evidence_text": text, "gold_component": str(gold), "pred_component": str(pred)}
    if current is None or rank[evidence] > rank[current["evidence_level"]]: labels[label] = item


def repeated_generation(raw: str) -> bool:
    toks = [x.lower() for x in tokenize(raw)]
    if len(toks) < 30: return False
    for n in (3, 5):
        grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        if grams and (len(grams) - len(set(grams))) / len(grams) >= 0.25: return True
    return toks.count("select") >= 4 or toks.count("with") >= 4


def safe_results(db_path: str, sql: str, cache: dict[tuple[str, str], tuple[bool, list[tuple[Any, ...]] | None, str]]) -> tuple[bool, list[tuple[Any, ...]] | None, str]:
    key = (db_path, sql)
    if key in cache: return cache[key]
    if not sql.strip():
        cache[key] = (False, None, "No SQL extracted"); return cache[key]
    uri = f"file:{quote(str((ROOT / db_path).resolve()))}?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        budget = [0]
        def progress() -> int:
            budget[0] += 1
            return int(budget[0] > 20000)
        conn.set_progress_handler(progress, 1000)
        cur = conn.execute(sql)
        rows = cur.fetchmany(10001)
        if len(rows) > 10000: raise RuntimeError("result exceeds 10000-row diagnostic cap")
        cache[key] = (True, rows, "")
    except Exception as exc:
        cache[key] = (False, None, repr(exc))
    finally:
        if conn is not None: conn.close()
    return cache[key]


def typed(row: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(x).__name__, repr(x)) for x in row)


def result_labels(labels: dict[str, dict[str, str]], gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]]) -> None:
    g = Counter(typed(x) for x in gold_rows); p = Counter(typed(x) for x in pred_rows)
    if len(gold_rows) != len(pred_rows): add_label(labels, "WRONG_RESULT_CARDINALITY", "E1", "result_row_count", f"gold rows={len(gold_rows)}, pred rows={len(pred_rows)}", len(gold_rows), len(pred_rows))
    if len(gold_rows) == 1 and len(pred_rows) != 1 or len(gold_rows) != 1 and len(pred_rows) == 1: add_label(labels, "WRONG_SCALAR_VS_LIST", "E1", "result_shape", "scalar/list shape differs")
    if g - p: add_label(labels, "MISSING_ROWS", "E1", "typed_multiset_difference", f"missing multiplicity={sum((g-p).values())}")
    if p - g: add_label(labels, "EXTRA_ROWS", "E1", "typed_multiset_difference", f"extra multiplicity={sum((p-g).values())}")
    if any(v > 1 for v in p.values()) and sum(p.values()) > len(p): add_label(labels, "DUPLICATE_ROWS", "E1", "prediction_duplicates", "prediction result contains duplicate rows")
    if g == p and [typed(x) for x in gold_rows] != [typed(x) for x in pred_rows]: add_label(labels, "ORDER_ONLY_MISMATCH", "E1", "ordered_result_difference", "multisets match but row order differs")


def classify(row: dict[str, str], gf: dict[str, Any], pf: dict[str, Any], cache: dict) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    if as_bool(row["exec_match"]): return labels
    raw, pred, err = row.get("raw_output", ""), row.get("pred_sql", ""), row.get("pred_error", "")
    if not raw.strip(): add_label(labels, "EMPTY_RAW_OUTPUT", "E1", "empty_raw", "raw output is empty")
    if not pred.strip():
        add_label(labels, "EMPTY_SQL_EXTRACTION", "E1", "empty_pred_sql", "extracted SQL is empty")
        if raw.strip(): add_label(labels, "EXTRACTOR_FAILURE", "E1", "nonempty_raw_empty_sql", "raw output is nonempty but SQL extraction is empty")
    if pred.strip() and not pred.rstrip().endswith(";"): add_label(labels, "NO_TERMINATING_SEMICOLON", "E1", "semicolon_check", "extracted SQL does not end with semicolon")
    if len([x for x in split_top(tokenize(pred), ";") if x]) > 1: add_label(labels, "MULTIPLE_STATEMENTS", "E1", "top_level_semicolon_count", "multiple top-level statements detected")
    lowraw = raw.lower()
    if "```" in raw or re.search(r"\b(here is|explanation:)\b", lowraw): add_label(labels, "MARKDOWN_OR_EXPLANATION", "E1", "raw_marker", "markdown or explanatory text detected")
    if "<think>" in lowraw or "</think>" in lowraw: add_label(labels, "THINK_MARKER", "E1", "think_marker", "think marker detected")
    if "<|im_" in lowraw or "<|start_header_id|>" in lowraw: add_label(labels, "CHAT_TEMPLATE_ARTIFACT", "E1", "chat_special_token", "chat-template token detected in output")
    if repeated_generation(raw): add_label(labels, "REPETITIVE_GENERATION", "E2", "repeated_ngram_rule", "repeated n-gram or repeated query-opening rule triggered")
    if int(float(row.get("completion_tokens") or 0)) == int(float(row.get("run_max_new_tokens") or 256)):
        add_label(labels, "COMPLETION_LIMIT_REACHED", "E1", "completion_equals_limit", f"completion reached {row.get('run_max_new_tokens') or 256} tokens")
        if not pf["balanced"] or not pred.strip(): add_label(labels, "TRUNCATED_SQL", "E2", "limit_and_incomplete_structure", "limit reached with unbalanced or empty SQL")
    if not pf["parse_success"]:
        add_label(labels, "SQL_PARSE_ERROR", "E1", "fallback_parser_failure", "fallback parser did not find balanced SELECT structure")
        add_label(labels, "MANUAL_REVIEW_REQUIRED", "E4", "parser_failure_review", "component comparison is incomplete")
    if err:
        add_label(labels, "SQLITE_EXECUTION_ERROR", "E1", "stored_pred_error", err)
        el = err.lower()
        if "syntax error" in el or "incomplete input" in el or "unrecognized token" in el: add_label(labels, "SQL_SYNTAX_ERROR", "E1", "sqlite_error_class", err)
        elif "no such table" in el: add_label(labels, "UNKNOWN_TABLE", "E1", "sqlite_error_class", err)
        elif "no such column" in el: add_label(labels, "UNKNOWN_COLUMN", "E1", "sqlite_error_class", err)
        elif "ambiguous column" in el: add_label(labels, "AMBIGUOUS_COLUMN", "E1", "sqlite_error_class", err)
        elif "function" in el or "datatype" in el or "type" in el or "misuse of aggregate" in el: add_label(labels, "TYPE_OR_FUNCTION_ERROR", "E1", "sqlite_error_class", err)
        elif "no sql extracted" not in el: add_label(labels, "OTHER_EXECUTION_ERROR", "E1", "sqlite_error_class", err)
    if gf["parse_success"] and pf["parse_success"]:
        gt, pt = gf["tables"], pf["tables"]
        gc, pc = gf["columns"], pf["columns"]
        if gt - pt: add_label(labels, "MISSING_TABLE", "E1", "table_set_difference", ", ".join(sorted(gt-pt)), gt, pt)
        if pt - gt: add_label(labels, "EXTRA_TABLE", "E1", "table_set_difference", ", ".join(sorted(pt-gt)), gt, pt)
        if gt-pt and pt-gt:
            add_label(labels, "WRONG_TABLE", "E1", "table_substitution", "gold-only and prediction-only tables both present", gt, pt)
            add_label(labels, "WRONG_SCHEMA_LINK", "E2", "table_substitution", "table substitution implies schema-link difference")
        if gc - pc: add_label(labels, "MISSING_COLUMN", "E1", "column_set_difference", ", ".join(sorted(gc-pc)), gc, pc)
        if pc - gc: add_label(labels, "EXTRA_COLUMN", "E1", "column_set_difference", ", ".join(sorted(pc-gc)), gc, pc)
        if gc-pc and pc-gc:
            add_label(labels, "WRONG_COLUMN", "E1", "column_substitution", "gold-only and prediction-only columns both present", gc, pc)
            add_label(labels, "WRONG_SCHEMA_LINK", "E2", "column_substitution", "column substitution implies schema-link difference")
        if gf["select_count"] != pf["select_count"]: add_label(labels, "WRONG_SELECT_CARDINALITY", "E1", "select_expression_count", "SELECT expression count differs", gf["select_count"], pf["select_count"])
        if gf["select_count"] > pf["select_count"]: add_label(labels, "MISSING_SELECT_EXPRESSION", "E1", "select_expression_count", "prediction has fewer SELECT expressions")
        if gf["select_count"] < pf["select_count"]: add_label(labels, "EXTRA_SELECT_EXPRESSION", "E1", "select_expression_count", "prediction has more SELECT expressions")
        if set(gf["select"]) != set(pf["select"]): add_label(labels, "WRONG_SELECT_EXPRESSION", "E2", "normalized_select_difference", "normalized SELECT expressions differ", gf["select"], pf["select"])
        if gf["distinct"] != pf["distinct"]:
            add_label(labels, "WRONG_DISTINCT", "E1", "distinct_presence", "DISTINCT presence differs")
            add_label(labels, "MISSING_DISTINCT" if gf["distinct"] else "EXTRA_DISTINCT", "E1", "distinct_presence", "DISTINCT presence differs")
        if gf["aggs"] != pf["aggs"]:
            add_label(labels, "WRONG_AGGREGATION", "E1", "aggregate_multiset", "aggregate functions differ", gf["aggs"], pf["aggs"])
            if sum(gf["aggs"].values()) > sum(pf["aggs"].values()): add_label(labels, "MISSING_AGGREGATION", "E1", "aggregate_count", "prediction has fewer aggregate functions")
            elif sum(gf["aggs"].values()) < sum(pf["aggs"].values()): add_label(labels, "EXTRA_AGGREGATION", "E1", "aggregate_count", "prediction has more aggregate functions")
            if gf["aggs"].get("count") != pf["aggs"].get("count"): add_label(labels, "WRONG_COUNT_TARGET", "E2", "count_difference", "COUNT usage differs")
            if any(gf["aggs"].get(x) != pf["aggs"].get(x) for x in ("min", "max")): add_label(labels, "WRONG_MIN_MAX", "E1", "min_max_difference", "MIN/MAX usage differs")
            if any(gf["aggs"].get(x) != pf["aggs"].get(x) for x in ("sum", "avg")): add_label(labels, "WRONG_SUM_AVG", "E1", "sum_avg_difference", "SUM/AVG usage differs")
        if gf["joins"] != pf["joins"]:
            add_label(labels, "MISSING_JOIN" if gf["joins"] > pf["joins"] else "EXTRA_JOIN", "E1", "join_count", "JOIN count differs", gf["joins"], pf["joins"])
            if pf["joins"] > gf["joins"] and pt-gt:
                add_label(labels, "OVER_JOIN", "E2", "extra_join_and_table", "extra JOIN coincides with prediction-only table")
        if gf["joins"] and pf["joins"] and gf["join_on"] != pf["join_on"]: add_label(labels, "WRONG_JOIN_CONDITION", "E2", "normalized_on_difference", "first normalized ON clause differs")
        if gf["joins"] and pf["joins"] and gt != pt: add_label(labels, "WRONG_JOIN_PATH", "E2", "joined_table_set_difference", "joined table sets differ")
        if bool(gf["where"]) != bool(pf["where"]): add_label(labels, "MISSING_WHERE_CONDITION" if gf["where"] else "EXTRA_WHERE_CONDITION", "E1", "where_presence", "WHERE presence differs")
        elif gf["where"] != pf["where"]:
            if gf["operators"] != pf["operators"]: add_label(labels, "WRONG_OPERATOR", "E1", "operator_sequence", "filter operators differ", gf["operators"], pf["operators"])
            if Counter(gf["literals"]) != Counter(pf["literals"]): add_label(labels, "WRONG_LITERAL_VALUE", "E1", "literal_multiset", "filter literals differ", gf["literals"], pf["literals"])
            if (gf["and_count"], gf["or_count"]) != (pf["and_count"], pf["or_count"]): add_label(labels, "WRONG_LOGICAL_CONNECTOR", "E1", "boolean_connector_count", "AND/OR counts differ")
            if gf["not_count"] != pf["not_count"]: add_label(labels, "WRONG_NEGATION", "E1", "negation_count", "NOT count differs")
            if gf["null"] != pf["null"]: add_label(labels, "WRONG_NULL_HANDLING", "E1", "null_presence", "NULL handling differs")
            if gf["between"] != pf["between"]: add_label(labels, "WRONG_BETWEEN_OR_RANGE", "E1", "between_presence", "BETWEEN usage differs")
            if gf["like"] != pf["like"]: add_label(labels, "WRONG_LIKE_PATTERN", "E1", "like_presence", "LIKE usage differs")
            if gc != pc: add_label(labels, "WRONG_WHERE_COLUMN", "E2", "where_and_column_difference", "WHERE and column sets differ")
        for name, key, missing, extra, wrong in [
            ("GROUP BY", "group", "MISSING_GROUP_BY", "EXTRA_GROUP_BY", "WRONG_GROUP_BY"),
            ("HAVING", "having", "MISSING_HAVING", "EXTRA_HAVING", "WRONG_HAVING"),
            ("ORDER BY", "order", "MISSING_ORDER_BY", "EXTRA_ORDER_BY", "WRONG_ORDER_COLUMN"),
        ]:
            gv, pv = gf[key], pf[key]
            if bool(gv) != bool(pv): add_label(labels, missing if gv else extra, "E1", f"{key}_presence", f"{name} presence differs")
            elif gv != pv: add_label(labels, wrong, "E1" if key != "having" else "E2", f"{key}_difference", f"normalized {name} differs", gv, pv)
        if gf["order_dir"] != pf["order_dir"]: add_label(labels, "WRONG_SORT_DIRECTION", "E1", "sort_direction", "ASC/DESC sequence differs")
        if bool(gf["limit"]) != bool(pf["limit"]): add_label(labels, "MISSING_LIMIT" if gf["limit"] else "EXTRA_LIMIT", "E1", "limit_presence", "LIMIT presence differs")
        elif gf["limit"] != pf["limit"]: add_label(labels, "WRONG_LIMIT_VALUE", "E1", "limit_value", "LIMIT value differs", gf["limit"], pf["limit"])
        if gf["subqueries"] != pf["subqueries"]:
            add_label(labels, "MISSING_SUBQUERY" if gf["subqueries"] > pf["subqueries"] else "EXTRA_SUBQUERY", "E1", "subquery_count", "subquery count differs")
            add_label(labels, "WRONG_NESTING_LEVEL", "E1", "subquery_count", "nesting level differs")
        if gf["setops"] != pf["setops"]:
            add_label(labels, "WRONG_SET_OPERATION", "E1", "setop_multiset", "set operations differ", gf["setops"], pf["setops"])
            if sum(gf["setops"].values()) > sum(pf["setops"].values()): add_label(labels, "MISSING_SET_OPERATION", "E1", "setop_count", "prediction misses set operation")
            elif sum(gf["setops"].values()) < sum(pf["setops"].values()): add_label(labels, "EXTRA_SET_OPERATION", "E1", "setop_count", "prediction adds set operation")
            for op in SETOPS:
                if gf["setops"].get(op) != pf["setops"].get(op): add_label(labels, f"WRONG_{op.upper()}", "E1", "setop_type", f"{op.upper()} usage differs")
    gok, grows, _ = safe_results(row["db_path"], row["gold_sql"], cache)
    pok, prows, _ = safe_results(row["db_path"], pred, cache)
    if gok and pok and grows is not None and prows is not None: result_labels(labels, grows, prows)
    if not labels:
        add_label(labels, "UNCLASSIFIED", "E4", "no_reliable_rule", "no deterministic component label was available")
        add_label(labels, "MANUAL_REVIEW_REQUIRED", "E4", "unclassified_review", "manual review required")
    elif sum(1 for x in labels if LABEL_FAMILY.get(x) not in {"OUTPUT_CONTROL", "EXECUTION_SYNTAX", "RESULT_CARDINALITY", "UNCLASSIFIED_REVIEW"}) >= 4:
        add_label(labels, "COMPLEX_MULTI_COMPONENT_ERROR", "E3", "four_or_more_component_families", "multiple SQL components differ")
        add_label(labels, "HEURISTIC_ONLY", "E3", "complex_case_flag", "causal primary error cannot be isolated automatically")
    return labels


def exact_mcnemar(n01: int, n10: int) -> float:
    n = n01 + n10
    if not n: return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    vals = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        size = min(250, BOOTSTRAP_RESAMPLES - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        vals[start:start+size] = diff[idx].mean(axis=1)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda x: x[1]["mcnemar_p"])
    running = 0.0
    for rank, (idx, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(rows)-rank)*row["mcnemar_p"]))
        rows[idx]["holm_adjusted_p"] = running
        rows[idx]["significant_holm_0_05"] = running < .05


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists(): raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    if path.exists(): raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_plot(base: Path) -> None:
    for ext in ("png", "pdf"):
        p = base.with_suffix(f".{ext}")
        if p.exists(): raise RuntimeError(f"Refusing to overwrite {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()


def transition(base: dict[str, str], lora: dict[str, str]) -> tuple[str, str]:
    b, l = as_bool(base["exec_match"]), as_bool(lora["exec_match"])
    bok, lok = as_bool(base["pred_ok"]), as_bool(lora["pred_ok"])
    if not b and l: return "T1", "T1a" if not bok else "T1b"
    if b and not l: return "T2", "T2a" if not lok else "T2b"
    if not b and not l: return "T3", "T3a" if not bok and not lok else "T3b"
    return "T4", "T4"


def codebook_text() -> str:
    lines = [f"# SQL Error Analysis Codebook ({TAXONOMY_VERSION})", "", "## Scope and authority", "", "Execution Match is the only correctness decision. Labels diagnose observed outputs on the frozen Spider Dev database and never override EMA. Labels are multi-label. Structurally different execution-matching SQL is not an error and is recorded separately as `ALTERNATIVE_VALID_FORMULATION`.", "", "## Parser", "", f"`{PARSER_NAME}` version `{PARSER_VERSION}` is a deterministic SQLite-aware lexer/clause fallback. No local Spider AST parser, sqlglot, sqlparse, or sql_metadata installation was available. It extracts top-level clauses, table/column references, aggregation, joins, predicates, grouping, ordering, limits, subqueries, and set operations; complex or failed parses are flagged for review.", "", "## Evidence", "", "| Level | Meaning |", "|---|---|", "| E1 | Deterministic artifact, SQLite error/result, or direct component difference |", "| E2 | High-confidence rule requiring combined observations |", "| E3 | Heuristic association; not a secured causal fact |", "| E4 | Manual review required |", "", "## Label rules", ""]
    for family, labels in FAMILY_LABELS.items():
        lines += [f"### {family}", "", "| Label | Rule | Exclusion / caveat |", "|---|---|---|"]
        for label in labels:
            if label == "TRUNCATED_SQL": rule = "Token limit plus empty or structurally unbalanced extracted SQL"
            elif label == "OVER_JOIN": rule = "Prediction adds both JOIN(s) and table(s) relative to gold"
            elif label in {"MANUAL_REVIEW_REQUIRED", "UNCLASSIFIED"}: rule = "No sufficiently reliable automatic component diagnosis"
            elif label.startswith("WRONG_RESULT") or label in {"DUPLICATE_ROWS", "MISSING_ROWS", "EXTRA_ROWS", "WRONG_SCALAR_VS_LIST", "ORDER_ONLY_MISMATCH"}: rule = "Read-only typed result comparison"
            elif label in {"REPETITIVE_GENERATION", "LONGER_OR_UNSTABLE_OUTPUT"}: rule = "Repeated n-gram/query-opening or few-shot length-instability rule"
            else: rule = "Direct token/clause/component presence, count, or normalized-set comparison"
            caveat = "Not assigned to execution-matching predictions" if family != "UNCLASSIFIED_REVIEW" else "Never imputed into a more specific category"
            lines.append(f"| `{label}` | {rule} | {caveat} |")
        lines.append("")
    lines += ["## Few-shot-only heuristics", "", "`DEMO_STRUCTURE_COPY` (E3) requires a prediction to adopt at least one demo structure feature absent from gold and to be structurally closer to the demo. `DEMO_LITERAL_COPY` (E3) requires a demo literal in the prediction that is absent from gold. Neither establishes causation.", "", "## Result safety", "", "Databases are opened with URI `mode=ro`, `PRAGMA query_only=ON`, a progress handler, and a 10,000-row diagnostic cap. Result labels are omitted when reliable comparison is unavailable.", ""]
    return "\n".join(lines)


def main() -> None:
    for p in list(OUT.values()) + [x.with_suffix(ext) for x in PLOTS.values() for ext in (".png", ".pdf")]:
        if p.exists(): raise RuntimeError(f"Target already exists: {p}")
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected: raise RuntimeError(f"Source hash mismatch: {path}: {actual} != {expected}")
    manifest = json.loads((ROOT / CROSS_MANIFEST).read_text(encoding="utf-8"))
    for source in manifest["authoritative_sources"].values():
        if sha256(source["path"]) != source["sha256"]: raise RuntimeError(f"Authoritative source changed: {source['path']}")
    run_records = [r for r in manifest["runs"] if r["condition"] == "zero_shot" and r["model"] in MODEL_ORDER and r["role"] in {"base", "lora_v2"}]
    if len(run_records) != 6: raise RuntimeError(f"Expected 6 zero-shot runs, got {len(run_records)}")
    by_run: dict[tuple[str, str], dict[str, Any]] = {}
    case_hash = None
    for rec in run_records:
        for key in ("csv", "metadata", "config"):
            if sha256(rec[f"{key}_path"]) != rec[f"{key}_sha256"]: raise RuntimeError(f"Run source changed: {rec['run_id']} {key}")
        rows = read_csv(rec["csv_path"])
        ids = [r["id"] for r in rows]
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if len(rows) != N or len(set(ids)) != N or digest != rec["case_ids_sha256"]: raise RuntimeError(f"Case integrity failed: {rec['run_id']}")
        case_hash = case_hash or digest
        if digest != case_hash: raise RuntimeError("Case order differs between runs")
        rec = dict(rec); rec["rows"] = rows; by_run[(rec["model"], rec["role"])] = rec
    requested_testset = manifest["testset"]["path"]
    resolved_testset = requested_testset
    testset_path_warning = False
    if sha256(requested_testset) != manifest["testset"]["sha256"]:
        candidate = "data/testcases_spider_dev_full.jsonl"
        if not (ROOT / candidate).exists() or sha256(candidate) != manifest["testset"]["sha256"]:
            raise RuntimeError("Testset integrity failed and no byte-identical frozen copy exists")
        resolved_testset = candidate
        testset_path_warning = True
    tests = read_jsonl(resolved_testset)
    if len(tests) != N: raise RuntimeError("Resolved testset row count failed")
    test_by_id = {x["id"]: x for x in tests}
    cache: dict = {}
    feature_cache: dict[str, dict[str, Any]] = {}
    def feats(sql: str) -> dict[str, Any]:
        if sql not in feature_cache: feature_cache[sql] = sql_features(sql)
        return feature_cache[sql]
    cases: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    alternative: list[dict[str, Any]] = []
    label_sets: dict[tuple[str, str, str], set[str]] = {}
    label_details: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    transitions: list[dict[str, Any]] = []
    parse_counts = defaultdict(Counter)
    for model in MODEL_ORDER:
        base_rows = by_run[(model, "base")]["rows"]
        lora_rows = by_run[(model, "lora_v2")]["rows"]
        tcount = Counter(); subcount = Counter()
        for b, l in zip(base_rows, lora_rows):
            if b["id"] != l["id"]: raise RuntimeError("Paired case order mismatch")
            tg, sg = transition(b, l); tcount[tg] += 1; subcount[sg] += 1
            gf = feats(b["gold_sql"])
            per_role = {}
            for role, row in (("base", b), ("lora_v2", l)):
                pf = feats(row["pred_sql"])
                parse_counts[(model, role)]["gold_success"] += int(gf["parse_success"])
                parse_counts[(model, role)]["pred_success"] += int(pf["parse_success"])
                labels = classify(row, gf, pf, cache)
                label_sets[(model, role, row["id"])] = set(labels)
                label_details[(model, role, row["id"])] = labels
                per_role[role] = labels
                for label, detail in labels.items():
                    long_rows.append({
                        "model_line": MODEL_LABELS[model], "model_key": model, "role": role, "case_id": row["id"], "db_id": row["db_id"],
                        "question": row["question"], "gold_sql": row["gold_sql"], "pred_sql": row["pred_sql"], "raw_output": row["raw_output"],
                        "execution_success": as_bool(row["pred_ok"]), "execution_match": as_bool(row["exec_match"]), "transition_group": tg,
                        "error_label": label, "error_family": LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW"), **detail,
                        "review_required": detail["evidence_level"] == "E4" or label == "MANUAL_REVIEW_REQUIRED",
                    })
            bs, ls = set(per_role["base"]), set(per_role["lora_v2"])
            cases.append({
                "model_line": MODEL_LABELS[model], "model_key": model, "case_id": b["id"], "db_id": b["db_id"], "question": b["question"], "gold_sql": b["gold_sql"],
                "starting_pred_sql": b["pred_sql"], "lora_pred_sql": l["pred_sql"], "starting_raw_output": b["raw_output"], "lora_raw_output": l["raw_output"],
                "starting_execution_success": as_bool(b["pred_ok"]), "starting_execution_match": as_bool(b["exec_match"]),
                "lora_execution_success": as_bool(l["pred_ok"]), "lora_execution_match": as_bool(l["exec_match"]),
                "transition_group": tg, "transition_subgroup": sg, "starting_error_labels": ";".join(sorted(bs)), "lora_error_labels": ";".join(sorted(ls)),
                "repaired_labels": ";".join(sorted(bs-ls)), "introduced_labels": ";".join(sorted(ls-bs)), "persistent_labels": ";".join(sorted(bs&ls)),
            })
            if tg == "T4":
                bf, lf = feats(b["pred_sql"]), feats(l["pred_sql"])
                if b["pred_sql"] == l["pred_sql"]: kind = "exact_same"
                elif bf["normalized"] == lf["normalized"]: kind = "normalized_same"
                else: kind = "structurally_different_execution_equivalent"
                alternative.append({"model_line": MODEL_LABELS[model], "model_key": model, "case_id": b["id"], "db_id": b["db_id"], "question": b["question"], "gold_sql": b["gold_sql"], "starting_pred_sql": b["pred_sql"], "lora_pred_sql": l["pred_sql"], "formulation_class": kind, "label": "ALTERNATIVE_VALID_FORMULATION" if kind.startswith("structurally") else "", "execution_match_both": True})
        transitions.append({"model_line": MODEL_LABELS[model], "model_key": model, "T1_repairs": tcount["T1"], "T1a_nonexec_to_correct": subcount["T1a"], "T1b_execwrong_to_correct": subcount["T1b"], "T2_regressions": tcount["T2"], "T2a_correct_to_nonexec": subcount["T2a"], "T2b_correct_to_execwrong": subcount["T2b"], "T3_persistent_errors": tcount["T3"], "T3a_both_nonexec": subcount["T3a"], "T3b_at_least_one_execwrong": subcount["T3b"], "T4_stable_correct": tcount["T4"], "total": sum(tcount.values())})
    category_rows: list[dict[str, Any]] = []
    all_labels = sorted(set(x["error_label"] for x in long_rows))
    for model in MODEL_ORDER:
        incorrect = {role: sum(not as_bool(r["exec_match"]) for r in by_run[(model, role)]["rows"]) for role in ("base", "lora_v2")}
        for label in all_labels:
            bset = {cid for (m, role, cid), labs in label_sets.items() if m == model and role == "base" and label in labs}
            lset = {cid for (m, role, cid), labs in label_sets.items() if m == model and role == "lora_v2" and label in labs}
            evidence = Counter(x["evidence_level"] for x in long_rows if x["model_key"] == model and x["error_label"] == label)
            category_rows.append({"model_line": MODEL_LABELS[model], "model_key": model, "error_label": label, "error_family": LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW"), "starting_count": len(bset), "lora_count": len(lset), "repaired": len(bset-lset), "introduced": len(lset-bset), "persistent": len(bset&lset), "net_change_lora_minus_starting": len(lset)-len(bset), "starting_rate_per_1032": len(bset)/N, "lora_rate_per_1032": len(lset)/N, "starting_share_of_incorrect": len(bset)/incorrect["base"], "lora_share_of_incorrect": len(lset)/incorrect["lora_v2"], "E1": evidence["E1"], "E2": evidence["E2"], "E3": evidence["E3"], "E4": evidence["E4"], "manual_review_case_count": sum(label in label_sets[(model, role, cid)] and "MANUAL_REVIEW_REQUIRED" in label_sets[(model, role, cid)] for role in ("base", "lora_v2") for cid in [r["id"] for r in by_run[(model, role)]["rows"]])})
    stats: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for model in MODEL_ORDER:
        family_stats = []
        ids = [r["id"] for r in by_run[(model, "base")]["rows"]]
        for family in PRIMARY_FAMILIES:
            labels = set(FAMILY_LABELS[family])
            a = np.array([bool(label_sets[(model, "base", cid)] & labels) for cid in ids], dtype=np.int8)
            b = np.array([bool(label_sets[(model, "lora_v2", cid)] & labels) for cid in ids], dtype=np.int8)
            n01 = int(np.sum((a == 0) & (b == 1))); n10 = int(np.sum((a == 1) & (b == 0)))
            lo, hi = bootstrap_ci(b.astype(float)-a.astype(float), rng)
            family_stats.append({"model_line": MODEL_LABELS[model], "model_key": model, "error_family": family, "starting_count": int(a.sum()), "lora_count": int(b.sum()), "delta_count": int(b.sum()-a.sum()), "delta_rate_per_1032": float(b.mean()-a.mean()), "n01_introduced": n01, "n10_repaired": n10, "mcnemar_p": exact_mcnemar(n01, n10), "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi, "inference_class": "EXPLORATIVE ERROR-CATEGORY INFERENCE"})
        holm(family_stats); stats.extend(family_stats)

    # Qwen-2B termination evidence is kept as an external eight-condition diagnostic.
    capped = read_csv("audits/derived/qwen35_2b_base_maxnew256_vs_512_capped_case_analysis_20260716.csv")
    repetition = read_csv("audits/derived/qwen35_2b_base_maxnew256_vs_512_repetition_analysis_20260716.csv")
    cap_summary = {
        "observations": len(capped), "reached_512_again": sum(as_bool(x["capped_again_512"]) for x in capped),
        "terminated_before_512": sum(as_bool(x["terminated_before_512"]) for x in capped), "new_matches": sum(as_bool(x["newly_correct"]) for x in capped),
        "repetition_rule": sum(as_bool(x["repetition_rule_triggered"]) for x in repetition),
        "sql_extracted_at_256": sum(as_bool(x["old_sql_extracted"]) for x in capped),
        "semantically_correct_first_sql": sum(as_bool(x["old_exec_match"]) for x in capped),
        "executable_but_wrong_first_sql": sum(as_bool(x["old_pred_ok"]) and not as_bool(x["old_exec_match"]) for x in capped),
        "not_executable_first_sql": sum(not as_bool(x["old_pred_ok"]) for x in capped),
        "zero_shot_observations": sum(x["condition"] == "zero_shot" for x in capped),
    }

    # Secondary few-shot harm analysis.
    demo_pool = {x["id"]: x for x in read_jsonl("data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl")}
    static_demo = read_jsonl("data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl")[0]
    demo_pool[static_demo["id"]] = static_demo
    fewshot_rows: list[dict[str, Any]] = []
    selected_fs = {"qwen2b": ["top1", "static_seed42", "structure"], "llama3b": ["top1", "structure"]}
    run_lookup = {(x["model"], x["role"], x["condition"]): x for x in manifest["runs"]}
    for model, conditions in selected_fs.items():
        zero = {r["id"]: r for r in by_run[(model, "lora_v2")]["rows"]}
        for condition in conditions:
            rec = run_lookup[(model, "lora_v2", condition)]
            if sha256(rec["csv_path"]) != rec["csv_sha256"] or sha256(rec["trace_path"]) != rec["trace_sha256"]: raise RuntimeError("Few-shot source hash mismatch")
            fs = {r["id"]: r for r in read_csv(rec["csv_path"])}
            trace = {r["id"]: r for r in read_jsonl(rec["trace_path"])}
            for cid, zr in zero.items():
                fr = fs[cid]
                if not as_bool(zr["exec_match"]) or as_bool(fr["exec_match"]): continue
                tr = trace[cid]; did = tr.get("retrieved_ids", [""])[0] if tr.get("retrieved_ids") else ""; demo = demo_pool.get(did, {})
                gf, ff, df = feats(fr["gold_sql"]), feats(fr["pred_sql"]), feats(str(demo.get("gold_sql", "")))
                labels = classify(fr, gf, ff, cache)
                # Explicit few-shot-harm vocabulary.
                if ff["order"] != gf["order"]: add_label(labels, "WRONG_ORDER_BY", "E2", "fewshot_order_difference", "few-shot ORDER BY differs from gold")
                if int(fr.get("completion_tokens") or 0) > int(zr.get("completion_tokens") or 0) * 2 and int(fr.get("completion_tokens") or 0) >= 80: add_label(labels, "LONGER_OR_UNSTABLE_OUTPUT", "E2", "fewshot_length_ratio", "few-shot completion is over twice zero-shot length and at least 80 tokens")
                demo_literals = set(df["literals"]); copied = demo_literals & set(ff["literals"]) - set(gf["literals"])
                if copied: add_label(labels, "DEMO_LITERAL_COPY", "E3", "demo_literal_overlap", f"prediction-only literal(s) also occur in demo: {sorted(copied)}")
                structural = lambda f: np.array([len(f["tables"]), f["joins"], sum(f["aggs"].values()), int(f["distinct"]), f["subqueries"], len(f["group"]), len(f["order"]), sum(f["setops"].values())], dtype=float)
                pv, gv, dv = structural(ff), structural(gf), structural(df)
                adopted = np.any((pv == dv) & (gv != dv))
                if adopted and np.abs(pv-dv).sum() < np.abs(gv-dv).sum(): add_label(labels, "DEMO_STRUCTURE_COPY", "E3", "demo_structure_distance", "prediction structure is closer to demo and adopts a demo-only feature")
                for label, detail in labels.items():
                    if label not in {"EXTRA_TABLE", "OVER_JOIN", "WRONG_JOIN_PATH", "WRONG_AGGREGATION", "EXTRA_DISTINCT", "EXTRA_SUBQUERY", "WRONG_GROUP_BY", "WRONG_ORDER_BY", "DEMO_STRUCTURE_COPY", "DEMO_LITERAL_COPY", "LONGER_OR_UNSTABLE_OUTPUT"}: continue
                    fewshot_rows.append({"model_line": MODEL_LABELS[model], "model_key": model, "condition": condition, "case_id": cid, "db_id": fr["db_id"], "zero_shot_correct": True, "fewshot_correct": False, "demo_id": did, "demo_sql": demo.get("gold_sql", ""), "gold_sql": fr["gold_sql"], "zero_pred_sql": zr["pred_sql"], "fewshot_pred_sql": fr["pred_sql"], "error_label": label, "error_family": LABEL_FAMILY.get(label, "UNCLASSIFIED_REVIEW"), **detail, "analysis_class": "EXPLORATIVE FEW-SHOT ERROR ANALYSIS"})

    # Stratified, deterministic review sample.
    rng_py = random.Random(SEED)
    review_rows: list[dict[str, Any]] = []
    desired = {"T1": 20, "T2": 15, "T3": 20, "T4": 5}
    for model in MODEL_ORDER:
        pool = [x for x in cases if x["model_key"] == model]
        chosen: list[dict[str, Any]] = []
        for tg, count in desired.items():
            candidates = [x for x in pool if x["transition_group"] == tg and (tg != "T4" or any(a["model_key"] == model and a["case_id"] == x["case_id"] and a["formulation_class"].startswith("structurally") for a in alternative))]
            candidates.sort(key=lambda x: (x["db_id"], x["case_id"])); rng_py.shuffle(candidates); chosen.extend(candidates[:count])
        if len(chosen) < 60:
            remaining = [x for x in pool if x not in chosen]; rng_py.shuffle(remaining); chosen.extend(remaining[:60-len(chosen)])
        for i, x in enumerate(chosen[:60], 1):
            tc = test_by_id[x["case_id"]]
            evidence = []
            for role in ("base", "lora_v2"):
                for label, detail in label_details[(model, role, x["case_id"])].items(): evidence.append(f"{role}:{label}:{detail['evidence_level']}:{detail['evidence_text']}")
            review_rows.append({"review_id": f"{model.upper()}-{i:03d}", "model_line": MODEL_LABELS[model], "transition_group": x["transition_group"], "case_id": x["case_id"], "db_id": x["db_id"], "question": x["question"], "relevant_schema_excerpt": tc["schema_prompt"][:1800], "gold_sql": x["gold_sql"], "starting_raw_output": x["starting_raw_output"], "starting_pred_sql": x["starting_pred_sql"], "lora_raw_output": x["lora_raw_output"], "lora_pred_sql": x["lora_pred_sql"], "starting_execution_status": f"success={x['starting_execution_success']};match={x['starting_execution_match']}", "lora_execution_status": f"success={x['lora_execution_success']};match={x['lora_execution_match']}", "automatic_starting_labels": x["starting_error_labels"], "automatic_lora_labels": x["lora_error_labels"], "automatic_evidence": " | ".join(evidence), "human_labels_starting": "", "human_labels_lora": "", "human_primary_error": "", "human_repair_assessment": "", "human_regression_assessment": "", "human_notes": "", "reviewer": "", "review_date": "", "review_status": "UNREVIEWED"})
    blind: list[dict[str, Any]] = []; blind_key: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        candidates = [x for x in review_rows if x["model_line"] == MODEL_LABELS[model]]; rng_py.shuffle(candidates)
        for item in candidates[:10]:
            swap = int(hashlib.sha256(f"{SEED}:{item['case_id']}".encode()).hexdigest(), 16) % 2 == 1
            a = item["lora_pred_sql"] if swap else item["starting_pred_sql"]; b = item["starting_pred_sql"] if swap else item["lora_pred_sql"]
            sa = item["lora_execution_status"] if swap else item["starting_execution_status"]; sb = item["starting_execution_status"] if swap else item["lora_execution_status"]
            blind.append({"review_id": item["review_id"], "model_line": item["model_line"], "case_id": item["case_id"], "db_id": item["db_id"], "question": item["question"], "schema_excerpt": item["relevant_schema_excerpt"], "gold_sql": item["gold_sql"], "prediction_a": a, "prediction_b": b, "execution_status_a": sa, "execution_status_b": sb, "reviewer_labels": "", "reviewer_notes": ""})
            blind_key.append({"review_id": item["review_id"], "model_line": item["model_line"], "case_id": item["case_id"], "prediction_a_role": "LoRA v2" if swap else "Starting model", "prediction_b_role": "Starting model" if swap else "LoRA v2"})

    # Plots.
    colors = ["#2b8cbe", "#d95f0e", "#756bb1", "#4d4d4d"]
    x = np.arange(3); bottom = np.zeros(3)
    plt.figure(figsize=(9, 5.5))
    for key, label, color in zip(["T1_repairs", "T2_regressions", "T3_persistent_errors", "T4_stable_correct"], ["Repairs", "Regressions", "Persistent errors", "Stable correct"], colors):
        vals = np.array([r[key] for r in transitions]); plt.bar(x, vals, bottom=bottom, label=label, color=color); bottom += vals
    plt.xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER]); plt.ylabel("Cases (n=1,032 per line)"); plt.legend(ncol=2); plt.title("Paired zero-shot outcome transitions"); save_plot(PLOTS["transitions"])
    fam = [x for x in PRIMARY_FAMILIES]
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        rows = {r["error_family"]: r for r in stats if r["model_key"] == model}; xx=np.arange(len(fam)); ax.bar(xx-.2,[rows[f]["starting_count"] for f in fam],.4,label="Starting",color="#4c78a8"); ax.bar(xx+.2,[rows[f]["lora_count"] for f in fam],.4,label="LoRA v2",color="#f58518"); ax.set_title(MODEL_LABELS[model]); ax.set_ylabel("Cases"); ax.legend()
    axes[-1].set_xticks(np.arange(len(fam)),fam,rotation=45,ha="right"); fig.suptitle("Automatic error-family presence (multi-label)"); save_plot(PLOTS["families"])
    top_labels = sorted(all_labels, key=lambda label: sum(r["repaired"]+r["introduced"] for r in category_rows if r["error_label"]==label), reverse=True)[:12]
    plt.figure(figsize=(11,6)); xx=np.arange(len(top_labels)); repaired=[sum(r["repaired"] for r in category_rows if r["error_label"]==l) for l in top_labels]; introduced=[sum(r["introduced"] for r in category_rows if r["error_label"]==l) for l in top_labels]; plt.bar(xx-.2,repaired,.4,label="Repaired",color="#2ca25f"); plt.bar(xx+.2,introduced,.4,label="Introduced",color="#de2d26"); plt.xticks(xx,top_labels,rotation=50,ha="right"); plt.ylabel("Label transitions across three lines"); plt.title("Most frequent repaired and introduced automatic labels"); plt.legend(); save_plot(PLOTS["changes"])
    coarse_names = ["Output/Control","Syntax/Execution","Schema/Structure","Semantic Query Logic","Unclassified/Review"]
    fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
    for ax,model in zip(axes,MODEL_ORDER):
        vals=[]
        for role in ("base","lora_v2"):
            rolevals=[]
            ids=[r["id"] for r in by_run[(model,role)]["rows"]]
            for cn in coarse_names:
                rolevals.append(sum(any(COARSE.get(LABEL_FAMILY.get(l,"UNCLASSIFIED_REVIEW"),"Unclassified/Review")==cn for l in label_sets[(model,role,cid)]) for cid in ids))
            vals.append(rolevals)
        xx=np.arange(len(coarse_names)); ax.bar(xx-.2,vals[0],.4,label="Starting"); ax.bar(xx+.2,vals[1],.4,label="LoRA v2"); ax.set_title(MODEL_LABELS[model]); ax.set_xticks(xx,coarse_names,rotation=55,ha="right"); ax.legend()
    fig.suptitle("Technical and semantic diagnostic groups (overlapping case counts)"); save_plot(PLOTS["coarse"])
    fs_counter=Counter((r["model_key"],r["error_label"]) for r in fewshot_rows); fs_labels=[x for x,_ in Counter(r["error_label"] for r in fewshot_rows).most_common(10)]; plt.figure(figsize=(10,5.5)); xx=np.arange(len(fs_labels)); plt.bar(xx-.2,[fs_counter[("qwen2b",l)] for l in fs_labels],.4,label="Qwen 2B LoRA"); plt.bar(xx+.2,[fs_counter[("llama3b",l)] for l in fs_labels],.4,label="Llama 3B LoRA"); plt.xticks(xx,fs_labels,rotation=50,ha="right"); plt.ylabel("Label occurrences in zero-correct / few-shot-wrong cases"); plt.title("Explorative few-shot harm diagnostics"); plt.legend(); save_plot(PLOTS["fewshot"])

    write_csv_new(OUT["cases"], cases); write_csv_new(OUT["labels"], long_rows); write_csv_new(OUT["transitions"], transitions); write_csv_new(OUT["categories"], category_rows); write_csv_new(OUT["statistics"], stats); write_csv_new(OUT["alternative"], alternative); write_csv_new(OUT["review"], review_rows); write_csv_new(OUT["blind"], blind); write_csv_new(OUT["key"], blind_key); write_csv_new(OUT["fewshot"], fewshot_rows); write_text_new(OUT["codebook"], codebook_text())

    def top(model: str, key: str, n: int = 5) -> list[dict[str, Any]]:
        return sorted([r for r in category_rows if r["model_key"] == model], key=lambda r: (-r[key], r["error_label"]))[:n]
    thesis = ["# Thesis-ready Cross-model Zero-shot Error Analysis Tables", "", "All labels are automatic, multi-label diagnostics; Execution Match remains authoritative.", "", "## Table A: Outcome transitions", "", "| Model line | Repairs | Regressions | Persistent errors | Stable correct |", "|---|---:|---:|---:|---:|"]
    for r in transitions: thesis.append(f"| {r['model_line']} | {r['T1_repairs']} | {r['T2_regressions']} | {r['T3_persistent_errors']} | {r['T4_stable_correct']} |")
    thesis += ["", "## Tables B-E: Error labels", ""]
    for model in MODEL_ORDER:
        thesis += [f"### {MODEL_LABELS[model]}", "", "| Rank | Most repaired | n | Most introduced | n | Most persistent | n |", "|---:|---|---:|---|---:|---|---:|"]
        a,b,c=top(model,"repaired"),top(model,"introduced"),top(model,"persistent")
        for i in range(5): thesis.append(f"| {i+1} | {a[i]['error_label']} | {a[i]['repaired']} | {b[i]['error_label']} | {b[i]['introduced']} | {c[i]['error_label']} | {c[i]['persistent']} |")
        thesis.append("")
    thesis += ["## Table F: Technical versus semantic groups", "", "Counts overlap because a case can receive multiple labels.", "", "| Model | Role | Output/control | Syntax/execution | Schema/structure | Semantic logic | Review |", "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        for role in ("base","lora_v2"):
            ids=[r["id"] for r in by_run[(model,role)]["rows"]]; vals=[]
            for cn in coarse_names: vals.append(sum(any(COARSE.get(LABEL_FAMILY.get(l,"UNCLASSIFIED_REVIEW"),"Unclassified/Review")==cn for l in label_sets[(model,role,cid)]) for cid in ids))
            thesis.append(f"| {MODEL_LABELS[model]} | {role} | " + " | ".join(map(str,vals)) + " |")
    thesis += ["", "## Table G: Manual review sample", "", "| Model | T1 | T2 | T3 | T4 | Total |", "|---|---:|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        c=Counter(x["transition_group"] for x in review_rows if x["model_line"]==MODEL_LABELS[model]); thesis.append(f"| {MODEL_LABELS[model]} | {c['T1']} | {c['T2']} | {c['T3']} | {c['T4']} | {sum(c.values())} |")
    write_text_new(OUT["thesis"], "\n".join(thesis)+"\n")

    parse_summary = {f"{m}:{r}": dict(c) for (m,r),c in parse_counts.items()}
    alt_struct = sum(x["formulation_class"].startswith("structurally") for x in alternative)
    fs_summary = Counter((x["model_key"],x["condition"]) for x in fewshot_rows)
    lines = ["# Cross-model Zero-shot Error Analysis", "", f"**Status:** PASS MIT METHODISCHEN EINSCHRANKUNGEN", "", "## Executive Summary", "", f"The six frozen zero-shot runs comprise {len(cases):,} paired model-line cases and {len(cases)*2:,} predictions. The six run CSV/config/metadata hashes and the common 1,032-case order passed. LoRA produced 241/152/141 repairs and 90/90/83 regressions for Qwen 2B, Llama 3B, and Qwen 9B, respectively. The automatic taxonomy is diagnostic rather than manually validated.", "", "## Source Integrity", "", f"Cross manifest `{CROSS_MANIFEST}` matched `{EXPECTED[CROSS_MANIFEST]}`. All six CSV/config/metadata hashes and all eight authoritative upstream audit/manifest hashes matched. The manifest path `{requested_testset}` currently no longer contains the frozen 1,032-row artifact; analysis therefore used the existing byte-identical copy `{resolved_testset}` with the required SHA256 `{manifest['testset']['sha256']}`. This path drift is a documentation warning, not an analyzed-content difference. Case-order SHA256: `{case_hash}`.", "", "## Method", "", f"No installed SQL AST parser was available. `{PARSER_NAME}` `{PARSER_VERSION}` therefore served as the documented fallback. SQLite result diagnostics used `mode=ro`, `query_only`, a progress guard, and a 10,000-row cap. Multi-label differences are evidence-graded E1-E4 and never override Execution Match.", "", "## Outcome Transitions", "", "| Model | T1 | T1a | T1b | T2 | T2a | T2b | T3 | T3a | T3b | T4 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in transitions: lines.append(f"| {r['model_line']} | {r['T1_repairs']} | {r['T1a_nonexec_to_correct']} | {r['T1b_execwrong_to_correct']} | {r['T2_regressions']} | {r['T2a_correct_to_nonexec']} | {r['T2b_correct_to_execwrong']} | {r['T3_persistent_errors']} | {r['T3a_both_nonexec']} | {r['T3b_at_least_one_execwrong']} | {r['T4_stable_correct']} |")
    lines += ["", "## Automatic Error Profiles", ""]
    for model in MODEL_ORDER:
        lines += [f"### {MODEL_LABELS[model]}", "", "| Rank | Repaired label | n | Introduced label | n | Persistent label | n |", "|---:|---|---:|---|---:|---|---:|"]
        a,b,c=top(model,"repaired"),top(model,"introduced"),top(model,"persistent")
        for i in range(5): lines.append(f"| {i+1} | `{a[i]['error_label']}` | {a[i]['repaired']} | `{b[i]['error_label']}` | {b[i]['introduced']} | `{c[i]['error_label']}` | {c[i]['persistent']} |")
        lines += ["", "These are overlapping automatic labels; a repaired component label does not prove that this component alone caused the outcome transition.", ""]
    lines += ["## Qwen-2B Termination Integration", "", f"The independent eight-condition sensitivity audit contained {cap_summary['observations']:,} observations capped at 256. All {cap_summary['reached_512_again']:,} reached 512 again, none terminated normally, no additional Execution Match arose, and all {cap_summary['repetition_rule']:,} met the repetition rule. At 256, {cap_summary['semantically_correct_first_sql']:,} had an execution-matching extracted first statement, {cap_summary['executable_but_wrong_first_sql']:,} were executable but wrong, and {cap_summary['not_executable_first_sql']:,} were not executable. Thus `COMPLETION_LIMIT_REACHED` is paired with `REPETITIVE_GENERATION`, not automatically `TRUNCATED_SQL`. The primary zero-shot Qwen-2B run contributes only {cap_summary['zero_shot_observations']} of these observations.", "", "## Alternative Valid SQL", "", f"Among T4 cases, {alt_struct} starting/LoRA pairs were textually or structurally different while both execution-matched. They are recorded as `ALTERNATIVE_VALID_FORMULATION`, not errors. This demonstrates why String EM and normalized EM cannot replace execution-based evaluation; it does not prove equivalence on every possible database instance.", "", "## Explorative Category Inference", "", "Eleven prespecified families were compared with paired McNemar tests and 10,000 bootstrap resamples (seed 20260716), with Holm correction separately within each model line. Labels are rule-based outcomes rather than manually validated gold categories, so inference is explicitly exploratory.", "", "## Manual Validation Package", "", f"The stratified file contains {len(review_rows)} unreviewed cases (60 per line where available), and the blinded double-coding package contains {len(blind)} cases (10 per line). Human fields remain empty. No inter-rater reliability is claimed.", "", "## Explorative Few-shot Harm", "", f"The secondary analysis includes only zero-shot-correct/few-shot-wrong cases from the five prespecified significantly harmful LoRA conditions and stores {len(fewshot_rows)} label rows. Demo-copy labels are E3 associations. They do not establish that the demonstration caused the error.", "", "## Answers to Q1-Q10", ""]
    for qi, model in enumerate(MODEL_ORDER,1):
        a=top(model,"repaired",3); lines.append(f"**Q{qi}. {MODEL_LABELS[model]} repairs.** Most frequent automatic repaired labels: " + ", ".join(f"`{x['error_label']}` ({x['repaired']})" for x in a) + ".")
    lines += ["", "**Q4. Introduced errors.** All three lines include LoRA regressions; their introduced labels are reported above and include both execution/schema and semantic component differences.", "", "**Q5. Persistent errors.** The top persistent labels per line are reported above. Multi-component schema, projection, filter, join, and result differences remain visible; automatic labels do not isolate a single cause.", "", "**Q6. Source of LoRA gains.** Repairs include both non-executable-to-correct and executable-wrong-to-correct transitions. Therefore gains are compatible with improved executability, output control, and semantic query logic, but no complete causal decomposition is possible.", "", "**Q7. Technical-to-semantic shift.** The coarse grouped table permits a descriptive comparison. Because groups overlap and remaining errors are conditioned on different error totals, it supports only descriptive, not causal, language.", "", "**Q8. Qwen 2B versus Qwen 9B.** Their profiles differ descriptively under comparison class B+. Parameter count is not an identified cause.", "", "**Q9. Llama versus Qwen.** Differences are comparison class B because model type and native prompt format differ; architecture is not an identified cause.", "", "**Q10. Few-shot harm.** Added tables/joins, query-structure changes, and demo-overlap heuristics are quantified in the secondary table. They are exploratory associations only.", "", "## Thesis-ready Supported Statements", ""]
    supported = [
        f"LoRA yields net zero-shot gains despite regressions: repairs/regressions are 241/90 (Qwen 2B), 152/90 (Llama 3B), and 141/83 (Qwen 9B).",
        "A material subset of repairs begins with executable but incorrect SQL, so improved executability alone cannot explain the gains.",
        "Automatic multi-label profiles show that technical and semantic component differences coexist within individual cases.",
        "Qwen-2B generation-limit behavior is repetitive rather than merely late completion: 2,215/2,215 observations reached 512 again and produced no new match.",
        f"At least {alt_struct} stable-correct pairs use differing SQL formulations, illustrating the limitations of text exactness metrics.",
        "LoRA introduces regressions in every model line even when aggregate EMA improves.",
        "Persistent errors remain after LoRA and span multiple SQL component families.",
        "The Qwen 2B and Qwen 9B error profiles may be compared descriptively (class B+), not causally by parameter count.",
        "Llama-versus-Qwen profile differences are class B because prompt and model families differ.",
        "Few-shot harm labels are exploratory associations and require manual validation before mechanistic interpretation.",
    ]
    for i,s in enumerate(supported,1): lines.append(f"{i}. {s}")
    lines += ["", "## Statements Not Supported", ""]
    unsupported = ["Every Base-to-LoRA gain is a repair without regressions.", "LoRA fixes all syntax or join errors.", "Every structural difference from gold SQL is an error.", "Execution Match proves equivalence on every possible database instance.", "Qwen 9B has fewer errors exclusively because it has more parameters.", "Llama's architecture causally determines its profile.", "Few-shot demonstrations cause every newly observed error.", "The automatic taxonomy is error-free without human review.", "Repetition explains the full Qwen-2B LoRA gain.", "Unclassified cases may be assigned to the most likely category." ]
    for i,s in enumerate(unsupported,1): lines.append(f"{i}. {s}")
    lines += ["", "## Methodological Limitations", ""]
    limitations = ["Automatic labels are rule-based and not fully manually validated.", "Multiple labels may occur per case and are statistically dependent.", "A component difference is not necessarily the causal reason for a result mismatch.", "Execution Match evaluates the observed database instance only.", "Gold SQL is not the only possible valid formulation.", "The fallback parser can be incomplete on complex SQLite syntax.", "The stratified review sample is not a prevalence sample.", "Cross-model comparisons remain class B or B+.", "Few-shot harm analysis is exploratory.", "No authoritative Spider difficulty labels are reconstructed.", "Error families overlap.", "Semantic quality and output control cannot be causally separated completely."]
    for i,s in enumerate(limitations,1): lines.append(f"{i}. {s}")
    lines += ["", "## Read-only Confirmation", "", "No training, evaluation, generation, model/adapter/tokenizer/BGE load, download, or existing artifact modification occurred. SQLite was opened read-only. Only the additive files listed in the companion manifest were created.", "", "## Status", "", "`CROSS-MODEL-ZERO-SHOT-ERROR-ANALYSIS: PASS MIT METHODISCHEN EINSCHRANKUNGEN`", "", "`EXPERIMENTS MISSING: NEIN`", "", "`RERUN REQUIRED: NEIN`", ""]
    write_text_new(OUT["audit"], "\n".join(lines))

    script_hash = sha256(Path(__file__))
    new_paths = [p for k,p in OUT.items() if k != "manifest"] + [x.with_suffix(ext) for x in PLOTS.values() for ext in (".png",".pdf")]
    out_manifest = {
        "audit_status": "PASS MIT METHODISCHEN EINSCHRANKUNGEN", "date": DATE, "classification": "cross-model paired zero-shot error analysis", "read_only": True,
        "source_integrity": "PASS MIT WARNUNGEN", "cross_model_source_audit": {"path": CROSS_AUDIT, "sha256": EXPECTED[CROSS_AUDIT]}, "cross_model_source_manifest": {"path": CROSS_MANIFEST, "sha256": EXPECTED[CROSS_MANIFEST]}, "cross_model_table": {"path": CROSS_TABLE, "sha256": EXPECTED[CROSS_TABLE]},
        "authoritative_upstream_sources": manifest["authoritative_sources"],
        "zero_shot_runs": [{k:r.get(k) for k in ("model","role","run_id","config_path","config_sha256","csv_path","csv_sha256","metadata_path","metadata_sha256","case_count","case_ids_sha256")} for r in run_records],
        "model_provenance": MODEL_INFO, "testset": {**manifest["testset"], "requested_path_current_sha256": sha256(requested_testset), "resolved_frozen_path": resolved_testset, "resolved_frozen_sha256": sha256(resolved_testset), "path_drift_warning": testset_path_warning}, "cases_per_line": N, "paired_model_line_cases": len(cases), "predictions": len(cases)*2,
        "transitions": transitions, "taxonomy": {"version": TAXONOMY_VERSION, "families": FAMILY_LABELS, "evidence_levels": ["E1","E2","E3","E4"]},
        "parser": {"name": PARSER_NAME, "version": PARSER_VERSION, "fallback_used": True, "external_parser_available": False, "parse_counts": parse_summary},
        "automatic_classification": {"long_label_rows": len(long_rows), "multi_label": True, "execution_match_authoritative": True},
        "category_statistics": {"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED, "confidence": .95, "holm_families": "11 predefined families separately per model line", "inference": "explorative"},
        "alternative_valid_sql": {"all_t4_rows": len(alternative), "structurally_different": alt_struct},
        "qwen2b_termination_integration": cap_summary,
        "manual_review": {"sample_size": len(review_rows), "target_per_model": 60, "seed": SEED, "human_validated": False, "double_coding_size": len(blind), "roles_blinded": True},
        "fewshot_harm": {"analysis_class": "EXPLORATIVE FEW-SHOT ERROR ANALYSIS", "conditions": selected_fs, "label_rows": len(fewshot_rows), "counts_by_model_condition": {f"{m}:{c}":n for (m,c),n in fs_summary.items()}, "causal_claim": False},
        "methodological_limitations": limitations, "experiments_missing": False, "rerun_required": False,
        "analysis_script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": script_hash},
        "new_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in new_paths],
        "manifest_self_hash": None, "manifest_self_hash_note": "Computed externally after write to avoid a recursive self-hash.",
    }
    write_text_new(OUT["manifest"], json.dumps(out_manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"status": out_manifest["audit_status"], "transitions": transitions, "labels": len(long_rows), "alternative_structural": alt_struct, "review": len(review_rows), "fewshot_labels": len(fewshot_rows), "manifest_sha256": sha256(OUT["manifest"])}, indent=2))


if __name__ == "__main__":
    main()
