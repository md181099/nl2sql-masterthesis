#!/usr/bin/env python3
"""Read-only inventory and coverage preflight for locally available SQL AST parsers."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
N = 1032
EXPECTED_PREDICTIONS = 6192
EXPECTED_TOTAL = 7224
SOURCE_MANIFEST = ROOT / "audits/cross_model_zero_shot_error_analysis_manifest_20260716.json"
CROSS_MANIFEST = ROOT / "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json"
OUT_INVENTORY = ROOT / "audits/derived/local_sql_ast_parser_inventory_20260716.csv"
OUT_RESULTS = ROOT / "audits/derived/local_sql_ast_parser_preflight_results_20260716.csv"
OUT_SUMMARY = ROOT / "audits/derived/local_sql_ast_parser_preflight_summary_20260716.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "NOT_AVAILABLE"


def inventory() -> list[dict[str, Any]]:
    external = ["sqlglot", "sqlparse", "moz_sql_parser", "sql_metadata", "sqlfluff", "pglast", "sql_ast"]
    rows = []
    for name in external:
        spec = importlib.util.find_spec(name)
        ast = name not in {"sqlparse", "sql_metadata"}
        rows.append({
            "parser": name, "source": "installed Python environment", "locally_available": bool(spec),
            "version": package_version(name) if spec else "NOT_AVAILABLE", "parser_class": "AST parser" if ast else "tokenizer/metadata utility",
            "ast_capable": ast and bool(spec), "sqlite_support": "UNKNOWN" if spec else "NO",
            "alias_resolution": "UNKNOWN" if spec else "NO", "subqueries": "UNKNOWN" if spec else "NO",
            "set_operations": "UNKNOWN" if spec else "NO", "network_dependency": "not tested" if spec else "none (not installed)",
            "usable": False, "reason": "not installed" if not spec else "not selected before safety/API validation",
        })
    rows += [
        {"parser": "official/project Spider SQL parser", "source": "project search: src/scripts/third_party/vendor/evaluation/evaluators/spider/data", "locally_available": False, "version": "NOT_AVAILABLE", "parser_class": "official Spider component parser", "ast_capable": True, "sqlite_support": "Spider SQL grammar", "alias_resolution": "UNAVAILABLE", "subqueries": "UNAVAILABLE", "set_operations": "UNAVAILABLE", "network_dependency": "none", "usable": False, "reason": "no parser implementation found"},
        {"parser": "project-local-sqlite-clause-fallback", "source": "scripts/analyze_cross_model_zero_shot_error_taxonomy.py", "locally_available": True, "version": "1.0.0", "parser_class": "deterministic lexer/clause parser", "ast_capable": False, "sqlite_support": "partial clause-level", "alias_resolution": "partial", "subqueries": "count only", "set_operations": "count/type only", "network_dependency": "none", "usable": False, "reason": "authoritative v1 fallback; not an AST parser"},
        {"parser": "project SQL normalizers/extractor", "source": "src/06_batch_run.py and project analysis scripts", "locally_available": True, "version": "project source", "parser_class": "normalizer/extractor", "ast_capable": False, "sqlite_support": "text normalization/extraction", "alias_resolution": "NO", "subqueries": "NO", "set_operations": "NO", "network_dependency": "none", "usable": False, "reason": "not an AST parser"},
    ]
    return rows


def unavailable_row(source_type: str, model: str, role: str, row: dict[str, str]) -> dict[str, Any]:
    empty = not str(row.get("pred_sql") if source_type == "prediction" else row.get("gold_sql", "")).strip()
    sql = row.get("pred_sql", "") if source_type == "prediction" else row.get("gold_sql", "")
    unsupported = "UNSUPPORTED"
    return {
        "source_type": source_type, "model_line": model, "role": role, "case_id": row["id"], "db_id": row["db_id"], "sql": sql,
        "execution_status": "empty_extraction" if empty else ("correct" if row.get("exec_match", "").lower() == "true" else "executable_wrong" if row.get("pred_ok", "").lower() == "true" else "not_executable") if source_type == "prediction" else "gold",
        "parser_name": "NONE_LOCAL_AST", "parser_version": "NOT_AVAILABLE", "parse_success": False,
        "parse_error_type": "EMPTY_SQL" if empty else "PARSER_UNAVAILABLE", "parse_error_message": "No locally installed AST-capable SQL parser is available",
        "fallback_required": True, "statement_type": unsupported, "tables": unsupported, "aliases": unsupported, "columns": unsupported,
        "select_expressions": unsupported, "aggregations": unsupported, "distinct": unsupported, "join_count": unsupported,
        "join_types": unsupported, "join_tables": unsupported, "join_conditions": unsupported, "where_conditions": unsupported,
        "operators": unsupported, "literals": unsupported, "group_by": unsupported, "having": unsupported, "order_by": unsupported,
        "sort_directions": unsupported, "limit": unsupported, "subquery_count": unsupported, "nesting_depth": unsupported,
        "set_operations": unsupported, "negations": unsupported,
    }


def main() -> None:
    for path in (OUT_INVENTORY, OUT_RESULTS, OUT_SUMMARY):
        if path.exists(): raise RuntimeError(f"Target exists: {path}")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    cross = json.loads(CROSS_MANIFEST.read_text(encoding="utf-8"))
    expected_script = source_manifest["analysis_script"]
    if sha256(ROOT / expected_script["path"]) != expected_script["sha256"]:
        raise RuntimeError("Authoritative analysis script hash mismatch")
    runs = [x for x in cross["runs"] if x["condition"] == "zero_shot" and x["role"] in {"base", "lora_v2"}]
    if len(runs) != 6: raise RuntimeError("Expected six zero-shot runs")
    for run in runs:
        for kind in ("csv", "config", "metadata"):
            if sha256(ROOT / run[f"{kind}_path"]) != run[f"{kind}_sha256"]:
                raise RuntimeError(f"Source hash mismatch: {run['run_id']} {kind}")
    result_rows: list[dict[str, Any]] = []
    gold_source = next(x for x in runs if x["model"] == "qwen2b" and x["role"] == "base")
    gold_rows = read_csv(ROOT / gold_source["csv_path"])
    if len(gold_rows) != N: raise RuntimeError("Gold source row count")
    model_labels = {"qwen2b": "Qwen 3.5 2B", "llama3b": "Llama 3.2 3B Instruct", "qwen9b": "Qwen 3.5 9B"}
    for row in gold_rows:
        result_rows.append(unavailable_row("gold", "Spider Dev", "gold", row))
    coverage_by_role = {}
    for run in runs:
        rows = read_csv(ROOT / run["csv_path"])
        if len(rows) != N: raise RuntimeError(f"Prediction row count: {run['run_id']}")
        for row in rows:
            result_rows.append(unavailable_row("prediction", model_labels[run["model"]], run["role"], row))
        coverage_by_role[f"{run['model']}:{run['role']}"] = {"success": 0, "total": N, "rate": 0.0}
    if len(result_rows) != EXPECTED_TOTAL: raise RuntimeError("Preflight SQL count mismatch")
    inv = inventory()
    usable_ast = [x for x in inv if x["ast_capable"] and x["usable"]]
    summary = {
        "status": "PASS", "python": platform.python_version(), "local_ast_parser": "NOT_AVAILABLE",
        "gold": {"success": 0, "total": N, "rate": 0.0},
        "predictions": {"success": 0, "total": EXPECTED_PREDICTIONS, "rate": 0.0},
        "by_model_role": coverage_by_role,
        "by_execution_status": {"not_assessed": EXPECTED_TOTAL, "reason": "no local AST parser"},
        "by_complexity": {"not_assessed": EXPECTED_TOTAL, "reason": "no local AST parser; components are UNSUPPORTED"},
        "parse_errors": {"PARSER_UNAVAILABLE": EXPECTED_TOTAL - sum(not x["sql"].strip() for x in result_rows), "EMPTY_SQL": sum(not x["sql"].strip() for x in result_rows)},
        "decision_stage": "C", "full_ast_sensitivity_allowed": False, "limited_ast_sensitivity_allowed": False,
        "ast_sensitivity_justified": False, "reason": "No locally available AST-capable parser; operational Stage A/B coverage thresholds cannot be met.",
        "ast_sensitivity_outputs_created": False, "existing_clause_fallback_remains_authoritative": True,
        "network_used": False, "packages_installed": False, "environment_modified": False,
        "inventory_rows": len(inv), "usable_ast_parsers": len(usable_ast), "sql_strings": len(result_rows),
        "source_manifest": {"path": str(SOURCE_MANIFEST.relative_to(ROOT)), "sha256": sha256(SOURCE_MANIFEST)},
    }
    write_csv_new(OUT_INVENTORY, inv)
    write_csv_new(OUT_RESULTS, result_rows)
    write_json_new(OUT_SUMMARY, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
