#!/usr/bin/env python3
"""Build an additive AI-assisted preannotation package without touching human fields."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "audits/cross_model_zero_shot_error_analysis_manifest_20260716.json"
REVIEW = ROOT / "audits/derived/cross_model_zero_shot_error_manual_review_sample_20260716.csv"
CASES = ROOT / "audits/derived/cross_model_zero_shot_error_analysis_cases_20260716.csv"
BLIND = ROOT / "audits/derived/cross_model_zero_shot_error_double_coding_blinded_20260716.csv"
KEY = ROOT / "audits/derived/cross_model_zero_shot_error_double_coding_key_20260716.csv"
OUT = ROOT / "audits/derived/cross_model_zero_shot_error_ai_preannotation_20260716.csv"
OUT_SUMMARY = ROOT / "audits/derived/cross_model_zero_shot_error_ai_preannotation_summary_20260716.csv"
OUT_PRIORITY = ROOT / "audits/derived/cross_model_zero_shot_error_human_review_priority_20260716.csv"

META = {"COMPLEX_MULTI_COMPONENT_ERROR", "HEURISTIC_ONLY", "MANUAL_REVIEW_REQUIRED", "UNCLASSIFIED"}
HIGH_PRIORITY = {"WRONG_SUBQUERY", "MISSING_SUBQUERY", "EXTRA_SUBQUERY", "WRONG_CORRELATION", "WRONG_NESTING_LEVEL", "WRONG_SET_OPERATION", "WRONG_UNION", "WRONG_INTERSECT", "WRONG_EXCEPT", "WRONG_JOIN_PATH", "COMPLEX_MULTI_COMPONENT_ERROR", "HEURISTIC_ONLY", "MANUAL_REVIEW_REQUIRED", "UNCLASSIFIED"}
LOW_PRIORITY = {"EMPTY_SQL_EXTRACTION", "EXTRACTOR_FAILURE", "SQL_SYNTAX_ERROR", "UNKNOWN_TABLE", "UNKNOWN_COLUMN", "AMBIGUOUS_COLUMN", "TYPE_OR_FUNCTION_ERROR", "MISSING_TABLE", "WRONG_AGGREGATION"}
PRIMARY_ORDER = [
    "EMPTY_RAW_OUTPUT", "EMPTY_SQL_EXTRACTION", "EXTRACTOR_FAILURE", "SQL_SYNTAX_ERROR", "UNKNOWN_TABLE", "UNKNOWN_COLUMN", "AMBIGUOUS_COLUMN", "TYPE_OR_FUNCTION_ERROR", "OTHER_EXECUTION_ERROR",
    "WRONG_TABLE", "MISSING_TABLE", "EXTRA_TABLE", "WRONG_COLUMN", "MISSING_COLUMN", "EXTRA_COLUMN", "WRONG_JOIN_PATH", "WRONG_JOIN_CONDITION", "MISSING_JOIN", "EXTRA_JOIN", "OVER_JOIN",
    "WRONG_AGGREGATION", "MISSING_AGGREGATION", "EXTRA_AGGREGATION", "WRONG_SELECT_EXPRESSION", "MISSING_SELECT_EXPRESSION", "EXTRA_SELECT_EXPRESSION",
    "MISSING_WHERE_CONDITION", "EXTRA_WHERE_CONDITION", "WRONG_OPERATOR", "WRONG_LITERAL_VALUE", "WRONG_GROUP_BY", "MISSING_GROUP_BY", "EXTRA_GROUP_BY", "WRONG_ORDER_COLUMN", "MISSING_ORDER_BY", "EXTRA_ORDER_BY",
    "MISSING_SUBQUERY", "EXTRA_SUBQUERY", "WRONG_SET_OPERATION", "MISSING_ROWS", "EXTRA_ROWS", "WRONG_RESULT_CARDINALITY", "COMPLEX_MULTI_COMPONENT_ERROR", "UNCLASSIFIED",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def labels(value: str) -> set[str]:
    return {x for x in str(value).split(";") if x}


def primary(values: set[str]) -> str:
    for label in PRIMARY_ORDER:
        if label in values: return label
    return sorted(values)[0] if values else "NONE"


def levels(evidence: str) -> set[str]:
    return {level for level in ("E1", "E2", "E3", "E4") if f":{level}:" in evidence}


def join(values: set[str]) -> str:
    return ";".join(sorted(values))


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists(): raise RuntimeError(f"Refusing to overwrite {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    for path in (OUT, OUT_SUMMARY, OUT_PRIORITY):
        if path.exists(): raise RuntimeError(f"Target exists: {path}")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = {x["path"]: x["sha256"] for x in manifest["new_files"]}
    for path in (REVIEW, CASES, BLIND, KEY):
        rel = str(path.relative_to(ROOT))
        if expected.get(rel) != sha256(path): raise RuntimeError(f"Source hash mismatch: {rel}")
    review = read_csv(REVIEW); cases = read_csv(CASES); blind = read_csv(BLIND); key = read_csv(KEY)
    if len(review) != 180 or len(blind) != 30 or len(key) != 30: raise RuntimeError("Review package cardinality failed")
    for row in review:
        if any(row.get(k, "") for k in row if k.startswith("human_")): raise RuntimeError("Human field is not empty")
    if any(any(k.startswith(("ai_", "human_")) or "automatic" in k for k in row) for row in blind): raise RuntimeError("Blind package contains label leakage")
    case_lookup = {(x["model_line"], x["case_id"]): x for x in cases}
    output: list[dict[str, Any]] = []
    for row in review:
        case = case_lookup[(row["model_line"], row["case_id"])]
        start = labels(row["automatic_starting_labels"]); lora = labels(row["automatic_lora_labels"])
        repaired, introduced, persistent = start-lora, lora-start, start&lora
        evidence_levels = levels(row["automatic_evidence"])
        tg = row["transition_group"]
        added_start: set[str] = set(); added_lora: set[str] = set()
        alternative = "NOT_APPLICABLE"
        if tg == "T4":
            alternative = "YES_ON_CURRENT_INSTANCE"
            added_start.add("ALTERNATIVE_VALID_FORMULATION"); added_lora.add("ALTERNATIVE_VALID_FORMULATION")
        uncertain = bool((start | lora) & HIGH_PRIORITY) or "E3" in evidence_levels or "E4" in evidence_levels
        parser_conflict = False  # No AST parser exists; absence is not disagreement.
        if parser_conflict:
            review_status, confidence, priority = "PARSER_CONFLICT", "LOW", "HIGH"
        elif uncertain:
            review_status, confidence, priority = "AMBIGUOUS", "LOW", "HIGH"
        elif not (start or lora) and tg != "T4":
            review_status, confidence, priority = "INSUFFICIENT_EVIDENCE", "LOW", "HIGH"
        else:
            direct = bool((start | lora) & LOW_PRIORITY) and ("E1" in evidence_levels)
            review_status, confidence, priority = "REVIEWED", ("HIGH" if direct else "MEDIUM"), ("LOW" if direct and len(start|lora) <= 5 else "MEDIUM")
        if tg == "T1":
            repair_assessment = "CLEAR_REPAIR" if not lora else "PARTIAL_REPAIR"
            regression_assessment = "NOT_APPLICABLE"
        elif tg == "T2":
            repair_assessment = "NOT_APPLICABLE"; regression_assessment = "CLEAR_REGRESSION" if lora else "AMBIGUOUS"
        elif tg == "T3":
            repair_assessment = "PARTIAL_REPAIR" if repaired else "NO_CLEAR_REPAIR"
            regression_assessment = "PARTIAL_REGRESSION" if introduced else "NO_CLEAR_REGRESSION"
        else:
            repair_assessment = regression_assessment = "NOT_APPLICABLE"
            priority = "HIGH"  # Alternative-valid cases are scientifically useful manual checks.
        confirmed_start = start - {"HEURISTIC_ONLY", "MANUAL_REVIEW_REQUIRED"}
        confirmed_lora = lora - {"HEURISTIC_ONLY", "MANUAL_REVIEW_REQUIRED"}
        rejected_start: set[str] = set(); rejected_lora: set[str] = set()
        level = "E3" if uncertain else "E1" if "E1" in evidence_levels else "E2" if "E2" in evidence_levels else "E4"
        reason = (
            f"Transition {tg}: starting match={case['starting_execution_match']}, LoRA match={case['lora_execution_match']}. "
            f"Primary automatic diagnoses: starting={primary(start)}, LoRA={primary(lora)}. "
            + ("Both SQLs match on the current database instance; structural equivalence beyond this instance is not claimed." if tg == "T4" else "The assessment confirms only artifact-supported labels; causal primacy remains provisional.")
        )
        item = dict(row)
        item.update({
            "ai_review_status": review_status, "ai_starting_labels": join(start | added_start), "ai_lora_labels": join(lora | added_lora),
            "ai_primary_starting_error": primary(start), "ai_primary_lora_error": primary(lora),
            "ai_repaired_labels": join(repaired), "ai_introduced_labels": join(introduced), "ai_persistent_labels": join(persistent),
            "ai_repair_assessment": repair_assessment, "ai_regression_assessment": regression_assessment,
            "ai_alternative_valid_formulation": alternative,
            "ai_clause_labels_confirmed": f"starting={join(confirmed_start)}|lora={join(confirmed_lora)}",
            "ai_clause_labels_rejected": f"starting={join(rejected_start)}|lora={join(rejected_lora)}",
            "ai_ast_labels_confirmed": "NOT_AVAILABLE", "ai_ast_labels_rejected": "NOT_AVAILABLE",
            "ai_parser_disagreement": "AST_NOT_AVAILABLE", "ai_confidence": confidence, "ai_evidence_level": level,
            "ai_reasoning_summary": reason, "ai_human_review_priority": priority,
        })
        output.append(item)
    if any(any(out.get(k, "") != src.get(k, "") for k in src) for src, out in zip(review, output)):
        raise RuntimeError("Original review fields changed")
    model_rows: list[dict[str, Any]] = []
    for model in sorted({x["model_line"] for x in output}):
        rows = [x for x in output if x["model_line"] == model]; c = Counter(x["ai_review_status"] for x in rows); conf = Counter(x["ai_confidence"] for x in rows); pri = Counter(x["ai_human_review_priority"] for x in rows)
        model_rows.append({"summary_type": "model_line", "group": model, "cases": len(rows), "reviewed": c["REVIEWED"], "ambiguous": c["AMBIGUOUS"], "insufficient_evidence": c["INSUFFICIENT_EVIDENCE"], "parser_conflict": c["PARSER_CONFLICT"], "high_confidence": conf["HIGH"], "medium_confidence": conf["MEDIUM"], "low_confidence": conf["LOW"], "high_review_priority": pri["HIGH"], "medium_review_priority": pri["MEDIUM"], "low_review_priority": pri["LOW"]})
    transition_rows: list[dict[str, Any]] = []
    for tg in ("T1", "T2", "T3", "T4"):
        rows = [x for x in output if x["transition_group"] == tg]
        transition_rows.append({"summary_type": "transition", "group": tg, "cases": len(rows), "clear_confirmation": sum(x["ai_review_status"] == "REVIEWED" for x in rows), "label_change_proposed": sum("ALTERNATIVE_VALID_FORMULATION" in x["ai_starting_labels"] and "ALTERNATIVE_VALID_FORMULATION" not in x["automatic_starting_labels"] for x in rows), "parser_conflict": 0, "high_review_priority": sum(x["ai_human_review_priority"] == "HIGH" for x in rows)})
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_priority = sorted(output, key=lambda x: (priority_order[x["ai_human_review_priority"]], x["model_line"], x["transition_group"], x["review_id"]))
    priority_rows = [{"priority_rank": i, **row} for i, row in enumerate(sorted_priority, 1)]
    write_csv_new(OUT, output); write_csv_new(OUT_SUMMARY, model_rows + transition_rows); write_csv_new(OUT_PRIORITY, priority_rows)
    print(json.dumps({"cases": len(output), "status": Counter(x["ai_review_status"] for x in output), "confidence": Counter(x["ai_confidence"] for x in output), "priority": Counter(x["ai_human_review_priority"] for x in output), "human_fields_modified": False, "blind_package_modified": False}, default=dict, indent=2))


if __name__ == "__main__":
    main()
