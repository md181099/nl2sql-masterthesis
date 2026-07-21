#!/usr/bin/env python3
from __future__ import annotations

"""
Create overlap-cleaned SQL-Create-Context training artifacts against Spider-Dev.

Removes rows only when normalized (question, SQL) pair overlaps with eval set.
Question-only and SQL-only overlaps are only counted/reported.
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging


logger = logging.getLogger(__name__)

Q_RE = re.compile(r"(?s)Question:\s*\n(.*?)\n\nSQL:\s*<\|im_end\|>")
A_RE = re.compile(r"(?s)<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>")


def normalize_question(value: str) -> str:
    return " ".join((value or "").lower().split())


def normalize_sql(value: str) -> str:
    normalized = (value or "").strip().lower()
    normalized = re.sub(r";+\s*$", "", normalized)
    return " ".join(normalized.split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_sft_text(text: str) -> tuple[str, str] | None:
    q_match = Q_RE.search(text)
    a_match = A_RE.search(text)
    if not (q_match and a_match):
        return None
    question = q_match.group(1).strip()
    assistant_sql = a_match.group(1).strip()
    return question, assistant_sql


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SQL-Create-Context train artifacts by removing Spider-Dev pair overlaps."
    )
    parser.add_argument(
        "--train_input_path",
        default="data/sql_create_context/train.jsonl",
        help="Input SQL-Create-Context train JSONL.",
    )
    parser.add_argument(
        "--sft_input_path",
        default="data/sql_create_context/train_sft_qwen35_9b_chat_legacy.jsonl",
        help="Input SQL-Create-Context SFT JSONL.",
    )
    parser.add_argument(
        "--eval_path",
        default="data/testcases_spider_dev_full.jsonl",
        help="Spider-Dev eval JSONL path.",
    )
    parser.add_argument(
        "--train_output_path",
        default="data/sql_create_context/train_no_spider_dev_overlap.jsonl",
        help="Output cleaned train JSONL path.",
    )
    parser.add_argument(
        "--sft_output_path",
        default="data/sql_create_context/train_sft_qwen35_9b_chat_legacy_no_spider_dev_overlap.jsonl",
        help="Output cleaned SFT JSONL path.",
    )
    parser.add_argument(
        "--manifest_path",
        default="data/sql_create_context/no_spider_dev_overlap_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output file(s) already exist. Re-run with --overwrite to replace them: "
            + ", ".join(str(p) for p in existing)
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(__file__).resolve().parents[1]

    train_input_path = resolve_path(project_root, args.train_input_path)
    sft_input_path = resolve_path(project_root, args.sft_input_path)
    eval_path = resolve_path(project_root, args.eval_path)
    train_output_path = resolve_path(project_root, args.train_output_path)
    sft_output_path = resolve_path(project_root, args.sft_output_path)
    manifest_path = resolve_path(project_root, args.manifest_path)

    for required in (train_input_path, sft_input_path, eval_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing input file: {required}")

    ensure_outputs_do_not_exist(
        [train_output_path, sft_output_path, manifest_path],
        overwrite=args.overwrite,
    )
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_input_path)
    sft_rows = load_jsonl(sft_input_path)
    eval_rows = load_jsonl(eval_path)

    logger.info("Loaded train rows: %d", len(train_rows))
    logger.info("Loaded sft rows: %d", len(sft_rows))
    logger.info("Loaded eval rows: %d", len(eval_rows))

    eval_q_map: dict[str, list[str]] = {}
    eval_sql_map: dict[str, list[str]] = {}
    eval_pair_map: dict[tuple[str, str], list[str]] = {}
    for row in eval_rows:
        eval_id = str(row.get("id", ""))
        q_norm = normalize_question(str(row.get("question", "")))
        s_norm = normalize_sql(str(row.get("gold_sql", "")))
        eval_q_map.setdefault(q_norm, []).append(eval_id)
        eval_sql_map.setdefault(s_norm, []).append(eval_id)
        eval_pair_map.setdefault((q_norm, s_norm), []).append(eval_id)

    eval_q_set = set(eval_q_map.keys())
    eval_sql_set = set(eval_sql_map.keys())
    eval_pair_set = set(eval_pair_map.keys())

    train_q_set: set[str] = set()
    train_sql_set: set[str] = set()
    train_pair_set: set[tuple[str, str]] = set()
    removed_train_ids: list[str] = []
    removed_eval_ids_set: set[str] = set()
    kept_train_rows: list[dict[str, Any]] = []

    for row in train_rows:
        row_id = str(row.get("id", ""))
        q_norm = normalize_question(str(row.get("question", "")))
        s_norm = normalize_sql(str(row.get("gold_sql", "")))
        pair_key = (q_norm, s_norm)

        train_q_set.add(q_norm)
        train_sql_set.add(s_norm)
        train_pair_set.add(pair_key)

        if pair_key in eval_pair_set:
            removed_train_ids.append(row_id)
            removed_eval_ids_set.update(eval_pair_map[pair_key])
            continue
        kept_train_rows.append(row)

    question_overlap_count = len(train_q_set & eval_q_set)
    sql_overlap_count = len(train_sql_set & eval_sql_set)
    pair_overlap_count = len(train_pair_set & eval_pair_set)

    train_by_id = {str(row.get("id", "")): row for row in train_rows}
    sft_by_id = {str(row.get("id", "")): row for row in sft_rows}
    train_ids = set(train_by_id.keys())
    sft_ids = set(sft_by_id.keys())
    ids_1to1 = train_ids == sft_ids and len(train_rows) == len(sft_rows)

    sft_parsed_by_id: dict[str, tuple[str, str]] = {}
    sft_parse_fail_ids: list[str] = []
    for row in sft_rows:
        row_id = str(row.get("id", ""))
        parsed = parse_sft_text(str(row.get("text", "")))
        if parsed is None:
            sft_parse_fail_ids.append(row_id)
            continue
        sft_parsed_by_id[row_id] = parsed

    train_sft_norm_mismatch_count = 0
    for row_id in train_ids & set(sft_parsed_by_id.keys()):
        train_row = train_by_id[row_id]
        train_q = normalize_question(str(train_row.get("question", "")))
        train_s = normalize_sql(str(train_row.get("gold_sql", "")))
        sft_q_raw, sft_s_raw = sft_parsed_by_id[row_id]
        sft_q = normalize_question(sft_q_raw)
        sft_s = normalize_sql(sft_s_raw)
        if train_q != sft_q or train_s != sft_s:
            train_sft_norm_mismatch_count += 1

    removed_sft_ids: list[str] = []
    kept_sft_rows: list[dict[str, Any]] = []
    sft_removal_mode = "id_mapping"

    if ids_1to1 and train_sft_norm_mismatch_count == 0:
        removed_train_id_set = set(removed_train_ids)
        for row in sft_rows:
            row_id = str(row.get("id", ""))
            if row_id in removed_train_id_set:
                removed_sft_ids.append(row_id)
            else:
                kept_sft_rows.append(row)
    else:
        # Safer fallback when 1:1 mapping cannot be trusted.
        sft_removal_mode = "pair_fallback"
        for row in sft_rows:
            row_id = str(row.get("id", ""))
            parsed = sft_parsed_by_id.get(row_id)
            if parsed is None:
                kept_sft_rows.append(row)
                continue
            q_raw, s_raw = parsed
            if (normalize_question(q_raw), normalize_sql(s_raw)) in eval_pair_set:
                removed_sft_ids.append(row_id)
            else:
                kept_sft_rows.append(row)

    write_jsonl(train_output_path, kept_train_rows)
    write_jsonl(sft_output_path, kept_sft_rows)

    manifest = {
        "train_input_path": str(train_input_path),
        "sft_input_path": str(sft_input_path),
        "eval_path": str(eval_path),
        "train_output_path": str(train_output_path),
        "sft_output_path": str(sft_output_path),
        "original_train_count": len(train_rows),
        "original_sft_count": len(sft_rows),
        "eval_count": len(eval_rows),
        "removed_train_count": len(removed_train_ids),
        "removed_sft_count": len(removed_sft_ids),
        "kept_train_count": len(kept_train_rows),
        "kept_sft_count": len(kept_sft_rows),
        "removed_train_ids": removed_train_ids,
        "removed_sft_ids": removed_sft_ids,
        "removed_eval_ids": sorted(removed_eval_ids_set),
        "normalization_method": "lowercase + whitespace normalization + strip trailing semicolons",
        "question_overlap_count": question_overlap_count,
        "sql_overlap_count": sql_overlap_count,
        "pair_overlap_count": pair_overlap_count,
        "id_alignment": {
            "train_ids_count": len(train_ids),
            "sft_ids_count": len(sft_ids),
            "ids_1to1": ids_1to1,
            "train_sft_norm_mismatch_count": train_sft_norm_mismatch_count,
            "sft_parse_fail_count": len(sft_parse_fail_ids),
            "sft_removal_mode": sft_removal_mode,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Wrote cleaned train rows: %d -> %d (%s removed)", len(train_rows), len(kept_train_rows), len(removed_train_ids))
    logger.info("Wrote cleaned sft rows: %d -> %d (%s removed)", len(sft_rows), len(kept_sft_rows), len(removed_sft_ids))
    logger.info("Pair overlaps (unique normalized pair keys): %d", pair_overlap_count)
    logger.info("Manifest written to: %s", manifest_path)


if __name__ == "__main__":
    main()
