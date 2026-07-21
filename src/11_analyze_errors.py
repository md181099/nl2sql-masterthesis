#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging


logger = logging.getLogger(__name__)

ARTIFACT_PATTERNS = [
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"<think\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\|assistant\|>", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"```"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one run CSV for common NL2SQL error patterns.")
    parser.add_argument("--csv", required=True, help="Path to one results CSV file.")
    parser.add_argument("--top_n", type=int, default=10, help="Top N entries for db/error listings.")
    parser.add_argument(
        "--write_markdown",
        action="store_true",
        help="Also write a markdown report to results/error_analysis_<run>.md",
    )
    parser.add_argument(
        "--markdown_path",
        default="",
        help="Optional custom markdown output path (used only with --write_markdown).",
    )
    parser.add_argument(
        "--overwrite_markdown",
        action="store_true",
        help="Allow overwriting markdown output if it already exists.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _classify_error(pred_error: str) -> str:
    text = pred_error.strip().lower()
    if not text:
        return "no_error_text"
    if "no such column" in text:
        return "no such column"
    if "no such table" in text:
        return "no such table"
    if "syntax error" in text:
        return "syntax error"
    if "ambiguous column" in text or "ambiguous column name" in text:
        return "ambiguous column"
    return "other"


def _artifact_counts(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {
        "has_think_close": "</think>" in lower,
        "has_assistant_tag": "<|assistant|>" in lower,
        "has_artifact": any(p.search(text) is not None for p in ARTIFACT_PATTERNS),
    }


def _to_percent(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    if 0.0 <= avg <= 1.0:
        return avg * 100.0
    return avg


def _safe_cell(text: str, max_len: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _render_markdown(
    *,
    csv_name: str,
    total_rows: int,
    pred_ok_zero: int,
    exec_match_zero: int,
    error_counter: Counter[str],
    token_avg: float | None,
    char_avg: float | None,
    top_db_rows: list[tuple[str, int]],
    think_count: int,
    assistant_tag_count: int,
    artifact_count: int,
    example_rows: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Error Analysis: {csv_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: {total_rows}")
    lines.append(f"- pred_ok = 0: {pred_ok_zero}")
    lines.append(f"- exec_match = 0: {exec_match_zero}")
    lines.append(f"- token_accuracy avg: {token_avg:.2f}%" if token_avg is not None else "- token_accuracy avg: n/a")
    lines.append(f"- char_accuracy avg: {char_avg:.2f}%" if char_avg is not None else "- char_accuracy avg: n/a")
    lines.append("")
    lines.append("## SQL Error Categories")
    lines.append("")
    for label, count in error_counter.most_common():
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("## Artifact Counts")
    lines.append("")
    lines.append(f"- Contains `</think>`: {think_count}")
    lines.append(f"- Contains `<|assistant|>`: {assistant_tag_count}")
    lines.append(f"- Contains obvious artifact markers: {artifact_count}")
    lines.append("")
    lines.append("## Top Problematic DBs")
    lines.append("")
    for db_id, count in top_db_rows:
        lines.append(f"- {db_id}: {count}")
    lines.append("")
    lines.append("## Example Error Cases")
    lines.append("")
    for idx, row in enumerate(example_rows, start=1):
        lines.append(f"### Case {idx}")
        lines.append("")
        lines.append(f"- question: `{_safe_cell(row.get('question', ''))}`")
        lines.append(f"- gold_sql: `{_safe_cell(row.get('gold_sql', ''))}`")
        lines.append(f"- pred_sql: `{_safe_cell(row.get('pred_sql', ''))}`")
        lines.append(f"- pred_error: `{_safe_cell(row.get('pred_error', ''))}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(__file__).resolve().parents[1]

    csv_path = _resolve_path(project_root, args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    total_rows = 0
    pred_ok_zero = 0
    exec_match_zero = 0
    error_counter: Counter[str] = Counter()
    token_values: list[float] = []
    char_values: list[float] = []
    db_problem_counter: Counter[str] = Counter()

    think_count = 0
    assistant_tag_count = 0
    artifact_count = 0

    error_examples_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        for row in reader:
            total_rows += 1

            pred_ok = _parse_int(row.get("pred_ok"))
            exec_match = _parse_int(row.get("exec_match"))
            db_id = str(row.get("db_id", "")).strip() or "unknown_db"

            if pred_ok is not None and pred_ok == 0:
                pred_ok_zero += 1
                db_problem_counter[db_id] += 1

            if exec_match is not None and exec_match == 0:
                exec_match_zero += 1
                db_problem_counter[db_id] += 1

            token_acc = _parse_float(row.get("token_accuracy"))
            if token_acc is not None:
                token_values.append(token_acc)

            char_acc = _parse_float(row.get("char_accuracy"))
            if char_acc is not None:
                char_values.append(char_acc)

            pred_error = str(row.get("pred_error", "")).strip()
            bucket = _classify_error(pred_error)
            if pred_error:
                error_counter[bucket] += 1
            elif pred_ok is not None and pred_ok == 0:
                error_counter["other"] += 1

            combined_text = "\n".join(
                [
                    str(row.get("pred_sql", "") or ""),
                    str(row.get("raw_output", "") or ""),
                ]
            )
            flags = _artifact_counts(combined_text)
            if flags["has_think_close"]:
                think_count += 1
            if flags["has_assistant_tag"]:
                assistant_tag_count += 1
            if flags["has_artifact"]:
                artifact_count += 1

            if (pred_ok is not None and pred_ok == 0) or pred_error:
                if len(error_examples_by_bucket[bucket]) < max(1, args.top_n):
                    error_examples_by_bucket[bucket].append(
                        {
                            "question": str(row.get("question", "") or ""),
                            "gold_sql": str(row.get("gold_sql", "") or ""),
                            "pred_sql": str(row.get("pred_sql", "") or ""),
                            "pred_error": pred_error,
                        }
                    )

    token_avg = _to_percent(token_values)
    char_avg = _to_percent(char_values)

    top_db = db_problem_counter.most_common(max(1, args.top_n))
    top_error_buckets = error_counter.most_common(max(1, args.top_n))

    logger.info("Analyzed CSV: %s", csv_path)
    logger.info("Total rows: %d", total_rows)
    logger.info("pred_ok=0: %d", pred_ok_zero)
    logger.info("exec_match=0: %d", exec_match_zero)
    if token_avg is None:
        logger.info("token_accuracy avg: n/a")
    else:
        logger.info("token_accuracy avg: %.2f%%", token_avg)
    if char_avg is None:
        logger.info("char_accuracy avg: n/a")
    else:
        logger.info("char_accuracy avg: %.2f%%", char_avg)

    logger.info("SQL error categories:")
    for label, count in top_error_buckets:
        logger.info("  %s: %d", label, count)

    logger.info("Artifact counts:")
    logger.info("  contains </think>: %d", think_count)
    logger.info("  contains <|assistant|>: %d", assistant_tag_count)
    logger.info("  contains obvious artifacts: %d", artifact_count)

    logger.info("Top problematic db_id entries:")
    for db_id, count in top_db:
        logger.info("  %s: %d", db_id, count)

    logger.info("Example error cases:")
    displayed = 0
    for bucket, _ in top_error_buckets:
        for row in error_examples_by_bucket.get(bucket, [])[:2]:
            displayed += 1
            logger.info(
                "  [%s] question=%r | gold_sql=%r | pred_sql=%r | pred_error=%r",
                bucket,
                _safe_cell(row.get("question", "")),
                _safe_cell(row.get("gold_sql", "")),
                _safe_cell(row.get("pred_sql", "")),
                _safe_cell(row.get("pred_error", "")),
            )
            if displayed >= max(3, args.top_n):
                break
        if displayed >= max(3, args.top_n):
            break

    if args.write_markdown:
        if args.markdown_path.strip():
            md_path = _resolve_path(project_root, args.markdown_path)
        else:
            md_path = csv_path.with_name(f"error_analysis_{csv_path.stem}.md")
        if md_path.exists() and not args.overwrite_markdown:
            raise FileExistsError(
                f"Markdown file already exists: {md_path}. "
                "Use --overwrite_markdown to replace it."
            )

        collected_examples: list[dict[str, str]] = []
        for bucket, _ in top_error_buckets:
            for row in error_examples_by_bucket.get(bucket, [])[:2]:
                collected_examples.append(row)
                if len(collected_examples) >= max(3, args.top_n):
                    break
            if len(collected_examples) >= max(3, args.top_n):
                break

        md_text = _render_markdown(
            csv_name=csv_path.name,
            total_rows=total_rows,
            pred_ok_zero=pred_ok_zero,
            exec_match_zero=exec_match_zero,
            error_counter=error_counter,
            token_avg=token_avg,
            char_avg=char_avg,
            top_db_rows=top_db,
            think_count=think_count,
            assistant_tag_count=assistant_tag_count,
            artifact_count=artifact_count,
            example_rows=collected_examples,
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")
        logger.info("Markdown report written to: %s", md_path)


if __name__ == "__main__":
    main()
