#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging


logger = logging.getLogger(__name__)


SUMMARY_FIELDS = [
    "csv_file",
    "llm",
    "adapter",
    "prompt_tuning",
    "k",
    "system_prompt_variant",
    "max_input_tokens",
    "max_new_tokens",
    "total_testcases",
    "execution_success_rate",
    "execution_match_accuracy",
    "normalized_exact_match",
    "token_accuracy_avg",
    "char_accuracy_avg",
]

FILENAME_RE = re.compile(r"^run_(?P<adapter>.+)_(?P<ts>\d{8}_\d{6})\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment summaries from results/*.csv.")
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory containing run CSV files.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/experiment_summary.csv",
        help="Summary output CSV path.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Show top N runs in console by execution_match_accuracy.",
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


def _percent_mean(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    # Most project CSVs store normalized values in [0,1].
    if 0.0 <= avg <= 1.0:
        return avg * 100.0
    return avg


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _first_non_empty(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _infer_from_filename(csv_path: Path) -> dict[str, str]:
    name = csv_path.name
    match = FILENAME_RE.match(name)
    if not match:
        return {"adapter": "", "timestamp": ""}
    return {
        "adapter": match.group("adapter"),
        "timestamp": match.group("ts"),
    }


def _infer_llm_from_adapter(project_root: Path, adapter: str) -> str:
    adapter_name = adapter.strip()
    if not adapter_name or adapter_name == "base":
        return ""
    adapters_root = project_root / "adapters"
    if not adapters_root.exists():
        return ""
    candidates: list[str] = []
    for llm_dir in adapters_root.iterdir():
        if not llm_dir.is_dir():
            continue
        if (llm_dir / adapter_name).exists():
            candidates.append(llm_dir.name)
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _load_summary_row(csv_path: Path) -> dict[str, Any] | None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            logger.warning("Skipping %s (missing header).", csv_path)
            return None

        total = 0
        pred_ok_vals: list[float] = []
        exec_match_vals: list[float] = []
        normalized_exact_vals: list[float] = []
        token_accuracy_vals: list[float] = []
        char_accuracy_vals: list[float] = []

        llm = ""
        adapter = ""
        prompt_tuning = ""
        k = ""
        system_prompt_variant = ""
        max_input_tokens = ""
        max_new_tokens = ""

        for row in reader:
            total += 1
            if not llm:
                llm = _first_non_empty(
                    row,
                    ["run_llm", "llm", "model", "model_name", "model_id", "run_model_id"],
                )
            if not adapter:
                adapter = _first_non_empty(row, ["run_adapter", "adapter", "adapter_name"])
            if not prompt_tuning:
                prompt_tuning = _first_non_empty(
                    row,
                    ["run_prompt_tuning", "prompt_tuning", "prompt_mode"],
                )
            if not k:
                k = _first_non_empty(row, ["run_k", "k"])
            if not system_prompt_variant:
                system_prompt_variant = _first_non_empty(
                    row,
                    ["run_system_prompt_variant", "system_prompt_variant"],
                )
            if not max_input_tokens:
                max_input_tokens = _first_non_empty(
                    row,
                    ["run_max_input_tokens", "max_input_tokens"],
                )
            if not max_new_tokens:
                max_new_tokens = _first_non_empty(
                    row,
                    ["run_max_new_tokens", "max_new_tokens"],
                )

            pred_ok = _parse_float(row.get("pred_ok"))
            if pred_ok is not None:
                pred_ok_vals.append(pred_ok)

            exec_match = _parse_float(row.get("exec_match"))
            if exec_match is not None:
                exec_match_vals.append(exec_match)

            normalized_exact = _parse_float(row.get("normalized_exact"))
            if normalized_exact is not None:
                normalized_exact_vals.append(normalized_exact)

            token_acc = _parse_float(row.get("token_accuracy"))
            if token_acc is not None:
                token_accuracy_vals.append(token_acc)

            char_acc = _parse_float(row.get("char_accuracy"))
            if char_acc is not None:
                char_accuracy_vals.append(char_acc)

    inferred = _infer_from_filename(csv_path)
    if not adapter:
        adapter = inferred.get("adapter", "")

    return {
        "csv_file": csv_path.name,
        "llm": llm,
        "adapter": adapter,
        "prompt_tuning": prompt_tuning,
        "k": k,
        "system_prompt_variant": system_prompt_variant,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "total_testcases": total,
        "execution_success_rate": _percent_mean(pred_ok_vals),
        "execution_match_accuracy": _percent_mean(exec_match_vals),
        "normalized_exact_match": _percent_mean(normalized_exact_vals),
        "token_accuracy_avg": _percent_mean(token_accuracy_vals),
        "char_accuracy_avg": _percent_mean(char_accuracy_vals),
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
    value = row.get("execution_match_accuracy")
    if value is None:
        return (1, float("-inf"))
    return (0, float(value))


def _print_top(rows: list[dict[str, Any]], top_n: int) -> None:
    if not rows:
        logger.info("No rows available for ranking.")
        return
    logger.info("Top %d by Execution Match Accuracy:", min(top_n, len(rows)))
    for idx, row in enumerate(rows[:top_n], start=1):
        logger.info(
            "%d) %s | adapter=%s | exec_match=%.2f%% | exec_success=%.2f%% | n=%s",
            idx,
            row.get("csv_file", ""),
            row.get("adapter", "") or "n/a",
            float(row.get("execution_match_accuracy") or 0.0),
            float(row.get("execution_success_rate") or 0.0),
            row.get("total_testcases", 0),
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(__file__).resolve().parents[1]

    results_dir = _resolve_path(project_root, args.results_dir)
    output_csv = _resolve_path(project_root, args.output_csv)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files = sorted(
        p
        for p in results_dir.glob("*.csv")
        if p.name != output_csv.name and p.is_file()
    )

    logger.info("Found %d CSV files in %s", len(csv_files), results_dir)

    summary_rows: list[dict[str, Any]] = []
    for csv_file in csv_files:
        try:
            row = _load_summary_row(csv_file)
            if row is None:
                continue
            if not row.get("llm"):
                row["llm"] = _infer_llm_from_adapter(project_root, str(row.get("adapter", "")))
            summary_rows.append(row)
        except Exception as exc:
            logger.warning("Skipping %s due to error: %s", csv_file.name, exc)

    summary_rows.sort(key=_sort_key, reverse=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "csv_file": row.get("csv_file", ""),
                    "llm": row.get("llm", ""),
                    "adapter": row.get("adapter", ""),
                    "prompt_tuning": row.get("prompt_tuning", ""),
                    "k": row.get("k", ""),
                    "system_prompt_variant": row.get("system_prompt_variant", ""),
                    "max_input_tokens": row.get("max_input_tokens", ""),
                    "max_new_tokens": row.get("max_new_tokens", ""),
                    "total_testcases": row.get("total_testcases", ""),
                    "execution_success_rate": _format_metric(row.get("execution_success_rate")),
                    "execution_match_accuracy": _format_metric(row.get("execution_match_accuracy")),
                    "normalized_exact_match": _format_metric(row.get("normalized_exact_match")),
                    "token_accuracy_avg": _format_metric(row.get("token_accuracy_avg")),
                    "char_accuracy_avg": _format_metric(row.get("char_accuracy_avg")),
                }
            )

    logger.info("Successfully processed runs: %d", len(summary_rows))
    logger.info("Summary written to: %s", output_csv)
    _print_top(summary_rows, max(1, args.top_n))


if __name__ == "__main__":
    main()
