#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HISTORY_COLUMNS = [
    "step",
    "epoch",
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "entropy",
    "num_tokens",
    "mean_token_accuracy",
    "train_runtime",
    "timestamp",
    "elapsed_seconds",
]

METRIC_COLUMNS = [
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "entropy",
    "num_tokens",
    "mean_token_accuracy",
    "train_runtime",
]

LEGACY_METRIC_KEYS = set(METRIC_COLUMNS) | {"train_loss"}
DICT_PATTERN = re.compile(r"\{[^{}\n]*\}")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "training_run"


def run_slug_from_adapter_dir(project_root: Path, adapter_dir: Path) -> str:
    adapter_dir = adapter_dir.resolve()
    project_root = project_root.resolve()
    try:
        rel = adapter_dir.relative_to(project_root)
    except ValueError:
        return sanitize_filename(adapter_dir.name)

    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "adapters":
        return sanitize_filename(f"{parts[1]}__{parts[2]}")
    return sanitize_filename("__".join(parts))


def history_paths(adapter_dir: Path) -> tuple[Path, Path]:
    return adapter_dir / "training_history.csv", adapter_dir / "training_history.jsonl"


def central_metric_paths(project_root: Path, adapter_dir: Path) -> tuple[Path, Path]:
    run_slug = run_slug_from_adapter_dir(project_root, adapter_dir)
    metrics_dir = project_root / "results" / "training_metrics"
    return (
        metrics_dir / f"{run_slug}_training_history.csv",
        metrics_dir / f"{run_slug}_training_history.jsonl",
    )


def central_plot_dir(project_root: Path, adapter_dir: Path) -> Path:
    run_slug = run_slug_from_adapter_dir(project_root, adapter_dir)
    return project_root / "results" / "training_plots" / run_slug


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def coerce_int(value: Any) -> int | None:
    number = coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _format_value(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return ""
    return f"{number:.12g}"


def normalize_history_row(row: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for column in HISTORY_COLUMNS:
        value = row.get(column)
        if column in {"timestamp"}:
            normalized[column] = "" if value is None else str(value)
        elif column == "step":
            step = coerce_int(value)
            normalized[column] = "" if step is None else str(step)
        else:
            normalized[column] = _format_value(value)
    return normalized


def row_from_trainer_log(
    logs: dict[str, Any],
    *,
    step: int | None,
    epoch: Any,
    timestamp: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, str] | None:
    if not any(key in logs for key in LEGACY_METRIC_KEYS):
        return None
    loss = logs.get("loss", logs.get("train_loss"))
    raw_row: dict[str, Any] = {
        "step": step,
        "epoch": logs.get("epoch", epoch),
        "loss": loss,
        "eval_loss": logs.get("eval_loss"),
        "grad_norm": logs.get("grad_norm"),
        "learning_rate": logs.get("learning_rate"),
        "entropy": logs.get("entropy"),
        "num_tokens": logs.get("num_tokens"),
        "mean_token_accuracy": logs.get("mean_token_accuracy"),
        "train_runtime": logs.get("train_runtime"),
        "timestamp": timestamp if timestamp is not None else utc_timestamp(),
        "elapsed_seconds": elapsed_seconds,
    }
    return normalize_history_row(raw_row)


def _csv_has_header(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def append_history_row(csv_path: Path, jsonl_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_history_row(row)

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HISTORY_COLUMNS)
        if not _csv_has_header(csv_path):
            writer.writeheader()
        writer.writerow(normalized)

    with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")


def _merge_history_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, str]:
    merged = normalize_history_row(existing)
    normalized_incoming = normalize_history_row(incoming)
    for key, value in normalized_incoming.items():
        if value != "":
            merged[key] = value
    return normalize_history_row(merged)


def upsert_history_row(csv_path: Path, jsonl_path: Path, row: dict[str, Any]) -> None:
    normalized = normalize_history_row(row)
    incoming_step = coerce_int(normalized.get("step"))
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        rows = read_history_csv(csv_path)
    elif jsonl_path.exists():
        rows = read_history_jsonl(jsonl_path)

    updated = False
    if incoming_step is not None:
        for index, existing in enumerate(rows):
            if coerce_int(existing.get("step")) == incoming_step:
                rows[index] = _merge_history_row(existing, normalized)
                updated = True
                break

    if not updated:
        rows.append(normalized)

    rows.sort(
        key=lambda item: (
            coerce_int(item.get("step")) is None,
            coerce_int(item.get("step")) or 0,
        )
    )
    write_history_files(rows, csv_path, jsonl_path)


def write_history_files(rows: Iterable[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> int:
    rows_normalized = [normalize_history_row(row) for row in rows]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_normalized)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows_normalized:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows_normalized)


def mirror_history_files(source_csv: Path, source_jsonl: Path, target_csv: Path, target_jsonl: Path) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if source_csv.exists():
        shutil.copy2(source_csv, target_csv)
    if source_jsonl.exists():
        shutil.copy2(source_jsonl, target_jsonl)


def load_existing_steps(csv_path: Path, jsonl_path: Path) -> set[int]:
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        rows = read_history_csv(csv_path)
    elif jsonl_path.exists():
        rows = read_history_jsonl(jsonl_path)
    steps: set[int] = set()
    for row in rows:
        step = coerce_int(row.get("step"))
        if step is not None:
            steps.add(step)
    return steps


def read_history_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [normalize_history_row(row) for row in csv.DictReader(csv_file)]


def read_history_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(normalize_history_row(json.loads(stripped)))
    return rows


def _extract_steps_per_epoch(log_text: str) -> int | None:
    patterns = [
        r"trainer_expected_steps_per_epoch=(\d+)",
        r"expected_steps_per_epoch=(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return int(match.group(1))
    return None


def parse_legacy_training_log(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    steps_per_epoch = _extract_steps_per_epoch(text)
    rows: list[dict[str, str]] = []

    for match in DICT_PATTERN.finditer(text):
        try:
            payload = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict) or not any(key in payload for key in LEGACY_METRIC_KEYS):
            continue

        step = coerce_int(payload.get("step") or payload.get("global_step"))
        epoch = payload.get("epoch")
        if step is None and steps_per_epoch is not None:
            epoch_float = coerce_float(epoch)
            if epoch_float is not None:
                step = max(0, int(round(epoch_float * steps_per_epoch)))
        if step is None:
            step = (len(rows) + 1) * 10

        row = row_from_trainer_log(
            payload,
            step=step,
            epoch=epoch,
            timestamp="",
            elapsed_seconds=None,
        )
        if row is not None:
            rows.append(row)

    return _dedupe_rows_by_step(rows)


def parse_trainer_state(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    log_history = payload.get("log_history", [])
    if not isinstance(log_history, list):
        return []

    rows: list[dict[str, str]] = []
    for item in log_history:
        if not isinstance(item, dict):
            continue
        step = coerce_int(item.get("step"))
        row = row_from_trainer_log(
            item,
            step=step,
            epoch=item.get("epoch"),
            timestamp="",
            elapsed_seconds=None,
        )
        if row is not None:
            rows.append(row)
    return _dedupe_rows_by_step(rows)


def _dedupe_rows_by_step(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_step: dict[int, dict[str, str]] = {}
    fallback: list[dict[str, str]] = []
    for row in rows:
        step = coerce_int(row.get("step"))
        if step is None:
            fallback.append(normalize_history_row(row))
        else:
            by_step[step] = normalize_history_row(row)
    return [by_step[step] for step in sorted(by_step)] + fallback


def _find_latest_trainer_state(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("trainer_state.json"))
    candidates.extend(run_dir.glob("checkpoints/checkpoint-*/trainer_state.json"))
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        match = re.search(r"checkpoint-(\d+)", str(path))
        checkpoint_step = int(match.group(1)) if match else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return checkpoint_step, mtime

    return max(candidates, key=sort_key)


def load_history_from_path(path: Path) -> list[dict[str, str]]:
    path = path.resolve()
    if path.is_dir():
        csv_path, jsonl_path = history_paths(path)
        if csv_path.exists():
            return read_history_csv(csv_path)
        if jsonl_path.exists():
            return read_history_jsonl(jsonl_path)

        logs = sorted(path.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        if logs:
            return parse_legacy_training_log(logs[-1])

        trainer_state = _find_latest_trainer_state(path)
        if trainer_state is not None:
            return parse_trainer_state(trainer_state)
        return []

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_history_csv(path)
    if suffix == ".jsonl":
        return read_history_jsonl(path)
    if suffix == ".json" and path.name == "trainer_state.json":
        return parse_trainer_state(path)
    if suffix in {".log", ".txt", ".out"}:
        return parse_legacy_training_log(path)
    raise ValueError(f"Unsupported history input: {path}")


def materialize_history_for_adapter(
    rows: Iterable[dict[str, Any]],
    *,
    adapter_dir: Path,
    project_root: Path,
    mirror_central: bool = True,
) -> tuple[Path, Path, int]:
    csv_path, jsonl_path = history_paths(adapter_dir)
    row_count = write_history_files(rows, csv_path, jsonl_path)
    if mirror_central:
        central_csv, central_jsonl = central_metric_paths(project_root, adapter_dir)
        mirror_history_files(csv_path, jsonl_path, central_csv, central_jsonl)
    return csv_path, jsonl_path, row_count
