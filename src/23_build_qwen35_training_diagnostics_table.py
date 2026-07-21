#!/usr/bin/env python3
"""Build provenance-checked Qwen 3.5 training diagnostic tables.

The builder joins trainer state and generalized post-hoc loss results only
after validating model, step, epoch, adapter hash, validation hash, and file
provenance. Missing diagnostics remain null; no values are interpolated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from src.training_history_utils import coerce_float, coerce_int
except ModuleNotFoundError:
    from training_history_utils import coerce_float, coerce_int


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
STATUS_COMPLETE = "COMPLETE"
STATUS_TRAINING_ONLY = "TRAINING_ONLY"
STATUS_INCOMPLETE = "INCOMPLETE"

WIDE_COLUMNS = [
    "model",
    "model_id",
    "model_size",
    "run_name",
    "lora_r",
    "lora_alpha",
    "lora_alpha_over_r",
    "learning_rate",
    "seed",
    "training_dataset_sha256",
    "validation_dataset",
    "validation_sha256",
    "epoch",
    "global_step",
    "is_best_checkpoint",
    "is_last_checkpoint",
    "official_eval_loss",
    "train_loss_raw_last",
    "train_loss_epoch_mean",
    "train_loss_epoch_median",
    "train_loss_smoothed_at_checkpoint",
    "learning_rate_at_checkpoint",
    "grad_norm_at_checkpoint",
    "eval_runtime",
    "eval_samples_per_second",
    "train_runtime",
    "best_metric",
    "best_model_checkpoint",
    "total_flos",
    "eval_fullchat_pack_macro_loss",
    "eval_fullchat_token_micro_loss",
    "eval_assistant_completion_loss",
    "eval_sql_loss",
    "eval_sql_perplexity",
    "eval_sql_token_accuracy",
    "sqlcc_fullchat_loss",
    "train_others_fullchat_loss",
    "sqlcc_sql_loss",
    "train_others_sql_loss",
    "sqlcc_sql_token_accuracy",
    "train_others_sql_token_accuracy",
    "sql_source_loss_gap",
    "sql_source_accuracy_gap",
    "fullchat_tokens",
    "assistant_completion_tokens",
    "sql_tokens",
    "sqlcc_sql_tokens",
    "train_others_sql_tokens",
    "checkpoint_path",
    "checkpoint_adapter_sha256",
    "posthoc_result_path",
    "posthoc_result_sha256",
]

TRAIN_POINT_COLUMNS = [
    "model",
    "run_name",
    "global_step",
    "epoch",
    "training_loss_raw",
    "training_loss_smoothed",
    "learning_rate",
    "grad_norm",
]

LONG_ID_COLUMNS = [
    "model",
    "model_id",
    "model_size",
    "run_name",
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "seed",
    "validation_sha256",
    "epoch",
    "global_step",
    "is_best_checkpoint",
    "is_last_checkpoint",
]

LONG_METRICS = [
    "official_eval_loss",
    "train_loss_raw_last",
    "train_loss_epoch_mean",
    "train_loss_epoch_median",
    "train_loss_smoothed_at_checkpoint",
    "learning_rate_at_checkpoint",
    "grad_norm_at_checkpoint",
    "eval_runtime",
    "eval_samples_per_second",
    "train_runtime",
    "eval_fullchat_pack_macro_loss",
    "eval_fullchat_token_micro_loss",
    "eval_assistant_completion_loss",
    "eval_sql_loss",
    "eval_sql_perplexity",
    "eval_sql_token_accuracy",
    "sqlcc_fullchat_loss",
    "train_others_fullchat_loss",
    "sqlcc_sql_loss",
    "train_others_sql_loss",
    "sqlcc_sql_token_accuracy",
    "train_others_sql_token_accuracy",
    "sql_source_loss_gap",
    "sql_source_accuracy_gap",
    "fullchat_tokens",
    "assistant_completion_tokens",
    "sql_tokens",
    "sqlcc_sql_tokens",
    "train_others_sql_tokens",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    number = coerce_float(value)
    require(number is not None and math.isfinite(number), f"Missing/non-finite {name}")
    if minimum is not None:
        require(number >= minimum, f"{name} must be >= {minimum}: {number}")
    return float(number)


def optional_finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    nonfinite_as_missing: bool = False,
    warnings: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid non-numeric {name}: {value!r}") from None
    if not math.isfinite(number):
        if nonfinite_as_missing:
            if warnings is not None:
                warnings.append(
                    {
                        "metric": name,
                        "context": dict(context or {}),
                        "raw_value": repr(value),
                        "action": "treated_as_missing",
                    }
                )
            return None
        raise RuntimeError(f"Non-finite {name}: {value!r}")
    if minimum is not None:
        require(number >= minimum, f"{name} must be >= {minimum}: {number}")
    return float(number)


def accuracy(value: Any, name: str) -> float:
    number = finite_number(value, name)
    require(0.0 <= number <= 1.0, f"{name} outside [0, 1]: {number}")
    return number


def positive_int(value: Any, name: str) -> int:
    number = coerce_int(value)
    require(number is not None and number > 0, f"{name} must be a positive integer")
    return int(number)


def validate_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    require(match is not None, f"Invalid checkpoint directory name: {path}")
    return int(match.group(1))


def ema(values: Iterable[float], alpha: float) -> list[float]:
    require(0.0 < alpha <= 1.0, "EMA alpha must be in (0, 1]")
    result: list[float] = []
    for value in values:
        result.append(value if not result else alpha * value + (1.0 - alpha) * result[-1])
    return result


def last_at_or_before(points: list[dict[str, Any]], step: int, field: str) -> float | None:
    values = [point[field] for point in points if point["global_step"] <= step and point.get(field) is not None]
    return values[-1] if values else None


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"Temporary path collision: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row.get(column) is None else row.get(column) for column in columns})
    return buffer.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def validate_static_config(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == 1, "Unsupported plot config schema_version")
    require(
        config.get("purpose") in {"qwen35_v2_training_diagnostics", "model_training_diagnostics"},
        "Unexpected plot config purpose",
    )
    require(config.get("overwrite") is False, "overwrite must be false")
    require(config.get("language") in {"de", "en"}, "language must be de or en")
    require(isinstance(config.get("runs"), list) and config["runs"], "Config needs at least one run")
    parent = config["parent_mainline_manifest"]
    parent_path = resolve_path(parent["path"])
    validate_hash(parent_path, parent["sha256"], "parent mainline manifest")
    smoothing = config["smoothing"]
    require(smoothing.get("method") == "ema", "Only transparent EMA smoothing is supported")
    require(smoothing.get("cross_epoch_boundaries") is False, "Smoothing across epoch boundaries is forbidden")
    require(smoothing.get("cross_run_boundaries") is False, "Smoothing across run boundaries is forbidden")
    require(smoothing.get("replace_raw_values") is False, "Smoothed values must not replace raw values")
    require(smoothing.get("extrapolate") is False, "Smoothing extrapolation is forbidden")
    alpha = finite_number(smoothing.get("alpha"), "smoothing alpha")
    require(0.0 < alpha <= 1.0, "smoothing alpha must be in (0, 1]")
    seen_names: set[str] = set()
    summaries = []
    for run in config["runs"]:
        run_name = run.get("run_name")
        require(isinstance(run_name, str) and run_name and run_name not in seen_names, "Duplicate/invalid run_name")
        seen_names.add(run_name)
        training_cfg_path = resolve_path(run["training_config"]["path"])
        validate_hash(training_cfg_path, run["training_config"]["sha256"], f"training config {run_name}")
        training_cfg = load_json(training_cfg_path)
        training_dataset_path = resolve_path(run["training_dataset"]["path"])
        validate_hash(training_dataset_path, run["training_dataset"]["sha256"], f"training dataset {run_name}")
        validation_path = resolve_path(run["validation"]["path"])
        validate_hash(validation_path, run["validation"]["sha256"], f"validation {run_name}")
        require(training_cfg.get("dataset_path") == run["training_dataset"]["path"], f"Wrong training dataset in config {run_name}")
        require(training_cfg.get("eval_dataset_path") == run["validation"]["path"], f"Wrong validation in training config {run_name}")
        require(training_cfg.get("output_dir") == run["adapter_root"], f"Wrong adapter root in training config {run_name}")
        require(training_cfg.get("llm") == run["llm"], f"Wrong llm in training config {run_name}")
        require(training_cfg.get("seed") == run["seed"], f"Wrong seed in training config {run_name}")
        expected = run["hyperparameters"]
        require(training_cfg["lora"]["r"] == expected["lora_r"], f"LoRA r mismatch for {run_name}")
        require(training_cfg["lora"]["lora_alpha"] == expected["lora_alpha"], f"LoRA alpha mismatch for {run_name}")
        require(training_cfg["learning_rate"] == expected["learning_rate"], f"Learning-rate mismatch for {run_name}")
        require(training_cfg.get("completion_only_loss") is False, f"Completion-only loss enabled for {run_name}")
        require(training_cfg.get("assistant_only_loss") is False, f"Assistant-only loss enabled for {run_name}")
        require(training_cfg.get("metric_for_best_model") == "eval_loss", f"Wrong best metric for {run_name}")
        require(training_cfg.get("greater_is_better") is False, f"greater_is_better mismatch for {run_name}")
        posthoc_cfg_path = resolve_path(run["posthoc"]["config_path"])
        validate_hash(posthoc_cfg_path, run["posthoc"]["config_sha256"], f"post-hoc config {run_name}")
        evaluator_path = resolve_path(run["posthoc"]["evaluator_path"])
        validate_hash(evaluator_path, run["posthoc"]["evaluator_sha256"], f"post-hoc evaluator {run_name}")
        summaries.append(
            {
                "run_name": run_name,
                "training_config": project_path(training_cfg_path),
                "validation": project_path(validation_path),
                "adapter_root_exists": resolve_path(run["adapter_root"]).is_dir(),
                "posthoc_result_dir_exists": resolve_path(run["posthoc"]["result_dir"]).is_dir(),
            }
        )
    controlled = config.get("controlled_comparison_requirements")
    if controlled is not None:
        require(isinstance(controlled, dict), "controlled_comparison_requirements must be an object")
        for run in config["runs"]:
            require(run["training_dataset_sha256"] == controlled["training_dataset_sha256"], f"Uncontrolled training dataset: {run['run_name']}")
            require(run["validation"]["sha256"] == controlled["validation_sha256"], f"Uncontrolled validation: {run['run_name']}")
            require(run["seed"] == controlled["seed"], f"Uncontrolled seed: {run['run_name']}")
            require(run["hyperparameters"]["loss"] == controlled["loss"], f"Uncontrolled loss definition: {run['run_name']}")
    return {
        "status": "PASS",
        "config_path": project_path(config_path),
        "config_sha256": sha256_file(config_path),
        "runs": summaries,
        "output_dir": config["output_dir"],
    }


def load_trainer_run(run: dict[str, Any], alpha: float) -> dict[str, Any]:
    run_name = run["run_name"]
    adapter_root = resolve_path(run["adapter_root"])
    require(adapter_root.is_dir(), f"Adapter root not found for {run_name}: {adapter_root}")
    checkpoint_root = adapter_root / "checkpoints"
    require(checkpoint_root.is_dir(), f"Checkpoint directory missing for {run_name}")
    checkpoint_dirs = sorted(
        [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    require(checkpoint_dirs, f"No checkpoints found for {run_name}")
    checkpoint_records: list[dict[str, Any]] = []
    for path in checkpoint_dirs:
        step = checkpoint_step(path)
        state_path = path / "trainer_state.json"
        model_path = path / "adapter_model.safetensors"
        adapter_config_path = path / "adapter_config.json"
        require(state_path.is_file(), f"Missing trainer_state.json: {path}")
        require(model_path.is_file() and model_path.stat().st_size > 0, f"Missing adapter weights: {path}")
        require(adapter_config_path.is_file(), f"Missing adapter_config.json: {path}")
        state = load_json(state_path)
        require(coerce_int(state.get("global_step")) == step, f"Checkpoint name/state mismatch: {path}")
        adapter_config = load_json(adapter_config_path)
        hyper = run["hyperparameters"]
        require(adapter_config.get("base_model_name_or_path") == run["model_id"], f"Wrong base model in {adapter_config_path}")
        require(adapter_config.get("r") == hyper["lora_r"], f"LoRA r mismatch in {adapter_config_path}")
        require(adapter_config.get("lora_alpha") == hyper["lora_alpha"], f"LoRA alpha mismatch in {adapter_config_path}")
        checkpoint_records.append(
            {
                "path": path,
                "step": step,
                "state_path": state_path,
                "adapter_sha256": sha256_file(model_path),
                "adapter_config_sha256": sha256_file(adapter_config_path),
            }
        )
    steps = [record["step"] for record in checkpoint_records]
    require(all(left < right for left, right in zip(steps, steps[1:])), f"Checkpoint steps not strictly increasing: {run_name}")
    latest_state = load_json(checkpoint_records[-1]["state_path"])
    history = latest_state.get("log_history")
    require(isinstance(history, list) and history, f"Empty log_history for {run_name}")
    train_entries = [entry for entry in history if isinstance(entry, dict) and "loss" in entry and "eval_loss" not in entry]
    eval_entries = [entry for entry in history if isinstance(entry, dict) and "eval_loss" in entry]
    require(train_entries and eval_entries, f"Missing train/eval history for {run_name}")
    train_steps = [positive_int(entry.get("step"), "training step") for entry in train_entries]
    eval_steps = [positive_int(entry.get("step"), "eval step") for entry in eval_entries]
    require(all(a < b for a, b in zip(train_steps, train_steps[1:])), f"Training steps not strictly increasing: {run_name}")
    require(all(a < b for a, b in zip(eval_steps, eval_steps[1:])), f"Eval steps not strictly increasing: {run_name}")
    train_epochs = [finite_number(entry.get("epoch"), "training epoch", minimum=0.0) for entry in train_entries]
    eval_epochs = [finite_number(entry.get("epoch"), "eval epoch", minimum=0.0) for entry in eval_entries]
    require(all(a <= b for a, b in zip(train_epochs, train_epochs[1:])), f"Training epochs decrease: {run_name}")
    require(all(a <= b for a, b in zip(eval_epochs, eval_epochs[1:])), f"Eval epochs decrease: {run_name}")
    eval_by_step: dict[int, dict[str, Any]] = {}
    for entry in eval_entries:
        step = positive_int(entry.get("step"), "eval step")
        require(step not in eval_by_step, f"Duplicate eval entry at step {step}: {run_name}")
        finite_number(entry.get("eval_loss"), "eval_loss", minimum=0.0)
        eval_by_step[step] = entry
    require(set(steps) == set(eval_by_step), f"Checkpoint/eval step mismatch for {run_name}: {steps} vs {sorted(eval_by_step)}")
    best_path_raw = latest_state.get("best_model_checkpoint")
    require(isinstance(best_path_raw, str) and best_path_raw, f"Missing best_model_checkpoint for {run_name}")
    best_step = checkpoint_step(Path(best_path_raw))
    require(best_step in eval_by_step, f"Best checkpoint is not retained for {run_name}")
    best_metric = finite_number(latest_state.get("best_metric"), "best_metric", minimum=0.0)
    best_eval = finite_number(eval_by_step[best_step]["eval_loss"], "best eval_loss", minimum=0.0)
    require(math.isclose(best_metric, best_eval, rel_tol=0.0, abs_tol=1e-12), f"best_metric mismatch for {run_name}")
    minimum_eval = min(finite_number(entry["eval_loss"], "eval_loss", minimum=0.0) for entry in eval_entries)
    require(math.isclose(best_eval, minimum_eval, rel_tol=0.0, abs_tol=1e-12), f"Best checkpoint is not minimum eval_loss for {run_name}")
    root_model = adapter_root / "adapter_model.safetensors"
    require(root_model.is_file() and root_model.stat().st_size > 0, f"Root adapter weights missing for {run_name}")
    root_sha = sha256_file(root_model)
    best_record = next(record for record in checkpoint_records if record["step"] == best_step)
    require(root_sha == best_record["adapter_sha256"], f"Root adapter does not match official best checkpoint for {run_name}")

    raw_losses = [finite_number(entry["loss"], "training loss", minimum=0.0) for entry in train_entries]
    smoothed: list[float] = []
    current_epoch_bucket: int | None = None
    epoch_values: list[float] = []
    for entry, loss in zip(train_entries, raw_losses):
        epoch_value = finite_number(entry.get("epoch"), "training epoch", minimum=0.0)
        epoch_bucket = max(1, math.ceil(epoch_value - 1e-12))
        if epoch_bucket != current_epoch_bucket:
            current_epoch_bucket = epoch_bucket
            epoch_values = []
        epoch_values.append(loss)
        smoothed.append(ema(epoch_values, alpha)[-1])
    train_points: list[dict[str, Any]] = []
    optional_metric_warnings: list[dict[str, Any]] = []
    for entry, loss, smooth in zip(train_entries, raw_losses, smoothed):
        train_step = positive_int(entry.get("step"), "training step")
        train_epoch = finite_number(entry.get("epoch"), "training epoch", minimum=0.0)
        train_points.append(
            {
                "model": run["model_label"],
                "run_name": run_name,
                "global_step": train_step,
                "epoch": train_epoch,
                "training_loss_raw": loss,
                "training_loss_smoothed": smooth,
                "learning_rate": optional_finite(entry.get("learning_rate"), "learning_rate", minimum=0.0),
                "grad_norm": optional_finite(
                    entry.get("grad_norm"),
                    "grad_norm",
                    minimum=0.0,
                    nonfinite_as_missing=True,
                    warnings=optional_metric_warnings,
                    context={"run_name": run_name, "log_type": "training", "step": train_step, "epoch": train_epoch},
                ),
            }
        )
    grad_norm_warnings = [warning for warning in optional_metric_warnings if warning["metric"] == "grad_norm"]
    optional_metric_anomalies = {
        "grad_norm": {
            "nonfinite_count": len(grad_norm_warnings),
            "total_training_entries": len(train_entries),
            "affected_steps": sorted({int(warning["context"]["step"]) for warning in grad_norm_warnings}),
            "action": "treated_as_missing_in_derived_diagnostics",
            "source_artifact_modified": False,
        }
    }
    final_train_entry = next((entry for entry in reversed(history) if isinstance(entry, dict) and "train_runtime" in entry), {})
    train_runtime = optional_finite(final_train_entry.get("train_runtime"), "train_runtime", minimum=0.0)
    rows: list[dict[str, Any]] = []
    previous_step = 0
    for record in checkpoint_records:
        step = record["step"]
        eval_entry = eval_by_step[step]
        epoch_losses = [point["training_loss_raw"] for point in train_points if previous_step < point["global_step"] <= step]
        require(epoch_losses, f"No training loss points for checkpoint step {step}: {run_name}")
        row = {column: None for column in WIDE_COLUMNS}
        hyper = run["hyperparameters"]
        row.update(
            {
                "model": run["model_label"],
                "model_id": run["model_id"],
                "model_size": run["model_size"],
                "run_name": run_name,
                "lora_r": hyper["lora_r"],
                "lora_alpha": hyper["lora_alpha"],
                "lora_alpha_over_r": hyper["lora_alpha"] / hyper["lora_r"],
                "learning_rate": hyper["learning_rate"],
                "seed": run["seed"],
                "training_dataset_sha256": run["training_dataset_sha256"],
                "validation_dataset": run["validation"]["label"],
                "validation_sha256": run["validation"]["sha256"],
                "epoch": finite_number(eval_entry.get("epoch"), "eval epoch", minimum=0.0),
                "global_step": step,
                "is_best_checkpoint": step == best_step,
                "is_last_checkpoint": step == steps[-1],
                "official_eval_loss": finite_number(eval_entry["eval_loss"], "eval_loss", minimum=0.0),
                "train_loss_raw_last": last_at_or_before(train_points, step, "training_loss_raw"),
                "train_loss_epoch_mean": statistics.fmean(epoch_losses),
                "train_loss_epoch_median": statistics.median(epoch_losses),
                "train_loss_smoothed_at_checkpoint": last_at_or_before(train_points, step, "training_loss_smoothed"),
                "learning_rate_at_checkpoint": last_at_or_before(train_points, step, "learning_rate"),
                "grad_norm_at_checkpoint": last_at_or_before(train_points, step, "grad_norm"),
                "eval_runtime": optional_finite(eval_entry.get("eval_runtime"), "eval_runtime", minimum=0.0),
                "eval_samples_per_second": optional_finite(eval_entry.get("eval_samples_per_second"), "eval_samples_per_second", minimum=0.0),
                "train_runtime": train_runtime,
                "best_metric": best_metric,
                "best_model_checkpoint": project_path(resolve_path(best_path_raw)),
                "total_flos": optional_finite(latest_state.get("total_flos"), "total_flos", minimum=0.0),
                "checkpoint_path": project_path(record["path"]),
                "checkpoint_adapter_sha256": record["adapter_sha256"],
            }
        )
        rows.append(row)
        previous_step = step
    return {
        "rows": rows,
        "train_points": train_points,
        "checkpoint_records": checkpoint_records,
        "best_step": best_step,
        "last_step": steps[-1],
        "root_adapter_sha256": root_sha,
        "latest_trainer_state_path": project_path(checkpoint_records[-1]["state_path"]),
        "latest_trainer_state_sha256": sha256_file(checkpoint_records[-1]["state_path"]),
        "optional_metric_anomalies": optional_metric_anomalies,
        "optional_metric_warnings": optional_metric_warnings,
    }


def parse_micro_metric(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"Missing metric object: {name}")
    result = {
        "loss": finite_number(value.get("loss"), f"{name}.loss", minimum=0.0),
        "perplexity": finite_number(value.get("perplexity"), f"{name}.perplexity", minimum=1.0),
        "token_accuracy": accuracy(value.get("token_accuracy"), f"{name}.token_accuracy"),
        "tokens": positive_int(value.get("tokens"), f"{name}.tokens"),
    }
    expected = math.exp(result["loss"])
    require(math.isclose(result["perplexity"], expected, rel_tol=1e-9, abs_tol=1e-12), f"Perplexity mismatch in {name}")
    return result


def load_posthoc_results(run: dict[str, Any], trainer: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result_dir = resolve_path(run["posthoc"]["result_dir"])
    if not result_dir.exists():
        return {}
    require(result_dir.is_dir(), f"Post-hoc result path is not a directory: {result_dir}")
    files = sorted(result_dir.glob(run["posthoc"]["result_glob"]))
    by_step: dict[int, dict[str, Any]] = {}
    by_adapter_hash: dict[str, Path] = {}
    checkpoint_by_step = {record["step"]: record for record in trainer["checkpoint_records"]}
    for path in files:
        result = load_json(path)
        require(result.get("status") == "complete", f"Incomplete post-hoc result: {path}")
        require(result.get("diagnostic_only") is True, f"Post-hoc result not marked diagnostic: {path}")
        require(result.get("checkpoint_selection_changed") is False, f"Post-hoc result changed checkpoint selection: {path}")
        require(result.get("generation_performed") is False, f"Generation flag set in post-hoc result: {path}")
        require(result.get("sql_execution_performed") is False, f"SQL execution flag set in post-hoc result: {path}")
        model = result.get("model")
        require(isinstance(model, dict) and model.get("model_id") == run["model_id"], f"Wrong model in {path}")
        require(result.get("evaluator_sha256") == run["posthoc"]["evaluator_sha256"], f"Wrong evaluator hash in {path}")
        require(result.get("config_sha256") == run["posthoc"]["config_sha256"], f"Wrong post-hoc config hash in {path}")
        validation = result.get("validation")
        require(isinstance(validation, dict) and validation.get("validation_sha256") == run["validation"]["sha256"], f"Wrong validation hash in {path}")
        checkpoint = result.get("checkpoint")
        require(isinstance(checkpoint, dict), f"Missing checkpoint metadata in {path}")
        step = positive_int(checkpoint.get("global_step"), "post-hoc global_step")
        require(step in checkpoint_by_step, f"Post-hoc step has no trainer checkpoint: {step}")
        if step in by_step:
            raise RuntimeError(f"Duplicate post-hoc result for step {step}: {path} and {by_step[step]['path']}")
        adapter_hash = checkpoint.get("adapter_model_sha256")
        require(isinstance(adapter_hash, str) and len(adapter_hash) == 64, f"Invalid adapter hash in {path}")
        require(adapter_hash == checkpoint_by_step[step]["adapter_sha256"], f"Adapter hash mismatch at step {step}")
        if adapter_hash in by_adapter_hash:
            raise RuntimeError(f"Duplicate adapter weight result: {path} and {by_adapter_hash[adapter_hash]}")
        by_adapter_hash[adapter_hash] = path
        checkpoint_epoch = finite_number(checkpoint.get("epoch"), "post-hoc epoch", minimum=0.0)
        trainer_epoch = next(row["epoch"] for row in trainer["rows"] if row["global_step"] == step)
        require(math.isclose(checkpoint_epoch, trainer_epoch, rel_tol=0.0, abs_tol=1e-9), f"Epoch mismatch at step {step}")
        metrics = result.get("metrics")
        require(isinstance(metrics, dict), f"Missing metrics in {path}")
        fullchat_micro = parse_micro_metric(metrics.get("eval_full_chat_micro"), "eval_full_chat_micro")
        completion_micro = parse_micro_metric(metrics.get("eval_assistant_completion_micro"), "eval_assistant_completion_micro")
        sql_micro = parse_micro_metric(metrics.get("eval_sql_micro"), "eval_sql_micro")
        fullchat_pack = finite_number(metrics.get("eval_full_chat_pack_macro_loss"), "eval_full_chat_pack_macro_loss", minimum=0.0)
        by_source = metrics.get("by_source_micro")
        require(isinstance(by_source, dict), f"Missing by_source_micro in {path}")
        source_metrics = {}
        for source in ("sqlcc", "train_others"):
            require(isinstance(by_source.get(source), dict), f"Missing source {source} in {path}")
            source_metrics[source] = {
                "full_chat": parse_micro_metric(by_source[source].get("full_chat"), f"{source}.full_chat"),
                "assistant_completion": parse_micro_metric(by_source[source].get("assistant_completion"), f"{source}.assistant_completion"),
                "sql": parse_micro_metric(by_source[source].get("sql"), f"{source}.sql"),
            }
        by_step[step] = {
            "path": path,
            "sha256": sha256_file(path),
            "adapter_sha256": adapter_hash,
            "fullchat_pack": fullchat_pack,
            "fullchat_micro": fullchat_micro,
            "completion_micro": completion_micro,
            "sql_micro": sql_micro,
            "sources": source_metrics,
        }
    if by_step:
        token_tuples = {
            (
                item["fullchat_micro"]["tokens"],
                item["completion_micro"]["tokens"],
                item["sql_micro"]["tokens"],
                item["sources"]["sqlcc"]["sql"]["tokens"],
                item["sources"]["train_others"]["sql"]["tokens"],
            )
            for item in by_step.values()
        }
        require(len(token_tuples) == 1, f"Evaluated token counts vary between checkpoints for {run['run_name']}")
    return by_step


def merge_posthoc(trainer: dict[str, Any], posthoc: dict[int, dict[str, Any]]) -> None:
    for row in trainer["rows"]:
        item = posthoc.get(row["global_step"])
        if item is None:
            continue
        sqlcc = item["sources"]["sqlcc"]
        train_others = item["sources"]["train_others"]
        row.update(
            {
                "eval_fullchat_pack_macro_loss": item["fullchat_pack"],
                "eval_fullchat_token_micro_loss": item["fullchat_micro"]["loss"],
                "eval_assistant_completion_loss": item["completion_micro"]["loss"],
                "eval_sql_loss": item["sql_micro"]["loss"],
                "eval_sql_perplexity": item["sql_micro"]["perplexity"],
                "eval_sql_token_accuracy": item["sql_micro"]["token_accuracy"],
                "sqlcc_fullchat_loss": sqlcc["full_chat"]["loss"],
                "train_others_fullchat_loss": train_others["full_chat"]["loss"],
                "sqlcc_sql_loss": sqlcc["sql"]["loss"],
                "train_others_sql_loss": train_others["sql"]["loss"],
                "sqlcc_sql_token_accuracy": sqlcc["sql"]["token_accuracy"],
                "train_others_sql_token_accuracy": train_others["sql"]["token_accuracy"],
                "sql_source_loss_gap": train_others["sql"]["loss"] - sqlcc["sql"]["loss"],
                "sql_source_accuracy_gap": sqlcc["sql"]["token_accuracy"] - train_others["sql"]["token_accuracy"],
                "fullchat_tokens": item["fullchat_micro"]["tokens"],
                "assistant_completion_tokens": item["completion_micro"]["tokens"],
                "sql_tokens": item["sql_micro"]["tokens"],
                "sqlcc_sql_tokens": sqlcc["sql"]["tokens"],
                "train_others_sql_tokens": train_others["sql"]["tokens"],
                "posthoc_result_path": project_path(item["path"]),
                "posthoc_result_sha256": item["sha256"],
            }
        )


def long_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for wide in wide_rows:
        for metric in LONG_METRICS:
            value = wide.get(metric)
            if value is None:
                continue
            row = {key: wide.get(key) for key in LONG_ID_COLUMNS}
            row.update({"metric": metric, "value": value})
            rows.append(row)
    return rows


def data_dictionary() -> str:
    return """# Training Diagnostics Data Dictionary

All checkpoint rows are joined through model ID, global step, epoch, adapter
SHA256, and validation SHA256. Blank CSV cells and JSON `null` values mean
that a metric was not measured; they are never interpolated.

| Field | Definition |
|---|---|
| `official_eval_loss` | Full-Chat `eval_loss` stored by the Trainer; the only official checkpoint-selection metric. |
| `train_loss_raw_last` | Last logged raw training loss at or before the checkpoint. |
| `train_loss_epoch_mean` | Arithmetic mean of raw training-loss logs since the previous epoch checkpoint. |
| `train_loss_epoch_median` | Median of the same epoch-local raw training-loss logs. |
| `train_loss_smoothed_at_checkpoint` | Last epoch-local EMA value at or before the checkpoint; smoothing resets at every epoch and run boundary. |
| `grad_norm` | Optional Trainer diagnostic. Missing and non-finite values are stored as null/blank, counted in `optional_metric_anomalies`, and never imputed. Negative finite values remain invalid. |
| `eval_fullchat_pack_macro_loss` | Post-hoc unweighted mean of pack-level Full-Chat losses, trainer-compatible at eval batch size 1. |
| `eval_fullchat_token_micro_loss` | Token-weighted Full-Chat causal NLL. |
| `eval_assistant_completion_loss` | Token-weighted Assistant SQL plus end-marker NLL. |
| `eval_sql_loss` | Token-weighted Assistant SQL-only NLL. |
| `eval_sql_perplexity` | `exp(eval_sql_loss)`, verified numerically. |
| `eval_sql_token_accuracy` | SQL next-token accuracy in the interval [0, 1]. |
| `sqlcc_*` / `train_others_*` | Token-weighted source-specific metrics. |
| `sql_source_loss_gap` | `train_others_sql_loss - sqlcc_sql_loss`; descriptive only. |
| `sql_source_accuracy_gap` | `sqlcc_sql_token_accuracy - train_others_sql_token_accuracy`; descriptive only. |
| `is_best_checkpoint` | Exact checkpoint selected by the Trainer's `best_model_checkpoint`; SQL diagnostics cannot change it. |
| `is_last_checkpoint` | Highest retained completed epoch checkpoint. |

The raw training-loss series is stored separately in
`training_loss_points.csv`. The long table contains only measured non-null
values from the wide table.
"""


def aggregate_optional_metric_anomalies(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grad_by_run = []
    for run in run_summaries:
        grad = run["optional_metric_anomalies"]["grad_norm"]
        grad_by_run.append(
            {
                "run_name": run["run_name"],
                "nonfinite_count": grad["nonfinite_count"],
                "total_training_entries": grad["total_training_entries"],
                "affected_steps": grad["affected_steps"],
            }
        )
    return {
        "grad_norm": {
            "nonfinite_count": sum(item["nonfinite_count"] for item in grad_by_run),
            "total_training_entries": sum(item["total_training_entries"] for item in grad_by_run),
            "affected_steps": sorted({step for item in grad_by_run for step in item["affected_steps"]}),
            "action": "treated_as_missing_in_derived_diagnostics",
            "source_artifact_modified": False,
            "by_run": grad_by_run,
        }
    }


def build_outputs(config_path: Path, config: dict[str, Any], *, require_posthoc: bool) -> dict[str, Any]:
    static = validate_static_config(config_path, config)
    alpha = float(config["smoothing"]["alpha"])
    all_wide: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    run_summaries = []
    statuses = []
    input_hashes: dict[str, str] = {project_path(config_path): sha256_file(config_path)}
    for run in config["runs"]:
        trainer = load_trainer_run(run, alpha)
        posthoc = load_posthoc_results(run, trainer)
        merge_posthoc(trainer, posthoc)
        checkpoint_steps = {row["global_step"] for row in trainer["rows"]}
        posthoc_steps = set(posthoc)
        missing = sorted(checkpoint_steps - posthoc_steps)
        if not posthoc_steps:
            status = STATUS_TRAINING_ONLY
        elif missing:
            status = STATUS_INCOMPLETE
        else:
            status = STATUS_COMPLETE
        statuses.append(status)
        if require_posthoc:
            require(status == STATUS_COMPLETE, f"Post-hoc diagnostics incomplete for {run['run_name']}: missing {missing}")
        all_wide.extend(trainer["rows"])
        all_train.extend(trainer["train_points"])
        input_hashes[trainer["latest_trainer_state_path"]] = trainer["latest_trainer_state_sha256"]
        for record in trainer["checkpoint_records"]:
            input_hashes[project_path(record["path"] / "adapter_model.safetensors")] = record["adapter_sha256"]
        for item in posthoc.values():
            input_hashes[project_path(item["path"])] = item["sha256"]
        run_summaries.append(
            {
                "run_name": run["run_name"],
                "status": status,
                "checkpoint_steps": sorted(checkpoint_steps),
                "posthoc_steps": sorted(posthoc_steps),
                "missing_posthoc_steps": missing,
                "best_step": trainer["best_step"],
                "last_step": trainer["last_step"],
                "root_adapter_sha256": trainer["root_adapter_sha256"],
                "optional_metric_anomalies": trainer["optional_metric_anomalies"],
                "optional_metric_warnings": trainer["optional_metric_warnings"],
            }
        )
    all_wide.sort(key=lambda row: (row["run_name"], row["global_step"]))
    all_train.sort(key=lambda row: (row["run_name"], row["global_step"]))
    combined_status = STATUS_COMPLETE if all(status == STATUS_COMPLETE for status in statuses) else (
        STATUS_TRAINING_ONLY if all(status == STATUS_TRAINING_ONLY for status in statuses) else STATUS_INCOMPLETE
    )
    optional_metric_anomalies = aggregate_optional_metric_anomalies(run_summaries)
    optional_metric_warnings = [warning for run in run_summaries for warning in run["optional_metric_warnings"]]
    long = long_rows(all_wide)
    output_dir = resolve_path(config["output_dir"])
    tables_dir = output_dir / "tables"
    manifests_dir = output_dir / "manifests"
    outputs = {
        tables_dir / "training_diagnostics_wide.csv": csv_bytes(all_wide, WIDE_COLUMNS),
        tables_dir / "training_diagnostics_long.csv": csv_bytes(long, LONG_ID_COLUMNS + ["metric", "value"]),
        tables_dir / "training_loss_points.csv": csv_bytes(all_train, TRAIN_POINT_COLUMNS),
        tables_dir / "training_diagnostics.json": json_bytes(
            {
                "status": combined_status,
                "diagnostic_only": True,
                "official_checkpoint_selection_metric": "eval_loss",
                "optional_metric_anomalies": optional_metric_anomalies,
                "optional_metric_warnings": optional_metric_warnings,
                "runs": run_summaries,
                "wide_rows": all_wide,
                "training_loss_points": all_train,
            }
        ),
        tables_dir / "training_diagnostics_data_dictionary.md": data_dictionary().encode("utf-8"),
    }
    for path in outputs:
        require(not path.exists(), f"Refusing to overwrite existing table output: {path}")
    output_hashes = {project_path(path): hashlib.sha256(payload).hexdigest() for path, payload in outputs.items()}
    manifest_path = manifests_dir / "training_diagnostics_manifest.json"
    require(not manifest_path.exists(), f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest = {
        "schema_version": 1,
        "status": combined_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_only": bool(config.get("test_only", False)),
        "config_path": project_path(config_path),
        "config_sha256": sha256_file(config_path),
        "builder_path": project_path(Path(__file__)),
        "builder_sha256": sha256_file(Path(__file__)),
        "smoothing": config["smoothing"],
        "official_checkpoint_selection_metric": "eval_loss",
        "sql_metrics_change_checkpoint_selection": False,
        "optional_metric_anomalies": optional_metric_anomalies,
        "optional_metric_warnings": optional_metric_warnings,
        "runs": run_summaries,
        "input_hashes": dict(sorted(input_hashes.items())),
        "output_hashes": dict(sorted(output_hashes.items())),
        "missing_values_policy": "null_or_blank; optional non-finite grad_norm treated as missing with warning; no interpolation and no synthetic values",
    }
    for path, payload in outputs.items():
        atomic_write_bytes(path, payload)
    atomic_write_bytes(manifest_path, json_bytes(manifest))
    return {
        "status": combined_status,
        "config": static,
        "runs": run_summaries,
        "optional_metric_anomalies": optional_metric_anomalies,
        "optional_metric_warnings": optional_metric_warnings,
        "outputs": [project_path(path) for path in outputs] + [project_path(manifest_path)],
        "message": (
            "Post-hoc SQL diagnostics not available yet. Run the configured post-hoc evaluator first."
            if combined_status == STATUS_TRAINING_ONLY
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate static config and hashes without requiring training outputs.")
    parser.add_argument("--require-posthoc", action="store_true", help="Fail unless every retained checkpoint has diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    require(config_path.is_file(), f"Config missing: {config_path}")
    config = load_json(config_path)
    if args.preflight:
        result = validate_static_config(config_path, config)
        alpha = finite_number(config["smoothing"]["alpha"], "smoothing alpha")
        for run_summary, run in zip(result["runs"], config["runs"]):
            if not run_summary["adapter_root_exists"]:
                run_summary["training_state"] = "NOT_AVAILABLE_YET"
                continue
            trainer = load_trainer_run(run, alpha)
            run_summary.update(
                {
                    "training_state": "PASS",
                    "checkpoint_steps": [record["step"] for record in trainer["checkpoint_records"]],
                    "best_checkpoint_step": trainer["best_step"],
                    "last_checkpoint_step": trainer["last_step"],
                    "root_matches_official_best_checkpoint": True,
                    "optional_metric_anomalies": trainer["optional_metric_anomalies"],
                    "optional_metric_warnings": trainer["optional_metric_warnings"],
                }
            )
        result["training_or_posthoc_started"] = False
    else:
        result = build_outputs(config_path, config, require_posthoc=args.require_posthoc)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
