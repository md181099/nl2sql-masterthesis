#!/usr/bin/env python3
"""TEST_ONLY preflight for the Qwen 3.5 diagnostics table/plot pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv_flash/bin/python3"
BUILDER = PROJECT_ROOT / "src/23_build_qwen35_training_diagnostics_table.py"
PLOTTER = PROJECT_ROOT / "src/24_plot_qwen35_training_diagnostics.py"
EVALUATOR = PROJECT_ROOT / "src/21_eval_qwen35_posthoc_loss_general.py"
PARENT_MANIFEST = PROJECT_ROOT / "audits/qwen35_2b_9b_v2_mixedval_schemaheaderfix_mainline_manifest_20260712.json"
VALIDATION = PROJECT_ROOT / "data/sql_create_context/val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl"
VALIDATION_SHA = "711b23a6dfca40234a33e9aca66506eb33df197f69b6f466fd875854bdb89c08"
OLD25K_SHA = "c4b72a87d175b79895081a83f525997b71a230fd9088a7f8c59c40673fa0a40d"
PARENT_SHA = "c54113b52f63d9a80e4d2626303cbff8fc26f34e341ab640f8a91eb789a136c6"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"Refusing to overwrite fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def write_bytes(path: Path, value: bytes) -> None:
    require(not path.exists(), f"Refusing to overwrite fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def run_command(arguments: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/qwen35_matplotlib_cache"
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)
    if expect_success:
        require(completed.returncode == 0, f"Command failed ({completed.returncode}): {' '.join(arguments)}\n{completed.stdout}\n{completed.stderr}")
    else:
        require(completed.returncode != 0, f"Negative test unexpectedly passed: {' '.join(arguments)}")
    return completed


def micro(loss: float, accuracy: float, tokens: int) -> dict[str, Any]:
    return {"loss": loss, "perplexity": math.exp(loss), "token_accuracy": accuracy, "tokens": tokens}


def history(best_path: Path, global_step: int) -> dict[str, Any]:
    train = [
        {"step": 20, "epoch": 0.2, "loss": 1.20, "learning_rate": 0.0001, "grad_norm": 1.1},
        {"step": 40, "epoch": 0.4, "loss": 0.90, "learning_rate": 0.0001, "grad_norm": 0.8},
        {"step": 60, "epoch": 0.6, "loss": 0.72, "learning_rate": 0.0001, "grad_norm": 0.6},
        {"step": 80, "epoch": 0.8, "loss": 0.61, "learning_rate": 0.0001, "grad_norm": 0.5},
        {"step": 100, "epoch": 1.0, "loss": 0.55, "learning_rate": 0.0001, "grad_norm": 0.45},
        {"step": 120, "epoch": 1.2, "loss": 0.50, "learning_rate": 0.0001, "grad_norm": 0.42},
        {"step": 140, "epoch": 1.4, "loss": 0.47, "learning_rate": 0.0001, "grad_norm": 0.40},
        {"step": 160, "epoch": 1.6, "loss": 0.44, "learning_rate": 0.0001, "grad_norm": 0.38},
        {"step": 180, "epoch": 1.8, "loss": 0.42, "learning_rate": 0.0001, "grad_norm": 0.36},
        {"step": 200, "epoch": 2.0, "loss": 0.40, "learning_rate": 0.0001, "grad_norm": 0.34},
        {"step": 220, "epoch": 2.2, "loss": 0.39, "learning_rate": 0.0001, "grad_norm": 0.35},
        {"step": 240, "epoch": 2.4, "loss": 0.38, "learning_rate": 0.0001, "grad_norm": 0.36},
        {"step": 260, "epoch": 2.6, "loss": 0.37, "learning_rate": 0.0001, "grad_norm": 0.37},
        {"step": 280, "epoch": 2.8, "loss": 0.36, "learning_rate": 0.0001, "grad_norm": 0.38},
        {"step": 300, "epoch": 3.0, "loss": 0.35, "learning_rate": 0.0001, "grad_norm": 0.39},
    ]
    evaluations = [
        {"step": 100, "epoch": 1.0, "eval_loss": 0.50, "eval_runtime": 10.0, "eval_samples_per_second": 44.4},
        {"step": 200, "epoch": 2.0, "eval_loss": 0.45, "eval_runtime": 10.2, "eval_samples_per_second": 43.5},
        {"step": 300, "epoch": 3.0, "eval_loss": 0.47, "eval_runtime": 10.1, "eval_samples_per_second": 44.0},
    ]
    logs: list[dict[str, Any]] = []
    for entry in train:
        logs.append(entry)
        if entry["step"] in {100, 200, 300}:
            logs.append(next(item for item in evaluations if item["step"] == entry["step"]))
    logs.append({"step": 300, "epoch": 3.0, "train_runtime": 123.0, "train_loss": 0.46})
    return {
        "global_step": global_step,
        "epoch": global_step / 100.0,
        "best_global_step": 200,
        "best_metric": 0.45,
        "best_model_checkpoint": str(best_path),
        "max_steps": 500,
        "num_train_epochs": 5,
        "total_flos": 123456789.0,
        "log_history": logs if global_step == 300 else [item for item in logs if int(item.get("step", 0)) <= global_step],
    }


def make_run(root: Path, *, size: str, model_offset: float = 0.0, with_posthoc: bool = True) -> dict[str, Any]:
    lower = size.lower()
    model_id = f"Qwen/Qwen3.5-{size}-Base"
    llm = f"qwen35_{lower}_base"
    run_name = f"TEST_ONLY Qwen 3.5 {size} r8/alpha16 v2"
    model_label = f"TEST_ONLY Qwen 3.5 {size}"
    adapter_root = root / f"adapter_{lower}"
    checkpoint_root = adapter_root / "checkpoints"
    training_dataset = PROJECT_ROOT / (
        f"data/sql_create_context/train_sft_qwen35_{lower}_base_full_chat_v1_clean_anti_overjoin_mix_"
        "spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
    )
    require(sha256_file(training_dataset) == OLD25K_SHA, "old25k fixture reference mismatch")
    adapter_hashes: dict[int, str] = {}
    best_path = checkpoint_root / "checkpoint-200"
    for step in (100, 200, 300):
        checkpoint = checkpoint_root / f"checkpoint-{step}"
        model_bytes = f"TEST_ONLY_{size}_ADAPTER_STEP_{step}\n".encode("ascii")
        model_path = checkpoint / "adapter_model.safetensors"
        write_bytes(model_path, model_bytes)
        adapter_hashes[step] = sha256_file(model_path)
        write_json(
            checkpoint / "adapter_config.json",
            {"base_model_name_or_path": model_id, "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj"]},
        )
        write_json(checkpoint / "trainer_state.json", history(best_path, step))
    write_bytes(adapter_root / "adapter_model.safetensors", (best_path / "adapter_model.safetensors").read_bytes())
    write_json(adapter_root / "adapter_config.json", {"base_model_name_or_path": model_id, "r": 8, "lora_alpha": 16})

    train_config_path = root / f"TEST_ONLY_train_{lower}.json"
    training_config = {
        "llm": llm,
        "dataset_path": str(training_dataset),
        "eval_dataset_path": str(VALIDATION),
        "output_dir": str(adapter_root),
        "seed": 42,
        "learning_rate": 0.0001,
        "completion_only_loss": False,
        "assistant_only_loss": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "lora": {"r": 8, "lora_alpha": 16},
    }
    write_json(train_config_path, training_config)
    posthoc_config_path = root / f"TEST_ONLY_posthoc_{lower}.json"
    write_json(posthoc_config_path, {"test_only": True, "model_id": model_id, "validation_sha256": VALIDATION_SHA})
    result_dir = root / f"posthoc_{lower}"
    if with_posthoc:
        result_dir.mkdir(parents=True)
        sql_losses = {100: 0.30 + model_offset, 200: 0.26 + model_offset, 300: 0.24 + model_offset}
        for step, epoch in ((100, 1.0), (200, 2.0), (300, 3.0)):
            sql_loss = sql_losses[step]
            sqlcc_loss = sql_loss - 0.10
            train_others_loss = sql_loss + 0.18
            sql_accuracy = 0.88 - model_offset - (step - 100) * -0.0001
            metrics = {
                "eval_full_chat_pack_macro_loss": {100: 0.501, 200: 0.451, 300: 0.471}[step] + model_offset,
                "eval_full_chat_micro": micro({100: 0.505, 200: 0.455, 300: 0.475}[step] + model_offset, 0.90 - model_offset, 873241),
                "eval_assistant_completion_micro": micro(sql_loss + 0.03, sql_accuracy - 0.01, 93690),
                "eval_sql_micro": micro(sql_loss, sql_accuracy, 91190),
                "by_source_micro": {
                    "sqlcc": {
                        "full_chat": micro(0.40 + model_offset, 0.93 - model_offset, 500000),
                        "assistant_completion": micro(sqlcc_loss + 0.03, 0.94 - model_offset, 52334),
                        "sql": micro(sqlcc_loss, 0.95 - model_offset, 50534),
                    },
                    "train_others": {
                        "full_chat": micro(0.62 + model_offset, 0.84 - model_offset, 373241),
                        "assistant_completion": micro(train_others_loss + 0.03, 0.80 - model_offset, 41356),
                        "sql": micro(train_others_loss, 0.81 - model_offset, 40656),
                    },
                },
            }
            result = {
                "status": "complete",
                "diagnostic_only": True,
                "checkpoint_selection_changed": False,
                "generation_performed": False,
                "sql_execution_performed": False,
                "model": {"model_id": model_id, "label": f"qwen35_{lower}_v2"},
                "checkpoint": {
                    "path": str(checkpoint_root / f"checkpoint-{step}"),
                    "kind": "checkpoint",
                    "global_step": step,
                    "epoch": epoch,
                    "adapter_model_sha256": adapter_hashes[step],
                    "aliases": [str(checkpoint_root / f"checkpoint-{step}")] + ([str(adapter_root)] if step == 200 else []),
                },
                "validation": {"validation_sha256": VALIDATION_SHA},
                "config_path": str(posthoc_config_path),
                "config_sha256": sha256_file(posthoc_config_path),
                "evaluator_path": str(EVALUATOR),
                "evaluator_sha256": sha256_file(EVALUATOR),
                "metrics": metrics,
            }
            result_path = result_dir / f"TEST_ONLY_qwen35_{lower}_v2_step{step:06d}_adapter-{adapter_hashes[step][:12]}_data-{VALIDATION_SHA[:12]}.json"
            write_json(result_path, result)

    return {
        "run_name": run_name,
        "model_label": model_label,
        "model_size": size,
        "model_id": model_id,
        "llm": llm,
        "seed": 42,
        "adapter_root": str(adapter_root),
        "training_config": {"path": str(train_config_path), "sha256": sha256_file(train_config_path)},
        "training_dataset": {"label": "old25k", "path": str(training_dataset), "sha256": OLD25K_SHA},
        "training_dataset_sha256": OLD25K_SHA,
        "validation": {"label": "MixedVal2500-v2 schemaheaderfix", "path": str(VALIDATION), "sha256": VALIDATION_SHA},
        "posthoc": {
            "result_dir": str(result_dir),
            "result_glob": f"TEST_ONLY_qwen35_{lower}_v2_step*_adapter-*_data-{VALIDATION_SHA[:12]}.json",
            "config_path": str(posthoc_config_path),
            "config_sha256": sha256_file(posthoc_config_path),
            "evaluator_path": str(EVALUATOR),
            "evaluator_sha256": sha256_file(EVALUATOR),
        },
        "hyperparameters": {"lora_r": 8, "lora_alpha": 16, "learning_rate": 0.0001, "max_epochs": 5, "loss": "full_chat"},
    }


def make_plot_config(path: Path, runs: list[dict[str, Any]], output_dir: Path, *, mode: str = "single") -> Path:
    config = {
        "schema_version": 1,
        "purpose": "qwen35_v2_training_diagnostics",
        "mode": mode,
        "test_only": True,
        "language": "de",
        "image_formats": ["png", "pdf", "svg"],
        "dpi": 300,
        "overwrite": False,
        "best_checkpoint_marker": "trainer_state.best_model_checkpoint",
        "smoothing": {"method": "ema", "alpha": 0.15, "replace_raw_values": False, "cross_epoch_boundaries": False, "cross_run_boundaries": False, "extrapolate": False},
        "output_dir": str(output_dir),
        "parent_mainline_manifest": {"path": str(PARENT_MANIFEST), "sha256": PARENT_SHA},
        "runs": runs,
    }
    if mode == "comparison":
        config["controlled_comparison_requirements"] = {
            "training_dataset_sha256": OLD25K_SHA,
            "validation_sha256": VALIDATION_SHA,
            "seed": 42,
            "loss": "full_chat",
        }
    write_json(path, config)
    return path


def load_wide(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "tables/training_diagnostics_wide.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_hashes(output_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted((output_dir / "plots").glob("*/*")):
        result[str(path.relative_to(output_dir / "plots"))] = sha256_file(path)
    return result


def copy_json_with_mutation(source: Path, target: Path, mutate: Any) -> None:
    value = json.loads(source.read_text(encoding="utf-8"))
    mutate(value)
    write_json(target, value)


def mutate_latest_trainer_state(run: dict[str, Any], mutate: Any) -> None:
    checkpoint_root = Path(run["adapter_root"]) / "checkpoints"
    latest = max(checkpoint_root.glob("checkpoint-*/trainer_state.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    state = json.loads(latest.read_text(encoding="utf-8"))
    mutate(state)
    latest.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def training_log_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in state["log_history"] if "loss" in entry and "eval_loss" not in entry]


def load_training_points(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "tables/training_loss_points.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_plot_manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "manifests/plot_manifest.json").read_text(encoding="utf-8"))


def load_table_manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "manifests/training_diagnostics_manifest.json").read_text(encoding="utf-8"))


def run_suite(root: Path) -> dict[str, Any]:
    require(PYTHON.is_file(), f"Missing interpreter: {PYTHON}")
    require(sha256_file(PARENT_MANIFEST) == PARENT_SHA, "Parent mainline manifest changed")
    require(sha256_file(VALIDATION) == VALIDATION_SHA, "MixedVal-v2 changed")
    fixture_root = root / "fixtures"
    output_root = root / "outputs"
    run2 = make_run(fixture_root / "complete", size="2B", model_offset=0.0, with_posthoc=True)
    run9 = make_run(fixture_root / "complete", size="9B", model_offset=-0.03, with_posthoc=True)

    config2 = make_plot_config(fixture_root / "TEST_ONLY_plot_2b.json", [run2], output_root / "complete_2b")
    run_command([str(PYTHON), str(BUILDER), "--config", str(config2), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(config2)])
    rows = load_wide(output_root / "complete_2b")
    require(len(rows) == 3, "Expected three checkpoint rows")
    best = [row for row in rows if row["is_best_checkpoint"] == "True"]
    require(len(best) == 1 and best[0]["global_step"] == "200", "Official best checkpoint marking failed")
    require(min(rows, key=lambda row: float(row["eval_sql_loss"]))["global_step"] == "300", "Fixture must make SQL-best differ from official best")
    require(math.isclose(float(rows[0]["sql_source_loss_gap"]), 0.28, abs_tol=1e-12), "Loss gap formula failed")
    require(math.isclose(float(rows[0]["sql_source_accuracy_gap"]), 0.14, abs_tol=1e-12), "Accuracy gap formula failed")

    config_compare = make_plot_config(fixture_root / "TEST_ONLY_plot_comparison.json", [run2, run9], output_root / "comparison", mode="comparison")
    run_command([str(PYTHON), str(BUILDER), "--config", str(config_compare), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(config_compare)])

    training_only_run = copy.deepcopy(run2)
    training_only_run["posthoc"]["result_dir"] = str(fixture_root / "missing_posthoc")
    config_training_only = make_plot_config(fixture_root / "TEST_ONLY_plot_training_only.json", [training_only_run], output_root / "training_only")
    build_training_only = run_command([str(PYTHON), str(BUILDER), "--config", str(config_training_only)])
    require("Post-hoc SQL diagnostics not available yet" in build_training_only.stdout, "Missing training-only message")
    run_command([str(PYTHON), str(PLOTTER), "--config", str(config_training_only)])
    training_only_plots = plot_hashes(output_root / "training_only")
    require(set(training_only_plots) == {f"{fmt}/01_training_and_fullchat_validation_loss.{fmt}" for fmt in ("png", "pdf", "svg")}, "Training-only mode produced SQL plots")

    # Determinism: same fixture values, separate paths and output directory.
    deterministic_root = root / "deterministic"
    deterministic_run = make_run(deterministic_root, size="2B", model_offset=0.0, with_posthoc=True)
    deterministic_config = make_plot_config(deterministic_root / "TEST_ONLY_plot_2b_repeat.json", [deterministic_run], output_root / "repeat_2b")
    run_command([str(PYTHON), str(BUILDER), "--config", str(deterministic_config), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(deterministic_config)])
    require(plot_hashes(output_root / "complete_2b") == plot_hashes(output_root / "repeat_2b"), "Plot hashes are not deterministic")

    # Overwrite protection.
    run_command([str(PYTHON), str(BUILDER), "--config", str(config2), "--require-posthoc"], expect_success=False)
    run_command([str(PYTHON), str(PLOTTER), "--config", str(config2)], expect_success=False)

    # Partial diagnostics: explicit INCOMPLETE without require, failure with require.
    partial_root = root / "partial"
    partial_run = make_run(partial_root, size="2B", with_posthoc=True)
    partial_files = sorted(Path(partial_run["posthoc"]["result_dir"]).glob("*.json"))
    partial_files[-1].unlink()
    partial_config = make_plot_config(partial_root / "TEST_ONLY_plot_partial.json", [partial_run], output_root / "partial")
    partial_result = run_command([str(PYTHON), str(BUILDER), "--config", str(partial_config)])
    require('"status": "INCOMPLETE"' in partial_result.stdout, "Partial post-hoc status not INCOMPLETE")
    partial_fail_config = make_plot_config(partial_root / "TEST_ONLY_plot_partial_fail.json", [partial_run], output_root / "partial_fail")
    run_command([str(PYTHON), str(BUILDER), "--config", str(partial_fail_config), "--require-posthoc"], expect_success=False)

    # Optional trainer metrics are validated by log type. Evaluation and
    # completion records need no grad_norm; missing training values stay blank.
    optional_tests: dict[str, str] = {
        "evaluation_without_grad_norm": "PASS",
        "completion_without_grad_norm": "PASS",
    }
    missing_root = root / "optional_missing_grad_norm"
    missing_run = make_run(missing_root, size="2B", with_posthoc=True)
    mutate_latest_trainer_state(missing_run, lambda state: training_log_entries(state)[0].pop("grad_norm"))
    missing_config = make_plot_config(missing_root / "TEST_ONLY_plot.json", [missing_run], output_root / "optional_missing_grad_norm")
    run_command([str(PYTHON), str(BUILDER), "--config", str(missing_config), "--require-posthoc"])
    missing_points = load_training_points(output_root / "optional_missing_grad_norm")
    require(missing_points[0]["grad_norm"] == "", "Missing grad_norm was not preserved as a blank cell")
    optional_tests["training_without_grad_norm"] = "PASS_BLANK"

    empty_root = root / "optional_empty_grad_norm"
    empty_run = make_run(empty_root, size="2B", with_posthoc=True)
    mutate_latest_trainer_state(empty_run, lambda state: training_log_entries(state)[0].update({"grad_norm": ""}))
    empty_config = make_plot_config(empty_root / "TEST_ONLY_plot.json", [empty_run], output_root / "optional_empty_grad_norm")
    run_command([str(PYTHON), str(BUILDER), "--config", str(empty_config), "--require-posthoc"])
    require(load_training_points(output_root / "optional_empty_grad_norm")[0]["grad_norm"] == "", "Empty grad_norm was not preserved as blank")
    optional_tests["empty_grad_norm"] = "PASS_BLANK"

    valid_root = root / "optional_valid_grad_norm"
    valid_run = make_run(valid_root, size="2B", with_posthoc=True)
    mutate_latest_trainer_state(valid_run, lambda state: training_log_entries(state)[0].update({"grad_norm": "0.1376"}))
    valid_config = make_plot_config(valid_root / "TEST_ONLY_plot.json", [valid_run], output_root / "optional_valid_grad_norm")
    run_command([str(PYTHON), str(BUILDER), "--config", str(valid_config), "--require-posthoc"])
    require(math.isclose(float(load_training_points(output_root / "optional_valid_grad_norm")[0]["grad_norm"]), 0.1376), "Valid string grad_norm was not normalized")
    optional_tests["valid_grad_norm"] = "PASS_NORMALIZED"

    no_grad_root = root / "optional_no_grad_norms"
    no_grad_run = make_run(no_grad_root, size="2B", with_posthoc=True)
    mutate_latest_trainer_state(no_grad_run, lambda state: [entry.update({"grad_norm": float("nan")}) for entry in training_log_entries(state)])
    no_grad_config = make_plot_config(no_grad_root / "TEST_ONLY_plot.json", [no_grad_run], output_root / "optional_no_grad_norms")
    run_command([str(PYTHON), str(BUILDER), "--config", str(no_grad_config), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(no_grad_config)])
    no_grad_manifest = load_plot_manifest(output_root / "optional_no_grad_norms")
    require("07_gradient_norm_over_training" in no_grad_manifest["skipped"], "Gradient plot was not skipped without valid data")
    require(not any("07_gradient_norm_over_training" in name for name in plot_hashes(output_root / "optional_no_grad_norms")), "Gradient plot exists without valid data")
    no_grad_anomaly = load_table_manifest(output_root / "optional_no_grad_norms")["optional_metric_anomalies"]["grad_norm"]
    require(no_grad_anomaly["nonfinite_count"] == 15 and no_grad_anomaly["affected_steps"] == list(range(20, 301, 20)), "All-NaN anomaly summary mismatch")
    optional_tests["no_gradient_norm_plot"] = "PASS_WARNED_AND_SKIPPED"

    mixed_grad_root = root / "optional_mixed_grad_norms"
    mixed_grad_run = make_run(mixed_grad_root, size="2B", with_posthoc=True)
    mutate_latest_trainer_state(mixed_grad_run, lambda state: training_log_entries(state)[5].update({"grad_norm": float("nan")}))
    mixed_grad_config = make_plot_config(mixed_grad_root / "TEST_ONLY_plot.json", [mixed_grad_run], output_root / "optional_mixed_grad_norms")
    run_command([str(PYTHON), str(BUILDER), "--config", str(mixed_grad_config), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(mixed_grad_config)])
    mixed_points = load_training_points(output_root / "optional_mixed_grad_norms")
    require(sum(bool(row["grad_norm"]) for row in mixed_points) == 14, "Mixed grad_norm fixture did not retain exactly fourteen real values")
    require(mixed_points[5]["grad_norm"] == "", "Non-finite grad_norm was not stored as blank")
    mixed_anomaly = load_table_manifest(output_root / "optional_mixed_grad_norms")["optional_metric_anomalies"]["grad_norm"]
    require(mixed_anomaly["nonfinite_count"] == 1 and mixed_anomaly["affected_steps"] == [120], "Mixed anomaly summary mismatch")
    mixed_master = json.loads((output_root / "optional_mixed_grad_norms/tables/training_diagnostics.json").read_text(encoding="utf-8"))
    require(mixed_master["optional_metric_anomalies"]["grad_norm"]["affected_steps"] == [120], "Master-table anomaly summary missing")
    require(mixed_master["optional_metric_warnings"][0]["action"] == "treated_as_missing", "Master-table anomaly warning missing")
    require(any("07_gradient_norm_over_training" in name for name in plot_hashes(output_root / "optional_mixed_grad_norms")), "Gradient plot missing for real data")
    mixed_caption = (output_root / "optional_mixed_grad_norms/captions/plot_captions.md").read_text(encoding="utf-8")
    require("nicht-endliche optionale Gradientennorm-Messpunkte" in mixed_caption, "Optional anomaly caption missing")
    optional_tests["mixed_gradient_norm_plot"] = "PASS_WARNING_GAP_AND_REAL_POINTS_ONLY"

    for name, nonfinite_value in (("nan_grad_norm", float("nan")), ("inf_grad_norm", float("inf"))):
        warning_root = root / f"optional_{name}"
        warning_run = make_run(warning_root, size="2B", with_posthoc=True)
        mutate_latest_trainer_state(warning_run, lambda state, value=nonfinite_value: training_log_entries(state)[0].update({"grad_norm": value}))
        warning_config = make_plot_config(warning_root / "TEST_ONLY_plot.json", [warning_run], output_root / f"optional_{name}")
        run_command([str(PYTHON), str(BUILDER), "--config", str(warning_config), "--require-posthoc"])
        require(load_training_points(output_root / f"optional_{name}")[0]["grad_norm"] == "", f"{name} was not stored as blank")
        anomaly = load_table_manifest(output_root / f"optional_{name}")["optional_metric_anomalies"]["grad_norm"]
        require(anomaly["nonfinite_count"] == 1 and anomaly["affected_steps"] == [20], f"{name} anomaly summary mismatch")
        optional_tests[name] = "PASS_WARNING_AND_NULL"

    strict_cases = {
        "negative_grad_norm": lambda state: training_log_entries(state)[0].update({"grad_norm": -0.1}),
        "nan_training_loss": lambda state: training_log_entries(state)[0].update({"loss": float("nan")}),
        "nan_eval_loss": lambda state: next(entry for entry in state["log_history"] if "eval_loss" in entry).update({"eval_loss": float("nan")}),
        "nan_learning_rate": lambda state: training_log_entries(state)[0].update({"learning_rate": float("nan")}),
    }
    for name, mutation in strict_cases.items():
        strict_root = root / f"strict_{name}"
        strict_run = make_run(strict_root, size="2B", with_posthoc=True)
        mutate_latest_trainer_state(strict_run, mutation)
        strict_config = make_plot_config(strict_root / "TEST_ONLY_plot.json", [strict_run], output_root / f"strict_{name}")
        run_command([str(PYTHON), str(BUILDER), "--config", str(strict_config), "--require-posthoc"], expect_success=False)
        optional_tests[name] = "PASS_REJECTED"

    negative_tests = {}
    mutations = {
        "wrong_dataset_hash": lambda value: value["validation"].update({"validation_sha256": "0" * 64}),
        "wrong_model": lambda value: value["model"].update({"model_id": "Qwen/WRONG"}),
        "perplexity_mismatch": lambda value: value["metrics"]["eval_sql_micro"].update({"perplexity": 999.0}),
        "nan_loss": lambda value: value["metrics"]["eval_sql_micro"].update({"loss": float("nan")}),
    }
    for name, mutation in mutations.items():
        case_root = root / f"negative_{name}"
        case_run = make_run(case_root, size="2B", with_posthoc=True)
        first = sorted(Path(case_run["posthoc"]["result_dir"]).glob("*.json"))[0]
        temporary = first.with_suffix(".mutated")
        copy_json_with_mutation(first, temporary, mutation)
        first.unlink()
        temporary.rename(first)
        case_config = make_plot_config(case_root / "TEST_ONLY_plot.json", [case_run], output_root / f"negative_{name}")
        run_command([str(PYTHON), str(BUILDER), "--config", str(case_config)], expect_success=False)
        negative_tests[name] = "PASS_REJECTED"

    duplicate_root = root / "negative_duplicate"
    duplicate_run = make_run(duplicate_root, size="2B", with_posthoc=True)
    duplicate_files = sorted(Path(duplicate_run["posthoc"]["result_dir"]).glob("*.json"))
    duplicate_name = duplicate_files[0].name.replace("_data-", "_duplicate_data-")
    shutil.copyfile(duplicate_files[0], duplicate_files[0].with_name(duplicate_name))
    duplicate_config = make_plot_config(duplicate_root / "TEST_ONLY_plot.json", [duplicate_run], output_root / "negative_duplicate")
    run_command([str(PYTHON), str(BUILDER), "--config", str(duplicate_config)], expect_success=False)
    negative_tests["duplicate_posthoc"] = "PASS_REJECTED"

    nonmonotone_root = root / "negative_nonmonotone"
    nonmonotone_run = make_run(nonmonotone_root, size="2B", with_posthoc=True)
    latest_state = Path(nonmonotone_run["adapter_root"]) / "checkpoints/checkpoint-300/trainer_state.json"
    state = json.loads(latest_state.read_text())
    train_entries = [item for item in state["log_history"] if "loss" in item and "eval_loss" not in item]
    train_entries[3]["step"] = 10
    latest_state.write_text(json.dumps(state, indent=2) + "\n")
    nonmonotone_config = make_plot_config(nonmonotone_root / "TEST_ONLY_plot.json", [nonmonotone_run], output_root / "negative_nonmonotone")
    run_command([str(PYTHON), str(BUILDER), "--config", str(nonmonotone_config)], expect_success=False)
    negative_tests["nonmonotone_steps"] = "PASS_REJECTED"

    return {
        "status": "PASS",
        "complete_2b_plot_hashes": plot_hashes(output_root / "complete_2b"),
        "comparison_plot_hashes": plot_hashes(output_root / "comparison"),
        "training_only_plot_hashes": training_only_plots,
        "negative_tests": negative_tests,
        "root_best_deduplication": "PASS",
        "official_best_ignores_sql_best": "PASS",
        "partial_posthoc": "INCOMPLETE_AND_REQUIRE_REJECTED",
        "optional_trainer_metrics": optional_tests,
        "overwrite_protection": "PASS_REJECTED",
        "deterministic_repeat": "PASS_IDENTICAL_PLOT_HASHES",
        "png_pdf_svg": "PASS",
    }


def materialize_audit_artifacts() -> dict[str, Any]:
    fixture_target = PROJECT_ROOT / "audits/fixtures/qwen35_v2_training_diagnostics_epoch_reset"
    output_target = PROJECT_ROOT / "audits/test_outputs/qwen35_v2_training_diagnostics_epoch_reset"
    require(not fixture_target.exists(), f"Fixture target already exists: {fixture_target}")
    require(not output_target.exists(), f"Test output target already exists: {output_target}")
    run2 = make_run(fixture_target / "complete", size="2B", model_offset=0.0, with_posthoc=True)
    run9 = make_run(fixture_target / "complete", size="9B", model_offset=-0.03, with_posthoc=True)
    complete_config = make_plot_config(fixture_target / "TEST_ONLY_plot_2b.json", [run2], output_target / "complete_2b")
    comparison_config = make_plot_config(fixture_target / "TEST_ONLY_plot_comparison.json", [run2, run9], output_target / "comparison", mode="comparison")
    training_only_run = copy.deepcopy(run2)
    training_only_run["posthoc"]["result_dir"] = str(fixture_target / "missing_posthoc")
    training_only_config = make_plot_config(fixture_target / "TEST_ONLY_plot_training_only.json", [training_only_run], output_target / "training_only")
    run_command([str(PYTHON), str(BUILDER), "--config", str(complete_config), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(complete_config)])
    run_command([str(PYTHON), str(BUILDER), "--config", str(comparison_config), "--require-posthoc"])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(comparison_config)])
    run_command([str(PYTHON), str(BUILDER), "--config", str(training_only_config)])
    run_command([str(PYTHON), str(PLOTTER), "--config", str(training_only_config)])
    hashes = {}
    for base in (fixture_target, output_target):
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            hashes[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return {
        "fixtures": str(fixture_target.relative_to(PROJECT_ROOT)),
        "test_outputs": str(output_target.relative_to(PROJECT_ROOT)),
        "artifact_hashes": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-audit-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="qwen35-diagnostics-preflight-", dir="/tmp") as temporary:
        result = run_suite(Path(temporary))
    if args.write_audit_artifacts:
        result["materialized"] = materialize_audit_artifacts()
    else:
        result["materialized"] = None
    result.update(
        {
            "training_started": False,
            "posthoc_evaluation_started": False,
            "generative_evaluation_started": False,
            "model_loaded": False,
            "adapter_loaded": False,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
