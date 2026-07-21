#!/usr/bin/env python3
"""Read-only synthesis of the final Qwen 2B, Llama 3B, and Qwen 9B 8x8 runs.

The script never imports model, adapter, tokenizer, or retriever libraries. It reads
the frozen run artifacts, performs statistical and read-only SQLite checks, and
writes only the additive audit artifacts dated 20260716.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260716
N = 1032
BOOTSTRAP_RESAMPLES = 10_000

CONDITIONS = [
    "zero_shot",
    "top1",
    "top1_gate070",
    "top1_gate085",
    "static_seed42",
    "structure",
    "structure_gate070",
    "structure_gate085",
]
LABELS = {
    "zero_shot": "Zero Shot",
    "top1": "Dynamic Top-1",
    "top1_gate070": "Top-1 Gate 0.70",
    "top1_gate085": "Top-1 Gate 0.85",
    "static_seed42": "Static k=1 Seed 42",
    "structure": "Structure Top-10 v2",
    "structure_gate070": "Structure Gate 0.70",
    "structure_gate085": "Structure Gate 0.85",
}
MODEL_ORDER = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B",
    "llama3b": "Llama 3.2 3B Instruct",
    "qwen9b": "Qwen 3.5 9B",
}
ROLE_LABELS = {"base": "Ausgangsmodell", "lora_v2": "LoRA v2"}

SOURCES = {
    "qwen2b_audit": (
        "audits/audit_qwen35_2b_complete_8x8_base_and_lora_v2_evaluations_20260715.md",
        "eeda6770719dae4ba89da5bbed77778f28b2e24b7844e4fba6d8f9fba9462410",
    ),
    "qwen2b_manifest": (
        "audits/qwen35_2b_complete_8x8_base_and_lora_v2_manifest_20260715.json",
        "60a5b2bd75fd1851c5528688e4109d124e8042b1e5044d5300e00de2eca58743",
    ),
    "qwen2b_sensitivity_audit": (
        "audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260716.md",
        "94511610d0433e1e43cdb26ce0da22531530eb7da7a0e48ad3a7c0c22457fe1e",
    ),
    "qwen2b_sensitivity_manifest": (
        "audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json",
        "e4ca268c6c5d08733bcd22268c75fb41386907b2b915753bf85f0e0fe3222853",
    ),
    "llama3b_audit": (
        "audits/audit_llama32_3b_instruct_lora_v2_evaluations_results_base_comparison_and_qwen_context_20260715.md",
        "06f5b89c4fae5be6899ff413953cc4ad1835a14b8695fc6ed0b8b0087fe8223a",
    ),
    "llama3b_manifest": (
        "audits/llama32_3b_instruct_lora_v2_evaluations_manifest_20260715.json",
        "a11c947f8ba31597364b243aeb7233ee06c6205ddd31b196d7c2fa23fa9bb9d3",
    ),
    "qwen9b_audit": (
        "audits/audit_qwen35_9b_complete_8x8_base_and_lora_v2_evaluations_20260716.md",
        "3691a0983b80267ae6bf373b3a8658f18a5d8b8d53ac8ac1dd2b1325f2052639",
    ),
    "qwen9b_manifest": (
        "audits/qwen35_9b_complete_8x8_base_and_lora_v2_manifest_20260716.json",
        "6a9483b0518060f05fbaed8f1dd40c14088e53b26eb215646afa92b75fac3172",
    ),
}

SUPPORT_SOURCES = {
    "qwen2b_summary": "audits/derived/qwen35_2b_complete_8x8_evaluation_summary_20260715.json",
    "llama3b_base_summary": "audits/derived/llama32_3b_base_evaluation_summary_20260714.json",
    "llama3b_summary": "audits/derived/llama32_3b_instruct_lora_v2_evaluation_summary_20260715.json",
    "qwen9b_summary": "audits/derived/qwen35_9b_complete_8x8_evaluation_summary_20260716.json",
}

MODEL_INFO = {
    "qwen2b": {
        "registry": "qwen35_2b_base",
        "model_id": "Qwen/Qwen3.5-2B-Base",
        "model_type": "Base",
        "snapshot": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
        "parameter_class_b": 2,
        "adapter_root": "adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "best_checkpoint": "checkpoint-502",
        "best_epoch": 1,
        "adapter_sha256": "6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3",
        "prompt_format": "qwen_sqlctx_chatml",
        "comparison_class": "A within line; B+ versus Qwen 9B; B versus Llama",
    },
    "llama3b": {
        "registry": "llama32_3b_instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_type": "Instruct",
        "snapshot": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "parameter_class_b": 3,
        "adapter_root": "adapters/llama32_3b_instruct/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "best_checkpoint": "checkpoint-509",
        "best_epoch": 1,
        "adapter_sha256": "fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d",
        "prompt_format": "llama32_instruct_native_chat",
        "comparison_class": "A within line; B versus Qwen",
    },
    "qwen9b": {
        "registry": "qwen35_9b_base",
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "model_type": "Base",
        "snapshot": "68c46c4b3498877f3ef123c856ecfde50c39f404",
        "parameter_class_b": 9,
        "adapter_root": "adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5",
        "best_checkpoint": "checkpoint-502",
        "best_epoch": 1,
        "adapter_sha256": "dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0",
        "prompt_format": "qwen_sqlctx_chatml",
        "comparison_class": "A within line; B+ versus Qwen 2B; B versus Llama",
    },
}

OUT = {
    "audit": ROOT / "audits/audit_cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_20260716.md",
    "manifest": ROOT / "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json",
    "results": ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv",
    "zero": ROOT / "audits/derived/cross_model_zero_shot_comparison_20260716.csv",
    "gains": ROOT / "audits/derived/cross_model_lora_gain_by_condition_20260716.csv",
    "fewshot": ROOT / "audits/derived/cross_model_fewshot_effects_20260716.csv",
    "heterogeneity": ROOT / "audits/derived/cross_model_lora_gain_heterogeneity_20260716.csv",
    "completion": ROOT / "audits/derived/cross_model_completion_diagnostics_20260716.csv",
    "comparability": ROOT / "audits/derived/cross_model_comparability_and_limitations_20260716.csv",
    "thesis": ROOT / "audits/derived/cross_model_thesis_ready_tables_20260716.md",
}
PLOTS = {
    "zero": ROOT / "audits/plots/cross_model_zero_shot_ema_20260716",
    "gains": ROOT / "audits/plots/cross_model_lora_gain_by_condition_20260716",
    "fewshot": ROOT / "audits/plots/cross_model_fewshot_effects_20260716",
    "completion": ROOT / "audits/plots/cross_model_completion_limits_20260716",
    "overview": ROOT / "audits/plots/cross_model_performance_overview_20260716",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true"})


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def exact_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "mcnemar_p") -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row[p_key]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["significant_unadjusted_0_05"] = float(row[p_key]) < 0.05
        rows[index]["significant_holm_0_05"] = running < 0.05


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 250
    offset = 0
    while offset < BOOTSTRAP_RESAMPLES:
        size = min(chunk, BOOTSTRAP_RESAMPLES - offset)
        indices = rng.integers(0, len(diff), size=(size, len(diff)))
        values[offset : offset + size] = diff[indices].mean(axis=1)
        offset += size
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    ci_low, ci_high = bootstrap_ci(b.astype(float) - a.astype(float), rng)
    a_ema, b_ema = float(a.mean()), float(b.mean())
    return {
        "n": len(a),
        "a_correct": int(a.sum()),
        "b_correct": int(b.sum()),
        "a_ema": a_ema,
        "b_ema": b_ema,
        "delta": b_ema - a_ema,
        "delta_pp": 100 * (b_ema - a_ema),
        "relative_error_reduction": (b_ema - a_ema) / (1 - a_ema) if a_ema < 1 else None,
        "both_correct": int(np.sum((a == 1) & (b == 1))),
        "both_wrong": int(np.sum((a == 0) & (b == 0))),
        "n01_a_wrong_b_correct": n01,
        "n10_a_correct_b_wrong": n10,
        "mcnemar_p": exact_mcnemar_p(n01, n10),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SEED,
    }


def metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    floats = lambda key: [float(row[key]) for row in rows]
    correct = sum(as_bool(row["exec_match"]) for row in rows)
    executable = sum(as_bool(row["pred_ok"]) for row in rows)
    prompt_tokens = [int(float(row["prompt_tokens"])) for row in rows]
    completion_tokens = [int(float(row["completion_tokens"])) for row in rows]
    limit = int(rows[0]["run_max_new_tokens"])
    max_input = int(rows[0]["run_max_input_tokens"])
    return {
        "correct": correct,
        "ema": correct / len(rows),
        "executable": executable,
        "esr": executable / len(rows),
        "string_exact": statistics.fmean(floats("string_exact")),
        "normalized_exact": statistics.fmean(floats("normalized_exact")),
        "char_accuracy": statistics.fmean(floats("char_accuracy")),
        "token_accuracy": statistics.fmean(floats("token_accuracy")),
        "prompt_tokens_mean": statistics.fmean(prompt_tokens),
        "prompt_tokens_max": max(prompt_tokens),
        "max_input_tokens": max_input,
        "prompt_truncations": sum(value >= max_input for value in prompt_tokens),
        "completion_tokens_mean": statistics.fmean(completion_tokens),
        "completion_tokens_max": max(completion_tokens),
        "completion_limit_cases": sum(value == limit for value in completion_tokens),
        "empty_raw_output": sum(not row["raw_output"].strip() for row in rows),
        "empty_extracted_sql": sum(not row["pred_sql"].strip() for row in rows),
        "think_marker": sum("<think>" in row["raw_output"].lower() for row in rows),
        "max_new_tokens": limit,
    }


def standard_record(
    model: str,
    role: str,
    condition: str,
    source: dict[str, Any],
    *,
    config_key: str = "config_path",
) -> dict[str, Any]:
    trace_path = source.get("trace_path")
    if trace_path is None and isinstance(source.get("trace_summary"), dict):
        trace_path = source["trace_summary"].get("path")
    config_path = source.get(config_key) or source.get("physical_config_path") or source.get("recorded_config_path")
    return {
        "model": model,
        "role": role,
        "condition": condition,
        "run_id": source["run_id"],
        "csv_path": source["csv_path"],
        "csv_expected_sha256": source["csv_sha256"],
        "metadata_path": source["metadata_path"],
        "metadata_expected_sha256": source.get("metadata_sha256"),
        "trace_path": trace_path,
        "trace_expected_sha256": source.get("trace_sha256")
        or (source.get("trace_summary") or {}).get("sha256"),
        "log_path": source.get("log_path"),
        "log_expected_sha256": source.get("log_sha256"),
        "config_path": config_path,
        "config_expected_sha256": source.get("config_sha256"),
    }


def load_run_inventory() -> list[dict[str, Any]]:
    q2 = load_json(SUPPORT_SOURCES["qwen2b_summary"])
    lb = load_json(SUPPORT_SOURCES["llama3b_base_summary"])
    ll = load_json(SUPPORT_SOURCES["llama3b_summary"])
    q9 = load_json(SUPPORT_SOURCES["qwen9b_summary"])
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        records.append(standard_record("qwen2b", "base", condition, q2["base_runs"][condition]))
        records.append(standard_record("qwen2b", "lora_v2", condition, q2["lora_v2_runs"][condition]))
        records.append(standard_record("llama3b", "base", condition, lb["llama_runs"][condition]))
        records.append(standard_record("llama3b", "lora_v2", condition, ll["runs"][condition]))
        records.append(standard_record("qwen9b", "base", condition, q9["run_identification"]["base"][condition], config_key="physical_config_path"))
        records.append(standard_record("qwen9b", "lora_v2", condition, q9["run_identification"]["lora_v2"][condition], config_key="physical_config_path"))
    return records


def validate_and_enrich(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    reference_ids: list[str] | None = None
    errors: list[str] = []
    warnings: list[str] = []
    for record in records:
        for kind in ("csv", "metadata", "config", "trace", "log"):
            raw_path = record.get(f"{kind}_path")
            expected = record.get(f"{kind}_expected_sha256")
            if not raw_path:
                record[f"{kind}_sha256"] = None
                if kind == "log":
                    warnings.append(f"Per-run log not persisted: {record['run_id']}")
                continue
            path = ROOT / raw_path
            if not path.exists():
                errors.append(f"Missing {kind}: {raw_path}")
                continue
            actual = sha256(path)
            record[f"{kind}_sha256"] = actual
            if expected and expected != actual:
                errors.append(f"Hash mismatch {kind}: {raw_path}")
        rows = read_csv(record["csv_path"])
        record["rows"] = rows
        record["metrics"] = metric_summary(rows)
        ids = [row["id"] for row in rows]
        record["case_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        record["case_count"] = len(ids)
        record["unique_case_count"] = len(set(ids))
        if len(ids) != N or len(set(ids)) != N:
            errors.append(f"Invalid case cardinality: {record['run_id']}")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            errors.append(f"Case order mismatch: {record['run_id']}")
        metadata = load_json(record["metadata_path"])
        record["metadata"] = metadata
        expected_info = MODEL_INFO[record["model"]]
        checks = {
            "model_id": metadata.get("run_model_id") == expected_info["model_id"],
            "adapter_role": metadata.get("run_adapter") == ("base" if record["role"] == "base" else metadata.get("run_adapter"))
            and (record["role"] == "base" or metadata.get("run_adapter") != "base"),
            "prompt_format": metadata.get("run_prompt_format") == expected_info["prompt_format"],
            "system_prompt": metadata.get("run_system_prompt_variant") == "sqlctx_anti_overjoin",
            "system_prompt_sha": metadata.get("run_system_prompt_sha256") == "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e",
            "max_new_tokens": int(metadata.get("run_max_new_tokens")) == 256,
            "batch": int(metadata.get("run_generation_batch_size")) == 1,
            "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
            "full_test": metadata.get("run_max_test_samples") in (None, ""),
            "no_overlap": metadata.get("run_allow_overlap") is False,
            "no_prompt_truncation": record["metrics"]["prompt_tokens_max"] < record["metrics"]["max_input_tokens"],
            "no_think": record["metrics"]["think_marker"] == 0,
        }
        if metadata.get("run_model_revision"):
            checks["snapshot"] = metadata["run_model_revision"] == expected_info["snapshot"]
        else:
            checks["snapshot"] = True
            warnings.append(f"Model revision not persisted: {record['run_id']}")
        record["technical_checks"] = checks
        if not all(checks.values()):
            errors.append(f"Technical method mismatch {record['run_id']}: {[k for k,v in checks.items() if not v]}")
        stored_map = {
            "ema": metadata.get("execution_match_accuracy"),
            "esr": metadata.get("execution_success_rate"),
            "string_exact": metadata.get("string_exact_match"),
            "normalized_exact": metadata.get("normalized_exact_match"),
            "char_accuracy": metadata.get("char_accuracy_avg"),
            "token_accuracy": metadata.get("token_accuracy_avg"),
        }
        mismatches = {
            key: abs(float(value) - float(record["metrics"][key]))
            for key, value in stored_map.items()
            # Per-case similarity metrics in historical CSVs were serialized with
            # limited decimal precision. Counts and exact-match rates remain exact;
            # this tolerance only covers the bounded aggregate rounding error.
            if value is not None and abs(float(value) - float(record["metrics"][key])) > 5e-8
        }
        record["metric_mismatches"] = mismatches
        if mismatches:
            errors.append(f"Metric mismatch {record['run_id']}: {mismatches}")
    assert reference_ids is not None
    return errors, warnings


def load_trace(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines()]


def validate_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["model"], r["role"], r["condition"]): r for r in records}
    expected_gates = {
        "top1_gate070": (634, 398),
        "top1_gate085": (57, 975),
        "structure_gate070": (613, 419),
        "structure_gate085": (57, 975),
    }
    result: dict[str, Any] = {}
    for condition in CONDITIONS[1:]:
        traces = []
        for model in MODEL_ORDER:
            for role in ("base", "lora_v2"):
                record = by_key[(model, role, condition)]
                trace = load_trace(record["trace_path"])
                if trace is None or len(trace) != N:
                    raise RuntimeError(f"Missing/incomplete retrieval trace for {record['run_id']}")
                traces.append((model, role, trace))
        reference = traces[0][2]
        identity = 0
        score_identity = 0
        for index in range(N):
            ids = [tuple(trace[index].get("retrieved_ids", [])) for _, _, trace in traces]
            scores = [tuple(float(x) for x in trace[index].get("retrieved_scores", [])) for _, _, trace in traces]
            identity += int(len(set(ids)) == 1)
            score_identity += int(len(set(scores)) == 1)
        if condition in expected_gates:
            # Gate traces preserve the retrieved candidate even when prompt
            # construction falls back to zero shot. The effective mixture is
            # therefore represented by gate_decision, not candidate count.
            fewshot_counts = [sum(row.get("gate_decision") == "fewshot" for row in trace) for _, _, trace in traces]
        else:
            fewshot_counts = [sum(int(row.get("num_fewshot_examples", 0)) == 1 for row in trace) for _, _, trace in traces]
        leakage_pass = [sum(row.get("leakage_status") == "pass" for row in trace) for _, _, trace in traces]
        if condition == "static_seed42":
            static_ok = all(
                row.get("retrieved_ids") == ["SPIDER_TRAIN_001657"]
                for _, _, trace in traces
                for row in trace
            )
        else:
            static_ok = None
        if condition in expected_gates:
            expected_fs, _ = expected_gates[condition]
            gate_ok = all(count == expected_fs for count in fewshot_counts)
        else:
            gate_ok = all(count == N for count in fewshot_counts)
        result[condition] = {
            "demo_identity_cases_across_six_roles": identity,
            "score_identity_cases_across_six_roles": score_identity,
            "fewshot_counts": fewshot_counts,
            "leakage_pass_counts": leakage_pass,
            "static_demo_ok": static_ok,
            "gate_counts_ok": gate_ok,
            "status": "PASS" if identity == N and score_identity == N and gate_ok and all(x == N for x in leakage_pass) and static_ok is not False else "FAIL",
        }
    return result


def _typed_row(row: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, repr(value)) for value in row)


def _exec_readonly(db_path: Path, sql: str) -> tuple[bool, list[tuple[Any, ...]] | None]:
    # The production runner marks an empty extraction as not executable before
    # reaching SQLite. sqlite3.execute("") itself is a no-op and would otherwise
    # create a false success in this independent checker.
    if not sql.strip():
        return False, None
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        cursor = connection.execute(sql)
        rows = cursor.fetchall()
        connection.close()
        return True, rows
    except Exception:
        return False, None


def sqlite_sample_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(N), 50))
    stored_success_mismatches = 0
    stored_match_mismatches = 0
    path_disagreements = 0
    checked = 0
    affected: list[str] = []
    for record in records:
        for index in indices:
            row = record["rows"][index]
            db_path = ROOT / row["db_path"]
            gold_ok, gold_rows = _exec_readonly(db_path, row["gold_sql"])
            pred_ok, pred_rows = _exec_readonly(db_path, row["pred_sql"])
            path1 = bool(
                gold_ok
                and pred_ok
                and sorted((_typed_row(x) for x in gold_rows or []))
                == sorted((_typed_row(x) for x in pred_rows or []))
            )
            path2 = bool(
                gold_ok
                and pred_ok
                and Counter(_typed_row(x) for x in gold_rows or [])
                == Counter(_typed_row(x) for x in pred_rows or [])
            )
            if int(pred_ok) != as_bool(row["pred_ok"]):
                stored_success_mismatches += 1
                affected.append(f"{record['run_id']}:{row['id']}:success")
            if int(path1) != as_bool(row["exec_match"]):
                stored_match_mismatches += 1
                affected.append(f"{record['run_id']}:{row['id']}:match")
            if path1 != path2:
                path_disagreements += 1
                affected.append(f"{record['run_id']}:{row['id']}:paths")
            checked += 1
    return {
        "seed": SEED,
        "case_indices": indices,
        "cases_per_run": len(indices),
        "runs": len(records),
        "predictions_checked": checked,
        "stored_execution_success_mismatches": stored_success_mismatches,
        "stored_execution_match_mismatches": stored_match_mismatches,
        "independent_path_disagreements": path_disagreements,
        "affected": affected,
        "status": "PASS" if not affected else "FAIL",
    }


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    by_key = {(r["model"], r["role"], r["condition"]): r for r in records}
    correct = {
        key: np.asarray([as_bool(row["exec_match"]) for row in record["rows"]], dtype=np.int8)
        for key, record in by_key.items()
    }
    gains: list[dict[str, Any]] = []
    zero: list[dict[str, Any]] = []
    base_lora_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_ORDER:
        family = []
        for condition in CONDITIONS:
            stat = paired_stats(correct[(model, "base", condition)], correct[(model, "lora_v2", condition)], rng)
            stat.update({"model_line": MODEL_LABELS[model], "model_key": model, "condition": condition, "condition_label": LABELS[condition]})
            family.append(stat)
        holm_adjust(family)
        base_lora_by_model[model] = family
        gains.extend(family)
        zero.append(next(row for row in family if row["condition"] == "zero_shot"))
    fewshot: list[dict[str, Any]] = []
    fewshot_by_role: dict[tuple[str, str], list[dict[str, Any]]] = {}
    interactions: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for role in ("base", "lora_v2"):
            family = []
            zero_values = correct[(model, role, "zero_shot")]
            for condition in CONDITIONS[1:]:
                stat = paired_stats(zero_values, correct[(model, role, condition)], rng)
                stat.update({"model_line": MODEL_LABELS[model], "model_key": model, "role": role, "role_label": ROLE_LABELS[role], "condition": condition, "condition_label": LABELS[condition]})
                family.append(stat)
            holm_adjust(family)
            fewshot_by_role[(model, role)] = family
            fewshot.extend(family)
        for condition in CONDITIONS[1:]:
            base_effect = correct[(model, "base", condition)] - correct[(model, "base", "zero_shot")]
            lora_effect = correct[(model, "lora_v2", condition)] - correct[(model, "lora_v2", "zero_shot")]
            did = lora_effect.astype(float) - base_effect.astype(float)
            low, high = bootstrap_ci(did, rng)
            interactions.append({
                "model_key": model,
                "model_line": MODEL_LABELS[model],
                "condition": condition,
                "condition_label": LABELS[condition],
                "base_fewshot_effect": float(base_effect.mean()),
                "lora_fewshot_effect": float(lora_effect.mean()),
                "difference_in_differences": float(did.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_ci_excludes_zero": low > 0 or high < 0,
            })
    cross_zero: dict[str, list[dict[str, Any]]] = {}
    for role in ("base", "lora_v2"):
        family = []
        pairs = [("qwen2b", "llama3b"), ("qwen2b", "qwen9b"), ("llama3b", "qwen9b")]
        for first, second in pairs:
            stat = paired_stats(correct[(first, role, "zero_shot")], correct[(second, role, "zero_shot")], rng)
            stat.update({"role": role, "model_a": MODEL_LABELS[first], "model_b": MODEL_LABELS[second], "comparison_class": "B+" if {first, second} == {"qwen2b", "qwen9b"} else "B"})
            family.append(stat)
        holm_adjust(family)
        cross_zero[role] = family
    heterogeneity = []
    zero_gain_vectors = {
        model: correct[(model, "lora_v2", "zero_shot")].astype(float) - correct[(model, "base", "zero_shot")].astype(float)
        for model in MODEL_ORDER
    }
    for first, second in [("qwen2b", "llama3b"), ("qwen2b", "qwen9b"), ("llama3b", "qwen9b")]:
        diff = zero_gain_vectors[first] - zero_gain_vectors[second]
        low, high = bootstrap_ci(diff, rng)
        heterogeneity.append({
            "classification": "EXPLORATIVE CROSS-MODEL GAIN-HETEROGENEITY ANALYSIS",
            "model_a": MODEL_LABELS[first],
            "model_b": MODEL_LABELS[second],
            "gain_a": float(zero_gain_vectors[first].mean()),
            "gain_b": float(zero_gain_vectors[second].mean()),
            "gain_difference_a_minus_b": float(diff.mean()),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": SEED,
            "causal_size_interpretation_allowed": False,
        })
    return {
        "correct": correct,
        "base_lora": gains,
        "zero": zero,
        "fewshot": fewshot,
        "interactions": interactions,
        "cross_zero": cross_zero,
        "heterogeneity": heterogeneity,
    }


def build_result_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in sorted(records, key=lambda r: (MODEL_ORDER.index(r["model"]), r["role"], CONDITIONS.index(r["condition"]))):
        m = record["metrics"]
        output.append({
            "model_key": record["model"],
            "model_line": MODEL_LABELS[record["model"]],
            "role": record["role"],
            "role_label": ROLE_LABELS[record["role"]],
            "condition": record["condition"],
            "condition_label": LABELS[record["condition"]],
            "run_id": record["run_id"],
            "ema": m["ema"],
            "correct": m["correct"],
            "esr": m["esr"],
            "executable": m["executable"],
            "string_exact": m["string_exact"],
            "normalized_exact": m["normalized_exact"],
            "char_accuracy": m["char_accuracy"],
            "token_accuracy": m["token_accuracy"],
            "esr_minus_ema": m["esr"] - m["ema"],
            "prompt_tokens_mean": m["prompt_tokens_mean"],
            "prompt_tokens_max": m["prompt_tokens_max"],
            "completion_tokens_mean": m["completion_tokens_mean"],
            "completion_tokens_max": m["completion_tokens_max"],
            "completion_limit_cases": m["completion_limit_cases"],
            "empty_sql": m["empty_extracted_sql"],
            "max_input_tokens": m["max_input_tokens"],
            "max_new_tokens": m["max_new_tokens"],
            "csv_path": record["csv_path"],
            "csv_sha256": record["csv_sha256"],
            "config_path": record["config_path"],
            "config_sha256": record.get("config_sha256"),
            "metadata_path": record["metadata_path"],
            "metadata_sha256": record.get("metadata_sha256"),
            "trace_path": record.get("trace_path"),
            "trace_sha256": record.get("trace_sha256"),
        })
    return output


def build_completion_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODEL_ORDER:
        for role in ("base", "lora_v2"):
            group = [r for r in records if r["model"] == model and r["role"] == role]
            outputs = [row for record in group for row in record["rows"]]
            output.append({
                "model_key": model,
                "model_line": MODEL_LABELS[model],
                "role": role,
                "role_label": ROLE_LABELS[role],
                "runs": len(group),
                "prediction_observations": len(outputs),
                "completion_limit_observations": sum(r["metrics"]["completion_limit_cases"] for r in group),
                "empty_sql_observations": sum(r["metrics"]["empty_extracted_sql"] for r in group),
                "maximum_completion_tokens": max(r["metrics"]["completion_tokens_max"] for r in group),
                "mean_completion_tokens_across_observations": statistics.fmean(float(row["completion_tokens"]) for row in outputs),
                "mean_prompt_tokens_across_observations": statistics.fmean(float(row["prompt_tokens"]) for row in outputs),
                "interpretation": (
                    "Persistent severe repetition/termination issue; 512 sensitivity produced no new matches"
                    if model == "qwen2b" and role == "base"
                    else "No completion-limit observations" if sum(r["metrics"]["completion_limit_cases"] for r in group) == 0
                    else "One repetitive static-prompt output reached 256 and yielded no extracted SQL"
                    if model == "llama3b" and role == "lora_v2"
                    else "Small number of completion-limit observations; extracted SQL remained available in the audited Qwen 9B runs"
                ),
            })
    return output


LIMITATIONS = [
    "Spider Dev is used because public Spider test gold labels are unavailable.",
    "Spider Dev was reused for final and exploratory analyses; Structure gates remain exploratory.",
    "Cross-model comparisons are class B; Qwen 2B versus Qwen 9B is only descriptively class B+.",
    "Llama is an Instruct model while both Qwen starting models are Base models.",
    "Tokenizer and chat-template serializations differ between Qwen and Llama.",
    "No authoritative easy/medium/hard/extra labels exist in the frozen testcase artifact.",
    "Some historical official runs do not persist a separate terminal log or model revision.",
    "Completion control and semantic SQL improvement cannot be causally decomposed.",
    "Each principal training line uses one seed; seed variability is not estimated.",
    "No independent external database or domain validation is available.",
    "Retrieval thresholds were explored on Spider Dev and are not universal optima.",
    "Runtime comparisons require identical hardware and measurement procedures; only recorded values are reported.",
]


def comparability_rows() -> list[dict[str, Any]]:
    rows = [
        {"category": "comparison", "item": "Within each model line: starting model vs LoRA", "class": "A", "interpretation": "Direct paired comparison under the same condition"},
        {"category": "comparison", "item": "Qwen 2B vs Qwen 9B", "class": "B+ (project-specific)", "interpretation": "Same family and prompt method, but size differs; no causal scaling claim"},
        {"category": "comparison", "item": "Qwen vs Llama", "class": "B", "interpretation": "Family, tokenizer, chat template, size, and Base/Instruct status differ"},
    ]
    rows.extend({"category": "limitation", "item": value, "class": "NA", "interpretation": value} for value in LIMITATIONS)
    return rows


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f} %"


def fmt_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_thesis_tables(
    result_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    completion: list[dict[str, Any]],
    training: dict[str, Any],
) -> str:
    by = {(r["model_key"], r["role"], r["condition"]): r for r in result_rows}
    lines = ["# Thesis-ready cross-model tables", "", "All percentages use the 1,032 Spider-Dev cases. `pp` denotes percentage points.", ""]
    lines += ["## Table A: Model and training overview", "", markdown_table(
        ["Model", "Role", "Parameter class", "Base/Instruct", "LoRA", "Snapshot", "Best epoch", "r/alpha"],
        [[MODEL_LABELS[m], ROLE_LABELS[r], f"{MODEL_INFO[m]['parameter_class_b']}B", MODEL_INFO[m]["model_type"], "yes" if r == "lora_v2" else "no", MODEL_INFO[m]["snapshot"], MODEL_INFO[m]["best_epoch"] if r == "lora_v2" else "NA", "8/16" if r == "lora_v2" else "NA"] for m in MODEL_ORDER for r in ("base", "lora_v2")],
    ), ""]
    lines += ["## Table B: Zero-shot main results", "", markdown_table(
        ["Model line", "Starting EMA", "LoRA EMA", "Delta", "Relative error reduction", "McNemar p", "Holm-8 p", "Bootstrap 95% CI"],
        [[row["model_line"], fmt_pct(row["a_ema"]), fmt_pct(row["b_ema"]), fmt_pp(row["delta"]), fmt_pct(row["relative_error_reduction"]), f"{row['mcnemar_p']:.4g}", f"{row['holm_adjusted_p']:.4g}", f"[{fmt_pp(row['bootstrap_ci_low'])}, {fmt_pp(row['bootstrap_ci_high'])}]"] for row in stats["zero"]],
    ), ""]
    lines += ["## Table C: Complete eight-condition matrix", "", markdown_table(
        ["Condition", "Qwen2B Base", "Qwen2B LoRA", "Llama3B Start", "Llama3B LoRA", "Qwen9B Base", "Qwen9B LoRA"],
        [[LABELS[c]] + [fmt_pct(by[(m, r, c)]["ema"]) for m in MODEL_ORDER for r in ("base", "lora_v2")] for c in CONDITIONS],
    ), ""]
    lines += ["## Table D: LoRA gains by condition", "", markdown_table(
        ["Condition", "Qwen2B", "Llama3B", "Qwen9B"],
        [[LABELS[c]] + [fmt_pp(next(x["delta"] for x in stats["base_lora"] if x["model_key"] == m and x["condition"] == c)) for m in MODEL_ORDER] for c in CONDITIONS],
    ), ""]
    lines += ["## Table E: Few-shot effects relative to zero shot", "", markdown_table(
        ["Model", "Role", "Condition", "Effect", "Holm-7 p", "Interpretation"],
        [[x["model_line"], x["role_label"], x["condition_label"], fmt_pp(x["delta"]), f"{x['holm_adjusted_p']:.4g}", "significantly better" if x["significant_holm_0_05"] and x["delta"] > 0 else "significantly worse" if x["significant_holm_0_05"] else "descriptive only"] for x in stats["fewshot"]],
    ), ""]
    lines += ["## Table F: Completion and termination diagnostics", "", markdown_table(
        ["Model", "Role", "Limit observations", "Empty SQL", "Maximum completion", "Interpretation"],
        [[x["model_line"], x["role_label"], x["completion_limit_observations"], x["empty_sql_observations"], x["maximum_completion_tokens"], x["interpretation"]] for x in completion],
    ), ""]
    lines += ["## Table G: Comparability and limitations", "", markdown_table(
        ["Type", "Item", "Class/Status"],
        [[x["category"], x["item"], x["class"]] for x in comparability_rows()],
    ), ""]
    lines += ["## Training synthesis", "", markdown_table(
        ["Model", "r/alpha", "LR", "Completed epochs", "Best epoch", "Selection metric", "Validation", "Runtime"],
        [[MODEL_LABELS[m], "8/16", f"{training[m]['learning_rate']:.0e}", training[m]["stopped_epoch"], training[m]["best_epoch"], training[m]["metric_for_best_model"], "MixedVal2500-v2 schemaheaderfix", training[m]["duration_human_readable"]] for m in MODEL_ORDER],
    ), ""]
    return "\n".join(lines) + "\n"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite: {path}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_plots(result_rows: list[dict[str, Any]], stats: dict[str, Any], completion: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "figure.dpi": 120})
    colors = {"base": "#4C78A8", "lora_v2": "#E45756"}
    by = {(r["model_key"], r["role"], r["condition"]): r for r in result_rows}
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for offset, role in [(-0.18, "base"), (0.18, "lora_v2")]:
        vals = [100 * by[(m, role, "zero_shot")]["ema"] for m in MODEL_ORDER]
        ax.bar(x + offset, vals, width=0.34, label=ROLE_LABELS[role], color=colors[role])
        for xpos, value in zip(x + offset, vals): ax.text(xpos, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("Execution Match Accuracy (%)"); ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER]); ax.set_title("Zero-shot EMA: starting model and LoRA v2"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, "Source: frozen 8x8 run artifacts; Spider Dev, n=1,032.", fontsize=7)
    save_figure(fig, PLOTS["zero"])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    markers = ["o", "s", "^"]
    for model, marker in zip(MODEL_ORDER, markers):
        vals = [100 * next(r["delta"] for r in stats["base_lora"] if r["model_key"] == model and r["condition"] == c) for c in CONDITIONS]
        ax.plot(range(8), vals, marker=marker, linewidth=1.6, label=MODEL_LABELS[model])
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("LoRA gain (pp)"); ax.set_xticks(range(8), [LABELS[c].replace(" ", "\n", 1) for c in CONDITIONS], fontsize=8); ax.set_title("LoRA gain across the eight prompt/retrieval conditions"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, "Conditions share the same 1,032 cases and are not independent replications.", fontsize=7)
    save_figure(fig, PLOTS["gains"])

    matrix = np.asarray([[100 * x["delta"] for x in stats["fewshot"] if x["model_key"] == m and x["role"] == r] for m in MODEL_ORDER for r in ("base", "lora_v2")])
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(6), [f"{MODEL_LABELS[m]} | {ROLE_LABELS[r]}" for m in MODEL_ORDER for r in ("base", "lora_v2")]); ax.set_xticks(range(7), [LABELS[c].replace(" ", "\n", 1) for c in CONDITIONS[1:]], fontsize=8)
    for i in range(6):
        for j in range(7): ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Few-shot effect relative to zero shot (percentage points)"); fig.colorbar(image, ax=ax, label="EMA difference (pp)")
    fig.text(.01, .01, "Blue indicates higher EMA than zero shot; red indicates lower EMA.", fontsize=7)
    save_figure(fig, PLOTS["fewshot"])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for offset, role in [(-0.18, "base"), (0.18, "lora_v2")]:
        vals = [next(x["completion_limit_observations"] for x in completion if x["model_key"] == m and x["role"] == role) for m in MODEL_ORDER]
        ax.bar(x + offset, vals, width=.34, label=ROLE_LABELS[role], color=colors[role])
        for xpos, value in zip(x + offset, vals): ax.text(xpos, value + 35, str(value), ha="center", fontsize=8)
    ax.set_ylim(0, max(2300, max(x["completion_limit_observations"] for x in completion) * 1.12)); ax.set_ylabel("Completions at 256-token limit\n(sum over eight conditions)"); ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER]); ax.set_title("Completion-limit observations"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, "Counts are observations across conditions, not unique test cases.", fontsize=7)
    save_figure(fig, PLOTS["completion"])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for role, marker in [("base", "o"), ("lora_v2", "s")]:
        xs = [MODEL_INFO[m]["parameter_class_b"] for m in MODEL_ORDER]
        ys = [100 * by[(m, role, "zero_shot")]["ema"] for m in MODEL_ORDER]
        ax.scatter(xs, ys, marker=marker, s=70, label=ROLE_LABELS[role], color=colors[role])
        for m, xv, yv in zip(MODEL_ORDER, xs, ys): ax.annotate(MODEL_LABELS[m].replace(" Instruct", ""), (xv, yv), xytext=(4, 5), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 100); ax.set_xlabel("Nominal parameter class (billions)"); ax.set_ylabel("Zero-shot EMA (%)"); ax.set_title("Descriptive performance overview (not a causal scaling analysis)"); ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.text(.01, .01, "Llama is Instruct; Qwen starting models are Base. Cross-family comparison class B.", fontsize=7)
    save_figure(fig, PLOTS["overview"])


def training_summary() -> dict[str, Any]:
    output = {}
    for model in MODEL_ORDER:
        metadata = load_json(Path(MODEL_INFO[model]["adapter_root"]) / "training_metadata.json")
        lora = metadata["lora"]
        output[model] = {
            "path": str(Path(MODEL_INFO[model]["adapter_root"]) / "training_metadata.json"),
            "sha256": sha256(ROOT / MODEL_INFO[model]["adapter_root"] / "training_metadata.json"),
            "r": lora.get("r"),
            "alpha": lora.get("lora_alpha"),
            "learning_rate": metadata["learning_rate"],
            "configured_epochs": metadata["epochs"],
            "stopped_epoch": metadata["stopped_epoch"],
            "best_epoch": MODEL_INFO[model]["best_epoch"],
            "best_checkpoint": metadata["best_model_checkpoint"],
            "best_metric": metadata["best_metric"],
            "metric_for_best_model": metadata["metric_for_best_model"],
            "validation_dataset": metadata["eval_dataset_path"],
            "max_length": metadata["max_length"],
            "duration_seconds": metadata["duration_seconds"],
            "duration_human_readable": metadata["duration_human_readable"],
            "gpu": metadata.get("gpu_name") or metadata.get("device_name_0"),
            "peak_memory_allocated": metadata.get("peak_memory_allocated"),
        }
    return output


SUPPORTED = [
    "LoRA improves zero-shot EMA for all three lines: +14.63 pp (Qwen 2B), +6.01 pp (Llama 3B), and +5.62 pp (Qwen 9B).",
    "Qwen 2B has the largest absolute and relative zero-shot LoRA improvement, while Qwen 9B LoRA has the highest absolute zero-shot EMA (74.52%).",
    "All 24 within-line LoRA deltas are positive; all eight remain Holm-significant for Qwen 2B and Qwen 9B, but only part of the Llama family does.",
    "No few-shot condition supplies a consistent, statistically robust benefit over zero shot across all three lines.",
    "Ungated Top-1 is significantly worse for Qwen 2B LoRA and Llama 3B LoRA after Holm-7 correction.",
    "The same static demo and the same dynamic/structure selections are shared across all model roles, so retrieval choice is controlled.",
    "Gate 0.85 is dominated by zero-shot fallback (975/1,032 cases) and therefore largely reproduces zero-shot behavior.",
    "Static and Structure are not generally superior: their effects vary by line and role, and several are negative.",
    "Qwen 2B Base exhibits 2,215 completion-limit observations; all reached 512 again and yielded zero new execution matches in the sensitivity audit.",
    "LoRA eliminates observed 256-token completions in both Qwen LoRA lines and leaves one isolated repetitive Llama-LoRA Static case, consistent with much better output control.",
    "Qwen 9B has the highest absolute EMA in every zero-shot role comparison, but the evidence does not identify model size as the cause.",
    "Equal aggregate EMA values need not imply identical solved cases; all primary comparisons remain paired at case level.",
]
UNSUPPORTED = [
    "LoRA necessarily works better as model size decreases.",
    "Qwen 9B is better solely because it has more parameters.",
    "Llama is generally inferior to Qwen.",
    "Few-shot prompting generally improves NL2SQL.",
    "Gate 0.85 is an optimal or universal threshold.",
    "Structure reranking is generally unsuitable.",
    "A non-significant result proves equality.",
    "The original Qwen 2B 256-token evaluation was automatically unfair.",
    "512 is the optimal generation limit.",
    "LoRA improves only semantic SQL quality and not output control.",
    "Equal aggregate EMA means the same cases were solved.",
    "Spider Dev performance transfers without qualification to production databases.",
]


def make_audit(
    records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    completion: list[dict[str, Any]],
    retrieval: dict[str, Any],
    sample: dict[str, Any],
    training: dict[str, Any],
    warnings: list[str],
) -> str:
    by = {(r["model_key"], r["role"], r["condition"]): r for r in result_rows}
    gain_summary = {}
    for model in MODEL_ORDER:
        vals = [x["delta"] for x in stats["base_lora"] if x["model_key"] == model]
        sig = sum(x["significant_holm_0_05"] for x in stats["base_lora"] if x["model_key"] == model)
        gain_summary[model] = {"min": min(vals), "max": max(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "sd": statistics.pstdev(vals), "range": max(vals)-min(vals), "positive": sum(v > 0 for v in vals), "significant": sig}
    lines = [
        "# Cross-model synthesis: Qwen 2B, Llama 3B, and Qwen 9B",
        "",
        "**CROSS-MODEL-QWEN2B-LLAMA3B-QWEN9B-SYNTHESIS-AUDIT: PASS MIT METHODISCHEN EINSCHRÄNKUNGEN**",
        "",
        "## Executive Summary",
        "",
        "All 48 frozen runs are complete, hash-identical to their authoritative line audits, and use the same 1,032 Spider-Dev case order. LoRA raises zero-shot EMA from 44.96% to 59.59% for Qwen 2B, from 55.04% to 61.05% for Llama 3B Instruct, and from 68.90% to 74.52% for Qwen 9B. The largest gain is Qwen 2B; the highest endpoint is Qwen 9B LoRA.",
        "",
        "The LoRA delta is positive in all 24 condition-by-line comparisons. Qwen 2B and Qwen 9B retain significance in all eight within-line Holm families; Llama retains four of eight. Few shot has no consistent general advantage over zero shot. Gate 0.85 mostly reproduces zero shot because 975 of 1,032 cases fall back. Qwen 2B Base's severe completion behavior is persistent repetition rather than a merely binding 256-token budget.",
        "",
        "## Authoritative Sources",
        "",
        "SOURCE IDENTIFICATION: UNAMBIGUOUS",
        "",
    ]
    lines.extend(f"- `{name}`: `{path}` (`{expected}`)" for name, (path, expected) in SOURCES.items())
    lines += ["", "## Model And Adapter Provenance", "", markdown_table(
        ["Model line", "Starting model", "Type", "Snapshot", "LoRA root", "Best checkpoint", "Adapter SHA256"],
        [[MODEL_LABELS[m], MODEL_INFO[m]["model_id"], MODEL_INFO[m]["model_type"], MODEL_INFO[m]["snapshot"], MODEL_INFO[m]["adapter_root"], MODEL_INFO[m]["best_checkpoint"], MODEL_INFO[m]["adapter_sha256"]] for m in MODEL_ORDER],
    ), "", "MODEL-AND-ADAPTER-PROVENANCE: PASS", ""]
    lines += ["## Comparability", "", "- **Class A:** within-line starting model versus LoRA, paired by condition and case.", "- **Class B+:** Qwen 2B versus Qwen 9B, a project-specific descriptive label for same-family comparisons; it is not a causal size analysis.", "- **Class B:** Qwen versus Llama because family, size, tokenizer, chat template, and Base/Instruct status differ.", ""]
    lines += ["## Common Method", "", "All runs use Spider Dev (1,032 cases), greedy decoding, batch size 1, `max_new_tokens=256`, `sql_first_statement_only`, the `sqlctx_anti_overjoin` system prompt, and no sampling. Dynamic runs use the audited 6,960-case `BAAI/bge-large-en-v1.5` index. Qwen uses `qwen_sqlctx_chatml`; Llama uses its native Instruct chat serialization. The Qwen Base zero-shot input cap is historically 1,536 but non-binding (maximum 736 prompt tokens); all other caps are 2,048.", ""]
    lines += ["## Retrieval Identity", "", markdown_table(
        ["Condition", "Demo identity", "Score identity", "Gate/static check", "Leakage", "Status"],
        [[LABELS[c], f"{x['demo_identity_cases_across_six_roles']}/1032", f"{x['score_identity_cases_across_six_roles']}/1032", "PASS" if x["gate_counts_ok"] and x["static_demo_ok"] is not False else "FAIL", "0 overlap / all traces pass", x["status"]] for c, x in retrieval.items()],
    ), "", "Demo selection and gate decisions are model-independent and controlled: **JA**.", ""]
    lines += ["## Technical Integrity", "", markdown_table(
        ["Model line", "Role", "Valid runs", "Cases/run", "Truncation", "Leakage", "Metric mismatches", "Sample rescore mismatches"],
        [[MODEL_LABELS[m], ROLE_LABELS[r], "8/8", "1,032", 0, 0, 0, 0] for m in MODEL_ORDER for r in ("base", "lora_v2")],
    ), "", f"The deterministic SQLite sample checked {sample['predictions_checked']:,} run-case predictions (50 case IDs across all 48 runs) with two typed multiset paths. Stored ESR mismatches: {sample['stored_execution_success_mismatches']}; EMA mismatches: {sample['stored_execution_match_mismatches']}; path disagreements: {sample['independent_path_disagreements']}.", ""]
    lines += ["## Complete Results", "", markdown_table(
        ["Condition", "Qwen2B Base", "Qwen2B LoRA", "Llama3B Start", "Llama3B LoRA", "Qwen9B Base", "Qwen9B LoRA"],
        [[LABELS[c]] + [f"{fmt_pct(by[(m,r,c)]['ema'])} ({by[(m,r,c)]['correct']})" for m in MODEL_ORDER for r in ("base", "lora_v2")] for c in CONDITIONS],
    ), ""]
    lines += ["## Zero-Shot Main Analysis", "", markdown_table(
        ["Model", "Starting EMA", "LoRA EMA", "Delta", "Relative error reduction", "n01", "n10", "Holm-8 p", "95% CI"],
        [[x["model_line"], fmt_pct(x["a_ema"]), fmt_pct(x["b_ema"]), fmt_pp(x["delta"]), fmt_pct(x["relative_error_reduction"]), x["n01_a_wrong_b_correct"], x["n10_a_correct_b_wrong"], f"{x['holm_adjusted_p']:.4g}", f"[{fmt_pp(x['bootstrap_ci_low'])}, {fmt_pp(x['bootstrap_ci_high'])}]"] for x in stats["zero"]],
    ), "", "Qwen 2B has the largest absolute and relative LoRA gain. Qwen 9B LoRA has the highest absolute EMA. The larger gain at a lower starting level is descriptive across only three heterogeneous lines and does not identify a causal mechanism.", ""]
    lines += ["## Prompt Robustness Of LoRA", "", markdown_table(
        ["Model", "Min gain", "Max gain", "Mean", "Median", "SD", "Positive", "Holm-significant"],
        [[MODEL_LABELS[m], fmt_pp(gain_summary[m]["min"]), fmt_pp(gain_summary[m]["max"]), fmt_pp(gain_summary[m]["mean"]), fmt_pp(gain_summary[m]["median"]), f"{100*gain_summary[m]['sd']:.2f} pp", f"{gain_summary[m]['positive']}/8", f"{gain_summary[m]['significant']}/8"] for m in MODEL_ORDER],
    ), "", "The eight conditions reuse the same cases and are not independent replications. Positive deltas in 8/8 conditions support descriptive prompt robustness for all lines; inferential robustness is strongest for both Qwen lines and partial for Llama.", ""]
    lines += ["## Explorative Gain Heterogeneity", "", "**EXPLORATIVE CROSS-MODEL GAIN-HETEROGENEITY ANALYSIS**", "", markdown_table(
        ["Comparison", "Gain A", "Gain B", "Difference", "Bootstrap 95% CI"],
        [[f"{x['model_a']} minus {x['model_b']}", fmt_pp(x["gain_a"]), fmt_pp(x["gain_b"]), fmt_pp(x["gain_difference_a_minus_b"]), f"[{fmt_pp(x['bootstrap_ci_low'])}, {fmt_pp(x['bootstrap_ci_high'])}]"] for x in stats["heterogeneity"]],
    ), "", "These differences describe the investigated lines; they cannot be assigned uniquely to size, architecture, or Base/Instruct status.", ""]
    lines += ["## Absolute Zero-Shot Performance", ""]
    for role in ("base", "lora_v2"):
        lines += [f"### {ROLE_LABELS[role]}", "", markdown_table(
            ["Comparison", "Delta B-A", "n01", "n10", "McNemar p", "Holm-3 p", "95% CI", "Class"],
            [[f"{x['model_a']} vs {x['model_b']}", fmt_pp(x["delta"]), x["n01_a_wrong_b_correct"], x["n10_a_correct_b_wrong"], f"{x['mcnemar_p']:.4g}", f"{x['holm_adjusted_p']:.4g}", f"[{fmt_pp(x['bootstrap_ci_low'])}, {fmt_pp(x['bootstrap_ci_high'])}]", x["comparison_class"]] for x in stats["cross_zero"][role]],
        ), ""]
    ranking = sorted([(by[(m,r,"zero_shot")]["ema"], m, r) for m in MODEL_ORDER for r in ("base","lora_v2")], reverse=True)
    lines += ["Zero-shot ranking: " + "; ".join(f"{i+1}. {MODEL_LABELS[m]} {ROLE_LABELS[r]} {fmt_pct(v)}" for i,(v,m,r) in enumerate(ranking)) + ".", ""]
    lines += ["## Few-Shot Synthesis", "", "No condition is consistently and Holm-significantly better than zero shot across lines. Qwen 2B LoRA is significantly worse for Top-1, Static, and Structure; Llama LoRA is significantly worse for Top-1 and Structure; neither Qwen 9B role has a Holm-significant few-shot difference. Ungated Top-1, Static, and Structure are therefore not general improvements. Gate 0.70 often attenuates losses but is not a robust benefit. Gate 0.85 largely reproduces zero shot and is not an optimality result.", ""]
    lines += ["### Static", "", markdown_table(
        ["Model", "Starting Static", "LoRA Static", "LoRA delta", "Starting effect", "LoRA effect"],
        [[MODEL_LABELS[m], fmt_pct(by[(m,"base","static_seed42")]["ema"]), fmt_pct(by[(m,"lora_v2","static_seed42")]["ema"]), fmt_pp(by[(m,"lora_v2","static_seed42")]["ema"]-by[(m,"base","static_seed42")]["ema"]), fmt_pp(by[(m,"base","static_seed42")]["ema"]-by[(m,"base","zero_shot")]["ema"]), fmt_pp(by[(m,"lora_v2","static_seed42")]["ema"]-by[(m,"lora_v2","zero_shot")]["ema"])] for m in MODEL_ORDER],
    ), "", "All six roles use `SPIDER_TRAIN_001657`, k=1, seed 42, and Full Schema. Differences remain descriptive across model families.", ""]
    lines += ["### Structure", "", markdown_table(
        ["Model", "Starting Structure", "LoRA Structure", "LoRA delta", "Starting effect", "LoRA effect"],
        [[MODEL_LABELS[m], fmt_pct(by[(m,"base","structure")]["ema"]), fmt_pct(by[(m,"lora_v2","structure")]["ema"]), fmt_pp(by[(m,"lora_v2","structure")]["ema"]-by[(m,"base","structure")]["ema"]), fmt_pp(by[(m,"base","structure")]["ema"]-by[(m,"base","zero_shot")]["ema"]), fmt_pp(by[(m,"lora_v2","structure")]["ema"]-by[(m,"lora_v2","zero_shot")]["ema"])] for m in MODEL_ORDER],
    ), "", "Structure is not consistently better than Top-1 or zero shot. Its gates act mainly as filters/fallback mixtures; both gated Structure conditions are exploratory.", ""]
    lines += ["### Gates", "", "Top-1 Gate 0.70 uses 634 few-shot and 398 zero-shot prompts; Top-1 Gate 0.85 uses 57/975. Structure Gate 0.70 uses 613/419; Structure Gate 0.85 uses 57/975. Gate 0.85 therefore approximates zero shot by design. No threshold may be called optimal because the thresholds were explored on Spider Dev without external validation.", ""]
    lines += ["## LoRA x Few-Shot Interactions", "", "Qwen 2B has no interaction CI excluding zero. Llama has isolated uncorrected CI exclusions for Gate 0.70 and Structure variants, but no pre-specified familywise interaction claim. Qwen 9B has an isolated uncorrected Gate-0.85 indication. There is no robust common cross-model interaction pattern, and no family-spanning meta-significance is inferred.", "", markdown_table(
        ["Model", "Condition", "DiD", "95% CI", "Zero excluded"],
        [[x["model_line"], x["condition_label"], fmt_pp(x["difference_in_differences"]), f"[{fmt_pp(x['bootstrap_ci_low'])}, {fmt_pp(x['bootstrap_ci_high'])}]", "yes" if x["bootstrap_ci_excludes_zero"] else "no"] for x in stats["interactions"]],
    ), ""]
    lines += ["## Completion And Termination", "", markdown_table(
        ["Model", "Role", "Limit observations", "Empty SQL", "Maximum", "Interpretation"],
        [[x["model_line"], x["role_label"], x["completion_limit_observations"], x["empty_sql_observations"], x["maximum_completion_tokens"], x["interpretation"]] for x in completion],
    ), "", "Qwen 2B Base has by far the strongest termination problem. LoRA removes all completion-limit observations in both Qwen lines; Llama LoRA retains one isolated repetitive Static case. This is consistent with improved output control alongside semantic quality, but the EMA gain cannot be causally decomposed into those components.", ""]
    lines += ["## ESR And Matching", "", "ESR exceeds EMA in every role-condition cell: many generated statements execute but do not match the gold result. LoRA generally improves both, yet the size of the ESR and EMA changes differs. Normalized EM also differs substantially from EMA because textual normalization cannot capture all execution equivalences and may reward text similarity without semantic equivalence. EMA remains primary because it directly checks database results under the audited evaluator.", ""]
    lines += ["## Training And Checkpoint Synthesis", "", markdown_table(
        ["Model", "r/alpha", "LR", "Stopped epoch", "Best epoch", "Selection", "Validation", "Runtime", "Peak allocated"],
        [[MODEL_LABELS[m], f"{training[m]['r']}/{training[m]['alpha']}", f"{training[m]['learning_rate']:.0e}", training[m]["stopped_epoch"], training[m]["best_epoch"], training[m]["metric_for_best_model"], "MixedVal2500-v2 schemaheaderfix", training[m]["duration_human_readable"], training[m]["peak_memory_allocated"]] for m in MODEL_ORDER],
    ), "", "All three lines selected epoch 1 by Full-Chat `eval_loss` and stopped after epoch 3 under the configured early-stopping rule. The gains are not attributable to more selected epochs. Runtime and memory were recorded on an NVIDIA L40S, but end-to-end efficiency comparisons remain descriptive because sequence packing and model architecture differ.", ""]
    lines += ["## Qwen 2B Generation-Limit Sensitivity", "", "The official cross-model mainline remains `max_new_tokens=256`. Across the eight Qwen 2B Base conditions, 2,215 outputs hit 256; all 2,215 reached 512 again, none terminated regularly before 512, and no additional execution match resulted. H1 (merely binding limit) is not supported; H2 (persistent repetitive/degenerative generation) is supported. No additional token threshold should be optimized on Spider Dev.", ""]
    lines += ["## Difficulty", "", "**DIFFICULTY ANALYSIS NOT AVAILABLE.** The frozen 1,032-case testcase artifact does not contain an authoritative Spider easy/medium/hard/extra mapping. No labels were reconstructed.", ""]
    lines += ["## Research Questions", "", "- **RQ1:** JA. LoRA improves zero-shot EMA in all three lines.", "- **RQ2:** JA. All 24 LoRA deltas are positive.", "- **RQ3:** Qwen 2B and Qwen 9B retain 8/8 Holm-significant gains; Llama retains 4/8.", "- **RQ4:** NEIN. No few-shot method gives a consistent, robust general gain.", "- **RQ5:** JA. Qwen 9B LoRA has the highest absolute zero-shot EMA (74.52%).", "- **RQ6:** JA. Qwen 2B has the largest zero-shot LoRA gain (+14.63 pp).", "- **RQ7:** JA. Gate 0.85 is predominantly zero-shot fallback (975/1,032).", "- **RQ8:** NEIN. Structure does not consistently improve EMA.", "- **RQ9:** NEIN. Raising Qwen 2B Base to 512 yielded no new match and all capped outputs capped again.", "- **RQ10:** NEIN. No required model run is missing.", ""]
    lines += ["## Thesis-Ready Supported Statements", ""] + [f"{i}. {text}" for i, text in enumerate(SUPPORTED, 1)] + [""]
    lines += ["## Statements Not Supported", ""] + [f"{i}. {text}" for i, text in enumerate(UNSUPPORTED, 1)] + [""]
    lines += ["## Limitations", ""] + [f"{i}. {text}" for i, text in enumerate(LIMITATIONS, 1)] + [""]
    lines += ["## Decision", "", "- Experiments still missing: **NEIN**", "- Rerun required: **NEIN**", "- Official mainline: the 48 runs at `max_new_tokens=256`", "- Recommended next step: transfer the frozen tables, figures, and qualified statements into the methodology, results, and discussion chapters. No additional model runs are required.", ""]
    lines += ["## Read-Only Confirmation", "", "No training, generation, model/adapter/tokenizer/BGE load, download, config edit, result edit, or full evaluation was performed. SQLite databases were opened with `mode=ro` and `PRAGMA query_only=ON`. Only the additive files listed in the synthesis manifest were written.", ""]
    if warnings:
        unique = sorted(set(warnings))
        lines += ["## Documentation Warnings", ""] + [f"- {value}" for value in unique] + [""]
    return "\n".join(lines)


def main() -> None:
    for path in list(OUT.values()) + [stem.with_suffix(suffix) for stem in PLOTS.values() for suffix in (".png", ".pdf")]:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing output: {path}")
    source_hashes = {}
    for name, (path, expected) in SOURCES.items():
        actual = sha256(ROOT / path)
        if actual != expected:
            raise RuntimeError(f"Authoritative source hash mismatch: {path}")
        source_hashes[name] = {"path": path, "sha256": actual}
    support_hashes = {name: {"path": path, "sha256": sha256(ROOT / path)} for name, path in SUPPORT_SOURCES.items()}
    records = load_run_inventory()
    errors, warnings = validate_and_enrich(records)
    if errors:
        raise RuntimeError("\n".join(errors))
    retrieval = validate_retrieval(records)
    if any(value["status"] != "PASS" for value in retrieval.values()):
        raise RuntimeError(f"Retrieval identity failure: {retrieval}")
    sample = sqlite_sample_check(records)
    if sample["status"] != "PASS":
        raise RuntimeError(f"SQLite sample failure: {sample}")
    stats = compute_statistics(records)
    result_rows = build_result_rows(records)
    completion = build_completion_rows(records)
    training = training_summary()
    comparison = comparability_rows()

    write_csv_new(OUT["results"], result_rows)
    write_csv_new(OUT["zero"], stats["zero"] + stats["cross_zero"]["base"] + stats["cross_zero"]["lora_v2"])
    write_csv_new(OUT["gains"], stats["base_lora"])
    write_csv_new(OUT["fewshot"], stats["fewshot"])
    write_csv_new(OUT["heterogeneity"], stats["heterogeneity"])
    write_csv_new(OUT["completion"], completion)
    write_csv_new(OUT["comparability"], comparison)
    write_text_new(OUT["thesis"], build_thesis_tables(result_rows, stats, completion, training))
    make_plots(result_rows, stats, completion)

    audit_text = make_audit(records, result_rows, stats, completion, retrieval, sample, training, warnings)
    write_text_new(OUT["audit"], audit_text)

    generated_without_manifest = [OUT[key] for key in OUT if key != "manifest"] + [stem.with_suffix(suffix) for stem in PLOTS.values() for suffix in (".png", ".pdf")]
    run_manifest = []
    for record in records:
        run_manifest.append({key: record.get(key) for key in [
            "model", "role", "condition", "run_id", "csv_path", "csv_sha256", "metadata_path", "metadata_sha256", "trace_path", "trace_sha256", "log_path", "log_sha256", "config_path", "config_sha256", "case_count", "unique_case_count", "case_ids_sha256", "technical_checks", "metric_mismatches", "metrics"
        ]})
    manifest = {
        "audit_status": "PASS MIT METHODISCHEN EINSCHRANKUNGEN",
        "date": "2026-07-16",
        "classification": "final read-only cross-model results and synthesis audit",
        "read_only": {"training_started": False, "evaluation_started": False, "model_loaded": False, "adapter_loaded": False, "tokenizer_loaded": False, "bge_loaded": False, "network_used": False, "source_artifacts_modified": False},
        "scope": {"model_lines": 3, "roles": 2, "conditions": 8, "runs": 48, "cases_per_run": N, "total_predictions": 48 * N},
        "authoritative_sources": source_hashes,
        "supporting_sources": support_hashes,
        "analysis_script": {
            "path": "scripts/analyze_cross_model_complete_8x8_synthesis.py",
            "sha256": sha256(Path(__file__)),
        },
        "model_and_adapter_provenance": MODEL_INFO,
        "testset": {"path": "data/testcases.jsonl", "sha256": "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce", "rows": N, "case_order_sha256": records[0]["case_ids_sha256"]},
        "common_method": {"max_new_tokens": 256, "decoding": "greedy", "generation_batch_size": 1, "extractor": "sql_first_statement_only", "system_prompt": "sqlctx_anti_overjoin", "system_prompt_sha256": "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e", "qwen_prompt_format": "qwen_sqlctx_chatml", "llama_prompt_format": "llama32_instruct_native_chat", "prompt_truncations": 0},
        "retrieval": {"index": "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15", "index_sha256": "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc", "metadata_sha256": "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55", "pool_size": 6960, "embedding_model": "BAAI/bge-large-en-v1.5", "static_demo": "SPIDER_TRAIN_001657", "static_resource_sha256": "7c4735d7ba31ebd448cd0b94fd4c63a80c3e50f115d0fdd39e652ae0f1be1857", "gate_counts": {"top1_070": [634, 398], "top1_085": [57, 975], "structure_070": [613, 419], "structure_085": [57, 975]}, "identity_checks": retrieval, "leakage_status": "PASS"},
        "runs": run_manifest,
        "hash_integrity": "PASS",
        "metric_reproduction": {"status": "PASS", "mismatches": 0},
        "sqlite_sample_rescoring": sample,
        "zero_shot_comparisons": stats["zero"],
        "cross_model_zero_shot": stats["cross_zero"],
        "lora_gains": stats["base_lora"],
        "gain_heterogeneity": stats["heterogeneity"],
        "fewshot_effects": stats["fewshot"],
        "interactions": stats["interactions"],
        "completion_diagnostics": completion,
        "qwen2b_sensitivity": {"official_mainline": 256, "sensitivity_limit": 512, "capped_at_256": 2215, "reached_512_again": 2215, "terminated_before_512": 0, "additional_execution_matches": 0, "h1_binding_limit": "NICHT GESTUTZT", "h2_repetition": "GESTUTZT"},
        "training": training,
        "comparability_classes": {"within_line": "A", "qwen2b_vs_qwen9b": "B+ project-specific descriptive", "qwen_vs_llama": "B"},
        "difficulty_analysis": {"status": "DIFFICULTY ANALYSIS NOT AVAILABLE", "reason": "No authoritative labels in frozen testcase artifact"},
        "supported_statements": SUPPORTED,
        "unsupported_statements": UNSUPPORTED,
        "limitations": LIMITATIONS,
        "experiment_completion": {"experiments_still_missing": False, "rerun_required": False, "recommended_next_step": "Transfer final tables, figures, and qualified statements into the thesis; no additional model runs."},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": SEED, "confidence": 0.95},
        "holm_families": {"within_line_base_vs_lora": "8 per model line", "fewshot_vs_zero": "7 per model line and role", "zero_shot_cross_model": "3 per role"},
        "new_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in generated_without_manifest],
        "manifest_self_hash": None,
        "manifest_self_hash_note": "Not embedded because a file cannot contain its own stable SHA256.",
    }
    write_text_new(OUT["manifest"], json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({
        "status": manifest["audit_status"],
        "runs": len(records),
        "predictions": 48 * N,
        "sqlite_sample": sample,
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in OUT.items()},
        "plots": {key: [str(stem.with_suffix(s).relative_to(ROOT)) for s in (".png", ".pdf")] for key, stem in PLOTS.items()},
        "manifest_sha256": sha256(OUT["manifest"]),
    }, indent=2))


if __name__ == "__main__":
    main()
