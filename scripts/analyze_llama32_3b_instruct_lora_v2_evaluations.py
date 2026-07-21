#!/usr/bin/env python3
"""Reproducible read-only audit analysis for completed Llama 3.2 LoRA-v2 runs.

The script reads immutable evaluation artifacts and Spider SQLite databases. It
creates new derived audit files with exclusive creation and never loads an LLM,
LoRA adapter, tokenizer, or embedding model.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
INDEX_DIR = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
STATIC_RESOURCE = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
ADAPTER_ROOT = ROOT / "adapters/llama32_3b_instruct/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
ADAPTER_MODEL = ADAPTER_ROOT / "adapter_model.safetensors"
BEST_CHECKPOINT = ADAPTER_ROOT / "checkpoints/checkpoint-509"

OUT_SUMMARY = ROOT / "audits/derived/llama32_3b_instruct_lora_v2_evaluation_summary_20260715.json"
OUT_CASES = ROOT / "audits/derived/llama32_3b_instruct_lora_v2_case_comparison_20260715.csv"
OUT_BASE_STATS = ROOT / "audits/derived/llama32_3b_instruct_base_vs_lora_paired_statistics_20260715.csv"
OUT_FEWSHOT_STATS = ROOT / "audits/derived/llama32_3b_instruct_lora_fewshot_paired_statistics_20260715.csv"
OUT_SIMILARITY = ROOT / "audits/derived/llama32_3b_instruct_lora_similarity_bin_analysis_20260715.csv"
OUT_QWEN = ROOT / "audits/derived/llama32_3b_instruct_qwen_cross_model_comparison_20260715.csv"

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
ADAPTER_ALIAS = "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
ADAPTER_SHA256 = "fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d"
TEST_SHA256 = "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce"
INDEX_SHA256 = "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc"
INDEX_METADATA_SHA256 = "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55"
STATIC_SHA256 = "7c4735d7ba31ebd448cd0b94fd4c63a80c3e50f115d0fdd39e652ae0f1be1857"
CHAT_TEMPLATE_SHA256 = "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"
SYSTEM_PROMPT_SHA256 = "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e"
RUNNER_SHA256 = "a37286649920f4224999b5184e6117ea31f24968ad2c353ff338397c99a7a3c9"
BOOTSTRAP_SEED = 20260715
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

DISPLAY = {
    "zero_shot": "Zero Shot",
    "top1": "Dynamic Top-1",
    "top1_gate070": "Top-1 Gate 0.70",
    "top1_gate085": "Top-1 Gate 0.85",
    "static_seed42": "Static k=1 Seed 42",
    "structure": "Structure Top-10 v2",
    "structure_gate070": "Structure Gate 0.70",
    "structure_gate085": "Structure Gate 0.85",
}

LLAMA_LORA_RUNS = {
    "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_215808",
    "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_222657",
    "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_225553",
    "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_232516",
    "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_235408",
    "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_002034",
    "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_005046",
    "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260715_012027",
}

LLAMA_BASE_RUNS = {
    "zero_shot": "run_base_20260714_162526",
    "top1": "run_base_20260714_164116",
    "top1_gate070": "run_base_20260714_165432",
    "top1_gate085": "run_base_20260714_170748",
    "static_seed42": "run_base_20260714_172224",
    "structure": "run_base_20260714_173302",
    "structure_gate070": "run_base_20260714_174639",
    "structure_gate085": "run_base_20260714_180015",
}

CONFIG_SHA256 = {
    "zero_shot": "8d0de1fa0d169924bfadd3ae7584bd42359f6ebcb2008827f936a57732f79be0",
    "top1": "5c02652cb2b51211224a0321db890f441c4786f778a8c1032633bb6d5aa69e79",
    "top1_gate070": "83e70474d632aa71f108e62a6f606d2937db7ca1823df313dc90ef12bd0b2ee4",
    "top1_gate085": "3a1e79753c7477ef4a5633dd1a40b132291fa2d6a3a979bbc6f96e7f0b4693d5",
    "static_seed42": "b998b34446cc11883ccb94e9113187ae7230c99526e6e5b415ecb11b7d5ae277",
    "structure": "608399e5b47e85609fc2082a0a92f149e3cf440b693aa9832ba0cb6f4e7448b9",
    "structure_gate070": "071fd48d9f1b4d90878ae3d7847be1eb0b73eeec8a10aca2dbaf4ac8c30db580",
    "structure_gate085": "adeebe74aad37955ee1188e78de17db1d8aa4b5e5b621a386e7d95f81074fdd8",
}

QWEN_LORA_RUNS = {
    "qwen35_2b_lora_v2": {
        "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_083452",
        "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_091541",
        "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_095759",
        "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_103913",
        "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_112013",
        "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_120540",
        "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_124740",
        "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260714_132822",
    },
    "qwen35_9b_lora_v2": {
        "zero_shot": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_120126",
        "top1": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_125744",
        "top1_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_135127",
        "top1_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_144738",
        "static_seed42": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_154004",
        "structure": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_163754",
        "structure_gate070": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_202137",
        "structure_gate085": "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5_20260713_211349",
    },
}

QWEN_BASE_RUNS = {
    "qwen35_2b_base": {
        "zero_shot": "run_base_20260627_211410",
        "top1": "run_base_20260712_171240",
        "top1_gate070": "run_base_20260712_183739",
        "top1_gate085": "run_base_20260712_194508",
        "structure": "run_base_20260712_202105",
    },
    "qwen35_9b_base": {
        "zero_shot": "run_base_20260624_221131",
        "top1": "run_base_20260712_143438",
        "top1_gate070": "run_base_20260712_150257",
        "top1_gate085": "run_base_20260712_153056",
        "structure_gate070": "run_base_20260712_160614",
        "structure_gate085": "run_base_20260712_163705",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def write_csv_new(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing derived CSV is not byte-identical: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def as_bool(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true"})


def sql_tokens(sql: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(
            r"'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<=|>=|<>|!=|==|[-+*/(),.;=<>]",
            sql,
        )
    ]


def normalized_sql(sql: str) -> str:
    return " ".join(sql_tokens(sql))


def char_accuracy(pred: str, gold: str) -> float:
    length = max(len(pred), len(gold))
    if length == 0:
        return 1.0
    return sum(i < len(pred) and i < len(gold) and pred[i] == gold[i] for i in range(length)) / length


def token_accuracy(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = sql_tokens(pred), sql_tokens(gold)
    length = max(len(pred_tokens), len(gold_tokens))
    if length == 0:
        return 1.0
    return sum(
        i < len(pred_tokens) and i < len(gold_tokens) and pred_tokens[i] == gold_tokens[i]
        for i in range(length)
    ) / length


def quantiles(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "min": int(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def exact_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "mcnemar_p") -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1][p_key])
    running = 0.0
    count = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row[p_key]))
        running = max(running, adjusted)
        rows[original_index]["holm_adjusted_p"] = running
        rows[original_index]["significant_unadjusted_0_05"] = float(row[p_key]) < 0.05
        rows[original_index]["significant_holm_0_05"] = running < 0.05


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = b.astype(float) - a.astype(float)
    estimates: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_RESAMPLES, 250):
        count = min(250, BOOTSTRAP_RESAMPLES - sum(len(block) for block in estimates))
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
    values = np.concatenate(estimates)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    *,
    comparison: str,
    condition: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    both_correct = int(np.sum((a == 1) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))
    a_ema, b_ema = float(a.mean()), float(b.mean())
    ci_low, ci_high = paired_bootstrap_difference(a, b, rng=rng)
    relative_error_reduction = (b_ema - a_ema) / (1.0 - a_ema) if a_ema < 1.0 else None
    return {
        "comparison": comparison,
        "condition": condition,
        "condition_label": DISPLAY.get(condition, condition),
        "n": len(a),
        "a_correct": int(a.sum()),
        "b_correct": int(b.sum()),
        "a_ema": a_ema,
        "b_ema": b_ema,
        "delta": b_ema - a_ema,
        "delta_percentage_points": 100.0 * (b_ema - a_ema),
        "relative_error_reduction": relative_error_reduction,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "n01_a_wrong_b_correct": n01,
        "n10_a_correct_b_wrong": n10,
        "net_additional_correct": n01 - n10,
        "mcnemar_p": exact_mcnemar_p(n01, n10),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def run_paths(run_id: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "results" / f"{run_id}.csv",
        ROOT / "results" / f"{run_id}_metadata.json",
        ROOT / "results/retrieval_traces" / f"{run_id}_retrieval_traces.jsonl",
    )


def trace_signature(row: dict[str, Any]) -> tuple[str | None, float | None]:
    ids = row.get("retrieved_ids") or []
    demo = row.get("selected_example_id") or (ids[0] if ids else None)
    score = row.get("gate_score")
    if score is None:
        score = row.get("retrieval_similarity")
    if score is None:
        scores = row.get("retrieved_scores") or []
        score = scores[0] if scores else None
    return demo, float(score) if score is not None else None


def find_log(config_path: Path) -> Path:
    candidates = sorted((ROOT / "logs").rglob(f"{config_path.stem}.log"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one log for {config_path}, found {len(candidates)}")
    return candidates[0]


def validate_config(condition: str, config: dict[str, Any]) -> None:
    required = {
        "llm": "llama32_3b_instruct",
        "adapter": ADAPTER_ALIAS,
        "testcases_path": "data/testcases_spider_dev_full.jsonl",
        "max_test_samples": None,
        "max_input_tokens": 2048,
        "max_new_tokens": 256,
        "generation_batch_size": 1,
        "compute_perplexity": False,
        "allow_overlap": False,
        "same_db_only": False,
        "prompt_format": "llama32_instruct_native_chat",
        "system_prompt_variant": "sqlctx_anti_overjoin",
        "extractor_mode": "sql_first_statement_only",
    }
    failures = [key for key, expected in required.items() if config.get(key) != expected]
    if condition == "zero_shot":
        failures += [key for key, expected in {"prompt_tuning": "none", "k": 0}.items() if config.get(key) != expected]
    else:
        failures += [key for key, expected in {"k": 1, "fewshot_example_schema_mode": "full", "fewshot_example_mode": "schema_with_rules"}.items() if config.get(key) != expected]
    if condition == "static_seed42":
        if config.get("prompt_tuning") != "static_fewshot" or config.get("retrieval_method") != "static_seeded":
            failures.append("static_method")
    elif condition != "zero_shot":
        if config.get("prompt_tuning") != "dynamic_fewshot" or config.get("retrieval_method") != "sentence_transformer_faiss":
            failures.append("dynamic_method")
        if config.get("retrieval_index_path") != str(INDEX_DIR.relative_to(ROOT)):
            failures.append("retrieval_index_path")
    if condition.startswith("structure"):
        expected = {"retrieval_rerank_method": "structure_topk_v2", "retrieval_rerank_top_n": 10, "retrieval_structure_bonus_max": 0.08}
        failures += [key for key, value in expected.items() if config.get(key) != value]
    if condition == "top1" and config.get("retrieval_rerank_method", "none") not in {None, "none"}:
        failures.append("top1_reranker")
    thresholds = {"top1_gate070": 0.7, "top1_gate085": 0.85, "structure_gate070": 0.7, "structure_gate085": 0.85}
    if condition in thresholds:
        if config.get("fewshot_gate_enabled") is not True or config.get("fewshot_gate_threshold") != thresholds[condition]:
            failures.append("gate")
    elif config.get("fewshot_gate_enabled") not in {False, None}:
        failures.append("unexpected_gate")
    if failures:
        raise RuntimeError(f"Config validation failed for {condition}: {failures}")


def output_issue_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        raw, pred = row["raw_output"], row["pred_sql"]
        counts["empty_raw_output"] += not raw.strip()
        counts["empty_extracted_sql"] += not pred.strip()
        counts["nonempty_raw_but_empty_extraction"] += bool(raw.strip()) and not pred.strip()
        counts["think_marker"] += "<think>" in raw.lower()
        counts["markdown_fence"] += "```" in raw
        counts["missing_semicolon_in_extracted_sql"] += bool(pred.strip()) and not pred.rstrip().endswith(";")
        counts["multiple_statement_terminators"] += pred.count(";") > 1
        counts["text_before_sql"] += bool(re.match(r"(?is)^\s*(?!select\b|with\b|pragma\b|insert\b|update\b|delete\b)", raw))
        after = raw.strip()
        counts["text_after_extracted_sql"] += bool(pred.strip()) and not after.endswith(pred.strip())
    return dict(counts)


def audit_run(condition: str, run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Missing result artifact for {run_id}")
    rows, metadata = load_csv(csv_path), load_json(metadata_path)
    config_path = ROOT / metadata["run_config_path"]
    config = load_json(config_path)
    log_path = find_log(config_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    validate_config(condition, config)
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032:
        raise RuntimeError(f"Incomplete run: {run_id}")
    if len({row["id"] for row in rows}) != 1032 or [row["id"] for row in rows] != [row["id"] for row in tests]:
        raise RuntimeError(f"Case alignment failure: {run_id}")
    for row, test in zip(rows, tests):
        if (row["db_id"], row["question"], row["gold_sql"]) != (test["db_id"], test["question"], test["gold_sql"]):
            raise RuntimeError(f"Test content mismatch in {run_id}: {row['id']}")

    expected_checks = {
        "config_sha256": sha256(config_path) == CONFIG_SHA256[condition] == metadata["provenance"]["config_sha256"],
        "model_id": metadata.get("run_model_id") == MODEL_ID,
        "model_revision": metadata.get("run_model_revision") == MODEL_REVISION,
        "adapter_alias": metadata.get("run_adapter") == ADAPTER_ALIAS,
        "test_sha256": metadata["provenance"].get("testcases_sha256") == TEST_SHA256,
        "chat_template": metadata.get("run_tokenizer_chat_template_sha256") == CHAT_TEMPLATE_SHA256,
        "system_prompt": metadata.get("run_system_prompt_sha256") == SYSTEM_PROMPT_SHA256,
        "runner": metadata["provenance"].get("code_sha256", {}).get("runner") == RUNNER_SHA256,
        "generation_limits": metadata.get("run_max_input_tokens") == 2048 and metadata.get("run_max_new_tokens") == 256,
        "batch": metadata.get("run_generation_batch_size") == 1,
        "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
        "native_prompt": metadata.get("run_prompt_format") == "llama32_instruct_native_chat",
        "no_limit": metadata.get("run_max_test_samples") == "",
        "log_no_error": not re.search(r"Traceback|CUDA out of memory|\bERROR\b", log_text),
        "resolved_adapter": str(ADAPTER_ROOT.resolve()) in log_text,
        "native_assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\\n\\n" in log_text,
        "greedy": True,
    }
    # The greedy flag is hard-coded in the audited runner; avoid making success depend on log wording.
    expected_checks["greedy"] = metadata.get("run_generation_batch_size") == 1
    if not all(expected_checks.values()):
        raise RuntimeError(f"Provenance failure {condition}: {[key for key, value in expected_checks.items() if not value]}")

    exec_values = np.asarray([as_bool(row["exec_match"]) for row in rows], dtype=np.int8)
    pred_ok = np.asarray([as_bool(row["pred_ok"]) for row in rows], dtype=np.int8)
    string_exact = np.asarray([int(row["pred_sql"] == row["gold_sql"]) for row in rows], dtype=np.int8)
    normalized_exact = np.asarray([int(normalized_sql(row["pred_sql"]) == normalized_sql(row["gold_sql"])) for row in rows], dtype=np.int8)
    char_values = np.asarray([char_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    token_values = np.asarray([token_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    stored_metric_mismatches = {
        "string_exact": sum(int(row["string_exact"]) != value for row, value in zip(rows, string_exact)),
        "normalized_exact": sum(int(row["normalized_exact"]) != value for row, value in zip(rows, normalized_exact)),
        "char_accuracy": sum(abs(float(row["char_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, char_values)),
        "token_accuracy": sum(abs(float(row["token_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, token_values)),
    }
    if any(stored_metric_mismatches.values()):
        raise RuntimeError(f"CSV metric mismatch {condition}: {stored_metric_mismatches}")
    reproduced = {
        "execution_match_accuracy": float(exec_values.mean()),
        "execution_success_rate": float(pred_ok.mean()),
        "string_exact_match": float(string_exact.mean()),
        "normalized_exact_match": float(normalized_exact.mean()),
        "char_accuracy_avg": float(char_values.mean()),
        "token_accuracy_avg": float(token_values.mean()),
    }
    for key, value in reproduced.items():
        if abs(float(metadata[key]) - value) > 5.1e-10:
            raise RuntimeError(f"Aggregate metric mismatch {condition} {key}: {metadata[key]} != {value}")

    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    completion_tokens = [int(row["completion_tokens"]) for row in rows]
    total_tokens = [int(row["total_tokens"]) for row in rows]
    finite_columns = ["char_accuracy", "token_accuracy", "generation_time_seconds", "tokens_per_second"]
    nonfinite_count = sum(not math.isfinite(float(row[column])) for row in rows for column in finite_columns)
    if nonfinite_count:
        raise RuntimeError(f"Non-finite CSV metrics in {condition}: {nonfinite_count}")

    traces: list[dict[str, Any]] = []
    trace_summary: dict[str, Any] | None = None
    if condition != "zero_shot":
        if not trace_path.is_file():
            raise RuntimeError(f"Missing retrieval trace for {condition}")
        traces = load_jsonl(trace_path)
        if len(traces) != 1032 or [row["id"] for row in traces] != [row["id"] for row in tests]:
            raise RuntimeError(f"Trace alignment failure: {condition}")
        if any(not row.get("retrieval_success") or row.get("leakage_status") != "pass" for row in traces):
            raise RuntimeError(f"Retrieval/leakage failure: {condition}")
        signatures = [trace_signature(row) for row in traces]
        scores = [score for _, score in signatures if score is not None]
        gate_counts = Counter(row.get("gate_decision") for row in traces if row.get("gate_decision") is not None)
        threshold = config.get("fewshot_gate_threshold")
        gate_mismatches = 0
        if threshold is not None:
            for trace, (_, score) in zip(traces, signatures):
                expected = "fewshot" if score is not None and score >= threshold else "zero_shot"
                gate_mismatches += trace.get("gate_decision") != expected
        trace_summary = {
            "path": str(trace_path.relative_to(ROOT)),
            "sha256": sha256(trace_path),
            "rows": len(traces),
            "retrieval_success": sum(bool(row.get("retrieval_success")) for row in traces),
            "leakage_pass": sum(row.get("leakage_status") == "pass" for row in traces),
            "unique_selected_demo_ids": len({demo for demo, _ in signatures}),
            "mean_similarity": float(np.mean(scores)) if scores else None,
            "min_similarity": min(scores) if scores else None,
            "max_similarity": max(scores) if scores else None,
            "gate_counts": dict(gate_counts),
            "gate_decision_mismatches": gate_mismatches,
            "target_id_as_demo": sum(demo == test["id"] for (demo, _), test in zip(signatures, tests)),
        }
        if gate_mismatches or trace_summary["target_id_as_demo"]:
            raise RuntimeError(f"Trace semantic mismatch in {condition}")

    return {
        "condition": condition,
        "run_id": run_id,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha256(metadata_path),
        "trace_path": str(trace_path.relative_to(ROOT)) if trace_path.is_file() else None,
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else None,
        "log_path": str(log_path.relative_to(ROOT)),
        "log_sha256": sha256(log_path),
        "config_path": metadata["run_config_path"],
        "config_sha256": sha256(config_path),
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
        "rows": rows,
        "metadata": metadata,
        "traces": traces,
        "exec": exec_values,
        "pred_ok": pred_ok,
        "checks": expected_checks,
        "metrics": {
            "execution_match_count": int(exec_values.sum()),
            "ema": reproduced["execution_match_accuracy"],
            "execution_success_count": int(pred_ok.sum()),
            "esr": reproduced["execution_success_rate"],
            "string_exact_count": int(string_exact.sum()),
            "string_exact": reproduced["string_exact_match"],
            "normalized_exact_count": int(normalized_exact.sum()),
            "normalized_exact": reproduced["normalized_exact_match"],
            "char_accuracy": reproduced["char_accuracy_avg"],
            "token_accuracy": reproduced["token_accuracy_avg"],
            "runtime_seconds": float(metadata["duration_seconds"]),
            "seconds_per_case": float(metadata["duration_seconds"]) / len(rows),
            "prompt_tokens": quantiles(prompt_tokens),
            "completion_tokens": quantiles(completion_tokens),
            "total_tokens": {"min": min(total_tokens), "mean": float(np.mean(total_tokens)), "max": max(total_tokens)},
            "prompts_over_2048": sum(value > 2048 for value in prompt_tokens),
            "completions_at_256": sum(value == 256 for value in completion_tokens),
            "nonfinite_numeric_values": nonfinite_count,
            "output_issues": output_issue_counts(rows),
        },
        "reproduced_metrics": reproduced,
        "stored_metric_mismatches": stored_metric_mismatches,
        "trace_summary": trace_summary,
    }


def load_reference(run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    rows, metadata = load_csv(csv_path), load_json(metadata_path)
    if len(rows) != 1032 or [row["id"] for row in rows] != [row["id"] for row in tests]:
        raise RuntimeError(f"Reference run alignment failure: {run_id}")
    return {
        "run_id": run_id,
        "rows": rows,
        "metadata": metadata,
        "exec": np.asarray([as_bool(row["exec_match"]) for row in rows], dtype=np.int8),
        "trace": load_jsonl(trace_path) if trace_path.is_file() else [],
        "csv_path": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha256(metadata_path),
    }


def compare_trace_sets(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    same_demo = same_score = 0
    deltas: list[float] = []
    for left_row, right_row in zip(left, right):
        left_demo, left_score = trace_signature(left_row)
        right_demo, right_score = trace_signature(right_row)
        same_demo += left_demo == right_demo
        if left_score is not None and right_score is not None:
            delta = abs(left_score - right_score)
            deltas.append(delta)
            same_score += delta <= 1e-12
    return {
        "same_demo_ids": same_demo,
        "different_demo_ids": len(left) - same_demo,
        "same_scores": same_score,
        "different_scores": len(deltas) - same_score,
        "mean_absolute_score_delta": float(np.mean(deltas)) if deltas else None,
        "max_absolute_score_delta": max(deltas) if deltas else None,
    }


def gate_reference_check(runs: dict[str, dict[str, Any]], gated: str, fewshot: str) -> dict[str, Any]:
    zero_rows, few_rows, gate_rows, traces = (
        runs["zero_shot"]["rows"],
        runs[fewshot]["rows"],
        runs[gated]["rows"],
        runs[gated]["traces"],
    )
    prompt_matches = raw_matches = pred_matches = 0
    accepted, rejected = [], []
    for zero, few, gate, trace in zip(zero_rows, few_rows, gate_rows, traces):
        decision = trace["gate_decision"]
        reference = few if decision == "fewshot" else zero
        prompt_matches += gate["prompt_tokens"] == reference["prompt_tokens"]
        raw_matches += gate["raw_output"] == reference["raw_output"]
        pred_matches += gate["pred_sql"] == reference["pred_sql"]
        _, score = trace_signature(trace)
        (accepted if decision == "fewshot" else rejected).append(score)
    return {
        "cases": len(gate_rows),
        "prompt_token_matches_selected_reference": prompt_matches,
        "raw_output_matches_selected_reference": raw_matches,
        "pred_sql_matches_selected_reference": pred_matches,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_mean_similarity": float(np.mean(accepted)),
        "rejected_mean_similarity": float(np.mean(rejected)),
    }


def normalize_overlap_question(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_overlap_sql(value: str) -> str:
    value = value.strip()
    if value and not value.endswith(";"):
        value += ";"
    return " ".join(value.lower().split())


def retrieval_overlap_audit(tests: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = load_jsonl(INDEX_DIR / "metadata.jsonl")
    if len(metadata) != 6960:
        raise RuntimeError("Unexpected retrieval pool size")
    test_ids = {row["id"] for row in tests}
    test_questions = {normalize_overlap_question(row["question"]) for row in tests}
    test_sql = {normalize_overlap_sql(row["gold_sql"]) for row in tests}
    test_pairs = {(normalize_overlap_question(row["question"]), normalize_overlap_sql(row["gold_sql"])) for row in tests}
    return {
        "rows": len(metadata),
        "index_sha256": sha256(INDEX_DIR / "index.faiss"),
        "metadata_sha256": sha256(INDEX_DIR / "metadata.jsonl"),
        "manifest_sha256": sha256(INDEX_DIR / "manifest.json"),
        "id_overlap": sum(row.get("id") in test_ids for row in metadata),
        "question_overlap": sum(normalize_overlap_question(row.get("question", "")) in test_questions for row in metadata),
        "sql_overlap": sum(normalize_overlap_sql(row.get("gold_sql", "")) in test_sql for row in metadata),
        "pair_overlap": sum((normalize_overlap_question(row.get("question", "")), normalize_overlap_sql(row.get("gold_sql", ""))) in test_pairs for row in metadata),
    }


def import_runner() -> Any:
    src_dir = ROOT / "src"
    sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("batch_run_audit", src_dir / "06_batch_run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import audited runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def typed_row_counter(rows: list[tuple[Any, ...]]) -> Counter[tuple[tuple[str, str], ...]]:
    return Counter(tuple((type(value).__name__, repr(value)) for value in row) for row in rows)


def independent_execute(conn: sqlite3.Connection, sql: str) -> tuple[bool, Counter[Any] | None, str | None]:
    try:
        rows = conn.execute(sql).fetchall()
        return True, typed_row_counter(rows), None
    except Exception as exc:  # exact error text is diagnostic only
        return False, None, repr(exc)


def execution_rescore(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runner = import_runner()
    result: dict[str, Any] = {}
    for condition, run in runs.items():
        connections: dict[str, sqlite3.Connection] = {}
        path1_esr_mismatch: list[str] = []
        path1_ema_mismatch: list[str] = []
        path2_esr_mismatch: list[str] = []
        path2_ema_mismatch: list[str] = []
        path_disagreement: list[str] = []
        gold_failures: list[str] = []
        try:
            for row in run["rows"]:
                db_path = (ROOT / row["db_path"]).resolve()
                key = str(db_path)
                if key not in connections:
                    uri = f"file:{db_path}?mode=ro"
                    connection = sqlite3.connect(uri, uri=True)
                    connection.execute("PRAGMA query_only = ON")
                    connections[key] = connection
                conn = connections[key]
                gold1 = runner.run_sql(conn, row["gold_sql"])
                pred1 = runner.run_sql(conn, row["pred_sql"]) if row["pred_sql"] else runner.ExecResult(False, None, "No SQL extracted")
                ok1 = bool(pred1.ok)
                match1 = bool(runner.execution_match(pred1, gold1))

                gold2_ok, gold2_rows, _ = independent_execute(conn, row["gold_sql"])
                if row["pred_sql"]:
                    pred2_ok, pred2_rows, _ = independent_execute(conn, row["pred_sql"])
                else:
                    pred2_ok, pred2_rows = False, None
                match2 = bool(pred2_ok and gold2_ok and pred2_rows == gold2_rows)
                stored_ok, stored_match = bool(as_bool(row["pred_ok"])), bool(as_bool(row["exec_match"]))
                if not gold2_ok:
                    gold_failures.append(row["id"])
                if ok1 != stored_ok:
                    path1_esr_mismatch.append(row["id"])
                if match1 != stored_match:
                    path1_ema_mismatch.append(row["id"])
                if pred2_ok != stored_ok:
                    path2_esr_mismatch.append(row["id"])
                if match2 != stored_match:
                    path2_ema_mismatch.append(row["id"])
                if ok1 != pred2_ok or match1 != match2:
                    path_disagreement.append(row["id"])
        finally:
            for connection in connections.values():
                connection.close()
        result[condition] = {
            "existing_runner_path": {
                "esr_mismatch_count": len(path1_esr_mismatch),
                "ema_mismatch_count": len(path1_ema_mismatch),
                "case_ids": sorted(set(path1_esr_mismatch + path1_ema_mismatch)),
            },
            "independent_sqlite_path": {
                "esr_mismatch_count": len(path2_esr_mismatch),
                "ema_mismatch_count": len(path2_ema_mismatch),
                "case_ids": sorted(set(path2_esr_mismatch + path2_ema_mismatch)),
            },
            "path_disagreement_count": len(path_disagreement),
            "path_disagreement_case_ids": path_disagreement,
            "gold_execution_failure_count": len(gold_failures),
            "gold_execution_failure_case_ids": gold_failures,
        }
    return result


def classify_sql_error(row: dict[str, str]) -> str:
    pred, gold, error = row["pred_sql"].lower(), row["gold_sql"].lower(), row["pred_error"].lower()
    if not row["pred_sql"].strip():
        return "empty_extraction"
    for needle, label in [
        ("no such table", "nonexistent_table"),
        ("no such column", "nonexistent_column"),
        ("syntax error", "syntax_error"),
        ("ambiguous column", "ambiguous_column"),
        ("misuse of aggregate", "aggregate_execution_error"),
    ]:
        if needle in error:
            return label
    pred_joins, gold_joins = len(re.findall(r"\bjoin\b", pred)), len(re.findall(r"\bjoin\b", gold))
    if pred_joins > gold_joins:
        return "unnecessary_join"
    if pred_joins < gold_joins:
        return "missing_join"
    feature_patterns = [
        (r"\bdistinct\b", "distinct_mismatch"),
        (r"\b(count|sum|avg|min|max)\s*\(", "aggregation_mismatch"),
        (r"\bgroup\s+by\b|\bhaving\b", "group_by_having_mismatch"),
        (r"\border\s+by\b|\blimit\b", "order_by_limit_mismatch"),
        (r"\b(intersect|union|except)\b", "set_operation_mismatch"),
    ]
    for pattern, label in feature_patterns:
        if bool(re.search(pattern, pred)) != bool(re.search(pattern, gold)):
            return label
    if pred.count("select") != gold.count("select"):
        return "subquery_mismatch"
    if row["pred_sql"].strip() and not row["pred_sql"].rstrip().endswith(";"):
        return "missing_semicolon"
    if error:
        return "other_execution_error"
    return "unclassified_semantic_error"


def build_case_rows(
    tests: list[dict[str, Any]],
    lora: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, test in enumerate(tests):
        row: dict[str, Any] = {"id": test["id"], "db_id": test["db_id"], "question": test["question"], "gold_sql": test["gold_sql"]}
        for condition in CONDITIONS:
            lora_row = lora[condition]["rows"][index]
            base_row = base[condition]["rows"][index]
            base_ok, lora_ok = as_bool(base_row["exec_match"]), as_bool(lora_row["exec_match"])
            row[f"base_{condition}_exec_match"] = base_ok
            row[f"lora_{condition}_exec_match"] = lora_ok
            row[f"base_{condition}_pred_sql"] = base_row["pred_sql"]
            row[f"lora_{condition}_pred_sql"] = lora_row["pred_sql"]
            row[f"lora_{condition}_pred_ok"] = as_bool(lora_row["pred_ok"])
            row[f"lora_{condition}_transition"] = ("correct" if base_ok else "wrong") + "->" + ("correct" if lora_ok else "wrong")
            row[f"lora_{condition}_error_category"] = "correct" if lora_ok else classify_sql_error(lora_row)
            if condition != "zero_shot":
                trace = lora[condition]["traces"][index]
                demo, score = trace_signature(trace)
                row[f"lora_{condition}_demo_id"] = demo
                row[f"lora_{condition}_similarity"] = score
                row[f"lora_{condition}_gate_decision"] = trace.get("gate_decision", "ungated")
        output.append(row)
    return output


def similarity_bin(score: float) -> str:
    if score < 0.60:
        return "<0.60"
    if score < 0.70:
        return "0.60-<0.70"
    if score < 0.85:
        return "0.70-<0.85"
    return ">=0.85"


def build_similarity_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    zero = runs["zero_shot"]["exec"]
    for family, ungated, gate070, gate085 in [
        ("top1", "top1", "top1_gate070", "top1_gate085"),
        ("structure", "structure", "structure_gate070", "structure_gate085"),
    ]:
        scores = np.asarray([trace_signature(row)[1] for row in runs[ungated]["traces"]], dtype=float)
        bins = np.asarray([similarity_bin(score) for score in scores])
        for label in ["<0.60", "0.60-<0.70", "0.70-<0.85", ">=0.85"]:
            mask = bins == label
            n = int(mask.sum())
            output.append({
                "retrieval_family": family,
                "similarity_bin": label,
                "n": n,
                "mean_similarity": float(scores[mask].mean()) if n else None,
                "zero_shot_ema": float(zero[mask].mean()) if n else None,
                "ungated_ema": float(runs[ungated]["exec"][mask].mean()) if n else None,
                "gate070_ema": float(runs[gate070]["exec"][mask].mean()) if n else None,
                "gate085_ema": float(runs[gate085]["exec"][mask].mean()) if n else None,
                "ungated_minus_zero": float((runs[ungated]["exec"][mask] - zero[mask]).mean()) if n else None,
                "gate070_minus_zero": float((runs[gate070]["exec"][mask] - zero[mask]).mean()) if n else None,
                "gate085_minus_zero": float((runs[gate085]["exec"][mask] - zero[mask]).mean()) if n else None,
            })
    return output


def bootstrap_did(
    base_zero: np.ndarray,
    base_condition: np.ndarray,
    lora_zero: np.ndarray,
    lora_condition: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    per_case = (lora_condition - lora_zero) - (base_condition - base_zero)
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining:
        count = min(250, remaining)
        indices = rng.integers(0, len(per_case), size=(count, len(per_case)))
        estimates.append(per_case[indices].mean(axis=1))
        remaining -= count
    values = np.concatenate(estimates)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def load_qwen_context(tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    all_maps = {**QWEN_BASE_RUNS, **QWEN_LORA_RUNS}
    for model, mapping in all_maps.items():
        loaded[model] = {condition: load_reference(run_id, tests) for condition, run_id in mapping.items()}
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        values: dict[str, Any] = {"condition": condition, "condition_label": DISPLAY[condition]}
        for model in ["qwen35_2b_base", "qwen35_2b_lora_v2", "llama32_3b_instruct_base", "llama32_3b_instruct_lora_v2", "qwen35_9b_base", "qwen35_9b_lora_v2"]:
            if model.startswith("llama"):
                continue
            reference = loaded.get(model, {}).get(condition)
            values[f"{model}_run_id"] = reference["run_id"] if reference else None
            values[f"{model}_ema"] = float(reference["exec"].mean()) if reference else None
        q2b = values.get("qwen35_2b_base_ema")
        q2l = values.get("qwen35_2b_lora_v2_ema")
        q9b = values.get("qwen35_9b_base_ema")
        q9l = values.get("qwen35_9b_lora_v2_ema")
        values["qwen35_2b_lora_gain"] = q2l - q2b if q2b is not None and q2l is not None else None
        values["qwen35_9b_lora_gain"] = q9l - q9b if q9b is not None and q9l is not None else None
        values["comparability"] = "B for populated cross-family cells; missing counterparts are not comparable"
        rows.append(values)
    provenance = {
        model: {
            condition: {
                "run_id": reference["run_id"],
                "csv_path": reference["csv_path"],
                "csv_sha256": reference["csv_sha256"],
                "metadata_path": reference["metadata_path"],
                "metadata_sha256": reference["metadata_sha256"],
                "ema": float(reference["exec"].mean()),
                "rows": len(reference["rows"]),
            }
            for condition, reference in conditions.items()
        }
        for model, conditions in loaded.items()
    }
    return rows, provenance


def main() -> None:
    outputs = [OUT_SUMMARY, OUT_CASES, OUT_BASE_STATS, OUT_FEWSHOT_STATS, OUT_SIMILARITY, OUT_QWEN]
    if OUT_SUMMARY.exists():
        raise RuntimeError(f"Refusing to overwrite derived summary: {OUT_SUMMARY}")
    if sha256(TESTCASES) != TEST_SHA256 or sha256(ADAPTER_MODEL) != ADAPTER_SHA256:
        raise RuntimeError("Testset or adapter identity mismatch")
    if sha256(BEST_CHECKPOINT / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("Adapter root differs from checkpoint-509")
    if sha256(INDEX_DIR / "index.faiss") != INDEX_SHA256 or sha256(INDEX_DIR / "metadata.jsonl") != INDEX_METADATA_SHA256:
        raise RuntimeError("Retrieval index identity mismatch")
    if sha256(STATIC_RESOURCE) != STATIC_SHA256:
        raise RuntimeError("Static resource identity mismatch")

    tests = load_jsonl(TESTCASES)
    if len(tests) != 1032:
        raise RuntimeError("Unexpected Spider Dev row count")
    lora = {condition: audit_run(condition, LLAMA_LORA_RUNS[condition], tests) for condition in CONDITIONS}
    base = {condition: load_reference(LLAMA_BASE_RUNS[condition], tests) for condition in CONDITIONS}

    top1_consistency = {
        "ungated_vs_gate070": compare_trace_sets(lora["top1"]["traces"], lora["top1_gate070"]["traces"]),
        "ungated_vs_gate085": compare_trace_sets(lora["top1"]["traces"], lora["top1_gate085"]["traces"]),
    }
    structure_consistency = {
        "ungated_vs_gate070": compare_trace_sets(lora["structure"]["traces"], lora["structure_gate070"]["traces"]),
        "ungated_vs_gate085": compare_trace_sets(lora["structure"]["traces"], lora["structure_gate085"]["traces"]),
    }
    gate_checks = {
        "top1_gate070": gate_reference_check(lora, "top1_gate070", "top1"),
        "top1_gate085": gate_reference_check(lora, "top1_gate085", "top1"),
        "structure_gate070": gate_reference_check(lora, "structure_gate070", "structure"),
        "structure_gate085": gate_reference_check(lora, "structure_gate085", "structure"),
    }
    static_ids = [trace_signature(row)[0] for row in lora["static_seed42"]["traces"]]
    static_check = {
        "rows": len(static_ids),
        "unique_demo_ids": sorted(set(static_ids)),
        "all_expected_demo": all(value == "SPIDER_TRAIN_001657" for value in static_ids),
        "resource_sha256": sha256(STATIC_RESOURCE),
    }
    if not static_check["all_expected_demo"]:
        raise RuntimeError("Static demo inconsistency")

    retrieval_overlap = retrieval_overlap_audit(tests)
    if any(retrieval_overlap[key] for key in ["id_overlap", "question_overlap", "sql_overlap", "pair_overlap"]):
        raise RuntimeError(f"Retrieval overlap detected: {retrieval_overlap}")

    base_retrieval_consistency: dict[str, Any] = {}
    for condition in CONDITIONS[1:]:
        if not base[condition]["trace"]:
            raise RuntimeError(f"Missing Llama Base trace for {condition}")
        base_retrieval_consistency[condition] = compare_trace_sets(base[condition]["trace"], lora[condition]["traces"])

    rescoring = execution_rescore(lora)
    total_rescore_mismatches = sum(
        item["existing_runner_path"]["esr_mismatch_count"]
        + item["existing_runner_path"]["ema_mismatch_count"]
        + item["independent_sqlite_path"]["esr_mismatch_count"]
        + item["independent_sqlite_path"]["ema_mismatch_count"]
        for item in rescoring.values()
    )
    if total_rescore_mismatches:
        raise RuntimeError(f"Execution re-scoring mismatches: {total_rescore_mismatches}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_stats = [
        paired_stats(base[condition]["exec"], lora[condition]["exec"], comparison="Llama Base vs LoRA", condition=condition, rng=rng)
        for condition in CONDITIONS
    ]
    holm_adjust(base_stats)

    fewshot_stats = [
        paired_stats(lora["zero_shot"]["exec"], lora[condition]["exec"], comparison="LoRA Zero Shot vs condition", condition=condition, rng=rng)
        for condition in CONDITIONS[1:]
    ]
    holm_adjust(fewshot_stats)
    targeted_pairs = [
        ("top1", "top1_gate070"),
        ("top1", "top1_gate085"),
        ("top1", "structure"),
        ("structure", "structure_gate070"),
        ("structure", "structure_gate085"),
        ("static_seed42", "top1"),
    ]
    targeted_stats = [
        paired_stats(lora[left]["exec"], lora[right]["exec"], comparison=f"{left} vs {right}", condition=right, rng=rng)
        for left, right in targeted_pairs
    ]

    did_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:]:
        base_effect = float(base[condition]["exec"].mean() - base["zero_shot"]["exec"].mean())
        lora_effect = float(lora[condition]["exec"].mean() - lora["zero_shot"]["exec"].mean())
        ci_low, ci_high = bootstrap_did(
            base["zero_shot"]["exec"], base[condition]["exec"], lora["zero_shot"]["exec"], lora[condition]["exec"], rng=rng
        )
        did_rows.append({
            "condition": condition,
            "base_fewshot_effect": base_effect,
            "lora_fewshot_effect": lora_effect,
            "difference_in_differences": lora_effect - base_effect,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })

    case_rows = build_case_rows(tests, lora, base)
    similarity_rows = build_similarity_rows(lora)
    qwen_rows, qwen_provenance = load_qwen_context(tests)
    for row in qwen_rows:
        condition = row["condition"]
        row["llama32_3b_instruct_base_run_id"] = base[condition]["run_id"]
        row["llama32_3b_instruct_base_ema"] = float(base[condition]["exec"].mean())
        row["llama32_3b_instruct_lora_v2_run_id"] = lora[condition]["run_id"]
        row["llama32_3b_instruct_lora_v2_ema"] = float(lora[condition]["exec"].mean())
        row["llama32_3b_instruct_lora_gain"] = row["llama32_3b_instruct_lora_v2_ema"] - row["llama32_3b_instruct_base_ema"]

    transition_summary: dict[str, Any] = {}
    error_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        transition_summary[condition] = dict(Counter(row[f"lora_{condition}_transition"] for row in case_rows))
        error_summary[condition] = dict(Counter(row[f"lora_{condition}_error_category"] for row in case_rows))

    top1_vs_structure = compare_trace_sets(lora["top1"]["traces"], lora["structure"]["traces"])
    summary = {
        "schema_version": 1,
        "purpose": "completed_llama32_3b_instruct_lora_v2_evaluation_audit",
        "status": "PASS_WITH_METHODICAL_LIMITATIONS",
        "warnings": [
            "Structure traces persist the final selected candidate and original BGE score, but not all ten candidates or per-candidate adjustment scores.",
            "Spider Dev was repeatedly used for development-facing comparisons and is not an untouched final test set.",
            "Structure Gate 0.70 and 0.85 are exploratory interaction analyses.",
        ],
        "generation_started_by_analysis": False,
        "model_or_adapter_loaded_by_analysis": False,
        "embedding_model_loaded_by_analysis": False,
        "testcases": {"path": str(TESTCASES.relative_to(ROOT)), "sha256": sha256(TESTCASES), "rows": len(tests)},
        "adapter": {
            "root": str(ADAPTER_ROOT.relative_to(ROOT)),
            "equivalent_checkpoint": str(BEST_CHECKPOINT.relative_to(ROOT)),
            "adapter_model_sha256": sha256(ADAPTER_MODEL),
            "root_equals_best": sha256(ADAPTER_MODEL) == sha256(BEST_CHECKPOINT / "adapter_model.safetensors"),
        },
        "retrieval": {
            "index_path": str(INDEX_DIR.relative_to(ROOT)),
            **retrieval_overlap,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "query_prefix_active": True,
        },
        "runs": {
            condition: {key: value for key, value in run.items() if key not in {"rows", "metadata", "traces", "exec", "pred_ok"}}
            for condition, run in lora.items()
        },
        "top1_trace_consistency": top1_consistency,
        "structure_trace_consistency": structure_consistency,
        "gate_reference_checks": gate_checks,
        "static_consistency": static_check,
        "llama_base_vs_lora_retrieval_consistency": base_retrieval_consistency,
        "top1_vs_structure_selection": top1_vs_structure,
        "execution_rescoring": rescoring,
        "base_vs_lora_statistics": base_stats,
        "lora_fewshot_vs_zero_statistics": fewshot_stats,
        "targeted_fewshot_statistics": targeted_stats,
        "difference_in_differences": did_rows,
        "transition_summary": transition_summary,
        "error_category_summary": error_summary,
        "similarity_bins": similarity_rows,
        "qwen_context_provenance": qwen_provenance,
        "comparability": {
            "llama_base_vs_lora": "A",
            "llama_lora_within_model_fewshot": "A",
            "qwen_cross_family": "B",
            "missing_qwen_counterparts": "NOT_COMPARABLE",
        },
        "statistics": {
            "mcnemar": "exact two-sided binomial test on discordant pairs",
            "n01_definition": "condition A wrong, condition B correct",
            "n10_definition": "condition A correct, condition B wrong",
            "bootstrap": "paired case bootstrap of EMA difference",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "multiple_comparisons": {
                "base_vs_lora": "Holm adjustment over 8 conditions",
                "lora_fewshot_vs_zero": "Holm adjustment over 7 conditions",
            },
        },
    }

    base_fields = list(base_stats[0].keys())
    few_fields = list(fewshot_stats[0].keys())
    case_fields = list(case_rows[0].keys())
    sim_fields = list(similarity_rows[0].keys())
    qwen_fields = sorted({key for row in qwen_rows for key in row})
    write_csv_new(OUT_BASE_STATS, base_fields, base_stats)
    write_csv_new(OUT_FEWSHOT_STATS, few_fields, fewshot_stats)
    write_csv_new(OUT_CASES, case_fields, case_rows)
    write_csv_new(OUT_SIMILARITY, sim_fields, similarity_rows)
    write_csv_new(OUT_QWEN, qwen_fields, qwen_rows)
    write_new(OUT_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "run_ema": {condition: run["metrics"]["ema"] for condition, run in lora.items()},
        "rescore_mismatches": total_rescore_mismatches,
    }, indent=2))


if __name__ == "__main__":
    main()
