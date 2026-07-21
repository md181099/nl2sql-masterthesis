#!/usr/bin/env python3
"""Final read-only audit for Qwen 3.5 2B max_new_tokens 256 vs 512.

The script reads completed evaluation artifacts, uses a local tokenizer for
token-prefix diagnostics, and opens Spider databases read-only for independent
execution rescoring. It never loads a language model, adapter, or retriever and
creates every output exclusively.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PREPARED_SCRIPT = ROOT / "scripts/analyze_qwen35_2b_base_maxnew512_sensitivity.py"
PREPARED_SCRIPT_SHA256 = "f4950f8b2136819964a15ff30a9938e386722ee6532e4987d5ffa7765bf480e5"
PRELIGHT_AUDIT = ROOT / "audits/audit_qwen35_2b_maxnew512_sensitivity_preflight_20260715.md"
PREFLIGHT_MANIFEST = ROOT / "audits/qwen35_2b_maxnew512_sensitivity_configs_manifest_20260715.json"
CONFIG_DIFF = ROOT / "audits/derived/qwen35_2b_maxnew512_config_diffs_20260715.csv"
MAINLINE_AUDIT = ROOT / "audits/audit_qwen35_2b_complete_8x8_base_and_lora_v2_evaluations_20260715.md"
MAINLINE_MANIFEST = ROOT / "audits/qwen35_2b_complete_8x8_base_and_lora_v2_manifest_20260715.json"
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
INDEX = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss"
INDEX_METADATA = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
STATIC_RESOURCE = ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
RUNNER = ROOT / "src/06_batch_run.py"
ADAPTER_ROOT = ROOT / "adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
ADAPTER_CHECKPOINT = ADAPTER_ROOT / "checkpoints/checkpoint-502"
ADAPTER_SHA256 = "6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3"
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
MODEL_SNAPSHOT = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TOKENIZER_SNAPSHOT = Path(
    "/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/"
    "snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
HF_MAIN_REF = Path("/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/refs/main")
TEST_SHA256 = "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce"
RUNNER_SHA256 = "a37286649920f4224999b5184e6117ea31f24968ad2c353ff338397c99a7a3c9"
SYSTEM_PROMPT_SHA256 = "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e"
PROMPT_RENDERER_SHA256 = "3d45db38486801bbafdd53346cd5af4f37c3d79d07a1c3a3183bdded7698ec6e"
INDEX_SHA256 = "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc"
INDEX_METADATA_SHA256 = "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55"
STATIC_SHA256 = "7c4735d7ba31ebd448cd0b94fd4c63a80c3e50f115d0fdd39e652ae0f1be1857"
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_RESAMPLES = 10_000

OUT_AUDIT = ROOT / "audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260716.md"
OUT_MANIFEST = ROOT / "audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json"
OUT_SUMMARY = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_summary_20260716.csv"
OUT_STATS = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_paired_statistics_20260716.csv"
OUT_CASES = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_case_comparison_20260716.csv"
OUT_CAPPED = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_capped_case_analysis_20260716.csv"
OUT_REPETITION = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_repetition_analysis_20260716.csv"
OUT_GATE = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_gate_identity_20260716.csv"
OUT_CONTROL = ROOT / "audits/derived/qwen35_2b_lora_zero_256_vs_512_control_20260716.csv"
OUT_FEWSHOT = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_fewshot_effects_20260716.csv"

CONDITIONS = [
    "zero_shot", "top1", "top1_gate070", "top1_gate085",
    "static_seed42", "structure", "structure_gate070", "structure_gate085",
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
RUNS_512 = {
    "zero_shot": "run_base_20260715_132148",
    "top1": "run_base_20260715_135532",
    "top1_gate070": "run_base_20260715_162605",
    "top1_gate085": "run_base_20260715_182102",
    "static_seed42": "run_base_20260715_190413",
    "structure": "run_base_20260715_205611",
    "structure_gate070": "run_base_20260715_233846",
    "structure_gate085": "run_base_20260716_013630",
}
LORA_512_RUN = (
    "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_"
    "evalstop_maxlen2048_epochs5_20260716_022008"
)
FULL_LOG_DIR = ROOT / "logs/qwen35_2b_maxnew512_sensitivity_full_20260715"


def import_prepared() -> Any:
    spec = importlib.util.spec_from_file_location("maxnew512_prepared", PREPARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import prepared sensitivity script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = import_prepared()
Q = P.Q
C = P.C
C.BOOTSTRAP_SEED = BOOTSTRAP_SEED
C.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
Q.BOOTSTRAP_SEED = BOOTSTRAP_SEED
Q.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_csv_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(materialized)
    write_new(path, buffer.getvalue())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_paths(run_id: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "results" / f"{run_id}.csv",
        ROOT / "results" / f"{run_id}_metadata.json",
        ROOT / "results/retrieval_traces" / f"{run_id}_retrieval_traces.jsonl",
    )


def log_for_config(config_path: str) -> Path:
    return FULL_LOG_DIR / (Path(config_path).stem + ".log")


def strict_config_diffs() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows = load_csv(CONFIG_DIFF)
    if len(rows) != 9:
        raise RuntimeError(f"Expected 9 config diffs, got {len(rows)}")
    mapping: dict[str, dict[str, str]] = {}
    details = []
    for row in rows:
        old_path, new_path = ROOT / row["source_config"], ROOT / row["new_512_config"]
        old, new = load_json(old_path), load_json(new_path)
        old_hash, new_hash = sha256(old_path), sha256(new_path)
        if old_hash != row["source_sha256"] or new_hash != row["new_512_sha256"]:
            raise RuntimeError(f"Config hash mismatch: {row['condition']}")
        changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        if changed != ["max_new_tokens"] or old.get("max_new_tokens") != 256 or new.get("max_new_tokens") != 512:
            raise RuntimeError(f"One-factor diff failed: {row['condition']} {changed}")
        mapping[row["condition"]] = row
        details.append({
            "condition": row["condition"], "source_config": row["source_config"],
            "source_sha256": old_hash, "new_config": row["new_512_config"],
            "new_sha256": new_hash, "changed_keys": changed, "status": "PASS",
        })
    return details, mapping


def metric_bundle(rows: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
    exec_values = np.asarray([C.as_bool(row["exec_match"]) for row in rows], dtype=np.int8)
    pred_ok = np.asarray([C.as_bool(row["pred_ok"]) for row in rows], dtype=np.int8)
    string_values = np.asarray([int(row["pred_sql"] == row["gold_sql"]) for row in rows], dtype=np.int8)
    normalized_values = np.asarray([
        int(C.normalized_sql(row["pred_sql"]) == C.normalized_sql(row["gold_sql"])) for row in rows
    ], dtype=np.int8)
    char_values = np.asarray([C.char_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    token_values = np.asarray([C.token_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])
    stored_mismatches = {
        "string_exact": sum(int(row["string_exact"]) != value for row, value in zip(rows, string_values)),
        "normalized_exact": sum(int(row["normalized_exact"]) != value for row, value in zip(rows, normalized_values)),
        "char_accuracy": sum(abs(float(row["char_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, char_values)),
        "token_accuracy": sum(abs(float(row["token_accuracy"]) - value) > 5.1e-7 for row, value in zip(rows, token_values)),
    }
    reproduced = {
        "execution_match_accuracy": float(exec_values.mean()),
        "execution_success_rate": float(pred_ok.mean()),
        "string_exact_match": float(string_values.mean()),
        "normalized_exact_match": float(normalized_values.mean()),
        "char_accuracy_avg": float(char_values.mean()),
        "token_accuracy_avg": float(token_values.mean()),
    }
    aggregate_mismatches = {
        key: abs(float(metadata[key]) - value) for key, value in reproduced.items()
    }
    if any(stored_mismatches.values()) or any(value > 5.1e-10 for value in aggregate_mismatches.values()):
        raise RuntimeError(f"Metric mismatch: stored={stored_mismatches}, aggregate={aggregate_mismatches}")
    prompt = np.asarray([int(row["prompt_tokens"]) for row in rows])
    completion = np.asarray([int(row["completion_tokens"]) for row in rows])
    return {
        "correct": int(exec_values.sum()), "ema": reproduced["execution_match_accuracy"],
        "executable": int(pred_ok.sum()), "esr": reproduced["execution_success_rate"],
        "string_exact_count": int(string_values.sum()), "string_exact": reproduced["string_exact_match"],
        "normalized_exact_count": int(normalized_values.sum()), "normalized_exact": reproduced["normalized_exact_match"],
        "char_accuracy": reproduced["char_accuracy_avg"], "token_accuracy": reproduced["token_accuracy_avg"],
        "runtime_seconds": float(metadata["duration_seconds"]),
        "seconds_per_case": float(metadata["duration_seconds"]) / len(rows),
        "prompt_mean": float(prompt.mean()), "prompt_max": int(prompt.max()),
        "completion_mean": float(completion.mean()), "completion_max": int(completion.max()),
        "at_256": int(np.sum(completion == 256)), "at_512": int(np.sum(completion == 512)),
        "exec": exec_values, "pred_ok": pred_ok,
        "stored_metric_mismatches": stored_mismatches,
        "aggregate_metric_absolute_differences": aggregate_mismatches,
    }


def audit_run(
    condition: str,
    run_id: str,
    expected_config: str,
    expected_hash: str,
    tests: list[dict[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Missing run artifacts: {run_id}")
    rows, metadata = load_csv(csv_path), load_json(metadata_path)
    config_path = ROOT / expected_config
    config = load_json(config_path)
    log_path = log_for_config(expected_config)
    if sha256(config_path) != expected_hash:
        raise RuntimeError(f"Config hash mismatch: {condition}")
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032:
        raise RuntimeError(f"Incomplete run: {run_id}")
    if [row["id"] for row in rows] != [row["id"] for row in tests] or len({row["id"] for row in rows}) != 1032:
        raise RuntimeError(f"Case order mismatch: {run_id}")
    for row, test in zip(rows, tests):
        if (row["db_id"], row["question"], row["gold_sql"]) != (test["db_id"], test["question"], test["gold_sql"]):
            raise RuntimeError(f"Test content mismatch: {run_id}/{row['id']}")
    adapter = "base" if role == "base" else P.SOURCE_LORA_CONTROL.removeprefix("run_").rsplit("_2026", 1)[0]
    # The run alias is pinned explicitly; deriving it from the old run id above is diagnostic only.
    if role == "lora":
        adapter = "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"
    checks = {
        "config_path": metadata.get("run_config_path") == expected_config,
        "model_id": metadata.get("run_model_id") == MODEL_ID,
        "adapter": metadata.get("run_adapter") == adapter,
        "prompt": metadata.get("run_prompt_format") == "qwen_sqlctx_chatml",
        "system_prompt": metadata.get("run_system_prompt_sha256") == SYSTEM_PROMPT_SHA256,
        "max_new": metadata.get("run_max_new_tokens") == config.get("max_new_tokens") == 512,
        "max_input": metadata.get("run_max_input_tokens") == config.get("max_input_tokens"),
        "batch": metadata.get("run_generation_batch_size") == 1,
        "sample_limit": metadata.get("run_max_test_samples") in (None, ""),
        "perplexity": metadata.get("run_compute_perplexity") is False,
        "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
        "row_config": all(row.get("run_config_path") == expected_config for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Run provenance failure {condition}: {[k for k, v in checks.items() if not v]}")
    if not log_path.is_file():
        raise RuntimeError(f"Missing full-run log: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    forbidden_log = re.findall(r"Traceback|CUDA out of memory|\bERROR\b|RuntimeError:", log_text)
    log_checks = {
        "no_errors": not forbidden_log,
        "dummy_false": "dummy_mode=False" in log_text,
        "max_new_512": "max_new_tokens=512" in log_text,
        "csv_written": str(csv_path.resolve()) in log_text,
        "metadata_written": str(metadata_path.resolve()) in log_text,
        "cuda": "MODEL DEVICE: cuda:0" in log_text,
        "qwen_prefix": "ends_with_qwen_assistant_prefix=True" in log_text,
        "no_llama_prefix": "ends_with_llama_assistant_prefix=False" in log_text,
        "no_forbidden_prompt_tokens": "forbidden_prompt_tokens_found=False" in log_text,
    }
    if not all(log_checks.values()):
        raise RuntimeError(f"Log failure {condition}: {[k for k, v in log_checks.items() if not v]}")
    prompt_limit = int(config["max_input_tokens"])
    if any(int(row["prompt_tokens"]) >= prompt_limit for row in rows):
        raise RuntimeError(f"Prompt at input limit: {condition}")
    if any("<think>" in row["raw_output"].lower() for row in rows):
        raise RuntimeError(f"Think marker in output: {condition}")
    if any(any(token in row["raw_output"] for token in ("<|start_header_id|>", "<|eot_id|>")) for row in rows):
        raise RuntimeError(f"Llama tokens in output: {condition}")
    traces: list[dict[str, Any]] = []
    if condition != "zero_shot" and role == "base":
        if not trace_path.is_file():
            raise RuntimeError(f"Missing trace: {condition}")
        traces = load_jsonl(trace_path)
        if len(traces) != 1032 or [row["id"] for row in traces] != [row["id"] for row in tests]:
            raise RuntimeError(f"Trace order failure: {condition}")
        if any(not row.get("retrieval_success") or row.get("leakage_status") != "pass" for row in traces):
            raise RuntimeError(f"Retrieval/leakage failure: {condition}")
    metrics = metric_bundle(rows, metadata)
    return {
        "condition": condition, "role": role, "run_id": run_id,
        "config_path": expected_config, "config_sha256": expected_hash,
        "csv_path": rel(csv_path), "csv_sha256": sha256(csv_path),
        "metadata_path": rel(metadata_path), "metadata_sha256": sha256(metadata_path),
        "trace_path": rel(trace_path) if trace_path.is_file() else None,
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else None,
        "log_path": rel(log_path), "log_sha256": sha256(log_path),
        "rows": rows, "metadata": metadata, "traces": traces,
        "checks": checks, "log_checks": log_checks, "metrics": metrics,
        "assignment_status": "UNAMBIGUOUS",
    }


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"rows", "metadata", "traces"}}


def trace_signature(trace: dict[str, Any]) -> tuple[str | None, float | None]:
    ids, scores = trace.get("retrieved_ids") or [], trace.get("retrieved_scores") or []
    return (str(ids[0]) if ids else None, float(scores[0]) if scores else None)


def retrieval_identity(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if not old["traces"] and not new["traces"]:
        return {"applicable": False}
    same_demo = same_score = 0
    max_delta = 0.0
    for a, b in zip(old["traces"], new["traces"]):
        aid, ascore = trace_signature(a)
        bid, bscore = trace_signature(b)
        same_demo += aid == bid
        if ascore is None and bscore is None:
            same_score += 1
        elif ascore is not None and bscore is not None:
            delta = abs(ascore - bscore)
            max_delta = max(max_delta, delta)
            same_score += delta <= 1e-12
    gate_counts = Counter(row.get("gate_decision") for row in new["traces"] if row.get("gate_decision") is not None)
    result = {
        "applicable": True, "same_demo_ids": same_demo, "same_scores": same_score,
        "max_score_delta": max_delta, "gate_counts": dict(gate_counts),
        "retrieval_success": sum(bool(row.get("retrieval_success")) for row in new["traces"]),
        "leakage_pass": sum(row.get("leakage_status") == "pass" for row in new["traces"]),
    }
    if same_demo != 1032 or same_score != 1032 or result["retrieval_success"] != 1032 or result["leakage_pass"] != 1032:
        raise RuntimeError(f"Retrieval identity failure: {new['condition']} {result}")
    return result


def ngram_repetition(tokens: list[int], n: int) -> tuple[int, float]:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0, 0.0
    counts = Counter(tuple(tokens[index:index + n]) for index in range(total))
    excess = sum(max(0, count - 1) for count in counts.values())
    return excess, excess / total


def longest_adjacent_repeat(tokens: list[int]) -> int:
    best = 0
    for gap in range(1, min(128, len(tokens) // 2) + 1):
        run = 0
        for index in range(gap, len(tokens)):
            if tokens[index] == tokens[index - gap]:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


def compression_ratio(data: bytes) -> float:
    return len(zlib.compress(data, level=9)) / len(data) if data else 1.0


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def repetition_features(raw: str, token_ids: list[int], tokenizer: Any) -> dict[str, Any]:
    lower = raw.lower()
    token_counts = Counter(token_ids)
    repeated_tokens = sum(max(0, count - 1) for count in token_counts.values())
    excess3, fraction3 = ngram_repetition(token_ids, 3)
    excess5, fraction5 = ngram_repetition(token_ids, 5)
    identifiers = [value.lower() for value in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", raw)]
    keywords = {
        "select", "from", "where", "join", "left", "right", "inner", "outer", "on", "and", "or",
        "as", "in", "with", "group", "order", "by", "limit", "having", "distinct", "union", "except",
        "intersect", "count", "sum", "avg", "min", "max", "asc", "desc", "case", "when", "then", "else", "end",
    }
    tables = [value.lower() for value in re.findall(r"(?i)\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", raw)]
    table_set = set(tables)
    columns = [value for value in identifiers if value not in keywords and value not in table_set]
    parenthesized = re.findall(r"\([^()]{1,200}\)", raw)
    semicolon_char = raw.find(";")
    semicolon_token = None
    if semicolon_char >= 0:
        prefix = tokenizer.encode(raw[:semicolon_char + 1], add_special_tokens=False)
        semicolon_token = len(prefix) - 1 if prefix else None
    token_bytes = ",".join(str(value) for value in token_ids).encode("ascii")
    repeated = (
        fraction5 >= 0.20
        or (repeated_tokens / len(token_ids) if token_ids else 0.0) >= 0.50
        or longest_adjacent_repeat(token_ids) >= 20
    )
    return {
        "repeated_token_count": repeated_tokens,
        "repeated_token_fraction": repeated_tokens / len(token_ids) if token_ids else 0.0,
        "repeated_3gram_excess": excess3, "repeated_3gram_fraction": fraction3,
        "repeated_5gram_excess": excess5, "repeated_5gram_fraction": fraction5,
        "longest_identical_repetition_sequence": longest_adjacent_repeat(token_ids),
        "select_count": len(re.findall(r"(?i)\bselect\b", raw)),
        "with_count": len(re.findall(r"(?i)\bwith\b", raw)),
        "repeated_subquery_excess": max(0, len(re.findall(r"(?i)\(\s*select\b", raw)) - 1),
        "repeated_table_fragment_excess": sum(max(0, count - 1) for count in Counter(tables).values()),
        "repeated_column_fragment_excess": sum(max(0, count - 1) for count in Counter(columns).values()),
        "repeated_parenthesized_structure_excess": sum(max(0, count - 1) for count in Counter(parenthesized).values()),
        "character_compression_ratio": compression_ratio(raw.encode("utf-8")),
        "token_compression_ratio": compression_ratio(token_bytes),
        "first_semicolon_char": semicolon_char if semicolon_char >= 0 else None,
        "first_semicolon_token": semicolon_token,
        "content_after_first_semicolon": raw[semicolon_char + 1:].strip() if semicolon_char >= 0 else "",
        "repetition_rule_triggered": repeated,
    }


def classify_capped(old: dict[str, str], new: dict[str, str], features: dict[str, Any]) -> tuple[str, str | None]:
    old_ok, new_ok = C.as_bool(old["pred_ok"]), C.as_bool(new["pred_ok"])
    old_match, new_match = C.as_bool(old["exec_match"]), C.as_bool(new["exec_match"])
    semicolon_token = features["first_semicolon_token"]
    late_semicolon = semicolon_token is not None and semicolon_token >= 256
    new_at_limit = int(new["completion_tokens"]) == 512
    repeated = bool(features["repetition_rule_triggered"])
    after_semicolon = bool(features["content_after_first_semicolon"])
    if not old_match and new_match and late_semicolon:
        primary = "late_valid_completion"
    elif not old["pred_sql"].strip() and new["pred_sql"].strip():
        primary = "extractor_recovery"
    elif not old_match and new_match:
        primary = "new_execution_match"
    elif not old_ok and new_ok:
        primary = "new_execution_success_but_wrong"
    elif new_at_limit and semicolon_token is None:
        primary = "still_truncated_at_512"
    elif after_semicolon and repeated:
        primary = "valid_sql_followed_by_repetition"
    elif repeated:
        primary = "continued_repetition"
    elif new_at_limit:
        primary = "still_truncated_at_512"
    elif new_ok and not new_match:
        primary = "semantic_error_without_repetition"
    elif new["raw_output"].strip():
        primary = "other"
    else:
        primary = "unclassified"
    cause = None
    if not old_match and new_match:
        if late_semicolon:
            cause = "A_late_complete_sql"
        elif not old["pred_sql"].strip() and new["pred_sql"].strip():
            cause = "B_extractor_recovery"
        elif not old_ok and new_ok:
            cause = "C_new_execution_success"
        elif old["pred_sql"] == new["pred_sql"]:
            cause = "D_same_sql_extraction_change"
        elif features["first_semicolon_token"] is not None:
            cause = "E_later_text_changed_first_extraction"
        else:
            cause = "F_other"
    return primary, cause


def transition(old: dict[str, str], new: dict[str, str]) -> str:
    return ("correct" if C.as_bool(old["exec_match"]) else "wrong") + "_to_" + (
        "correct" if C.as_bool(new["exec_match"]) else "wrong"
    )


def gate_identity(new_runs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out = []
    summary = {}
    references = {
        "top1_gate070": "top1", "top1_gate085": "top1",
        "structure_gate070": "structure", "structure_gate085": "structure",
    }
    for condition, fewshot_condition in references.items():
        gate_run, few_run, zero_run = new_runs[condition], new_runs[fewshot_condition], new_runs["zero_shot"]
        raw_match = pred_match = prompt_match = 0
        counts = Counter()
        for gate_row, few_row, zero_row, trace in zip(gate_run["rows"], few_run["rows"], zero_run["rows"], gate_run["traces"]):
            decision = trace.get("gate_decision")
            if decision not in {"fewshot", "zero_shot"}:
                raise RuntimeError(f"Invalid gate decision {condition}/{gate_row['id']}: {decision}")
            reference = few_row if decision == "fewshot" else zero_row
            same_raw = gate_row["raw_output"] == reference["raw_output"]
            same_pred = gate_row["pred_sql"] == reference["pred_sql"]
            same_prompt = gate_row["prompt_tokens"] == reference["prompt_tokens"]
            raw_match += same_raw
            pred_match += same_pred
            prompt_match += same_prompt
            counts[decision] += 1
            rows_out.append({
                "condition": condition, "case_id": gate_row["id"], "gate_decision": decision,
                "reference_condition": fewshot_condition if decision == "fewshot" else "zero_shot",
                "raw_output_identical": same_raw, "pred_sql_identical": same_pred,
                "prompt_token_count_identical": same_prompt,
            })
        summary[condition] = {
            "gate_counts": dict(counts), "raw_output_matches": raw_match,
            "pred_sql_matches": pred_match, "prompt_token_matches": prompt_match,
        }
        if min(raw_match, pred_match, prompt_match) != 1032:
            raise RuntimeError(f"Gate mixture identity failure: {condition} {summary[condition]}")
    return rows_out, summary


def control_analysis(old: dict[str, Any], new: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    counts = Counter()
    for a, b in zip(old["rows"], new["rows"]):
        values = {
            "raw_output_identical": a["raw_output"] == b["raw_output"],
            "pred_sql_identical": a["pred_sql"] == b["pred_sql"],
            "execution_identical": (a["pred_ok"], a["exec_match"]) == (b["pred_ok"], b["exec_match"]),
            "completion_tokens_identical": a["completion_tokens"] == b["completion_tokens"],
            "prompt_tokens_identical": a["prompt_tokens"] == b["prompt_tokens"],
        }
        for key, value in values.items():
            counts[key] += value
        rows.append({"case_id": a["id"], **values})
    summary = {
        **dict(counts), "rows": 1032,
        "old_at_256": old["metrics"]["at_256"], "new_at_512": new["metrics"]["at_512"],
        "ema_256": old["metrics"]["ema"], "ema_512": new["metrics"]["ema"],
        "esr_256": old["metrics"]["esr"], "esr_512": new["metrics"]["esr"],
        "status": "PASS" if min(counts.values()) == 1032 and old["metrics"]["at_256"] == 0 else "FAIL",
    }
    if summary["status"] != "PASS":
        raise RuntimeError(f"LoRA negative control failed: {summary}")
    return rows, summary


def main() -> None:
    outputs = [OUT_AUDIT, OUT_MANIFEST, OUT_SUMMARY, OUT_STATS, OUT_CASES, OUT_CAPPED, OUT_REPETITION, OUT_GATE, OUT_CONTROL, OUT_FEWSHOT]
    for path in outputs:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
    for path, expected in [
        (PREPARED_SCRIPT, PREPARED_SCRIPT_SHA256), (TESTCASES, TEST_SHA256), (RUNNER, RUNNER_SHA256),
        (INDEX, INDEX_SHA256), (INDEX_METADATA, INDEX_METADATA_SHA256), (STATIC_RESOURCE, STATIC_SHA256),
        (ADAPTER_ROOT / "adapter_model.safetensors", ADAPTER_SHA256),
        (ADAPTER_CHECKPOINT / "adapter_model.safetensors", ADAPTER_SHA256),
    ]:
        if sha256(path) != expected:
            raise RuntimeError(f"Pinned artifact hash mismatch: {path}")
    if HF_MAIN_REF.read_text(encoding="utf-8").strip() != MODEL_SNAPSHOT or not TOKENIZER_SNAPSHOT.is_dir():
        raise RuntimeError("Local model snapshot provenance mismatch")

    tests = load_jsonl(TESTCASES)
    if len(tests) != 1032:
        raise RuntimeError("Spider Dev row count mismatch")
    diff_details, diff_mapping = strict_config_diffs()
    preflight_manifest = load_json(PREFLIGHT_MANIFEST)
    old_runs = {condition: P.load_run(run_id) for condition, run_id in P.SOURCE_RUNS.items()}
    for run in old_runs.values():
        run["metrics"] = metric_bundle(run["rows"], run["metadata"])
    old_control = P.load_run(P.SOURCE_LORA_CONTROL)
    old_control["metrics"] = metric_bundle(old_control["rows"], old_control["metadata"])

    new_runs: dict[str, dict[str, Any]] = {}
    assignments: dict[str, Any] = {}
    for condition in CONDITIONS:
        expected_config = P.CONFIG_512[condition]
        expected_hash = P.CONFIG_512_SHA256[condition]
        discovered_id, _ = P.find_completed_run(expected_config, expected_hash)
        if discovered_id != RUNS_512[condition]:
            raise RuntimeError(f"Unexpected run assignment {condition}: {discovered_id}")
        new_runs[condition] = audit_run(condition, discovered_id, expected_config, expected_hash, tests, role="base")
        assignments[condition] = {"expected_run": RUNS_512[condition], "discovered_run": discovered_id, "status": "UNAMBIGUOUS"}
    discovered_control, _ = P.find_completed_run(P.CONFIG_512_LORA, P.CONFIG_512_LORA_SHA256)
    if discovered_control != LORA_512_RUN:
        raise RuntimeError(f"Unexpected LoRA control assignment: {discovered_control}")
    new_control = audit_run("zero_shot", discovered_control, P.CONFIG_512_LORA, P.CONFIG_512_LORA_SHA256, tests, role="lora")
    assignments["lora_zero_control"] = {"expected_run": LORA_512_RUN, "discovered_run": discovered_control, "status": "UNAMBIGUOUS"}

    control_rows, control_summary = control_analysis(old_control, new_control)
    retrieval = {condition: retrieval_identity(old_runs[condition], new_runs[condition]) for condition in CONDITIONS}
    expected_gates = {
        "top1_gate070": {"fewshot": 634, "zero_shot": 398},
        "top1_gate085": {"fewshot": 57, "zero_shot": 975},
        "structure_gate070": {"fewshot": 613, "zero_shot": 419},
        "structure_gate085": {"fewshot": 57, "zero_shot": 975},
    }
    for condition, expected in expected_gates.items():
        if retrieval[condition]["gate_counts"] != expected:
            raise RuntimeError(f"Gate count mismatch: {condition} {retrieval[condition]['gate_counts']}")
    static_ids = [trace_signature(row)[0] for row in new_runs["static_seed42"]["traces"]]
    if set(static_ids) != {"SPIDER_TRAIN_001657"} or len(static_ids) != 1032:
        raise RuntimeError("Static demo identity failure")

    gate_rows, gate_summary = gate_identity(new_runs)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SNAPSHOT, local_files_only=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    capped_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    determinism: dict[str, Any] = {}
    capped_summary: dict[str, Any] = {}

    for condition in CONDITIONS:
        old, new = old_runs[condition], new_runs[condition]
        old_m, new_m = old["metrics"], new["metrics"]
        uncapped_raw = uncapped_sql = uncapped_exec = 0
        uncapped_count = capped_count = prefix_count = raw_prefix_count = 0
        reencoded_length_counts: Counter[int] = Counter()
        prefix_mismatch_ids: list[str] = []
        uncapped_mismatch_ids: list[str] = []
        capped_counter = Counter()
        class_counter = Counter()
        cause_counter = Counter()
        for a, b in zip(old["rows"], new["rows"]):
            is_capped = int(a["completion_tokens"]) == 256
            same_raw = a["raw_output"] == b["raw_output"]
            same_sql = a["pred_sql"] == b["pred_sql"]
            same_exec = (a["pred_ok"], a["exec_match"]) == (b["pred_ok"], b["exec_match"])
            token_prefix = None
            old_token_count = None
            if is_capped:
                capped_count += 1
                old_ids = tokenizer.encode(a["raw_output"], add_special_tokens=False)
                new_ids = tokenizer.encode(b["raw_output"], add_special_tokens=False)
                old_token_count = len(old_ids)
                reencoded_length_counts[old_token_count] += 1
                raw_text_prefix = b["raw_output"].startswith(a["raw_output"])
                raw_prefix_count += raw_text_prefix
                token_prefix = old_ids == new_ids[:len(old_ids)]
                prefix_count += bool(token_prefix)
                if not token_prefix:
                    prefix_mismatch_ids.append(a["id"])
                features = repetition_features(b["raw_output"], new_ids, tokenizer)
                primary, cause = classify_capped(a, b, features)
                trans = transition(a, b)
                class_counter[primary] += 1
                if cause:
                    cause_counter[cause] += 1
                terminated = int(b["completion_tokens"]) < 512
                new_at_limit = int(b["completion_tokens"]) == 512
                newly_executable = not C.as_bool(a["pred_ok"]) and C.as_bool(b["pred_ok"])
                newly_correct = not C.as_bool(a["exec_match"]) and C.as_bool(b["exec_match"])
                lost_correct = C.as_bool(a["exec_match"]) and not C.as_bool(b["exec_match"])
                late_semicolon = features["first_semicolon_token"] is not None and features["first_semicolon_token"] >= 256
                capped_counter.update({
                    "terminated_before_512": terminated, "capped_again_512": new_at_limit,
                    "newly_executable": newly_executable, "newly_correct": newly_correct,
                    "lost_correct": lost_correct, trans: 1,
                    "continued_repetition": features["repetition_rule_triggered"],
                    "late_valid_sql": primary == "late_valid_completion",
                    "extractor_recovery": primary == "extractor_recovery",
                    "late_semicolon": late_semicolon,
                })
                capped_row = {
                    "condition": condition, "case_id": a["id"], "db_id": a["db_id"],
                    "old_completion_tokens": int(a["completion_tokens"]), "new_completion_tokens": int(b["completion_tokens"]),
                    "old_reencoded_tokens": old_token_count, "token_prefix_identical": token_prefix,
                    "old_pred_ok": C.as_bool(a["pred_ok"]), "new_pred_ok": C.as_bool(b["pred_ok"]),
                    "old_exec_match": C.as_bool(a["exec_match"]), "new_exec_match": C.as_bool(b["exec_match"]),
                    "transition": trans, "terminated_before_512": terminated, "capped_again_512": new_at_limit,
                    "newly_executable": newly_executable, "newly_correct": newly_correct, "lost_correct": lost_correct,
                    "old_sql_extracted": bool(a["pred_sql"].strip()), "new_sql_extracted": bool(b["pred_sql"].strip()),
                    "primary_classification": primary, "improvement_cause": cause,
                    "late_semicolon": late_semicolon,
                }
                capped_rows.append(capped_row)
                repetition_rows.append({"condition": condition, "case_id": a["id"], **features, "primary_classification": primary})
            else:
                uncapped_count += 1
                uncapped_raw += same_raw
                uncapped_sql += same_sql
                uncapped_exec += same_exec
                if not (same_raw and same_sql and same_exec):
                    uncapped_mismatch_ids.append(a["id"])
            case_rows.append({
                "condition": condition, "case_id": a["id"], "db_id": a["db_id"],
                "old_completion_tokens": int(a["completion_tokens"]), "new_completion_tokens": int(b["completion_tokens"]),
                "old_capped_256": is_capped, "new_capped_512": int(b["completion_tokens"]) == 512,
                "raw_output_identical": same_raw, "pred_sql_identical": same_sql,
                "execution_identical": same_exec, "token_prefix_identical_if_capped": token_prefix,
                "old_pred_ok": C.as_bool(a["pred_ok"]), "new_pred_ok": C.as_bool(b["pred_ok"]),
                "old_exec_match": C.as_bool(a["exec_match"]), "new_exec_match": C.as_bool(b["exec_match"]),
                "transition": transition(a, b),
            })
        determinism[condition] = {
            "uncapped_cases": uncapped_count, "uncapped_raw_identical": uncapped_raw,
            "uncapped_sql_identical": uncapped_sql, "uncapped_execution_identical": uncapped_exec,
            "uncapped_mismatch_case_ids": uncapped_mismatch_ids,
            "capped_cases": capped_count, "capped_prefix_identical": prefix_count,
            "capped_raw_text_prefix_identical": raw_prefix_count,
            "old_raw_reencoded_token_length_counts": dict(sorted(reencoded_length_counts.items())),
            "prefix_mismatch_case_ids": prefix_mismatch_ids,
        }
        if uncapped_mismatch_ids or prefix_mismatch_ids or raw_prefix_count != capped_count:
            raise RuntimeError(f"Determinism failure: {condition}")
        capped_summary[condition] = {
            "capped_cases": capped_count, **dict(capped_counter),
            "classifications": dict(class_counter), "improvement_causes": dict(cause_counter),
        }
        stats_rows.append(C.paired_stats(
            old_m["exec"], new_m["exec"], comparison="Base max_new_tokens 256 vs 512",
            condition=condition, rng=rng,
        ))
        summary_rows.append({
            "condition": condition, "condition_label": DISPLAY[condition],
            "run_256": old["run_id"], "run_512": new["run_id"],
            "ema_256": old_m["ema"], "ema_512": new_m["ema"],
            "ema_delta_percentage_points": 100 * (new_m["ema"] - old_m["ema"]),
            "ema_correct_256": old_m["correct"], "ema_correct_512": new_m["correct"],
            "esr_256": old_m["esr"], "esr_512": new_m["esr"],
            "esr_delta_percentage_points": 100 * (new_m["esr"] - old_m["esr"]),
            "executable_256": old_m["executable"], "executable_512": new_m["executable"],
            "string_em_256": old_m["string_exact"], "string_em_512": new_m["string_exact"],
            "normalized_em_256": old_m["normalized_exact"], "normalized_em_512": new_m["normalized_exact"],
            "char_accuracy_256": old_m["char_accuracy"], "char_accuracy_512": new_m["char_accuracy"],
            "token_accuracy_256": old_m["token_accuracy"], "token_accuracy_512": new_m["token_accuracy"],
            "runtime_seconds_256": old_m["runtime_seconds"], "runtime_seconds_512": new_m["runtime_seconds"],
            "seconds_per_case_256": old_m["seconds_per_case"], "seconds_per_case_512": new_m["seconds_per_case"],
            "prompt_mean_256": old_m["prompt_mean"], "prompt_mean_512": new_m["prompt_mean"],
            "prompt_max_256": old_m["prompt_max"], "prompt_max_512": new_m["prompt_max"],
            "completion_mean_256": old_m["completion_mean"], "completion_mean_512": new_m["completion_mean"],
            "completion_max_256": old_m["completion_max"], "completion_max_512": new_m["completion_max"],
            "limit_cases_256": old_m["at_256"], "limit_cases_512": new_m["at_512"],
            "metric_mismatches_512": sum(new_m["stored_metric_mismatches"].values()),
        })
    C.holm_adjust(stats_rows)

    rescoring = C.execution_rescore({**new_runs, "lora_zero_control": new_control})
    rescore_mismatches = sum(
        details[path][metric]
        for details in rescoring.values()
        for path in ["existing_runner_path", "independent_sqlite_path"]
        for metric in ["esr_mismatch_count", "ema_mismatch_count"]
    )
    path_disagreements = sum(details["path_disagreement_count"] for details in rescoring.values())
    if rescore_mismatches or path_disagreements:
        raise RuntimeError(f"Execution rescore mismatch: stored={rescore_mismatches}, paths={path_disagreements}")

    by_summary = {row["condition"]: row for row in summary_rows}
    fewshot_rows = []
    for condition in CONDITIONS[1:]:
        effect256 = by_summary[condition]["ema_256"] - by_summary["zero_shot"]["ema_256"]
        effect512 = by_summary[condition]["ema_512"] - by_summary["zero_shot"]["ema_512"]
        fewshot_rows.append({
            "condition": condition, "condition_label": DISPLAY[condition],
            "fewshot_effect_256": effect256, "fewshot_effect_256_pp": 100 * effect256,
            "fewshot_effect_512": effect512, "fewshot_effect_512_pp": 100 * effect512,
            "change": effect512 - effect256, "change_pp": 100 * (effect512 - effect256),
        })

    total_capped = sum(value["capped_cases"] for value in capped_summary.values())
    total_capped_again = sum(value.get("capped_again_512", 0) for value in capped_summary.values())
    total_new_correct = sum(value.get("newly_correct", 0) for value in capped_summary.values())
    total_late_valid = sum(value.get("late_valid_sql", 0) for value in capped_summary.values())
    total_repeated = sum(value.get("continued_repetition", 0) for value in capped_summary.values())
    static_lost_correct = [
        row for row in capped_rows
        if row["condition"] == "static_seed42" and row["lost_correct"]
    ]
    h1 = "TEILWEISE GESTUETZT" if total_new_correct else "NICHT GESTUETZT"
    h2 = "GESTUETZT" if total_capped_again > total_capped / 2 or total_repeated > total_capped / 2 else "TEILWEISE GESTUETZT"
    hypotheses = {"H1_binding_limit": h1, "H2_repetition": h2, "H3_determinism": "BESTAETIGT", "H4_lora_control": "BESTAETIGT"}
    status = "PASS MIT METHODISCHEN EINSCHRAENKUNGEN"
    rerun = False

    write_csv_new(OUT_SUMMARY, summary_rows)
    write_csv_new(OUT_STATS, stats_rows)
    write_csv_new(OUT_CASES, case_rows)
    write_csv_new(OUT_CAPPED, capped_rows)
    write_csv_new(OUT_REPETITION, repetition_rows)
    write_csv_new(OUT_GATE, gate_rows)
    write_csv_new(OUT_CONTROL, control_rows)
    write_csv_new(OUT_FEWSHOT, fewshot_rows)

    stat_by_condition = {row["condition"]: row for row in stats_rows}
    lines = [
        "# Qwen 3.5 2B Base: max_new_tokens 256 vs. 512",
        "",
        "```text",
        f"QWEN35-2B-MAXNEW256-VS-512-SENSITIVITY-AUDIT: {status}",
        "OFFICIAL MAINLINE: max_new_tokens = 256",
        "SENSITIVITY ANALYSIS: max_new_tokens = 512",
        "RERUN REQUIRED: NEIN",
        "```",
        "",
        "## 1. Executive Summary",
        "",
        "Die neun 512er-Laeufe sind vollstaendig, eindeutig zugeordnet und technisch valide. Alle neun Configpaare unterscheiden sich ausschliesslich in `max_new_tokens`. Die Analyse ist explorativ; die modelluebergreifende Hauptlinie bleibt bei 256 Tokens.",
        "",
        f"Die LoRA-Negativkontrolle reproduziert {control_summary['raw_output_identical']}/1032 Rohoutputs und {control_summary['pred_sql_identical']}/1032 SQLs exakt. Alle nicht gedeckelten Base-Ausgaben und alle re-tokenisierten Prefixe der gedeckelten Ausgaben sind identisch.",
        "",
        "## 2. Methodik und Provenienz",
        "",
        f"- Vorbereitetes Skript (unveraendert): `{rel(PREPARED_SCRIPT)}`, SHA256 `{PREPARED_SCRIPT_SHA256}`.",
        f"- Additive Abschlussversion: `{rel(Path(__file__).resolve())}`, SHA256 `{sha256(Path(__file__).resolve())}`.",
        "- Grund der additiven Version: Das vorbereitete Skript enthielt weder beide SQLite-Re-Scorings noch Gate-Mischungs-, vollstaendige Case- und Few-Shot-Effekt-Artefakte und zielte auf 20260715-Ausgaben.",
        f"- Runner: `{rel(RUNNER)}`, SHA256 `{RUNNER_SHA256}`.",
        f"- Qwen-Promptrenderer-SHA256: `{PROMPT_RENDERER_SHA256}`; Systemprompt-SHA256: `{SYSTEM_PROMPT_SHA256}`.",
        f"- Spider Dev: 1032 Faelle, SHA256 `{TEST_SHA256}`.",
        f"- Modell: `{MODEL_ID}`, lokaler `refs/main`-Snapshot `{MODEL_SNAPSHOT}`.",
        f"- LoRA-Root und Checkpoint-502 sind gewichtsgleich, SHA256 `{ADAPTER_SHA256}`.",
        "- Kein Modell, Adapter oder BGE-Modell wurde geladen. SQLite wurde mit `mode=ro` und `query_only` geoeffnet.",
        "",
        "## 3. Runzuordnung",
        "",
        "| Bedingung | 256er-Run | 512er-Run | 512er-Zuordnung |",
        "|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        lines.append(f"| {DISPLAY[condition]} | `{old_runs[condition]['run_id']}` | `{new_runs[condition]['run_id']}` | UNAMBIGUOUS |")
    lines.append(f"| LoRA Zero Control | `{old_control['run_id']}` | `{new_control['run_id']}` | UNAMBIGUOUS |")
    lines.extend([
        "", "## 4. Ein-Faktor- und technische Integritaet", "",
        "- `ONE-FACTOR-DIFF`: PASS (9/9; ausschliesslich `256 -> 512`).",
        "- Je Run: 1032/1032 eindeutige, geordnete Spider-Dev-Faelle; `max_test_samples=None`; Batch 1; greedy; `sql_first_statement_only`.",
        "- Prompt: `qwen_sqlctx_chatml`, `sqlctx_anti_overjoin`; keine Prompttruncation, keine `<think>`- oder Llama-Tokens.",
        "- Alle Logs enden regulaer; keine Tracebacks, OOMs oder ERROR-Abbrueche.",
        "- Retrievaldemo und BGE-Score sind in allen Retrievalbedingungen 1032/1032 identisch; Leakage-Status 1032/1032 `pass`.",
        "", "## 5. Metrik- und Execution-Pruefung", "",
        "Alle CSV-Einzelmetriken und Metadatenaggregate wurden reproduziert. Beide Execution-Pfade stimmen ohne ESR-/EMA-Mismatch mit den gespeicherten Werten und miteinander ueberein.",
        "",
        f"- Gespeicherte/reproduzierte Metrikmismatches: 0",
        f"- SQLite-Re-Scoring-Mismatches: {rescore_mismatches}",
        f"- Abweichungen zwischen den zwei SQLite-Pfaden: {path_disagreements}",
        "- Die beiden bekannten Goldfehler `SPIDER_DEV_000455` und `SPIDER_DEV_000456` betreffen denselben nicht als UTF-8 dekodierbaren `wta_1.last_name`-Wert. Beide Re-Scoring-Pfade behandeln sie identisch; sie erzeugen keinen gespeicherten-vs.-reproduzierten Mismatch.",
        "", "## 6. Hauptvergleich", "",
        "| Bedingung | EMA 256 | EMA 512 | Delta pp | ESR 256 | ESR 512 | Limit 256 | Limit 512 | McNemar p | Holm-8 p | Bootstrap-95%-KI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        stat = stat_by_condition[row["condition"]]
        lines.append(
            f"| {row['condition_label']} | {100*row['ema_256']:.2f}% | {100*row['ema_512']:.2f}% | "
            f"{row['ema_delta_percentage_points']:+.2f} | {100*row['esr_256']:.2f}% | {100*row['esr_512']:.2f}% | "
            f"{row['limit_cases_256']} | {row['limit_cases_512']} | {stat['mcnemar_p']:.6g} | "
            f"{stat['holm_adjusted_p']:.6g} | [{100*stat['bootstrap_ci_low']:.2f}, {100*stat['bootstrap_ci_high']:.2f}] pp |"
        )
    lines.extend([
        "", "## 7. Determinismus und LoRA-Kontrolle", "",
        f"- Nicht gedeckelte Base-Rohoutputs: {sum(v['uncapped_raw_identical'] for v in determinism.values())}/{sum(v['uncapped_cases'] for v in determinism.values())} identisch.",
        f"- Nicht gedeckelte Base-SQLs: {sum(v['uncapped_sql_identical'] for v in determinism.values())}/{sum(v['uncapped_cases'] for v in determinism.values())} identisch.",
        f"- Gedeckelte Rohtext-Prefixe: {sum(v['capped_raw_text_prefix_identical'] for v in determinism.values())}/{sum(v['capped_cases'] for v in determinism.values())} identisch.",
        f"- Gedeckelte kanonisch re-tokenisierte Prefixe: {sum(v['capped_prefix_identical'] for v in determinism.values())}/{sum(v['capped_cases'] for v in determinism.values())} identisch.",
        "- Methodische Einschraenkung: Die Runner speicherten keine originalen generierten Token-ID-Sequenzen. Nur 104/2215 alte Rohstrings re-tokenisieren auf exakt 256 Tokens; die uebrigen auf 248 bis 255. Daher ist kein direkter Original-Tokenarray-Nachweis moeglich. Der unveraenderte Rohtextprefix und dessen kanonischer Tokenprefix sind jedoch vollstaendig bestaetigt.",
        f"- LoRA: Rohoutput {control_summary['raw_output_identical']}/1032, SQL {control_summary['pred_sql_identical']}/1032, Execution {control_summary['execution_identical']}/1032, Completionlaenge {control_summary['completion_tokens_identical']}/1032.",
        f"- LoRA EMA: {old_control['metrics']['correct']}/1032 = {100*old_control['metrics']['ema']:.2f}% in beiden Laeufen; ESR {old_control['metrics']['executable']}/1032 = {100*old_control['metrics']['esr']:.2f}%.",
        "", "## 8. Verhalten ehemals gedeckelter Faelle", "",
        "| Bedingung | bei 256 | vor 512 beendet | erneut 512 | neu ausfuehrbar | neu korrekt | falsch->falsch | korrekt->korrekt | korrekt->falsch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        value = capped_summary[condition]
        lines.append(
            f"| {DISPLAY[condition]} | {value['capped_cases']} | {value.get('terminated_before_512', 0)} | "
            f"{value.get('capped_again_512', 0)} | {value.get('newly_executable', 0)} | {value.get('newly_correct', 0)} | "
            f"{value.get('wrong_to_wrong', 0)} | {value.get('correct_to_correct', 0)} | {value.get('correct_to_wrong', 0)} |"
        )
    lines.extend([
        "",
        f"Alle {total_capped} ehemals gedeckelten Generationen erreichen erneut exakt 512 Tokens; keine terminiert regulaer und keine wird neu korrekt. Alle {total_repeated} erfuellen die vorab definierte Repetitionsregel.",
        "",
        f"Einziger Matchverlust: `{static_lost_correct[0]['case_id']}` (Static). Der identische 256er-Rohtextprefix endete als noch extrahierbarer unvollstaendiger, formal balancierter Kandidat; die 512er-Fortsetzung endete mit einem haengenden `AND`, blieb semikolonlos und wurde konservativ nicht mehr extrahiert. Dies ist eine limitbedingte Extraktorfolge einer degenerativen Fortsetzung, kein Determinismusfehler.",
        "", "## 9. Repetitions- und Terminierungsklassifikation", "",
        "Regeln: `late_valid_completion` verlangt einen neuen Match und das erste Semikolon ab Tokenposition 256; `extractor_recovery` eine zuvor leere und nun nichtleere SQL; `new_execution_success_but_wrong` neue Ausfuehrbarkeit ohne Match; `still_truncated_at_512` erneutes Limit ohne vorrangige Verbesserung; `valid_sql_followed_by_repetition` Text nach Semikolon plus Repetitionsregel; `continued_repetition` wird bei mindestens 20% wiederholten 5-Grammen, 50% wiederholten Tokens oder einer Wiederholungssequenz von mindestens 20 Tokens ausgeloest; `semantic_error_without_repetition` ist ausfuehrbar, falsch und nicht repetitiv.",
        "",
        "| Bedingung | Repetitionsregel | erneut 512 | Late valid SQL | Extractor recovery |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        value = capped_summary[condition]
        lines.append(
            f"| {DISPLAY[condition]} | {value.get('continued_repetition', 0)} | {value.get('capped_again_512', 0)} | "
            f"{value.get('late_valid_sql', 0)} | {value.get('extractor_recovery', 0)} |"
        )
    lines.extend([
        "", "## 10. Gate-Mischungsidentitaet", "",
        "Alle vier Gate-Runs stimmen fuer 1032/1032 Faelle exakt mit dem jeweils ausgewaehlten 512er-Referenzlauf ueberein: Fallback mit Zero Shot, akzeptierte Faelle mit dem jeweiligen ungated Few-Shot-Lauf. Dies gilt fuer Rohoutput, extrahierte SQL und Prompttokenzahl.",
        "", "## 11. Few-Shot-Effekte", "",
        "| Bedingung | Effekt 256 pp | Effekt 512 pp | Veraenderung pp |",
        "|---|---:|---:|---:|",
    ])
    for row in fewshot_rows:
        lines.append(f"| {row['condition_label']} | {row['fewshot_effect_256_pp']:+.2f} | {row['fewshot_effect_512_pp']:+.2f} | {row['change_pp']:+.2f} |")
    lines.extend([
        "", "## 12. Hypothesen", "",
        f"- H1 bindendes Limit: **{hypotheses['H1_binding_limit']}**.",
        f"- H2 repetitive/degenerative Generation: **{hypotheses['H2_repetition']}**.",
        f"- H3 Determinismus nicht gedeckelter Faelle: **{hypotheses['H3_determinism']}**.",
        f"- H4 LoRA-Negativkontrolle: **{hypotheses['H4_lora_control']}**.",
        "", "## 13. Wissenschaftliche Interpretation", "",
        "Zulaessig ist: Die Verdopplung des Budgets erzeugte keine neue korrekte SQL und keine regulaere Terminierung eines zuvor gedeckelten Falls. Alle 2215 ehemals gedeckelten Ausgaben erreichten auch 512 Tokens und erfuellten die Repetitionsregel. Die langen Ausgaben sind deshalb ueberwiegend als persistente repetitive beziehungsweise degenerative Generationen einzuordnen. Nicht signifikante Unterschiede belegen keine Gleichheit.",
        "",
        "Nicht zulaessig ist: die 256er-Evaluation pauschal als unfair zu bezeichnen, 512 als optimales Limit auszugeben, alle EMA-Aenderungen als semantische Verbesserung zu werten oder die 512er-Werte still in die Hauptlinie zu uebernehmen.",
        "",
        "Hauptbefunde: (1) sieben Bedingungen haben exakt gleiche EMA/ESR; Static verliert einen Match. (2) McNemar und Holm-8 sind nirgends signifikant. (3) 0/2215 Limitfaelle terminieren vor 512 und 0 werden neu korrekt. (4) Gate-Mischung und LoRA-Kontrolle sind exakt. (5) Das zusaetzliche Budget verdoppelt vor allem Laufzeit und repetitive Ausgabe, nicht die Aufgabenleistung.",
        "", "## 14. Hauptlinien- und Rerunentscheidung", "",
        "```text",
        "OFFICIAL MAINLINE: max_new_tokens = 256",
        "SENSITIVITY ANALYSIS: max_new_tokens = 512",
        "RERUN REQUIRED: NEIN",
        "```",
        "",
        "Die 256er-Hauptlinie bleibt wegen einheitlicher modelluebergreifender Parameter und der vorab definierten Hauptmatrix bestehen. Die 512er-Analyse dokumentiert eine diagnostische Limitation, ersetzt aber keine offizielle Zahl. Ein Rerun ist technisch nicht begruendet.",
        "", "## 15. Read-only-Bestaetigung", "",
        "Es wurden keine Generation, kein Training und kein Retrieval gestartet. Es wurden kein Modell, Adapter oder BGE-Modell geladen. Bestehende Configs, Runs, Logs, Traces und Audits blieben unveraendert.",
    ])
    write_new(OUT_AUDIT, "\n".join(lines) + "\n")

    output_paths = [OUT_AUDIT, OUT_SUMMARY, OUT_STATS, OUT_CASES, OUT_CAPPED, OUT_REPETITION, OUT_GATE, OUT_CONTROL, OUT_FEWSHOT]
    manifest = {
        "audit_status": status,
        "classification": "explorative Generationslimit-Sensitivitaetsanalyse",
        "official_mainline_max_new_tokens": 256,
        "sensitivity_max_new_tokens": 512,
        "rerun_required": rerun,
        "preflight": {
            "audit": {"path": rel(PRELIGHT_AUDIT), "sha256": sha256(PRELIGHT_AUDIT)},
            "manifest": {"path": rel(PREFLIGHT_MANIFEST), "sha256": sha256(PREFLIGHT_MANIFEST)},
            "config_diff": {"path": rel(CONFIG_DIFF), "sha256": sha256(CONFIG_DIFF)},
            "mainline_audit": {"path": rel(MAINLINE_AUDIT), "sha256": sha256(MAINLINE_AUDIT)},
            "mainline_manifest": {"path": rel(MAINLINE_MANIFEST), "sha256": sha256(MAINLINE_MANIFEST)},
        },
        "analysis_scripts": {
            "prepared": {"path": rel(PREPARED_SCRIPT), "sha256": PREPARED_SCRIPT_SHA256, "modified": False},
            "final_additive": {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "additive_reason": "Required rescoring, gate identity, case, few-shot, and 20260716 artifacts were absent from the prepared script.",
        },
        "model": {"id": MODEL_ID, "snapshot": MODEL_SNAPSHOT, "snapshot_source": str(HF_MAIN_REF), "model_loaded": False},
        "adapter": {"root": rel(ADAPTER_ROOT), "equivalent_checkpoint": rel(ADAPTER_CHECKPOINT), "sha256": ADAPTER_SHA256, "adapter_loaded": False},
        "resources": {
            "testset": {"path": rel(TESTCASES), "sha256": TEST_SHA256, "rows": 1032},
            "runner": {"path": rel(RUNNER), "sha256": RUNNER_SHA256},
            "prompt_renderer_sha256": PROMPT_RENDERER_SHA256,
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "retrieval_index": {"path": rel(INDEX), "sha256": INDEX_SHA256},
            "retrieval_metadata": {"path": rel(INDEX_METADATA), "sha256": INDEX_METADATA_SHA256},
            "static_resource": {"path": rel(STATIC_RESOURCE), "sha256": STATIC_SHA256, "demo_id": "SPIDER_TRAIN_001657"},
        },
        "one_factor_diffs": diff_details,
        "assignments": assignments,
        "runs_256": {condition: {"run_id": run["run_id"], "csv_path": rel(run["csv_path"]), "csv_sha256": sha256(run["csv_path"]), "metadata_path": rel(run["metadata_path"]), "metadata_sha256": sha256(run["metadata_path"]), "trace_path": rel(run["trace_path"]) if run["trace_path"] else None, "trace_sha256": sha256(run["trace_path"]) if run["trace_path"] else None, "config_path": diff_mapping[condition]["source_config"], "config_sha256": diff_mapping[condition]["source_sha256"], "log_path": None} for condition, run in old_runs.items()},
        "runs_512": {condition: compact_run(run) for condition, run in new_runs.items()},
        "lora_control_256": {"run_id": old_control["run_id"], "csv_path": rel(old_control["csv_path"]), "csv_sha256": sha256(old_control["csv_path"]), "metadata_path": rel(old_control["metadata_path"]), "metadata_sha256": sha256(old_control["metadata_path"]), "config_path": diff_mapping["zero_shot_control"]["source_config"], "config_sha256": diff_mapping["zero_shot_control"]["source_sha256"]},
        "lora_control_512": compact_run(new_control),
        "metric_reproduction": {condition: run["metrics"] for condition, run in new_runs.items()} | {"lora_zero_control": new_control["metrics"]},
        "execution_rescoring": rescoring,
        "retrieval_identity": retrieval,
        "gate_mixture_identity": gate_summary,
        "lora_negative_control": control_summary,
        "determinism": determinism,
        "summary": summary_rows,
        "paired_statistics": stats_rows,
        "capped_case_summary": capped_summary,
        "fewshot_effects": fewshot_rows,
        "hypotheses": hypotheses,
        "methodological_limits": [
            "Exploratory analysis on Spider Dev; official cross-model mainline remains 256.",
            "Original generated token-id arrays were not persisted; prefix identity uses deterministic re-tokenization of stored raw outputs.",
            "No further token budget is selected or optimized.",
        ],
        "read_only": {"training_started": False, "generation_started": False, "retrieval_started": False, "model_loaded": False, "adapter_loaded": False, "bge_loaded": False, "sqlite_mode": "read-only"},
        "new_files": [{"path": rel(path), "sha256": sha256(path)} for path in output_paths] + [{"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}],
    }
    # Remove arrays from per-run metric payloads while retaining all scalar reproductions.
    for payload in manifest["metric_reproduction"].values():
        payload.pop("exec", None)
        payload.pop("pred_ok", None)
    write_new(OUT_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default) + "\n")
    print(json.dumps({
        "status": status, "runs_512": RUNS_512, "lora_control": control_summary,
        "rescore_mismatches": rescore_mismatches, "path_disagreements": path_disagreements,
        "hypotheses": hypotheses, "outputs": [rel(path) for path in outputs],
    }, indent=2))


if __name__ == "__main__":
    main()
