#!/usr/bin/env python3
"""Read-only analysis of Qwen 3.5 2B max-new-tokens 256 vs 512.

Run this only after all nine full sensitivity evaluations have completed. The
script discovers runs by their exact config path and config SHA256, never by
timestamp. It loads a local tokenizer for deterministic token-prefix analysis,
but never loads a language model, adapter, or embedding model.
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
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PRIOR_ANALYSIS = ROOT / "scripts/analyze_qwen35_2b_base_and_lora_v2_evaluations.py"
SNAPSHOT = Path(
    "/home/ec2-user/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_RESAMPLES = 10_000

OUT_AUDIT = ROOT / "audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260715.md"
OUT_MANIFEST = ROOT / "audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260715.json"
OUT_SUMMARY = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_summary_20260715.csv"
OUT_STATS = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_paired_statistics_20260715.csv"
OUT_CAPPED = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_capped_case_analysis_20260715.csv"
OUT_REPETITION = ROOT / "audits/derived/qwen35_2b_base_maxnew256_vs_512_repetition_analysis_20260715.csv"
OUT_CONTROL = ROOT / "audits/derived/qwen35_2b_lora_zero_256_vs_512_control_20260715.csv"

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

SOURCE_RUNS = {
    "zero_shot": "run_base_20260627_211410",
    "top1": "run_base_20260712_171240",
    "top1_gate070": "run_base_20260712_183739",
    "top1_gate085": "run_base_20260712_194508",
    "static_seed42": "run_base_20260715_091427",
    "structure": "run_base_20260712_202105",
    "structure_gate070": "run_base_20260715_102049",
    "structure_gate085": "run_base_20260715_112920",
}
SOURCE_LORA_CONTROL = (
    "run_lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_"
    "evalstop_maxlen2048_epochs5_20260714_083452"
)

CONFIG_512 = {
    "zero_shot": "configs/eval_qwen35_2b_base_zero_shot_maxinput1536_maxnew512_sensitivity.json",
    "top1": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top1_k1_full_schema_maxinput2048_maxnew512_sensitivity.json",
    "top1_gate070": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top1_k1_full_schema_similarity_gate070_maxinput2048_maxnew512_sensitivity.json",
    "top1_gate085": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top1_k1_full_schema_similarity_gate085_maxinput2048_maxnew512_sensitivity.json",
    "static_seed42": "configs/eval_qwen35_2b_base_static_fewshot_k1_full_schema_seed42_spidertrain6960_maxinput2048_maxnew512_sensitivity.json",
    "structure": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top10_structure_rerank_v2_k1_full_schema_maxinput2048_maxnew512_sensitivity.json",
    "structure_gate070": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top10_structure_rerank_v2_k1_full_schema_similarity_gate070_maxinput2048_maxnew512_sensitivity.json",
    "structure_gate085": "configs/eval_qwen35_2b_base_dynamic_fewshot_bge_large_top10_structure_rerank_v2_k1_full_schema_similarity_gate085_maxinput2048_maxnew512_sensitivity.json",
}
CONFIG_512_SHA256 = {
    "zero_shot": "c99db8024e68f368caa0707ecbe25713e07bf3b6dcc75f7f9b44de735d56743a",
    "top1": "c5db2bc113df7ea2717a55584d14562fb489b40ad3d168465beb1bf3df428282",
    "top1_gate070": "29f4580667a423b7e7607d851e91fa075fde21f278ef01ec27c92fbb04a04723",
    "top1_gate085": "0a81cc22eefaff6c4b05a693a2f40cb76be9c8317acef485d1e7a2dbccbe191b",
    "static_seed42": "19574e2016f7f5b8097dfc85953870c50b1820ca22e6746029a9ade82291273a",
    "structure": "15df14419a20c83ca9223a41dfed73c0be71ed9fcf3da0f7cc848fd2bf92b18c",
    "structure_gate070": "95514faa920178d9512620c6cc330da2d34b6002a52577cc2243c45f27b681aa",
    "structure_gate085": "56e1016bfdd65c4a79f784f487a321c3c1c1897fb2bb3d31037ad331872eb095",
}
CONFIG_512_LORA = (
    "configs/eval_qwen35_2b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_"
    "bestepoch1_zero_shot_maxinput2048_maxnew512_sensitivity_control.json"
)
CONFIG_512_LORA_SHA256 = "13dc7ea24f82aa76e7971c4f3a8e9a2780996d57973f39d8198b8945086a481b"


def load_prior() -> Any:
    spec = importlib.util.spec_from_file_location("qwen2_sensitivity_common", PRIOR_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import prior audited Qwen analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    module.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    module.COMMON.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    module.COMMON.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    return module


Q = load_prior()
C = Q.COMMON


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


def run_paths(run_id: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "results" / f"{run_id}.csv",
        ROOT / "results" / f"{run_id}_metadata.json",
        ROOT / "results/retrieval_traces" / f"{run_id}_retrieval_traces.jsonl",
    )


def find_completed_run(config_path: str, expected_hash: str) -> tuple[str, dict[str, Any]]:
    actual_hash = sha256(ROOT / config_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Config hash mismatch for {config_path}: expected {expected_hash}, got {actual_hash}"
        )
    matches = []
    for metadata_path in (ROOT / "results").glob("*_metadata.json"):
        metadata = load_json(metadata_path)
        if metadata.get("run_config_path") != config_path:
            continue
        if metadata.get("total_testcases") != 1032 or metadata.get("run_max_new_tokens") != 512:
            continue
        csv_path = Path(str(metadata.get("csv_path", "")))
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        if not csv_path.is_file():
            continue
        matches.append((metadata_path.name.removesuffix("_metadata.json"), metadata))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one complete 512 run for {config_path}, found {len(matches)}")
    return matches[0]


def load_run(run_id: str) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    rows = load_csv(csv_path)
    metadata = load_json(metadata_path)
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032:
        raise RuntimeError(f"Incomplete run: {run_id}")
    traces = load_jsonl(trace_path) if trace_path.is_file() else []
    return {
        "run_id": run_id,
        "rows": rows,
        "metadata": metadata,
        "traces": traces,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "trace_path": trace_path if trace_path.is_file() else None,
    }


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    exec_values = np.asarray([C.as_bool(row["exec_match"]) for row in rows], dtype=np.int8)
    pred_ok = np.asarray([C.as_bool(row["pred_ok"]) for row in rows], dtype=np.int8)
    completion = np.asarray([int(row["completion_tokens"]) for row in rows])
    return {
        "correct": int(exec_values.sum()),
        "ema": float(exec_values.mean()),
        "executable": int(pred_ok.sum()),
        "esr": float(pred_ok.mean()),
        "string_em": float(np.mean([int(row["pred_sql"] == row["gold_sql"]) for row in rows])),
        "normalized_em": float(np.mean([int(C.normalized_sql(row["pred_sql"]) == C.normalized_sql(row["gold_sql"])) for row in rows])),
        "char_accuracy": float(np.mean([C.char_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])),
        "token_accuracy": float(np.mean([C.token_accuracy(row["pred_sql"], row["gold_sql"]) for row in rows])),
        "mean_completion_tokens": float(completion.mean()),
        "max_completion_tokens": int(completion.max()),
        "at_256": int(np.sum(completion == 256)),
        "at_512": int(np.sum(completion == 512)),
        "runtime_seconds": float(np.sum([float(row["generation_time_seconds"]) for row in rows])),
        "exec": exec_values,
        "pred_ok": pred_ok,
    }


def trace_identity(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if not a["traces"] and not b["traces"]:
        return {"applicable": False}
    if len(a["traces"]) != 1032 or len(b["traces"]) != 1032:
        raise RuntimeError("Trace length mismatch")
    comparison = C.compare_trace_sets(a["traces"], b["traces"])
    gate_same = sum(x.get("gate_decision") == y.get("gate_decision") for x, y in zip(a["traces"], b["traces"]))
    if comparison["different_demo_ids"] or comparison["different_scores"] or gate_same != 1032:
        raise RuntimeError(f"Retrieval identity failure: {comparison}, gate_same={gate_same}")
    return {"applicable": True, **comparison, "same_gate_decisions": gate_same}


def repeated_ngram_excess(tokens: list[int], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return sum(max(0, count - 1) for count in counts.values())


def longest_adjacent_repeated_sequence(tokens: list[int]) -> int:
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


def repetition_features(raw: str, token_ids: list[int], tokenizer: Any) -> dict[str, Any]:
    lower = raw.lower()
    identifiers = [x.lower() for x in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", raw)]
    keywords = {"select", "from", "where", "join", "on", "and", "or", "as", "in", "with", "group", "order", "by", "limit"}
    fragments = Counter(x for x in identifiers if x not in keywords)
    semicolon_char = raw.find(";")
    semicolon_token = None
    if semicolon_char >= 0:
        prefix_ids = tokenizer.encode(raw[:semicolon_char + 1], add_special_tokens=False)
        semicolon_token = len(prefix_ids) - 1 if prefix_ids else None
    return {
        "repeated_token_fraction": 1.0 - (len(set(token_ids)) / len(token_ids)) if token_ids else 0.0,
        "repeated_3gram_excess": repeated_ngram_excess(token_ids, 3),
        "repeated_5gram_excess": repeated_ngram_excess(token_ids, 5),
        "longest_adjacent_repeated_sequence": longest_adjacent_repeated_sequence(token_ids),
        "select_count": len(re.findall(r"\bselect\b", lower)),
        "with_count": len(re.findall(r"\bwith\b", lower)),
        "repeated_subquery_count": max(0, lower.count("(select") - 1),
        "repeated_identifier_fragment_excess": sum(max(0, count - 1) for count in fragments.values()),
        "first_semicolon_char": semicolon_char if semicolon_char >= 0 else None,
        "first_semicolon_token_approx": semicolon_token,
        "content_after_first_semicolon": raw[semicolon_char + 1:].strip() if semicolon_char >= 0 else "",
    }


def classify_capped(old: dict[str, str], new: dict[str, str], features: dict[str, Any]) -> str:
    new_tokens = int(new["completion_tokens"])
    old_pred = bool(old["pred_sql"].strip())
    new_pred = bool(new["pred_sql"].strip())
    repeated = features["repeated_5gram_excess"] >= 5 or features["repeated_token_fraction"] >= 0.55
    if new_tokens == 512 and not new_pred:
        return "still_truncated_at_512"
    if not old_pred and new_pred:
        return "extractor_recovery" if not C.as_bool(new["exec_match"]) else "late_valid_completion"
    if features["content_after_first_semicolon"] and repeated:
        return "valid_sql_followed_by_repetition"
    if repeated:
        return "continued_repetition"
    if new_tokens < 512 and new_pred:
        return "late_valid_completion" if C.as_bool(new["exec_match"]) else "semantic_error_without_repetition"
    return "other"


def main() -> None:
    outputs = [OUT_AUDIT, OUT_MANIFEST, OUT_SUMMARY, OUT_STATS, OUT_CAPPED, OUT_REPETITION, OUT_CONTROL]
    for path in outputs:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
    if not SNAPSHOT.is_dir():
        raise RuntimeError(f"Local tokenizer snapshot missing: {SNAPSHOT}")

    old = {condition: load_run(run_id) for condition, run_id in SOURCE_RUNS.items()}
    new: dict[str, dict[str, Any]] = {}
    for condition, config_path in CONFIG_512.items():
        run_id, metadata = find_completed_run(config_path, CONFIG_512_SHA256[condition])
        new[condition] = load_run(run_id)
        if metadata.get("run_adapter") != "base" or metadata.get("run_model_id") != "Qwen/Qwen3.5-2B-Base":
            raise RuntimeError(f"Model/adapter mismatch: {condition}")

    old_control = load_run(SOURCE_LORA_CONTROL)
    new_control_id, control_meta = find_completed_run(CONFIG_512_LORA, CONFIG_512_LORA_SHA256)
    new_control = load_run(new_control_id)
    if control_meta.get("run_adapter") == "base":
        raise RuntimeError("LoRA negative control resolved to base")

    case_ids = [row["id"] for row in old["zero_shot"]["rows"]]
    for role_runs in [old, new]:
        for condition, run in role_runs.items():
            if [row["id"] for row in run["rows"]] != case_ids:
                raise RuntimeError(f"Case order mismatch: {condition}")
    if [row["id"] for row in new_control["rows"]] != case_ids:
        raise RuntimeError("Control case order mismatch")

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    stats_rows = []
    capped_rows = []
    repetition_rows = []
    determinism: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}

    for condition in CONDITIONS:
        a, b = old[condition], new[condition]
        ma, mb = metrics(a["rows"]), metrics(b["rows"])
        retrieval[condition] = trace_identity(a, b)
        uncapped_diff = []
        capped_prefix_diff = []
        for old_row, new_row in zip(a["rows"], b["rows"]):
            capped = int(old_row["completion_tokens"]) == 256
            if not capped:
                if old_row["raw_output"] != new_row["raw_output"] or old_row["pred_sql"] != new_row["pred_sql"]:
                    uncapped_diff.append(old_row["id"])
                continue
            old_ids = tokenizer.encode(old_row["raw_output"], add_special_tokens=False)
            new_ids = tokenizer.encode(new_row["raw_output"], add_special_tokens=False)
            if old_ids != new_ids[:len(old_ids)]:
                capped_prefix_diff.append(old_row["id"])
            features = repetition_features(new_row["raw_output"], new_ids, tokenizer)
            category = classify_capped(old_row, new_row, features)
            capped_rows.append({
                "condition": condition,
                "case_id": old_row["id"],
                "db_id": old_row["db_id"],
                "old_exec_match": C.as_bool(old_row["exec_match"]),
                "new_exec_match": C.as_bool(new_row["exec_match"]),
                "old_pred_ok": C.as_bool(old_row["pred_ok"]),
                "new_pred_ok": C.as_bool(new_row["pred_ok"]),
                "old_sql_extracted": bool(old_row["pred_sql"].strip()),
                "new_sql_extracted": bool(new_row["pred_sql"].strip()),
                "new_completion_tokens": int(new_row["completion_tokens"]),
                "terminated_between_257_and_511": 256 < int(new_row["completion_tokens"]) < 512,
                "reached_512": int(new_row["completion_tokens"]) == 512,
                "token_prefix_match": old_ids == new_ids[:len(old_ids)],
                "classification": category,
            })
            repetition_rows.append({"condition": condition, "case_id": old_row["id"], **features, "classification": category})
        determinism[condition] = {
            "uncapped_cases": 1032 - ma["at_256"],
            "uncapped_output_mismatch_count": len(uncapped_diff),
            "uncapped_output_mismatch_case_ids": uncapped_diff,
            "capped_cases": ma["at_256"],
            "capped_prefix_mismatch_count": len(capped_prefix_diff),
            "capped_prefix_mismatch_case_ids": capped_prefix_diff,
        }
        paired = C.paired_stats(ma["exec"], mb["exec"], comparison="Base max_new_tokens 256 vs 512", condition=condition, rng=rng)
        stats_rows.append(paired)
        summary_rows.append({
            "condition": condition,
            "condition_label": DISPLAY[condition],
            "run_256": a["run_id"],
            "run_512": b["run_id"],
            "ema_256": ma["ema"],
            "ema_512": mb["ema"],
            "ema_delta_percentage_points": 100 * (mb["ema"] - ma["ema"]),
            "esr_256": ma["esr"],
            "esr_512": mb["esr"],
            "string_em_256": ma["string_em"],
            "string_em_512": mb["string_em"],
            "normalized_em_256": ma["normalized_em"],
            "normalized_em_512": mb["normalized_em"],
            "char_accuracy_256": ma["char_accuracy"],
            "char_accuracy_512": mb["char_accuracy"],
            "token_accuracy_256": ma["token_accuracy"],
            "token_accuracy_512": mb["token_accuracy"],
            "runtime_seconds_256": ma["runtime_seconds"],
            "runtime_seconds_512": mb["runtime_seconds"],
            "mean_completion_tokens_256": ma["mean_completion_tokens"],
            "mean_completion_tokens_512": mb["mean_completion_tokens"],
            "max_completion_tokens_256": ma["max_completion_tokens"],
            "max_completion_tokens_512": mb["max_completion_tokens"],
            "cases_at_256_limit": ma["at_256"],
            "cases_at_512_limit": mb["at_512"],
        })
    C.holm_adjust(stats_rows)

    control_rows = []
    raw_same = pred_same = metric_same = 0
    for a, b in zip(old_control["rows"], new_control["rows"]):
        same_raw = a["raw_output"] == b["raw_output"]
        same_pred = a["pred_sql"] == b["pred_sql"]
        same_metrics = all(a[key] == b[key] for key in ["pred_ok", "exec_match", "string_exact", "normalized_exact"])
        raw_same += same_raw
        pred_same += same_pred
        metric_same += same_metrics
        control_rows.append({"case_id": a["id"], "raw_output_identical": same_raw, "pred_sql_identical": same_pred, "metrics_identical": same_metrics})

    invalid = {
        "uncapped_output_mismatches": sum(x["uncapped_output_mismatch_count"] for x in determinism.values()),
        "capped_prefix_mismatches": sum(x["capped_prefix_mismatch_count"] for x in determinism.values()),
        "control_raw_mismatches": 1032 - raw_same,
        "control_pred_mismatches": 1032 - pred_same,
        "control_metric_mismatches": 1032 - metric_same,
    }
    status = "PASS" if not any(invalid.values()) else "SENSITIVITY_COMPARISON_INVALID"

    write_csv_new(OUT_SUMMARY, summary_rows)
    write_csv_new(OUT_STATS, stats_rows)
    write_csv_new(OUT_CAPPED, capped_rows)
    write_csv_new(OUT_REPETITION, repetition_rows)
    write_csv_new(OUT_CONTROL, control_rows)

    lines = [
        "# Qwen 3.5 2B Base: max_new_tokens 256 vs 512 Sensitivitaetsanalyse",
        "",
        f"```text\nSENSITIVITY-ANALYSIS-STATUS: {status}\n```",
        "",
        "Die offizielle Hauptlinie bleibt bei `max_new_tokens=256`; die 512er-Laeufe sind explorativ.",
        "",
        "## Ergebnisse",
        "",
        "| Bedingung | EMA 256 | EMA 512 | Delta pp | am Limit 256 | am Limit 512 | Holm-8 p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_condition = {row["condition"]: row for row in stats_rows}
    for row in summary_rows:
        stat = by_condition[row["condition"]]
        lines.append(
            f"| {row['condition_label']} | {100*row['ema_256']:.2f} % | {100*row['ema_512']:.2f} % | "
            f"{row['ema_delta_percentage_points']:+.2f} | {row['cases_at_256_limit']} | {row['cases_at_512_limit']} | "
            f"{stat['holm_adjusted_p']:.6g} |"
        )
    lines.extend([
        "", "## Determinismus", "",
        f"- Nicht gedeckelte Outputabweichungen: {invalid['uncapped_output_mismatches']}",
        f"- Gedeckelte Token-Prefixabweichungen: {invalid['capped_prefix_mismatches']}",
        f"- LoRA-Control Rohoutputabweichungen: {invalid['control_raw_mismatches']}",
        f"- LoRA-Control SQL-Abweichungen: {invalid['control_pred_mismatches']}",
        "", "## Read-only-Bestaetigung", "",
        "Das Skript startete keine Generation und lud weder Modell, Adapter noch BGE-Modell.",
    ])
    write_new(OUT_AUDIT, "\n".join(lines) + "\n")

    manifest = {
        "status": status,
        "official_mainline_max_new_tokens": 256,
        "sensitivity_max_new_tokens": 512,
        "exploratory": True,
        "base_runs_256": SOURCE_RUNS,
        "base_runs_512": {condition: run["run_id"] for condition, run in new.items()},
        "lora_control_256": SOURCE_LORA_CONTROL,
        "lora_control_512": new_control_id,
        "config_512": CONFIG_512,
        "config_512_lora_control": CONFIG_512_LORA,
        "determinism": determinism,
        "retrieval_identity": retrieval,
        "invalidity_counts": invalid,
        "statistics": {"holm_family": 8, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
        "tokenizer_snapshot": str(SNAPSHOT),
        "tokenizer_loaded": True,
        "model_loaded": False,
        "adapter_loaded": False,
        "embedding_model_loaded": False,
        "generation_started": False,
        "outputs": [],
    }
    for path in [OUT_AUDIT, OUT_SUMMARY, OUT_STATS, OUT_CAPPED, OUT_REPETITION, OUT_CONTROL]:
        manifest["outputs"].append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    write_new(OUT_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "invalidity_counts": invalid, "base_runs_512": manifest["base_runs_512"], "lora_control_512": new_control_id}, indent=2))


if __name__ == "__main__":
    main()
