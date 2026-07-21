#!/usr/bin/env python3
"""Read-only analysis of the complete Qwen 3.5 2B Base/LoRA-v2 8x8 matrix.

This script only reads completed evaluation artifacts and Spider SQLite
databases. It never loads a model, adapter, tokenizer, or embedding model. All
derived outputs are created exclusively and existing files are never replaced.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR_SCRIPT = ROOT / "scripts/analyze_qwen35_2b_base_and_lora_v2_evaluations.py"

OUT_SUMMARY = ROOT / "audits/derived/qwen35_2b_complete_8x8_evaluation_summary_20260715.json"
OUT_CASES = ROOT / "audits/derived/qwen35_2b_complete_8x8_base_vs_lora_case_comparison_20260715.csv"
OUT_BASE_LORA = ROOT / "audits/derived/qwen35_2b_complete_8x8_base_vs_lora_statistics_20260715.csv"
OUT_BASE_FS = ROOT / "audits/derived/qwen35_2b_complete_8x8_base_fewshot_statistics_20260715.csv"
OUT_LORA_FS = ROOT / "audits/derived/qwen35_2b_complete_8x8_lora_fewshot_statistics_20260715.csv"
OUT_INTERACTION = ROOT / "audits/derived/qwen35_2b_complete_8x8_interaction_analysis_20260715.csv"
OUT_STATIC_CROSS = ROOT / "audits/derived/qwen35_2b_complete_static_cross_model_comparison_20260715.csv"

BASE_RUNS = {
    "zero_shot": "run_base_20260627_211410",
    "top1": "run_base_20260712_171240",
    "top1_gate070": "run_base_20260712_183739",
    "top1_gate085": "run_base_20260712_194508",
    "static_seed42": "run_base_20260715_091427",
    "structure": "run_base_20260712_202105",
    "structure_gate070": "run_base_20260715_102049",
    "structure_gate085": "run_base_20260715_112920",
}


def load_prior() -> Any:
    spec = importlib.util.spec_from_file_location("qwen2_prior_audit", PRIOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import prior audited Qwen 2B analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = load_prior()
C = Q.COMMON


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=C.json_default)
        handle.write("\n")


def trace_demo_ids(run: dict[str, Any]) -> list[str | None]:
    return [Q.trace_signature(row)[0] for row in run["traces"]]


def static_reference(run_id: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = Q.run_paths(run_id)
    rows = Q.load_csv(csv_path)
    metadata = Q.load_json(metadata_path)
    traces = Q.load_jsonl(trace_path)
    if len(rows) != 1032 or len(traces) != 1032:
        raise RuntimeError(f"Incomplete static cross-model reference: {run_id}")
    if [row["id"] for row in rows] != [row["id"] for row in tests]:
        raise RuntimeError(f"Static cross-model case mismatch: {run_id}")
    demo_ids = [Q.trace_signature(row)[0] for row in traces]
    if set(demo_ids) != {"SPIDER_TRAIN_001657"}:
        raise RuntimeError(f"Static demo mismatch: {run_id}")
    return {
        "run_id": run_id,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha256(metadata_path),
        "trace_path": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256(trace_path),
        "ema": float(np.mean([C.as_bool(row["exec_match"]) for row in rows])),
        "correct": sum(C.as_bool(row["exec_match"]) for row in rows),
        "demo_id": "SPIDER_TRAIN_001657",
        "k": metadata.get("run_k"),
        "prompt_format": metadata.get("run_prompt_format"),
        "model_id": metadata.get("run_model_id"),
        "adapter": metadata.get("run_adapter"),
    }


def static_cross_model(
    tests: list[dict[str, Any]], base: dict[str, dict[str, Any]], lora: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = {
        "qwen35_2b_base": {
            "run_id": base["static_seed42"]["run_id"],
            "ema": base["static_seed42"]["metrics"]["ema"],
            "correct": base["static_seed42"]["metrics"]["correct"],
            "demo_id": "SPIDER_TRAIN_001657",
            "k": 1,
            "model_id": Q.MODEL_ID,
            "adapter": "base",
        },
        "qwen35_2b_lora_v2": {
            "run_id": lora["static_seed42"]["run_id"],
            "ema": lora["static_seed42"]["metrics"]["ema"],
            "correct": lora["static_seed42"]["metrics"]["correct"],
            "demo_id": "SPIDER_TRAIN_001657",
            "k": 1,
            "model_id": Q.MODEL_ID,
            "adapter": Q.ADAPTER_ALIAS,
        },
        "llama32_3b_instruct_base": static_reference(Q.LLAMA_BASE_RUNS["static_seed42"], tests),
        "llama32_3b_instruct_lora_v2": static_reference(Q.LLAMA_LORA_RUNS["static_seed42"], tests),
        "qwen35_9b_lora_v2": static_reference(Q.QWEN9_LORA_RUNS["static_seed42"], tests),
    }
    rows = []
    for model, item in refs.items():
        rows.append({
            "model_role": model,
            "run_id": item["run_id"],
            "ema": item["ema"],
            "correct": item["correct"],
            "demo_id": item["demo_id"],
            "k": item["k"],
            "full_schema": True,
            "semantic_static_control": True,
            "cross_family_comparability": "B",
        })
    rows.append({
        "model_role": "qwen35_9b_base",
        "run_id": None,
        "ema": None,
        "correct": None,
        "demo_id": None,
        "k": None,
        "full_schema": None,
        "semantic_static_control": False,
        "cross_family_comparability": "MISSING_MATCHED_RUN",
    })
    return rows, refs


def main() -> None:
    outputs = [
        OUT_SUMMARY, OUT_CASES, OUT_BASE_LORA, OUT_BASE_FS, OUT_LORA_FS,
        OUT_INTERACTION, OUT_STATIC_CROSS,
    ]
    for path in outputs:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")

    tests = Q.load_jsonl(Q.TESTCASES)
    if len(tests) != 1032 or sha256(Q.TESTCASES) != Q.TEST_SHA256:
        raise RuntimeError("Spider Dev identity failure")

    base = {condition: Q.audit_run(condition, "base", run_id, tests) for condition, run_id in BASE_RUNS.items()}
    lora = {condition: Q.audit_run(condition, "lora", run_id, tests) for condition, run_id in Q.LORA_RUNS.items()}

    static_base_ids = trace_demo_ids(base["static_seed42"])
    static_lora_ids = trace_demo_ids(lora["static_seed42"])
    static_check = {
        "base_rows": len(static_base_ids),
        "lora_rows": len(static_lora_ids),
        "base_unique_demo_ids": sorted(set(static_base_ids)),
        "lora_unique_demo_ids": sorted(set(static_lora_ids)),
        "base_lora_same_demo_cases": sum(a == b for a, b in zip(static_base_ids, static_lora_ids)),
        "all_expected_demo": set(static_base_ids) == set(static_lora_ids) == {"SPIDER_TRAIN_001657"},
        "resource_path": str(Q.STATIC_RESOURCE.relative_to(ROOT)),
        "resource_sha256": sha256(Q.STATIC_RESOURCE),
    }
    if not static_check["all_expected_demo"] or static_check["base_lora_same_demo_cases"] != 1032:
        raise RuntimeError(f"Static consistency failure: {static_check}")

    structure_consistency: dict[str, Any] = {}
    gate_reference_checks: dict[str, Any] = {}
    for condition in ["structure_gate070", "structure_gate085"]:
        structure_consistency[condition] = C.compare_trace_sets(
            base["structure"]["traces"], base[condition]["traces"]
        )
        gate_reference_checks[condition] = Q.gate_reference_check(base, condition, "structure")
        trace_check = structure_consistency[condition]
        gate_check = gate_reference_checks[condition]
        if trace_check["different_demo_ids"] or trace_check["different_scores"]:
            raise RuntimeError(f"Structure trace mismatch: {condition}")
        if min(
            gate_check["prompt_token_matches_selected_reference"],
            gate_check["raw_output_matches_selected_reference"],
            gate_check["pred_sql_matches_selected_reference"],
        ) != 1032:
            raise RuntimeError(f"Gate reference mismatch: {condition}")

    overlap = Q.retrieval_overlap(tests)
    if any(overlap[key] for key in ["id_overlap", "question_overlap", "sql_overlap", "pair_overlap"]):
        raise RuntimeError(f"Retrieval leakage: {overlap}")

    rescoring = {"base": C.execution_rescore(base), "lora_v2": C.execution_rescore(lora)}
    rescore_mismatches = sum(
        details[path][metric]
        for role in rescoring.values()
        for details in role.values()
        for path in ["existing_runner_path", "independent_sqlite_path"]
        for metric in ["esr_mismatch_count", "ema_mismatch_count"]
    )
    if rescore_mismatches:
        raise RuntimeError(f"Execution rescoring mismatches: {rescore_mismatches}")

    rng = np.random.default_rng(Q.BOOTSTRAP_SEED)
    base_lora_stats = [
        C.paired_stats(base[c]["exec"], lora[c]["exec"], comparison="Qwen 2B Base vs LoRA v2", condition=c, rng=rng)
        for c in Q.CONDITIONS
    ]
    C.holm_adjust(base_lora_stats)
    base_fs_stats = [
        C.paired_stats(base["zero_shot"]["exec"], base[c]["exec"], comparison="Base Zero Shot vs condition", condition=c, rng=rng)
        for c in Q.CONDITIONS[1:]
    ]
    C.holm_adjust(base_fs_stats)
    lora_fs_stats = [
        C.paired_stats(lora["zero_shot"]["exec"], lora[c]["exec"], comparison="LoRA Zero Shot vs condition", condition=c, rng=rng)
        for c in Q.CONDITIONS[1:]
    ]
    C.holm_adjust(lora_fs_stats)

    interaction = []
    for condition in Q.CONDITIONS[1:]:
        base_effect = float(base[condition]["exec"].mean() - base["zero_shot"]["exec"].mean())
        lora_effect = float(lora[condition]["exec"].mean() - lora["zero_shot"]["exec"].mean())
        low, high = C.bootstrap_did(
            base["zero_shot"]["exec"], base[condition]["exec"],
            lora["zero_shot"]["exec"], lora[condition]["exec"], rng=rng,
        )
        interaction.append({
            "condition": condition,
            "condition_label": Q.DISPLAY[condition],
            "base_fewshot_effect": base_effect,
            "lora_fewshot_effect": lora_effect,
            "difference_in_differences": lora_effect - base_effect,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "bootstrap_seed": Q.BOOTSTRAP_SEED,
            "bootstrap_resamples": Q.BOOTSTRAP_RESAMPLES,
        })

    structure_gate_targeted = []
    for condition in ["structure_gate070", "structure_gate085"]:
        structure_gate_targeted.append(C.paired_stats(
            base["structure"]["exec"], base[condition]["exec"],
            comparison="Base Structure ungated vs gate", condition=condition, rng=rng,
        ))

    cases = Q.build_case_rows(tests, base, lora)
    static_cross_rows, static_cross_provenance = static_cross_model(tests, base, lora)

    summary = {
        "schema_version": 1,
        "purpose": "complete_qwen35_2b_base_vs_official_lora_v2_8x8_evaluation_audit",
        "status": "PASS_WITH_METHODICAL_LIMITATIONS",
        "generation_started_by_analysis": False,
        "model_loaded_by_analysis": False,
        "adapter_loaded_by_analysis": False,
        "embedding_model_loaded_by_analysis": False,
        "prior_audit": "audits/audit_qwen35_2b_base_and_lora_v2_evaluations_results_comparison_20260715.md",
        "prior_manifest": "audits/qwen35_2b_base_and_lora_v2_evaluations_manifest_20260715.json",
        "analysis_code_provenance": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "prior_qwen2_analysis_script": str(PRIOR_SCRIPT.relative_to(ROOT)),
            "prior_qwen2_analysis_script_sha256": sha256(PRIOR_SCRIPT),
            "shared_llama_analysis_script": "scripts/analyze_llama32_3b_instruct_lora_v2_evaluations.py",
            "shared_llama_analysis_script_sha256": sha256(ROOT / "scripts/analyze_llama32_3b_instruct_lora_v2_evaluations.py"),
        },
        "official_model": {
            "registry_key": Q.MODEL_REGISTRY_KEY,
            "model_id": Q.MODEL_ID,
            "model_revision": Q.MODEL_REVISION,
            "tokenizer_revision": Q.TOKENIZER_REVISION,
            "model_type": Q.MODEL_TYPE,
        },
        "official_adapter": {
            "root": str(Q.ADAPTER_ROOT.relative_to(ROOT)),
            "best_checkpoint": str(Q.BEST_CHECKPOINT.relative_to(ROOT)),
            "adapter_sha256": sha256(Q.ADAPTER_ROOT / "adapter_model.safetensors"),
            "root_equals_best": sha256(Q.ADAPTER_ROOT / "adapter_model.safetensors") == sha256(Q.BEST_CHECKPOINT / "adapter_model.safetensors"),
        },
        "testset": {"path": str(Q.TESTCASES.relative_to(ROOT)), "sha256": sha256(Q.TESTCASES), "rows": len(tests)},
        "new_base_runs": {condition: Q.compact(base[condition]) for condition in ["static_seed42", "structure_gate070", "structure_gate085"]},
        "base_runs": {condition: Q.compact(base[condition]) for condition in Q.CONDITIONS},
        "lora_v2_runs": {condition: Q.compact(lora[condition]) for condition in Q.CONDITIONS},
        "static_consistency": static_check,
        "structure_consistency": structure_consistency,
        "gate_reference_checks": gate_reference_checks,
        "retrieval_overlap": overlap,
        "execution_rescoring": rescoring,
        "execution_rescoring_mismatch_count": rescore_mismatches,
        "base_vs_lora_statistics": base_lora_stats,
        "base_fewshot_vs_zero_statistics": base_fs_stats,
        "lora_fewshot_vs_zero_statistics": lora_fs_stats,
        "difference_in_differences": interaction,
        "base_structure_gate_targeted_statistics": structure_gate_targeted,
        "static_cross_model": static_cross_provenance,
        "statistics": {
            "mcnemar": "exact two-sided binomial test",
            "bootstrap_seed": Q.BOOTSTRAP_SEED,
            "bootstrap_resamples": Q.BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "holm_families": {"base_vs_lora": 8, "base_fewshot": 7, "lora_fewshot": 7},
        },
        "comparability": {
            "all_eight_qwen2_base_vs_lora": "A",
            "within_qwen2": "A",
            "static_cross_family": "B",
            "structure_gates": "exploratory",
        },
        "warnings": [
            "Spider Dev is development-facing rather than an untouched final test set.",
            "Structure Gate 0.70 and 0.85 are exploratory interaction analyses.",
            "The Qwen registry does not persist an explicit revision string per historical run; the local snapshot is reconstructed from cache and training provenance.",
            "Two Spider Dev gold queries fail under the current SQLite environment in both rescoring paths, while stored ESR/EMA are nevertheless reproduced exactly.",
        ],
        "rerun_required": False,
        "new_experiments_missing": ["matched Qwen 3.5 9B Base Static Seed-42 run for a complete cross-size static comparison"],
    }

    Q.write_csv_new(OUT_CASES, cases)
    Q.write_csv_new(OUT_BASE_LORA, base_lora_stats)
    Q.write_csv_new(OUT_BASE_FS, base_fs_stats)
    Q.write_csv_new(OUT_LORA_FS, lora_fs_stats)
    Q.write_csv_new(OUT_INTERACTION, interaction)
    Q.write_csv_new(OUT_STATIC_CROSS, static_cross_rows)
    write_json_new(OUT_SUMMARY, summary)

    print(json.dumps({
        "status": summary["status"],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "base_ema": {condition: base[condition]["metrics"]["ema"] for condition in Q.CONDITIONS},
        "lora_ema": {condition: lora[condition]["metrics"]["ema"] for condition in Q.CONDITIONS},
        "rescore_mismatches": rescore_mismatches,
    }, indent=2))


if __name__ == "__main__":
    main()
