#!/usr/bin/env python3
"""Audit completed Llama 3.2 3B Base evaluations without model execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "audits/derived/llama32_3b_base_evaluation_summary_20260714.json"
OUT_CSV = ROOT / "audits/derived/llama32_3b_base_case_comparison_20260714.csv"
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"

LLAMA_RUNS = {
    "zero_shot": "run_base_20260714_162526",
    "top1": "run_base_20260714_164116",
    "top1_gate070": "run_base_20260714_165432",
    "top1_gate085": "run_base_20260714_170748",
    "static_seed42": "run_base_20260714_172224",
    "structure": "run_base_20260714_173302",
    "structure_gate070": "run_base_20260714_174639",
    "structure_gate085": "run_base_20260714_180015",
}

QWEN_RUNS = {
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

EXPECTED = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "test_sha256": "6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce",
    "template_sha256": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
    "system_prompt_sha256": "d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e",
    "runner_sha256": "a37286649920f4224999b5184e6117ea31f24968ad2c353ff338397c99a7a3c9",
    "retrieval_index": "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15",
    "retrieval_index_sha256": "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc",
    "retrieval_metadata_sha256": "05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55",
    "static_demo": "SPIDER_TRAIN_001657",
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


def atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def as_bool(value: str) -> int:
    return int(str(value).strip().lower() in {"1", "true"})


def exact_mcnemar_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - radius, center + radius]


def paired(a: list[int], b: list[int]) -> dict[str, Any]:
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    return {
        "a_wrong_b_correct_n01": n01,
        "a_correct_b_wrong_n10": n10,
        "net_correct_b_minus_a": n01 - n10,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(n01, n10),
    }


def run_paths(run_id: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "results" / f"{run_id}.csv",
        ROOT / "results" / f"{run_id}_metadata.json",
        ROOT / "results/retrieval_traces" / f"{run_id}_retrieval_traces.jsonl",
    )


def audit_run(condition: str, run_id: str, test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Missing run artifact for {run_id}")
    rows = load_csv(csv_path)
    metadata = load_json(metadata_path)
    if len(rows) != 1032 or metadata.get("total_testcases") != 1032:
        raise RuntimeError(f"Incomplete run {run_id}")
    if [row["id"] for row in rows] != [row["id"] for row in test_rows]:
        raise RuntimeError(f"Test ID/order mismatch in {run_id}")
    for row, test in zip(rows, test_rows):
        if row["db_id"] != test["db_id"] or row["question"] != test["question"] or row["gold_sql"] != test["gold_sql"]:
            raise RuntimeError(f"Test content mismatch in {run_id}: {row['id']}")

    exec_values = [as_bool(row["exec_match"]) for row in rows]
    pred_ok = [as_bool(row["pred_ok"]) for row in rows]
    string_exact = [as_bool(row["string_exact"]) for row in rows]
    normalized_exact = [as_bool(row["normalized_exact"]) for row in rows]
    prompt_tokens = [int(float(row["prompt_tokens"])) for row in rows]
    completion_tokens = [int(float(row["completion_tokens"])) for row in rows]
    numeric_columns = ["char_accuracy", "token_accuracy", "generation_time_seconds", "tokens_per_second"]
    nonfinite = sum(
        not math.isfinite(float(row[column]))
        for row in rows
        for column in numeric_columns
        if row[column] not in {"", None}
    )
    raw_outputs = [row["raw_output"] for row in rows]
    config_path = ROOT / metadata["run_config_path"]
    provenance = metadata.get("provenance", {})
    checks = {
        "model": metadata.get("run_model_id") == EXPECTED["model_id"],
        "revision": metadata.get("run_model_revision") == EXPECTED["revision"],
        "base_adapter": metadata.get("run_adapter") == "base",
        "native_prompt": metadata.get("run_prompt_format") == "llama32_instruct_native_chat",
        "system_prompt": metadata.get("run_system_prompt_sha256") == EXPECTED["system_prompt_sha256"],
        "max_input_2048": metadata.get("run_max_input_tokens") == 2048,
        "max_new_256": metadata.get("run_max_new_tokens") == 256,
        "batch_1": metadata.get("run_generation_batch_size") == 1,
        "extractor": metadata.get("run_extractor_mode") == "sql_first_statement_only",
        "full_testset": metadata.get("run_max_test_samples") == "",
        "no_overlap": metadata.get("run_allow_overlap") is False,
        "test_hash": provenance.get("testcases_sha256") == EXPECTED["test_sha256"],
        "config_hash": provenance.get("config_sha256") == sha256(config_path),
        "template_hash": metadata.get("run_tokenizer_chat_template_sha256") == EXPECTED["template_sha256"],
        "stop_ids": metadata.get("run_generation_eos_token_ids") == [128001, 128008, 128009],
        "pad_id": metadata.get("run_generation_pad_token_id") == 128009,
        "runner_hash": provenance.get("code_sha256", {}).get("runner") == EXPECTED["runner_sha256"],
        "metrics_recomputed": all(
            abs(a - b) < 1e-12
            for a, b in (
                (mean(exec_values), metadata["execution_match_accuracy"]),
                (mean(pred_ok), metadata["execution_success_rate"]),
                (mean(string_exact), metadata["string_exact_match"]),
                (mean(normalized_exact), metadata["normalized_exact_match"]),
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Failed provenance checks for {run_id}: {[key for key, value in checks.items() if not value]}")

    trace_summary = None
    traces: list[dict[str, Any]] = []
    if condition != "zero_shot":
        if not trace_path.is_file():
            raise RuntimeError(f"Missing trace for {run_id}")
        traces = load_jsonl(trace_path)
        if len(traces) != 1032 or [row["id"] for row in traces] != [row["id"] for row in test_rows]:
            raise RuntimeError(f"Trace alignment failure in {run_id}")
        if any(row.get("leakage_status") != "pass" or not row.get("retrieval_success") for row in traces):
            raise RuntimeError(f"Trace leakage/retrieval failure in {run_id}")
        gate_counts: dict[str, int] = {}
        for row in traces:
            if "gate_decision" in row:
                gate_counts[row["gate_decision"]] = gate_counts.get(row["gate_decision"], 0) + 1
        trace_summary = {
            "path": str(trace_path.relative_to(ROOT)),
            "sha256": sha256(trace_path),
            "rows": len(traces),
            "retrieval_success": sum(bool(row.get("retrieval_success")) for row in traces),
            "leakage_pass": sum(row.get("leakage_status") == "pass" for row in traces),
            "unique_selected_demo_ids": len({row.get("selected_example_id") or tuple(row.get("retrieved_ids", [])) for row in traces}),
            "gate_counts": gate_counts,
        }

    return {
        "condition": condition,
        "run_id": run_id,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha256(metadata_path),
        "config_path": metadata["run_config_path"],
        "config_sha256": sha256(config_path),
        "rows": rows,
        "metadata": metadata,
        "traces": traces,
        "checks": checks,
        "metrics": {
            "execution_match_count": sum(exec_values),
            "ema": mean(exec_values),
            "ema_wilson_95": wilson(sum(exec_values), len(exec_values)),
            "execution_success_count": sum(pred_ok),
            "esr": mean(pred_ok),
            "string_exact": mean(string_exact),
            "normalized_exact": mean(normalized_exact),
            "char_accuracy": metadata["char_accuracy_avg"],
            "token_accuracy": metadata["token_accuracy_avg"],
            "duration_seconds": metadata["duration_seconds"],
            "avg_prompt_tokens": metadata["avg_prompt_tokens"],
            "max_prompt_tokens": max(prompt_tokens),
            "prompts_over_2048": sum(value > 2048 for value in prompt_tokens),
            "avg_completion_tokens": metadata["avg_completion_tokens"],
            "max_completion_tokens": max(completion_tokens),
            "completions_at_256": sum(value >= 256 for value in completion_tokens),
            "prediction_failures": len(rows) - sum(pred_ok),
            "empty_predictions": sum(not row["pred_sql"].strip() for row in rows),
            "empty_raw_outputs": sum(not value.strip() for value in raw_outputs),
            "think_outputs": sum("<think>" in value.lower() for value in raw_outputs),
            "markdown_fence_outputs": sum("```" in value for value in raw_outputs),
            "nonfinite_numeric_values": nonfinite,
        },
        "trace_summary": trace_summary,
    }


def trace_signature(row: dict[str, Any]) -> tuple[Any, Any]:
    demo = row.get("selected_example_id") or (row.get("retrieved_ids") or [None])[0]
    score = row.get("retrieval_similarity")
    if score is None:
        values = row.get("retrieved_scores") or [None]
        score = values[0]
    return demo, score


def trace_consistency(runs: dict[str, dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    reference = runs[conditions[0]]["traces"]
    result: dict[str, Any] = {}
    for condition in conditions[1:]:
        candidate = runs[condition]["traces"]
        demo_equal = 0
        score_equal = 0
        deltas = []
        for left, right in zip(reference, candidate):
            left_demo, left_score = trace_signature(left)
            right_demo, right_score = trace_signature(right)
            demo_equal += left_demo == right_demo
            if left_score is not None and right_score is not None:
                delta = abs(float(left_score) - float(right_score))
                deltas.append(delta)
                score_equal += delta <= 1e-12
        result[f"{conditions[0]}_vs_{condition}"] = {
            "same_demo_ids": demo_equal,
            "same_scores": score_equal,
            "mean_absolute_score_delta": mean(deltas) if deltas else None,
            "max_absolute_score_delta": max(deltas) if deltas else None,
        }
    return result


def gate_prompt_consistency(runs: dict[str, dict[str, Any]], gated: str, fewshot_reference: str) -> dict[str, Any]:
    zero_rows = runs["zero_shot"]["rows"]
    fewshot_rows = runs[fewshot_reference]["rows"]
    gated_rows = runs[gated]["rows"]
    traces = runs[gated]["traces"]
    zero_matches = fewshot_matches = prediction_matches = 0
    for zero, fewshot, row, trace in zip(zero_rows, fewshot_rows, gated_rows, traces):
        reference = fewshot if trace["gate_decision"] == "fewshot" else zero
        zero_matches += trace["gate_decision"] == "zero_shot" and row["prompt_tokens"] == zero["prompt_tokens"]
        fewshot_matches += trace["gate_decision"] == "fewshot" and row["prompt_tokens"] == fewshot["prompt_tokens"]
        prediction_matches += row["raw_output"] == reference["raw_output"]
    return {
        "zero_shot_prompt_token_matches": zero_matches,
        "fewshot_prompt_token_matches": fewshot_matches,
        "raw_output_matches_selected_reference": prediction_matches,
        "cases": len(gated_rows),
    }


def load_reference_run(run_id: str, test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path, metadata_path, trace_path = run_paths(run_id)
    rows = load_csv(csv_path)
    metadata = load_json(metadata_path)
    if len(rows) != 1032 or [row["id"] for row in rows] != [row["id"] for row in test_rows]:
        raise RuntimeError(f"Qwen reference misaligned: {run_id}")
    return {
        "run_id": run_id,
        "rows": rows,
        "metadata": metadata,
        "trace": load_jsonl(trace_path) if trace_path.is_file() else [],
        "csv_sha256": sha256(csv_path),
        "metadata_sha256": sha256(metadata_path),
    }


def main() -> None:
    if OUT_JSON.exists() or OUT_CSV.exists():
        raise RuntimeError("Refusing to overwrite existing derived audit output")
    test_rows = load_jsonl(TESTCASES)
    if len(test_rows) != 1032 or sha256(TESTCASES) != EXPECTED["test_sha256"]:
        raise RuntimeError("Unexpected Spider Dev test set")
    llama = {condition: audit_run(condition, run_id, test_rows) for condition, run_id in LLAMA_RUNS.items()}

    zero_exec = [as_bool(row["exec_match"]) for row in llama["zero_shot"]["rows"]]
    within_llama = {}
    for condition, run in llama.items():
        if condition == "zero_shot":
            continue
        within_llama[condition] = paired(zero_exec, [as_bool(row["exec_match"]) for row in run["rows"]])

    top1_consistency = trace_consistency(llama, ["top1", "top1_gate070", "top1_gate085"])
    structure_consistency = trace_consistency(llama, ["structure", "structure_gate070", "structure_gate085"])
    gate_prompt_checks = {
        condition: gate_prompt_consistency(llama, condition, "top1" if condition.startswith("top1") else "structure")
        for condition in ("top1_gate070", "top1_gate085", "structure_gate070", "structure_gate085")
    }
    static_traces = llama["static_seed42"]["traces"]
    static_check = {
        "same_demo_all_cases": all(row.get("retrieved_ids") == [EXPECTED["static_demo"]] for row in static_traces),
        "demo_id": EXPECTED["static_demo"],
        "unique_demo_sets": len({tuple(row.get("retrieved_ids", [])) for row in static_traces}),
    }

    qwen_summary: dict[str, Any] = {}
    cross_model: dict[str, Any] = {}
    qwen_loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for model, mapping in QWEN_RUNS.items():
        qwen_loaded[model] = {}
        qwen_summary[model] = {}
        cross_model[model] = {}
        for condition, run_id in mapping.items():
            reference = load_reference_run(run_id, test_rows)
            qwen_loaded[model][condition] = reference
            qwen_exec = [as_bool(row["exec_match"]) for row in reference["rows"]]
            llama_exec = [as_bool(row["exec_match"]) for row in llama[condition]["rows"]]
            metadata = reference["metadata"]
            qwen_summary[model][condition] = {
                "run_id": run_id,
                "ema": mean(qwen_exec),
                "ema_count": sum(qwen_exec),
                "esr": metadata["execution_success_rate"],
                "max_input_tokens": metadata["run_max_input_tokens"],
                "prompt_format": metadata["run_prompt_format"],
                "csv_sha256": reference["csv_sha256"],
                "metadata_sha256": reference["metadata_sha256"],
            }
            cross_model[model][condition] = {
                "llama_minus_qwen_ema_percentage_points": 100.0 * (mean(llama_exec) - mean(qwen_exec)),
                **paired(qwen_exec, llama_exec),
            }

    retrieval_cross_model: dict[str, Any] = {}
    for model, conditions in qwen_loaded.items():
        retrieval_cross_model[model] = {}
        for condition, reference in conditions.items():
            if not reference["trace"] or not llama[condition]["traces"]:
                continue
            same_demo = same_score = 0
            deltas = []
            for qwen_trace, llama_trace in zip(reference["trace"], llama[condition]["traces"]):
                q_demo, q_score = trace_signature(qwen_trace)
                l_demo, l_score = trace_signature(llama_trace)
                same_demo += q_demo == l_demo
                if q_score is not None and l_score is not None:
                    delta = abs(float(q_score) - float(l_score))
                    deltas.append(delta)
                    same_score += delta <= 1e-12
            retrieval_cross_model[model][condition] = {
                "same_demo_ids": same_demo,
                "same_scores": same_score,
                "mean_absolute_score_delta": mean(deltas) if deltas else None,
                "max_absolute_score_delta": max(deltas) if deltas else None,
            }

    result = {
        "schema_version": 1,
        "purpose": "llama32_3b_base_completed_evaluation_audit_and_qwen_base_comparison",
        "status": "PASS MIT METHODISCHEN EINSCHRAENKUNGEN",
        "testcases": {"path": str(TESTCASES.relative_to(ROOT)), "sha256": sha256(TESTCASES), "rows": 1032},
        "retrieval": {
            "index_path": EXPECTED["retrieval_index"],
            "index_sha256": sha256(ROOT / EXPECTED["retrieval_index"] / "index.faiss"),
            "metadata_sha256": sha256(ROOT / EXPECTED["retrieval_index"] / "metadata.jsonl"),
        },
        "llama_runs": {
            condition: {key: value for key, value in run.items() if key not in {"rows", "metadata", "traces"}}
            for condition, run in llama.items()
        },
        "within_llama_vs_zero_shot": within_llama,
        "top1_trace_consistency": top1_consistency,
        "structure_trace_consistency": structure_consistency,
        "gate_prompt_and_generation_consistency": gate_prompt_checks,
        "static_consistency": static_check,
        "qwen_reference_runs": qwen_summary,
        "cross_model_comparison": cross_model,
        "cross_model_retrieval_consistency": retrieval_cross_model,
        "comparability": {
            "within_llama_prompting": "CONTROLLED",
            "llama_vs_qwen_dynamic_same_condition": "METHODICALLY ALIGNED BUT NOT A CONTROLLED MODEL-SIZE EFFECT",
            "llama_vs_qwen_zero_shot": "LARGELY ALIGNED; Qwen limit 1536 was non-binding",
            "confounds": ["model family", "parameter count", "Base versus Instruct status", "native tokenizer and chat serialization"],
            "missing_qwen_counterparts": {
                "static_seed42": ["qwen35_2b_base", "qwen35_9b_base"],
                "structure": ["qwen35_9b_base"],
                "structure_gate070": ["qwen35_2b_base"],
                "structure_gate085": ["qwen35_2b_base"],
            },
        },
    }

    case_columns = ["id", "db_id", "question"]
    for condition in LLAMA_RUNS:
        case_columns.extend([f"llama_{condition}_exec_match", f"llama_{condition}_pred_ok", f"llama_{condition}_pred_sql"])
    for model, conditions in QWEN_RUNS.items():
        for condition in conditions:
            case_columns.append(f"{model}_{condition}_exec_match")
    csv_rows = []
    for index, test in enumerate(test_rows):
        row: dict[str, Any] = {"id": test["id"], "db_id": test["db_id"], "question": test["question"]}
        for condition, run in llama.items():
            source = run["rows"][index]
            row[f"llama_{condition}_exec_match"] = as_bool(source["exec_match"])
            row[f"llama_{condition}_pred_ok"] = as_bool(source["pred_ok"])
            row[f"llama_{condition}_pred_sql"] = source["pred_sql"]
        for model, conditions in qwen_loaded.items():
            for condition, run in conditions.items():
                row[f"{model}_{condition}_exec_match"] = as_bool(run["rows"][index]["exec_match"])
        csv_rows.append(row)

    import io

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=case_columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_payload = csv_buffer.getvalue().encode("utf-8")
    result["derived_case_csv"] = {
        "path": str(OUT_CSV.relative_to(ROOT)),
        "rows": len(csv_rows),
        "sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    json_payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    atomic_new(OUT_CSV, csv_payload)
    atomic_new(OUT_JSON, json_payload)
    print(json.dumps({"status": result["status"], "json": str(OUT_JSON.relative_to(ROOT)), "csv": str(OUT_CSV.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
