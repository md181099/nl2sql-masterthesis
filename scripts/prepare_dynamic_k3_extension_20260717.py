#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
OUT_MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
OUT_DEPENDENCIES = ROOT / "audits/derived/dynamic_k3_implementation_dependencies_20260717.csv"
RESULTS_DIR = "results/k3_extension_20260717"

SOURCE_CONDITIONS = (
    "top1",
    "top1_gate070",
    "top1_gate085",
    "structure",
    "structure_gate070",
    "structure_gate085",
)

NEW_CONDITIONS = {
    "top1": "top3",
    "top1_gate070": "top3_gate070",
    "top1_gate085": "top3_gate085",
    "structure": "structure_top3",
    "structure_gate070": "structure_top3_gate070",
    "structure_gate085": "structure_top3_gate085",
}

MODEL_PREFIXES = {
    ("qwen2b", "base"): "eval_qwen35_2b_base",
    ("qwen2b", "lora_v2"): (
        "eval_qwen35_2b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1"
    ),
    ("llama3b", "base"): "eval_llama32_3b_instruct_base",
    ("llama3b", "lora_v2"): (
        "eval_llama32_3b_instruct_lora_v2_old25k_r8_alpha16_mixedval2500_v2"
    ),
    ("qwen9b", "base"): "eval_qwen35_9b_base",
    ("qwen9b", "lora_v2"): (
        "eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1"
    ),
}

MODEL_REVISIONS = {
    "qwen2b": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    "llama3b": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "qwen9b": "68c46c4b3498877f3ef123c856ecfde50c39f404",
}

CONDITION_STEMS = {
    "top3": "dynamic_bge_large_top3_k3_full_schema_maxin4352_full_aliasnames",
    "top3_gate070": (
        "dynamic_bge_large_top3_gate070_k3_full_schema_maxin4352_full_aliasnames"
    ),
    "top3_gate085": (
        "dynamic_bge_large_top3_gate085_k3_full_schema_maxin4352_full_aliasnames"
    ),
    "structure_top3": (
        "dynamic_bge_large_top10_structure_rerank_v2_top3_k3_full_schema_"
        "maxin4352_full_aliasnames"
    ),
    "structure_top3_gate070": (
        "dynamic_bge_large_top10_structure_rerank_v2_top3_gate070_k3_full_schema_"
        "maxin4352_full_aliasnames"
    ),
    "structure_top3_gate085": (
        "dynamic_bge_large_top10_structure_rerank_v2_top3_gate085_k3_full_schema_"
        "maxin4352_full_aliasnames"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new(path: Path, text: str) -> None:
    require(not path.exists(), f"Refusing to overwrite existing additive file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def config_path(model_key: str, role: str, condition: str) -> Path:
    prefix = MODEL_PREFIXES[(model_key, role)]
    return ROOT / "configs" / f"{prefix}_{CONDITION_STEMS[condition]}.json"


def changed_fields(reference: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(reference) | set(new))
    return {
        key: {"reference": reference.get(key), "new": new.get(key)}
        for key in keys
        if reference.get(key) != new.get(key)
    }


def validate_reference(config: dict[str, Any], source_condition: str) -> None:
    expected_reranker = "structure_topk_v2" if source_condition.startswith("structure") else None
    require(config.get("k") == 1, f"Reference is not k=1: {source_condition}")
    require(config.get("prompt_tuning") == "dynamic_fewshot", "Reference is not dynamic_fewshot")
    require(config.get("max_input_tokens") == 2048, "Reference max_input_tokens is not 2048")
    require(config.get("max_new_tokens") == 256, "Reference max_new_tokens is not 256")
    require(config.get("generation_batch_size") == 1, "Reference batch size is not 1")
    require(config.get("max_test_samples") is None, "Reference is not a full run config")
    require(config.get("allow_overlap") is False, "Reference allows overlap")
    require(config.get("fewshot_example_schema_mode") == "full", "Reference is not Full Schema")
    require(
        config.get("retrieval_index_path")
        == "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15",
        "Reference uses the wrong retrieval index",
    )
    require(config.get("retrieval_rerank_method") == expected_reranker, "Unexpected reranker")
    if expected_reranker:
        require(config.get("retrieval_rerank_top_n") == 10, "Structure reference is not Top-10")
        require(config.get("retrieval_structure_bonus_max") == 0.08, "Unexpected structure cap")


def build_config(
    reference: dict[str, Any],
    *,
    model_key: str,
    role: str,
    source_condition: str,
    new_condition: str,
) -> dict[str, Any]:
    config = dict(reference)
    config["k"] = 3
    config["max_input_tokens"] = 4352
    config["expected_model_revision"] = MODEL_REVISIONS[model_key]
    config["results_dir"] = RESULTS_DIR
    config["run_output_prefix"] = f"run_k3_{model_key}_{role}_{new_condition}_maxin4352"
    if "gate" in new_condition:
        config["fewshot_gate_enabled"] = True
        config["fewshot_gate_mode"] = "set_min_similarity"
        config["fewshot_gate_debug"] = True
    else:
        config["fewshot_gate_enabled"] = False
        config.pop("fewshot_gate_mode", None)
        config.pop("fewshot_gate_threshold", None)
        config.pop("fewshot_gate_debug", None)
        config.pop("fewshot_gate_features", None)
        config.pop("fewshot_rerank_top_n", None)
    return config


def main() -> None:
    require(SOURCE_TABLE.is_file(), f"Missing source table: {SOURCE_TABLE}")
    require(not OUT_MATRIX.exists(), f"Refusing to overwrite: {OUT_MATRIX}")
    require(not OUT_DEPENDENCIES.exists(), f"Refusing to overwrite: {OUT_DEPENDENCIES}")
    with SOURCE_TABLE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    selected = [row for row in source_rows if row["condition"] in SOURCE_CONDITIONS]
    require(len(selected) == 36, f"Expected 36 authoritative references, found {len(selected)}")

    matrix_rows: list[dict[str, Any]] = []
    for row in selected:
        model_key = row["model_key"]
        role = row["role"]
        source_condition = row["condition"]
        new_condition = NEW_CONDITIONS[source_condition]
        reference_path = ROOT / row["config_path"]
        require(reference_path.is_file(), f"Missing reference config: {reference_path}")
        require(sha256(reference_path) == row["config_sha256"], f"Reference hash drift: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        validate_reference(reference, source_condition)
        new = build_config(
            reference,
            model_key=model_key,
            role=role,
            source_condition=source_condition,
            new_condition=new_condition,
        )
        target = config_path(model_key, role, new_condition)
        write_new(target, json.dumps(new, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        diffs = changed_fields(reference, new)
        allowed = {
            "k",
            "expected_model_revision",
            "results_dir",
            "run_output_prefix",
            "fewshot_gate_mode",
            "fewshot_gate_debug",
            "max_input_tokens",
        }
        require(set(diffs) <= allowed, f"Disallowed config differences for {target}: {sorted(set(diffs) - allowed)}")
        unchanged = sorted(key for key in reference if key in new and reference[key] == new[key])
        matrix_rows.append(
            {
                "new_k3_config": str(target.relative_to(ROOT)),
                "reference_k1_config": str(reference_path.relative_to(ROOT)),
                "model_key": model_key,
                "model_line": row["model_line"],
                "role": role,
                "source_condition": source_condition,
                "condition": new_condition,
                "changed_fields": json.dumps(diffs, ensure_ascii=False, sort_keys=True),
                "unchanged_fields": json.dumps(unchanged, ensure_ascii=False),
                "reference_config_sha256": sha256(reference_path),
                "config_sha256": sha256(target),
                "one_factor_extension_status": "PASS",
            }
        )

    matrix_fields = list(matrix_rows[0])
    lines: list[str] = []
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=matrix_fields)
    writer.writeheader()
    writer.writerows(matrix_rows)
    write_new(OUT_MATRIX, output.getvalue())

    dependencies = [
        {
            "component": "evaluation_runner",
            "current_file": "src/06_batch_run.py",
            "current_function_or_class": "main/build_case_record",
            "k1_behavior": "Builds one-demo prompts and writes standard run artifacts.",
            "required_k3_change": "Enforce k=3, immutable revisions, unique outputs, and full runtime provenance.",
            "additive_implementation_path": "src/06_batch_run_dynamic_k3_v1.py",
            "risk": "high",
        },
        {
            "component": "faiss_retrieval",
            "current_file": "src/retrieval_utils.py",
            "current_function_or_class": "FaissFewShotRetriever.select",
            "k1_behavior": "Returns the highest-ranked valid demonstration.",
            "required_k3_change": "Return three distinct ranked demonstrations and deterministic stable-ID tie fallback.",
            "additive_implementation_path": "src/retrieval_utils_dynamic_k3_v1.py",
            "risk": "medium",
        },
        {
            "component": "set_gate",
            "current_file": "src/06_batch_run.py",
            "current_function_or_class": "evaluate_fewshot_gate",
            "k1_behavior": "Thresholds only the first selected BGE score.",
            "required_k3_change": "Threshold min(score_1, score_2, score_3) and use binary k=3/k=0.",
            "additive_implementation_path": "src/06_batch_run_dynamic_k3_v1.py",
            "risk": "high",
        },
        {
            "component": "structure_rerank",
            "current_file": "src/structure_rerank_v2.py",
            "current_function_or_class": "structure_rerank_adjustment",
            "k1_behavior": "Computes the established v2 adjustment for Top-10 candidates.",
            "required_k3_change": "No heuristic change; select three by final score, BGE score, BGE rank, stable ID.",
            "additive_implementation_path": "src/retrieval_utils_dynamic_k3_v1.py",
            "risk": "medium",
        },
        {
            "component": "prompt_builder",
            "current_file": "src/06_batch_run.py",
            "current_function_or_class": "build_prompt_schema_fewshot",
            "k1_behavior": "Renders every demo in the supplied list using Full Schema.",
            "required_k3_change": "No code change; verify three blocks and token capacity in preflight.",
            "additive_implementation_path": "src/06_batch_run_dynamic_k3_v1.py (copied implementation)",
            "risk": "high_prompt_length",
        },
    ]
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(dependencies[0]))
    writer.writeheader()
    writer.writerows(dependencies)
    write_new(OUT_DEPENDENCIES, output.getvalue())
    print(json.dumps({
        "status": "PASS",
        "configs_created": len(matrix_rows),
        "matrix": str(OUT_MATRIX.relative_to(ROOT)),
        "dependencies": str(OUT_DEPENDENCIES.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
