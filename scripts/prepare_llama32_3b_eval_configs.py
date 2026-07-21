#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
FUTURE_ADAPTER = (
    "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_"
    "schemaheaderfix_evalstop_maxlen2048_epochs5"
)

BASE_PREFIX = "eval_llama32_3b_instruct_base"
LORA_PREFIX = "eval_llama32_3b_instruct_lora_v2_old25k_r8_alpha16_mixedval2500_v2"


COMMON: dict[str, Any] = {
    "llm": "llama32_3b_instruct",
    "max_input_tokens": 2048,
    "max_new_tokens": 256,
    "generation_batch_size": 1,
    "compute_perplexity": False,
    "allow_overlap": False,
    "same_db_only": False,
    "testcases_path": "data/testcases_spider_dev_full.jsonl",
    "traincases_path": (
        "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
    ),
    "max_test_samples": None,
    "progress_log_every": 25,
    "prompt_format": "llama32_instruct_native_chat",
    "system_prompt_variant": "sqlctx_anti_overjoin",
    "extractor_mode": "sql_first_statement_only",
}

DYNAMIC: dict[str, Any] = {
    "prompt_tuning": "dynamic_fewshot",
    "k": 1,
    "retrieval_pool_path": (
        "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
    ),
    "retrieval_index_path": "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15",
    "retrieval_method": "sentence_transformer_faiss",
    "fewshot_example_schema_mode": "full",
    "fewshot_example_mode": "schema_with_rules",
    "embedding_model": "BAAI/bge-large-en-v1.5",
}


def with_adapter(config: dict[str, Any], adapter: str) -> dict[str, Any]:
    value = deepcopy(COMMON)
    value["adapter"] = adapter
    value.update(deepcopy(config))
    return value


def zero_shot() -> dict[str, Any]:
    return {"prompt_tuning": "none", "k": 0}


def top1(*, threshold: float | None = None) -> dict[str, Any]:
    value = deepcopy(DYNAMIC)
    value["fewshot_gate_enabled"] = threshold is not None
    if threshold is not None:
        value.update(
            {
                "fewshot_gate_mode": "similarity_only",
                "fewshot_gate_threshold": threshold,
                "fewshot_gate_features": [],
                "fewshot_gate_debug": False,
                "fewshot_rerank_top_n": None,
            }
        )
    return value


def static() -> dict[str, Any]:
    return {
        "prompt_tuning": "static_fewshot",
        "k": 1,
        "retrieval_method": "static_seeded",
        "retrieval_pool_path": (
            "data/fewshot_static/static_fewshot_k1_full_schema_seed42_"
            "spider_train_no_dev_overlap.jsonl"
        ),
        "fewshot_example_schema_mode": "full",
        "fewshot_example_mode": "schema_with_rules",
        "fewshot_gate_enabled": False,
    }


def structure(*, threshold: float | None = None) -> dict[str, Any]:
    value = top1(threshold=threshold)
    value.update(
        {
            "retrieval_rerank_method": "structure_topk_v2",
            "retrieval_rerank_top_n": 10,
            "retrieval_structure_bonus_max": 0.08,
        }
    )
    return value


def matrix(prefix: str, adapter: str) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}_zero_shot_maxinput2048_full_aliasnames.json": with_adapter(zero_shot(), adapter),
        f"{prefix}_dynamic_bge_large_top1_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(top1(), adapter),
        f"{prefix}_dynamic_bge_large_top1_gate070_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(top1(threshold=0.70), adapter),
        f"{prefix}_dynamic_bge_large_top1_gate085_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(top1(threshold=0.85), adapter),
        f"{prefix}_static_k1_full_schema_seed42_maxinput2048_full_aliasnames.json": with_adapter(static(), adapter),
        f"{prefix}_dynamic_bge_large_top10_structure_rerank_v2_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(structure(), adapter),
        f"{prefix}_dynamic_bge_large_top10_structure_rerank_v2_gate070_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(structure(threshold=0.70), adapter),
        f"{prefix}_dynamic_bge_large_top10_structure_rerank_v2_gate085_k1_full_schema_maxinput2048_full_aliasnames.json": with_adapter(structure(threshold=0.85), adapter),
    }


def main() -> None:
    configs: dict[str, dict[str, Any]] = {}
    configs.update(matrix(BASE_PREFIX, "base"))
    configs.update(matrix(LORA_PREFIX, FUTURE_ADAPTER))

    base_zero_smoke = with_adapter(zero_shot(), "base")
    base_zero_smoke["max_test_samples"] = 5
    configs[f"{BASE_PREFIX}_zero_shot_smoke5_maxinput2048_full_aliasnames.json"] = base_zero_smoke

    base_top1_smoke = with_adapter(top1(), "base")
    base_top1_smoke["max_test_samples"] = 5
    configs[f"{BASE_PREFIX}_dynamic_bge_large_top1_k1_full_schema_smoke5_maxinput2048_full_aliasnames.json"] = base_top1_smoke

    lora_zero_smoke = with_adapter(zero_shot(), FUTURE_ADAPTER)
    lora_zero_smoke["max_test_samples"] = 5
    configs[f"{LORA_PREFIX}_zero_shot_smoke5_maxinput2048_full_aliasnames.json"] = lora_zero_smoke

    existing = [CONFIG_DIR / name for name in configs if (CONFIG_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation configs:\n"
            + "\n".join(str(path) for path in existing)
        )
    for name, config in configs.items():
        path = CONFIG_DIR / name
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
