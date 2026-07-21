#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEED = 42
TRAIN_SPIDER = 5700
RETRIEVAL_SPIDER = 700
VAL_SPIDER = 560
TRAIN_SQLCC_TOTAL = 19300
OLD_SQLCC_EXPECTED = 18040
EXTRA_SQLCC = TRAIN_SQLCC_TOTAL - OLD_SQLCC_EXPECTED
VAL_SQLCC = 1940
AGG_TARGET_RATE = 0.58

RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_oldsqlccpreserved_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_TRAIN = Path(
    "data/sql_create_context/"
    "train_sft_qwen35_2b_clean_split_oldsqlccpreserved_full_chat_v1_clean_anti_overjoin_"
    "spider5700_sqlcc19300_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_VAL = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_2b_clean_split_oldsqlccpreserved_spider560_sqlcc1940_full_chat_v1_clean_"
    "anti_overjoin_no_train_no_retrieval_no_dev_2500_seed42.jsonl"
)
RETRIEVAL_POOL = Path(
    "data/retrieval_pools/clean_split_oldsqlccpreserved_spider700_no_train_no_val_no_dev_seed42.jsonl"
)
RETRIEVAL_INDEX = Path(
    "data/retrieval_indexes/clean_split_oldsqlccpreserved_spider700_no_train_no_val_no_dev_bge_large_en_v15"
)
STATIC_FEWSHOT = Path(
    "data/fewshot_static/static_fewshot_clean_split_oldsqlccpreserved_spider700_k1_full_schema_seed42.jsonl"
)

TRAIN_CONFIG = Path(
    "configs/train_lora_qwen35_2b_base_clean_split_oldsqlccpreserved_r8_alpha16_"
    "evalval2500_earlystop_maxlen2048_oomsafe.json"
)
EVAL_ZERO = Path(
    "configs/eval_qwen35_2b_lora_clean_split_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_zero_shot_full_aliasnames.json"
)
EVAL_FULL = Path(
    "configs/eval_qwen35_2b_lora_clean_split_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_maxinput2048_full_aliasnames.json"
)
EVAL_GATE085 = Path(
    "configs/eval_qwen35_2b_lora_clean_split_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate085_"
    "maxinput2048_full_aliasnames.json"
)
EVAL_GATE070 = Path(
    "configs/eval_qwen35_2b_lora_clean_split_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate070_"
    "maxinput2048_full_aliasnames.json"
)
EVAL_STATIC = Path(
    "configs/eval_qwen35_2b_lora_clean_split_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_static_fewshot_k1_full_schema_clean_retrieval_maxinput2048_full_aliasnames.json"
)

OUTPUT_ADAPTER = (
    "adapters/qwen35_2b_base/"
    "lora_clean_split_oldsqlccpreserved_qwen35_2b_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_oomsafe"
)

CURRENT_MIX = Path(
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
SQLCC_RAW = Path("data/sql_create_context/train.jsonl")
SPIDER_DEV = Path("data/testcases_spider_dev_full.jsonl")
TRAIN_CONFIG_REF = Path(
    "configs/train_lora_qwen35_2b_base_v1_fullchat_mix_spider_train_sqlcc_"
    "complexity_enriched_25k_seed42_flashattn2_no_overlap_r8_alpha16_"
    "evalval2500_earlystop_maxlen2048_oomsafe.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare old-SQLCC-preserved clean split artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated v2 artifacts.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sets_for(rows: list[dict[str, Any]], base: Any) -> dict[str, set[Any]]:
    return base.row_sets(rows)


def no_overlap_filter(rows: list[dict[str, Any]], protected_rows: list[dict[str, Any]], base: Any) -> list[dict[str, Any]]:
    protected = sets_for(protected_rows, base)
    kept: list[dict[str, Any]] = []
    for row in rows:
        q = base.normalize_question(str(row.get("question", "")))
        s = base.normalize_sql(str(row.get("gold_sql", "")))
        pair = (q, s)
        if q in protected["question"] or s in protected["sql"] or pair in protected["pair"]:
            continue
        kept.append(row)
    return kept


def component_conflicts_with_sets(component: list[dict[str, Any]], sqlcc_sets: dict[str, set[Any]], base: Any) -> bool:
    for row in component:
        q = base.normalize_question(str(row.get("question", "")))
        s = base.normalize_sql(str(row.get("gold_sql", "")))
        pair = (q, s)
        if q in sqlcc_sets["question"] or s in sqlcc_sets["sql"] or pair in sqlcc_sets["pair"]:
            return True
    return False


def split_spider_around_fixed_sqlcc(
    spider_rows: list[dict[str, Any]],
    sqlcc_train: list[dict[str, Any]],
    mix_module: Any,
    base: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    components = base.build_overlap_components(spider_rows)
    len_cuts = base.quantile_bins([int(base.structure(row)["sql_len_chars"]) for row in spider_rows])
    schema_cuts = base.quantile_bins([int(base.structure(row)["schema_table_count"]) for row in spider_rows])
    sqlcc_sets = sets_for(sqlcc_train, base)
    forced_train: list[list[dict[str, Any]]] = []
    eligible: list[list[dict[str, Any]]] = []
    for comp in components:
        if component_conflicts_with_sets(comp, sqlcc_sets, base):
            forced_train.append(comp)
        else:
            eligible.append(comp)

    retrieval_components, remaining = base.select_components(
        eligible,
        target_rows=RETRIEVAL_SPIDER,
        mix_module=mix_module,
        len_cuts=len_cuts,
        schema_cuts=schema_cuts,
        seed=SEED,
    )
    validation_components, remaining_after_val = base.select_components(
        remaining,
        target_rows=VAL_SPIDER,
        mix_module=mix_module,
        len_cuts=len_cuts,
        schema_cuts=schema_cuts,
        seed=SEED + 1,
    )
    train_components = forced_train + remaining_after_val
    spider_train = base.flatten_components(train_components)
    spider_retrieval = base.flatten_components(retrieval_components)
    spider_validation = base.flatten_components(validation_components)
    if len(spider_train) != TRAIN_SPIDER:
        raise RuntimeError(f"Spider train size mismatch: {len(spider_train)} != {TRAIN_SPIDER}")
    stats = {
        "component_count": len(components),
        "eligible_component_count": len(eligible),
        "forced_train_component_count": len(forced_train),
        "forced_train_rows_due_to_sqlcc_overlap": sum(len(comp) for comp in forced_train),
        "component_size_distribution": dict(sorted(Counter(len(comp) for comp in components).items())),
        "len_cuts": len_cuts,
        "schema_table_cuts": schema_cuts,
    }
    return spider_train, spider_retrieval, spider_validation, stats


def select_extra_sqlcc(
    old_sqlcc: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    base: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_agg = sum(bool(base.structure(row)["aggregation"]) for row in old_sqlcc)
    desired_final_agg = round(TRAIN_SQLCC_TOTAL * AGG_TARGET_RATE)
    agg_needed = max(0, desired_final_agg - current_agg)
    rng = random.Random(SEED)
    agg_rows = [dict(row) for row in candidates if base.structure(row)["aggregation"]]
    simple_rows = [dict(row) for row in candidates if not base.structure(row)["aggregation"]]
    rng.shuffle(agg_rows)
    rng.shuffle(simple_rows)

    agg_take = min(agg_needed, EXTRA_SQLCC, len(agg_rows))
    simple_take = EXTRA_SQLCC - agg_take
    if simple_take > len(simple_rows):
        raise RuntimeError(f"Not enough simple SQLCC rows for extra fill: {simple_take} > {len(simple_rows)}")
    selected = agg_rows[:agg_take] + simple_rows[:simple_take]
    if len(selected) < EXTRA_SQLCC:
        selected.extend(agg_rows[agg_take : agg_take + (EXTRA_SQLCC - len(selected))])
    if len(selected) != EXTRA_SQLCC:
        raise RuntimeError(f"Extra SQLCC size mismatch: {len(selected)} != {EXTRA_SQLCC}")
    for row in selected[:agg_take]:
        row["selection_bucket"] = "sqlcc_oldmix_preserved_extra_aggregation_fill"
    for row in selected[agg_take:]:
        row["selection_bucket"] = "sqlcc_oldmix_preserved_extra_simple_fill"
    stats = {
        "old_sqlcc_rows_preserved": len(old_sqlcc),
        "extra_sqlcc_target": EXTRA_SQLCC,
        "aggregation_target_rate_for_final_sqlcc_train": AGG_TARGET_RATE,
        "old_sqlcc_aggregation_rows": current_agg,
        "desired_final_sqlcc_aggregation_rows": desired_final_agg,
        "extra_aggregation_rows_selected": agg_take,
        "extra_simple_rows_selected": len(selected) - agg_take,
        "available_extra_aggregation_candidates": len(agg_rows),
        "available_extra_simple_candidates": len(simple_rows),
    }
    return selected, stats


def write_configs(root: Path, base: Any) -> dict[str, Any]:
    train_ref = json.loads((root / TRAIN_CONFIG_REF).read_text(encoding="utf-8"))
    train_cfg = dict(train_ref)
    train_cfg["dataset_path"] = base.rel(root / SFT_TRAIN)
    train_cfg["eval_dataset_path"] = base.rel(root / SFT_VAL)
    train_cfg["output_dir"] = OUTPUT_ADAPTER
    train_cfg["num_train_epochs"] = 10
    train_cfg["early_stopping"]["early_stopping_threshold"] = 0.01
    train_cfg["lora"]["r"] = 8
    train_cfg["lora"]["lora_alpha"] = 16
    base.write_json(root / TRAIN_CONFIG, train_cfg)

    adapter = Path(OUTPUT_ADAPTER).name
    common_eval = {
        "llm": "qwen35_2b_base",
        "adapter": adapter,
        "max_new_tokens": 256,
        "prompt_format": "qwen_sqlctx_chatml",
        "system_prompt_variant": "sqlctx_anti_overjoin",
        "extractor_mode": "sql_first_statement_only",
        "generation_batch_size": 1,
        "compute_perplexity": False,
        "allow_overlap": False,
        "same_db_only": False,
        "testcases_path": "data/testcases_spider_dev_full.jsonl",
        "traincases_path": base.rel(root / RAW_TRAIN),
        "max_test_samples": None,
        "progress_log_every": 25,
    }
    zero = dict(common_eval)
    zero.update({"prompt_tuning": "none", "k": 0, "max_input_tokens": 1536})
    base.write_json(root / EVAL_ZERO, zero)

    dynamic = dict(common_eval)
    dynamic.update(
        {
            "prompt_tuning": "dynamic_fewshot",
            "k": 1,
            "max_input_tokens": 2048,
            "retrieval_pool_path": base.rel(root / RETRIEVAL_INDEX / "metadata.jsonl"),
            "retrieval_index_path": base.rel(root / RETRIEVAL_INDEX),
            "retrieval_method": "sentence_transformer_faiss",
            "fewshot_example_schema_mode": "full",
            "fewshot_example_mode": "schema_with_rules",
            "embedding_model": base.EMBEDDING_MODEL,
        }
    )
    full = dict(dynamic)
    full["fewshot_gate_enabled"] = False
    base.write_json(root / EVAL_FULL, full)

    gate085 = dict(dynamic)
    gate085.update(
        {
            "fewshot_gate_enabled": True,
            "fewshot_gate_mode": "similarity_only",
            "fewshot_gate_threshold": 0.85,
            "fewshot_rerank_top_n": None,
            "fewshot_gate_features": [],
            "fewshot_gate_debug": False,
        }
    )
    base.write_json(root / EVAL_GATE085, gate085)
    gate070 = dict(gate085)
    gate070["fewshot_gate_threshold"] = 0.70
    base.write_json(root / EVAL_GATE070, gate070)

    static_cfg = dict(common_eval)
    static_cfg.update(
        {
            "prompt_tuning": "static_fewshot",
            "k": 1,
            "max_input_tokens": 2048,
            "retrieval_pool_path": base.rel(root / STATIC_FEWSHOT),
            "retrieval_method": "static_seeded",
            "fewshot_example_schema_mode": "full",
            "fewshot_example_mode": "schema_with_rules",
            "fewshot_gate_enabled": False,
        }
    )
    base.write_json(root / EVAL_STATIC, static_cfg)
    return {
        "train_config": base.rel(root / TRAIN_CONFIG),
        "eval_configs": [
            base.rel(root / path)
            for path in (EVAL_ZERO, EVAL_FULL, EVAL_GATE085, EVAL_GATE070, EVAL_STATIC)
        ],
        "output_adapter": OUTPUT_ADAPTER,
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    base = load_module(root / "src/14_prepare_qwen35_2b_clean_split_pipeline.py", "base_clean_split")
    mix_module = load_module(root / "src/04_build_spider_sqlcc_complexity_mix.py", "mix_builder_oldsqlcc")
    sft_module = load_module(root / "src/02_make_sft_dataset_v1_clean_full_chat.py", "sft_builder_oldsqlcc")
    prompt_module = load_module(root / "src/prompt_presets.py", "prompt_presets_oldsqlcc")

    base.RETRIEVAL_POOL = RETRIEVAL_POOL
    base.STATIC_FEWSHOT = STATIC_FEWSHOT

    base.ensure_free(
        [
            RAW_TRAIN,
            base.manifest_path(RAW_TRAIN),
            SFT_TRAIN,
            base.manifest_path(SFT_TRAIN),
            SFT_VAL,
            base.manifest_path(SFT_VAL),
            RETRIEVAL_POOL,
            base.manifest_path(RETRIEVAL_POOL),
            STATIC_FEWSHOT,
            base.manifest_path(STATIC_FEWSHOT),
            TRAIN_CONFIG,
            EVAL_ZERO,
            EVAL_FULL,
            EVAL_GATE085,
            EVAL_GATE070,
            EVAL_STATIC,
        ],
        overwrite=args.overwrite,
    )
    if (root / RETRIEVAL_INDEX).exists() and not args.overwrite:
        raise FileExistsError(f"Retrieval index path already exists: {RETRIEVAL_INDEX}")
    if (root / OUTPUT_ADAPTER).exists():
        raise FileExistsError(f"Output adapter path already exists: {OUTPUT_ADAPTER}")

    current_rows = base.read_jsonl(root / CURRENT_MIX)
    spider_rows = [dict(row) for row in current_rows if row.get("source_dataset") == "spider_train"]
    old_sqlcc = [dict(row) for row in current_rows if row.get("source_dataset") == "sql_create_context"]
    if len(spider_rows) != 6960:
        raise RuntimeError(f"Expected 6960 Spider rows, got {len(spider_rows)}")
    if len(old_sqlcc) != OLD_SQLCC_EXPECTED:
        raise RuntimeError(f"Expected {OLD_SQLCC_EXPECTED} old SQLCC rows, got {len(old_sqlcc)}")

    dev_q, dev_s, _dev_pair = mix_module.load_dev_overlap_sets(root / SPIDER_DEV)
    all_spider_q = {base.normalize_question(str(row.get("question", ""))) for row in spider_rows}
    all_spider_s = {base.normalize_sql(str(row.get("gold_sql", ""))) for row in spider_rows}
    all_spider_pair = {
        (base.normalize_question(str(row.get("question", ""))), base.normalize_sql(str(row.get("gold_sql", ""))))
        for row in spider_rows
    }
    sqlcc_pool, sqlcc_pool_stats, _examples = mix_module.build_sqlcc_pool(
        sqlcc_path=root / SQLCC_RAW,
        dev_question_set=dev_q,
        dev_sql_set=dev_s,
        spider_question_set=all_spider_q,
        spider_sql_set=all_spider_s,
        spider_pair_set=all_spider_pair,
    )
    old_ids = {str(row.get("id", "")) for row in old_sqlcc}
    pool_not_old = [row for row in sqlcc_pool if str(row.get("id", "")) not in old_ids]
    strict_pool_not_old, strict_filter_stats = base.strict_filter_sqlcc_against_spider(pool_not_old, spider_rows)
    extra_sqlcc, extra_stats = select_extra_sqlcc(old_sqlcc, strict_pool_not_old, base)
    sqlcc_train = old_sqlcc + extra_sqlcc

    spider_train, spider_retrieval, spider_validation, spider_stats = split_spider_around_fixed_sqlcc(
        spider_rows,
        sqlcc_train,
        mix_module,
        base,
    )

    val_candidates = no_overlap_filter(
        [row for row in strict_pool_not_old if str(row.get("id", "")) not in {str(x.get("id", "")) for x in extra_sqlcc}],
        sqlcc_train + spider_rows,
        base,
    )
    sqlcc_validation, sqlcc_val_stats = base.select_sqlcc_validation(val_candidates)

    train_rows = [
        base.enrich_row(row, role="train", order=i)
        for i, row in enumerate(spider_train + sqlcc_train)
    ]
    validation_raw = [
        base.enrich_row(row, role="validation", order=i)
        for i, row in enumerate(spider_validation + sqlcc_validation)
    ]
    retrieval_rows = [base.retrieval_row(row, order=i) for i, row in enumerate(spider_retrieval)]

    if len(train_rows) != 25000 or len(retrieval_rows) != 700 or len(validation_raw) != 2500:
        raise RuntimeError("Corrected clean split counts do not match requested sizes.")

    dev_rows = base.read_jsonl(root / SPIDER_DEV)
    overlap_matrix = {
        "train_vs_retrieval": base.overlap_counts(train_rows, retrieval_rows),
        "train_vs_validation": base.overlap_counts(train_rows, validation_raw),
        "retrieval_vs_validation": base.overlap_counts(retrieval_rows, validation_raw),
        "train_vs_spider_dev": base.overlap_counts(train_rows, dev_rows),
        "retrieval_vs_spider_dev": base.overlap_counts(retrieval_rows, dev_rows),
        "validation_vs_spider_dev": base.overlap_counts(validation_raw, dev_rows),
    }
    if any(any(value != 0 for value in counts.values()) for counts in overlap_matrix.values()):
        raise RuntimeError("Overlap matrix is not clean: " + json.dumps(overlap_matrix, sort_keys=True))

    train_sft, train_sft_manifest = base.sft_rows_from_raw(train_rows, sft_module, prompt_module)
    val_sft, val_sft_manifest = base.sft_rows_from_raw(validation_raw, sft_module, prompt_module)

    base.write_jsonl(root / RAW_TRAIN, train_rows)
    base.write_jsonl(root / SFT_TRAIN, train_sft)
    base.write_jsonl(root / SFT_VAL, val_sft)
    base.write_jsonl(root / RETRIEVAL_POOL, retrieval_rows)

    static_row, static_manifest = base.select_static_fewshot(retrieval_rows)
    base.write_jsonl(root / STATIC_FEWSHOT, [static_row])

    created_at = datetime.now(timezone.utc).isoformat()
    common = {
        "created_at": created_at,
        "seed": SEED,
        "builder_script": "src/15_prepare_qwen35_2b_clean_split_oldsqlcc_preserved_pipeline.py",
        "correction_reason": (
            "Preserve all SQL Create Context examples from the previous 25k mix, then fill "
            "the remaining SQLCC training quota with similar SQLCC examples."
        ),
        "stratification": {
            "spider": [
                "overlap_components_by_question_and_sql",
                "sqlcc_train_conflicting_components_forced_to_train",
                "db_id",
                "sql_length_bin",
                "schema_table_count_bin",
                "sql_table_count_bin",
                "join_bin",
                "where",
                "aggregation",
                "group_by",
                "having",
                "order_by",
                "limit",
                "distinct",
                "subquery",
                "set_operation",
                "rare_aggregation_simple_bucket",
            ],
            "sqlcc_train": "all previous 25k-mix SQLCC rows + extra fill matching final aggregation target",
            "sqlcc_validation": [
                "aggregation_vs_simple",
                "sql_length_bin",
                "where",
                "count_avg_sum_min_max",
            ],
        },
        "overlap_matrix": overlap_matrix,
    }

    raw_manifest = {
        **common,
        "path": base.rel(root / RAW_TRAIN),
        "sha256": base.sha256_file(root / RAW_TRAIN),
        "counts": {
            "rows": len(train_rows),
            "spider_train": len(spider_train),
            "sql_create_context": len(sqlcc_train),
            "sqlcc_from_previous_25k_mix": len(old_sqlcc),
            "sqlcc_extra_fill": len(extra_sqlcc),
        },
        "source_counts": base.source_counts(train_rows),
        "structure_distribution": base.structure_distribution(train_rows, mix_module),
        "spider_component_stats": spider_stats,
        "sqlcc_pool_stats": sqlcc_pool_stats,
        "strict_extra_sqlcc_filter_stats": strict_filter_stats,
        "sqlcc_extra_selection_stats": extra_stats,
    }
    base.write_json(root / base.manifest_path(RAW_TRAIN), raw_manifest)

    train_sft_manifest.update(
        {
            **common,
            "path": base.rel(root / SFT_TRAIN),
            "raw_train_path": base.rel(root / RAW_TRAIN),
            "sha256": base.sha256_file(root / SFT_TRAIN),
            "raw_train_sha256": base.sha256_file(root / RAW_TRAIN),
            "counts": raw_manifest["counts"],
        }
    )
    base.write_json(root / base.manifest_path(SFT_TRAIN), train_sft_manifest)

    val_manifest = {
        **common,
        **val_sft_manifest,
        "path": base.rel(root / SFT_VAL),
        "sha256": base.sha256_file(root / SFT_VAL),
        "counts": {
            "rows": len(validation_raw),
            "spider_train": len(spider_validation),
            "sql_create_context": len(sqlcc_validation),
        },
        "source_counts": base.source_counts(validation_raw),
        "structure_distribution": base.structure_distribution(validation_raw, mix_module),
        "sqlcc_validation_selection_stats": sqlcc_val_stats,
    }
    base.write_json(root / base.manifest_path(SFT_VAL), val_manifest)

    retrieval_manifest = {
        **common,
        "path": base.rel(root / RETRIEVAL_POOL),
        "sha256": base.sha256_file(root / RETRIEVAL_POOL),
        "counts": {"rows": len(retrieval_rows), "spider_train": len(retrieval_rows), "sql_create_context": 0},
        "source_counts": base.source_counts(retrieval_rows),
        "structure_distribution": base.structure_distribution(retrieval_rows, mix_module),
        "embedding_model_for_index": base.EMBEDDING_MODEL,
        "bge_query_prefix": base.BGE_QUERY_PREFIX,
    }
    base.write_json(root / base.manifest_path(RETRIEVAL_POOL), retrieval_manifest)

    static_manifest.update(
        {
            **common,
            "path": base.rel(root / STATIC_FEWSHOT),
            "sha256": base.sha256_file(root / STATIC_FEWSHOT),
            "resource_path": base.rel(root / STATIC_FEWSHOT),
        }
    )
    base.write_json(root / base.manifest_path(STATIC_FEWSHOT), static_manifest)

    config_summary = write_configs(root, base)
    summary_path = root / "results/analyses/clean_split_oldsqlccpreserved_pipeline_preparation_summary.json"
    summary = {
        **common,
        "generated_files": [
            base.rel(root / RAW_TRAIN),
            base.rel(root / base.manifest_path(RAW_TRAIN)),
            base.rel(root / SFT_TRAIN),
            base.rel(root / base.manifest_path(SFT_TRAIN)),
            base.rel(root / SFT_VAL),
            base.rel(root / base.manifest_path(SFT_VAL)),
            base.rel(root / RETRIEVAL_POOL),
            base.rel(root / base.manifest_path(RETRIEVAL_POOL)),
            base.rel(root / STATIC_FEWSHOT),
            base.rel(root / base.manifest_path(STATIC_FEWSHOT)),
            config_summary["train_config"],
            *config_summary["eval_configs"],
        ],
        "config_summary": config_summary,
        "split_counts": {
            "train": len(train_rows),
            "train_spider": len(spider_train),
            "train_sqlcc": len(sqlcc_train),
            "train_sqlcc_from_previous_25k_mix": len(old_sqlcc),
            "train_sqlcc_extra_fill": len(extra_sqlcc),
            "retrieval": len(retrieval_rows),
            "validation": len(validation_raw),
            "validation_spider": len(spider_validation),
            "validation_sqlcc": len(sqlcc_validation),
        },
        "sha256": {
            "raw_train": base.sha256_file(root / RAW_TRAIN),
            "sft_train": base.sha256_file(root / SFT_TRAIN),
            "sft_validation": base.sha256_file(root / SFT_VAL),
            "retrieval_pool": base.sha256_file(root / RETRIEVAL_POOL),
            "static_fewshot": base.sha256_file(root / STATIC_FEWSHOT),
        },
        "structure": {
            "train": raw_manifest["structure_distribution"],
            "validation": val_manifest["structure_distribution"],
            "retrieval": retrieval_manifest["structure_distribution"],
        },
    }
    base.write_json(summary_path, summary)
    print(json.dumps({"status": "prepared", "summary_path": base.rel(summary_path), "split_counts": summary["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
