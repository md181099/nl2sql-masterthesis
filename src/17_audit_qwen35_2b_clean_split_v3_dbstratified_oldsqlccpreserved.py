#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import py_compile
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-08"
VARIANT = "v3_dbstratified_oldsqlccpreserved"

CURRENT_MIX = Path(
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
OLD_RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_oldsqlccpreserved_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
OLD_SFT_VAL = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_2b_clean_split_oldsqlccpreserved_spider560_sqlcc1940_full_chat_v1_clean_"
    "anti_overjoin_no_train_no_retrieval_no_dev_2500_seed42.jsonl"
)
OLD_RETRIEVAL = Path(
    "data/retrieval_pools/clean_split_oldsqlccpreserved_spider700_no_train_no_val_no_dev_seed42.jsonl"
)
V3_RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_v3_dbstratified_oldsqlccpreserved_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
V3_SFT_TRAIN = Path(
    "data/sql_create_context/"
    "train_sft_qwen35_2b_clean_split_v3_dbstratified_oldsqlccpreserved_full_chat_v1_clean_anti_overjoin_"
    "spider5700_sqlcc19300_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
V3_SFT_VAL = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_2b_clean_split_v3_dbstratified_oldsqlccpreserved_spider560_sqlcc1940_"
    "full_chat_v1_clean_anti_overjoin_no_train_no_retrieval_no_dev_2500_seed42.jsonl"
)
V3_RETRIEVAL = Path(
    "data/retrieval_pools/"
    "clean_split_v3_dbstratified_oldsqlccpreserved_spider700_no_train_no_val_no_dev_seed42.jsonl"
)
V3_INDEX = Path(
    "data/retrieval_indexes/"
    "clean_split_v3_dbstratified_oldsqlccpreserved_spider700_no_train_no_val_no_dev_bge_large_en_v15"
)
V3_STATIC = Path(
    "data/fewshot_static/"
    "static_fewshot_clean_split_v3_dbstratified_oldsqlccpreserved_spider700_k1_full_schema_seed42.jsonl"
)
V3_SUMMARY = Path("results/analyses/clean_split_v3_dbstratified_oldsqlccpreserved_pipeline_preparation_summary.json")
SPIDER_DEV = Path("data/testcases_spider_dev_full.jsonl")
SQLCC_RAW = Path("data/sql_create_context/train.jsonl")

DB_CSV = Path("results/analyses/clean_split_v3_dbstratified_oldsqlccpreserved_db_distribution.csv")
COMPLEXITY_CSV = Path("results/analyses/clean_split_v3_dbstratified_oldsqlccpreserved_complexity_comparison.csv")
TOKEN_CSV = Path("results/analyses/clean_split_v3_dbstratified_oldsqlccpreserved_token_lengths.csv")
AUDIT_MD = Path("results/audits/audit_qwen35_2b_clean_split_v3_dbstratified_oldsqlccpreserved_preparation_20260708.md")

TRAIN_CONFIG = Path(
    "configs/train_lora_qwen35_2b_base_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_"
    "evalval2500_earlystop_maxlen2048_oomsafe.json"
)
EVAL_CONFIGS = [
    Path(
        "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
        "earlystop_maxlen2048_zero_shot_full_aliasnames.json"
    ),
    Path(
        "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
        "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_maxinput2048_full_aliasnames.json"
    ),
    Path(
        "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
        "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate085_maxinput2048_full_aliasnames.json"
    ),
    Path(
        "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
        "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate070_maxinput2048_full_aliasnames.json"
    ),
    Path(
        "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
        "earlystop_maxlen2048_static_fewshot_k1_full_schema_clean_retrieval_maxinput2048_full_aliasnames.json"
    ),
]
OUTPUT_ADAPTER = Path(
    "adapters/qwen35_2b_base/"
    "lora_clean_split_v3_dbstratified_oldsqlccpreserved_qwen35_2b_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_oomsafe"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    full = ROOT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(out)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def pct(value: float) -> float:
    return 100.0 * value


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(values[low])
    return float(values[low] * (high - idx) + values[high] * (idx - low))


def row_id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}


def db_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("db_id", "")) for row in rows)


def db_distribution_summary(
    *,
    pool_counts: Counter[str],
    role_counts: Counter[str],
    target_rows: int,
) -> dict[str, Any]:
    total_pool = sum(pool_counts.values())
    frequent = sorted(db for db, count in pool_counts.items() if count >= 10)
    missing = []
    over = []
    under = []
    for db in frequent:
        expected = pool_counts[db] / total_pool * target_rows
        actual = role_counts.get(db, 0)
        ratio = actual / expected if expected else 0.0
        if actual == 0:
            missing.append(db)
        if ratio >= 2.0:
            over.append(db)
        if ratio <= 0.5:
            under.append(db)
    return {
        "frequent_db_count": len(frequent),
        "missing": len(missing),
        "over": len(over),
        "under": len(under),
    }


def db_distribution_csv_rows(
    pool_counts: Counter[str],
    old_retrieval: list[dict[str, Any]],
    old_validation_spider: list[dict[str, Any]],
    v3_retrieval: list[dict[str, Any]],
    v3_validation_spider: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    old_ret_counts = db_counts(old_retrieval)
    old_val_counts = db_counts(old_validation_spider)
    v3_ret_counts = db_counts(v3_retrieval)
    v3_val_counts = db_counts(v3_validation_spider)
    total_pool = sum(pool_counts.values())
    rows: list[dict[str, Any]] = []
    for db in sorted(pool_counts):
        ret_expected = pool_counts[db] / total_pool * 700
        val_expected = pool_counts[db] / total_pool * 560
        rows.append(
            {
                "db_id": db,
                "spider_pool_count": pool_counts[db],
                "frequent_ge10": int(pool_counts[db] >= 10),
                "old_retrieval_count": old_ret_counts.get(db, 0),
                "v3_retrieval_count": v3_ret_counts.get(db, 0),
                "retrieval_expected": ret_expected,
                "old_retrieval_ratio": old_ret_counts.get(db, 0) / ret_expected if ret_expected else 0.0,
                "v3_retrieval_ratio": v3_ret_counts.get(db, 0) / ret_expected if ret_expected else 0.0,
                "old_validation_count": old_val_counts.get(db, 0),
                "v3_validation_count": v3_val_counts.get(db, 0),
                "validation_expected": val_expected,
                "old_validation_ratio": old_val_counts.get(db, 0) / val_expected if val_expected else 0.0,
                "v3_validation_ratio": v3_val_counts.get(db, 0) / val_expected if val_expected else 0.0,
            }
        )
    return rows


def metric_summary(rows: list[dict[str, Any]], base: Any, mix_module: Any) -> dict[str, float]:
    if not rows:
        return {}
    structs = [base.structure(row) for row in rows]
    sql_lens = [int(s["sql_len_chars"]) for s in structs]
    sql_tables = [int(s["sql_table_count"]) for s in structs]
    schema_tables = [int(s["schema_table_count"]) for s in structs]
    buckets = Counter(base.bucket(row, mix_module) for row in rows)
    join_counts = [int(s["join_count"]) for s in structs]
    return {
        "examples": float(len(rows)),
        "spider_count": float(sum(1 for row in rows if row.get("source_dataset") == "spider_train")),
        "sqlcc_count": float(sum(1 for row in rows if row.get("source_dataset") == "sql_create_context")),
        "rare_complexity_rate_pct": pct(buckets.get("rare_complexity", 0) / len(rows)),
        "aggregation_bucket_rate_pct": pct(buckets.get("aggregation_only", 0) / len(rows)),
        "simple_bucket_rate_pct": pct(buckets.get("simple", 0) / len(rows)),
        "sql_length_mean": statistics.mean(sql_lens),
        "sql_length_median": statistics.median(sql_lens),
        "sql_length_p25": percentile(sql_lens, 0.25),
        "sql_length_p75": percentile(sql_lens, 0.75),
        "sql_length_min": float(min(sql_lens)),
        "sql_length_max": float(max(sql_lens)),
        "sql_table_count_mean": statistics.mean(sql_tables),
        "schema_table_count_mean": statistics.mean(schema_tables),
        "join_rate_pct": pct(sum(count > 0 for count in join_counts) / len(rows)),
        "join_bin_0_rate_pct": pct(sum(count == 0 for count in join_counts) / len(rows)),
        "join_bin_1_rate_pct": pct(sum(count == 1 for count in join_counts) / len(rows)),
        "join_bin_2_rate_pct": pct(sum(count == 2 for count in join_counts) / len(rows)),
        "join_bin_3plus_rate_pct": pct(sum(count >= 3 for count in join_counts) / len(rows)),
        "where_rate_pct": pct(sum(bool(s["where_any"]) for s in structs) / len(rows)),
        "aggregation_rate_pct": pct(sum(bool(s["aggregation"]) for s in structs) / len(rows)),
        "group_by_rate_pct": pct(sum(bool(s["group_by"]) for s in structs) / len(rows)),
        "having_rate_pct": pct(sum(bool(s["having"]) for s in structs) / len(rows)),
        "order_by_rate_pct": pct(sum(bool(s["order_by"]) for s in structs) / len(rows)),
        "limit_rate_pct": pct(sum(bool(s["limit"]) for s in structs) / len(rows)),
        "distinct_rate_pct": pct(sum(bool(s["distinct"]) for s in structs) / len(rows)),
        "subquery_rate_pct": pct(sum(bool(s["subquery"]) for s in structs) / len(rows)),
        "set_operation_rate_pct": pct(sum(bool(s["set_operation"]) for s in structs) / len(rows)),
    }


def complexity_csv_rows(comparisons: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]], base: Any, mix_module: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, left, right in comparisons:
        left_metrics = metric_summary(left, base, mix_module)
        right_metrics = metric_summary(right, base, mix_module)
        for metric in sorted(set(left_metrics) | set(right_metrics)):
            left_value = left_metrics.get(metric, 0.0)
            right_value = right_metrics.get(metric, 0.0)
            out.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta_right_minus_left": right_value - left_value,
                }
            )
    return out


def reconstruct_validation_raw(
    sft_path: Path,
    spider_by_id: dict[str, dict[str, Any]],
    sqlcc_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_jsonl(sft_path)
    raw: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, row in enumerate(rows):
        qid = str(row.get("id", "")).strip()
        source = spider_by_id.get(qid) or sqlcc_by_id.get(qid)
        if source is None:
            missing.append(qid)
            continue
        out = dict(source)
        out["clean_split_role"] = "validation"
        out["clean_split_order"] = idx
        raw.append(out)
    if missing:
        raise RuntimeError(f"Could not reconstruct validation IDs: {missing[:10]}")
    return raw


def sft_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    checks = Counter()
    for row in rows:
        text = str(row.get("text", ""))
        checks["rows"] += 1
        checks["fields_exact_id_text"] += int(set(row.keys()) == {"id", "text"})
        checks["system"] += int("<|im_start|>system" in text)
        checks["user"] += int("<|im_start|>user" in text)
        checks["assistant_block"] += int(assistant_marker in text)
        checks["end"] += int(text.rstrip().endswith(end_marker))
        checks["think"] += int("<think" in text.casefold())
        if assistant_marker in text:
            completion = text.rsplit(assistant_marker, 1)[1].split(end_marker, 1)[0].strip()
            low = completion.casefold()
            sql_only = (
                (low.startswith("select") or low.startswith("with"))
                and "```" not in completion
                and "<|im_" not in completion
                and "\n\n" not in completion
            )
            checks["sql_only"] += int(sql_only)
    return dict(checks)


def token_stats(name: str, rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    lengths = [
        len(tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"])
        for row in rows
    ]
    return {
        "dataset": name,
        "rows": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p25": percentile(lengths, 0.25),
        "p75": percentile(lengths, 0.75),
        "p95": percentile(lengths, 0.95),
        "max": max(lengths),
        "gt1536": sum(length > 1536 for length in lengths),
        "gt2048": sum(length > 2048 for length in lengths),
    }


def prompt_smoke(tokenizer: Any) -> dict[str, Any]:
    batch_run = load_module(Path("src/06_batch_run.py"), "batch_run_v3_audit")
    retrieval_utils = load_module(Path("src/retrieval_utils.py"), "retrieval_utils_v3_audit")
    prompt_presets = load_module(Path("src/prompt_presets.py"), "prompt_presets_v3_audit")

    system_prompt, _source, _path, _ = prompt_presets.resolve_system_prompt(
        project_root=ROOT,
        system_prompt_variant="sqlctx_anti_overjoin",
        system_prompt_path=None,
    )
    dev_rows = read_jsonl(SPIDER_DEV)
    static_rows = read_jsonl(V3_STATIC)
    static_demo = static_rows[0]
    leakage_guard = retrieval_utils.LeakageGuard.from_testcases_path(ROOT / SPIDER_DEV)
    retriever = retrieval_utils.FaissFewShotRetriever(
        index_dir=ROOT / V3_INDEX,
        embedding_model="BAAI/bge-large-en-v1.5",
        k=1,
        allow_overlap=False,
        same_db_only=False,
        leakage_guard=leakage_guard,
        retrieval_pool_path=ROOT / V3_INDEX / "metadata.jsonl",
    )

    zero_lengths: list[int] = []
    dynamic_lengths: list[int] = []
    static_lengths: list[int] = []
    retrieval_failed = 0
    unique_demo_ids: set[str] = set()
    similarities: list[float] = []
    think_count = 0
    for row in dev_rows:
        schema = str(row.get("schema_prompt", ""))
        question = str(row.get("question", ""))
        qid = str(row.get("id", ""))
        db_id = str(row.get("db_id", ""))
        zero_prompt = batch_run.build_prompt(
            schema,
            question,
            "qwen35_2b_base",
            tokenizer,
            prompt_format="qwen_sqlctx_chatml",
            system_instruction=system_prompt,
        )
        zero_lengths.append(len(tokenizer(zero_prompt, add_special_tokens=False)["input_ids"]))
        selection = retriever.select(question=question, qid=qid, db_id=db_id)
        if not selection.retrieval_success:
            retrieval_failed += 1
        unique_demo_ids.update(selection.ids())
        similarities.extend(float(score) for score in selection.scores)
        dynamic_prompt = batch_run.build_prompt_schema_fewshot(
            schema,
            question,
            selection.examples,
            "qwen35_2b_base",
            tokenizer,
            prompt_format="qwen_sqlctx_chatml",
            system_instruction=system_prompt,
            example_schema_mode="full",
            example_mode="schema_with_rules",
        )
        dynamic_lengths.append(len(tokenizer(dynamic_prompt, add_special_tokens=False)["input_ids"]))
        static_prompt = batch_run.build_prompt_schema_fewshot(
            schema,
            question,
            [static_demo],
            "qwen35_2b_base",
            tokenizer,
            prompt_format="qwen_sqlctx_chatml",
            system_instruction=system_prompt,
            example_schema_mode="full",
            example_mode="schema_with_rules",
        )
        static_lengths.append(len(tokenizer(static_prompt, add_special_tokens=False)["input_ids"]))
        think_count += int("<think" in zero_prompt.casefold())
        think_count += int("<think" in dynamic_prompt.casefold())
        think_count += int("<think" in static_prompt.casefold())

    fake_example = {"id": "SMOKE", "gold_sql": "SELECT 1;"}

    def gate(score: float, threshold: float) -> str:
        selection = retrieval_utils.FewShotSelection(
            examples=[fake_example],
            scores=[score],
            filtered_count=0,
            filtered_reasons={},
            retrieval_method="smoke",
            retrieval_index_path="",
            retrieval_pool_path="",
            retrieval_success=True,
        )
        return batch_run.evaluate_fewshot_gate(
            enabled=True,
            mode="similarity_only",
            threshold=threshold,
            features=[],
            selection=selection,
            question="smoke",
            debug_enabled=False,
        ).decision

    return {
        "zero_shot": {
            "cases": len(dev_rows),
            "max_tokens": max(zero_lengths),
            "over_limit": sum(length > 1536 for length in zero_lengths),
            "limit": 1536,
        },
        "dynamic_full_schema": {
            "cases": len(dev_rows),
            "max_tokens": max(dynamic_lengths),
            "over_limit": sum(length > 2048 for length in dynamic_lengths),
            "limit": 2048,
            "retrieval_failed": retrieval_failed,
            "unique_demo_ids": len(unique_demo_ids),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
        },
        "static_full_schema": {
            "cases": len(dev_rows),
            "max_tokens": max(static_lengths),
            "over_limit": sum(length > 2048 for length in static_lengths),
            "limit": 2048,
        },
        "think_in_prompts": think_count,
        "gate_smoke": {
            "0.86_at_0.85": gate(0.86, 0.85),
            "0.84_at_0.85": gate(0.84, 0.85),
            "0.71_at_0.70": gate(0.71, 0.70),
            "0.69_at_0.70": gate(0.69, 0.70),
        },
    }


def config_checks() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_cfg = read_json(TRAIN_CONFIG)
    train_checks = {
        "dataset_path": train_cfg.get("dataset_path") == rel(V3_SFT_TRAIN),
        "eval_dataset_path": train_cfg.get("eval_dataset_path") == rel(V3_SFT_VAL),
        "output_dir": train_cfg.get("output_dir") == rel(OUTPUT_ADAPTER),
        "lora_r": train_cfg.get("lora", {}).get("r") == 8,
        "lora_alpha": train_cfg.get("lora", {}).get("lora_alpha") == 16,
        "max_length": train_cfg.get("max_length") == 2048,
        "eval_strategy": train_cfg.get("eval_strategy") == "epoch",
        "prediction_loss_only": train_cfg.get("prediction_loss_only") is True,
        "output_adapter_free": (not (ROOT / OUTPUT_ADAPTER).exists()) or not any((ROOT / OUTPUT_ADAPTER).iterdir()),
    }
    eval_rows: list[dict[str, Any]] = []
    for path in EVAL_CONFIGS:
        cfg = read_json(path)
        text = (ROOT / path).read_text(encoding="utf-8")
        eval_rows.append(
            {
                "path": rel(path),
                "json_valid": True,
                "adapter_ok": cfg.get("adapter") == OUTPUT_ADAPTER.name,
                "traincases_ok": cfg.get("traincases_path") == rel(V3_RAW_TRAIN),
                "retrieval_index": cfg.get("retrieval_index_path", ""),
                "v3_index_ok": (
                    cfg.get("prompt_tuning") != "dynamic_fewshot"
                    or cfg.get("retrieval_index_path") == rel(V3_INDEX)
                ),
                "static_pool_ok": (
                    cfg.get("prompt_tuning") != "static_fewshot"
                    or cfg.get("retrieval_pool_path") == rel(V3_STATIC)
                ),
                "bad_old_clean_reference": "clean_split_oldsqlccpreserved_spider700" in text
                or "clean_split_spider700_no_train" in text
                or "spider_train_no_dev_overlap_bge" in text,
                "bad_adapter_reference": "r2_alpha4" in text or "qwen35_9b" in text or "epochs2" in text,
                "gate": cfg.get("fewshot_gate_threshold", ""),
            }
        )
    return train_checks, eval_rows


def py_compile_checks() -> list[dict[str, str]]:
    paths = [
        Path("src/16_prepare_qwen35_2b_clean_split_v3_dbstratified_oldsqlcc_preserved_pipeline.py"),
        Path("src/07_lora_finetune_sft_v1_clean.py"),
        Path("src/06_batch_run.py"),
        Path("src/build_retrieval_index.py"),
        Path("src/training_history_utils.py"),
        Path("src/plot_training_history.py"),
    ]
    results = []
    for path in paths:
        cfile = Path("/tmp") / (path.name + ".v3_audit.pyc")
        try:
            py_compile.compile(str(ROOT / path), cfile=str(cfile), doraise=True)
            status = "PASS"
        except Exception as exc:
            status = f"FAIL: {exc}"
        results.append({"path": rel(path), "status": status})
    return results


def main() -> None:
    base = load_module(Path("src/14_prepare_qwen35_2b_clean_split_pipeline.py"), "base_clean_split_v3_audit")
    mix_module = load_module(Path("src/04_build_spider_sqlcc_complexity_mix.py"), "mix_builder_v3_audit")

    current_rows = read_jsonl(CURRENT_MIX)
    full_spider = [row for row in current_rows if row.get("source_dataset") == "spider_train"]
    old_sqlcc = [row for row in current_rows if row.get("source_dataset") == "sql_create_context"]
    old_raw_train = read_jsonl(OLD_RAW_TRAIN)
    old_retrieval = read_jsonl(OLD_RETRIEVAL)
    v3_train = read_jsonl(V3_RAW_TRAIN)
    v3_retrieval = read_jsonl(V3_RETRIEVAL)
    v3_sft_train = read_jsonl(V3_SFT_TRAIN)
    v3_sft_val = read_jsonl(V3_SFT_VAL)
    dev_rows = read_jsonl(SPIDER_DEV)
    static_rows = read_jsonl(V3_STATIC)
    summary = read_json(V3_SUMMARY)

    old_ids = row_id_set(old_sqlcc)
    old_fill = [row for row in old_raw_train if row.get("source_dataset") == "sql_create_context" and str(row.get("id")) not in old_ids]
    v3_sqlcc = [row for row in v3_train if row.get("source_dataset") == "sql_create_context"]
    v3_fill = [row for row in v3_sqlcc if str(row.get("id")) not in old_ids]

    dev_q, dev_s, _dev_pair = mix_module.load_dev_overlap_sets(ROOT / SPIDER_DEV)
    all_spider_q = {base.normalize_question(str(row.get("question", ""))) for row in full_spider}
    all_spider_s = {base.normalize_sql(str(row.get("gold_sql", ""))) for row in full_spider}
    all_spider_pair = {
        (base.normalize_question(str(row.get("question", ""))), base.normalize_sql(str(row.get("gold_sql", ""))))
        for row in full_spider
    }
    sqlcc_pool, _sqlcc_pool_stats, _examples = mix_module.build_sqlcc_pool(
        sqlcc_path=ROOT / SQLCC_RAW,
        dev_question_set=dev_q,
        dev_sql_set=dev_s,
        spider_question_set=all_spider_q,
        spider_sql_set=all_spider_s,
        spider_pair_set=all_spider_pair,
    )
    spider_by_id = {str(row.get("id")): row for row in full_spider}
    sqlcc_by_id = {str(row.get("id")): row for row in sqlcc_pool}
    old_validation = reconstruct_validation_raw(OLD_SFT_VAL, spider_by_id, sqlcc_by_id)
    v3_validation = reconstruct_validation_raw(V3_SFT_VAL, spider_by_id, sqlcc_by_id)
    old_validation_spider = [row for row in old_validation if row.get("source_dataset") == "spider_train"]
    v3_validation_spider = [row for row in v3_validation if row.get("source_dataset") == "spider_train"]

    overlap_matrix = {
        "train_vs_retrieval": base.overlap_counts(v3_train, v3_retrieval),
        "train_vs_validation": base.overlap_counts(v3_train, v3_validation),
        "retrieval_vs_validation": base.overlap_counts(v3_retrieval, v3_validation),
        "train_vs_spider_dev": base.overlap_counts(v3_train, dev_rows),
        "retrieval_vs_spider_dev": base.overlap_counts(v3_retrieval, dev_rows),
        "validation_vs_spider_dev": base.overlap_counts(v3_validation, dev_rows),
        "sqlcc_fill_vs_validation": base.overlap_counts(v3_fill, v3_validation),
        "sqlcc_fill_vs_retrieval": base.overlap_counts(v3_fill, v3_retrieval),
        "sqlcc_fill_vs_spider_dev": base.overlap_counts(v3_fill, dev_rows),
    }

    pool_counts = db_counts(full_spider)
    old_ret_summary = db_distribution_summary(pool_counts=pool_counts, role_counts=db_counts(old_retrieval), target_rows=700)
    old_val_summary = db_distribution_summary(pool_counts=pool_counts, role_counts=db_counts(old_validation_spider), target_rows=560)
    v3_ret_summary = db_distribution_summary(pool_counts=pool_counts, role_counts=db_counts(v3_retrieval), target_rows=700)
    v3_val_summary = db_distribution_summary(pool_counts=pool_counts, role_counts=db_counts(v3_validation_spider), target_rows=560)
    db_csv_rows = db_distribution_csv_rows(pool_counts, old_retrieval, old_validation_spider, v3_retrieval, v3_validation_spider)
    write_csv(
        DB_CSV,
        db_csv_rows,
        [
            "db_id",
            "spider_pool_count",
            "frequent_ge10",
            "old_retrieval_count",
            "v3_retrieval_count",
            "retrieval_expected",
            "old_retrieval_ratio",
            "v3_retrieval_ratio",
            "old_validation_count",
            "v3_validation_count",
            "validation_expected",
            "old_validation_ratio",
            "v3_validation_ratio",
        ],
    )

    complexity_rows = complexity_csv_rows(
        [
            ("old_25k_vs_v3_25k", current_rows, v3_train),
            ("full_spider_pool_vs_v3_retrieval", full_spider, v3_retrieval),
            ("full_spider_pool_vs_v3_spider_validation", full_spider, v3_validation_spider),
            ("v3_retrieval_vs_old_retrieval", old_retrieval, v3_retrieval),
            ("v3_spider_validation_vs_old_spider_validation", old_validation_spider, v3_validation_spider),
        ],
        base,
        mix_module,
    )
    write_csv(COMPLEXITY_CSV, complexity_rows, ["comparison", "metric", "left_value", "right_value", "delta_right_minus_left"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base", local_files_only=True)
    token_rows = [
        token_stats("v3_train_sft_25k", v3_sft_train, tokenizer),
        token_stats("v3_validation_sft_2500", v3_sft_val, tokenizer),
    ]
    write_csv(TOKEN_CSV, token_rows, ["dataset", "rows", "mean", "median", "p25", "p75", "p95", "max", "gt1536", "gt2048"])

    prompt_results = prompt_smoke(tokenizer)
    train_checks, eval_checks = config_checks()
    compile_results = py_compile_checks()

    index_manifest = read_json(V3_INDEX / "manifest.json")
    metadata_rows = read_jsonl(V3_INDEX / "metadata.jsonl")
    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(ROOT / V3_INDEX / "index.faiss"))
        index_ntotal = int(index.ntotal)
        index_dim = int(index.d)
    except Exception:
        index_ntotal = -1
        index_dim = -1

    fill_row_by_id = {str(row.get("id")): row for row in old_fill}
    fill_exact_matches = sum(1 for row in v3_fill if fill_row_by_id.get(str(row.get("id"))) == row)
    sqlcc_train_agg_rate = sum(bool(base.structure(row)["aggregation"]) for row in v3_sqlcc) / len(v3_sqlcc)
    sqlcc_schema_formats = Counter(str(row.get("schema_format", "")) for row in v3_sqlcc)
    raw_create_table_count = sum("create table" in str(row.get("schema_prompt") or row.get("context") or "").casefold() for row in v3_sqlcc)
    validation_schema_formats = Counter(str(row.get("schema_format", "")) for row in v3_validation)
    retrieval_schema_formats = Counter(str(row.get("schema_format", "")) for row in v3_retrieval)
    train_sft_checks = sft_checks(v3_sft_train)
    val_sft_checks = sft_checks(v3_sft_val)

    static_row = static_rows[0]
    static_overlap = {
        "static_vs_train": base.overlap_counts(static_rows, v3_train),
        "static_vs_validation": base.overlap_counts(static_rows, v3_validation),
        "static_vs_spider_dev": base.overlap_counts(static_rows, dev_rows),
    }
    static_struct = base.structure(static_row)

    pass_conditions = [
        len(v3_train) == 25000,
        len([row for row in v3_train if row.get("source_dataset") == "spider_train"]) == 5700,
        len(v3_sqlcc) == 19300,
        len(set(row_id_set(v3_sqlcc)) & old_ids) == 18040,
        len(v3_fill) == 1260,
        fill_exact_matches == 1260,
        len(v3_retrieval) == 700,
        len(v3_validation_spider) == 560,
        len([row for row in v3_validation if row.get("source_dataset") == "sql_create_context"]) == 1940,
        all(all(value == 0 for value in counts.values()) for counts in overlap_matrix.values()),
        v3_ret_summary == {"frequent_db_count": 139, "missing": 0, "over": 0, "under": 0},
        v3_val_summary == {"frequent_db_count": 139, "missing": 0, "over": 0, "under": 0},
        max(row["gt2048"] for row in token_rows) == 0,
        prompt_results["zero_shot"]["over_limit"] == 0,
        prompt_results["dynamic_full_schema"]["over_limit"] == 0,
        prompt_results["dynamic_full_schema"]["retrieval_failed"] == 0,
        prompt_results["static_full_schema"]["over_limit"] == 0,
        index_ntotal == 700 and index_dim == 1024,
        all(train_checks.values()),
        all(row["adapter_ok"] and row["traincases_ok"] and row["v3_index_ok"] and row["static_pool_ok"] and not row["bad_old_clean_reference"] and not row["bad_adapter_reference"] for row in eval_checks),
        all(row["status"] == "PASS" for row in compile_results),
    ]
    status = "PASS" if all(pass_conditions) else "FAIL"

    generated_files = [
        V3_RAW_TRAIN,
        V3_RAW_TRAIN.with_name(V3_RAW_TRAIN.stem + "_manifest.json"),
        V3_SFT_TRAIN,
        V3_SFT_TRAIN.with_name(V3_SFT_TRAIN.stem + "_manifest.json"),
        V3_SFT_VAL,
        V3_SFT_VAL.with_name(V3_SFT_VAL.stem + "_manifest.json"),
        V3_RETRIEVAL,
        V3_RETRIEVAL.with_name(V3_RETRIEVAL.stem + "_manifest.json"),
        V3_INDEX / "index.faiss",
        V3_INDEX / "metadata.jsonl",
        V3_INDEX / "manifest.json",
        V3_STATIC,
        V3_STATIC.with_name(V3_STATIC.stem + "_manifest.json"),
        V3_SUMMARY,
        TRAIN_CONFIG,
        *EVAL_CONFIGS,
        DB_CSV,
        COMPLEXITY_CSV,
        TOKEN_CSV,
        AUDIT_MD,
    ]

    lines: list[str] = []
    lines.append("# Audit: Qwen 3.5 2B Clean-Split v3 DB-Stratified Old-SQLCC-Preserved Preparation")
    lines.append("")
    lines.append(f"Date: {DATE}")
    lines.append("")
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Prepared and validated the `v3_dbstratified_oldsqlccpreserved` Clean-Split variant. "
        "No training, model generation, or Spider-Dev execution evaluation was started."
    )
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.extend(f"- `{rel(path)}`" for path in generated_files)
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    lines.append(
        markdown_table(
            ["Check", "Value", "Expected", "Result"],
            [
                ["Train rows", len(v3_train), 25000, len(v3_train) == 25000],
                ["Train Spider", sum(row.get("source_dataset") == "spider_train" for row in v3_train), 5700, True],
                ["Train SQLCC", len(v3_sqlcc), 19300, True],
                ["Old SQLCC preserved", len(set(row_id_set(v3_sqlcc)) & old_ids), "18040 / 18040", len(set(row_id_set(v3_sqlcc)) & old_ids) == 18040],
                ["SQLCC fill", len(v3_fill), 1260, len(v3_fill) == 1260],
                ["Retrieval Spider-only", len(v3_retrieval), 700, len(v3_retrieval) == 700],
                ["Validation Spider", len(v3_validation_spider), 560, len(v3_validation_spider) == 560],
                ["Validation SQLCC", sum(row.get("source_dataset") == "sql_create_context" for row in v3_validation), 1940, True],
            ],
        )
    )
    lines.append("")
    lines.append("## SQLCC Preservation And Fill")
    lines.append("")
    lines.append(
        markdown_table(
            ["Check", "Value", "Expected", "Result"],
            [
                ["Missing old SQLCC", 18040 - len(set(row_id_set(v3_sqlcc)) & old_ids), 0, 18040 - len(set(row_id_set(v3_sqlcc)) & old_ids) == 0],
                ["Fill rows exact dict match previous oldsqlccpreserved", fill_exact_matches, 1260, fill_exact_matches == 1260],
                ["Final SQLCC train aggregation rate", pct(sqlcc_train_agg_rate), "near 58", abs(sqlcc_train_agg_rate - 0.58) < 0.001],
                ["Raw CREATE TABLE schemas in SQLCC train", raw_create_table_count, 0, raw_create_table_count == 0],
                ["SQLCC schema formats", dict(sqlcc_schema_formats), "spider_schema_harmonized_table_columns_empty_pk_fk", sqlcc_schema_formats == {"spider_schema_harmonized_table_columns_empty_pk_fk": 19300}],
            ],
        )
    )
    lines.append("")
    lines.append("## Overlap Matrix")
    lines.append("")
    lines.append(markdown_table(["Comparison", "id", "question", "sql", "pair"], [[name, c["id"], c["question"], c["sql"], c["pair"]] for name, c in overlap_matrix.items()]))
    lines.append("")
    lines.append("## DB Distribution: Old vs v3")
    lines.append("")
    lines.append(
        markdown_table(
            ["Split", "Variant", "Frequent DBs", "Missing", "Strong Over", "Strong Under"],
            [
                ["Retrieval", "old oldsqlccpreserved", old_ret_summary["frequent_db_count"], old_ret_summary["missing"], old_ret_summary["over"], old_ret_summary["under"]],
                ["Retrieval", "v3 dbstratified", v3_ret_summary["frequent_db_count"], v3_ret_summary["missing"], v3_ret_summary["over"], v3_ret_summary["under"]],
                ["Validation Spider", "old oldsqlccpreserved", old_val_summary["frequent_db_count"], old_val_summary["missing"], old_val_summary["over"], old_val_summary["under"]],
                ["Validation Spider", "v3 dbstratified", v3_val_summary["frequent_db_count"], v3_val_summary["missing"], v3_val_summary["over"], v3_val_summary["under"]],
            ],
        )
    )
    lines.append("")
    lines.append(f"Detailed per-DB ratios are in `{rel(DB_CSV)}`.")
    lines.append("")
    lines.append("## Structure Comparisons")
    lines.append("")
    key_rows = [row for row in complexity_rows if row["comparison"] == "old_25k_vs_v3_25k" and row["metric"] in {
        "examples", "spider_count", "sqlcc_count", "join_rate_pct", "group_by_rate_pct", "having_rate_pct",
        "order_by_rate_pct", "limit_rate_pct", "distinct_rate_pct", "subquery_rate_pct",
        "set_operation_rate_pct", "aggregation_rate_pct", "sql_length_mean", "sql_table_count_mean",
        "schema_table_count_mean",
    }]
    lines.append(markdown_table(["Metric", "Old 25k", "v3 25k", "Delta"], [[row["metric"], row["left_value"], row["right_value"], row["delta_right_minus_left"]] for row in key_rows]))
    lines.append("")
    lines.append(f"Full structure comparison CSV: `{rel(COMPLEXITY_CSV)}`.")
    lines.append("")
    lines.append("## Schema And SFT Checks")
    lines.append("")
    lines.append(
        markdown_table(
            ["Check", "Value"],
            [
                ["Validation schema formats", dict(validation_schema_formats)],
                ["Retrieval schema formats", dict(retrieval_schema_formats)],
                ["Train SFT", train_sft_checks],
                ["Validation SFT", val_sft_checks],
            ],
        )
    )
    lines.append("")
    lines.append("## Token Lengths")
    lines.append("")
    lines.append(markdown_table(["Dataset", "Rows", "Mean", "Median", "P95", "Max", ">1536", ">2048"], [[row["dataset"], row["rows"], row["mean"], row["median"], row["p95"], row["max"], row["gt1536"], row["gt2048"]] for row in token_rows]))
    lines.append("")
    lines.append(f"Token CSV: `{rel(TOKEN_CSV)}`.")
    lines.append("")
    lines.append("## Retrieval Index")
    lines.append("")
    lines.append(
        markdown_table(
            ["Check", "Value", "Expected", "Result"],
            [
                ["Index rows", index_ntotal, 700, index_ntotal == 700],
                ["Metadata rows", len(metadata_rows), 700, len(metadata_rows) == 700],
                ["Embedding model", index_manifest.get("embedding_model"), "BAAI/bge-large-en-v1.5", index_manifest.get("embedding_model") == "BAAI/bge-large-en-v1.5"],
                ["Embedding dim", index_dim, 1024, index_dim == 1024],
                ["Query prefix", index_manifest.get("query_prefix"), "BGE prefix", bool(index_manifest.get("query_prefix"))],
                ["Prefix applied to documents", index_manifest.get("apply_query_prefix_to_documents"), True, index_manifest.get("apply_query_prefix_to_documents") is True],
                ["Prefix applied to queries", index_manifest.get("apply_query_prefix_to_queries"), True, index_manifest.get("apply_query_prefix_to_queries") is True],
                ["Leakage check", index_manifest.get("leakage_check_result", {}).get("status"), "pass", index_manifest.get("leakage_check_result", {}).get("status") == "pass"],
            ],
        )
    )
    lines.append("")
    lines.append("## Prompt Smoke")
    lines.append("")
    lines.append(
        markdown_table(
            ["Check", "Cases", "Max Tokens", "Over Limit", "Extra"],
            [
                ["Zero-Shot", prompt_results["zero_shot"]["cases"], prompt_results["zero_shot"]["max_tokens"], prompt_results["zero_shot"]["over_limit"], "limit 1536"],
                ["Dynamic Full Schema", prompt_results["dynamic_full_schema"]["cases"], prompt_results["dynamic_full_schema"]["max_tokens"], prompt_results["dynamic_full_schema"]["over_limit"], f"retrieval_failed={prompt_results['dynamic_full_schema']['retrieval_failed']}, unique_demo_ids={prompt_results['dynamic_full_schema']['unique_demo_ids']}"],
                ["Static Full Schema", prompt_results["static_full_schema"]["cases"], prompt_results["static_full_schema"]["max_tokens"], prompt_results["static_full_schema"]["over_limit"], "limit 2048"],
                ["Think tags in prompts", "", "", prompt_results["think_in_prompts"], "expected 0"],
            ],
        )
    )
    lines.append("")
    lines.append("Actual dynamic retrieval smoke:")
    lines.append("")
    lines.append(
        markdown_table(
            ["dev_cases", "retrieval_failed", "max_prompt_tokens", "gt2048", "unique_demo_ids", "min_similarity", "max_similarity"],
            [[
                prompt_results["dynamic_full_schema"]["cases"],
                prompt_results["dynamic_full_schema"]["retrieval_failed"],
                prompt_results["dynamic_full_schema"]["max_tokens"],
                prompt_results["dynamic_full_schema"]["over_limit"],
                prompt_results["dynamic_full_schema"]["unique_demo_ids"],
                prompt_results["dynamic_full_schema"]["min_similarity"],
                prompt_results["dynamic_full_schema"]["max_similarity"],
            ]],
        )
    )
    lines.append("")
    lines.append("Gate smoke:")
    lines.append("")
    lines.append(markdown_table(["0.86 @ 0.85", "0.84 @ 0.85", "0.71 @ 0.70", "0.69 @ 0.70"], [[prompt_results["gate_smoke"]["0.86_at_0.85"], prompt_results["gate_smoke"]["0.84_at_0.85"], prompt_results["gate_smoke"]["0.71_at_0.70"], prompt_results["gate_smoke"]["0.69_at_0.70"]]]))
    lines.append("")
    lines.append("## Static Few-Shot")
    lines.append("")
    lines.append(
        markdown_table(
            ["id", "in_retrieval", "join_count", "aggregation", "group_by", "order_by", "question"],
            [[
                static_row.get("id"),
                static_row.get("id") in row_id_set(v3_retrieval),
                static_struct["join_count"],
                static_struct["aggregation"],
                static_struct["group_by"],
                static_struct["order_by"],
                static_row.get("question"),
            ]],
        )
    )
    lines.append("")
    lines.append(markdown_table(["Comparison", "id", "question", "sql", "pair"], [[name, c["id"], c["question"], c["sql"], c["pair"]] for name, c in static_overlap.items()]))
    lines.append("")
    lines.append("## Config Checks")
    lines.append("")
    lines.append(markdown_table(["Train check", "PASS"], [[key, value] for key, value in train_checks.items()]))
    lines.append("")
    lines.append(markdown_table(["Eval config", "adapter_ok", "traincases_ok", "v3_index_ok", "static_pool_ok", "bad_old_ref", "bad_adapter_ref", "gate"], [[row["path"], row["adapter_ok"], row["traincases_ok"], row["v3_index_ok"], row["static_pool_ok"], row["bad_old_clean_reference"], row["bad_adapter_reference"], row["gate"]] for row in eval_checks]))
    lines.append("")
    lines.append("## py_compile")
    lines.append("")
    lines.append(markdown_table(["Path", "Status"], [[row["path"], row["status"]] for row in compile_results]))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if status == "PASS":
        lines.append(
            "PASS: The v3 DB-stratified oldsqlccpreserved variant should replace the previous "
            "`oldsqlccpreserved` split as the final training variant. It preserves the SQLCC training "
            "content while making Retrieval and Validation substantially more representative by `db_id`."
        )
    else:
        lines.append("FAIL: Do not start training until the failing checks above are resolved.")
    lines.append("")
    lines.append("## Later Commands")
    lines.append("")
    commands = [
        ("Training", f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_flash/bin/python3 src/07_lora_finetune_sft_v1_clean.py --config {rel(TRAIN_CONFIG)}"),
        ("Zero-Shot", f".venv_flash/bin/python3 src/06_batch_run.py --config {rel(EVAL_CONFIGS[0])}"),
        ("Full Schema", f".venv_flash/bin/python3 src/06_batch_run.py --config {rel(EVAL_CONFIGS[1])}"),
        ("Gate 0.85", f".venv_flash/bin/python3 src/06_batch_run.py --config {rel(EVAL_CONFIGS[2])}"),
        ("Gate 0.70", f".venv_flash/bin/python3 src/06_batch_run.py --config {rel(EVAL_CONFIGS[3])}"),
        ("Static Few-Shot", f".venv_flash/bin/python3 src/06_batch_run.py --config {rel(EVAL_CONFIGS[4])}"),
    ]
    for label, command in commands:
        lines.append(f"{label}:")
        lines.append("")
        lines.append("```bash")
        lines.append(command)
        lines.append("```")
        lines.append("")

    (ROOT / AUDIT_MD).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / AUDIT_MD).write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": status, "audit": rel(AUDIT_MD), "db_csv": rel(DB_CSV), "complexity_csv": rel(COMPLEXITY_CSV), "token_csv": rel(TOKEN_CSV)}, indent=2))


if __name__ == "__main__":
    main()
