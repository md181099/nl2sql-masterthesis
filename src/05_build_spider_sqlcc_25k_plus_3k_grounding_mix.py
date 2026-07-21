#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_NAME = "spider_train_sqlcc_25k_plus_3k_grounding"
DEFAULT_BASE_MIX = (
    "data/sql_create_context/archive/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
DEFAULT_OUTPUT = (
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_25k_plus_3k_grounding_seed42_no_dev_overlap.jsonl"
)
DEFAULT_MANIFEST = (
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_25k_plus_3k_grounding_seed42_no_dev_overlap_manifest.json"
)
TARGET_BUCKETS: OrderedDict[str, int] = OrderedDict(
    [
        ("schema_grounding", 900),
        ("table_selection", 500),
        ("join_decision", 500),
        ("difficult_filters", 500),
        ("aggregation_group_order_repair", 400),
        ("robustness_fill", 200),
    ]
)
BUCKET_LABELS = {
    "schema_grounding": "Schema Grounding / Column Disambiguation",
    "table_selection": "Table Selection / No-such-table Robustness",
    "join_decision": "Join Decision",
    "difficult_filters": "Schwierige Filter",
    "aggregation_group_order_repair": "Aggregation / GROUP / ORDER Repair",
    "robustness_fill": "Robustness-Fill",
}
BUCKET_OUTPUT_NAMES = {
    "schema_grounding": "sqlcc_25k_plus_3k_schema_grounding",
    "table_selection": "sqlcc_25k_plus_3k_table_selection",
    "join_decision": "sqlcc_25k_plus_3k_join_decision",
    "difficult_filters": "sqlcc_25k_plus_3k_difficult_filters",
    "aggregation_group_order_repair": "sqlcc_25k_plus_3k_aggregation_group_order_repair",
    "robustness_fill": "sqlcc_25k_plus_3k_robustness_fill",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 25k best mix plus 3k targeted SQLCC grounding examples."
    )
    parser.add_argument("--base_mix_path", default=DEFAULT_BASE_MIX)
    parser.add_argument("--sqlcc_path", default="data/sql_create_context/train.jsonl")
    parser.add_argument("--spider_dir", default="data/spider/spider_data")
    parser.add_argument("--spider_train_path", default="data/spider/spider_data/train_spider.json")
    parser.add_argument("--dev_reference_path", default="data/testcases_spider_dev_full.jsonl")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--additional_sqlcc_size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_module(project_root: Path, relative_path: str, module_name: str):
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def strip_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", str(sql or ""))


def where_clause(sql: str) -> str:
    low = strip_literals(sql).lower()
    match = re.search(r"\bwhere\b(.+)", low, flags=re.DOTALL)
    if match is None:
        return ""
    return re.split(
        r"\b(group\s+by|having|order\s+by|limit|union|intersect|except)\b",
        match.group(1),
        maxsplit=1,
        flags=re.DOTALL,
    )[0].strip()


def normalize_pair(helpers, row: dict[str, Any]) -> tuple[str, str]:
    return (
        helpers.normalize_question(str(row.get("question", ""))),
        helpers.normalize_sql(str(row.get("gold_sql", ""))),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def category_flags(row: dict[str, Any], relevance_builder) -> dict[str, bool]:
    sql = str(row.get("gold_sql", ""))
    features = row.get("sql_features") or {}
    selection_features = row.get("selection_features") or relevance_builder.make_selection_features(row)
    where = where_clause(sql)
    join_count = int(selection_features.get("join_count", 0))
    query_table_count = int(selection_features.get("query_table_count", 0))
    schema_table_count = int(selection_features.get("schema_table_count", 0))
    column_count = int(selection_features.get("column_count", 0))
    has_projection = relevance_builder.is_projection_query(sql)
    count_only_simple = relevance_builder.count_only_simple(sql, features)

    schema_grounding = bool(
        query_table_count <= 1
        and column_count >= 3
        and (where or has_projection)
    )
    # Strict table-selection evidence outside the 25k mix is expected to be scarce:
    # it requires either actual multi-table use or a multi-table schema where one table is enough.
    table_selection = bool(
        query_table_count >= 2
        or join_count > 0
        or (schema_table_count >= 2 and query_table_count == 1)
    )
    join_decision = bool(
        join_count > 0
        or (query_table_count <= 1 and column_count >= 4 and (where or has_projection))
    )
    difficult_filters = bool(
        where
        and (
            int(selection_features.get("condition_count", 0)) >= 2
            or selection_features.get("has_and")
            or selection_features.get("has_or")
            or selection_features.get("has_range")
            or selection_features.get("has_like")
            or selection_features.get("has_in")
        )
    )
    aggregation_group_order_repair = bool(
        features.get("any_aggregation")
        or features.get("group_by")
        or features.get("having")
        or features.get("order_by")
        or features.get("limit")
        or features.get("distinct")
    )
    robustness_fill = bool(column_count > 0 and not count_only_simple)
    return {
        "schema_grounding": schema_grounding,
        "table_selection": table_selection,
        "join_decision": join_decision,
        "difficult_filters": difficult_filters,
        "aggregation_group_order_repair": aggregation_group_order_repair,
        "robustness_fill": robustness_fill,
    }


def candidate_score(row: dict[str, Any], bucket: str, relevance_builder, rng_values: dict[str, float]) -> tuple[int, int, float]:
    sql = str(row.get("gold_sql", ""))
    features = row.get("sql_features") or {}
    selection_features = row.get("selection_features") or relevance_builder.make_selection_features(row)
    score = 0
    if bucket == "schema_grounding":
        score = (
            5 * bool(where_clause(sql))
            + 4 * min(int(selection_features.get("column_count", 0)), 8)
            + 3 * bool(relevance_builder.is_projection_query(sql))
            + 3 * bool(int(selection_features.get("condition_count", 0)) >= 2)
        )
    elif bucket == "table_selection":
        score = (
            12 * bool(int(selection_features.get("query_table_count", 0)) >= 2)
            + 8 * bool(int(selection_features.get("schema_table_count", 0)) >= 2)
            + 4 * bool(where_clause(sql))
        )
    elif bucket == "join_decision":
        score = (
            12 * bool(int(selection_features.get("join_count", 0)) > 0)
            + 7 * bool(int(selection_features.get("query_table_count", 0)) <= 1)
            + 3 * min(int(selection_features.get("column_count", 0)), 10)
            + 3 * bool(where_clause(sql))
        )
    elif bucket == "difficult_filters":
        score = (
            6 * min(int(selection_features.get("condition_count", 0)), 4)
            + 5 * bool(selection_features.get("has_range"))
            + 5 * bool(selection_features.get("has_or"))
            + 3 * bool(selection_features.get("has_and"))
            + 3 * bool(selection_features.get("filtered_min_max"))
            + 3 * bool(selection_features.get("filtered_count"))
        )
    elif bucket == "aggregation_group_order_repair":
        score = (
            7 * bool(features.get("group_by"))
            + 8 * bool(features.get("having"))
            + 6 * bool(features.get("order_by"))
            + 6 * bool(features.get("limit"))
            + 5 * bool(features.get("any_aggregation"))
            + 4 * bool(features.get("distinct"))
        )
    elif bucket == "robustness_fill":
        score = (
            3 * bool(where_clause(sql))
            + 2 * min(int(selection_features.get("column_count", 0)), 8)
            + 2 * bool(int(selection_features.get("condition_count", 0)) >= 2)
            - 10 * bool(relevance_builder.count_only_simple(sql, features))
        )
    else:
        score = sum(candidate_score(row, b, relevance_builder, rng_values)[0] for b in TARGET_BUCKETS)
    return (score, int(selection_features.get("column_count", 0)), rng_values[str(row["id"])])


def category_flag_counts(rows: list[dict[str, Any]], relevance_builder) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        flags = category_flags(row, relevance_builder)
        for key, value in flags.items():
            if value:
                counts[BUCKET_LABELS[key]] += 1
    return dict(counts)


def rich_distribution(rows: list[dict[str, Any]], relevance_builder) -> dict[str, Any]:
    return relevance_builder.rich_distribution(rows)


def select_additional_sqlcc(
    *,
    candidates: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
    relevance_builder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_size != sum(TARGET_BUCKETS.values()):
        raise ValueError(f"Expected additional target {sum(TARGET_BUCKETS.values())}, got {target_size}")
    rng_values = {str(row["id"]): rng.random() for row in candidates}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    gaps: dict[str, int] = {}

    for bucket, target in TARGET_BUCKETS.items():
        bucket_candidates = [
            row
            for row in candidates
            if row["id"] not in selected_ids and category_flags(row, relevance_builder)[bucket]
        ]
        bucket_candidates.sort(
            key=lambda row: candidate_score(row, bucket, relevance_builder, rng_values),
            reverse=True,
        )
        take = bucket_candidates[:target]
        for row in take:
            row["selection_bucket"] = BUCKET_OUTPUT_NAMES[bucket]
            row["selection_primary_class"] = BUCKET_LABELS[bucket]
            row["selection_score"] = candidate_score(row, bucket, relevance_builder, rng_values)[0]
            row["mix_component"] = "additional_sqlcc_3k"
            row["is_new_sqlcc_addition"] = True
            selected_ids.add(str(row["id"]))
        selected.extend(take)
        if len(take) < target:
            gaps[BUCKET_LABELS[bucket]] = target - len(take)

    if len(selected) < target_size:
        remaining = [
            row
            for row in candidates
            if row["id"] not in selected_ids
            and any(category_flags(row, relevance_builder).values())
        ]
        remaining.sort(
            key=lambda row: (
                sum(candidate_score(row, bucket, relevance_builder, rng_values)[0] for bucket in TARGET_BUCKETS),
                int((row.get("selection_features") or {}).get("column_count", 0)),
                rng_values[str(row["id"])],
            ),
            reverse=True,
        )
        fallback_take = remaining[: target_size - len(selected)]
        for row in fallback_take:
            row["selection_bucket"] = "sqlcc_25k_plus_3k_fallback_fill"
            row["selection_primary_class"] = "Fallback aus naechstbesten geeigneten Kategorien"
            row["selection_score"] = sum(
                candidate_score(row, bucket, relevance_builder, rng_values)[0]
                for bucket in TARGET_BUCKETS
            )
            row["mix_component"] = "additional_sqlcc_3k"
            row["is_new_sqlcc_addition"] = True
            selected_ids.add(str(row["id"]))
        selected.extend(fallback_take)

    if len(selected) != target_size:
        raise RuntimeError(f"Could not select {target_size} additions; got {len(selected)}")

    return selected, {
        "target_size": target_size,
        "target_bucket_sizes": {BUCKET_LABELS[k]: v for k, v in TARGET_BUCKETS.items()},
        "priority_order": [BUCKET_LABELS[k] for k in TARGET_BUCKETS],
        "available_category_flags": category_flag_counts(candidates, relevance_builder),
        "selected_category_flags": category_flag_counts(selected, relevance_builder),
        "selected_bucket_sizes": dict(sorted(Counter(row["selection_bucket"] for row in selected).items())),
        "bucket_gaps": gaps,
        "selected_distribution": rich_distribution(selected, relevance_builder),
    }


def leakage_counts(helpers, rows: list[dict[str, Any]], dev_sets) -> dict[str, int]:
    dev_q, dev_s, dev_pair = dev_sets
    return {
        "normalized_question_overlap": sum(
            helpers.normalize_question(str(row.get("question", ""))) in dev_q for row in rows
        ),
        "normalized_sql_overlap": sum(
            helpers.normalize_sql(str(row.get("gold_sql", ""))) in dev_s for row in rows
        ),
        "normalized_question_sql_pair_overlap": sum(
            normalize_pair(helpers, row) in dev_pair for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    helpers = import_module(project_root, "src/04_build_spider_sqlcc_complexity_mix.py", "complexity_helpers")
    relevance_builder = import_module(project_root, "src/05_build_spider_sqlcc_relevance_selected_mix.py", "relevance_helpers")

    base_mix_path = resolve_path(project_root, args.base_mix_path)
    sqlcc_path = resolve_path(project_root, args.sqlcc_path)
    spider_train_path = resolve_path(project_root, args.spider_train_path)
    spider_dir = resolve_path(project_root, args.spider_dir)
    dev_reference_path = resolve_path(project_root, args.dev_reference_path)
    output_path = resolve_path(project_root, args.output_path)
    manifest_path = resolve_path(project_root, args.manifest_path)

    for path in (base_mix_path, sqlcc_path, spider_train_path, dev_reference_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
    for path in (output_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    rng = random.Random(args.seed)
    base_rows_raw = load_jsonl(base_mix_path)
    base_source_counts = Counter(str(row.get("source_dataset", "unknown")) for row in base_rows_raw)
    if len(base_rows_raw) != 25000 or base_source_counts.get("spider_train") != 6960 or base_source_counts.get("sql_create_context") != 18040:
        raise RuntimeError(f"Unexpected base mix composition: {len(base_rows_raw)}, {dict(base_source_counts)}")

    base_rows: list[dict[str, Any]] = []
    for row in base_rows_raw:
        row = dict(row)
        row["mix_component"] = "existing_25k_mix"
        row["is_new_sqlcc_addition"] = False
        base_rows.append(row)

    base_ids = {str(row.get("id")) for row in base_rows}
    base_pairs = {normalize_pair(helpers, row) for row in base_rows}
    base_sqlcc_ids = {
        str(row.get("id"))
        for row in base_rows
        if row.get("source_dataset") == "sql_create_context"
    }
    base_sqlcc_pairs = {
        normalize_pair(helpers, row)
        for row in base_rows
        if row.get("source_dataset") == "sql_create_context"
    }

    build_schema_prompt = helpers.load_spider_schema_builder(project_root)
    dev_sets = helpers.load_dev_overlap_sets(dev_reference_path)
    dev_question_set, dev_sql_set, _ = dev_sets
    spider_rows_raw = helpers.load_json_array(spider_train_path)
    spider_rows, _spider_stats = helpers.build_spider_rows(
        spider_rows_raw=spider_rows_raw,
        spider_dir=spider_dir,
        build_schema_prompt=build_schema_prompt,
        dev_question_set=dev_question_set,
        dev_sql_set=dev_sql_set,
    )
    spider_question_set = {helpers.normalize_question(str(row["question"])) for row in spider_rows}
    spider_sql_set = {helpers.normalize_sql(str(row["gold_sql"])) for row in spider_rows}
    spider_pair_set = {
        (helpers.normalize_question(str(row["question"])), helpers.normalize_sql(str(row["gold_sql"])))
        for row in spider_rows
    }

    sqlcc_pool, sqlcc_pool_stats, conversion_examples = relevance_builder.build_sqlcc_pool(
        helpers=helpers,
        sqlcc_path=sqlcc_path,
        dev_sets=dev_sets,
        spider_question_set=spider_question_set,
        spider_sql_set=spider_sql_set,
        spider_pair_set=spider_pair_set,
    )

    excluded_from_pool = Counter()
    candidates: list[dict[str, Any]] = []
    for row in sqlcc_pool:
        row_id = str(row.get("id"))
        pair = normalize_pair(helpers, row)
        if row_id in base_ids:
            excluded_from_pool["id_overlap_with_25k_mix"] += 1
            continue
        if pair in base_pairs:
            excluded_from_pool["question_sql_pair_overlap_with_25k_mix"] += 1
            continue
        row["selection_features"] = relevance_builder.make_selection_features(row)
        candidates.append(row)

    additional_rows, selection_stats = select_additional_sqlcc(
        candidates=candidates,
        target_size=args.additional_sqlcc_size,
        rng=rng,
        relevance_builder=relevance_builder,
    )

    additional_ids = {str(row.get("id")) for row in additional_rows}
    additional_pairs = {normalize_pair(helpers, row) for row in additional_rows}
    additional_validation = {
        "id_overlap_with_25k_mix_sqlcc": len(additional_ids & base_sqlcc_ids),
        "id_overlap_with_25k_mix_all": len(additional_ids & base_ids),
        "question_sql_pair_overlap_with_25k_mix_sqlcc": len(additional_pairs & base_sqlcc_pairs),
        "question_sql_pair_overlap_with_25k_mix_all": len(additional_pairs & base_pairs),
        "dev_leakage": leakage_counts(helpers, additional_rows, dev_sets),
    }

    final_rows = base_rows + additional_rows
    rng.shuffle(final_rows)
    final_counts = Counter()
    for row in final_rows:
        if row.get("mix_component") == "additional_sqlcc_3k":
            final_counts["additional_sqlcc_3k"] += 1
        elif row.get("source_dataset") == "spider_train":
            final_counts["spider_train_existing_25k"] += 1
        elif row.get("source_dataset") == "sql_create_context":
            final_counts["existing_25k_sqlcc"] += 1
        else:
            final_counts["unknown"] += 1

    distribution = helpers.dataset_distribution(final_rows)
    validation = {
        "target_size_ok": len(final_rows) == 28000,
        "component_counts": dict(final_counts),
        "source_counts": dict(Counter(str(row.get("source_dataset", "unknown")) for row in final_rows)),
        "additional_examples": additional_validation,
        "final_dev_leakage": leakage_counts(helpers, final_rows, dev_sets),
        "duplicates": helpers.duplicate_counts(final_rows),
        "schema_validation": {
            "create_table_count": distribution["schema_create_table_count"],
            "missing_required_schema_labels": distribution["missing_required_schema_labels"],
            "missing_foreign_keys_label": distribution["missing_foreign_keys_label"],
        },
    }
    validation["all_passed"] = (
        validation["target_size_ok"]
        and final_counts["spider_train_existing_25k"] == 6960
        and final_counts["existing_25k_sqlcc"] == 18040
        and final_counts["additional_sqlcc_3k"] == args.additional_sqlcc_size
        and additional_validation["id_overlap_with_25k_mix_all"] == 0
        and additional_validation["question_sql_pair_overlap_with_25k_mix_all"] == 0
        and additional_validation["dev_leakage"]["normalized_question_overlap"] == 0
        and additional_validation["dev_leakage"]["normalized_sql_overlap"] == 0
        and additional_validation["dev_leakage"]["normalized_question_sql_pair_overlap"] == 0
        and validation["final_dev_leakage"]["normalized_question_overlap"] == 0
        and validation["final_dev_leakage"]["normalized_sql_overlap"] == 0
        and validation["final_dev_leakage"]["normalized_question_sql_pair_overlap"] == 0
        and validation["schema_validation"]["create_table_count"] == 0
        and validation["schema_validation"]["missing_required_schema_labels"] == 0
    )
    if not validation["all_passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    helpers.write_jsonl(output_path, final_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/05_build_spider_sqlcc_25k_plus_3k_grounding_mix.py",
        "target_name": TARGET_NAME,
        "seed": args.seed,
        "target_size": 28000,
        "actual_size": len(final_rows),
        "base_mix_size": len(base_rows),
        "additional_sqlcc_target_size": args.additional_sqlcc_size,
        "source_paths": {
            "base_mix": str(base_mix_path.relative_to(project_root)),
            "sql_create_context": str(sqlcc_path.relative_to(project_root)),
            "spider_train": str(spider_train_path.relative_to(project_root)),
            "spider_dir": str(spider_dir.relative_to(project_root)),
            "dev_reference": str(dev_reference_path.relative_to(project_root)),
        },
        "output_path": str(output_path.relative_to(project_root)),
        "sampling_strategy": (
            "Keep the existing 25k best mix unchanged as content, then add exactly 3000 "
            "new SQLCC examples not present by ID or normalized Question+SQL pair. "
            "Prioritize schema grounding, table selection, join decision, difficult filters, "
            "aggregation/group/order repair, and robustness fill."
        ),
        "selection_targets": {BUCKET_LABELS[k]: v for k, v in TARGET_BUCKETS.items()},
        "priority_order": [BUCKET_LABELS[k] for k in TARGET_BUCKETS],
        "quality_filters": {
            "schema_ok": "SQLite executes original CREATE TABLE context and EXPLAIN QUERY PLAN for SQL",
            "spider_schema_harmonizable": True,
            "dev_overlap_clean": True,
            "exclude_existing_25k_id": True,
            "exclude_existing_25k_question_sql_pair": True,
            "sqlcc_internal_question_dedupe": True,
            "sqlcc_internal_sql_dedupe": True,
            "no_create_table_in_harmonized_schema": True,
        },
        "base_mix": {
            "rows": len(base_rows),
            "source_distribution": dict(base_source_counts),
            "bucket_distribution": dict(Counter(str(row.get("selection_bucket", "unknown")) for row in base_rows)),
            "distribution": helpers.dataset_distribution(base_rows),
            "rich_distribution": rich_distribution(base_rows, relevance_builder),
        },
        "sql_create_context_pool": {
            **sqlcc_pool_stats,
            "strict_pool_before_25k_exclusion": len(sqlcc_pool),
            "candidate_after_25k_exclusion": len(candidates),
            "excluded_from_pool": dict(excluded_from_pool),
            "available_category_flags": category_flag_counts(candidates, relevance_builder),
        },
        "additional_sqlcc_selection": selection_stats,
        "final_distribution": distribution,
        "final_rich_distribution": rich_distribution(final_rows, relevance_builder),
        "validation": validation,
        "conversion_examples": conversion_examples,
        "input_sha256": {
            "base_mix": helpers.sha256_file(base_mix_path),
            "sql_create_context": helpers.sha256_file(sqlcc_path),
            "spider_train": helpers.sha256_file(spider_train_path),
            "dev_reference": helpers.sha256_file(dev_reference_path),
        },
    }
    manifest["output_sha256"] = helpers.sha256_file(output_path)
    helpers.write_json(manifest_path, manifest)

    print(f"Wrote 28k mix dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(manifest["additional_sqlcc_selection"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
