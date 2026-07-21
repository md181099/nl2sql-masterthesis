#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
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

BUILDER_SCRIPT = "src/16_prepare_qwen35_2b_clean_split_v3_dbstratified_oldsqlcc_preserved_pipeline.py"

RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_v3_dbstratified_oldsqlccpreserved_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_TRAIN = Path(
    "data/sql_create_context/"
    "train_sft_qwen35_2b_clean_split_v3_dbstratified_oldsqlccpreserved_full_chat_v1_clean_anti_overjoin_"
    "spider5700_sqlcc19300_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_VAL = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_2b_clean_split_v3_dbstratified_oldsqlccpreserved_spider560_sqlcc1940_"
    "full_chat_v1_clean_anti_overjoin_no_train_no_retrieval_no_dev_2500_seed42.jsonl"
)
RETRIEVAL_POOL = Path(
    "data/retrieval_pools/"
    "clean_split_v3_dbstratified_oldsqlccpreserved_spider700_no_train_no_val_no_dev_seed42.jsonl"
)
RETRIEVAL_INDEX = Path(
    "data/retrieval_indexes/"
    "clean_split_v3_dbstratified_oldsqlccpreserved_spider700_no_train_no_val_no_dev_bge_large_en_v15"
)
STATIC_FEWSHOT = Path(
    "data/fewshot_static/"
    "static_fewshot_clean_split_v3_dbstratified_oldsqlccpreserved_spider700_k1_full_schema_seed42.jsonl"
)
SUMMARY_PATH = Path(
    "results/analyses/clean_split_v3_dbstratified_oldsqlccpreserved_pipeline_preparation_summary.json"
)

TRAIN_CONFIG = Path(
    "configs/train_lora_qwen35_2b_base_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_"
    "evalval2500_earlystop_maxlen2048_oomsafe.json"
)
EVAL_ZERO = Path(
    "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_zero_shot_full_aliasnames.json"
)
EVAL_FULL = Path(
    "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_maxinput2048_full_aliasnames.json"
)
EVAL_GATE085 = Path(
    "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate085_"
    "maxinput2048_full_aliasnames.json"
)
EVAL_GATE070 = Path(
    "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate070_"
    "maxinput2048_full_aliasnames.json"
)
EVAL_STATIC = Path(
    "configs/eval_qwen35_2b_lora_clean_split_v3_dbstratified_oldsqlccpreserved_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_static_fewshot_k1_full_schema_clean_retrieval_maxinput2048_full_aliasnames.json"
)

OUTPUT_ADAPTER = (
    "adapters/qwen35_2b_base/"
    "lora_clean_split_v3_dbstratified_oldsqlccpreserved_qwen35_2b_r8_alpha16_evalval2500_"
    "earlystop_maxlen2048_oomsafe"
)

CURRENT_MIX = Path(
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
PREVIOUS_OLDSQLCC_RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_oldsqlccpreserved_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SQLCC_RAW = Path("data/sql_create_context/train.jsonl")
SPIDER_DEV = Path("data/testcases_spider_dev_full.jsonl")
TRAIN_CONFIG_REF = Path(
    "configs/train_lora_qwen35_2b_base_clean_split_oldsqlccpreserved_r8_alpha16_"
    "evalval2500_earlystop_maxlen2048_oomsafe.json"
)


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v3 DB-stratified old-SQLCC-preserved clean-split artifacts."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated v3 artifacts.")
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


def stable_hash(value: Any) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12], 16)


def sets_for(rows: list[dict[str, Any]], base: Any) -> dict[str, set[Any]]:
    return base.row_sets(rows)


def no_overlap_filter(rows: list[dict[str, Any]], protected_rows: list[dict[str, Any]], base: Any) -> list[dict[str, Any]]:
    protected = sets_for(protected_rows, base)
    kept: list[dict[str, Any]] = []
    for row in rows:
        q = base.normalize_question(str(row.get("question", "")))
        s = base.normalize_sql(str(row.get("gold_sql", "")))
        pair = (q, s)
        if (
            str(row.get("id", "")).strip() in protected["id"]
            or q in protected["question"]
            or s in protected["sql"]
            or pair in protected["pair"]
        ):
            continue
        kept.append(row)
    return kept


def build_overlap_components_by_all_keys(rows: list[dict[str, Any]], base: Any) -> list[list[dict[str, Any]]]:
    dsu = DSU(len(rows))
    maps: list[dict[Any, int]] = [{}, {}, {}, {}]
    for idx, row in enumerate(rows):
        q = base.normalize_question(str(row.get("question", "")))
        s = base.normalize_sql(str(row.get("gold_sql", "")))
        keys = [str(row.get("id", "")).strip(), q, s, (q, s)]
        for key_idx, key in enumerate(keys):
            if not key:
                continue
            if key in maps[key_idx]:
                dsu.union(idx, maps[key_idx][key])
            else:
                maps[key_idx][key] = idx

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[dsu.find(idx)].append(row)
    components = list(grouped.values())
    for comp in components:
        comp.sort(key=lambda row: str(row.get("id", "")))
    return components


def component_conflicts_with_sets(component: list[dict[str, Any]], sqlcc_sets: dict[str, set[Any]], base: Any) -> bool:
    for row in component:
        q = base.normalize_question(str(row.get("question", "")))
        s = base.normalize_sql(str(row.get("gold_sql", "")))
        pair = (q, s)
        if (
            str(row.get("id", "")).strip() in sqlcc_sets["id"]
            or q in sqlcc_sets["question"]
            or s in sqlcc_sets["sql"]
            or pair in sqlcc_sets["pair"]
        ):
            return True
    return False


def db_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("db_id", "")) for row in rows)


def feature_flags(row: dict[str, Any], base: Any) -> dict[str, bool]:
    s = base.structure(row)
    join_count = int(s["join_count"])
    return {
        "join_bin_0": join_count == 0,
        "join_bin_1": join_count == 1,
        "join_bin_2": join_count == 2,
        "join_bin_3plus": join_count >= 3,
        "where": bool(s["where_any"]),
        "aggregation": bool(s["aggregation"]),
        "group_by": bool(s["group_by"]),
        "having": bool(s["having"]),
        "order_by": bool(s["order_by"]),
        "limit": bool(s["limit"]),
        "distinct": bool(s["distinct"]),
        "subquery": bool(s["subquery"]),
        "set_operation": bool(s["set_operation"]),
    }


def feature_counts(rows: list[dict[str, Any]], base: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for name, value in feature_flags(row, base).items():
            counts[name] += int(value)
    return counts


def apportion_with_mins(
    *,
    total: int,
    weights: Counter[str],
    mins: dict[str, int],
    caps: Counter[str],
) -> dict[str, int]:
    keys = list(weights.keys())
    alloc = {key: min(int(caps.get(key, 0)), int(mins.get(key, 0))) for key in keys}
    remaining = total - sum(alloc.values())
    if remaining < 0:
        raise RuntimeError(f"Minimum quotas exceed total={total}: {sum(alloc.values())}")

    weight_sum = sum(weights.values())
    targets = {key: total * weights[key] / weight_sum for key in keys}
    ordered = sorted(
        keys,
        key=lambda key: (targets[key] - math.floor(targets[key]), weights[key], str(key)),
        reverse=True,
    )
    for key in ordered:
        desired = min(int(caps.get(key, 0)), max(alloc[key], int(math.floor(targets[key]))))
        add = min(remaining, desired - alloc[key])
        if add > 0:
            alloc[key] += add
            remaining -= add

    while remaining > 0:
        candidates = [key for key in keys if alloc[key] < int(caps.get(key, 0))]
        if not candidates:
            raise RuntimeError(f"No DB capacity left while apportioning total={total}")
        key = max(candidates, key=lambda item: (targets[item] - alloc[item], weights[item], -stable_hash(item)))
        alloc[key] += 1
        remaining -= 1
    return alloc


def summarize_db_distribution(
    *,
    pool_counts: Counter[str],
    role_counts: Counter[str],
    target_rows: int,
    frequent_threshold: int = 10,
) -> dict[str, Any]:
    total_pool = sum(pool_counts.values())
    frequent_dbs = sorted(db for db, count in pool_counts.items() if count >= frequent_threshold)
    missing: list[str] = []
    over: list[dict[str, Any]] = []
    under: list[dict[str, Any]] = []
    for db in frequent_dbs:
        expected = pool_counts[db] / total_pool * target_rows
        actual = role_counts.get(db, 0)
        ratio = actual / expected if expected else 0.0
        entry = {
            "db_id": db,
            "pool_count": pool_counts[db],
            "actual_count": actual,
            "expected_count": expected,
            "ratio_vs_expected": ratio,
        }
        if actual == 0:
            missing.append(db)
        if ratio >= 2.0:
            over.append(entry)
        if ratio <= 0.5:
            under.append(entry)
    return {
        "frequent_db_threshold": frequent_threshold,
        "frequent_db_count": len(frequent_dbs),
        "missing_frequent_db_count": len(missing),
        "missing_frequent_dbs": missing,
        "strongly_overrepresented_count": len(over),
        "strongly_underrepresented_count": len(under),
        "strongly_overrepresented": sorted(over, key=lambda item: (-item["ratio_vs_expected"], item["db_id"]))[:25],
        "strongly_underrepresented": sorted(under, key=lambda item: (item["ratio_vs_expected"], item["db_id"]))[:25],
    }


def split_spider_dbstratified(
    spider_rows: list[dict[str, Any]],
    sqlcc_train: list[dict[str, Any]],
    base: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    components = build_overlap_components_by_all_keys(spider_rows, base)
    sqlcc_sets = sets_for(sqlcc_train, base)
    forced_train: list[list[dict[str, Any]]] = []
    eligible: list[list[dict[str, Any]]] = []
    for comp in components:
        if component_conflicts_with_sets(comp, sqlcc_sets, base):
            forced_train.append(comp)
        else:
            eligible.append(comp)

    pool_counts = db_counts(spider_rows)
    frequent_dbs = {db for db, count in pool_counts.items() if count >= 10}
    total_spider = len(spider_rows)
    component_meta: list[dict[str, Any]] = []
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_db_counts: Counter[str] = Counter()
    feature_rate_targets = {
        name: count / len(spider_rows)
        for name, count in feature_counts(spider_rows, base).items()
    }
    for idx, comp in enumerate(eligible):
        meta = {
            "idx": idx,
            "rows": comp,
            "size": len(comp),
            "db_counts": db_counts(comp),
            "feature_counts": feature_counts(comp, base),
            "hash": stable_hash([row.get("id") for row in comp]),
        }
        component_meta.append(meta)
        eligible_db_counts.update(meta["db_counts"])
        for db in meta["db_counts"]:
            by_db[db].append(meta)

    missing_eligible = sorted(db for db in frequent_dbs if eligible_db_counts[db] < 2)
    if missing_eligible:
        raise RuntimeError(
            "Frequent DBs lack the two eligible holdout rows needed for retrieval+validation coverage: "
            + ", ".join(missing_eligible[:20])
        )

    def anti_under_floor(target_rows: int, db: str) -> int:
        expected = pool_counts[db] / total_spider * target_rows
        return max(1, math.floor(0.5 * expected) + 1)

    retrieval_floors = {db: anti_under_floor(RETRIEVAL_SPIDER, db) for db in frequent_dbs}
    validation_floors = {db: anti_under_floor(VAL_SPIDER, db) for db in frequent_dbs}
    retrieval_targets = apportion_with_mins(
        total=RETRIEVAL_SPIDER,
        weights=pool_counts,
        mins=retrieval_floors,
        caps=eligible_db_counts,
    )
    validation_targets = apportion_with_mins(
        total=VAL_SPIDER,
        weights=pool_counts,
        mins=validation_floors,
        caps=eligible_db_counts,
    )

    used: set[int] = set()
    retrieval_components: list[dict[str, Any]] = []
    validation_components: list[dict[str, Any]] = []
    retrieval_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    retrieval_features: Counter[str] = Counter()
    validation_features: Counter[str] = Counter()
    retrieval_rows = 0
    validation_rows = 0

    def add_component(role: str, meta: dict[str, Any]) -> None:
        nonlocal retrieval_rows, validation_rows
        used.add(int(meta["idx"]))
        if role == "retrieval":
            retrieval_components.append(meta)
            retrieval_counts.update(meta["db_counts"])
            retrieval_features.update(meta["feature_counts"])
            retrieval_rows += int(meta["size"])
        elif role == "validation":
            validation_components.append(meta)
            validation_counts.update(meta["db_counts"])
            validation_features.update(meta["feature_counts"])
            validation_rows += int(meta["size"])
        else:
            raise ValueError(role)

    def role_state(role: str) -> tuple[Counter[str], Counter[str], int, int, dict[str, int], Counter[str]]:
        if role == "retrieval":
            return (
                retrieval_counts,
                retrieval_features,
                retrieval_rows,
                RETRIEVAL_SPIDER,
                retrieval_targets,
                validation_counts,
            )
        return (
            validation_counts,
            validation_features,
            validation_rows,
            VAL_SPIDER,
            validation_targets,
            retrieval_counts,
        )

    def pick_for_db(role: str, db: str, *, require_other_role_coverable: bool) -> bool:
        counts, _features, current_rows, target_rows, targets, other_counts = role_state(role)
        candidates: list[tuple[Any, ...]] = []
        for meta in by_db[db]:
            if int(meta["idx"]) in used:
                continue
            if current_rows + int(meta["size"]) > target_rows:
                continue
            if require_other_role_coverable and other_counts[db] == 0:
                if not any(int(other["idx"]) not in used and int(other["idx"]) != int(meta["idx"]) for other in by_db[db]):
                    continue
            over = max(0, counts[db] + meta["db_counts"][db] - targets.get(db, 0))
            exact_penalty = 0 if counts[db] + meta["db_counts"][db] <= targets.get(db, 0) else 1
            newly_covered = sum(1 for candidate_db in meta["db_counts"] if candidate_db in frequent_dbs and counts[candidate_db] == 0)
            candidates.append((over, exact_penalty, int(meta["size"]), -newly_covered, int(meta["hash"]), meta))
        if not candidates:
            return False
        candidates.sort()
        add_component(role, candidates[0][-1])
        return True

    frequent_order = sorted(
        frequent_dbs,
        key=lambda db: (len(by_db[db]), eligible_db_counts[db], pool_counts[db], str(db)),
    )
    for db in frequent_order:
        if retrieval_counts[db] == 0 and not pick_for_db("retrieval", db, require_other_role_coverable=True):
            raise RuntimeError(f"Could not cover frequent DB in retrieval split: {db}")
    for db in frequent_order:
        if validation_counts[db] == 0 and not pick_for_db("validation", db, require_other_role_coverable=False):
            raise RuntimeError(f"Could not cover frequent DB in validation split: {db}")

    coverage_rows = {"retrieval": retrieval_rows, "validation": validation_rows}

    def satisfy_floor(role: str, floors: dict[str, int]) -> None:
        counts, _features, _current_rows, _target_rows, _targets, _other_counts = role_state(role)
        changed = True
        while changed:
            changed = False
            ordered = sorted(
                frequent_dbs,
                key=lambda db: (counts[db] - floors[db], len(by_db[db]), pool_counts[db], str(db)),
            )
            for db in ordered:
                while counts[db] < floors[db]:
                    if not pick_for_db(role, db, require_other_role_coverable=False):
                        break
                    changed = True

    satisfy_floor("retrieval", retrieval_floors)
    satisfy_floor("validation", validation_floors)
    floor_rows = {"retrieval": retrieval_rows, "validation": validation_rows}

    def fill_role(role: str) -> None:
        nonlocal retrieval_rows, validation_rows
        while True:
            counts, features, current_rows, target_rows, targets, _other_counts = role_state(role)
            if current_rows == target_rows:
                return
            remaining = target_rows - current_rows
            candidates = [
                meta
                for meta in component_meta
                if int(meta["idx"]) not in used and int(meta["size"]) <= remaining
            ]
            if not candidates:
                raise RuntimeError(f"Could not fill {role} to exact size; current_rows={current_rows}")

            def score(meta: dict[str, Any]) -> tuple[float, int, int]:
                db_score = 0.0
                for db, count in meta["db_counts"].items():
                    deficit = targets.get(db, 0) - counts[db]
                    if deficit > 0:
                        db_score += 100.0 * min(count, deficit) + 10.0 * deficit * count
                    else:
                        db_score -= 15.0 * count * abs(deficit)
                feature_score = 0.0
                for name, count in meta["feature_counts"].items():
                    desired = feature_rate_targets[name] * target_rows
                    deficit = desired - features[name]
                    if deficit > 0:
                        feature_score += 0.5 * min(count, deficit)
                    else:
                        feature_score -= 0.05 * count
                return (db_score + feature_score, -int(meta["size"]), -int(meta["hash"]))

            add_component(role, max(candidates, key=score))

    fill_role("retrieval")
    fill_role("validation")

    retrieval_selected = [meta["rows"] for meta in retrieval_components]
    validation_selected = [meta["rows"] for meta in validation_components]
    selected_ids = {int(meta["idx"]) for meta in retrieval_components + validation_components}
    train_components = forced_train + [
        comp for idx, comp in enumerate(eligible) if idx not in selected_ids
    ]

    spider_train = base.flatten_components(train_components)
    spider_retrieval = base.flatten_components(retrieval_selected)
    spider_validation = base.flatten_components(validation_selected)

    if len(spider_train) != TRAIN_SPIDER:
        raise RuntimeError(f"Spider train size mismatch: {len(spider_train)} != {TRAIN_SPIDER}")
    if len(spider_retrieval) != RETRIEVAL_SPIDER:
        raise RuntimeError(f"Spider retrieval size mismatch: {len(spider_retrieval)} != {RETRIEVAL_SPIDER}")
    if len(spider_validation) != VAL_SPIDER:
        raise RuntimeError(f"Spider validation size mismatch: {len(spider_validation)} != {VAL_SPIDER}")

    stats = {
        "component_key_policy": ["id", "question", "sql", "question_sql_pair"],
        "component_count": len(components),
        "eligible_component_count": len(eligible),
        "forced_train_component_count": len(forced_train),
        "forced_train_rows_due_to_sqlcc_overlap": sum(len(comp) for comp in forced_train),
        "eligible_rows": sum(len(comp) for comp in eligible),
        "component_size_distribution": dict(sorted(Counter(len(comp) for comp in components).items())),
        "eligible_component_size_distribution": dict(sorted(Counter(len(comp) for comp in eligible).items())),
        "frequent_db_threshold": 10,
        "frequent_db_count": len(frequent_dbs),
        "frequent_dbs_with_at_least_two_eligible_rows": sum(1 for db in frequent_dbs if eligible_db_counts[db] >= 2),
        "coverage_rows_after_minimum_db_pass": coverage_rows,
        "rows_after_anti_underrepresentation_floor": floor_rows,
        "retrieval_db_targets": dict(sorted(retrieval_targets.items())),
        "validation_db_targets": dict(sorted(validation_targets.items())),
        "retrieval_db_floors": dict(sorted(retrieval_floors.items())),
        "validation_db_floors": dict(sorted(validation_floors.items())),
        "retrieval_db_summary": summarize_db_distribution(
            pool_counts=pool_counts,
            role_counts=db_counts(spider_retrieval),
            target_rows=RETRIEVAL_SPIDER,
        ),
        "validation_db_summary": summarize_db_distribution(
            pool_counts=pool_counts,
            role_counts=db_counts(spider_validation),
            target_rows=VAL_SPIDER,
        ),
    }
    return spider_train, spider_retrieval, spider_validation, stats


def load_preserved_extra_sqlcc(
    *,
    root: Path,
    old_sqlcc: list[dict[str, Any]],
    base: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_rows = base.read_jsonl(root / PREVIOUS_OLDSQLCC_RAW_TRAIN)
    old_ids = {str(row.get("id", "")) for row in old_sqlcc}
    extra_rows = [
        dict(row)
        for row in previous_rows
        if row.get("source_dataset") == "sql_create_context" and str(row.get("id", "")) not in old_ids
    ]
    if len(extra_rows) != EXTRA_SQLCC:
        raise RuntimeError(f"Expected {EXTRA_SQLCC} preserved SQLCC fill rows, got {len(extra_rows)}")
    current_ids = [str(row.get("id", "")) for row in extra_rows]
    if len(set(current_ids)) != EXTRA_SQLCC:
        raise RuntimeError("Preserved SQLCC fill rows contain duplicate IDs.")
    stats = {
        "source_path": base.rel(root / PREVIOUS_OLDSQLCC_RAW_TRAIN),
        "extra_sqlcc_rows_reused": len(extra_rows),
        "id_sha256": hashlib.sha256("\n".join(current_ids).encode("utf-8")).hexdigest(),
        "aggregation_rows": sum(bool(base.structure(row)["aggregation"]) for row in extra_rows),
        "simple_rows": sum(not bool(base.structure(row)["aggregation"]) for row in extra_rows),
        "reuse_policy": "byte/id-identical rows copied from existing oldsqlccpreserved raw train before re-enrichment",
    }
    return extra_rows, stats


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
    base = load_module(root / "src/14_prepare_qwen35_2b_clean_split_pipeline.py", "base_clean_split_v3_dbstrat")
    mix_module = load_module(root / "src/04_build_spider_sqlcc_complexity_mix.py", "mix_builder_v3_dbstrat")
    sft_module = load_module(root / "src/02_make_sft_dataset_v1_clean_full_chat.py", "sft_builder_v3_dbstrat")
    prompt_module = load_module(root / "src/prompt_presets.py", "prompt_presets_v3_dbstrat")

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
            SUMMARY_PATH,
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

    extra_sqlcc, extra_stats = load_preserved_extra_sqlcc(root=root, old_sqlcc=old_sqlcc, base=base)
    sqlcc_train = old_sqlcc + extra_sqlcc

    spider_train, spider_retrieval, spider_validation, spider_stats = split_spider_dbstratified(
        spider_rows,
        sqlcc_train,
        base,
    )

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
    extra_ids = {str(row.get("id", "")) for row in extra_sqlcc}
    pool_not_old = [row for row in sqlcc_pool if str(row.get("id", "")) not in old_ids]
    strict_pool_not_old, strict_filter_stats = base.strict_filter_sqlcc_against_spider(pool_not_old, spider_rows)
    val_candidates = no_overlap_filter(
        [row for row in strict_pool_not_old if str(row.get("id", "")) not in extra_ids],
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
        raise RuntimeError("v3 clean split counts do not match requested sizes.")

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
        "builder_script": BUILDER_SCRIPT,
        "variant": "v3_dbstratified_oldsqlccpreserved",
        "correction_reason": (
            "Preserve the previous 18,040 SQLCC rows and 1,260 SQLCC fill rows, "
            "then rebuild only the Spider train/retrieval/validation split with DB-first stratification."
        ),
        "stratification": {
            "spider": [
                "overlap_components_by_id_question_sql_pair",
                "sqlcc_train_conflicting_components_forced_to_train",
                "db_first_proportional_targets",
                "minimum_coverage_for_frequent_dbs_ge_10",
                "anti_underrepresentation_floor",
                "secondary_structure_balance",
            ],
            "sqlcc_train": "all previous 25k-mix SQLCC rows + byte/id-preserved 1,260 oldsqlccpreserved fill rows",
            "sqlcc_validation": [
                "same deterministic SQLCC validation selection policy as oldsqlccpreserved",
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
        "sqlcc_extra_reuse_stats": extra_stats,
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
        "db_distribution_summary": spider_stats["retrieval_db_summary"],
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
        "spider_dbstratification": spider_stats,
        "sqlcc_extra_reuse_stats": extra_stats,
    }
    base.write_json(root / SUMMARY_PATH, summary)
    print(
        json.dumps(
            {
                "status": "prepared",
                "summary_path": base.rel(root / SUMMARY_PATH),
                "split_counts": summary["split_counts"],
                "retrieval_db_summary": spider_stats["retrieval_db_summary"],
                "validation_db_summary": spider_stats["validation_db_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
