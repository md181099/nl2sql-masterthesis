#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SEED = 42
TRAIN_SPIDER = 5700
RETRIEVAL_SPIDER = 700
VAL_SPIDER = 560
TRAIN_SQLCC = 19300
VAL_SQLCC = 1940
SYSTEM_PROMPT_VARIANT = "sqlctx_anti_overjoin"
CHAT_FORMAT = "qwen_sqlctx_chatml"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RAW_TRAIN = Path(
    "data/sql_create_context/"
    "train_mix_clean_split_qwen35_2b_spider5700_sqlcc19300_"
    "complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_TRAIN = Path(
    "data/sql_create_context/"
    "train_sft_qwen35_2b_clean_split_full_chat_v1_clean_anti_overjoin_"
    "spider5700_sqlcc19300_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
SFT_VAL = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_2b_clean_split_spider560_sqlcc1940_full_chat_v1_clean_"
    "anti_overjoin_no_train_no_retrieval_no_dev_2500_seed42.jsonl"
)
RETRIEVAL_POOL = Path("data/retrieval_pools/clean_split_spider700_no_train_no_val_no_dev_seed42.jsonl")
RETRIEVAL_INDEX = Path("data/retrieval_indexes/clean_split_spider700_no_train_no_val_no_dev_bge_large_en_v15")
STATIC_FEWSHOT = Path("data/fewshot_static/static_fewshot_clean_split_spider700_k1_full_schema_seed42.jsonl")

TRAIN_CONFIG = Path("configs/train_lora_qwen35_2b_base_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_oomsafe.json")
EVAL_ZERO = Path("configs/eval_qwen35_2b_lora_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_zero_shot_full_aliasnames.json")
EVAL_FULL = Path("configs/eval_qwen35_2b_lora_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_maxinput2048_full_aliasnames.json")
EVAL_GATE085 = Path("configs/eval_qwen35_2b_lora_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate085_maxinput2048_full_aliasnames.json")
EVAL_GATE070 = Path("configs/eval_qwen35_2b_lora_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_dynamic_fewshot_bge_large_k1_full_schema_similarity_gate070_maxinput2048_full_aliasnames.json")
EVAL_STATIC = Path("configs/eval_qwen35_2b_lora_clean_split_r8_alpha16_evalval2500_earlystop_maxlen2048_static_fewshot_k1_full_schema_clean_retrieval_maxinput2048_full_aliasnames.json")

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

SQL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|<=|>=|<>|!=|[(),.*=<>;+/-]")
TABLE_REF_RE = re.compile(
    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_.$]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])",
    re.IGNORECASE,
)
AGG_RE = re.compile(r"\b(count|avg|sum|min|max)\s*\(", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qwen 2B clean-split pipeline artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated clean-split files.")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + "_manifest.json")


def rel(path: Path) -> str:
    root = project_root()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def ensure_free(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        full = project_root() / path
        if full.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing clean-split artifact: {path}")
        if full.is_dir() and overwrite:
            raise FileExistsError(f"Refusing to overwrite directory from this builder: {path}")


def normalize_question(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_sql(value: str) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r";+\s*$", "", text)
    return " ".join(text.split())


def clean_ident(value: str) -> str:
    value = value.strip().strip("`\"[]")
    if "." in value:
        value = value.split(".")[-1]
    return value.casefold()


def sql_table_count(sql: str) -> int:
    return len({clean_ident(match.group(1)) for match in TABLE_REF_RE.finditer(sql) if clean_ident(match.group(1))})


def schema_table_count(row: dict[str, Any]) -> int:
    if row.get("schema_table_count") is not None:
        try:
            return int(row["schema_table_count"])
        except Exception:
            pass
    schema = str(row.get("schema_prompt") or row.get("context") or "")
    return len(re.findall(r"(?m)^Table:\s+", schema))


def structure(row: dict[str, Any]) -> dict[str, Any]:
    sql = str(row.get("gold_sql", ""))
    low = sql.casefold()
    features = row.get("sql_features") or {}
    set_op = bool(
        features.get("union")
        or features.get("intersect")
        or features.get("except")
        or features.get("compound")
        or re.search(r"\b(union|intersect|except)\b", low)
    )
    agg_counts = {
        name: len(re.findall(rf"\b{name}\s*\(", low))
        for name in ("count", "avg", "sum", "min", "max")
    }
    return {
        "sql_len_chars": len(sql),
        "sql_len_tokens": len(SQL_TOKEN_RE.findall(sql)),
        "sql_table_count": sql_table_count(sql),
        "schema_table_count": schema_table_count(row),
        "join_count": int(features.get("join_count", len(re.findall(r"\bjoin\b", low)))),
        "join_any": bool(int(features.get("join_count", len(re.findall(r"\bjoin\b", low)))) > 0),
        "where_any": bool(re.search(r"\bwhere\b", low)),
        "aggregation": bool(features.get("any_aggregation") or AGG_RE.search(sql)),
        "group_by": bool(features.get("group_by") or re.search(r"\bgroup\s+by\b", low)),
        "having": bool(features.get("having") or re.search(r"\bhaving\b", low)),
        "order_by": bool(features.get("order_by") or re.search(r"\border\s+by\b", low)),
        "limit": bool(features.get("limit") or re.search(r"\blimit\b", low)),
        "distinct": bool(features.get("distinct") or re.search(r"\bdistinct\b", low)),
        "subquery": bool(features.get("nested_select") or len(re.findall(r"\bselect\b", low)) > 1),
        "set_operation": set_op,
        **{f"{name}_agg": count for name, count in agg_counts.items()},
    }


def bucket(row: dict[str, Any], mix_module: Any) -> str:
    features = row.get("sql_features") or mix_module.sql_features(str(row.get("gold_sql", "")))
    if mix_module.rare_complexity(features):
        return "rare_complexity"
    if features.get("any_aggregation"):
        return "aggregation_only"
    return "simple"


def quantile_bins(values: list[int], cuts: int = 4) -> list[int]:
    values_sorted = sorted(values)
    return [values_sorted[round((len(values_sorted) - 1) * q / cuts)] for q in range(1, cuts)]


def bin_value(value: int, cuts: list[int]) -> int:
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return idx
    return len(cuts)


def join_bin(count: int) -> str:
    if count >= 3:
        return "3plus"
    return str(count)


def component_signature(rows: list[dict[str, Any]], mix_module: Any, len_cuts: list[int], schema_cuts: list[int]) -> tuple[Any, ...]:
    row = rows[0]
    s = structure(row)
    return (
        str(row.get("db_id", "")),
        bucket(row, mix_module),
        join_bin(int(s["join_count"])),
        bool(s["where_any"]),
        bool(s["aggregation"]),
        bool(s["group_by"]),
        bool(s["having"]),
        bool(s["order_by"]),
        bool(s["limit"]),
        bool(s["distinct"]),
        bool(s["subquery"]),
        bool(s["set_operation"]),
        bin_value(int(s["sql_len_chars"]), len_cuts),
        bin_value(int(s["schema_table_count"]), schema_cuts),
        bin_value(int(s["sql_table_count"]), [1, 2, 3]),
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


def build_overlap_components(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    dsu = DSU(len(rows))
    by_question: dict[str, int] = {}
    by_sql: dict[str, int] = {}
    for idx, row in enumerate(rows):
        q = normalize_question(str(row.get("question", "")))
        s = normalize_sql(str(row.get("gold_sql", "")))
        if q in by_question:
            dsu.union(idx, by_question[q])
        else:
            by_question[q] = idx
        if s in by_sql:
            dsu.union(idx, by_sql[s])
        else:
            by_sql[s] = idx
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[dsu.find(idx)].append(row)
    components = list(grouped.values())
    for comp in components:
        comp.sort(key=lambda row: str(row.get("id", "")))
    return components


def apportion(total: int, sizes: dict[Any, int]) -> dict[Any, int]:
    denom = sum(sizes.values())
    raw = {key: sizes[key] / denom * total for key in sizes}
    alloc = {key: min(sizes[key], int(raw[key])) for key in sizes}
    remaining = total - sum(alloc.values())
    order = sorted(sizes, key=lambda key: (raw[key] - int(raw[key]), sizes[key]), reverse=True)
    idx = 0
    while remaining > 0:
        key = order[idx % len(order)]
        if alloc[key] < sizes[key]:
            alloc[key] += 1
            remaining -= 1
        idx += 1
    return alloc


def select_components(
    components: list[list[dict[str, Any]]],
    *,
    target_rows: int,
    mix_module: Any,
    len_cuts: list[int],
    schema_cuts: list[int],
    seed: int,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    by_sig: dict[tuple[Any, ...], list[list[dict[str, Any]]]] = defaultdict(list)
    for comp in components:
        by_sig[component_signature(comp, mix_module, len_cuts, schema_cuts)].append(comp)
    for sig, comps in by_sig.items():
        comps.sort(key=lambda comp: str(comp[0].get("id", "")))
        random.Random(seed + int(sha256_text(str(sig))[:8], 16)).shuffle(comps)

    sig_sizes = {sig: sum(len(comp) for comp in comps) for sig, comps in by_sig.items()}
    sig_targets = apportion(target_rows, sig_sizes)
    selected: list[list[dict[str, Any]]] = []
    selected_ids: set[int] = set()
    selected_rows = 0

    for sig, target in sorted(sig_targets.items(), key=lambda item: str(item[0])):
        for comp in by_sig[sig]:
            if selected_rows + len(comp) <= target_rows and selected_rows + len(comp) <= sum(sig_targets.values()):
                if sum(len(c) for c in selected if component_signature(c, mix_module, len_cuts, schema_cuts) == sig) + len(comp) <= target:
                    selected.append(comp)
                    selected_ids.add(id(comp))
                    selected_rows += len(comp)

    remaining_components = [comp for comp in components if id(comp) not in selected_ids]
    remaining_components.sort(key=lambda comp: (len(comp), str(comp[0].get("id", ""))))
    for comp in remaining_components:
        if selected_rows == target_rows:
            break
        if selected_rows + len(comp) <= target_rows:
            selected.append(comp)
            selected_ids.add(id(comp))
            selected_rows += len(comp)

    if selected_rows != target_rows:
        raise RuntimeError(f"Could not select exact component target: {selected_rows} != {target_rows}")
    rest = [comp for comp in components if id(comp) not in selected_ids]
    return selected, rest


def flatten_components(components: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comp in components:
        rows.extend(comp)
    return rows


def split_spider(rows: list[dict[str, Any]], mix_module: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    components = build_overlap_components(rows)
    len_cuts = quantile_bins([int(structure(row)["sql_len_chars"]) for row in rows])
    schema_cuts = quantile_bins([int(structure(row)["schema_table_count"]) for row in rows])
    retrieval_components, remaining = select_components(
        components,
        target_rows=RETRIEVAL_SPIDER,
        mix_module=mix_module,
        len_cuts=len_cuts,
        schema_cuts=schema_cuts,
        seed=SEED,
    )
    validation_components, train_components = select_components(
        remaining,
        target_rows=VAL_SPIDER,
        mix_module=mix_module,
        len_cuts=len_cuts,
        schema_cuts=schema_cuts,
        seed=SEED + 1,
    )
    train = flatten_components(train_components)
    retrieval = flatten_components(retrieval_components)
    validation = flatten_components(validation_components)
    for role, role_rows in (("train", train), ("retrieval", retrieval), ("validation", validation)):
        for row in role_rows:
            row["clean_split_role"] = role
    stats = {
        "component_count": len(components),
        "component_size_distribution": dict(sorted(Counter(len(comp) for comp in components).items())),
        "len_cuts": len_cuts,
        "schema_table_cuts": schema_cuts,
    }
    return train, retrieval, validation, stats


def strict_filter_sqlcc_against_spider(sqlcc_rows: list[dict[str, Any]], spider_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spider_questions = {normalize_question(str(row.get("question", ""))) for row in spider_rows}
    spider_sqls = {normalize_sql(str(row.get("gold_sql", ""))) for row in spider_rows}
    kept: list[dict[str, Any]] = []
    removed = Counter()
    for row in sqlcc_rows:
        q = normalize_question(str(row.get("question", "")))
        s = normalize_sql(str(row.get("gold_sql", "")))
        q_hit = q in spider_questions
        s_hit = s in spider_sqls
        if q_hit or s_hit:
            removed["question_overlap_with_spider"] += int(q_hit)
            removed["sql_overlap_with_spider"] += int(s_hit)
            removed["question_or_sql_overlap_with_spider"] += 1
            continue
        kept.append(row)
    return kept, {"input": len(sqlcc_rows), "kept": len(kept), "removed": dict(removed)}


def clone_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def select_sqlcc_train(rows: list[dict[str, Any]], mix_module: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, stats = mix_module.select_sqlcc_rows(
        rows=clone_rows(rows),
        target_size=TRAIN_SQLCC,
        aggregation_target_rate=0.58,
        rng=random.Random(SEED),
    )
    for row in selected:
        row["clean_split_role"] = "train"
    return selected, stats


def remove_ids(rows: list[dict[str, Any]], used: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_ids = {str(row.get("id", "")) for row in used}
    return [row for row in rows if str(row.get("id", "")) not in used_ids]


def sqlcc_val_signature(row: dict[str, Any], len_cuts: list[int]) -> tuple[Any, ...]:
    s = structure(row)
    return (
        "aggregation" if s["aggregation"] else "simple",
        bool(s["where_any"]),
        bool(s["count_agg"]),
        bool(s["avg_agg"]),
        bool(s["sum_agg"]),
        bool(s["min_agg"]),
        bool(s["max_agg"]),
        bin_value(int(s["sql_len_chars"]), len_cuts),
    )


def select_sqlcc_validation(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    len_cuts = quantile_bins([int(structure(row)["sql_len_chars"]) for row in rows])
    agg_rows = [row for row in rows if structure(row)["aggregation"]]
    simple_rows = [row for row in rows if not structure(row)["aggregation"]]
    agg_target = round(VAL_SQLCC * 0.40)
    simple_target = VAL_SQLCC - agg_target

    def pick(pool: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
        by_sig: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_sig[sqlcc_val_signature(row, len_cuts)].append(row)
        for sig, items in by_sig.items():
            items.sort(key=lambda row: str(row.get("id", "")))
            random.Random(seed + int(sha256_text(str(sig))[:8], 16)).shuffle(items)
        sizes = {sig: len(items) for sig, items in by_sig.items()}
        targets = apportion(target, sizes)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for sig, sig_target in targets.items():
            for row in by_sig[sig][:sig_target]:
                selected.append(row)
                selected_ids.add(str(row.get("id", "")))
        if len(selected) < target:
            rest = [row for row in pool if str(row.get("id", "")) not in selected_ids]
            rest.sort(key=lambda row: str(row.get("id", "")))
            selected.extend(rest[: target - len(selected)])
        if len(selected) != target:
            raise RuntimeError(f"Could not select SQLCC validation rows: {len(selected)} != {target}")
        return selected

    selected = pick(agg_rows, agg_target, SEED + 2) + pick(simple_rows, simple_target, SEED + 3)
    for row in selected:
        row["clean_split_role"] = "validation"
        row["selection_bucket"] = "sqlcc_clean_validation_aggregation" if structure(row)["aggregation"] else "sqlcc_clean_validation_simple"
    stats = {
        "len_cuts": len_cuts,
        "target_size": VAL_SQLCC,
        "aggregation_target": agg_target,
        "simple_target": simple_target,
        "available_aggregation": len(agg_rows),
        "available_simple": len(simple_rows),
    }
    return selected, stats


def row_sets(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    return {
        "id": {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()},
        "question": {normalize_question(str(row.get("question", ""))) for row in rows},
        "sql": {normalize_sql(str(row.get("gold_sql", ""))) for row in rows},
        "pair": {
            (normalize_question(str(row.get("question", ""))), normalize_sql(str(row.get("gold_sql", ""))))
            for row in rows
        },
    }


def overlap_counts(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, int]:
    sa = row_sets(a)
    sb = row_sets(b)
    return {key: len(sa[key] & sb[key]) for key in ("id", "question", "sql", "pair")}


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source_dataset", "unknown")) for row in rows).items()))


def structure_distribution(rows: list[dict[str, Any]], mix_module: Any) -> dict[str, Any]:
    if not rows:
        return {}
    structs = [structure(row) for row in rows]
    buckets = Counter(bucket(row, mix_module) for row in rows)
    source = Counter(str(row.get("source_dataset", "unknown")) for row in rows)
    return {
        "rows": len(rows),
        "source_counts": dict(sorted(source.items())),
        "bucket_counts": dict(sorted(buckets.items())),
        "sql_length_mean": mean(s["sql_len_chars"] for s in structs),
        "sql_tokens_mean": mean(s["sql_len_tokens"] for s in structs),
        "sql_table_count_mean": mean(s["sql_table_count"] for s in structs),
        "schema_table_count_mean": mean(s["schema_table_count"] for s in structs),
        "join_rate": sum(bool(s["join_any"]) for s in structs) / len(structs),
        "where_rate": sum(bool(s["where_any"]) for s in structs) / len(structs),
        "aggregation_rate": sum(bool(s["aggregation"]) for s in structs) / len(structs),
        "group_by_rate": sum(bool(s["group_by"]) for s in structs) / len(structs),
        "having_rate": sum(bool(s["having"]) for s in structs) / len(structs),
        "order_by_rate": sum(bool(s["order_by"]) for s in structs) / len(structs),
        "limit_rate": sum(bool(s["limit"]) for s in structs) / len(structs),
        "distinct_rate": sum(bool(s["distinct"]) for s in structs) / len(structs),
        "subquery_rate": sum(bool(s["subquery"]) for s in structs) / len(structs),
        "set_operation_rate": sum(bool(s["set_operation"]) for s in structs) / len(structs),
        "agg_function_counts": {
            name: sum(int(s[f"{name}_agg"] > 0) for s in structs)
            for name in ("count", "avg", "sum", "min", "max")
        },
    }


def sft_rows_from_raw(rows: list[dict[str, Any]], sft_module: Any, prompt_module: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    system_prompt, system_prompt_source, resolved_system_prompt_path, _ = prompt_module.resolve_system_prompt(
        project_root=project_root(),
        system_prompt_variant=SYSTEM_PROMPT_VARIANT,
        system_prompt_path=None,
    )
    output_rows: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", f"CLEAN_SPLIT_{idx:06d}"))
        try:
            question = str(row["question"]).strip()
            schema = sft_module.normalize_schema_text(str(row.get("schema_prompt") or row.get("context") or ""))
            completion = sft_module.sanitize_completion(str(row["gold_sql"]))
            user_prompt = sft_module.build_user_prompt(schema, question)
            prompt_prefix = sft_module.build_prompt_prefix(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_format=CHAT_FORMAT,
            )
            errors = sft_module.check_no_prompt_leakage(
                prompt_prefix,
                completion,
                assistant_marker=sft_module.assistant_marker_for_chat_format(CHAT_FORMAT),
            )
            if errors:
                dropped.append({"id": qid, "reason": "; ".join(errors)})
                continue
            output_rows.append(
                {
                    "id": qid,
                    "text": sft_module.build_full_chat_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        completion=completion,
                        chat_format=CHAT_FORMAT,
                    ),
                }
            )
        except Exception as exc:
            dropped.append({"id": qid, "reason": str(exc)})
    manifest = {
        "dataset_format": "full_chat_text",
        "chat_format": CHAT_FORMAT,
        "system_prompt_variant": SYSTEM_PROMPT_VARIANT,
        "system_prompt_source": system_prompt_source,
        "system_prompt_path": resolved_system_prompt_path,
        "system_prompt_sha256": sha256_text(system_prompt),
        "input_rows": len(rows),
        "written_examples": len(output_rows),
        "dropped_examples": len(dropped),
        "dropped_preview": dropped[:20],
        "contains_think": any("<think" in row["text"].casefold() for row in output_rows),
    }
    if dropped:
        raise RuntimeError(f"SFT rendering dropped {len(dropped)} rows: {dropped[:3]}")
    return output_rows, manifest


def enrich_row(row: dict[str, Any], *, role: str, order: int) -> dict[str, Any]:
    out = dict(row)
    out["clean_split_role"] = role
    out["clean_split_seed"] = SEED
    out["clean_split_order"] = order
    out["row_hash"] = "sha256:" + sha256_text(
        json.dumps(
            {
                "id": out.get("id", ""),
                "question": out.get("question", ""),
                "gold_sql": out.get("gold_sql", ""),
                "schema_prompt": out.get("schema_prompt") or out.get("context") or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return out


def retrieval_row(row: dict[str, Any], order: int) -> dict[str, Any]:
    out = enrich_row(row, role="retrieval", order=order)
    out["embedding_text"] = str(out.get("question", ""))
    out["source_split"] = "clean_split_retrieval"
    if out.get("schema_prompt") and not out.get("spider_schema"):
        out["spider_schema"] = out["schema_prompt"]
    return out


def select_static_fewshot(retrieval_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    lengths = [structure(row)["sql_len_chars"] for row in retrieval_rows]
    median_len = sorted(lengths)[len(lengths) // 2]
    for row in retrieval_rows:
        s = structure(row)
        if not s["join_any"] or not s["aggregation"]:
            continue
        score = 0.0
        score += abs(int(s["join_count"]) - 1) * 3.0
        score += 0.0 if s["group_by"] else 1.5
        score += 0.5 if s["order_by"] else 0.0
        score += 0.5 if s["limit"] else 0.0
        score += 2.0 if s["having"] else 0.0
        score += 2.0 if s["subquery"] else 0.0
        score += 2.0 if s["set_operation"] else 0.0
        score += 1.5 if s["distinct"] else 0.0
        score += abs(int(s["sql_len_chars"]) - median_len) / 100.0
        candidates.append((score, str(row.get("id", "")), row))
    if not candidates:
        raise RuntimeError("No clean retrieval candidate with JOIN and aggregation found for static few-shot.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = dict(candidates[0][2])
    selected["selection_method"] = (
        "deterministic_clean_split_retrieval_join_aggregation_medium_complexity_lowest_score"
    )
    s = structure(selected)
    manifest = {
        "selection_random": False,
        "selection_seed": SEED,
        "source": rel(RETRIEVAL_POOL),
        "num_candidates_join_and_aggregation": len(candidates),
        "selected_score": candidates[0][0],
        "id": selected.get("id"),
        "db_id": selected.get("db_id"),
        "question": selected.get("question"),
        "gold_sql": selected.get("gold_sql"),
        "sql_constructs": {
            "join_count": s["join_count"],
            "aggregation": s["aggregation"],
            "group_by": s["group_by"],
            "having": s["having"],
            "order_by": s["order_by"],
            "limit": s["limit"],
            "distinct": s["distinct"],
            "nested_select": s["subquery"],
            "set_operation": s["set_operation"],
        },
    }
    return selected, manifest


def write_configs() -> dict[str, Any]:
    root = project_root()
    train_ref = json.loads((root / TRAIN_CONFIG_REF).read_text(encoding="utf-8"))
    train_cfg = dict(train_ref)
    train_cfg["dataset_path"] = rel(root / SFT_TRAIN)
    train_cfg["eval_dataset_path"] = rel(root / SFT_VAL)
    train_cfg["output_dir"] = "adapters/qwen35_2b_base/lora_clean_split_qwen35_2b_r8_alpha16_evalval2500_earlystop_maxlen2048_oomsafe"
    train_cfg["num_train_epochs"] = 10
    train_cfg["early_stopping"]["early_stopping_threshold"] = 0.01
    train_cfg["lora"]["r"] = 8
    train_cfg["lora"]["lora_alpha"] = 16
    write_json(root / TRAIN_CONFIG, train_cfg)

    adapter = "lora_clean_split_qwen35_2b_r8_alpha16_evalval2500_earlystop_maxlen2048_oomsafe"
    base_eval = {
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
        "traincases_path": rel(root / RAW_TRAIN),
        "max_test_samples": None,
        "progress_log_every": 25,
    }
    zero = dict(base_eval)
    zero.update({"prompt_tuning": "none", "k": 0, "max_input_tokens": 1536})
    write_json(root / EVAL_ZERO, zero)

    dyn = dict(base_eval)
    dyn.update(
        {
            "prompt_tuning": "dynamic_fewshot",
            "k": 1,
            "max_input_tokens": 2048,
            "retrieval_pool_path": rel(root / RETRIEVAL_INDEX / "metadata.jsonl"),
            "retrieval_index_path": rel(root / RETRIEVAL_INDEX),
            "retrieval_method": "sentence_transformer_faiss",
            "fewshot_example_schema_mode": "full",
            "fewshot_example_mode": "schema_with_rules",
            "embedding_model": EMBEDDING_MODEL,
        }
    )
    full = dict(dyn)
    full.update({"fewshot_gate_enabled": False})
    write_json(root / EVAL_FULL, full)
    gate085 = dict(dyn)
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
    write_json(root / EVAL_GATE085, gate085)
    gate070 = dict(gate085)
    gate070["fewshot_gate_threshold"] = 0.70
    write_json(root / EVAL_GATE070, gate070)

    static_cfg = dict(base_eval)
    static_cfg.update(
        {
            "prompt_tuning": "static_fewshot",
            "k": 1,
            "max_input_tokens": 2048,
            "retrieval_pool_path": rel(root / STATIC_FEWSHOT),
            "retrieval_method": "static_seeded",
            "fewshot_example_schema_mode": "full",
            "fewshot_example_mode": "schema_with_rules",
            "fewshot_gate_enabled": False,
        }
    )
    write_json(root / EVAL_STATIC, static_cfg)
    return {
        "train_config": rel(root / TRAIN_CONFIG),
        "eval_configs": [rel(root / path) for path in (EVAL_ZERO, EVAL_FULL, EVAL_GATE085, EVAL_GATE070, EVAL_STATIC)],
        "output_adapter": train_cfg["output_dir"],
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    ensure_free(
        [
            RAW_TRAIN,
            manifest_path(RAW_TRAIN),
            SFT_TRAIN,
            manifest_path(SFT_TRAIN),
            SFT_VAL,
            manifest_path(SFT_VAL),
            RETRIEVAL_POOL,
            manifest_path(RETRIEVAL_POOL),
            STATIC_FEWSHOT,
            manifest_path(STATIC_FEWSHOT),
            TRAIN_CONFIG,
            EVAL_ZERO,
            EVAL_FULL,
            EVAL_GATE085,
            EVAL_GATE070,
            EVAL_STATIC,
        ],
        overwrite=args.overwrite,
    )
    adapter_output = root / "adapters/qwen35_2b_base/lora_clean_split_qwen35_2b_r8_alpha16_evalval2500_earlystop_maxlen2048_oomsafe"
    if adapter_output.exists():
        raise FileExistsError(f"Output adapter path already exists: {adapter_output}")

    mix_module = load_module(root / "src/04_build_spider_sqlcc_complexity_mix.py", "mix_builder_clean_split")
    sft_module = load_module(root / "src/02_make_sft_dataset_v1_clean_full_chat.py", "sft_builder_clean_split")
    prompt_module = load_module(root / "src/prompt_presets.py", "prompt_presets_clean_split")

    current_rows = read_jsonl(root / CURRENT_MIX)
    spider_rows = [dict(row) for row in current_rows if row.get("source_dataset") == "spider_train"]
    if len(spider_rows) != 6960:
        raise RuntimeError(f"Expected 6960 Spider rows, got {len(spider_rows)}")
    spider_train, spider_retrieval, spider_validation, spider_stats = split_spider(spider_rows, mix_module)

    dev_q, dev_s, _dev_pair = mix_module.load_dev_overlap_sets(root / SPIDER_DEV)
    all_spider_q = {normalize_question(str(row.get("question", ""))) for row in spider_rows}
    all_spider_s = {normalize_sql(str(row.get("gold_sql", ""))) for row in spider_rows}
    all_spider_pair = {
        (normalize_question(str(row.get("question", ""))), normalize_sql(str(row.get("gold_sql", ""))))
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
    strict_sqlcc_pool, strict_filter_stats = strict_filter_sqlcc_against_spider(sqlcc_pool, spider_rows)
    sqlcc_train, sqlcc_train_stats = select_sqlcc_train(strict_sqlcc_pool, mix_module)
    sqlcc_leftover = remove_ids(strict_sqlcc_pool, sqlcc_train)
    sqlcc_validation, sqlcc_val_stats = select_sqlcc_validation(sqlcc_leftover)

    train_rows = [enrich_row(row, role="train", order=i) for i, row in enumerate(spider_train + sqlcc_train)]
    validation_raw = [enrich_row(row, role="validation", order=i) for i, row in enumerate(spider_validation + sqlcc_validation)]
    retrieval_rows = [retrieval_row(row, order=i) for i, row in enumerate(spider_retrieval)]

    if len(train_rows) != 25000 or len(retrieval_rows) != 700 or len(validation_raw) != 2500:
        raise RuntimeError("Clean split counts do not match requested sizes.")

    dev_rows = read_jsonl(root / SPIDER_DEV)
    overlap_matrix = {
        "train_vs_retrieval": overlap_counts(train_rows, retrieval_rows),
        "train_vs_validation": overlap_counts(train_rows, validation_raw),
        "retrieval_vs_validation": overlap_counts(retrieval_rows, validation_raw),
        "train_vs_spider_dev": overlap_counts(train_rows, dev_rows),
        "retrieval_vs_spider_dev": overlap_counts(retrieval_rows, dev_rows),
        "validation_vs_spider_dev": overlap_counts(validation_raw, dev_rows),
    }
    if any(any(value != 0 for value in counts.values()) for counts in overlap_matrix.values()):
        raise RuntimeError("Overlap matrix is not clean: " + json.dumps(overlap_matrix, sort_keys=True))

    train_sft, train_sft_manifest = sft_rows_from_raw(train_rows, sft_module, prompt_module)
    val_sft, val_sft_manifest = sft_rows_from_raw(validation_raw, sft_module, prompt_module)

    write_jsonl(root / RAW_TRAIN, train_rows)
    write_jsonl(root / SFT_TRAIN, train_sft)
    write_jsonl(root / SFT_VAL, val_sft)
    write_jsonl(root / RETRIEVAL_POOL, retrieval_rows)

    static_row, static_manifest = select_static_fewshot(retrieval_rows)
    write_jsonl(root / STATIC_FEWSHOT, [static_row])

    created_at = datetime.now(timezone.utc).isoformat()
    common = {
        "created_at": created_at,
        "seed": SEED,
        "builder_script": "src/14_prepare_qwen35_2b_clean_split_pipeline.py",
        "stratification": {
            "spider": [
                "overlap_components_by_question_and_sql",
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
            "sqlcc_train": "rare -> aggregation-fill -> simple-fill",
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
        "path": rel(root / RAW_TRAIN),
        "sha256": sha256_file(root / RAW_TRAIN),
        "counts": {
            "rows": len(train_rows),
            "spider_train": len(spider_train),
            "sql_create_context": len(sqlcc_train),
        },
        "source_counts": source_counts(train_rows),
        "structure_distribution": structure_distribution(train_rows, mix_module),
        "spider_component_stats": spider_stats,
        "sqlcc_pool_stats": sqlcc_pool_stats,
        "strict_sqlcc_filter_stats": strict_filter_stats,
        "sqlcc_train_selection_stats": sqlcc_train_stats,
    }
    write_json(root / manifest_path(RAW_TRAIN), raw_manifest)

    train_sft_manifest.update(
        {
            **common,
            "path": rel(root / SFT_TRAIN),
            "raw_train_path": rel(root / RAW_TRAIN),
            "sha256": sha256_file(root / SFT_TRAIN),
            "raw_train_sha256": sha256_file(root / RAW_TRAIN),
            "counts": raw_manifest["counts"],
        }
    )
    write_json(root / manifest_path(SFT_TRAIN), train_sft_manifest)

    val_manifest = {
        **common,
        **val_sft_manifest,
        "path": rel(root / SFT_VAL),
        "sha256": sha256_file(root / SFT_VAL),
        "counts": {
            "rows": len(validation_raw),
            "spider_train": len(spider_validation),
            "sql_create_context": len(sqlcc_validation),
        },
        "source_counts": source_counts(validation_raw),
        "structure_distribution": structure_distribution(validation_raw, mix_module),
        "sqlcc_validation_selection_stats": sqlcc_val_stats,
    }
    write_json(root / manifest_path(SFT_VAL), val_manifest)

    retrieval_manifest = {
        **common,
        "path": rel(root / RETRIEVAL_POOL),
        "sha256": sha256_file(root / RETRIEVAL_POOL),
        "counts": {"rows": len(retrieval_rows), "spider_train": len(retrieval_rows), "sql_create_context": 0},
        "source_counts": source_counts(retrieval_rows),
        "structure_distribution": structure_distribution(retrieval_rows, mix_module),
        "embedding_model_for_index": EMBEDDING_MODEL,
        "bge_query_prefix": BGE_QUERY_PREFIX,
    }
    write_json(root / manifest_path(RETRIEVAL_POOL), retrieval_manifest)

    static_manifest.update(
        {
            **common,
            "path": rel(root / STATIC_FEWSHOT),
            "sha256": sha256_file(root / STATIC_FEWSHOT),
            "resource_path": rel(root / STATIC_FEWSHOT),
        }
    )
    write_json(root / manifest_path(STATIC_FEWSHOT), static_manifest)

    config_summary = write_configs()

    summary_path = root / "results/analyses/clean_split_pipeline_preparation_summary.json"
    summary = {
        **common,
        "generated_files": [
            rel(root / RAW_TRAIN),
            rel(root / manifest_path(RAW_TRAIN)),
            rel(root / SFT_TRAIN),
            rel(root / manifest_path(SFT_TRAIN)),
            rel(root / SFT_VAL),
            rel(root / manifest_path(SFT_VAL)),
            rel(root / RETRIEVAL_POOL),
            rel(root / manifest_path(RETRIEVAL_POOL)),
            rel(root / STATIC_FEWSHOT),
            rel(root / manifest_path(STATIC_FEWSHOT)),
            config_summary["train_config"],
            *config_summary["eval_configs"],
        ],
        "config_summary": config_summary,
        "split_counts": {
            "train": len(train_rows),
            "train_spider": len(spider_train),
            "train_sqlcc": len(sqlcc_train),
            "retrieval": len(retrieval_rows),
            "validation": len(validation_raw),
            "validation_spider": len(spider_validation),
            "validation_sqlcc": len(sqlcc_validation),
        },
        "sha256": {
            "raw_train": sha256_file(root / RAW_TRAIN),
            "sft_train": sha256_file(root / SFT_TRAIN),
            "sft_validation": sha256_file(root / SFT_VAL),
            "retrieval_pool": sha256_file(root / RETRIEVAL_POOL),
            "static_fewshot": sha256_file(root / STATIC_FEWSHOT),
        },
        "structure": {
            "train": raw_manifest["structure_distribution"],
            "validation": val_manifest["structure_distribution"],
            "retrieval": retrieval_manifest["structure_distribution"],
        },
    }
    write_json(summary_path, summary)
    print(json.dumps({"status": "prepared", "summary_path": rel(summary_path), "split_counts": summary["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
