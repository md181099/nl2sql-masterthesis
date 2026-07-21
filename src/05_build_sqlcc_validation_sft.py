#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QWEN_CHAT_FORMAT = "qwen_sqlctx_chatml"
SYSTEM_PROMPT_VARIANT = "sqlctx_anti_overjoin"
THINK_RE = re.compile(r"(?i)<\s*/?\s*think\b")
QWEN_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed SQL-Create-Context-only Qwen full-chat validation "
            "dataset with no overlap against the 25k train mix, Spider Train, or Spider Dev."
        )
    )
    parser.add_argument("--sqlcc_path", default="data/sql_create_context/train.jsonl")
    parser.add_argument(
        "--train_mix_path",
        default=(
            "data/sql_create_context/"
            "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
            "25k_seed42_no_dev_overlap.jsonl"
        ),
    )
    parser.add_argument("--spider_train_path", default="data/spider/spider_data/train_spider.json")
    parser.add_argument("--spider_dev_path", default="data/testcases_spider_dev_full.jsonl")
    parser.add_argument(
        "--output_path",
        default=(
            "data/sql_create_context/"
            "val_sft_qwen35_full_chat_v1_clean_anti_overjoin_sqlcc_only_no_spider_"
            "no_train_overlap_2500_seed42.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest_path",
        default=(
            "data/sql_create_context/"
            "val_sft_qwen35_full_chat_v1_clean_anti_overjoin_sqlcc_only_no_spider_"
            "no_train_overlap_2500_seed42_manifest.json"
        ),
    )
    parser.add_argument("--target_size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_module(project_root: Path, relative_path: str, module_name: str) -> Any:
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array: {path}")
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_overlap_sets(
    rows: list[dict[str, Any]],
    *,
    normalize_question,
    normalize_sql,
    sql_key: str,
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    questions: set[str] = set()
    sqls: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        question = str(row.get("question", "")).strip()
        sql = str(row.get(sql_key, "") or row.get("query", "") or row.get("answer", "")).strip()
        if not question or not sql:
            continue
        q_norm = normalize_question(question)
        s_norm = normalize_sql(sql)
        questions.add(q_norm)
        sqls.add(s_norm)
        pairs.add((q_norm, s_norm))
    return questions, sqls, pairs


def overlap_counts(
    rows: list[dict[str, Any]],
    reference: tuple[set[str], set[str], set[tuple[str, str]]],
    *,
    normalize_question,
    normalize_sql,
) -> dict[str, int]:
    ref_questions, ref_sqls, ref_pairs = reference
    q_overlap = 0
    s_overlap = 0
    pair_overlap = 0
    for row in rows:
        q_norm = normalize_question(str(row.get("question", "")))
        s_norm = normalize_sql(str(row.get("gold_sql", "")))
        q_overlap += q_norm in ref_questions
        s_overlap += s_norm in ref_sqls
        pair_overlap += (q_norm, s_norm) in ref_pairs
    return {
        "question_overlap": q_overlap,
        "sql_overlap": s_overlap,
        "question_sql_pair_overlap": pair_overlap,
    }


def validate_qwen_chatml(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = str(row.get("text", ""))
        row_errors: list[str] = []
        if not text.startswith("<|im_start|>system\n"):
            row_errors.append("missing system start")
        if "<|im_start|>user\n" not in text:
            row_errors.append("missing user start")
        if text.count(QWEN_ASSISTANT_MARKER) != 1:
            row_errors.append("assistant marker count is not 1")
        if not text.endswith("<|im_end|>\n"):
            row_errors.append("missing final im_end")
        if "<|start_header_id|>" in text or "<|eot_id|>" in text:
            row_errors.append("contains non-Qwen chat token")
        if THINK_RE.search(text):
            row_errors.append("contains think tag")
        if row_errors:
            errors.append({"index": index, "id": row.get("id"), "errors": row_errors})
    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors_preview": errors[:20],
    }


def duplicate_counts(rows: list[dict[str, Any]], *, normalize_question, normalize_sql) -> dict[str, int]:
    ids = [str(row.get("id", "")) for row in rows]
    questions = [normalize_question(str(row.get("question", ""))) for row in rows]
    sqls = [normalize_sql(str(row.get("gold_sql", ""))) for row in rows]
    pairs = list(zip(questions, sqls))
    return {
        "duplicate_id": len(ids) - len(set(ids)),
        "duplicate_question": len(questions) - len(set(questions)),
        "duplicate_sql": len(sqls) - len(set(sqls)),
        "duplicate_question_sql": len(pairs) - len(set(pairs)),
    }


def main() -> None:
    args = parse_args()
    if args.target_size < 1:
        raise ValueError("target_size must be >= 1")

    project_root = Path(__file__).resolve().parents[1]
    mix_builder = load_module(project_root, "src/04_build_spider_sqlcc_complexity_mix.py", "mix_builder")
    sft_builder = load_module(project_root, "src/02_make_sft_dataset_v1_clean_full_chat.py", "sft_builder")

    sqlcc_path = resolve_path(project_root, args.sqlcc_path)
    train_mix_path = resolve_path(project_root, args.train_mix_path)
    spider_train_path = resolve_path(project_root, args.spider_train_path)
    spider_dev_path = resolve_path(project_root, args.spider_dev_path)
    output_path = resolve_path(project_root, args.output_path)
    manifest_path = resolve_path(project_root, args.manifest_path)

    for input_path in (sqlcc_path, train_mix_path, spider_train_path, spider_dev_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
    for path in (output_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    train_mix_rows = load_jsonl(train_mix_path)
    spider_train_rows = load_json_array(spider_train_path)
    spider_dev_rows = load_jsonl(spider_dev_path)
    normalize_question = mix_builder.normalize_question
    normalize_sql = mix_builder.normalize_sql
    train_sets = build_overlap_sets(
        train_mix_rows,
        normalize_question=normalize_question,
        normalize_sql=normalize_sql,
        sql_key="gold_sql",
    )
    spider_train_sets = build_overlap_sets(
        spider_train_rows,
        normalize_question=normalize_question,
        normalize_sql=normalize_sql,
        sql_key="query",
    )
    spider_dev_sets = build_overlap_sets(
        spider_dev_rows,
        normalize_question=normalize_question,
        normalize_sql=normalize_sql,
        sql_key="gold_sql",
    )

    system_prompt, system_prompt_source, resolved_system_prompt_path, _prompt_hash = (
        sft_builder.resolve_system_prompt(
            project_root=project_root,
            system_prompt_variant=SYSTEM_PROMPT_VARIANT,
            system_prompt_path=None,
        )
    )
    prompt_template_hash = sft_builder.sha256_text(
        json.dumps(
            {
                "system_prompt": system_prompt,
                "user_prompt_template": sft_builder.USER_PROMPT_TEMPLATE,
                "chat_format": QWEN_CHAT_FORMAT,
                "assistant_end": sft_builder.assistant_end_for_chat_format(QWEN_CHAT_FORMAT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    removed = Counter()
    seen_questions: set[str] = set()
    seen_sqls: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    render_rows_by_id: dict[str, dict[str, Any]] = {}
    conversion_examples: list[dict[str, str]] = []

    with sqlcc_path.open("r", encoding="utf-8") as handle:
        for source_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            sql_raw = str(item.get("gold_sql") or item.get("answer") or "").strip()
            context = str(item.get("schema_prompt") or item.get("context") or "").strip()
            row_id = str(item.get("id") or f"SCC_TRAIN_{source_idx + 1:06d}")
            if not question or not sql_raw or not context:
                removed["missing_required_field"] += 1
                continue

            q_norm = normalize_question(question)
            s_norm = normalize_sql(sql_raw)
            pair = (q_norm, s_norm)

            if q_norm in train_sets[0]:
                removed["train_question_overlap"] += 1
            if s_norm in train_sets[1]:
                removed["train_sql_overlap"] += 1
            if pair in train_sets[2]:
                removed["train_question_sql_pair_overlap"] += 1
            if q_norm in train_sets[0] or s_norm in train_sets[1] or pair in train_sets[2]:
                removed["train_any_overlap"] += 1
                continue

            if q_norm in spider_train_sets[0]:
                removed["spider_train_question_overlap"] += 1
            if s_norm in spider_train_sets[1]:
                removed["spider_train_sql_overlap"] += 1
            if pair in spider_train_sets[2]:
                removed["spider_train_question_sql_pair_overlap"] += 1
            if q_norm in spider_train_sets[0] or s_norm in spider_train_sets[1] or pair in spider_train_sets[2]:
                removed["spider_train_any_overlap"] += 1
                continue

            if q_norm in spider_dev_sets[0]:
                removed["spider_dev_question_overlap"] += 1
            if s_norm in spider_dev_sets[1]:
                removed["spider_dev_sql_overlap"] += 1
            if pair in spider_dev_sets[2]:
                removed["spider_dev_question_sql_pair_overlap"] += 1
            if q_norm in spider_dev_sets[0] or s_norm in spider_dev_sets[1] or pair in spider_dev_sets[2]:
                removed["spider_dev_any_overlap"] += 1
                continue

            if q_norm in seen_questions:
                removed["duplicate_question"] += 1
                continue
            if s_norm in seen_sqls:
                removed["duplicate_sql"] += 1
                continue
            if pair in seen_pairs:
                removed["duplicate_question_sql_pair"] += 1
                continue

            try:
                schema_prompt, table_count, column_count = mix_builder.convert_create_context_to_spider_schema(context)
                completion = sft_builder.sanitize_completion(sql_raw)
                user_prompt = sft_builder.build_user_prompt(schema_prompt, question)
                prompt_prefix = sft_builder.build_prompt_prefix(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    chat_format=QWEN_CHAT_FORMAT,
                )
                leakage_errors = sft_builder.check_no_prompt_leakage(
                    prompt_prefix,
                    completion,
                    assistant_marker=sft_builder.assistant_marker_for_chat_format(QWEN_CHAT_FORMAT),
                )
                if leakage_errors:
                    removed["prompt_leakage"] += 1
                    continue
                text = sft_builder.build_full_chat_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    completion=completion,
                    chat_format=QWEN_CHAT_FORMAT,
                )
            except Exception:
                removed["nonrenderable_or_nonparseable"] += 1
                continue

            seen_questions.add(q_norm)
            seen_sqls.add(s_norm)
            seen_pairs.add(pair)
            features = mix_builder.sql_features(completion)
            raw_row = {
                "id": row_id,
                "source_dataset": "sql_create_context",
                "split": str(item.get("split") or "train"),
                "source_path": "data/sql_create_context/train.jsonl",
                "source_idx": source_idx,
                "question": question,
                "context": schema_prompt,
                "schema_prompt": schema_prompt,
                "gold_sql": completion,
                "schema_format": "spider_schema_harmonized_table_columns_empty_pk_fk",
                "schema_table_count": table_count,
                "schema_column_count": column_count,
                "sql_features": features,
                "selection_bucket": "sqlcc_validation_seed42",
                "source_row_sha256": mix_builder.sha256_text(
                    json.dumps(
                        {"question": question, "context": context, "gold_sql": sql_raw},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            }
            candidates.append(raw_row)
            render_rows_by_id[row_id] = {"id": row_id, "text": text}
            if len(conversion_examples) < 10:
                conversion_examples.append(
                    {
                        "id": row_id,
                        "original": context,
                        "converted": schema_prompt,
                    }
                )

    if len(candidates) < args.target_size:
        raise RuntimeError(
            f"Not enough SQLCC validation candidates after filters: {len(candidates)} < {args.target_size}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected_raw = candidates[: args.target_size]
    selected_sft = [render_rows_by_id[str(row["id"])] for row in selected_raw]

    qwen_validation = validate_qwen_chatml(selected_sft)
    overlap_validation = {
        "train_25k_mix": overlap_counts(
            selected_raw,
            train_sets,
            normalize_question=normalize_question,
            normalize_sql=normalize_sql,
        ),
        "spider_train": overlap_counts(
            selected_raw,
            spider_train_sets,
            normalize_question=normalize_question,
            normalize_sql=normalize_sql,
        ),
        "spider_dev": overlap_counts(
            selected_raw,
            spider_dev_sets,
            normalize_question=normalize_question,
            normalize_sql=normalize_sql,
        ),
    }
    duplicates = duplicate_counts(
        selected_raw,
        normalize_question=normalize_question,
        normalize_sql=normalize_sql,
    )
    validation = {
        "target_size_ok": len(selected_sft) == args.target_size,
        "jsonl_schema": {
            "fields": ["id", "text"],
            "all_rows_have_id_and_text": all(
                set(row.keys()) == {"id", "text"} and row["id"] and row["text"]
                for row in selected_sft
            ),
        },
        "qwen_chatml": qwen_validation,
        "no_think": all("<think" not in row["text"].casefold() for row in selected_sft),
        "overlap": overlap_validation,
        "duplicates": duplicates,
    }
    validation["all_passed"] = (
        validation["target_size_ok"]
        and validation["jsonl_schema"]["all_rows_have_id_and_text"]
        and validation["qwen_chatml"]["passed"]
        and validation["no_think"]
        and all(
            counts["question_overlap"] == 0
            and counts["sql_overlap"] == 0
            and counts["question_sql_pair_overlap"] == 0
            for counts in overlap_validation.values()
        )
        and duplicates["duplicate_id"] == 0
        and duplicates["duplicate_question"] == 0
        and duplicates["duplicate_sql"] == 0
        and duplicates["duplicate_question_sql"] == 0
    )
    if not validation["all_passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    write_jsonl(output_path, selected_sft)
    output_sha256 = sft_builder.sha256_file(output_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/05_build_sqlcc_validation_sft.py",
        "pipeline_version": "v1_clean_full_chat_validation",
        "seed": args.seed,
        "target_size": args.target_size,
        "actual_size": len(selected_sft),
        "source_policy": "SQL-Create-Context only; strict question/sql/pair exclusion against 25k train mix, Spider Train, and Spider Dev.",
        "source_paths": {
            "sql_create_context": str(sqlcc_path.relative_to(project_root)),
            "train_25k_mix": str(train_mix_path.relative_to(project_root)),
            "spider_train": str(spider_train_path.relative_to(project_root)),
            "spider_dev": str(spider_dev_path.relative_to(project_root)),
        },
        "output_path": str(output_path.relative_to(project_root)),
        "dataset_path": str(output_path.relative_to(project_root)),
        "dataset_format": "full_chat_text",
        "chat_format": QWEN_CHAT_FORMAT,
        "system_prompt_source": system_prompt_source,
        "system_prompt_variant": SYSTEM_PROMPT_VARIANT,
        "system_prompt_path": resolved_system_prompt_path,
        "prompt_template_sha256": prompt_template_hash,
        "system_prompt_sha256": sft_builder.sha256_text(system_prompt),
        "candidate_counts": {
            "available_after_filters": len(candidates),
            "selected": len(selected_sft),
            "removed": dict(sorted(removed.items())),
        },
        "selected_source_idx_min": min(int(row["source_idx"]) for row in selected_raw),
        "selected_source_idx_max": max(int(row["source_idx"]) for row in selected_raw),
        "selected_source_idx_preview": [int(row["source_idx"]) for row in selected_raw[:20]],
        "selected_distribution": mix_builder.dataset_distribution(selected_raw),
        "validation": validation,
        "conversion_examples": conversion_examples,
        "input_sha256": {
            "sql_create_context": sft_builder.sha256_file(sqlcc_path),
            "train_25k_mix": sft_builder.sha256_file(train_mix_path),
            "spider_train": sft_builder.sha256_file(spider_train_path),
            "spider_dev": sft_builder.sha256_file(spider_dev_path),
        },
        "output_sha256": output_sha256,
        "sha256": output_sha256,
        "num_examples": len(selected_sft),
    }
    write_json(manifest_path, manifest)

    print(f"Wrote validation SFT dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps({"sha256": output_sha256, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
