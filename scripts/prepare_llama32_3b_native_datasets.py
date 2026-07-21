#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llama32_native_chat import (  # noqa: E402
    LLAMA32_3B_INSTRUCT_MODEL_ID,
    LLAMA32_3B_INSTRUCT_REVISION,
    LLAMA32_NATIVE_CHAT_FORMAT,
    LLAMA32_NATIVE_TEMPLATE_DATE,
    llama32_native_template_kwargs,
    render_llama32_native_chat,
)


QWEN_TRAIN = PROJECT_ROOT / (
    "data/sql_create_context/"
    "train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_"
    "sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl"
)
QWEN_VALIDATION = PROJECT_ROOT / (
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl"
)
LLAMA_TRAIN = PROJECT_ROOT / (
    "data/sql_create_context/"
    "train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl"
)
LLAMA_VALIDATION = PROJECT_ROOT / (
    "data/sql_create_context/"
    "val_sft_llama32_3b_instruct_full_chat_v2_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl"
)
DATASET_MANIFEST = PROJECT_ROOT / (
    "data/sql_create_context/"
    "llama32_3b_instruct_native_chat_v2_dataset_manifest_20260714.json"
)
SPIDER_DEV = PROJECT_ROOT / "data/testcases_spider_dev_full.jsonl"

CHATML_MESSAGE_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?",
    flags=re.DOTALL,
)
QUESTION_RE = re.compile(r"\nQuestion:\n(.*?)\n\nSQL:\s*$", flags=re.DOTALL)
SCHEMA_RE = re.compile(r"^Database schema:\n(.*?)\n\nRules:\n", flags=re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_sql(value: str) -> str:
    return normalize_text(value.rstrip(";") + ";")


def normalize_schema(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip().lower()


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    remainder = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * remainder


def parse_qwen_row(row: dict[str, Any], line_number: int, path: Path) -> dict[str, Any]:
    example_id = row.get("id")
    text = row.get("text")
    if not isinstance(example_id, str) or not example_id:
        raise RuntimeError(f"Invalid id at {path}:{line_number}")
    if not isinstance(text, str) or not text:
        raise RuntimeError(f"Invalid text at {path}:{line_number}")
    matches = list(CHATML_MESSAGE_RE.finditer(text))
    reconstructed = "".join(match.group(0) for match in matches)
    if reconstructed != text:
        raise RuntimeError(f"Non-canonical ChatML at {path}:{line_number}")
    messages = [
        {"role": match.group(1), "content": match.group(2)}
        for match in matches
    ]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise RuntimeError(f"Unexpected roles at {path}:{line_number}")
    user_text = messages[1]["content"]
    question_match = QUESTION_RE.search(user_text)
    schema_match = SCHEMA_RE.search(user_text)
    if question_match is None or schema_match is None:
        raise RuntimeError(f"Could not parse question/schema at {path}:{line_number}")
    sql = messages[2]["content"]
    if not sql.strip():
        raise RuntimeError(f"Empty assistant SQL at {path}:{line_number}")
    return {
        "id": example_id,
        "messages": messages,
        "question": question_match.group(1),
        "schema": schema_match.group(1),
        "sql": sql,
    }


def read_qwen_semantics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(parse_qwen_row(json.loads(line), line_number, path))
    return rows


def source_name(example_id: str) -> str:
    if example_id.startswith("SPIDER_TRAIN_OTHERS_"):
        return "train_others"
    if example_id.startswith("SPIDER_TRAIN_"):
        return "spider_train"
    if example_id.startswith("SCC_"):
        return "sqlcc"
    return "unknown"


def materialize(
    source_path: Path,
    output_path: Path,
    tokenizer: Any,
    *,
    create: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_qwen_semantics(source_path)
    if create and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_path}")

    lengths: list[int] = []
    rendered_rows: list[dict[str, Any]] = []
    for row in source_rows:
        rendered = render_llama32_native_chat(
            tokenizer,
            row["messages"],
            add_generation_prompt=False,
        )
        direct_encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            date_string=LLAMA32_NATIVE_TEMPLATE_DATE,
        )
        direct_ids = direct_encoded["input_ids"]
        rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if list(direct_ids) != list(rendered_ids):
            raise RuntimeError(f"Native template token roundtrip mismatch: {row['id']}")
        if rendered.count("<|begin_of_text|>") != 1:
            raise RuntimeError(f"BOS count mismatch: {row['id']}")
        if rendered.count("<|eot_id|>") != 3:
            raise RuntimeError(f"EOT count mismatch: {row['id']}")
        if "<|im_start|>" in rendered or "<|im_end|>" in rendered or "<think>" in rendered.lower():
            raise RuntimeError(f"Forbidden token in native Llama rendering: {row['id']}")
        lengths.append(len(direct_ids))
        rendered_rows.append(
            {
                "id": row["id"],
                "messages": row["messages"],
                "chat_template_kwargs": llama32_native_template_kwargs(),
                "text": rendered,
            }
        )

    if create:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            for row in rendered_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:
        if not output_path.is_file():
            raise FileNotFoundError(f"Expected materialized dataset: {output_path}")
        materialized = []
        with output_path.open(encoding="utf-8") as handle:
            materialized = [json.loads(line) for line in handle if line.strip()]
        if materialized != rendered_rows:
            raise RuntimeError(f"Materialized dataset differs from deterministic rendering: {output_path}")

    stats = {
        "rows": len(source_rows),
        "unique_ids": len({row["id"] for row in source_rows}),
        "source_rows": dict(Counter(source_name(row["id"]) for row in source_rows)),
        "token_lengths": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p95": quantile(lengths, 0.95),
            "p99": quantile(lengths, 0.99),
            "exactly_2048": sum(length == 2048 for length in lengths),
            "over_2048": sum(length > 2048 for length in lengths),
            "truncations": 0,
        },
        "duplicate_schema_headers": sum(
            row["messages"][1]["content"].count("Database schema:") != 1
            for row in source_rows
        ),
        "qwen_special_token_rows": sum(
            "<|im_start|>" in row["text"] or "<|im_end|>" in row["text"]
            for row in rendered_rows
        ),
        "think_token_rows": sum("<think" in row["text"].lower() for row in rendered_rows),
    }
    return source_rows, stats


def set_view(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    return {
        "id": {row["id"] for row in rows},
        "question": {normalize_text(row["question"]) for row in rows},
        "sql": {normalize_sql(row["sql"]) for row in rows},
        "pair": {
            (normalize_text(row["question"]), normalize_sql(row["sql"]))
            for row in rows
        },
        "schema_signature": {normalize_schema(row["schema"]) for row in rows},
    }


def read_dev(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            schema = str(row.get("schema_prompt", ""))
            if schema.lower().startswith("database schema:\n"):
                schema = schema.split("\n", 1)[1]
            rows.append(
                {
                    "id": str(row["id"]),
                    "question": str(row["question"]),
                    "sql": str(row["gold_sql"]),
                    "schema": schema,
                }
            )
    return rows


def overlaps(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    left_sets = set_view(left)
    right_sets = set_view(right)
    return {name: len(left_sets[name] & right_sets[name]) for name in left_sets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create additive datasets and manifest")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing deterministic outputs")
    args = parser.parse_args()
    if args.create == args.verify_only:
        raise SystemExit("Choose exactly one of --create or --verify-only")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA32_3B_INSTRUCT_MODEL_ID,
        revision=LLAMA32_3B_INSTRUCT_REVISION,
        local_files_only=True,
    )
    train_rows, train_stats = materialize(
        QWEN_TRAIN,
        LLAMA_TRAIN,
        tokenizer,
        create=args.create,
    )
    validation_rows, validation_stats = materialize(
        QWEN_VALIDATION,
        LLAMA_VALIDATION,
        tokenizer,
        create=args.create,
    )
    dev_rows = read_dev(SPIDER_DEV)

    manifest = {
        "schema_version": 1,
        "purpose": "llama32_3b_native_chat_dataset_equivalence_and_leakage_manifest",
        "model_id": LLAMA32_3B_INSTRUCT_MODEL_ID,
        "model_revision": LLAMA32_3B_INSTRUCT_REVISION,
        "chat_format": LLAMA32_NATIVE_CHAT_FORMAT,
        "native_template_date": LLAMA32_NATIVE_TEMPLATE_DATE,
        "source": {
            "train_path": str(QWEN_TRAIN.relative_to(PROJECT_ROOT)),
            "train_sha256": sha256_file(QWEN_TRAIN),
            "validation_path": str(QWEN_VALIDATION.relative_to(PROJECT_ROOT)),
            "validation_sha256": sha256_file(QWEN_VALIDATION),
        },
        "materialized": {
            "train_path": str(LLAMA_TRAIN.relative_to(PROJECT_ROOT)),
            "train_sha256": sha256_file(LLAMA_TRAIN),
            "validation_path": str(LLAMA_VALIDATION.relative_to(PROJECT_ROOT)),
            "validation_sha256": sha256_file(LLAMA_VALIDATION),
        },
        "equivalence": {
            "same_ids": True,
            "same_order": True,
            "same_questions": True,
            "same_sql": True,
            "same_schema_semantics": True,
            "only_chat_serialization_changed": True,
        },
        "train": train_stats,
        "validation": validation_stats,
        "leakage": {
            "train_vs_validation": overlaps(train_rows, validation_rows),
            "train_vs_spider_dev": overlaps(train_rows, dev_rows),
            "validation_vs_spider_dev": overlaps(validation_rows, dev_rows),
            "interpretation": (
                "ID, normalized-question, and normalized question-SQL-pair overlap are hard leakage checks. "
                "SQL-only and schema-signature overlap are reported diagnostics because generic SQL structures "
                "and schema shapes can legitimately recur across independent examples."
            ),
        },
    }
    if train_stats["token_lengths"]["over_2048"] or validation_stats["token_lengths"]["over_2048"]:
        raise RuntimeError("At least one native Llama training/validation row exceeds 2048 tokens")
    for comparison in manifest["leakage"].values():
        if isinstance(comparison, dict):
            if comparison["id"] or comparison["question"] or comparison["pair"]:
                raise RuntimeError(f"Hard leakage detected: {comparison}")

    if args.create:
        if DATASET_MANIFEST.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {DATASET_MANIFEST}")
        DATASET_MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        existing = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("Dataset manifest differs from fresh verification")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
