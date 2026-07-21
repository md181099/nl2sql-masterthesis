#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare a local, reproducible split from the Hugging Face dataset
`philschmid/sql-create-context-copy`.

This dataset contains schema/context + question + SQL text, but no executable
SQLite database files. Therefore it is suitable for training / text-based
validation / retrieval, while Spider remains the execution-evaluation source.
"""

import argparse
import hashlib
import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = "You are a SQL expert. Return exactly one SQL query, no explanation."
THINK_BLOCK_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")
THINK_TAG_RE = re.compile(r"(?i)</?think\b[^>]*>")
SQL_START_RE = re.compile(r"(?i)\b(?:select|with)\b")


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def canonical_key(context: str, question: str, answer: str) -> str:
    return "|||".join(
        [
            normalize_text(context),
            normalize_text(question),
            normalize_text(answer),
        ]
    )


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_sql_answer(answer: str) -> str:
    cleaned = THINK_BLOCK_RE.sub(" ", answer)
    cleaned = THINK_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    # Keep only the SQL segment starting at the first SELECT/WITH.
    start_match = SQL_START_RE.search(cleaned)
    if start_match is None:
        return ""
    return cleaned[start_match.start() :].strip()


def row_to_output(
    *,
    row_hash: str,
    dataset_name: str,
    context: str,
    question: str,
    answer: str,
    split: str,
    idx: int,
) -> dict[str, Any]:
    item_id = f"SCC_{split.upper()}_{idx:06d}"
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    return {
        "id": item_id,
        "source_dataset": dataset_name,
        "question": question,
        "context": context,
        "schema_prompt": context,
        "gold_sql": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ],
        "split": split,
        "row_hash": row_hash,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare deterministic local train/test JSONL splits for sql-create-context."
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="philschmid/sql-create-context-copy",
        help="Hugging Face dataset id.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/sql_create_context",
        help="Output directory for train.jsonl/test.jsonl/metadata.json.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    p.add_argument("--train_size", type=int, default=50000, help="Number of train examples.")
    p.add_argument("--test_size", type=int, default=15000, help="Number of test examples.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _atomic_write_outputs(
    *,
    train_path: Path,
    test_path: Path,
    metadata_path: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    suffix = f".tmp.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    train_tmp = train_path.with_name(train_path.name + suffix)
    test_tmp = test_path.with_name(test_path.name + suffix)
    metadata_tmp = metadata_path.with_name(metadata_path.name + suffix)
    tmp_paths = [train_tmp, test_tmp, metadata_tmp]
    try:
        _write_jsonl(train_tmp, train_rows)
        _write_jsonl(test_tmp, test_rows)
        _write_json(metadata_tmp, metadata)
        train_tmp.replace(train_path)
        test_tmp.replace(test_path)
        metadata_tmp.replace(metadata_path)
    finally:
        for tmp_path in tmp_paths:
            if tmp_path.exists():
                tmp_path.unlink()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(__file__).resolve().parents[1]

    if args.train_size < 1 or args.test_size < 1:
        raise ValueError("train_size and test_size must both be >= 1")

    out_dir = _resolve_path(project_root, args.output_dir)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    metadata_path = out_dir / "metadata.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in (train_path, test_path, metadata_path) if p.exists()]
    if existing and not args.overwrite:
        existing_str = ", ".join(str(p) for p in existing)
        raise FileExistsError(
            "Output file(s) already exist. Re-run with --overwrite to replace them: "
            f"{existing_str}"
        )

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install project requirements first."
        ) from exc

    logger.info("Loading dataset: %s", args.dataset_name)
    ds = load_dataset(args.dataset_name, split="train")

    total_loaded = len(ds)
    logger.info("Loaded rows: %s", total_loaded)

    deduped: dict[str, dict[str, str]] = {}
    total_valid = 0
    for row in ds:
        question_raw = row.get("question", "")
        context_raw = row.get("context", "")
        answer_raw = row.get("answer", "")

        question = str(question_raw).strip()
        context = str(context_raw).strip()
        answer = sanitize_sql_answer(str(answer_raw))

        if not question or not context or not answer:
            continue

        total_valid += 1
        key = canonical_key(context=context, question=question, answer=answer)
        row_hash = sha256_text(key)
        if row_hash not in deduped:
            deduped[row_hash] = {
                "question": question,
                "context": context,
                "answer": answer,
            }

    total_after_dedupe = len(deduped)
    duplicates_removed = total_valid - total_after_dedupe
    logger.info(
        "After cleaning: total_valid=%s, total_after_dedupe=%s, duplicates_removed=%s",
        total_valid,
        total_after_dedupe,
        duplicates_removed,
    )

    needed = args.train_size + args.test_size
    if total_after_dedupe < needed:
        raise RuntimeError(
            "Not enough deduplicated rows for requested split sizes: "
            f"need={needed}, available={total_after_dedupe}"
        )

    all_hashes = sorted(deduped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_hashes)
    train_hashes = all_hashes[: args.train_size]
    test_hashes = all_hashes[args.train_size : args.train_size + args.test_size]

    overlap_count = len(set(train_hashes) & set(test_hashes))
    if overlap_count != 0:
        raise RuntimeError(
            f"Leakage check failed: overlap_count={overlap_count} between train and test row_hash sets."
        )

    train_rows = [
        row_to_output(
            row_hash=row_hash,
            dataset_name=args.dataset_name,
            context=deduped[row_hash]["context"],
            question=deduped[row_hash]["question"],
            answer=deduped[row_hash]["answer"],
            split="train",
            idx=i,
        )
        for i, row_hash in enumerate(train_hashes, start=1)
    ]
    test_rows = [
        row_to_output(
            row_hash=row_hash,
            dataset_name=args.dataset_name,
            context=deduped[row_hash]["context"],
            question=deduped[row_hash]["question"],
            answer=deduped[row_hash]["answer"],
            split="test",
            idx=i,
        )
        for i, row_hash in enumerate(test_hashes, start=1)
    ]

    dataset_fingerprint = getattr(ds, "_fingerprint", None)
    metadata = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "total_loaded": total_loaded,
        "total_valid": total_valid,
        "total_after_dedupe": total_after_dedupe,
        "duplicates_removed": duplicates_removed,
        "overlap_count": overlap_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_train": str(train_path),
        "output_test": str(test_path),
    }
    if dataset_fingerprint is not None:
        metadata["dataset_fingerprint"] = str(dataset_fingerprint)

    _atomic_write_outputs(
        train_path=train_path,
        test_path=test_path,
        metadata_path=metadata_path,
        train_rows=train_rows,
        test_rows=test_rows,
        metadata=metadata,
    )

    logger.info("Wrote train split: %s (%s rows)", train_path, len(train_rows))
    logger.info("Wrote test split: %s (%s rows)", test_path, len(test_rows))
    logger.info("Wrote metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
