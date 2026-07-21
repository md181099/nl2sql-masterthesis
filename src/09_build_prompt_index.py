#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging


logger = logging.getLogger(__name__)


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().split())


def normalize_sql(sql: str) -> str:
    return " ".join(ensure_semicolon(sql).strip().lower().split())


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def overlap_stats(train_items: list[dict], test_items: list[dict]) -> tuple[int, int, int]:
    train_q = {normalize_question(str(x.get("question", ""))) for x in train_items}
    train_s = {normalize_sql(str(x.get("gold_sql", ""))) for x in train_items}
    q_overlap = sum(1 for x in test_items if normalize_question(str(x.get("question", ""))) in train_q)
    s_overlap = sum(1 for x in test_items if normalize_sql(str(x.get("gold_sql", ""))) in train_s)
    both_overlap = sum(
        1
        for x in test_items
        if (
            normalize_question(str(x.get("question", ""))) in train_q
            and normalize_sql(str(x.get("gold_sql", ""))) in train_s
        )
    )
    return q_overlap, s_overlap, both_overlap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prompt-retrieval index from traincases with overlap guard.")
    p.add_argument("--allow_overlap", action="store_true", help="Allow overlap between traincases and testcases.")
    p.add_argument(
        "--traincases_path",
        default="data/traincases.jsonl",
        help="Path to retrieval pool source JSONL (traincases only).",
    )
    p.add_argument(
        "--testcases_path",
        default="data/testcases.jsonl",
        help="Path to testcases JSONL used only for overlap/leakage checks.",
    )
    p.add_argument(
        "--index_json_path",
        default="data/prompt_index.json",
        help="Output path for retrieval index JSON.",
    )
    p.add_argument(
        "--index_emb_path",
        default="data/prompt_index_embeddings.npy",
        help="Output path for retrieval index embeddings (.npy).",
    )
    p.add_argument(
        "--embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return p.parse_args()


def main() -> None:
    """
    Builds a retrieval index for prompt-tuning (few-shot) from traincases only.

    Outputs:
      - data/prompt_index.json           (list of {id, question, gold_sql, db_id, [schema_prompt]})
      - data/prompt_index_embeddings.npy (float32 array shape [N, D])
    """
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(__file__).resolve().parents[1]
    traincases_path = Path(args.traincases_path)
    if not traincases_path.is_absolute():
        traincases_path = project_root / traincases_path
    testcases_path = Path(args.testcases_path)
    if not testcases_path.is_absolute():
        testcases_path = project_root / testcases_path
    index_json_path = Path(args.index_json_path)
    if not index_json_path.is_absolute():
        index_json_path = project_root / index_json_path
    index_emb_path = Path(args.index_emb_path)
    if not index_emb_path.is_absolute():
        index_emb_path = project_root / index_emb_path
    embed_model_name = str(args.embed_model).strip()
    if not embed_model_name:
        raise ValueError("embed_model must be non-empty")

    logger.info("traincases_path=%s", traincases_path)
    logger.info("testcases_path=%s", testcases_path)
    logger.info("index_json_path=%s", index_json_path)
    logger.info("index_emb_path=%s", index_emb_path)
    logger.info("embed_model=%s", embed_model_name)
    logger.info("Index source = traincases only (testcases are used only for overlap checks).")

    if not traincases_path.exists():
        raise FileNotFoundError(f"Missing: {traincases_path}")

    if testcases_path.exists():
        train_rows = load_jsonl(traincases_path)
        test_rows = load_jsonl(testcases_path)
        q_overlap, s_overlap, both_overlap = overlap_stats(train_rows, test_rows)
        if not args.allow_overlap and (q_overlap > 0 or s_overlap > 0):
            raise RuntimeError(
                "Train/Test overlap detected before building prompt index. "
                f"question_overlap={q_overlap}, sql_overlap={s_overlap}, both_overlap={both_overlap}. "
                "Run src/00_prepare_spider_subset.py to generate Spider dataset."
            )

    # Load traincases
    items: list[dict] = []
    with traincases_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Require question + gold_sql for demonstrations
            if "question" not in obj or "gold_sql" not in obj:
                continue
            question = str(obj["question"]).strip()
            gold_sql = str(obj["gold_sql"]).strip()
            if not question or not gold_sql:
                continue

            item = {
                "id": str(obj.get("id") or f"TRAIN_{row_idx:06d}"),
                "question": question,
                "gold_sql": gold_sql,
                "db_id": str(obj.get("db_id", "")).strip(),
            }
            schema_prompt = obj.get("schema_prompt", None)
            if isinstance(schema_prompt, str) and schema_prompt.strip():
                item["schema_prompt"] = schema_prompt
            items.append(item)

    if not items:
        raise RuntimeError("No usable items found in traincases.jsonl (need question + gold_sql).")

    questions = [it["question"] for it in items]

    logger.info("Loading embedding model: %s", embed_model_name)
    model = SentenceTransformer(embed_model_name)

    logger.info("Encoding %d questions...", len(questions))
    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity becomes dot product
        show_progress_bar=True,
    ).astype(np.float32)

    # Save index JSON and embeddings
    index_json_path.parent.mkdir(parents=True, exist_ok=True)
    index_emb_path.parent.mkdir(parents=True, exist_ok=True)
    index_json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(index_emb_path, embeddings)

    logger.info("Prompt-tuning index built")
    logger.info("Indexed train examples: %d", len(items))
    logger.info("Items saved to: %s", index_json_path)
    logger.info("Embeddings saved to: %s", index_emb_path)
    logger.info("Embeddings shape: %s", embeddings.shape)


if __name__ == "__main__":
    main()
