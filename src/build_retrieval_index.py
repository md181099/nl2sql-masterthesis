#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval_utils import (
    LeakageGuard,
    load_retrieval_pool,
    question_sql_signature,
    sha256_file,
)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_versions() -> dict[str, str | None]:
    return {
        "sentence-transformers": _package_version("sentence-transformers"),
        "faiss-cpu": _package_version("faiss-cpu"),
        "faiss-gpu": _package_version("faiss-gpu"),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
        "scikit-learn": _package_version("scikit-learn"),
    }


def _resolved_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _leakage_check(
    *,
    examples: list[dict[str, Any]],
    source_path: Path,
    testcases_path: Path,
) -> dict[str, Any]:
    guard = LeakageGuard.from_testcases_path(testcases_path)
    source_resolved = str(source_path.resolve())
    test_resolved = str(testcases_path.resolve()) if testcases_path.exists() else str(testcases_path)
    id_overlaps = []
    question_overlaps = []
    sql_overlaps = []
    pair_overlaps = []
    for example in examples:
        example_id = str(example.get("id", "")).strip()
        if example_id and example_id in guard.testcase_ids:
            id_overlaps.append(example_id)
        pair = question_sql_signature(
            str(example.get("question", "")),
            str(example.get("gold_sql", "")),
        )
        if pair[0] and pair[0] in guard.testcase_questions:
            question_overlaps.append(example_id or str(example.get("source_idx", "")))
        if pair[1] and pair[1] in guard.testcase_sqls:
            sql_overlaps.append(example_id or str(example.get("source_idx", "")))
        if pair in guard.question_sql_pairs:
            pair_overlaps.append(example_id or str(example.get("source_idx", "")))
    source_matches_testcases = source_resolved == test_resolved
    status = (
        "fail"
        if source_matches_testcases or id_overlaps or question_overlaps or sql_overlaps or pair_overlaps
        else "pass"
    )
    return {
        "status": status,
        "testcases_path": str(testcases_path),
        "source_matches_testcases": source_matches_testcases,
        "num_id_overlaps": len(id_overlaps),
        "num_question_overlaps": len(question_overlaps),
        "num_sql_overlaps": len(sql_overlaps),
        "num_question_sql_overlaps": len(pair_overlaps),
        "sample_id_overlaps": id_overlaps[:10],
        "sample_question_overlap_ids": question_overlaps[:10],
        "sample_sql_overlap_ids": sql_overlaps[:10],
        "sample_question_sql_overlap_ids": pair_overlaps[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SentenceTransformer + FAISS retrieval index for NL2SQL few-shot prompts."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Retrieval-pool JSONL source path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for index.faiss, metadata.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--method",
        default="sentence_transformer",
        choices=["sentence_transformer"],
        help="Embedding method.",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model id or local path.",
    )
    parser.add_argument(
        "--index_type",
        default="faiss_ip",
        choices=["faiss_ip"],
        help="FAISS index type. faiss_ip uses IndexFlatIP.",
    )
    parser.add_argument(
        "--normalize",
        type=_parse_bool,
        default=True,
        help="Normalize embeddings before indexing (true/false).",
    )
    parser.add_argument(
        "--query_prefix",
        default="",
        help=(
            "Optional prefix applied to retrieval queries and, when enabled, "
            "to indexed document questions. Useful for BGE instruction models."
        ),
    )
    parser.add_argument(
        "--apply_query_prefix_to_documents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply query_prefix to indexed retrieval-pool questions.",
    )
    parser.add_argument(
        "--apply_query_prefix_to_queries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Record whether eval-time queries should receive query_prefix. "
            "Defaults to apply_query_prefix_to_documents when query_prefix is set."
        ),
    )
    parser.add_argument(
        "--testcases",
        default="data/testcases_spider_dev_full.jsonl",
        help="Testcases JSONL used for leakage checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded in the manifest.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="SentenceTransformer encoding batch size.",
    )
    parser.add_argument(
        "--allow_overlap",
        action="store_true",
        help="Allow leakage-check failures and still build the index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in output_dir if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = _resolved_path(Path(args.source))
    output_dir = _resolved_path(Path(args.output_dir))
    testcases_path = _resolved_path(Path(args.testcases))

    if not source_path.exists():
        raise FileNotFoundError(f"Missing retrieval source JSONL: {source_path}")
    if not testcases_path.exists():
        raise FileNotFoundError(f"Missing leakage-check testcases JSONL: {testcases_path}")

    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only when you intentionally want to replace this index."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "faiss is not installed. Run src/check_embedding_retrieval_env.py first; "
            "install faiss-cpu/faiss-gpu only after confirming the environment."
        ) from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required to build the retrieval index.") from exc

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run src/check_embedding_retrieval_env.py first."
        ) from exc

    examples = load_retrieval_pool(source_path)
    if not examples:
        raise RuntimeError(
            f"No usable retrieval examples found in {source_path}; "
            "each row needs question, gold_sql/sql/query, and schema_prompt/context."
        )

    leakage_check_result = _leakage_check(
        examples=examples,
        source_path=source_path,
        testcases_path=testcases_path,
    )
    if leakage_check_result["status"] != "pass" and not args.allow_overlap:
        raise RuntimeError(
            "Leakage check failed; refusing to build retrieval index. "
            + json.dumps(leakage_check_result, ensure_ascii=False)
        )

    query_prefix = str(args.query_prefix or "")
    apply_query_prefix_to_documents = bool(args.apply_query_prefix_to_documents)
    apply_query_prefix_to_queries = (
        bool(args.apply_query_prefix_to_queries)
        if args.apply_query_prefix_to_queries is not None
        else bool(query_prefix and apply_query_prefix_to_documents)
    )

    def _prefixed_question(value: str) -> str:
        if query_prefix and apply_query_prefix_to_documents and not str(value).startswith(query_prefix):
            return query_prefix + str(value)
        return str(value)

    questions = [_prefixed_question(str(example["question"])) for example in examples]
    print(f"Loading embedding model: {args.embedding_model}")
    try:
        model = SentenceTransformer(str(args.embedding_model))
    except Exception:
        model = SentenceTransformer(str(args.embedding_model), local_files_only=True)
    print(f"Encoding {len(questions)} questions...")
    embeddings = model.encode(
        questions,
        batch_size=int(args.batch_size),
        convert_to_numpy=True,
        normalize_embeddings=bool(args.normalize),
        show_progress_bar=True,
    ).astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    embedding_dim = int(embeddings.shape[1])
    if args.index_type == "faiss_ip":
        index = faiss.IndexFlatIP(embedding_dim)
    else:  # pragma: no cover - argparse choices keep this unreachable
        raise ValueError(f"Unsupported index_type: {args.index_type}")
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    _write_jsonl(metadata_path, examples)
    metadata_sha256 = sha256_file(metadata_path)
    manifest = {
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "method": args.method,
        "embedding_model": str(args.embedding_model),
        "index_type": args.index_type,
        "normalize": bool(args.normalize),
        "query_prefix": query_prefix,
        "apply_query_prefix_to_documents": apply_query_prefix_to_documents,
        "apply_query_prefix_to_queries": apply_query_prefix_to_queries,
        "bge_query_format": {
            "prefix": query_prefix,
            "uses_recommended_prefix": query_prefix
            == "Represent this sentence for searching relevant passages: ",
            "rationale": (
                "Optional instruction prefix for retrieval-oriented embedding models. "
                "When the same prefix is applied to indexed questions and eval-time "
                "queries, question-to-question retrieval stays in the same embedding "
                "distribution."
            ),
        },
        "num_examples": len(examples),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(source_path),
        "metadata_sha256": metadata_sha256,
        "leakage_check_result": leakage_check_result,
        "embedding_dim": embedding_dim,
        "package_versions": _package_versions(),
        "seed": int(args.seed),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Retrieval index built successfully.")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Leakage status: {leakage_check_result['status']}")
    print(f"Embedding dim: {embedding_dim}")


if __name__ == "__main__":
    main()
