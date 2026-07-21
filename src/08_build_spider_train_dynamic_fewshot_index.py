#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGG_RE = re.compile(r"\b(count|avg|sum|min|max)\s*\(", re.IGNORECASE)
JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)
NESTED_RE = re.compile(r"\(\s*select\b", re.IGNORECASE | re.DOTALL)
SET_OP_RE = re.compile(r"\b(union|intersect|except)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Spider-Train-only SentenceTransformer + FAISS retrieval index "
            "for dynamic NL2SQL few-shot prompting."
        )
    )
    parser.add_argument("--spider_train_path", default="data/spider/spider_data/train_spider.json")
    parser.add_argument("--spider_dir", default="data/spider/spider_data")
    parser.add_argument("--dev_reference_path", default="data/testcases_spider_dev_full.jsonl")
    parser.add_argument(
        "--output_dir",
        default="data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15",
    )
    parser.add_argument(
        "--audit_path",
        default="results/audits/audit_dynamic_fewshot_bge_large_final_readiness_20260625.md",
    )
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-large-en-v1.5",
    )
    parser.add_argument(
        "--query_prefix",
        default="Represent this sentence for searching relevant passages: ",
        help=(
            "Prefix applied to retrieval queries. For question-to-question BGE retrieval "
            "this is also applied to indexed Spider-Train questions by default."
        ),
    )
    parser.add_argument(
        "--apply_query_prefix_to_documents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply query_prefix to indexed Spider-Train questions as well as Dev queries.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit_k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_semicolon(sql: str) -> str:
    sql = str(sql).strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def normalize_question(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_sql(value: str) -> str:
    value = ensure_semicolon(str(value)).strip().casefold()
    return " ".join(value.split())


def question_sql_pair(question: str, sql: str) -> tuple[str, str]:
    return normalize_question(question), normalize_sql(sql)


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    return {
        "sentence-transformers": package_version("sentence-transformers"),
        "faiss-cpu": package_version("faiss-cpu"),
        "faiss-gpu": package_version("faiss-gpu"),
        "numpy": package_version("numpy"),
        "torch": package_version("torch"),
        "scikit-learn": package_version("scikit-learn"),
    }


def hf_cache_probe(model_name: str) -> dict[str, Any]:
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
    except Exception as exc:
        return {
            "status": "unknown",
            "detail": f"huggingface_hub cache probe unavailable: {repr(exc)}",
        }

    checked_files = ["config.json", "modules.json", "sentence_bert_config.json"]
    cached_files: dict[str, str] = {}
    missing_files: list[str] = []
    for filename in checked_files:
        try:
            cached = try_to_load_from_cache(model_name, filename)
        except Exception as exc:
            missing_files.append(f"{filename}:probe_error:{repr(exc)}")
            continue
        if isinstance(cached, str):
            cached_files[filename] = cached
        else:
            missing_files.append(filename)
    return {
        "status": "cached" if cached_files else "not_cached_or_partial",
        "cached_files": cached_files,
        "missing_or_uncached_files": missing_files,
    }


def load_spider_schema_builder(project_root: Path):
    module_path = project_root / "src" / "00_prepare_spider_subset.py"
    spec = importlib.util.spec_from_file_location("prepare_spider_subset_for_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Spider schema builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_schema_prompt = getattr(module, "build_schema_prompt", None)
    if build_schema_prompt is None:
        raise RuntimeError("00_prepare_spider_subset.py has no build_schema_prompt")
    return build_schema_prompt


def build_retrieval_examples(
    *,
    spider_train_path: Path,
    spider_dir: Path,
    dev_reference_path: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = load_json_array(spider_train_path)
    dev_rows = load_jsonl(dev_reference_path)
    build_schema_prompt = load_spider_schema_builder(project_root)

    dev_questions = {normalize_question(row.get("question", "")) for row in dev_rows}
    dev_sqls = {
        normalize_sql(row.get("gold_sql") or row.get("sql") or row.get("query") or "")
        for row in dev_rows
    }
    dev_pairs = {
        question_sql_pair(
            str(row.get("question", "")),
            str(row.get("gold_sql") or row.get("sql") or row.get("query") or ""),
        )
        for row in dev_rows
    }

    examples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    schema_cache: dict[str, str] = {}
    stats = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    overlap_examples: list[dict[str, Any]] = []

    for source_idx, row in enumerate(train_rows):
        question = str(row.get("question", "")).strip()
        sql = ensure_semicolon(str(row.get("query", "")).strip())
        db_id = str(row.get("db_id", "")).strip()
        if not question or not sql or not db_id:
            stats["invalid_missing_required"] += 1
            continue

        q_norm, s_norm = question_sql_pair(question, sql)
        pair = (q_norm, s_norm)
        has_question_overlap = q_norm in dev_questions
        has_sql_overlap = s_norm in dev_sqls
        has_pair_overlap = pair in dev_pairs
        if has_question_overlap or has_sql_overlap or has_pair_overlap:
            stats["removed_dev_question_or_sql_overlap"] += 1
            stats["removed_dev_question_overlap"] += int(has_question_overlap)
            stats["removed_dev_sql_overlap"] += int(has_sql_overlap)
            stats["removed_dev_pair_overlap"] += int(has_pair_overlap)
            if len(overlap_examples) < 10:
                overlap_examples.append(
                    {
                        "source_idx": source_idx,
                        "db_id": db_id,
                        "question": question,
                        "sql": sql,
                        "question_overlap": has_question_overlap,
                        "sql_overlap": has_sql_overlap,
                        "pair_overlap": has_pair_overlap,
                    }
                )
            continue

        if pair in seen_pairs:
            stats["removed_duplicate_question_sql"] += 1
            if len(duplicate_examples) < 10:
                duplicate_examples.append(
                    {
                        "source_idx": source_idx,
                        "db_id": db_id,
                        "question": question,
                        "sql": sql,
                    }
                )
            continue
        seen_pairs.add(pair)

        db_abs = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        if db_id not in schema_cache:
            schema_cache[db_id] = build_schema_prompt(db_abs)
        db_rel = Path("data") / "spider" / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
        examples.append(
            {
                "id": f"SPIDER_TRAIN_{source_idx:06d}",
                "source_dataset": "spider_train",
                "source_split": "train_spider",
                "source_path": str(spider_train_path.relative_to(project_root))
                if spider_train_path.is_relative_to(project_root)
                else str(spider_train_path),
                "source_idx": source_idx,
                "db_id": db_id,
                "db_path": str(db_rel).replace("\\", "/"),
                "question": question,
                "gold_sql": sql,
                "schema_prompt": schema_cache[db_id],
                "spider_schema": schema_cache[db_id],
                "embedding_text": question,
                "schema_format": "spider_table_columns_pk_fk_from_sqlite",
            }
        )

    stats["input_rows"] = len(train_rows)
    stats["dev_reference_rows"] = len(dev_rows)
    stats["written_rows"] = len(examples)
    return examples, {
        "counts": dict(stats),
        "sample_removed_overlap_examples": overlap_examples,
        "sample_removed_duplicate_examples": duplicate_examples,
    }


def sql_features(sql: str) -> dict[str, Any]:
    text = str(sql)
    low = text.lower()
    join_count = len(JOIN_RE.findall(text))
    return {
        "join_count": join_count,
        "join_bucket": "0" if join_count == 0 else ("1" if join_count == 1 else "2+"),
        "aggregation": bool(AGG_RE.search(text)),
        "group_by": "group by" in low,
        "having": "having" in low,
        "order_by": "order by" in low,
        "limit": bool(re.search(r"\blimit\b", low)),
        "distinct": bool(re.search(r"\bdistinct\b", low)),
        "nested_select": bool(NESTED_RE.search(text)),
        "set_operation": bool(SET_OP_RE.search(text)),
    }


def structure_label(features: dict[str, Any]) -> str:
    labels = [f"joins={features['join_bucket']}"]
    for key in [
        "aggregation",
        "group_by",
        "having",
        "order_by",
        "limit",
        "distinct",
        "nested_select",
        "set_operation",
    ]:
        if features.get(key):
            labels.append(key)
    return ", ".join(labels)


def shorten(value: str, max_len: int = 220) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return float(values_sorted[0])
    pos = (len(values_sorted) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values_sorted) - 1)
    frac = pos - lower
    return float(values_sorted[lower] * (1 - frac) + values_sorted[upper] * frac)


def score_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)) if values else None,
        "min": float(min(values)) if values else None,
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)) if values else None,
    }


def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    try:
        return SentenceTransformer(model_name)
    except Exception:
        return SentenceTransformer(model_name, local_files_only=True)


def prefixed_text(value: str, prefix: str, enabled: bool) -> str:
    value = str(value)
    if enabled and prefix and not value.startswith(prefix):
        return prefix + value
    return value


def make_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    spider_train_path = resolve_path(project_root, args.spider_train_path)
    spider_dir = resolve_path(project_root, args.spider_dir)
    dev_reference_path = resolve_path(project_root, args.dev_reference_path)
    output_dir = resolve_path(project_root, args.output_dir)
    audit_path = resolve_path(project_root, args.audit_path)

    if not spider_train_path.exists():
        raise FileNotFoundError(f"Missing Spider train JSON: {spider_train_path}")
    if not spider_dir.exists():
        raise FileNotFoundError(f"Missing Spider directory: {spider_dir}")
    if not dev_reference_path.exists():
        raise FileNotFoundError(f"Missing dev reference JSONL: {dev_reference_path}")

    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    topk_audit_path = output_dir / f"dev_top{int(args.audit_k)}_retrieval_audit.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only when replacing the retrieval index intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss  # type: ignore
    import numpy as np

    examples, build_stats = build_retrieval_examples(
        spider_train_path=spider_train_path,
        spider_dir=spider_dir,
        dev_reference_path=dev_reference_path,
        project_root=project_root,
    )
    if len(examples) != 6960:
        raise RuntimeError(f"Expected 6960 Spider-Train retrieval examples, got {len(examples)}")

    cache_before = hf_cache_probe(str(args.embedding_model))
    model = load_sentence_transformer(str(args.embedding_model))
    cache_after = hf_cache_probe(str(args.embedding_model))
    query_prefix = str(args.query_prefix)
    apply_query_prefix_to_documents = bool(args.apply_query_prefix_to_documents)
    questions = [
        prefixed_text(str(example["question"]), query_prefix, apply_query_prefix_to_documents)
        for example in examples
    ]
    embeddings = model.encode(
        questions,
        batch_size=int(args.batch_size),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    write_jsonl(metadata_path, examples)

    dev_rows = load_jsonl(dev_reference_path)
    dev_questions = [
        prefixed_text(str(row.get("question", "")), query_prefix, True)
        for row in dev_rows
    ]
    dev_embeddings = model.encode(
        dev_questions,
        batch_size=int(args.batch_size),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    audit_k = max(1, int(args.audit_k))
    dev_scores_np, dev_indices_np = index.search(dev_embeddings, audit_k)

    dev_question_norms = [normalize_question(row.get("question", "")) for row in dev_rows]
    dev_sql_norms = [
        normalize_sql(row.get("gold_sql") or row.get("sql") or row.get("query") or "")
        for row in dev_rows
    ]
    dev_pair_norms = [
        question_sql_pair(
            str(row.get("question", "")),
            str(row.get("gold_sql") or row.get("sql") or row.get("query") or ""),
        )
        for row in dev_rows
    ]

    leakage = Counter()
    top1_scores: list[float] = []
    top3_scores: list[float] = []
    all_scores: list[float] = []
    top1_structure = Counter()
    top3_structure = Counter()
    trace_rows: list[dict[str, Any]] = []

    for dev_idx, dev in enumerate(dev_rows):
        gold_sql = str(dev.get("gold_sql") or dev.get("sql") or dev.get("query") or "")
        dev_features = sql_features(gold_sql)
        retrieved: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(
            zip(dev_indices_np[dev_idx].tolist(), dev_scores_np[dev_idx].tolist()),
            start=1,
        ):
            if idx < 0:
                continue
            example = examples[int(idx)]
            ex_q_norm, ex_s_norm = question_sql_pair(example["question"], example["gold_sql"])
            ex_pair = (ex_q_norm, ex_s_norm)
            leakage["same_id"] += int(str(example.get("id")) == str(dev.get("id")))
            leakage["same_question"] += int(ex_q_norm == dev_question_norms[dev_idx])
            leakage["same_sql"] += int(ex_s_norm == dev_sql_norms[dev_idx])
            leakage["same_question_sql_pair"] += int(ex_pair == dev_pair_norms[dev_idx])

            ex_features = sql_features(str(example.get("gold_sql", "")))
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": example["id"],
                    "db_id": example["db_id"],
                    "question": example["question"],
                    "gold_sql": example["gold_sql"],
                    "features": ex_features,
                }
            )
            all_scores.append(float(score))
            if rank == 1:
                top1_scores.append(float(score))
                top1_structure["same_join_bucket"] += int(ex_features["join_bucket"] == dev_features["join_bucket"])
                top1_structure["same_aggregation"] += int(ex_features["aggregation"] == dev_features["aggregation"])
                top1_structure["same_group_by"] += int(ex_features["group_by"] == dev_features["group_by"])
                top1_structure["same_having"] += int(ex_features["having"] == dev_features["having"])
                top1_structure["same_order_by"] += int(ex_features["order_by"] == dev_features["order_by"])
                top1_structure["same_nested_select"] += int(ex_features["nested_select"] == dev_features["nested_select"])
            if rank <= 3:
                top3_scores.append(float(score))

        first_three = retrieved[:3]
        top3_structure["contains_same_join_bucket"] += int(
            any(item["features"]["join_bucket"] == dev_features["join_bucket"] for item in first_three)
        )
        top3_structure["contains_same_aggregation"] += int(
            any(item["features"]["aggregation"] == dev_features["aggregation"] for item in first_three)
        )
        top3_structure["contains_same_group_by"] += int(
            any(item["features"]["group_by"] == dev_features["group_by"] for item in first_three)
        )
        top3_structure["contains_same_having"] += int(
            any(item["features"]["having"] == dev_features["having"] for item in first_three)
        )
        top3_structure["contains_same_order_by"] += int(
            any(item["features"]["order_by"] == dev_features["order_by"] for item in first_three)
        )
        top3_structure["contains_same_nested_select"] += int(
            any(item["features"]["nested_select"] == dev_features["nested_select"] for item in first_three)
        )

        trace_rows.append(
            {
                "id": dev.get("id"),
                "db_id": dev.get("db_id"),
                "question": dev.get("question"),
                "gold_sql": gold_sql,
                "gold_features": dev_features,
                "retrieved": retrieved,
            }
        )

    write_jsonl(topk_audit_path, trace_rows)

    self_k = min(6, len(examples))
    self_scores_np, self_indices_np = index.search(embeddings, self_k)
    near_duplicate_counts = Counter()
    near_duplicate_samples: list[dict[str, Any]] = []
    for row_idx, (idxs, scores) in enumerate(zip(self_indices_np.tolist(), self_scores_np.tolist())):
        best_other: tuple[int, float] | None = None
        for idx, score in zip(idxs, scores):
            if int(idx) != row_idx:
                best_other = (int(idx), float(score))
                break
        if best_other is None:
            continue
        other_idx, score = best_other
        near_duplicate_counts[">=0.95"] += int(score >= 0.95)
        near_duplicate_counts[">=0.90"] += int(score >= 0.90)
        near_duplicate_counts[">=0.85"] += int(score >= 0.85)
        if score >= 0.90 and len(near_duplicate_samples) < 20:
            near_duplicate_samples.append(
                {
                    "score": score,
                    "id_a": examples[row_idx]["id"],
                    "question_a": examples[row_idx]["question"],
                    "sql_a": examples[row_idx]["gold_sql"],
                    "id_b": examples[other_idx]["id"],
                    "question_b": examples[other_idx]["question"],
                    "sql_b": examples[other_idx]["gold_sql"],
                }
            )

    rng = random.Random(int(args.seed))
    sample_indices = sorted(rng.sample(range(len(trace_rows)), min(int(args.sample_size), len(trace_rows))))
    sample_rows = [trace_rows[i] for i in sample_indices]

    total_dev = len(dev_rows)
    pct = lambda value: round((value / total_dev * 100.0), 2) if total_dev else 0.0
    retrieval_quality = {
        "dev_rows": total_dev,
        "audit_k": audit_k,
        "top1_similarity": score_summary(top1_scores),
        "top3_similarity": score_summary(top3_scores),
        "all_retrieved_similarity": score_summary(all_scores),
        "top1_structure_match_rates": {
            key: {"count": int(value), "percent": pct(int(value))}
            for key, value in sorted(top1_structure.items())
        },
        "top3_structure_coverage_rates": {
            key: {"count": int(value), "percent": pct(int(value))}
            for key, value in sorted(top3_structure.items())
        },
    }

    leakage_audit = {
        "retrieval_source_only_spider_train": all(
            example.get("source_dataset") == "spider_train" for example in examples
        ),
        "retrieval_example_count": len(examples),
        "dev_reference_count": len(dev_rows),
        "index_contains_spider_dev_ids": sum(
            1 for example in examples if str(example.get("id", "")).startswith("SPIDER_DEV_")
        ),
        "dev_retrieval_same_id_hits": int(leakage["same_id"]),
        "dev_retrieval_same_question_hits": int(leakage["same_question"]),
        "dev_retrieval_same_sql_hits": int(leakage["same_sql"]),
        "dev_retrieval_same_question_sql_pair_hits": int(leakage["same_question_sql_pair"]),
        "near_duplicate_train_questions": dict(near_duplicate_counts),
        "near_duplicate_train_question_samples": near_duplicate_samples,
        "status": "PASS"
        if all(
            [
                all(example.get("source_dataset") == "spider_train" for example in examples),
                len(examples) == 6960,
                int(leakage["same_id"]) == 0,
                int(leakage["same_question"]) == 0,
                int(leakage["same_sql"]) == 0,
                int(leakage["same_question_sql_pair"]) == 0,
            ]
        )
        else "FAIL",
    }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/08_build_spider_train_dynamic_fewshot_index.py",
        "method": "sentence_transformer_faiss",
        "embedding_model": str(args.embedding_model),
        "embedding_text": "question",
        "query_prefix": query_prefix,
        "apply_query_prefix_to_queries": True,
        "apply_query_prefix_to_documents": apply_query_prefix_to_documents,
        "bge_query_format": {
            "uses_recommended_prefix": bool(query_prefix.strip()),
            "prefix": query_prefix,
            "rationale": (
                "BAAI/bge-large-en-v1.5 recommends an instruction prefix for retrieval queries. "
                "Because this pipeline performs question-to-question retrieval, the same prefix "
                "is applied to indexed Spider-Train questions and Dev queries to avoid a "
                "query/document embedding distribution mismatch."
            ),
        },
        "normalize_embeddings": True,
        "normalize": True,
        "index_type": "faiss.IndexFlatIP",
        "cosine_similarity": True,
        "embedding_dim": int(embeddings.shape[1]),
        "model_cache_status": {
            "before_load": cache_before,
            "after_load": cache_after,
            "download_status": (
                "already_cached"
                if cache_before.get("status") == "cached"
                else "downloaded_or_cache_populated_during_load"
                if cache_after.get("status") == "cached"
                else "not_confirmed"
            ),
        },
        "seed": int(args.seed),
        "source_paths": {
            "spider_train_path": str(spider_train_path),
            "spider_dir": str(spider_dir),
            "dev_reference_path": str(dev_reference_path),
        },
        "output_paths": {
            "index": str(index_path),
            "metadata": str(metadata_path),
            "manifest": str(manifest_path),
            "topk_audit_jsonl": str(topk_audit_path),
            "audit_markdown": str(audit_path),
        },
        "build_stats": build_stats,
        "leakage_audit": leakage_audit,
        "retrieval_quality": retrieval_quality,
        "package_versions": package_versions(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sample_sections: list[str] = []
    for sample_idx, row in enumerate(sample_rows, start=1):
        lines = [
            f"### Sample {sample_idx}: `{row['id']}` / `{row['db_id']}`",
            f"Question: {row['question']}",
            f"Gold SQL: `{shorten(row['gold_sql'])}`",
            f"Gold structure: {structure_label(row['gold_features'])}",
            "",
            make_markdown_table(
                ["Rank", "Score", "ID", "DB", "Retrieved Question", "Retrieved SQL", "Structure"],
                [
                    [
                        item["rank"],
                        f"{item['score']:.4f}",
                        f"`{item['id']}`",
                        f"`{item['db_id']}`",
                        shorten(item["question"], 120),
                        f"`{shorten(item['gold_sql'], 160)}`",
                        structure_label(item["features"]),
                    ]
                    for item in row["retrieved"][:3]
                ],
            ),
        ]
        sample_sections.append("\n".join(lines))

    quality_rows = [
        ["Top-1 mean", f"{retrieval_quality['top1_similarity']['mean']:.4f}"],
        ["Top-1 p50", f"{retrieval_quality['top1_similarity']['p50']:.4f}"],
        ["Top-1 p90", f"{retrieval_quality['top1_similarity']['p90']:.4f}"],
        ["Top-3 mean", f"{retrieval_quality['top3_similarity']['mean']:.4f}"],
        ["All top-k mean", f"{retrieval_quality['all_retrieved_similarity']['mean']:.4f}"],
    ]
    structure_rows = []
    for key, payload in retrieval_quality["top1_structure_match_rates"].items():
        structure_rows.append([f"Top-1 {key}", payload["count"], f"{payload['percent']:.2f}%"])
    for key, payload in retrieval_quality["top3_structure_coverage_rates"].items():
        structure_rows.append([f"Top-3 {key}", payload["count"], f"{payload['percent']:.2f}%"])

    audit_status = leakage_audit["status"]
    audit_md = f"""# Dynamic Few-Shot Spider-Train Retrieval Audit

Created: {manifest['created_at']}

## Gesamtstatus

{audit_status}

## Artefakte

- Builder: `src/08_build_spider_train_dynamic_fewshot_index.py`
- FAISS index: `{index_path.relative_to(project_root)}`
- Metadata: `{metadata_path.relative_to(project_root)}`
- Manifest: `{manifest_path.relative_to(project_root)}`
- Dev retrieval audit JSONL: `{topk_audit_path.relative_to(project_root)}`
- Audit report: `{audit_path.relative_to(project_root)}`

## Datenbasis

- Retrieval corpus: Spider Train only
- Raw Spider train rows: {build_stats['counts']['input_rows']}
- Removed for Spider-Dev question/SQL overlap: {build_stats['counts']['removed_dev_question_or_sql_overlap']}
- Removed duplicate Question+SQL pairs: {build_stats['counts']['removed_duplicate_question_sql']}
- Final retrieval examples: {len(examples)}
- Embedding model: `{args.embedding_model}`
- Embedding dimension: {int(embeddings.shape[1])}
- Embeddings normalized: yes
- FAISS metric: Inner Product over L2-normalized embeddings, equivalent to cosine similarity
- Query prefix: `{query_prefix}`
- Prefix applied to Spider-Train documents: {apply_query_prefix_to_documents}
- Model cache status: {manifest['model_cache_status']['download_status']}
- SQL Create Context used: no
- Spider Dev used in index: no

## Leakage Audit

{make_markdown_table(
    ["Check", "Result"],
    [
        ["Retrieval source only Spider Train", leakage_audit["retrieval_source_only_spider_train"]],
        ["Index contains Spider Dev IDs", leakage_audit["index_contains_spider_dev_ids"]],
        ["Same ID hits during Dev retrieval", leakage_audit["dev_retrieval_same_id_hits"]],
        ["Same question hits during Dev retrieval", leakage_audit["dev_retrieval_same_question_hits"]],
        ["Same SQL hits during Dev retrieval", leakage_audit["dev_retrieval_same_sql_hits"]],
        ["Same Question+SQL hits during Dev retrieval", leakage_audit["dev_retrieval_same_question_sql_pair_hits"]],
        ["Train near-duplicate question pairs >=0.95", leakage_audit["near_duplicate_train_questions"].get(">=0.95", 0)],
        ["Train near-duplicate question pairs >=0.90", leakage_audit["near_duplicate_train_questions"].get(">=0.90", 0)],
        ["Train near-duplicate question pairs >=0.85", leakage_audit["near_duplicate_train_questions"].get(">=0.85", 0)],
    ],
)}

Duplikate werden so behandelt: exakte normalisierte Question+SQL-Duplikate werden vor dem Indexbau entfernt. Nahezu identische Train-Fragen bleiben im Index, weil sie keine Dev-Leakage sind; sie werden im Manifest dokumentiert und koennen bei Bedarf spaeter per Similarity-Threshold gefiltert werden.

## Retrieval Quality

{make_markdown_table(["Metric", "Value"], quality_rows)}

{make_markdown_table(["Structure Metric", "Count", "Percent of Dev"], structure_rows)}

Qualitative Bewertung: Die Retrieval-Pipeline sucht semantisch ueber die Frage und verwendet normalisierte Embeddings mit Inner Product als Cosine Similarity. Strukturtreue ist nicht garantiert, wird aber ueber die Top-1/Top-3-Strukturmetriken transparent gemacht. Fuer eine Masterarbeits-Ablation ist das wissenschaftlich sauber, weil der Index nur aus Spider Train besteht und alle Leakage-Checks vor der Evaluation getrennt dokumentiert werden.

## 20 Random Dev Retrieval Examples

{chr(10).join(sample_sections)}
"""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(audit_md, encoding="utf-8")

    print("Spider-Train dynamic few-shot retrieval index built.")
    print(f"Examples: {len(examples)}")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Top-k audit JSONL: {topk_audit_path}")
    print(f"Audit: {audit_path}")
    print(f"Leakage status: {audit_status}")


if __name__ == "__main__":
    main()
