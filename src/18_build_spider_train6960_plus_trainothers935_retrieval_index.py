#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SEED = 42
EXPECTED_BASE_ROWS = 6960
EXPECTED_ADDED_ROWS = 935
EXPECTED_TOTAL_ROWS = 7895
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BUILDER_VERSION = "spider_train6960_plus_trainothers935_retrieval_v1"

BASE_INDEX = Path("data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15")
TRAIN_OTHERS = Path("data/spider/spider_data/train_others.json")
DEV_1032 = Path("data/testcases_spider_dev_full.jsonl")
DEV_1034 = Path("data/spider/spider_data/dev.json")
SPIDER_DIR = Path("data/spider/spider_data")
VALIDATION_MANIFEST = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_manifest.json"
)
OUTPUT_POOL = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_seed42.jsonl"
)
OUTPUT_POOL_MANIFEST = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_seed42_manifest.json"
)
OUTPUT_INDEX = Path(
    "data/retrieval_indexes/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_bge_large_en_v15"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the isolated Spider-Train + train_others retrieval ablation.")
    parser.add_argument("--write", action="store_true", help="Atomically write the new pool and index.")
    parser.add_argument("--device", default="cpu", help="SentenceTransformer device for the 935 new embeddings.")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def build_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    helper = load_module(root / "src/build_qwen35_mixed_validation_trainothers700_sqlcc1800.py", "retrieval_filter_helper")
    sft = load_module(root / "src/02_make_sft_dataset_v1_clean_full_chat.py", "retrieval_sft_helper")
    from transformers import AutoTokenizer

    base_rows = load_jsonl(root / BASE_INDEX / "metadata.jsonl")
    if len(base_rows) != EXPECTED_BASE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_BASE_ROWS} base rows, found {len(base_rows)}")
    train_others = load_json(root / TRAIN_OTHERS)
    dev_1032 = load_jsonl(root / DEV_1032)
    dev_1034_raw = load_json(root / DEV_1034)
    validation_manifest = load_json(root / VALIDATION_MANIFEST)
    reserved = set(validation_manifest["reserved_validation"]["reserved_train_others_validation_source_ids"])
    if len(reserved) != 700:
        raise RuntimeError(f"Expected 700 reserved train_others IDs, found {len(reserved)}")

    dev_1034 = [
        {
            "id": f"SPIDER_DEV_{index:06d}",
            "source_path": str(DEV_1034),
            "source_idx": index,
            "question": row.get("question", ""),
            "gold_sql": row.get("query", ""),
            "db_id": row.get("db_id", ""),
        }
        for index, row in enumerate(dev_1034_raw)
    ]
    dev_sets = {
        "dev1032": helper.identity_sets(dev_1032),
        "dev1034": helper.identity_sets(dev_1034),
    }
    base_sets = helper.identity_sets(base_rows)
    tokenizer = AutoTokenizer.from_pretrained(helper.TOKENIZER_ID, local_files_only=True)
    system_prompt, _, _, _ = sft.resolve_system_prompt(
        project_root=root,
        system_prompt_variant=helper.SYSTEM_PROMPT_VARIANT,
        system_prompt_path=None,
    )

    excluded: dict[str, list[Any]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    schema_cache: dict[str, str] = {}

    for index, raw in enumerate(train_others):
        row_id = f"SPIDER_TRAIN_OTHERS_{index:06d}"
        db_id = str(raw.get("db_id", "")).strip()
        question = str(raw.get("question", "")).strip()
        sql = str(raw.get("query", "")).strip()
        row = {
            "id": row_id,
            "source_dataset": "spider_train_others",
            "source_split": "train_others",
            "source_path": str(TRAIN_OTHERS),
            "source_idx": index,
            "db_id": db_id,
            "question": question,
            "gold_sql": sql if sql.endswith(";") else sql + ";",
            "embedding_text": question,
            "schema_format": "spider_table_columns_pk_fk_from_sqlite",
        }
        if row_id in reserved:
            excluded["reserved_mixed_validation"].append(row_id)
            continue
        if not db_id or not question or not sql:
            excluded["missing_required_field"].append(row_id)
            continue
        safe, reason = helper.statement_safety(sql)
        if not safe:
            excluded[f"sql_safety:{reason}"].append(row_id)
            continue
        identity = helper.row_identity(row)
        overlap = None
        for label, reference in dev_sets.items():
            for key in (
                "id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm",
                "pair_exact", "pair_norm", "schema_question", "schema_question_sql",
            ):
                if identity[key] in reference[key]:
                    overlap = f"{label}_{key}_overlap"
                    break
            if overlap:
                break
        if overlap:
            excluded[overlap].append(row_id)
            continue
        duplicate = next(
            (
                key for key in (
                    "question_exact", "question_norm", "pair_exact", "pair_norm",
                    "schema_question", "schema_question_sql",
                )
                if identity[key] in base_sets[key]
            ),
            None,
        )
        if duplicate:
            excluded[f"base_pool_{duplicate}_duplicate"].append(row_id)
            continue
        if identity["question_norm"] in seen_questions:
            excluded["internal_question_norm_duplicate"].append(row_id)
            continue
        if identity["pair_norm"] in seen_pairs:
            excluded["internal_pair_norm_duplicate"].append(row_id)
            continue
        db_path = root / SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        executable, execution_error = helper.execute_readonly(db_path, sql)
        if not executable:
            excluded[f"execution_error:{execution_error}"].append(row_id)
            continue
        if db_id not in schema_cache:
            schema_cache[db_id] = helper.sqlite_schema_prompt(db_path)
        row["schema_prompt"] = schema_cache[db_id]
        row["spider_schema"] = schema_cache[db_id]
        row["db_path"] = str(db_path.relative_to(root))
        rendered = helper.render_row(row, sft, system_prompt)
        token_length = len(tokenizer(rendered["text"], add_special_tokens=False)["input_ids"])
        if token_length > helper.MAX_LENGTH:
            excluded["over_2048_tokens"].append(row_id)
            continue
        row["full_chat_token_length"] = token_length
        selected.append(row)
        seen_questions.add(identity["question_norm"])
        seen_pairs.add(identity["pair_norm"])

    if len(selected) != EXPECTED_ADDED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ADDED_ROWS} eligible train_others rows, found {len(selected)}")

    dev_questions = [
        (row["id"], helper.normalize_question(row["question"]))
        for row in dev_1034
    ]
    near_pairs: list[dict[str, Any]] = []
    for row in selected:
        question_norm = helper.normalize_question(row["question"])
        tokens = set(question_norm.split())
        for dev_id, dev_question in dev_questions:
            dev_tokens = set(dev_question.split())
            union = tokens | dev_tokens
            jaccard = len(tokens & dev_tokens) / len(union) if union else 0.0
            if jaccard < 0.5:
                continue
            sequence = SequenceMatcher(None, question_norm, dev_question, autojunk=False).ratio()
            if sequence >= 0.90 or jaccard >= 0.80:
                near_pairs.append(
                    {
                        "pool_id": row["id"],
                        "dev_id": dev_id,
                        "sequence_similarity": sequence,
                        "token_jaccard": jaccard,
                    }
                )
    if near_pairs:
        raise RuntimeError(f"Conservative Dev near-question matches remain: {near_pairs[:20]}")

    combined = base_rows + selected
    if len(combined) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_ROWS} combined rows, found {len(combined)}")
    ids = [str(row.get("id", "")) for row in combined]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Combined retrieval pool contains duplicate IDs")

    lengths = [int(row["full_chat_token_length"]) for row in selected]
    stats = {
        "builder_version": BUILDER_VERSION,
        "seed": SEED,
        "counts": {
            "base_spider_train": len(base_rows),
            "train_others_input": len(train_others),
            "reserved_mixed_validation": len(reserved),
            "eligible_train_others": len(selected),
            "combined": len(combined),
        },
        "excluded_counts": {key: len(value) for key, value in sorted(excluded.items())},
        "excluded_ids_by_reason": {key: value for key, value in sorted(excluded.items())},
        "train_others_db_distribution": dict(sorted(Counter(row["db_id"] for row in selected).items())),
        "train_others_token_lengths": {
            "min": min(lengths),
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            "max": max(lengths),
            "over_2048": sum(value > 2048 for value in lengths),
        },
        "near_duplicate_thresholds": {"sequence_similarity": 0.90, "token_jaccard": 0.80},
        "near_dev_pairs": near_pairs,
        "selected_train_others_ids": [row["id"] for row in selected],
    }
    return combined, selected, stats


def build_index(root: Path, selected: list[dict[str, Any]], output_dir: Path, device: str, batch_size: int) -> dict[str, Any]:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    base_index_path = root / BASE_INDEX / "index.faiss"
    base_index = faiss.read_index(str(base_index_path))
    if base_index.ntotal != EXPECTED_BASE_ROWS:
        raise RuntimeError(f"Base FAISS rows mismatch: {base_index.ntotal}")
    dimension = int(base_index.d)
    base_vectors = base_index.reconstruct_n(0, int(base_index.ntotal))
    base_vectors = np.asarray(base_vectors, dtype=np.float32)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device, local_files_only=True)
    texts = [QUERY_PREFIX + row["question"] for row in selected]
    new_vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if new_vectors.shape != (EXPECTED_ADDED_ROWS, dimension):
        raise RuntimeError(f"Unexpected new embedding shape: {new_vectors.shape}")
    index = faiss.IndexFlatIP(dimension)
    index.add(base_vectors)
    index.add(new_vectors)
    if index.ntotal != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Combined FAISS rows mismatch: {index.ntotal}")

    output_dir.mkdir(parents=True, exist_ok=False)
    temporary = output_dir / "index.faiss.tmp"
    final = output_dir / "index.faiss"
    faiss.write_index(index, str(temporary))
    os.replace(temporary, final)
    reconstructed = index.reconstruct_n(0, EXPECTED_BASE_ROWS)
    max_base_delta = float(np.max(np.abs(np.asarray(reconstructed) - base_vectors)))
    if max_base_delta != 0.0:
        raise RuntimeError(f"Base embedding vectors changed during extension: {max_base_delta}")
    return {
        "index_type": "faiss.IndexFlatIP",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": dimension,
        "normalize_embeddings": True,
        "query_prefix": QUERY_PREFIX,
        "apply_query_prefix_to_documents": True,
        "apply_query_prefix_to_queries": True,
        "base_vectors_reused": EXPECTED_BASE_ROWS,
        "new_vectors_encoded": EXPECTED_ADDED_ROWS,
        "base_vector_max_abs_delta": max_base_delta,
        "device": device,
        "batch_size": batch_size,
        "packages": {
            "sentence-transformers": package_version("sentence-transformers"),
            "faiss-cpu": package_version("faiss-cpu"),
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / BASE_INDEX / "index.faiss",
        root / BASE_INDEX / "metadata.jsonl",
        root / BASE_INDEX / "manifest.json",
        root / TRAIN_OTHERS,
        root / DEV_1032,
        root / DEV_1034,
        root / VALIDATION_MANIFEST,
    ]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
    targets = [root / OUTPUT_POOL, root / OUTPUT_POOL_MANIFEST, root / OUTPUT_INDEX]
    conflicts = [str(path) for path in targets if path.exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing targets: " + ", ".join(conflicts))

    combined, selected, stats = build_rows(root)
    print(json.dumps({"dry_run": not args.write, **stats["counts"], "excluded_counts": stats["excluded_counts"]}, indent=2))
    if not args.write:
        return

    pool_text = jsonl_text(combined)
    atomic_write_text(root / OUTPUT_POOL, pool_text)
    index_stats = build_index(root, selected, root / OUTPUT_INDEX, args.device, args.batch_size)
    atomic_write_text(root / OUTPUT_INDEX / "metadata.jsonl", pool_text)

    manifest = {
        **stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_path": str(Path(__file__).resolve().relative_to(root)),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_paths": [str(path.relative_to(root)) for path in sources],
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sources},
        "output_pool": str(OUTPUT_POOL),
        "output_pool_sha256": sha256_file(root / OUTPUT_POOL),
        "output_index": str(OUTPUT_INDEX),
        "index": index_stats,
        "index_sha256": sha256_file(root / OUTPUT_INDEX / "index.faiss"),
        "metadata_sha256": sha256_file(root / OUTPUT_INDEX / "metadata.jsonl"),
        "retrieval_policy": {
            "candidate_generation": "BGE cosine/IP Top-10",
            "reranking": "structure_topk_v2",
            "final_demo_count": 1,
            "fewshot_representation": "full_schema_with_rules_question_gold_sql",
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(root / OUTPUT_POOL_MANIFEST, manifest_text)
    atomic_write_text(root / OUTPUT_INDEX / "manifest.json", manifest_text)
    print(json.dumps({
        "status": "PASS",
        "pool": str(OUTPUT_POOL),
        "pool_sha256": manifest["output_pool_sha256"],
        "index": str(OUTPUT_INDEX),
        "index_sha256": manifest["index_sha256"],
        "rows": len(combined),
    }, indent=2))


if __name__ == "__main__":
    main()
