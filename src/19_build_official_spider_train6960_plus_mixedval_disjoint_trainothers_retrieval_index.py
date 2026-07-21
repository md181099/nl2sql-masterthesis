#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SEED = 42
EXPECTED_BASE_ROWS = 6960
EXPECTED_ADDED_ROWS = 103
EXPECTED_TOTAL_ROWS = 7063
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BUILDER_VERSION = "official_spider_train6960_plus_mixedval_disjoint_trainothers_v1"

SOURCE_INDEX = Path(
    "data/retrieval_indexes/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_bge_large_en_v15"
)
SOURCE_POOL = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_seed42.jsonl"
)
VALIDATION = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42.jsonl"
)
VALIDATION_MANIFEST = Path(str(VALIDATION).replace(".jsonl", "_manifest.json"))
OLD25K = Path(
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
SPIDER_TRAIN = Path("data/spider/spider_data/train_spider.json")
TRAIN_OTHERS = Path("data/spider/spider_data/train_others.json")
SQLCC_RAW = Path("data/sql_create_context/train.jsonl")
DEV_1032 = Path("data/testcases_spider_dev_full.jsonl")
DEV_1034 = Path("data/spider/spider_data/dev.json")

OUTPUT_POOL = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers103_mixedval_disjoint_official_seed42.jsonl"
)
OUTPUT_POOL_MANIFEST = Path(str(OUTPUT_POOL).replace(".jsonl", "_manifest.json"))
OUTPUT_INDEX = Path(
    "data/retrieval_indexes/"
    "spider_train6960_plus_trainothers103_mixedval_disjoint_official_bge_large_en_v15"
)

IDENTITY_KEYS = (
    "id",
    "source_id",
    "question_exact",
    "question_norm",
    "sql_exact",
    "sql_norm",
    "pair_exact",
    "pair_norm",
    "schema_question",
    "schema_question_sql",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the official Mixed-Validation-disjoint Spider retrieval pool."
    )
    parser.add_argument("--write", action="store_true", help="Write all new artifacts atomically.")
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def source_rows(raw_rows: list[dict[str, Any]], source_path: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        rows.append(
            {
                "id": str(raw.get("id") or f"{prefix}_{index:06d}"),
                "source_path": str(source_path),
                "source_idx": index,
                "question": str(raw.get("question", "")).strip(),
                "gold_sql": str(raw.get("gold_sql") or raw.get("query") or raw.get("answer") or "").strip(),
                "schema_prompt": str(raw.get("schema_prompt") or raw.get("context") or "").strip(),
                "db_id": str(raw.get("db_id", "")).strip(),
            }
        )
    return rows


def reconstruct_validation_rows(
    manifest: dict[str, Any], train_others_raw: list[dict[str, Any]], sqlcc_raw: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["selected_provenance"]:
        source_path = str(item["source_path"])
        source_idx = int(item["source_idx"])
        if source_path == str(TRAIN_OTHERS):
            raw = train_others_raw[source_idx]
            sql = raw.get("query", "")
            schema = ""
        elif source_path == str(SQLCC_RAW):
            raw = sqlcc_raw[source_idx]
            sql = raw.get("answer", "")
            schema = raw.get("context", "")
        else:
            raise RuntimeError(f"Unexpected validation source: {source_path}")
        rows.append(
            {
                "id": str(item["id"]),
                "source_path": source_path,
                "source_idx": source_idx,
                "question": str(raw.get("question", "")).strip(),
                "gold_sql": str(sql).strip(),
                "schema_prompt": str(schema).strip(),
                "db_id": str(item.get("db_id") or "").strip(),
            }
        )
    if len(rows) != 2500:
        raise RuntimeError(f"Expected 2500 reconstructed validation rows, found {len(rows)}")
    return rows


def overlap_matrix(rows: list[dict[str, Any]], reference_sets: dict[str, set[Any]], helper: Any) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        identity = helper.row_identity(row)
        for key in IDENTITY_KEYS:
            if identity[key] in reference_sets[key]:
                counts[key] += 1
    return {key: int(counts[key]) for key in IDENTITY_KEYS}


def overlap_details(
    row: dict[str, Any], reference_sets: dict[str, set[Any]], helper: Any
) -> list[str]:
    identity = helper.row_identity(row)
    return [key for key in IDENTITY_KEYS if identity[key] in reference_sets[key]]


def prepare(root: Path) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    helper = load_module(
        root / "src/build_qwen35_mixed_validation_trainothers700_sqlcc1800.py",
        "official_pool_identity_helper",
    )
    structure = load_module(root / "src/structure_rerank_v2.py", "official_pool_structure_helper")

    expanded = load_jsonl(root / SOURCE_INDEX / "metadata.jsonl")
    source_pool_rows = load_jsonl(root / SOURCE_POOL)
    if expanded != source_pool_rows:
        raise RuntimeError("Expanded index metadata is not row-identical to its logical pool")
    if len(expanded) != 7895:
        raise RuntimeError(f"Expected 7895 expanded rows, found {len(expanded)}")

    validation_manifest = load_json(root / VALIDATION_MANIFEST)
    validation_sft = load_jsonl(root / VALIDATION)
    if len(validation_sft) != 2500:
        raise RuntimeError(f"Expected 2500 validation JSONL rows, found {len(validation_sft)}")
    train_others_raw = load_json(root / TRAIN_OTHERS)
    sqlcc_raw = load_jsonl(root / SQLCC_RAW)
    validation_rows = reconstruct_validation_rows(validation_manifest, train_others_raw, sqlcc_raw)
    validation_sets = helper.identity_sets(validation_rows)

    reserved = validation_manifest["reserved_validation"]
    reserved_ids = set(reserved["reserved_train_others_validation_source_ids"]) | set(
        reserved["reserved_sqlcc_validation_source_ids"]
    )
    question_hashes = set(reserved["normalized_question_hashes"])
    sql_hashes = set(reserved["normalized_sql_hashes"])
    pair_hashes = set(reserved["normalized_pair_hashes"])

    selected: list[dict[str, Any]] = []
    selected_positions: list[int] = []
    excluded: list[dict[str, Any]] = []
    excluded_reason_counts = Counter()
    for position, row in enumerate(expanded):
        identity = helper.row_identity(row)
        reasons = overlap_details(row, validation_sets, helper)
        if str(row.get("id", "")) in reserved_ids:
            reasons.append("reserved_source_id")
        if sha256_text(identity["question_norm"]) in question_hashes:
            reasons.append("reserved_question_hash")
        if sha256_text(identity["sql_norm"]) in sql_hashes:
            reasons.append("reserved_sql_hash")
        if sha256_text("\n".join(identity["pair_norm"])) in pair_hashes:
            reasons.append("reserved_pair_hash")
        reasons = sorted(set(reasons))
        if reasons:
            excluded.append(
                {
                    "id": row.get("id"),
                    "source_path": row.get("source_path"),
                    "source_idx": row.get("source_idx"),
                    "db_id": row.get("db_id"),
                    "reasons": reasons,
                }
            )
            excluded_reason_counts.update(reasons)
            continue
        selected.append(row)
        selected_positions.append(position)

    base_count = sum(
        str(row.get("source_split")) == "train_spider" for row in selected
    )
    added = [row for row in selected if str(row.get("source_split")) == "train_others"]
    if (base_count, len(added), len(selected)) != (
        EXPECTED_BASE_ROWS,
        EXPECTED_ADDED_ROWS,
        EXPECTED_TOTAL_ROWS,
    ):
        raise RuntimeError(
            f"Unexpected final composition: base={base_count}, added={len(added)}, total={len(selected)}"
        )
    if selected_positions[:EXPECTED_BASE_ROWS] != list(range(EXPECTED_BASE_ROWS)):
        raise RuntimeError("The official 6960-row base was not preserved in exact order")

    dev1032 = load_jsonl(root / DEV_1032)
    dev1034 = source_rows(load_json(root / DEV_1034), DEV_1034, "SPIDER_DEV")
    old25k = load_jsonl(root / OLD25K)
    spider_train = source_rows(load_json(root / SPIDER_TRAIN), SPIDER_TRAIN, "SPIDER_TRAIN")
    train_others = source_rows(train_others_raw, TRAIN_OTHERS, "SPIDER_TRAIN_OTHERS")

    reference_rows = {
        "mixed_validation": validation_rows,
        "spider_dev_1032": dev1032,
        "spider_dev_1034": dev1034,
        "old25k": old25k,
        "spider_train_raw": spider_train,
        "train_others_raw": train_others,
    }
    matrices = {
        name: overlap_matrix(selected, helper.identity_sets(rows), helper)
        for name, rows in reference_rows.items()
    }
    cross_checks = {
        "mixed_validation_vs_old25k": overlap_matrix(
            validation_rows, helper.identity_sets(old25k), helper
        ),
        "mixed_validation_vs_dev1032": overlap_matrix(
            validation_rows, helper.identity_sets(dev1032), helper
        ),
        "mixed_validation_vs_dev1034": overlap_matrix(
            validation_rows, helper.identity_sets(dev1034), helper
        ),
    }
    if any(matrices["mixed_validation"].values()):
        raise RuntimeError(f"Mixed-Validation overlap remains: {matrices['mixed_validation']}")
    if any(matrices["spider_dev_1032"].values()):
        raise RuntimeError(f"Spider Dev 1032 overlap remains: {matrices['spider_dev_1032']}")

    dev1034_sets = helper.identity_sets(dev1034)
    dev1034_details = []
    for row in selected:
        reasons = overlap_details(row, dev1034_sets, helper)
        if reasons:
            dev1034_details.append(
                {
                    "id": row.get("id"),
                    "db_id": row.get("db_id"),
                    "question": row.get("question"),
                    "gold_sql": row.get("gold_sql"),
                    "overlap_types": reasons,
                }
            )

    features = [structure.candidate_sql_features(row.get("gold_sql", "")) for row in added]
    feature_counts = Counter()
    for item in features:
        feature_counts.update(item["features"])
        feature_counts[f"join_bucket_{item['join_bucket']}"] += 1
    token_lengths = [int(row.get("full_chat_token_length", 0)) for row in added]

    stats = {
        "builder_version": BUILDER_VERSION,
        "seed": SEED,
        "counts": {
            "expanded_source_pool": len(expanded),
            "base_spider_train": base_count,
            "eligible_train_others": len(added),
            "excluded_from_expanded_pool": len(excluded),
            "combined": len(selected),
        },
        "selection_rule": (
            "Preserve all 6960 official Spider-Train base rows in order; retain an expanded-pool "
            "train_others row only when ID, source ID, exact/normalized Question, exact/normalized "
            "SQL, exact/normalized Pair, DB+Question, DB+Question+SQL, and all reserved Mixed-" 
            "Validation Question/SQL/Pair hashes are disjoint."
        ),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "excluded_rows": excluded,
        "selected_train_others_ids": [row["id"] for row in added],
        "train_others_db_distribution": dict(sorted(Counter(row["db_id"] for row in added).items())),
        "train_others_structure_distribution": dict(sorted(feature_counts.items())),
        "train_others_token_lengths": {
            "min": min(token_lengths),
            "mean": statistics.mean(token_lengths),
            "median": statistics.median(token_lengths),
            "p95": percentile(token_lengths, 0.95),
            "max": max(token_lengths),
            "over_2048": sum(value > 2048 for value in token_lengths),
        },
        "leakage_matrix": matrices,
        "dataset_cross_checks": cross_checks,
        "spider_dev_1034_overlap_details": dev1034_details,
    }
    return selected, selected_positions, stats


def stage_artifacts(
    root: Path, selected: list[dict[str, Any]], selected_positions: list[int], stats: dict[str, Any]
) -> dict[str, Any]:
    import faiss
    import numpy as np

    source_index = faiss.read_index(str(root / SOURCE_INDEX / "index.faiss"))
    if source_index.ntotal != 7895 or source_index.d != 1024:
        raise RuntimeError(
            f"Unexpected source index shape: ntotal={source_index.ntotal}, d={source_index.d}"
        )
    source_vectors = np.asarray(
        source_index.reconstruct_n(0, int(source_index.ntotal)), dtype=np.float32
    )
    vectors = source_vectors[np.asarray(selected_positions, dtype=np.int64)]
    index = faiss.IndexFlatIP(int(source_index.d))
    index.add(vectors)
    if index.ntotal != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Unexpected output index size: {index.ntotal}")
    reconstructed = np.asarray(index.reconstruct_n(0, EXPECTED_TOTAL_ROWS), dtype=np.float32)
    max_delta = float(np.max(np.abs(reconstructed - vectors)))
    if max_delta != 0.0:
        raise RuntimeError(f"Reused vectors changed: max absolute delta={max_delta}")

    pool_text = jsonl_text(selected)
    pool_parent = root / OUTPUT_POOL.parent
    index_parent = root / OUTPUT_INDEX.parent
    pool_parent.mkdir(parents=True, exist_ok=True)
    index_parent.mkdir(parents=True, exist_ok=True)
    pool_fd, pool_tmp_name = tempfile.mkstemp(prefix=OUTPUT_POOL.name + ".", suffix=".tmp", dir=pool_parent)
    manifest_fd, manifest_tmp_name = tempfile.mkstemp(
        prefix=OUTPUT_POOL_MANIFEST.name + ".", suffix=".tmp", dir=pool_parent
    )
    os.close(manifest_fd)
    index_tmp = Path(tempfile.mkdtemp(prefix=OUTPUT_INDEX.name + ".", dir=index_parent))
    pool_tmp = Path(pool_tmp_name)
    manifest_tmp = Path(manifest_tmp_name)
    try:
        with os.fdopen(pool_fd, "w", encoding="utf-8") as handle:
            handle.write(pool_text)
            handle.flush()
            os.fsync(handle.fileno())
        (index_tmp / "metadata.jsonl").write_text(pool_text, encoding="utf-8")
        faiss.write_index(index, str(index_tmp / "index.faiss"))

        sources = [
            SOURCE_INDEX / "index.faiss",
            SOURCE_INDEX / "metadata.jsonl",
            SOURCE_INDEX / "manifest.json",
            SOURCE_POOL,
            VALIDATION,
            VALIDATION_MANIFEST,
            OLD25K,
            SPIDER_TRAIN,
            TRAIN_OTHERS,
            SQLCC_RAW,
            DEV_1032,
            DEV_1034,
        ]
        manifest = {
            **stats,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder_path": str(Path(__file__).resolve().relative_to(root)),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "source_paths": [str(path) for path in sources],
            "source_sha256": {str(path): sha256_file(root / path) for path in sources},
            "output_pool": str(OUTPUT_POOL),
            "output_pool_sha256": sha256_file(pool_tmp),
            "output_index": str(OUTPUT_INDEX),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": int(index.d),
            "normalize": True,
            "normalize_embeddings": True,
            "query_prefix": QUERY_PREFIX,
            "apply_query_prefix_to_documents": True,
            "apply_query_prefix_to_queries": True,
            "index": {
                "index_type": "faiss.IndexFlatIP",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": int(index.d),
                "normalize_embeddings": True,
                "query_prefix": QUERY_PREFIX,
                "apply_query_prefix_to_documents": True,
                "apply_query_prefix_to_queries": True,
                "vectors_reused_from_existing_index": EXPECTED_TOTAL_ROWS,
                "vectors_newly_encoded": 0,
                "source_vector_positions": selected_positions,
                "max_abs_reconstruction_delta": max_delta,
                "vector_norm_min": float(np.linalg.norm(vectors, axis=1).min()),
                "vector_norm_mean": float(np.linalg.norm(vectors, axis=1).mean()),
                "vector_norm_max": float(np.linalg.norm(vectors, axis=1).max()),
                "model_loaded": False,
                "packages": {
                    "faiss-cpu": package_version("faiss-cpu"),
                    "numpy": package_version("numpy"),
                },
            },
            "index_sha256": sha256_file(index_tmp / "index.faiss"),
            "metadata_sha256": sha256_file(index_tmp / "metadata.jsonl"),
            "retrieval_variants": {
                "bge_top1": {"rerank_method": "none", "k": 1},
                "bge_top10_structure_rerank_v2": {
                    "candidate_top_n": 10,
                    "rerank_method": "structure_topk_v2",
                    "final_k": 1,
                    "structure_bonus_max": 0.08,
                },
            },
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        manifest_tmp.write_text(manifest_text, encoding="utf-8")
        (index_tmp / "manifest.json").write_text(manifest_text, encoding="utf-8")

        # All expensive checks have completed; publish only new paths.
        os.replace(pool_tmp, root / OUTPUT_POOL)
        os.replace(manifest_tmp, root / OUTPUT_POOL_MANIFEST)
        os.replace(index_tmp, root / OUTPUT_INDEX)
        return manifest
    except Exception:
        pool_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        if index_tmp.exists():
            shutil.rmtree(index_tmp)
        raise


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    required = [
        SOURCE_INDEX / "index.faiss",
        SOURCE_INDEX / "metadata.jsonl",
        SOURCE_INDEX / "manifest.json",
        SOURCE_POOL,
        VALIDATION,
        VALIDATION_MANIFEST,
        OLD25K,
        SPIDER_TRAIN,
        TRAIN_OTHERS,
        SQLCC_RAW,
        DEV_1032,
        DEV_1034,
    ]
    for path in required:
        if not (root / path).exists():
            raise FileNotFoundError(path)
    targets = [root / OUTPUT_POOL, root / OUTPUT_POOL_MANIFEST, root / OUTPUT_INDEX]
    conflicts = [str(path.relative_to(root)) for path in targets if path.exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing targets: " + ", ".join(conflicts))

    selected, selected_positions, stats = prepare(root)
    print(
        json.dumps(
            {
                "dry_run": not args.write,
                "counts": stats["counts"],
                "train_others_db_distribution": stats["train_others_db_distribution"],
                "mixed_validation": stats["leakage_matrix"]["mixed_validation"],
                "spider_dev_1032": stats["leakage_matrix"]["spider_dev_1032"],
                "spider_dev_1034": stats["leakage_matrix"]["spider_dev_1034"],
            },
            indent=2,
        )
    )
    if not args.write:
        return
    manifest = stage_artifacts(root, selected, selected_positions, stats)
    print(
        json.dumps(
            {
                "status": "PASS",
                "pool": str(OUTPUT_POOL),
                "pool_sha256": manifest["output_pool_sha256"],
                "index": str(OUTPUT_INDEX),
                "index_sha256": manifest["index_sha256"],
                "rows": len(selected),
                "model_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
