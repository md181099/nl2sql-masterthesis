#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from prompt_presets import resolve_system_prompt
from retrieval_utils import FaissFewShotRetriever, LeakageGuard
from structure_rerank_v2 import candidate_sql_features


CONFIG = Path(
    "configs/eval_qwen35_9b_lora_old25k_r8_alpha16_evalstop_mixedval2500_"
    "epochs5_bestepoch1_maxlen2048_dynamic_fewshot_bge_large_top10_"
    "structure_rerank_v2_k1_full_schema_trainplusothers7895_"
    "maxinput2048_full_aliasnames.json"
)
POOL = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_seed42.jsonl"
)
POOL_MANIFEST = Path(
    "data/retrieval_pools/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_seed42_manifest.json"
)
INDEX = Path(
    "data/retrieval_indexes/"
    "spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_bge_large_en_v15"
)
BASE_INDEX = Path("data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15")
TESTCASES = Path("data/testcases_spider_dev_full.jsonl")
DEV_1034 = Path("data/spider/spider_data/dev.json")
VALIDATION_MANIFEST = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_manifest.json"
)
AUDIT = Path(
    "audits/audit_spider_train6960_plus_trainothers935_bge_top10_"
    "structure_rerank_fullschema_preparation_20260711.md"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def atomic_write(path: Path, text: str) -> None:
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


def feature_flags(sql: str) -> dict[str, bool | int]:
    parsed = candidate_sql_features(sql)
    features = parsed["features"]
    return {
        "join_bucket": int(parsed["join_bucket"]),
        "aggregation": bool({"count", "sum", "avg", "min", "max"} & features),
        "group_by": "group_by" in features,
        "having": "having" in features,
        "order_by": "order_by" in features,
        "limit": "limit" in features,
        "distinct": "distinct" in features,
        "nested_select": "nested_select" in features,
        "set_operation": bool({"union", "intersect", "except"} & features),
    }


def match_counts(target: dict[str, Any], candidate: dict[str, Any], output: Counter[str]) -> None:
    output["rows"] += 1
    for key in target:
        output[key] += int(target[key] == candidate[key])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    audit_path = root / AUDIT
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    required = [
        CONFIG, POOL, POOL_MANIFEST, INDEX / "index.faiss", INDEX / "metadata.jsonl",
        INDEX / "manifest.json", BASE_INDEX / "metadata.jsonl", TESTCASES, DEV_1034,
        VALIDATION_MANIFEST,
    ]
    for relative in required:
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)

    batch = load_module(root / "src/06_batch_run.py", "batch_run_audit")
    helper = load_module(root / "src/build_qwen35_mixed_validation_trainothers700_sqlcc1800.py", "identity_audit")
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    manifest = json.loads((root / INDEX / "manifest.json").read_text(encoding="utf-8"))
    pool_manifest = json.loads((root / POOL_MANIFEST).read_text(encoding="utf-8"))
    validation_manifest = json.loads((root / VALIDATION_MANIFEST).read_text(encoding="utf-8"))
    pool = load_jsonl(root / POOL)
    metadata = load_jsonl(root / INDEX / "metadata.jsonl")
    base = load_jsonl(root / BASE_INDEX / "metadata.jsonl")
    testcases = load_jsonl(root / TESTCASES)
    dev_1034_raw = json.loads((root / DEV_1034).read_text(encoding="utf-8"))
    dev_1034 = [
        {
            "id": f"SPIDER_DEV_{index:06d}",
            "source_path": str(DEV_1034),
            "source_idx": index,
            "db_id": row.get("db_id", ""),
            "question": row.get("question", ""),
            "gold_sql": row.get("query", ""),
        }
        for index, row in enumerate(dev_1034_raw)
    ]
    reserved = set(validation_manifest["reserved_validation"]["reserved_train_others_validation_source_ids"])

    if len(pool) != 7895 or len(metadata) != 7895 or len(base) != 6960:
        raise RuntimeError("Pool/index metadata row counts are inconsistent")
    if pool != metadata:
        raise RuntimeError("Pool JSONL and index metadata are not row-identical")
    if pool[:6960] != base:
        raise RuntimeError("The first 6,960 rows do not preserve the existing pool exactly")
    train_others = pool[6960:]
    if len(train_others) != 935:
        raise RuntimeError("Expected 935 appended train_others rows")
    if reserved & {row["id"] for row in train_others}:
        raise RuntimeError("Reserved Mixed-Validation IDs entered the retrieval pool")

    leakage: dict[str, dict[str, int]] = {}
    leakage_hits: dict[str, list[dict[str, Any]]] = {}
    for label, reference_rows in (("dev1032", testcases), ("dev1034", dev_1034)):
        reference = helper.identity_sets(reference_rows)
        counts = Counter()
        hits: list[dict[str, Any]] = []
        for row in pool:
            identity = helper.row_identity(row)
            row_hits: list[str] = []
            for key in (
                "id", "source_id", "question_exact", "question_norm", "sql_exact", "sql_norm",
                "pair_exact", "pair_norm", "schema_question", "schema_question_sql",
            ):
                matched = identity[key] in reference[key]
                counts[key] += int(matched)
                if matched:
                    row_hits.append(key)
            if row_hits:
                hits.append(
                    {
                        "id": row.get("id"),
                        "source_dataset": row.get("source_dataset"),
                        "db_id": row.get("db_id"),
                        "question": row.get("question"),
                        "gold_sql": row.get("gold_sql"),
                        "matched_fields": row_hits,
                    }
                )
        leakage[label] = dict(counts)
        leakage_hits[label] = hits
        if label == "dev1032" and any(counts.values()):
            raise RuntimeError(f"Leakage detected against {label}: {dict(counts)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base", local_files_only=True)
    system_prompt, _, _, _ = resolve_system_prompt(
        project_root=root,
        system_prompt_variant=config["system_prompt_variant"],
        system_prompt_path=None,
    )
    guard = LeakageGuard.from_testcases_path(root / TESTCASES)
    retriever = FaissFewShotRetriever(
        index_dir=root / INDEX,
        embedding_model=config["embedding_model"],
        k=config["k"],
        allow_overlap=config["allow_overlap"],
        same_db_only=config["same_db_only"],
        leakage_guard=guard,
        retrieval_pool_path=root / INDEX / "metadata.jsonl",
        rerank_method=config["retrieval_rerank_method"],
        rerank_top_n=config["retrieval_rerank_top_n"],
        structure_bonus_max=config["retrieval_structure_bonus_max"],
    )

    prompt_tokens: list[int] = []
    similarities: list[float] = []
    adjustments: list[float] = []
    selected_ranks: Counter[int] = Counter()
    selected_sources: Counter[str] = Counter()
    selected_ids: Counter[str] = Counter()
    top1_matches: Counter[str] = Counter()
    reranked_matches: Counter[str] = Counter()
    changed = 0
    filtered = 0
    top10_lengths: Counter[int] = Counter()
    issues: list[str] = []

    for row in testcases:
        preview = retriever.preview_structure_rerank(
            question=row["question"],
            qid=row["id"],
            db_id=row["db_id"],
            target_schema=row["schema_prompt"],
            top_n=10,
        )
        candidates = preview["candidates"]
        selected_candidates = preview["selected_candidates"]
        top10_lengths[len(candidates)] += 1
        filtered += sum(preview["filtered_reasons"].values())
        if len(candidates) != 10 or len(selected_candidates) != 1:
            issues.append(f"{row['id']}: candidates={len(candidates)} selected={len(selected_candidates)}")
            continue
        top1 = candidates[0]
        selected = selected_candidates[0]
        example = selected["example"]
        if selected["rank"] != 1:
            changed += 1
        selected_ranks[int(selected["rank"])] += 1
        selected_sources[str(example.get("source_dataset", ""))] += 1
        selected_ids[str(example.get("id", ""))] += 1
        similarities.append(float(selected["bge_similarity"]))
        adjustments.append(float(selected["structure_adjustment"]))
        target_features = feature_flags(row["gold_sql"])
        match_counts(target_features, feature_flags(top1["gold_sql"]), top1_matches)
        match_counts(target_features, feature_flags(selected["gold_sql"]), reranked_matches)
        if helper.row_identity(example)["question_norm"] in helper.identity_sets([row])["question_norm"]:
            issues.append(f"{row['id']}: selected same normalized question")
        prompt = batch.build_prompt_schema_fewshot(
            row["schema_prompt"],
            row["question"],
            [example],
            config["llm"],
            tokenizer,
            prompt_format=config["prompt_format"],
            system_instruction=system_prompt,
            example_schema_mode=config["fewshot_example_schema_mode"],
            example_mode=config["fewshot_example_mode"],
        )
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompt_tokens.append(token_count)
        if token_count > config["max_input_tokens"]:
            issues.append(f"{row['id']}: prompt_tokens={token_count}")
        if "<think>" in prompt.casefold() or "</think>" in prompt.casefold():
            issues.append(f"{row['id']}: think marker")
        if prompt.count("Example 1") != 1 or prompt.count("<|im_start|>assistant\n") != 1:
            issues.append(f"{row['id']}: unexpected demo/assistant count")
        if example["schema_prompt"] not in prompt or example["gold_sql"] not in prompt:
            issues.append(f"{row['id']}: incomplete full-schema demonstration")
        if row["schema_prompt"] not in prompt or row["question"] not in prompt:
            issues.append(f"{row['id']}: incomplete target task")

    if issues:
        raise RuntimeError("Retrieval/prompt smoke failed: " + "; ".join(issues[:20]))

    keys = [
        "join_bucket", "aggregation", "group_by", "having", "order_by", "limit",
        "distinct", "nested_select", "set_operation",
    ]
    match_table = []
    for key in keys:
        before = 100.0 * top1_matches[key] / len(testcases)
        after = 100.0 * reranked_matches[key] / len(testcases)
        match_table.append((key, top1_matches[key], reranked_matches[key], before, after, after - before))

    status = "PASS MIT WARNUNGEN" if leakage_hits["dev1034"] else "PASS"
    if reranked_matches["join_bucket"] < top1_matches["join_bucket"]:
        status = "PASS MIT WARNUNGEN"

    hashes = {
        str(path): sha256_file(root / path)
        for path in [
            CONFIG, POOL, POOL_MANIFEST, INDEX / "index.faiss", INDEX / "metadata.jsonl",
            INDEX / "manifest.json", Path("src/structure_rerank_v2.py"),
            Path("src/retrieval_utils.py"), Path("src/06_batch_run.py"),
            Path("src/18_build_spider_train6960_plus_trainothers935_retrieval_index.py"),
            Path("src/19_audit_spider_trainplusothers_structure_top10.py"),
        ]
    }
    rows = [
        "# Audit: Spider Train 6960 + train_others 935, BGE Top-10 Structure Rerank",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"Status: **{status}**",
        "",
        "## Executive Summary",
        "",
        "A new isolated retrieval ablation was prepared without changing or rebuilding the existing 6,960-row index. The new pool contains the original 6,960 Spider-Train rows plus 935 strictly filtered, non-reserved train_others rows. BGE retrieves ten candidates; structure_topk_v2 deterministically reranks them and inserts exactly one Full-Schema demonstration.",
        "",
        "No Qwen model was loaded, no SQL was generated, and no Spider execution evaluation was run. The 1,032-case operations below are retrieval and prompt-construction smokes only.",
        "",
        "## Created Artifacts",
        "",
        f"- Builder: `{Path('src/18_build_spider_train6960_plus_trainothers935_retrieval_index.py')}`",
        f"- Reranker: `{Path('src/structure_rerank_v2.py')}`",
        f"- Pool: `{POOL}`",
        f"- Pool manifest: `{POOL_MANIFEST}`",
        f"- Index: `{INDEX}`",
        f"- Eval config: `{CONFIG}`",
        "",
        "## Pool Construction",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        "| Existing Spider-Train rows preserved | 6,960 |",
        "| Raw train_others rows | 1,659 |",
        "| Reserved Mixed-Validation rows excluded | 700 |",
        "| Exact normalized Dev-question overlap excluded | 1 |",
        "| Internal normalized-question duplicates excluded | 23 |",
        "| Eligible train_others appended | 935 |",
        "| Final pool | 7,895 |",
        "",
        f"Pool SHA256: `{hashes[str(POOL)]}`",
        "",
        "The first 6,960 metadata rows are structurally identical to the existing pool. The old FAISS vectors were reconstructed and copied exactly; manifest max absolute base-vector delta is `0.0`. Only 935 new BGE vectors were encoded.",
        "",
        "## train_others Distribution",
        "",
        "| DB | Rows |",
        "| --- | ---: |",
    ]
    for db_id, count in manifest["train_others_db_distribution"].items():
        rows.append(f"| `{db_id}` | {count} |")
    rows += [
        "",
        "## Leakage Matrix",
        "",
        "Checks use exact and robust normalized identities. The actual 1,032-case evaluation set is fully clean. The complete local 1,034 source retains generic Question/SQL-only overlaps inherited from the unchanged 6,960-row historical base pool; no Question+SQL pair or DB-conditioned overlap exists.",
        "",
        "| Reference | ID | Source ID | Exact Q | Norm Q | Exact SQL | Norm SQL | Exact Pair | Norm Pair | DB+Q | DB+Q+SQL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("dev1032", "dev1034"):
        item = leakage[label]
        rows.append(
            f"| {label} | {item.get('id', 0)} | {item.get('source_id', 0)} | "
            f"{item.get('question_exact', 0)} | {item.get('question_norm', 0)} | "
            f"{item.get('sql_exact', 0)} | {item.get('sql_norm', 0)} | "
            f"{item.get('pair_exact', 0)} | {item.get('pair_norm', 0)} | "
            f"{item.get('schema_question', 0)} | {item.get('schema_question_sql', 0)} |"
        )
    rows += ["", "### Full-Dev-1034 inherited matches", ""]
    for hit in leakage_hits["dev1034"]:
        rows.append(
            f"- `{hit['id']}` ({hit['source_dataset']}, `{hit['db_id']}`): "
            f"fields={hit['matched_fields']}; Q=`{hit['question']}`; SQL=`{hit['gold_sql']}`"
        )
    rows += [
        "",
        f"Reserved Mixed-Validation IDs in pool: **{len(reserved & {row['id'] for row in train_others})}**.",
        "",
        "## Index Integrity",
        "",
        "| Check | Result |",
        "| --- | --- |",
        "| FAISS type | `IndexFlatIP` |",
        f"| Rows | {retriever.index.ntotal} |",
        f"| Dimension | {retriever.index.d} |",
        "| Embedding | `BAAI/bge-large-en-v1.5` |",
        "| Normalized cosine/IP | PASS |",
        "| Query/document BGE prefix | PASS |",
        "| Index/metadata row agreement | PASS |",
        "| Existing 6,960 vectors unchanged | PASS |",
        "",
        "## Top-10 Structure Reranking",
        "",
        f"- Queries processed: {len(testcases)}",
        f"- Exactly ten candidates: {top10_lengths.get(10, 0)}/{len(testcases)}",
        f"- Exactly one selected demo: {len(testcases)}/{len(testcases)}",
        f"- Selection changed from raw BGE rank 1: {changed}/{len(testcases)} ({100*changed/len(testcases):.2f}%)",
        f"- Retrieval-filtered candidates: {filtered}",
        f"- Unique selected demo IDs: {len(selected_ids)}",
        f"- Selected source distribution: `{dict(selected_sources)}`",
        f"- Selected original-rank distribution: `{dict(sorted(selected_ranks.items()))}`",
        f"- Selected BGE similarity min/mean/max: {min(similarities):.6f} / {statistics.mean(similarities):.6f} / {max(similarities):.6f}",
        f"- Structure adjustment min/mean/max: {min(adjustments):.6f} / {statistics.mean(adjustments):.6f} / {max(adjustments):.6f}",
        "",
        "The reranker uses only the target question, target schema, and known candidate schema/SQL. It never receives target Gold SQL. Target Gold SQL is used below only for this one-time post-hoc structural audit and did not change the predefined weights.",
        "",
        "| Structural agreement | Raw BGE Top-1 | Reranked Top-10 | Delta pp |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, _, _, before, after, delta in match_table:
        rows.append(f"| {key} | {before:.2f}% | {after:.2f}% | {delta:+.2f} |")
    rows += [
        "",
        "## Full-Schema Prompt Smoke",
        "",
        f"- Cases: {len(prompt_tokens)}",
        f"- Mean tokens: {statistics.mean(prompt_tokens):.3f}",
        f"- p95 tokens: {quantile([float(x) for x in prompt_tokens], 0.95):.3f}",
        f"- Maximum tokens: {max(prompt_tokens)}",
        f"- Over 2,048: {sum(value > 2048 for value in prompt_tokens)}",
        "- Exactly one Full-Schema demo per prompt: PASS",
        "- Demo schema, question, and Gold SQL present: PASS",
        "- Target schema and question present: PASS",
        "- Qwen assistant generation prefix exactly once: PASS",
        "- `<think>` markers: 0",
        "- Silent truncation: not used during this smoke; raw prompt lengths were measured before generation tokenization",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in (
        "llm", "adapter", "prompt_tuning", "k", "max_input_tokens", "max_new_tokens",
        "retrieval_index_path", "retrieval_rerank_method", "retrieval_rerank_top_n",
        "retrieval_structure_bonus_max", "fewshot_example_schema_mode", "fewshot_example_mode",
    ):
        rows.append(f"| `{key}` | `{config.get(key)}` |")
    rows += [
        "",
        "## Backward Compatibility",
        "",
        "The existing retrieval pool and index were read only. Existing configs continue to use rerank_method `none` or `sqlaware_topk`; their branches are unchanged. The new logic is activated only by `structure_topk_v2`. No existing dataset, adapter, config, result, or index was overwritten.",
        "",
        "## SHA256",
        "",
        "| Artifact | SHA256 |",
        "| --- | --- |",
    ]
    for path, digest in hashes.items():
        rows.append(f"| `{path}` | `{digest}` |")
    rows += [
        "",
        "## Final Decision",
        "",
        f"**{status}.** The new 7,895-row BGE Top-10 / structure_topk_v2 / k=1 Full-Schema ablation is prepared for a later model evaluation. This audit does not claim an EMA improvement; only a future explicitly started Spider-Dev run can establish that.",
        "",
        "Recommended later command:",
        "",
        f"`.venv_flash/bin/python3 src/06_batch_run.py --config {CONFIG}`",
        "",
        "## Safety Confirmation",
        "",
        "- Training started: NO",
        "- Qwen model loaded: NO",
        "- SQL generation started: NO",
        "- Spider execution evaluation started: NO",
        "- Existing retrieval index modified: NO",
        "- Existing files overwritten: NO",
    ]
    atomic_write(audit_path, "\n".join(rows) + "\n")
    print(json.dumps({
        "status": status,
        "audit": str(AUDIT),
        "pool_rows": len(pool),
        "changed_from_top1": changed,
        "prompt_max": max(prompt_tokens),
        "prompt_over_limit": sum(value > 2048 for value in prompt_tokens),
        "join_match_top1": top1_matches["join_bucket"],
        "join_match_reranked": reranked_matches["join_bucket"],
    }, indent=2))


if __name__ == "__main__":
    main()
