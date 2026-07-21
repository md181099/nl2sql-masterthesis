#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import faiss  # type: ignore  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from retrieval_utils_dynamic_k3_v1 import (  # noqa: E402
    LeakageGuard,
    _filter_reason,
    load_jsonl,
)
from structure_rerank_v2 import structure_rerank_adjustment  # noqa: E402


MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
OUT_PROMPTS = ROOT / "audits/derived/dynamic_k3_prompt_preflight_20260717.csv"
OUT_RETRIEVAL = ROOT / "audits/derived/dynamic_k3_retrieval_selection_validation_20260717.csv"
OUT_SUMMARY = ROOT / "audits/derived/dynamic_k3_prompt_preflight_summary_20260717.json"
OUT_CONTEXT = ROOT / "audits/derived/dynamic_k3_model_context_capacity_20260717.json"
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
INDEX_DIR = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15"
EXPECTED_INTERPRETER = ROOT / ".venv_flash/bin/python"
MODEL_REVISIONS = {
    "qwen2b": ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
    "llama3b": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "qwen9b": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
}
CONDITION_SOURCE = {
    "top3": "top",
    "top3_gate070": "top",
    "top3_gate085": "top",
    "structure_top3": "structure",
    "structure_top3_gate070": "structure",
    "structure_top3_gate085": "structure",
}
THRESHOLDS = {
    "top3_gate070": 0.70,
    "top3_gate085": 0.85,
    "structure_top3_gate070": 0.70,
    "structure_top3_gate085": 0.85,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def write_scoped(path: Path, text: str) -> None:
    allowed = {OUT_PROMPTS, OUT_RETRIEVAL, OUT_SUMMARY, OUT_CONTEXT}
    require(path in allowed, f"Refusing out-of-scope preflight write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".maxin4352.tmp")
    require(not temporary.exists(), f"Temporary output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_runner():
    path = SRC / "06_batch_run_dynamic_k3_v1.py"
    spec = importlib.util.spec_from_file_location("dynamic_k3_preflight_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot_path(model_id: str, revision: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    require(path.is_dir(), f"Missing local tokenizer snapshot: {path}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        require((path / name).is_file(), f"Incomplete tokenizer snapshot: {path / name}")
    return path


def percentile(values: list[int], fraction: float) -> float:
    require(values, "Cannot compute percentile of empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def selection_from_ranked(
    ranked_pairs: list[tuple[int, float]],
    examples: list[dict[str, Any]],
    *,
    testcase: dict[str, Any],
    leakage_guard: LeakageGuard,
) -> tuple[list[dict[str, Any]], list[float], Counter[str]]:
    selected: list[dict[str, Any]] = []
    scores: list[float] = []
    seen_ids: set[str] = set()
    reasons: Counter[str] = Counter()
    for index_position, score in ranked_pairs:
        if index_position < 0:
            continue
        example = examples[index_position]
        reason = _filter_reason(
            example,
            target_id=str(testcase.get("id", "")),
            target_question=str(testcase.get("question", "")),
            target_db_id=str(testcase.get("db_id", "")),
            same_db_only=False,
            allow_overlap=False,
            leakage_guard=leakage_guard,
        )
        if reason:
            reasons[reason] += 1
            continue
        example_id = str(example.get("id", "")).strip()
        if example_id in seen_ids:
            reasons["duplicate_example_id"] += 1
            continue
        seen_ids.add(example_id)
        selected.append(example)
        scores.append(float(score))
        if len(selected) == 3:
            break
    return selected, scores, reasons


def build_retrieval_selections(testcases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prior_rows: list[dict[str, str]] = []
    if OUT_RETRIEVAL.is_file():
        with OUT_RETRIEVAL.open(newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedding_model = str(manifest.get("model") or manifest.get("embedding_model"))
    require(embedding_model == "BAAI/bge-large-en-v1.5", f"Unexpected embedding model: {embedding_model}")
    examples = load_jsonl(INDEX_DIR / "metadata.jsonl")
    require(len(examples) == 6960, f"Expected retrieval pool 6960, found {len(examples)}")
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    require(index.ntotal == 6960, f"Expected index.ntotal=6960, found {index.ntotal}")
    normalize = bool(manifest.get("normalize", True))
    query_prefix = str(manifest.get("query_prefix", ""))
    apply_prefix = bool(manifest.get("apply_query_prefix_to_queries", bool(query_prefix)))
    model = SentenceTransformer(embedding_model, local_files_only=True)
    query_texts = [
        (query_prefix + str(row["question"]))
        if apply_prefix and query_prefix and not str(row["question"]).startswith(query_prefix)
        else str(row["question"])
        for row in testcases
    ]
    embeddings = model.encode(
        query_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).astype(np.float32)
    search_scores, search_indices = index.search(embeddings, 60)
    leakage_guard = LeakageGuard.from_testcases_path(TESTCASES)
    top_selections: list[dict[str, Any]] = []
    structure_selections: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for case_index, testcase in enumerate(testcases):
        ranked_pairs = [
            (int(index_position), float(score))
            for index_position, score in zip(search_indices[case_index], search_scores[case_index])
        ]
        top_examples, top_scores, top_reasons = selection_from_ranked(
            ranked_pairs,
            examples,
            testcase=testcase,
            leakage_guard=leakage_guard,
        )
        require(len(top_examples) == 3, f"Top-3 selection incomplete for {testcase['id']}")

        structure_candidates: list[dict[str, Any]] = []
        structure_filtered: Counter[str] = Counter()
        for rank, (index_position, score) in enumerate(ranked_pairs[:10], start=1):
            require(index_position >= 0, f"Invalid Top-10 index for {testcase['id']}")
            example = examples[index_position]
            reason = _filter_reason(
                example,
                target_id=str(testcase.get("id", "")),
                target_question=str(testcase.get("question", "")),
                target_db_id=str(testcase.get("db_id", "")),
                same_db_only=False,
                allow_overlap=False,
                leakage_guard=leakage_guard,
            )
            if reason:
                structure_filtered[reason] += 1
                continue
            adjustment, details = structure_rerank_adjustment(
                question=str(testcase["question"]),
                target_schema=str(testcase.get("schema_prompt", "")),
                candidate_sql=str(example.get("gold_sql", "")),
                candidate_schema=str(example.get("schema_prompt", "")),
                max_adjustment=0.08,
            )
            structure_candidates.append(
                {
                    "rank": rank,
                    "id": str(example.get("id", "")),
                    "example": example,
                    "bge_similarity": float(score),
                    "structure_adjustment": float(adjustment),
                    "final_score": float(score) + float(adjustment),
                    "details": details,
                }
            )
        structure_candidates.sort(
            key=lambda item: (
                -item["final_score"],
                -item["bge_similarity"],
                item["rank"],
                item["id"],
            )
        )
        distinct_structure: list[dict[str, Any]] = []
        seen_structure_ids: set[str] = set()
        for candidate in structure_candidates:
            if candidate["id"] in seen_structure_ids:
                structure_filtered["duplicate_example_id"] += 1
                continue
            seen_structure_ids.add(candidate["id"])
            distinct_structure.append(candidate)
            if len(distinct_structure) == 3:
                break
        require(len(distinct_structure) == 3, f"Structure Top-3 incomplete for {testcase['id']}")

        top = {
            "examples": top_examples,
            "ids": [str(item["id"]) for item in top_examples],
            "scores": top_scores,
            "filtered_reasons": dict(top_reasons),
        }
        structure = {
            "examples": [item["example"] for item in distinct_structure],
            "ids": [item["id"] for item in distinct_structure],
            "scores": [item["bge_similarity"] for item in distinct_structure],
            "adjustments": [item["structure_adjustment"] for item in distinct_structure],
            "final_scores": [item["final_score"] for item in distinct_structure],
            "bge_ranks": [item["rank"] for item in distinct_structure],
            "filtered_reasons": dict(structure_filtered),
        }
        top_selections.append(top)
        structure_selections.append(structure)
        all_examples = top_examples + structure["examples"]
        leakage_reasons = [
            leakage_guard.leakage_reasons(example)
            for example in all_examples
        ]
        retrieval_rows.append(
            {
                "case_id": testcase["id"],
                "db_id": testcase["db_id"],
                "top3_demo_ids": json.dumps(top["ids"], ensure_ascii=False),
                "top3_scores": json.dumps(top["scores"]),
                "top3_score_min": min(top["scores"]),
                "top3_score_mean": statistics.fmean(top["scores"]),
                "top3_score_max": max(top["scores"]),
                "top3_distinct": int(len(set(top["ids"])) == 3),
                "structure_top3_demo_ids": json.dumps(structure["ids"], ensure_ascii=False),
                "structure_original_bge_scores": json.dumps(structure["scores"]),
                "structure_adjustments": json.dumps(structure["adjustments"]),
                "structure_final_scores": json.dumps(structure["final_scores"]),
                "structure_original_bge_ranks": json.dumps(structure["bge_ranks"]),
                "structure_score_min": min(structure["scores"]),
                "structure_score_mean": statistics.fmean(structure["scores"]),
                "structure_score_max": max(structure["scores"]),
                "structure_distinct": int(len(set(structure["ids"])) == 3),
                "target_or_dev_leakage": int(any(leakage_reasons)),
                "leakage_reasons": json.dumps(leakage_reasons, ensure_ascii=False),
                "top3_filtered_reasons": json.dumps(top["filtered_reasons"], sort_keys=True),
                "structure_filtered_reasons": json.dumps(structure["filtered_reasons"], sort_keys=True),
                "status": "PASS",
            }
        )

    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(retrieval_rows[0]))
    writer.writeheader()
    writer.writerows(retrieval_rows)
    prior_identity = {
        "available": bool(prior_rows),
        "rows": len(prior_rows),
        "case_order_matches": 0,
        "top3_demo_id_matches": 0,
        "top3_score_matches": 0,
        "structure_demo_id_matches": 0,
        "structure_score_matches": 0,
        "status": "NOT_AVAILABLE",
    }
    if prior_rows:
        require(len(prior_rows) == len(retrieval_rows), "Prior retrieval preflight is incomplete")
        for prior, current in zip(prior_rows, retrieval_rows):
            prior_identity["case_order_matches"] += int(prior["case_id"] == current["case_id"])
            prior_identity["top3_demo_id_matches"] += int(
                prior["top3_demo_ids"] == current["top3_demo_ids"]
            )
            prior_identity["top3_score_matches"] += int(
                prior["top3_scores"] == current["top3_scores"]
            )
            prior_identity["structure_demo_id_matches"] += int(
                prior["structure_top3_demo_ids"] == current["structure_top3_demo_ids"]
            )
            prior_identity["structure_score_matches"] += int(
                prior["structure_original_bge_scores"]
                == current["structure_original_bge_scores"]
            )
        prior_identity["status"] = (
            "PASS"
            if all(
                prior_identity[key] == len(retrieval_rows)
                for key in (
                    "case_order_matches",
                    "top3_demo_id_matches",
                    "top3_score_matches",
                    "structure_demo_id_matches",
                    "structure_score_matches",
                )
            )
            else "FAIL"
        )
    write_scoped(OUT_RETRIEVAL, output.getvalue())
    return top_selections, structure_selections, {
        "embedding_model": embedding_model,
        "pool_size": len(examples),
        "index_ntotal": index.ntotal,
        "index_sha256": sha256_file(INDEX_DIR / "index.faiss"),
        "metadata_sha256": sha256_file(INDEX_DIR / "metadata.jsonl"),
        "manifest_sha256": sha256_file(INDEX_DIR / "manifest.json"),
        "rows": len(retrieval_rows),
        "leakage_rows": sum(int(row["target_or_dev_leakage"]) for row in retrieval_rows),
        "invalid_distinct_rows": sum(
            not (int(row["top3_distinct"]) and int(row["structure_distinct"]))
            for row in retrieval_rows
        ),
        "prior_maxin2048_selection_identity": prior_identity,
    }


def load_tokenizers() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tokenizers: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for model_key, (model_id, revision) in MODEL_REVISIONS.items():
        path = snapshot_path(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizers[model_key] = tokenizer
        provenance[model_key] = {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path": str(path),
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(path / "tokenizer_config.json"),
        }
    return tokenizers, provenance


def validate_context_capacity(
    tokenizers: dict[str, Any],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    required_total = max_input_tokens + max_new_tokens
    models: list[dict[str, Any]] = []
    for model_key, (model_id, revision) in MODEL_REVISIONS.items():
        path = snapshot_path(model_id, revision)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config") or {}
        capacity = text_config.get("max_position_embeddings") or config.get(
            "max_position_embeddings"
        )
        require(isinstance(capacity, int) and capacity > 0, f"Missing context capacity: {model_id}")
        tokenizer_limit = int(getattr(tokenizers[model_key], "model_max_length", 0) or 0)
        models.append(
            {
                "model_key": model_key,
                "model_id": model_id,
                "revision": revision,
                "config_path": str(path / "config.json"),
                "config_sha256": sha256_file(path / "config.json"),
                "model_type": config.get("model_type"),
                "max_position_embeddings": config.get("max_position_embeddings"),
                "text_config_max_position_embeddings": text_config.get(
                    "max_position_embeddings"
                ),
                "resolved_context_capacity": capacity,
                "tokenizer_model_max_length": tokenizer_limit,
                "required_total_tokens": required_total,
                "capacity_sufficient": capacity >= required_total,
            }
        )
    result = {
        "status": "PASS" if all(row["capacity_sufficient"] for row in models) else "FAIL",
        "configured_max_input_tokens": max_input_tokens,
        "configured_max_new_tokens": max_new_tokens,
        "required_total_tokens": required_total,
        "runner_semantics": "input is truncated independently to max_input_tokens; generate receives max_new_tokens",
        "models": models,
    }
    write_scoped(OUT_CONTEXT, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    require(Path(sys.executable).absolute() == EXPECTED_INTERPRETER, (
        f"Authoritative interpreter required: expected={EXPECTED_INTERPRETER}, actual={sys.executable}"
    ))
    runner = load_runner()
    testcases = load_jsonl(TESTCASES)
    require(len(testcases) == 1032, f"Expected 1032 testcases, found {len(testcases)}")
    require(len({str(row['id']) for row in testcases}) == 1032, "Duplicate testcase IDs")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    require(len(matrix_rows) == 36, f"Expected 36 configs, found {len(matrix_rows)}")
    for row in matrix_rows:
        config_path = ROOT / row["new_k3_config"]
        require("maxin4352" in config_path.name, f"Config name lacks maxin4352: {config_path}")
        require(sha256_file(config_path) == row["config_sha256"], f"Config hash drift: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        require(config.get("max_input_tokens") == 4352, f"Wrong max_input_tokens: {config_path}")
        require(config.get("max_new_tokens") == 256, f"Wrong max_new_tokens: {config_path}")

    top_selections, structure_selections, retrieval_summary = build_retrieval_selections(testcases)
    tokenizers, tokenizer_provenance = load_tokenizers()
    context_capacity = validate_context_capacity(
        tokenizers,
        max_input_tokens=4352,
        max_new_tokens=256,
    )
    require(context_capacity["status"] == "PASS", "Model context capacity is insufficient")
    prior_prompt_rows: list[dict[str, str]] = []
    if OUT_PROMPTS.is_file():
        with OUT_PROMPTS.open(newline="", encoding="utf-8") as handle:
            prior_prompt_rows = list(csv.DictReader(handle))
    prompt_rows: list[dict[str, Any]] = []
    stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "tokens": [],
            "truncations": 0,
            "fallbacks": 0,
            "fewshot": 0,
            "invalid": 0,
        }
    )

    for matrix_row in matrix_rows:
        config_path = ROOT / matrix_row["new_k3_config"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_key = matrix_row["model_key"]
        role = matrix_row["role"]
        condition = matrix_row["condition"]
        tokenizer = tokenizers[model_key]
        resolved_system_prompt, _source, _path, system_prompt_sha = runner.resolve_system_prompt(
            project_root=ROOT,
            system_prompt_variant=config["system_prompt_variant"],
            system_prompt_path=config.get("system_prompt_path"),
        )
        add_special_tokens = config["prompt_format"] != runner.LLAMA32_NATIVE_CHAT_FORMAT
        native_assistant_prefix = (
            runner.llama32_assistant_generation_prefix(tokenizer)
            if config["prompt_format"] == runner.LLAMA32_NATIVE_CHAT_FORMAT
            else None
        )

        for index, testcase in enumerate(testcases):
            source = CONDITION_SOURCE[condition]
            selection = top_selections[index] if source == "top" else structure_selections[index]
            similarities = [float(value) for value in selection["scores"]]
            threshold = THRESHOLDS.get(condition)
            gate_set_score = min(similarities)
            fallback = threshold is not None and gate_set_score < threshold
            actual_k = 0 if fallback else 3
            schema = runner.normalize_schema_text(str(testcase.get("schema_prompt", "")))
            question = str(testcase["question"])
            if fallback:
                prompt = runner.build_prompt(
                    schema,
                    question,
                    config["llm"],
                    tokenizer,
                    prompt_format=config["prompt_format"],
                    system_instruction=resolved_system_prompt,
                )
                demo_ids: list[str] = []
            else:
                prompt = runner.build_prompt_schema_fewshot(
                    schema,
                    question,
                    selection["examples"],
                    config["llm"],
                    tokenizer,
                    prompt_format=config["prompt_format"],
                    system_instruction=resolved_system_prompt,
                    example_schema_mode=config["fewshot_example_schema_mode"],
                    example_mode=config["fewshot_example_mode"],
                )
                demo_ids = list(selection["ids"])
            token_ids = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )["input_ids"]
            token_count = len(token_ids)
            max_input = int(config["max_input_tokens"])
            would_truncate = token_count > max_input
            qwen_prefix_ok = (
                config["prompt_format"] != "qwen_sqlctx_chatml"
                or prompt.endswith(runner.V2_SQLCTX_ASSISTANT_PREFIX)
            )
            llama_prefix_ok = (
                config["prompt_format"] != runner.LLAMA32_NATIVE_CHAT_FORMAT
                or bool(native_assistant_prefix and prompt.endswith(native_assistant_prefix))
            )
            invalid_reasons: list[str] = []
            if actual_k not in {0, 3}:
                invalid_reasons.append("unexpected_actual_k")
            if actual_k == 3 and len(set(demo_ids)) != 3:
                invalid_reasons.append("duplicate_or_missing_demo")
            if not question or question not in prompt:
                invalid_reasons.append("target_question_missing")
            if not schema:
                invalid_reasons.append("empty_schema")
            if not qwen_prefix_ok or not llama_prefix_ok:
                invalid_reasons.append("assistant_prefix_invalid")
            if "<think" in prompt.lower() or "</think" in prompt.lower():
                invalid_reasons.append("think_token_present")
            if config["prompt_format"] == "qwen_sqlctx_chatml" and "<|start_header_id|>" in prompt:
                invalid_reasons.append("llama_token_in_qwen_prompt")
            if config["prompt_format"] == runner.LLAMA32_NATIVE_CHAT_FORMAT and "<|im_start|>" in prompt:
                invalid_reasons.append("qwen_token_in_llama_prompt")
            key = (model_key, role, condition)
            group = stats[key]
            group["tokens"].append(token_count)
            group["truncations"] += int(would_truncate)
            group["fallbacks"] += int(fallback)
            group["fewshot"] += int(not fallback)
            group["invalid"] += int(bool(invalid_reasons))
            prompt_rows.append(
                {
                    "model_key": model_key,
                    "model_line": matrix_row["model_line"],
                    "role": role,
                    "condition": condition,
                    "config_path": matrix_row["new_k3_config"],
                    "config_sha256": matrix_row["config_sha256"],
                    "case_id": testcase["id"],
                    "db_id": testcase["db_id"],
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_tokens": token_count,
                    "max_input_tokens": max_input,
                    "would_truncate": int(would_truncate),
                    "truncated_tokens": max(0, token_count - max_input),
                    "requested_k": 3,
                    "actual_k": actual_k,
                    "fallback": int(fallback),
                    "demo_ids": json.dumps(demo_ids, ensure_ascii=False),
                    "similarities": json.dumps(similarities),
                    "gate_set_score": gate_set_score if threshold is not None else "",
                    "gate_threshold": threshold if threshold is not None else "",
                    "target_question_present": int(question in prompt),
                    "target_schema_nonempty": int(bool(schema)),
                    "assistant_prefix_ok": int(qwen_prefix_ok and llama_prefix_ok),
                    "invalid_reasons": json.dumps(invalid_reasons),
                    "status": "PASS" if not invalid_reasons else "FAIL",
                }
            )

    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(prompt_rows[0]))
    writer.writeheader()
    writer.writerows(prompt_rows)
    prior_prompt_identity = {
        "available": bool(prior_prompt_rows),
        "rows": len(prior_prompt_rows),
        "case_key_matches": 0,
        "prompt_sha256_matches": 0,
        "demo_id_matches": 0,
        "similarity_matches": 0,
        "actual_k_matches": 0,
        "fallback_matches": 0,
        "status": "NOT_AVAILABLE",
    }
    if prior_prompt_rows:
        require(len(prior_prompt_rows) == len(prompt_rows), "Prior prompt preflight is incomplete")
        for prior, current in zip(prior_prompt_rows, prompt_rows):
            prior_key = (prior["model_key"], prior["role"], prior["condition"], prior["case_id"])
            current_key = (
                current["model_key"],
                current["role"],
                current["condition"],
                current["case_id"],
            )
            prior_prompt_identity["case_key_matches"] += int(prior_key == current_key)
            prior_prompt_identity["prompt_sha256_matches"] += int(
                prior["prompt_sha256"] == current["prompt_sha256"]
            )
            prior_prompt_identity["demo_id_matches"] += int(
                prior["demo_ids"] == current["demo_ids"]
            )
            prior_prompt_identity["similarity_matches"] += int(
                prior["similarities"] == current["similarities"]
            )
            prior_prompt_identity["actual_k_matches"] += int(
                prior["actual_k"] == str(current["actual_k"])
            )
            prior_prompt_identity["fallback_matches"] += int(
                prior["fallback"] == str(current["fallback"])
            )
        prior_prompt_identity["status"] = (
            "PASS"
            if all(
                prior_prompt_identity[key] == len(prompt_rows)
                for key in (
                    "case_key_matches",
                    "prompt_sha256_matches",
                    "demo_id_matches",
                    "similarity_matches",
                    "actual_k_matches",
                    "fallback_matches",
                )
            )
            else "FAIL"
        )
    write_scoped(OUT_PROMPTS, output.getvalue())

    group_summaries = []
    for (model_key, role, condition), values in sorted(stats.items()):
        tokens = values["tokens"]
        group_summaries.append(
            {
                "model_key": model_key,
                "role": role,
                "condition": condition,
                "cases": len(tokens),
                "minimum": min(tokens),
                "mean": statistics.fmean(tokens),
                "median": statistics.median(tokens),
                "p95": percentile(tokens, 0.95),
                "p99": percentile(tokens, 0.99),
                "maximum": max(tokens),
                "needed_input_limit_for_zero_truncation": max(tokens),
                "prompt_truncations": values["truncations"],
                "fewshot_cases": values["fewshot"],
                "fallback_cases": values["fallbacks"],
                "invalid_prompts": values["invalid"],
            }
        )
    total_truncations = sum(item["prompt_truncations"] for item in group_summaries)
    total_invalid = sum(item["invalid_prompts"] for item in group_summaries)
    feasibility = (
        "PASS"
        if total_truncations == 0
        and total_invalid == 0
        and context_capacity["status"] == "PASS"
        and retrieval_summary["prior_maxin2048_selection_identity"]["status"] in {"PASS", "NOT_AVAILABLE"}
        and prior_prompt_identity["status"] in {"PASS", "NOT_AVAILABLE"}
        else "BLOCKED-BY-PROMPT-TRUNCATION"
    )
    summary = {
        "status": feasibility,
        "full_runs_released": feasibility == "PASS",
        "full_runs_started": False,
        "authoritative_interpreter": str(EXPECTED_INTERPRETER),
        "sys_executable": sys.executable,
        "testcases": len(testcases),
        "configs": len(matrix_rows),
        "prompt_rows": len(prompt_rows),
        "total_prompt_truncations": total_truncations,
        "total_invalid_prompts": total_invalid,
        "configured_max_input_tokens": 4352,
        "configured_max_new_tokens": 256,
        "maximum_prompt_tokens": max(item["maximum"] for item in group_summaries),
        "retrieval": retrieval_summary,
        "prior_maxin2048_prompt_identity": prior_prompt_identity,
        "model_context_capacity": context_capacity,
        "tokenizers": tokenizer_provenance,
        "groups": group_summaries,
        "hard_stop_triggered": total_truncations > 0 or total_invalid > 0,
        "methodological_options_if_blocked": [
            "Keep the configured input limit and accept truncation",
            "Use a higher max_input_tokens limit",
            "Use a more compact demonstration representation",
            "Run a controlled k1/k3 matrix with the same higher input limit",
        ] if total_truncations > 0 else [],
        "source_files": {
            "config_matrix": str(MATRIX.relative_to(ROOT)),
            "config_matrix_sha256": sha256_file(MATRIX),
            "testcases": str(TESTCASES.relative_to(ROOT)),
            "testcases_sha256": sha256_file(TESTCASES),
            "runner": "src/06_batch_run_dynamic_k3_v1.py",
            "runner_sha256": sha256_file(SRC / "06_batch_run_dynamic_k3_v1.py"),
            "retrieval_utils": "src/retrieval_utils_dynamic_k3_v1.py",
            "retrieval_utils_sha256": sha256_file(SRC / "retrieval_utils_dynamic_k3_v1.py"),
        },
    }
    write_scoped(OUT_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": feasibility,
        "configs": len(matrix_rows),
        "prompt_rows": len(prompt_rows),
        "total_prompt_truncations": total_truncations,
        "total_invalid_prompts": total_invalid,
        "maximum_prompt_tokens": max(item["maximum"] for item in group_summaries),
        "full_runs_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
