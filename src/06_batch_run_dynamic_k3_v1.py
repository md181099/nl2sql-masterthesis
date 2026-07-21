#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import argparse

from chat_formatting import build_nl2sql_messages, render_messages
from config import get_param, load_config
from llm_client import LLMClient
from llama32_native_chat import (
    LLAMA32_NATIVE_CHAT_FORMAT,
    configure_llama32_padding,
    llama32_assistant_generation_prefix,
    llama32_generation_stop_token_ids,
    render_llama32_native_chat,
)
from logging_utils import setup_logging
from prompt_presets import CURRENT_SYSTEM_PROMPT, V2_SQLCTX_SYSTEM_PROMPT, resolve_system_prompt
from retrieval_utils_dynamic_k3_v1 import (
    FaissFewShotRetriever,
    FewShotSelection,
    LeakageGuard,
    StaticFewShotRetriever,
    load_retrieval_pool,
    selection_to_csv_values,
    selection_to_trace,
    sqlaware_structure_bonus,
)


DEFAULT_LLM = "llama32_1b"
NON_SELECT_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "pragma",
    "attach",
    "detach",
    "vacuum",
    "begin",
    "commit",
    "rollback",
    "replace",
    "truncate",
)
ANY_SQL_START_RE = re.compile(
    r"(?im)(?:^|;|\n)\s*(select|" + "|".join(NON_SELECT_KEYWORDS) + r")\b"
)
NON_SELECT_START_RE = re.compile(
    r"(?im)(?:^|;|\n)\s*(" + "|".join(NON_SELECT_KEYWORDS) + r")\b"
)
RESTART_MARKER_RE = re.compile(r"(?im)(?:^|\n)\s*(?:question|sql|example)\s*:")
CHAT_TOKEN_RE = re.compile(r"<\|(?:user|assistant|system)\|>|\|(?:user|assistant|system)\|>", flags=re.IGNORECASE)
MARKDOWN_FENCE_RE = re.compile(r"```(?:sql)?", flags=re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")
THINK_CLOSE_RE = re.compile(r"(?i)</think\b[^>]*>")
THINK_TAG_RE = re.compile(r"(?i)</?think\b[^>]*>")
IM_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end)\|>", flags=re.IGNORECASE)
WITH_CTE_START_RE = re.compile(
    r"(?is)^with\s+(?:recursive\s+)?[A-Za-z_][A-Za-z0-9_]*\s+as\s*\("
)
EXPLANATION_MARKER_RE = re.compile(
    r"(?is)\b(?:but note|however|then|let's|we need|the query|the question)\b"
)
FRAGMENT_BAD_PATTERN_RE = re.compile(r"(?is)(?:=\s*;|\band\s*;|\bor\s*;|,\s*;)")
ENDS_WITH_OPERATOR_RE = re.compile(r"(?is)(?:=|\band\b|\bor\b|,|\+|\-|\*|/)\s*;?\s*$")
SET_OPERATOR_TAIL_RE = re.compile(r"(?is)\b(?:intersect|except|union(?:\s+all)?)\s*$")

logger = logging.getLogger(__name__)

QWEN_SQLCTX_CHATML_FORMAT = "qwen_sqlctx_chatml"
QWEN_SQLCTX_LEGACY_PROMPT_FORMATS = {
    "qwen_v2_sqlctx",
    "qwen_v2_sqlctx_full_chat",
    "v2_prompt_completion_chatml",
}
QWEN_SQLCTX_PROMPT_FORMATS = {QWEN_SQLCTX_CHATML_FORMAT, *QWEN_SQLCTX_LEGACY_PROMPT_FORMATS}
V2_PROMPT_FORMATS = QWEN_SQLCTX_PROMPT_FORMATS
LLAMA32_INSTRUCT_SQLCTX_FORMAT = "llama32_instruct_sqlctx"
LLAMA32_V2_PROMPT_FORMATS = {"llama32_v2_sqlctx", LLAMA32_INSTRUCT_SQLCTX_FORMAT}
V2_SQLCTX_PROMPT_FORMATS = V2_PROMPT_FORMATS | LLAMA32_V2_PROMPT_FORMATS | {LLAMA32_NATIVE_CHAT_FORMAT}
V2_SQLCTX_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
LLAMA32_V2_SQLCTX_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
K3_RUNNER_VERSION = "dynamic_k3_v1"
EXPECTED_MODEL_REVISIONS = {
    "qwen35_2b_base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    "llama32_3b_instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "qwen35_9b_base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
}
@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    generation_time_seconds: float | None = None
    tokens_per_second: float | None = None


@dataclass
class FewShotGateDecision:
    enabled: bool
    mode: str
    score: float | None
    threshold: float | None
    decision: str
    reason: str
    retrieval_similarity: float | None
    selected_example_id: str
    number_of_retrieved_candidates: int
    debug: dict[str, Any]
    retrieval_similarities: tuple[float, ...] = ()
    retrieval_similarity_min: float | None = None
    retrieval_similarity_max: float | None = None
    retrieval_similarity_mean: float | None = None
    score_semantics: str = "first_selected_bge_similarity"


def _format_duration_hms(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_model_snapshot_provenance(
    model_id: str,
    expected_revision: str,
) -> dict[str, Any]:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = "models--" + model_id.replace("/", "--")
    snapshot_dir = Path(HF_HUB_CACHE) / repo_dir / "snapshots" / expected_revision
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (snapshot_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete local model snapshot {snapshot_dir}: missing={missing}"
        )
    return {
        "model_id": model_id,
        "revision": expected_revision,
        "snapshot_path": str(snapshot_dir),
        "file_sha256": {
            name: _sha256_file(snapshot_dir / name)
            for name in required
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def normalize_schema_text(schema_txt: str) -> str:
    lines = schema_txt.strip().splitlines()
    if lines and lines[0].strip().lower() == "database schema:":
        lines = lines[1:]
    return "\n".join(lines).strip()


def normalize_question_for_overlap(question: str) -> str:
    return " ".join(question.strip().lower().split())


def normalize_sql_for_overlap(sql: str) -> str:
    return " ".join(ensure_semicolon(sql).strip().lower().split())


def build_target_block(gold_sql: str) -> str:
    gold_sql = ensure_semicolon(gold_sql)
    return f"```sql\n{gold_sql}\n```"


def sql_tokens(sql: str) -> list[str]:
    return [
        t.lower()
        for t in re.findall(
            r"'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<=|>=|<>|!=|==|[-+*/(),.;=<>]",
            sql,
        )
    ]


def normalize_sql_for_exact(sql: str) -> str:
    return " ".join(sql_tokens(sql))


def char_accuracy(pred_sql: str, gold_sql: str) -> float:
    n = max(len(pred_sql), len(gold_sql))
    if n == 0:
        return 1.0
    same = sum(
        1 for i in range(n)
        if i < len(pred_sql) and i < len(gold_sql) and pred_sql[i] == gold_sql[i]
    )
    return same / n


def token_accuracy(pred_sql: str, gold_sql: str) -> float:
    pred_toks = sql_tokens(pred_sql)
    gold_toks = sql_tokens(gold_sql)
    n = max(len(pred_toks), len(gold_toks))
    if n == 0:
        return 1.0
    same = sum(
        1 for i in range(n)
        if i < len(pred_toks) and i < len(gold_toks) and pred_toks[i] == gold_toks[i]
    )
    return same / n


def compute_target_perplexity(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    max_length: int,
) -> float | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    if not input_ids:
        return None

    overflow = max(0, len(input_ids) - max_length)
    if overflow > 0:
        input_ids = input_ids[overflow:]
    label_start = max(0, len(prompt_ids) - overflow)
    if label_start >= len(input_ids):
        return None

    labels = [-100] * label_start + input_ids[label_start:]
    device = model.device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attn_tensor = torch.ones_like(input_tensor, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_tensor,
            attention_mask=attn_tensor,
            labels=label_tensor,
        )
    return float(torch.exp(out.loss).item())


def _slice_generated_token_ids(output_ids: Any, prompt_width: int) -> Any:
    return output_ids[prompt_width:]


def _decode_single_generation(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    add_special_tokens: bool = True,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
) -> str:
    return _decode_single_generation_with_metrics(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        add_special_tokens=add_special_tokens,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
    ).text


def _decode_single_generation_with_metrics(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    add_special_tokens: bool = True,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
) -> GenerationResult:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=add_special_tokens,
    ).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        generation_start = time.perf_counter()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if eos_token_ids is not None:
            generation_kwargs["eos_token_id"] = eos_token_ids
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )
        generation_time_seconds = time.perf_counter() - generation_start

    gen_ids = _slice_generated_token_ids(output[0], prompt_len)
    completion_tokens = int(len(gen_ids))
    tokens_per_second = (
        completion_tokens / generation_time_seconds
        if generation_time_seconds > 0
        else None
    )
    return GenerationResult(
        text=tokenizer.decode(gen_ids, skip_special_tokens=True),
        prompt_tokens=prompt_len,
        completion_tokens=completion_tokens,
        total_tokens=prompt_len + completion_tokens,
        reasoning_tokens=None,
        generation_time_seconds=generation_time_seconds,
        tokens_per_second=tokens_per_second,
    )


def _decode_batch_generation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
    add_special_tokens: bool = True,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
) -> list[str]:
    return [
        result.text
        for result in _decode_batch_generation_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            add_special_tokens=add_special_tokens,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
        )
    ]


def _decode_batch_generation_with_metrics(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
    add_special_tokens: bool = True,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
) -> list[GenerationResult]:
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=add_special_tokens,
        ).to(model.device)
        input_width = int(inputs["input_ids"].shape[1])
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        with torch.no_grad():
            generation_start = time.perf_counter()
            generation_kwargs: dict[str, Any] = {
                "position_ids": position_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if eos_token_ids is not None:
                generation_kwargs["eos_token_id"] = eos_token_ids
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            output = model.generate(
                **inputs,
                **generation_kwargs,
            )
            chunk_generation_time_seconds = time.perf_counter() - generation_start
        per_case_generation_time_seconds = (
            chunk_generation_time_seconds / len(prompts)
            if prompts
            else None
        )

        results: list[GenerationResult] = []
        for i in range(len(prompts)):
            gen_ids = _slice_generated_token_ids(output[i], input_width)
            prompt_tokens = int(attention_mask[i].sum().item())
            completion_tokens = int(len(gen_ids))
            tokens_per_second = (
                completion_tokens / per_case_generation_time_seconds
                if per_case_generation_time_seconds and per_case_generation_time_seconds > 0
                else None
            )
            results.append(
                GenerationResult(
                    text=tokenizer.decode(gen_ids, skip_special_tokens=True),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    reasoning_tokens=None,
                    generation_time_seconds=per_case_generation_time_seconds,
                    tokens_per_second=tokens_per_second,
                )
            )
        return results
    except Exception as exc:
        logger.warning(
            "Batch generation failed for chunk_size=%s; falling back to single-sample generation for this chunk: %s",
            len(prompts),
            repr(exc),
        )
    finally:
        tokenizer.padding_side = old_padding_side

    return [
        _decode_single_generation_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            add_special_tokens=add_special_tokens,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
        )
        for prompt in prompts
    ]


def _iter_chunks(items: list[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _split_first_statement(sql_text: str) -> tuple[str, str, bool]:
    """
    Split into first SQL statement + remainder.
    Statement boundary is the first semicolon outside quotes.
    """
    i = 0
    n = len(sql_text)
    in_single = False
    in_double = False
    while i < n:
        ch = sql_text[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < n and sql_text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < n and sql_text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return sql_text[: i + 1], sql_text[i + 1 :], True
        i += 1
    return sql_text, "", False


def _candidate_query_starts(text: str) -> list[int]:
    starts: list[int] = []
    # SELECT must be followed by whitespace, so instruction text like "SELECT." is ignored.
    for m in re.finditer(r"(?i)\bselect\s+", text):
        starts.append(m.start())

    # WITH is only considered when it looks like a real SQL CTE starter.
    for m in re.finditer(r"(?i)\bwith\b", text):
        fragment = text[m.start() :].lstrip()
        if WITH_CTE_START_RE.match(fragment):
            starts.append(m.start())
    return sorted(dict.fromkeys(starts))


def _clean_generated_sql_candidate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop full reasoning blocks first.
    text = THINK_BLOCK_RE.sub("\n", text)

    # If a stray closing </think> appears, keep only content after it.
    stray_close = THINK_CLOSE_RE.search(text)
    if stray_close:
        text = text[stray_close.end() :]

    # Remove remaining known artifact tags/tokens.
    text = THINK_TAG_RE.sub("\n", text)
    text = CHAT_TOKEN_RE.sub("\n", text)
    text = IM_TOKEN_RE.sub("\n", text)
    text = MARKDOWN_FENCE_RE.sub("\n", text)
    # Strip common lead-in labels to keep extraction tolerant to mild chatter.
    text = re.sub(r"(?im)^\s*(?:answer|response)\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*sql(?:\s+query)?\s*:\s*", "", text)
    return text.strip()


def _clean_generated_sql_candidate_robust(text: str) -> str:
    """
    Cleaning variant for robust_v2:
    - never hard-cuts at </think>
    - removes tag artifacts but preserves full text for multi-candidate ranking
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = THINK_BLOCK_RE.sub("\n", text)
    text = THINK_TAG_RE.sub("\n", text)
    text = CHAT_TOKEN_RE.sub("\n", text)
    text = IM_TOKEN_RE.sub("\n", text)
    text = MARKDOWN_FENCE_RE.sub("\n", text)
    text = re.sub(r"(?im)^\s*(?:answer|response)\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*sql(?:\s+query)?\s*:\s*", "", text)
    return text.strip()


def _truncate_at_restart_marker(text: str) -> str:
    match = RESTART_MARKER_RE.search(text)
    if match and match.start() > 0:
        return text[:match.start()].rstrip()
    return text


def _looks_like_readonly_sql(stmt: str) -> bool:
    stripped = stmt.lstrip()
    low = stripped.lower()
    if low.startswith("select"):
        return True
    if low.startswith("with"):
        # Accept only CTE-style WITH queries (avoid English "with ...").
        if WITH_CTE_START_RE.match(stripped) is None:
            return False
        # CTE must eventually feed a SELECT query.
        return re.search(r"(?i)\bselect\b", stripped) is not None
    return False


def _extract_single_select_from_text(
    text: str,
    *,
    allow_non_select_prefix: bool = False,
    reject_sql_after_first: bool = True,
) -> str | None:
    """
    Extract exactly one read-only SQL statement from text.
    Can optionally relax non-SELECT-prefix checks and post-query SQL checks for lenient recovery.
    """
    for start in _candidate_query_starts(text):
        prefix = text[:start]
        if not allow_non_select_prefix and NON_SELECT_START_RE.search(prefix):
            continue

        candidate = _truncate_at_restart_marker(text[start:])
        first_stmt, remainder, had_semicolon = _split_first_statement(candidate)
        stmt = first_stmt.strip()
        if not stmt:
            continue
        if not _looks_like_readonly_sql(stmt):
            continue

        if not had_semicolon:
            stmt = ensure_semicolon(stmt)

        # In strict mode, reject stacked SQL statements after first SELECT.
        if reject_sql_after_first and ANY_SQL_START_RE.search(remainder):
            continue

        return ensure_semicolon(stmt)
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_test_overlap_stats(traincases_path: Path, testcases: list[dict[str, Any]]) -> tuple[int, int, int]:
    if not traincases_path.exists():
        return (0, 0, 0)
    traincases = _load_jsonl(traincases_path)
    train_q = {normalize_question_for_overlap(str(x.get("question", ""))) for x in traincases}
    train_s = {normalize_sql_for_overlap(str(x.get("gold_sql", ""))) for x in traincases}
    train_pairs = {
        (
            normalize_question_for_overlap(str(x.get("question", ""))),
            normalize_sql_for_overlap(str(x.get("gold_sql", ""))),
        )
        for x in traincases
    }
    q_overlap = sum(
        1
        for t in testcases
        if normalize_question_for_overlap(str(t.get("question", ""))) in train_q
    )
    s_overlap = sum(
        1
        for t in testcases
        if normalize_sql_for_overlap(str(t.get("gold_sql", ""))) in train_s
    )
    both_overlap = sum(
        1
        for t in testcases
        if (
            normalize_question_for_overlap(str(t.get("question", ""))),
            normalize_sql_for_overlap(str(t.get("gold_sql", ""))),
        ) in train_pairs
    )
    return (q_overlap, s_overlap, both_overlap)


def extract_sql(text: str) -> str | None:
    """
    Robust SQL extraction from generated model text.
    Rules:
    - Prefer ```sql ... ``` code blocks.
    - Fallback to unlabeled code blocks.
    - Fallback to plain text.
    - Extract exactly one read-only query (SELECT / WITH ... SELECT).
    - Reject non-SELECT and stacked multi-statements.
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = []
    sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(sql_blocks)

    fenced_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    for block in fenced_blocks:
        if block.lstrip().lower().startswith("sql"):
            continue
        candidates.append(block)

    candidates.append(text)

    # Strict pass
    for raw in candidates:
        cleaned = _clean_generated_sql_candidate(raw)
        if not cleaned:
            continue
        sql = _extract_single_select_from_text(
            cleaned,
            allow_non_select_prefix=False,
            reject_sql_after_first=True,
        )
        if sql is not None:
            return sql

    # Lenient recovery pass: allow leading chatter/non-SELECT content and keep first plausible SELECT.
    for raw in candidates:
        cleaned = _clean_generated_sql_candidate(raw)
        if not cleaned:
            continue
        sql = _extract_single_select_from_text(
            cleaned,
            allow_non_select_prefix=True,
            reject_sql_after_first=False,
        )
        if sql is not None:
            return sql

    return None


def _is_hard_reject_candidate(stmt: str) -> bool:
    stmt_stripped = stmt.strip()
    if not stmt_stripped:
        return True
    if THINK_TAG_RE.search(stmt_stripped):
        return True
    if FRAGMENT_BAD_PATTERN_RE.search(stmt_stripped):
        return True
    if ENDS_WITH_OPERATOR_RE.search(stmt_stripped):
        return True
    return False


def _has_soft_contamination(stmt: str) -> bool:
    stmt_stripped = stmt.strip()
    if not stmt_stripped:
        return True
    if EXPLANATION_MARKER_RE.search(stmt_stripped):
        return True
    return False


def _collect_sql_candidates_from_source(
    source_text: str,
    *,
    source_name: str,
    base_offset: int,
    source_order: int,
) -> list[dict[str, Any]]:
    cleaned = _clean_generated_sql_candidate_robust(source_text)
    if not cleaned:
        return []

    candidates: list[dict[str, Any]] = []
    for start in _candidate_query_starts(cleaned):
        candidate_text = _truncate_at_restart_marker(cleaned[start:])
        first_stmt, _remainder, had_semicolon = _split_first_statement(candidate_text)
        stmt = first_stmt.strip()
        if not stmt:
            continue
        if not _looks_like_readonly_sql(stmt):
            continue

        sql_stmt = ensure_semicolon(stmt) if had_semicolon else stmt
        if _is_hard_reject_candidate(sql_stmt):
            continue
        contaminated = _has_soft_contamination(sql_stmt)

        candidates.append(
            {
                "sql": sql_stmt.strip(),
                "complete": bool(had_semicolon),
                "contaminated": contaminated,
                "source_name": source_name,
                "source_order": source_order,
                "start": start,
                "global_start": base_offset + start,
            }
        )
    return candidates


def _split_first_top_level_statement(sql_text: str) -> tuple[str, str, bool, bool]:
    """
    Split first statement by a semicolon that appears at top level
    (outside quotes and outside parentheses).
    Returns:
      (first_stmt, remainder, had_top_level_semicolon, is_structurally_balanced)
    """
    i = 0
    n = len(sql_text)
    in_single = False
    in_double = False
    paren_depth = 0
    unmatched_close = False
    while i < n:
        ch = sql_text[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < n and sql_text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < n and sql_text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                if paren_depth == 0:
                    unmatched_close = True
                else:
                    paren_depth -= 1
            elif ch == ";" and paren_depth == 0:
                balanced = (not unmatched_close) and (paren_depth == 0) and (not in_single) and (not in_double)
                return sql_text[: i + 1], sql_text[i + 1 :], True, balanced
        i += 1
    balanced = (not unmatched_close) and (paren_depth == 0) and (not in_single) and (not in_double)
    return sql_text, "", False, balanced


def _top_level_candidate_query_starts(text: str) -> list[int]:
    starts: list[int] = []
    i = 0
    n = len(text)
    in_single = False
    in_double = False
    paren_depth = 0

    def _is_ident_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < n:
        ch = text[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            if in_double and i + 1 < n and text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
            i += 1
            continue
        if not in_single and not in_double:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)

            if paren_depth == 0:
                # SELECT starts
                if text[i : i + 6].lower() == "select":
                    prev_ok = i == 0 or not _is_ident_char(text[i - 1])
                    next_ok = (i + 6) < n and text[i + 6].isspace()
                    if prev_ok and next_ok:
                        prefix = text[:i].rstrip()
                        if not SET_OPERATOR_TAIL_RE.search(prefix):
                            starts.append(i)
                    i += 6
                    continue

                # WITH starts (only CTE-style)
                if text[i : i + 4].lower() == "with":
                    prev_ok = i == 0 or not _is_ident_char(text[i - 1])
                    next_ok = (i + 4) < n and text[i + 4].isspace()
                    if prev_ok and next_ok:
                        fragment = text[i:]
                        if WITH_CTE_START_RE.match(fragment):
                            starts.append(i)
                    i += 4
                    continue
        i += 1
    return sorted(dict.fromkeys(starts))


def _collect_sql_candidates_from_source_v3(
    source_text: str,
    *,
    source_name: str,
    base_offset: int,
    source_order: int,
) -> list[dict[str, Any]]:
    cleaned = _clean_generated_sql_candidate_robust(source_text)
    if not cleaned:
        return []

    candidates: list[dict[str, Any]] = []
    for start in _top_level_candidate_query_starts(cleaned):
        candidate_text = _truncate_at_restart_marker(cleaned[start:])
        first_stmt, _remainder, had_semicolon, balanced = _split_first_top_level_statement(candidate_text)
        stmt = first_stmt.strip()
        if not stmt:
            continue
        if not _looks_like_readonly_sql(stmt):
            continue

        sql_stmt = ensure_semicolon(stmt) if had_semicolon else stmt
        if _is_hard_reject_candidate(sql_stmt):
            continue
        contaminated = _has_soft_contamination(sql_stmt)
        contains_set_op = re.search(r"(?i)\b(?:intersect|except|union)\b", stmt) is not None
        contains_subquery = re.search(r"(?is)\(\s*select\b", stmt) is not None

        candidates.append(
            {
                "sql": sql_stmt.strip(),
                "complete": bool(had_semicolon),
                "balanced": bool(balanced),
                "contaminated": contaminated,
                "contains_set_op": contains_set_op,
                "contains_subquery": contains_subquery,
                "source_name": source_name,
                "source_order": source_order,
                "start": start,
                "global_start": base_offset + start,
            }
        )
    return candidates


def extract_sql_robust_v2(text: str) -> str | None:
    """
    Optional robust extractor that ranks multiple SQL candidates.
    Keeps extraction purely text-based (no DB/gold-SQL usage).
    """
    if not text or not text.strip():
        return None

    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    sources: list[tuple[str, str, int, int]] = []
    source_idx = 0

    for match in re.finditer(r"```sql\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE):
        sources.append((match.group(1), "fenced_sql", match.start(1), source_idx))
        source_idx += 1

    close_match = THINK_CLOSE_RE.search(raw)
    if close_match:
        after = raw[close_match.end() :]
        before = raw[: close_match.start()]
        sources.append((after, "after_think", close_match.end(), source_idx))
        source_idx += 1
        sources.append((before, "before_think", 0, source_idx))
        source_idx += 1

    sources.append((raw, "full_text", 0, source_idx))

    candidates: list[dict[str, Any]] = []
    for source_text, source_name, base_offset, order in sources:
        candidates.extend(
            _collect_sql_candidates_from_source(
                source_text,
                source_name=source_name,
                base_offset=base_offset,
                source_order=order,
            )
        )

    if not candidates:
        return None

    # Deduplicate SQL text while preserving best-ranked candidate per SQL.
    dedup: dict[str, dict[str, Any]] = {}

    def _dedup_rank_key(c: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(c["complete"]),
            int(not c["contaminated"]),
            1 if c["source_name"] == "after_think" and c["complete"] and not c["contaminated"] else 0,
            int(c["global_start"]),
        )

    for c in candidates:
        norm_key = " ".join(c["sql"].lower().split())
        prev = dedup.get(norm_key)
        if prev is None or _dedup_rank_key(c) > _dedup_rank_key(prev):
            dedup[norm_key] = c

    unique_candidates = list(dedup.values())

    plausible = [c for c in unique_candidates if c["complete"] and not c["contaminated"]]
    if plausible:
        best = max(
            plausible,
            key=lambda c: (
                1 if c["source_name"] == "after_think" else 0,
                int(c["global_start"]),
                int(c["source_order"]),
            ),
        )
        return ensure_semicolon(best["sql"])

    complete_any = [c for c in unique_candidates if c["complete"]]
    if complete_any:
        best = max(complete_any, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
        return ensure_semicolon(best["sql"])

    incomplete_clean = [c for c in unique_candidates if not c["contaminated"]]
    if incomplete_clean:
        best = max(incomplete_clean, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
        return ensure_semicolon(best["sql"])

    best = max(unique_candidates, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
    return ensure_semicolon(best["sql"])


def extract_sql_robust_v3(text: str) -> str | None:
    """
    Optional robust extractor (v3):
    - text-only (no DB/gold usage)
    - keeps multi-source candidate collection
    - accepts only top-level SELECT/WITH starts
    - avoids subquery fragments and RHS set-operator fragments
    """
    if not text or not text.strip():
        return None

    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    sources: list[tuple[str, str, int, int]] = []
    source_idx = 0

    for match in re.finditer(r"```sql\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE):
        sources.append((match.group(1), "fenced_sql", match.start(1), source_idx))
        source_idx += 1

    close_match = THINK_CLOSE_RE.search(raw)
    if close_match:
        after = raw[close_match.end() :]
        before = raw[: close_match.start()]
        sources.append((after, "after_think", close_match.end(), source_idx))
        source_idx += 1
        sources.append((before, "before_think", 0, source_idx))
        source_idx += 1

    sources.append((raw, "full_text", 0, source_idx))

    candidates: list[dict[str, Any]] = []
    for source_text, source_name, base_offset, order in sources:
        candidates.extend(
            _collect_sql_candidates_from_source_v3(
                source_text,
                source_name=source_name,
                base_offset=base_offset,
                source_order=order,
            )
        )

    if not candidates:
        return None

    # Keep best structural evidence per normalized SQL.
    dedup: dict[str, dict[str, Any]] = {}

    source_priority = {
        "fenced_sql": 4,
        "full_text": 3,
        "after_think": 2,
        "before_think": 1,
    }

    def _rank_key(c: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
        return (
            int(c["complete"]),
            int(c["balanced"]),
            int(not c["contaminated"]),
            int(c["contains_set_op"]),
            int(c["contains_subquery"]),
            int(source_priority.get(c["source_name"], 0)),
            int(c["global_start"]),
        )

    for c in candidates:
        norm_key = " ".join(c["sql"].lower().split())
        prev = dedup.get(norm_key)
        if prev is None or _rank_key(c) > _rank_key(prev):
            dedup[norm_key] = c

    unique_candidates = list(dedup.values())

    def _pick_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return max(
            rows,
            key=lambda c: (
                int(c["contains_set_op"]),
                int(c["contains_subquery"]),
                int(source_priority.get(c["source_name"], 0)),
                int(c["global_start"]),
                int(c["source_order"]),
            ),
        )

    plausible = [
        c for c in unique_candidates
        if c["complete"] and c["balanced"] and (not c["contaminated"])
    ]
    if plausible:
        return ensure_semicolon(_pick_best(plausible)["sql"])

    complete_balanced = [c for c in unique_candidates if c["complete"] and c["balanced"]]
    if complete_balanced:
        return ensure_semicolon(_pick_best(complete_balanced)["sql"])

    incomplete_clean_balanced = [
        c for c in unique_candidates
        if (not c["complete"]) and c["balanced"] and (not c["contaminated"])
    ]
    if incomplete_clean_balanced:
        return ensure_semicolon(_pick_best(incomplete_clean_balanced)["sql"])

    return None


def extract_sql_first_statement_only(text: str) -> str | None:
    """
    Extract the first complete read-only SQL statement from model output.

    This mode is intentionally conservative for verbose NL2SQL outputs:
    - ignore prose before the first top-level SELECT/WITH candidate
    - cut at the first top-level semicolon outside quotes and parentheses
    - ignore prose after the first complete statement
    - prefer the earliest complete, balanced, non-contaminated statement
    """
    if not text or not text.strip():
        return None

    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _clean_generated_sql_candidate_robust(raw)
    if not cleaned:
        return None

    candidates = _collect_sql_candidates_from_source_v3(
        cleaned,
        source_name="full_text",
        base_offset=0,
        source_order=0,
    )
    if not candidates:
        return None

    def _is_clean_first_statement_candidate(candidate: dict[str, Any]) -> bool:
        sql = str(candidate["sql"]).lstrip()
        if not sql.lower().startswith("select"):
            return True
        # If a SELECT-started candidate contains another top-level SELECT/WITH
        # before its first semicolon, the first one is often prose like
        # "we need to select ..." followed by the real SQL query.
        return len(_top_level_candidate_query_starts(sql)) <= 1

    complete_clean = [
        c for c in candidates
        if c["complete"] and c["balanced"] and (not c["contaminated"]) and _is_clean_first_statement_candidate(c)
    ]
    if complete_clean:
        first = min(complete_clean, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
        return ensure_semicolon(first["sql"])

    complete_balanced = [
        c for c in candidates
        if c["complete"] and c["balanced"] and _is_clean_first_statement_candidate(c)
    ]
    if complete_balanced:
        first = min(complete_balanced, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
        return ensure_semicolon(first["sql"])

    incomplete_clean = [
        c for c in candidates
        if (
            (not c["complete"])
            and c["balanced"]
            and (not c["contaminated"])
            and _is_clean_first_statement_candidate(c)
        )
    ]
    if incomplete_clean:
        first = min(incomplete_clean, key=lambda c: (int(c["global_start"]), int(c["source_order"])))
        return ensure_semicolon(first["sql"])

    return None


def extract_sql_by_mode(text: str, extractor_mode: str) -> str | None:
    mode = extractor_mode.strip().lower()
    if mode == "legacy":
        return extract_sql(text)
    if mode == "robust_v2":
        return extract_sql_robust_v2(text)
    if mode == "robust_v3":
        return extract_sql_robust_v3(text)
    if mode in {"sql_first_statement_only", "first_statement"}:
        return extract_sql_first_statement_only(text)
    raise ValueError(
        f"Unknown extractor_mode '{extractor_mode}'. "
        "Use 'legacy', 'robust_v2', 'robust_v3', or 'sql_first_statement_only'."
    )


def _run_extract_sql_selftests() -> None:
    # A
    assert extract_sql("```sql\nSELECT COUNT(*) FROM customers;\n```") == "SELECT COUNT(*) FROM customers;"
    # B
    assert extract_sql("SELECT name FROM products") == "SELECT name FROM products;"
    # C
    assert extract_sql("Sure!\n```sql\nSELECT * FROM orders;\n```\nExplanation...") == "SELECT * FROM orders;"
    # D
    assert extract_sql("Here is SQL:\nSELECT * FROM orders; DROP TABLE customers;") == "SELECT * FROM orders;"
    # E
    assert extract_sql("```sql\nSELECT * FROM orders;\n```\n```sql\nSELECT * FROM customers;\n```") == "SELECT * FROM orders;"
    # F
    assert extract_sql("No query here") is None
    # G
    assert extract_sql("PRAGMA foreign_keys=ON; SELECT * FROM orders;") == "SELECT * FROM orders;"
    # H
    assert extract_sql("SELECT * FROM orders;\nQuestion:\nWhat next?") == "SELECT * FROM orders;"
    # I
    assert extract_sql("<|assistant|>\nSELECT * FROM products;\n<|user|>\nQuestion: x") == "SELECT * FROM products;"
    # J
    assert extract_sql("WITH t AS (SELECT * FROM orders) SELECT * FROM t;") == "WITH t AS (SELECT * FROM orders) SELECT * FROM t;"
    # K
    assert extract_sql("SQL query:\nSELECT * FROM orders;") == "SELECT * FROM orders;"
    # L
    assert extract_sql("SELECT AVG(Capacity), MAX(Capacity) FROM stadium;") == "SELECT AVG(Capacity), MAX(Capacity) FROM stadium;"
    # M
    assert extract_sql("</think>\nSELECT AVG(Capacity), MAX(Capacity) FROM stadium;") == "SELECT AVG(Capacity), MAX(Capacity) FROM stadium;"
    # N
    assert extract_sql("SELECT on the stadium table. We are to output SQL. </think> SELECT AVG(Capacity), MAX(Capacity) FROM stadium;") == "SELECT AVG(Capacity), MAX(Capacity) FROM stadium;"
    # O
    assert extract_sql("with columns: concert_ID, Year. </think> SELECT Year FROM concert GROUP BY Year ORDER BY COUNT(*) DESC LIMIT 1;") == "SELECT Year FROM concert GROUP BY Year ORDER BY COUNT(*) DESC LIMIT 1;"
    # P
    assert extract_sql("Some explanation\nWITH x AS (SELECT * FROM singer) SELECT * FROM x;") == "WITH x AS (SELECT * FROM singer) SELECT * FROM x;"
    # Q
    assert extract_sql("with SELECT. End with semicolon. No explanation. So: SELECT Song;") == "SELECT Song;"
    # R
    assert extract_sql("with the highest average attendance? We need to translate this.") is None
    # S
    assert extract_sql("Some text before. WITH cte AS (SELECT country FROM singer) SELECT * FROM cte;") == "WITH cte AS (SELECT country FROM singer) SELECT * FROM cte;"
    # T: legacy behavior remains unchanged for fragmentary SQL.
    legacy_fragment_input = "SELECT COUNT(*) FROM concert WHERE Year =;"
    assert extract_sql(legacy_fragment_input) == "SELECT COUNT(*) FROM concert WHERE Year =;"
    # U: robust_v2 prefers complete SQL before </think> over incomplete trailing SQL.
    robust_think_input = (
        "Reasoning draft. SELECT COUNT(*) FROM concert WHERE Year = 2014 OR Year = 2015;"
        "\n</think>\nSELECT COUNT(*) FROM concert WHERE Year ="
    )
    assert (
        extract_sql_robust_v2(robust_think_input)
        == "SELECT COUNT(*) FROM concert WHERE Year = 2014 OR Year = 2015;"
    )
    # V: robust_v2 prefers later final SQL when multiple complete candidates exist.
    robust_multi_input = (
        "SELECT Name FROM singer;\n"
        "However, corrected final answer:\n"
        "SELECT DISTINCT Name FROM singer;"
    )
    assert extract_sql_robust_v2(robust_multi_input) == "SELECT DISTINCT Name FROM singer;"
    # W: robust_v2 rejects contaminated candidates and keeps clean SQL candidate.
    robust_contaminated_input = (
        "SELECT Name FROM stadium EXCEPT SELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID "
        "But note this explanation keeps going;\n"
        "SELECT T1.Name FROM stadium AS T1 LEFT JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.concert_ID IS NULL;"
    )
    assert (
        extract_sql_robust_v2(robust_contaminated_input)
        == "SELECT T1.Name FROM stadium AS T1 LEFT JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.concert_ID IS NULL;"
    )
    # X: robust_v2 rejects obvious WHERE =; fragment.
    assert extract_sql_robust_v2("SELECT COUNT(*) FROM concert WHERE Year =;") is None
    # Y: robust_v3 must prefer full outer SQL over inner subquery fragment.
    robust_v3_outer_vs_inner = (
        "Draft: SELECT AVG(Age) FROM singer);\n"
        "Final: SELECT Song_Name FROM singer WHERE Age > (SELECT AVG(Age) FROM singer);"
    )
    assert (
        extract_sql_robust_v3(robust_v3_outer_vs_inner)
        == "SELECT Song_Name FROM singer WHERE Age > (SELECT AVG(Age) FROM singer);"
    )
    # Z: robust_v3 keeps full INTERSECT query, not only second SELECT part.
    robust_v3_intersect = (
        "SELECT Citizenship FROM singer WHERE Birth_Year < 1945 "
        "INTERSECT SELECT Citizenship FROM singer WHERE Birth_Year > 1955;"
    )
    assert (
        extract_sql_robust_v3(robust_v3_intersect)
        == "SELECT Citizenship FROM singer WHERE Birth_Year < 1945 INTERSECT SELECT Citizenship FROM singer WHERE Birth_Year > 1955;"
    )
    # AA: robust_v3 keeps full EXCEPT query as one statement.
    robust_v3_except = (
        "SELECT Name FROM employee EXCEPT "
        "SELECT T1.Name FROM employee AS T1 JOIN evaluation AS T2 ON T1.Employee_ID = T2.Employee_ID;"
    )
    assert (
        extract_sql_robust_v3(robust_v3_except)
        == "SELECT Name FROM employee EXCEPT SELECT T1.Name FROM employee AS T1 JOIN evaluation AS T2 ON T1.Employee_ID = T2.Employee_ID;"
    )
    # AB: robust_v3 rejects obvious WHERE =; fragment.
    assert extract_sql_robust_v3("SELECT COUNT(*) FROM concert WHERE Year =;") is None
    # AC: first-statement mode keeps the earliest complete SQL and drops later prose.
    first_statement_verbose = (
        "We are given the schema. Steps: identify rows. "
        "SELECT Name FROM singer WHERE Birth_Year > 1980; "
        "However, another possible answer is SELECT Name FROM singer;"
    )
    assert extract_sql_first_statement_only(first_statement_verbose) == "SELECT Name FROM singer WHERE Birth_Year > 1980;"
    # AD: first-statement mode cuts at top-level semicolon and preserves semicolons in strings.
    first_statement_quoted_semicolon = (
        "Answer: SELECT Name FROM singer WHERE Notes = 'born; active'; "
        "Explanation after SQL."
    )
    assert (
        extract_sql_first_statement_only(first_statement_quoted_semicolon)
        == "SELECT Name FROM singer WHERE Notes = 'born; active';"
    )
    # AE: first-statement mode uses top-level semicolon, not subquery-internal SELECT starts.
    first_statement_subquery = (
        "Steps...\n"
        "SELECT Song_Name FROM singer WHERE Age > (SELECT AVG(Age) FROM singer); "
        "More text."
    )
    assert (
        extract_sql_first_statement_only(first_statement_subquery)
        == "SELECT Song_Name FROM singer WHERE Age > (SELECT AVG(Age) FROM singer);"
    )
    # AF: first-statement mode skips prose "select ..." when followed by real SQL.
    first_statement_prose_select = (
        "We need to select the relevant singer name from the schema. "
        "SQL: SELECT Name FROM singer WHERE Birth_Year > 1980;"
    )
    assert (
        extract_sql_first_statement_only(first_statement_prose_select)
        == "SELECT Name FROM singer WHERE Birth_Year > 1980;"
    )


def _value_sort_key(value: Any) -> tuple[str, str]:
    # Technical sort key only: keeps exact value semantics unchanged.
    return (type(value).__name__, repr(value))


def _row_sort_key(row: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(_value_sort_key(v) for v in row)


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple[Any, ...]] | None
    error: str | None


def run_sql(conn: sqlite3.Connection, sql: str) -> ExecResult:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return ExecResult(ok=True, rows=rows, error=None)
    except Exception as e:
        return ExecResult(ok=False, rows=None, error=repr(e))


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(rows, key=_row_sort_key)


def execution_match(pred: ExecResult, gold: ExecResult) -> bool:
    if not pred.ok or not gold.ok:
        return False
    return normalize_rows(pred.rows) == normalize_rows(gold.rows)


def _run_execution_match_selftests() -> None:
    mixed_rows_a = [
        (1, "x", 2.5, None),
        (2, "y", 1.0, None),
    ]
    mixed_rows_b = [
        (2, "y", 1.0, None),
        (1, "x", 2.5, None),
    ]
    pred = ExecResult(ok=True, rows=mixed_rows_a, error=None)
    gold = ExecResult(ok=True, rows=mixed_rows_b, error=None)
    assert execution_match(pred, gold) is True


def _run_batch_slicing_selftests() -> None:
    # Decoder-only generate returns the padded prompt plus new tokens.
    output_ids = [0, 0, 11, 12, 101, 102]
    assert _slice_generated_token_ids(output_ids, 4) == [101, 102]

    tensor_ids = torch.tensor([0, 13, 14, 15, 201, 202, 203])
    assert _slice_generated_token_ids(tensor_ids, 4).tolist() == [201, 202, 203]


def uses_chat_format(llm_name: str) -> bool:
    """Return True if this model should be prompted via a chat template."""
    # TinyLlama is a chat/instruct model; plain prompts often work worse.
    return llm_name in {"tinyllama_11b"}


def resolve_prompt_format(llm_name: str, prompt_format: str) -> str:
    fmt = prompt_format.strip().lower()
    if fmt in QWEN_SQLCTX_PROMPT_FORMATS:
        return QWEN_SQLCTX_CHATML_FORMAT
    if fmt in LLAMA32_V2_PROMPT_FORMATS:
        return fmt
    if fmt == LLAMA32_NATIVE_CHAT_FORMAT:
        if llm_name != "llama32_3b_instruct":
            raise ValueError(
                f"{LLAMA32_NATIVE_CHAT_FORMAT!r} requires llm='llama32_3b_instruct'"
            )
        return fmt
    if fmt in {"chat", "chat_template"}:
        return "chat_template"
    if fmt == "plain":
        return "plain"
    if fmt == "auto":
        return "chat_template" if uses_chat_format(llm_name) else "plain"
    raise ValueError(
        "prompt_format must be 'auto', 'plain', 'chat', 'chat_template', "
        "'qwen_sqlctx_chatml' (preferred), legacy aliases 'qwen_v2_sqlctx', "
        "'qwen_v2_sqlctx_full_chat', 'v2_prompt_completion_chatml', "
        "'llama32_instruct_sqlctx', 'llama32_v2_sqlctx', or "
        f"'{LLAMA32_NATIVE_CHAT_FORMAT}'"
    )


def _base_prompt_text(schema_txt: str, question: str) -> str:
    return f"""
You are an assistant that translates natural language questions into SQLite SQL queries.

Database schema:
{schema_txt}

Rules:
- Use only the tables and columns from the schema.
- Output exactly one valid SQLite read query (SELECT or WITH...SELECT).
- Start directly with SELECT or WITH.
- End the query with a semicolon.
- No explanation or extra text.
- Do not use markdown.
- Do not output additional examples, prompt labels, or chat role tokens.
- Stop immediately after the first query.

Question:
{question}

SQL:
""".strip()


def _normalize_v2_schema_text(schema_txt: str) -> str:
    lines = str(schema_txt).strip().splitlines()
    if lines and lines[0].strip().lower() == "database schema:":
        lines = lines[1:]
    return "\n".join(lines).strip()


SQL_TABLE_REF_RE = re.compile(
    r"\b(?:from|join)\s+(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*))",
    flags=re.IGNORECASE,
)


def _schema_table_blocks(schema_txt: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in _normalize_v2_schema_text(schema_txt).splitlines():
        match = re.match(r"^\s*Table:\s*(.+?)\s*$", line)
        if match:
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = match.group(1).strip()
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        blocks.append((current_name, current_lines))
    return blocks


def _extract_sql_table_refs(
    sql: str,
    *,
    known_table_names: set[str] | None = None,
) -> set[str]:
    cleaned = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", str(sql))
    refs: set[str] = set()
    for match in SQL_TABLE_REF_RE.finditer(cleaned):
        ref = next((group for group in match.groups() if group), "")
        ref = ref.strip().strip("`\"[]")
        if not ref or ref.lower() == "select":
            continue
        refs.add(ref.lower())
    if known_table_names:
        from_spans = re.findall(
            r"\bfrom\b(.*?)(?=\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|\bunion\b|\bintersect\b|\bexcept\b|;|$)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for span in from_spans:
            for table_name in known_table_names:
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(table_name)}(?![A-Za-z0-9_])", span, flags=re.IGNORECASE):
                    refs.add(table_name.lower())
    return refs


def _filter_schema_block_foreign_keys(
    lines: list[str],
    *,
    selected_table_names_norm: set[str],
) -> list[str]:
    filtered: list[str] = []
    pending_fk_header: str | None = None
    kept_fk_lines: list[str] = []
    in_fk = False

    def flush_fk() -> None:
        nonlocal pending_fk_header, kept_fk_lines
        if pending_fk_header is not None and kept_fk_lines:
            filtered.append(pending_fk_header)
            filtered.extend(kept_fk_lines)
        pending_fk_header = None
        kept_fk_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "foreign keys:":
            flush_fk()
            in_fk = True
            pending_fk_header = line
            continue
        if in_fk and stripped.startswith("-"):
            tables = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.", stripped)
            if tables and all(table.lower() in selected_table_names_norm for table in tables):
                kept_fk_lines.append(line)
            continue
        if in_fk:
            flush_fk()
            in_fk = False
        filtered.append(line)
    flush_fk()
    return filtered


def reduce_schema_to_sql_relevant_tables(
    schema_txt: str,
    sql: str,
) -> tuple[str, dict[str, Any]]:
    normalized_schema = _normalize_v2_schema_text(schema_txt)
    blocks = _schema_table_blocks(normalized_schema)
    table_map = {name.lower(): (name, lines) for name, lines in blocks}
    table_refs = _extract_sql_table_refs(sql, known_table_names=set(table_map))
    selected = {ref for ref in table_refs if ref in table_map}
    unknown_refs = sorted(ref for ref in table_refs if ref not in table_map)
    fallback_reason = ""
    if not blocks:
        fallback_reason = "no_schema_table_blocks"
    elif not table_refs:
        fallback_reason = "no_sql_table_refs"
    elif unknown_refs:
        fallback_reason = "unknown_sql_table_refs"
    elif not selected:
        fallback_reason = "no_matching_sql_table_refs"

    if fallback_reason:
        return normalized_schema, {
            "fallback": True,
            "fallback_reason": fallback_reason,
            "schema_table_count": len(blocks),
            "reduced_table_count": len(blocks),
            "sql_table_count": len(table_refs),
            "sql_tables": sorted(table_refs),
            "unknown_sql_tables": unknown_refs,
            "selected_tables": [name for name, _lines in blocks],
        }

    reduced_blocks: list[str] = []
    for name, lines in blocks:
        if name.lower() not in selected:
            continue
        block_lines = _filter_schema_block_foreign_keys(
            lines,
            selected_table_names_norm=selected,
        )
        reduced_blocks.append("\n".join(block_lines).strip())

    if not reduced_blocks:
        return normalized_schema, {
            "fallback": True,
            "fallback_reason": "empty_reduced_schema",
            "schema_table_count": len(blocks),
            "reduced_table_count": len(blocks),
            "sql_table_count": len(table_refs),
            "sql_tables": sorted(table_refs),
            "unknown_sql_tables": unknown_refs,
            "selected_tables": [name for name, _lines in blocks],
        }

    selected_names = [name for name, _lines in blocks if name.lower() in selected]
    return "\n\n".join(reduced_blocks).strip(), {
        "fallback": False,
        "fallback_reason": "",
        "schema_table_count": len(blocks),
        "reduced_table_count": len(selected_names),
        "sql_table_count": len(table_refs),
        "sql_tables": sorted(table_refs),
        "unknown_sql_tables": unknown_refs,
        "selected_tables": selected_names,
    }


def _v2_sqlctx_prompt_text(schema_txt: str, question: str) -> str:
    schema_body = _normalize_v2_schema_text(schema_txt)
    return f"""Database schema:
{schema_body}

Rules:
- Use only the tables and columns from the schema.
- Output exactly ONE SQLite read query.
- Start directly with SELECT or WITH.
- End with a semicolon.
- Do NOT explain anything.
- Do NOT use markdown.

Question:
{question}

SQL:"""


def _render_qwen_sqlctx_chatml_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """
    Render OpenAI-style messages to the canonical Qwen SQLCTX ChatML prefix.

    Qwen's tokenizer chat template appends "<think>\\n" when
    add_generation_prompt=True. The SQLCTX training data was built without that
    token, so this renderer keeps the messages abstraction but emits ChatML manually.
    """
    chunks: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role for qwen_sqlctx_chatml rendering: {role!r}")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append(V2_SQLCTX_ASSISTANT_PREFIX)
    return "".join(chunks)


def _render_qwen_v2_sqlctx_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    return _render_qwen_sqlctx_chatml_messages(
        messages,
        add_generation_prompt=add_generation_prompt,
    )


def _render_llama32_v2_sqlctx_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """
    Render OpenAI-style messages with Llama-3.x Instruct header tokens.

    The tokenizer's apply_chat_template is available for Llama 3.2 Instruct,
    but it injects a date/knowledge-cutoff preamble. For controlled Spider
    comparisons we keep the official Llama header/eot token structure while
    preserving the SQLCTX system prompt content exactly.
    """
    chunks: list[str] = ["<|begin_of_text|>"]
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role for Llama SQLCTX rendering: {role!r}")
        chunks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    if add_generation_prompt:
        chunks.append(LLAMA32_V2_SQLCTX_ASSISTANT_PREFIX)
    return "".join(chunks)


def build_prompt(
    schema_txt: str,
    question: str,
    llm_name: str,
    tokenizer,
    prompt_format: str = "auto",
    chat_template: str | None = None,
    system_instruction: str = CURRENT_SYSTEM_PROMPT,
) -> str:
    """Build a model-appropriate prompt (plain or chat-template)."""
    fmt = resolve_prompt_format(llm_name, prompt_format)
    user_prompt = (
        _v2_sqlctx_prompt_text(schema_txt, question)
        if fmt in V2_SQLCTX_PROMPT_FORMATS
        else _base_prompt_text(schema_txt, question)
    )
    messages = build_nl2sql_messages(system_instruction=system_instruction, user_prompt=user_prompt)
    if fmt in V2_PROMPT_FORMATS:
        return _render_qwen_sqlctx_chatml_messages(messages, add_generation_prompt=True)
    if fmt in LLAMA32_V2_PROMPT_FORMATS:
        return _render_llama32_v2_sqlctx_messages(messages, add_generation_prompt=True)
    if fmt == LLAMA32_NATIVE_CHAT_FORMAT:
        return render_llama32_native_chat(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
    return render_messages(
        tokenizer=tokenizer,
        messages=messages,
        prompt_format=fmt,
        chat_template=chat_template,
        add_generation_prompt=True,
    )


def retrieve_examples(
    question: str,
    k: int,
    embedder: Any,
    index_items: list[dict],
    index_emb: Any,
    exclude_id: str | None = None,
    exclude_question: str | None = None,
    query_db_id: str | None = None,
    same_db_only: bool = False,
) -> list[dict]:
    import numpy as np

    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    sims = index_emb @ q_emb  # cosine similarity because embeddings are normalized
    ranked = np.argsort(-sims)
    examples = []
    ex_q_norm = (exclude_question or "").strip().lower()
    query_db_id_norm = (query_db_id or "").strip()
    for idx in ranked:
        item = index_items[int(idx)]
        if same_db_only and query_db_id_norm:
            item_db_id = str(item.get("db_id", "")).strip()
            if item_db_id != query_db_id_norm:
                continue
        if exclude_id and item.get("id") == exclude_id:
            continue
        item_q_norm = str(item.get("question", "")).strip().lower()
        if ex_q_norm and item_q_norm == ex_q_norm:
            continue
        examples.append(item)
        if len(examples) >= k:
            break
    return examples


def build_prompt_fewshot(
    schema_txt: str,
    question: str,
    demos: list[dict],
    llm_name: str,
    tokenizer,
    prompt_format: str = "auto",
    chat_template: str | None = None,
    system_instruction: str = CURRENT_SYSTEM_PROMPT,
) -> str:
    demo_blocks = []
    for idx, d in enumerate(demos, start=1):
        gold = d["gold_sql"].strip()
        if not gold.endswith(";"):
            gold += ";"
        demo_blocks.append(
            f"Example {idx} Question:\n{d['question']}\n"
            f"Example {idx} SQL:\n{gold}"
        )
    demos_text = "\n".join(demo_blocks).strip()

    base_text = f"""
You are an assistant that translates natural language questions into SQLite SQL queries.

Rules:
- Output exactly one valid SQLite read query (SELECT or WITH...SELECT).
- Start directly with SELECT or WITH.
- End the query with a semicolon.
- No explanation or extra text.
- Do not use markdown.
- Do not output additional examples, prompt labels, or chat role tokens.
- Stop immediately after the first query.

Below are example question-SQL pairs from other databases. They are provided only to illustrate SQL structure. Do not copy table names, column names, literal values, aliases, or database-specific identifiers from these examples. For the final answer, use only the target schema shown below.

BEGIN EXAMPLES
{demos_text}
END EXAMPLES

Target database schema:
{schema_txt}

Question:
{question}

Use only tables and columns from the target schema. Return exactly one SQLite query and no explanation.

SQL:
""".strip()

    messages = build_nl2sql_messages(system_instruction=system_instruction, user_prompt=base_text)
    fmt = resolve_prompt_format(llm_name, prompt_format)
    if fmt in V2_PROMPT_FORMATS:
        return _render_qwen_sqlctx_chatml_messages(messages, add_generation_prompt=True)
    if fmt in LLAMA32_V2_PROMPT_FORMATS:
        return _render_llama32_v2_sqlctx_messages(messages, add_generation_prompt=True)
    if fmt == LLAMA32_NATIVE_CHAT_FORMAT:
        return render_llama32_native_chat(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
    return render_messages(
        tokenizer=tokenizer,
        messages=messages,
        prompt_format=fmt,
        chat_template=chat_template,
        add_generation_prompt=True,
    )


def build_prompt_schema_fewshot(
    schema_txt: str,
    question: str,
    demos: list[dict],
    llm_name: str,
    tokenizer,
    prompt_format: str = "auto",
    chat_template: str | None = None,
    system_instruction: str = CURRENT_SYSTEM_PROMPT,
    example_schema_mode: str = "full",
    example_mode: str = "schema_with_rules",
) -> str:
    rules_text = """Rules:
- Use only the tables and columns from the schema.
- Output exactly ONE SQLite read query.
- Start directly with SELECT or WITH.
- End with a semicolon.
- Do NOT explain anything.
- Do NOT use markdown."""
    example_blocks: list[str] = []
    for idx, demo in enumerate(demos, start=1):
        demo_schema = _normalize_v2_schema_text(str(demo.get("schema_prompt", "")))
        gold_sql = ensure_semicolon(str(demo.get("gold_sql", "")))
        if example_mode == "question_sql_only":
            example_blocks.append(
                f"""Example {idx}

Question:
{demo.get("question", "")}

SQL:
{gold_sql}"""
            )
            continue
        if example_mode != "schema_with_rules":
            raise ValueError(
                "example_mode must be 'schema_with_rules' or 'question_sql_only'"
            )
        if example_schema_mode == "sql_relevant_only":
            demo_schema, _details = reduce_schema_to_sql_relevant_tables(
                demo_schema,
                gold_sql,
            )
        elif example_schema_mode != "full":
            raise ValueError(
                "example_schema_mode must be 'full' or 'sql_relevant_only'"
            )
        example_blocks.append(
            f"""Example {idx}
Database schema:
{demo_schema}

{rules_text}

Question:
{demo.get("question", "")}

SQL:
{gold_sql}"""
        )
    examples_text = "\n\n".join(example_blocks).strip()
    target_schema = _normalize_v2_schema_text(schema_txt)
    user_prompt = f"""{examples_text}

Now solve the following task.
Database schema:
{target_schema}

{rules_text}

Question:
{question}

SQL:"""

    messages = build_nl2sql_messages(system_instruction=system_instruction, user_prompt=user_prompt)
    fmt = resolve_prompt_format(llm_name, prompt_format)
    if fmt in V2_PROMPT_FORMATS:
        return _render_qwen_sqlctx_chatml_messages(messages, add_generation_prompt=True)
    if fmt in LLAMA32_V2_PROMPT_FORMATS:
        return _render_llama32_v2_sqlctx_messages(messages, add_generation_prompt=True)
    if fmt == LLAMA32_NATIVE_CHAT_FORMAT:
        return render_llama32_native_chat(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
    return render_messages(
        tokenizer=tokenizer,
        messages=messages,
        prompt_format=fmt,
        chat_template=chat_template,
        add_generation_prompt=True,
    )


def _first_retrieval_similarity(selection: FewShotSelection | None) -> float | None:
    if selection is None or not selection.scores:
        return None
    try:
        return float(selection.scores[0])
    except (TypeError, ValueError):
        return None


def _first_selected_example_id(selection: FewShotSelection | None) -> str:
    if selection is None or not selection.examples:
        return ""
    return str(selection.examples[0].get("id", "")).strip()


def _evaluate_fewshot_gate_k1_legacy(
    *,
    enabled: bool,
    mode: str | None,
    threshold: float | None,
    features: list[str],
    selection: FewShotSelection | None,
    question: str,
    debug_enabled: bool,
) -> FewShotGateDecision:
    retrieval_similarity = _first_retrieval_similarity(selection)
    selected_example_id = _first_selected_example_id(selection)
    retrieved_count = len(selection.examples) if selection is not None else 0
    mode_norm = (mode or "").strip().lower()
    debug: dict[str, Any] = {}

    if not enabled:
        return FewShotGateDecision(
            enabled=False,
            mode="disabled",
            score=retrieval_similarity,
            threshold=None,
            decision="fewshot",
            reason="gate_disabled",
            retrieval_similarity=retrieval_similarity,
            selected_example_id=selected_example_id,
            number_of_retrieved_candidates=retrieved_count,
            debug={},
        )

    if mode_norm not in {"similarity_only", "weighted_score", "rerank_similarity"}:
        raise ValueError(
            "fewshot_gate_mode must be one of: similarity_only, weighted_score, "
            "rerank_similarity"
        )
    if selection is None or not selection.examples:
        return FewShotGateDecision(
            enabled=True,
            mode=mode_norm,
            score=None,
            threshold=threshold,
            decision="zero_shot",
            reason="no_retrieval_examples",
            retrieval_similarity=retrieval_similarity,
            selected_example_id=selected_example_id,
            number_of_retrieved_candidates=retrieved_count,
            debug=debug,
        )
    if retrieval_similarity is None:
        return FewShotGateDecision(
            enabled=True,
            mode=mode_norm,
            score=None,
            threshold=threshold,
            decision="zero_shot",
            reason="no_similarity_score",
            retrieval_similarity=retrieval_similarity,
            selected_example_id=selected_example_id,
            number_of_retrieved_candidates=retrieved_count,
            debug=debug,
        )

    score = retrieval_similarity
    if mode_norm == "weighted_score":
        allowed_features = {"sqlaware_structure_bonus"}
        unknown_features = sorted(set(features) - allowed_features)
        if unknown_features:
            raise ValueError(
                "Unsupported fewshot_gate_features for weighted_score: "
                + ", ".join(unknown_features)
            )
        if "sqlaware_structure_bonus" in features:
            bonus, bonus_details = sqlaware_structure_bonus(
                question,
                str(selection.examples[0].get("gold_sql", "")),
            )
            score += bonus
            if debug_enabled:
                debug["sqlaware_structure_bonus"] = bonus
                debug["sqlaware_structure_bonus_details"] = bonus_details

    if threshold is None:
        if mode_norm == "rerank_similarity":
            return FewShotGateDecision(
                enabled=True,
                mode=mode_norm,
                score=score,
                threshold=None,
                decision="fewshot",
                reason="no_threshold_accept",
                retrieval_similarity=retrieval_similarity,
                selected_example_id=selected_example_id,
                number_of_retrieved_candidates=retrieved_count,
                debug=debug if debug_enabled else {},
            )
        raise ValueError(
            "fewshot_gate_threshold must be set for similarity_only or weighted_score"
        )

    if score >= float(threshold):
        decision = "fewshot"
        reason = "score_meets_threshold"
    else:
        decision = "zero_shot"
        reason = "below_threshold"

    return FewShotGateDecision(
        enabled=True,
        mode=mode_norm,
        score=score,
        threshold=float(threshold),
        decision=decision,
        reason=reason,
        retrieval_similarity=retrieval_similarity,
        selected_example_id=selected_example_id,
        number_of_retrieved_candidates=retrieved_count,
        debug=debug if debug_enabled else {},
    )


def evaluate_fewshot_gate(
    *,
    enabled: bool,
    mode: str | None,
    threshold: float | None,
    features: list[str],
    selection: FewShotSelection | None,
    question: str,
    debug_enabled: bool,
) -> FewShotGateDecision:
    raw_scores = tuple(float(value) for value in (selection.scores if selection else []))
    if any(not math.isfinite(value) for value in raw_scores):
        raise RuntimeError("Non-finite BGE similarity in few-shot selection")
    score_min = min(raw_scores) if raw_scores else None
    score_max = max(raw_scores) if raw_scores else None
    score_mean = sum(raw_scores) / len(raw_scores) if raw_scores else None
    mode_norm = (mode or "").strip().lower()

    if mode_norm != "set_min_similarity":
        decision = _evaluate_fewshot_gate_k1_legacy(
            enabled=enabled,
            mode=mode,
            threshold=threshold,
            features=features,
            selection=selection,
            question=question,
            debug_enabled=debug_enabled,
        )
        decision.retrieval_similarities = raw_scores
        decision.retrieval_similarity_min = score_min
        decision.retrieval_similarity_max = score_max
        decision.retrieval_similarity_mean = score_mean
        decision.score_semantics = "first_selected_bge_similarity"
        return decision

    examples = selection.examples if selection is not None else []
    selected_example_id = _first_selected_example_id(selection)
    retrieval_similarity = _first_retrieval_similarity(selection)
    retrieved_count = len(examples)
    if features:
        raise ValueError("set_min_similarity does not accept fewshot_gate_features")
    if examples and len(raw_scores) != len(examples):
        raise RuntimeError(
            "set_min_similarity requires one original BGE score per selected demonstration"
        )
    if not enabled:
        gate_score = retrieval_similarity
        gate_threshold = None
        gate_decision = "fewshot"
        gate_reason = "gate_disabled"
    elif not examples:
        gate_score = None
        gate_threshold = threshold
        gate_decision = "zero_shot"
        gate_reason = "no_retrieval_examples"
    elif score_min is None:
        gate_score = None
        gate_threshold = threshold
        gate_decision = "zero_shot"
        gate_reason = "no_similarity_score"
    else:
        if threshold is None:
            raise ValueError("fewshot_gate_threshold is required for set_min_similarity")
        gate_score = score_min
        gate_threshold = float(threshold)
        gate_decision = "fewshot" if gate_score >= gate_threshold else "zero_shot"
        gate_reason = (
            "set_min_meets_threshold"
            if gate_decision == "fewshot"
            else "set_min_below_threshold"
        )

    debug = {
        "gate_score_semantics": "minimum_original_bge_similarity_of_selected_set",
        "selected_original_bge_scores": list(raw_scores),
        "selected_score_min": score_min,
        "selected_score_max": score_max,
        "selected_score_mean": score_mean,
    }
    return FewShotGateDecision(
        enabled=enabled,
        mode="set_min_similarity",
        score=gate_score,
        threshold=gate_threshold,
        decision=gate_decision,
        reason=gate_reason,
        retrieval_similarity=retrieval_similarity,
        selected_example_id=selected_example_id,
        number_of_retrieved_candidates=retrieved_count,
        debug=debug,
        retrieval_similarities=raw_scores,
        retrieval_similarity_min=score_min,
        retrieval_similarity_max=score_max,
        retrieval_similarity_mean=score_mean,
        score_semantics="minimum_original_bge_similarity_of_selected_set",
    )


def gate_decision_to_csv_values(decision: FewShotGateDecision) -> dict[str, str | int]:
    return {
        "gate_enabled": int(decision.enabled),
        "gate_mode": decision.mode,
        "gate_score": "" if decision.score is None else f"{decision.score:.6f}",
        "gate_threshold": ""
        if decision.threshold is None
        else f"{decision.threshold:.6f}",
        "gate_decision": decision.decision,
        "gate_reason": decision.reason,
        "retrieval_similarity": ""
        if decision.retrieval_similarity is None
        else f"{decision.retrieval_similarity:.6f}",
        "selected_example_id": decision.selected_example_id,
        "number_of_retrieved_candidates": decision.number_of_retrieved_candidates,
        "gate_similarity_scores": json.dumps(
            [round(value, 6) for value in decision.retrieval_similarities],
            ensure_ascii=False,
        ),
        "gate_similarity_min": ""
        if decision.retrieval_similarity_min is None
        else f"{decision.retrieval_similarity_min:.6f}",
        "gate_similarity_max": ""
        if decision.retrieval_similarity_max is None
        else f"{decision.retrieval_similarity_max:.6f}",
        "gate_similarity_mean": ""
        if decision.retrieval_similarity_mean is None
        else f"{decision.retrieval_similarity_mean:.6f}",
        "gate_score_semantics": decision.score_semantics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch NL2SQL evaluation with Execution Match Accuracy.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file for evaluation parameters.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Additive output directory for the k=3 extension.",
    )
    parser.add_argument(
        "--run_output_prefix",
        type=str,
        default=None,
        help="Unique filename prefix for k=3 CSV, metadata, and retrieval trace outputs.",
    )
    parser.add_argument(
        "--expected_model_revision",
        type=str,
        default=None,
        help="Required immutable local model snapshot revision for the k=3 run.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        choices=LLMClient.list_llms(),
        help="Which base LLM to use (registry key).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Adapter to load. Use 'base' for no adapter. Otherwise uses ./adapters/<llm>/<adapter>.",
    )
    parser.add_argument(
        "--prompt_tuning",
        type=str,
        default=None,
        choices=["none", "fewshot", "static_fewshot", "dynamic_fewshot"],
        help=(
            "Prompting mode: none (zero-shot), static_fewshot, dynamic_fewshot, "
            "or legacy fewshot."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of retrieved demonstrations to include when --prompt_tuning=fewshot.",
    )
    parser.add_argument(
        "--index_json_path",
        type=str,
        default=None,
        help="Path to retrieval index JSON (used when --prompt_tuning=fewshot).",
    )
    parser.add_argument(
        "--index_emb_path",
        type=str,
        default=None,
        help="Path to retrieval index embeddings .npy (used when --prompt_tuning=fewshot).",
    )
    parser.add_argument(
        "--embed_model",
        "--embedding_model",
        dest="embed_model",
        type=str,
        default=None,
        help="SentenceTransformer embedding model (used when --prompt_tuning=fewshot).",
    )
    parser.add_argument(
        "--retrieval_pool_path",
        type=str,
        default=None,
        help="Retrieval-pool JSONL path for static_fewshot and leakage metadata.",
    )
    parser.add_argument(
        "--retrieval_index_path",
        type=str,
        default=None,
        help="FAISS retrieval index directory for dynamic_fewshot.",
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default=None,
        choices=["sentence_transformer_faiss", "static_seeded"],
        help="Retrieval method for Few-Shot modes.",
    )
    parser.add_argument(
        "--retrieval_rerank_method",
        type=str,
        default=None,
        choices=["none", "sqlaware_topk", "structure_topk_v2"],
        help="Optional dynamic few-shot candidate re-ranking method.",
    )
    parser.add_argument(
        "--retrieval_rerank_top_n",
        type=int,
        default=None,
        help="Number of initial BGE candidates considered by the selected re-ranking method.",
    )
    parser.add_argument(
        "--retrieval_structure_bonus_max",
        type=float,
        default=None,
        help="Maximum structure adjustment added to BGE similarity.",
    )
    parser.add_argument(
        "--fewshot_example_schema_mode",
        type=str,
        default=None,
        choices=["full", "sql_relevant_only"],
        help=(
            "Schema rendering for retrieved schema few-shot examples: full "
            "or SQL-relevant tables only."
        ),
    )
    parser.add_argument(
        "--fewshot_example_mode",
        type=str,
        default=None,
        choices=["schema_with_rules", "question_sql_only"],
        help=(
            "Rendering mode for retrieved schema few-shot examples: "
            "schema_with_rules (default) or question_sql_only."
        ),
    )
    parser.add_argument(
        "--fewshot_gate_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable optional Dynamic Few-Shot gating. Disabled by default; "
            "when disabled, legacy Few-Shot prompts and CSV fields are unchanged."
        ),
    )
    parser.add_argument(
        "--fewshot_gate_threshold",
        type=float,
        default=None,
        help="Gate threshold for similarity_only or weighted_score modes.",
    )
    parser.add_argument(
        "--fewshot_gate_mode",
        type=str,
        default=None,
        choices=[
            "similarity_only",
            "weighted_score",
            "rerank_similarity",
            "set_min_similarity",
        ],
        help="Optional Few-Shot gate mode.",
    )
    parser.add_argument(
        "--fewshot_rerank_top_n",
        type=int,
        default=None,
        help=(
            "Optional future-compatible top-n hint for gate/rerank experiments. "
            "The current BGE similarity path already selects the highest-similarity hit."
        ),
    )
    parser.add_argument(
        "--fewshot_gate_features",
        type=str,
        default=None,
        help=(
            "Comma-separated gate features. Currently supported for weighted_score: "
            "sqlaware_structure_bonus."
        ),
    )
    parser.add_argument(
        "--fewshot_gate_debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write optional gate debug details into retrieval traces when gate is enabled.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="Maximum input prompt length in tokens before truncation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate for SQL output.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=None,
        help="Number of prompts to generate in one model.generate call (default: 1).",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default=None,
        choices=[
            "auto",
            "plain",
            "chat",
            "chat_template",
            "qwen_sqlctx_chatml",
            "qwen_v2_sqlctx",
            "qwen_v2_sqlctx_full_chat",
            "v2_prompt_completion_chatml",
            "llama32_instruct_sqlctx",
            "llama32_v2_sqlctx",
            LLAMA32_NATIVE_CHAT_FORMAT,
        ],
        help=(
            "Prompt serialization format: auto (default), plain, chat, chat_template, "
            "qwen_sqlctx_chatml (preferred), legacy aliases qwen_v2_sqlctx, "
            "qwen_v2_sqlctx_full_chat, v2_prompt_completion_chatml, "
            "llama32_instruct_sqlctx, llama32_v2_sqlctx, or "
            f"{LLAMA32_NATIVE_CHAT_FORMAT}."
        ),
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Optional custom tokenizer chat template (Jinja).",
    )
    parser.add_argument(
        "--system_prompt_variant",
        type=str,
        default=None,
        help="System prompt preset variant (default: current). Ignored when system_prompt_path is set.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Optional path to a text file containing the system prompt.",
    )
    parser.add_argument(
        "--testcases_path",
        type=str,
        default=None,
        help="Path to evaluation testcases JSONL (default: data/testcases.jsonl).",
    )
    parser.add_argument(
        "--traincases_path",
        type=str,
        default=None,
        help=(
            "Path to traincases JSONL for overlap checks only "
            "(default: data/traincases.jsonl)."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Limit number of evaluated testcases (for local smoke tests).",
    )
    parser.add_argument(
        "--extractor_mode",
        type=str,
        default=None,
        choices=["legacy", "robust_v2", "robust_v3", "sql_first_statement_only", "first_statement"],
        help=(
            "SQL extractor mode: legacy (default), robust_v2, robust_v3, "
            "or sql_first_statement_only."
        ),
    )
    parser.add_argument(
        "--progress_log_every",
        type=int,
        default=None,
        help="Log progress every N processed testcases (<=0 disables periodic progress logs).",
    )
    parser.add_argument(
        "--compute_perplexity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also compute gold-target perplexity (slower).",
    )
    parser.add_argument(
        "--allow_overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow train/test overlap. By default, evaluation stops if overlap is detected.",
    )
    parser.add_argument(
        "--same_db_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When using fewshot retrieval, only retrieve demonstrations with the same db_id.",
    )
    parser.add_argument(
        "--dummy_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run pipeline without model loading/generation and use dummy_sql as raw model output.",
    )
    parser.add_argument(
        "--dummy_sql",
        type=str,
        default=None,
        help="Raw SQL text used as generated output when --dummy_mode is enabled.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log_format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Logging format: text (default) or json.",
    )
    args = parser.parse_args()
    setup_logging(args.log_level, args.log_format)
    cfg = load_config(args.config) if args.config else {}

    llm = get_param(args, cfg, "llm", DEFAULT_LLM)
    if llm not in LLMClient.list_llms():
        raise ValueError(f"Unknown llm '{llm}'. Available: {', '.join(LLMClient.list_llms())}")
    adapter = get_param(args, cfg, "adapter", "base")
    expected_model_revision = str(
        get_param(args, cfg, "expected_model_revision", "")
    ).strip()
    authoritative_revision = EXPECTED_MODEL_REVISIONS.get(llm)
    if authoritative_revision is None:
        raise ValueError(f"No authoritative k3 model revision registered for llm={llm!r}")
    if expected_model_revision != authoritative_revision:
        raise ValueError(
            "expected_model_revision does not match the authoritative model snapshot: "
            f"expected={authoritative_revision}, configured={expected_model_revision!r}"
        )
    prompt_tuning = str(get_param(args, cfg, "prompt_tuning", "none")).strip().lower()
    if prompt_tuning not in {"none", "fewshot", "static_fewshot", "dynamic_fewshot"}:
        raise ValueError(
            "prompt_tuning must be 'none', 'static_fewshot', 'dynamic_fewshot', "
            "or legacy 'fewshot'"
        )
    k = int(get_param(args, cfg, "k", 3))
    if k != 3:
        raise ValueError("The additive dynamic-k3 runner requires k=3 exactly")
    results_dir_cfg = str(
        get_param(args, cfg, "results_dir", "results/k3_extension_20260717")
    ).strip()
    run_output_prefix = str(
        get_param(args, cfg, "run_output_prefix", "run_dynamic_k3")
    ).strip()
    if not results_dir_cfg:
        raise ValueError("results_dir must be non-empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_output_prefix):
        raise ValueError("run_output_prefix must contain only letters, digits, '.', '_', or '-'")
    if "k3" not in run_output_prefix.lower():
        raise ValueError("run_output_prefix must explicitly contain 'k3'")
    index_json_path_cfg = get_param(args, cfg, "index_json_path", "data/prompt_index.json")
    index_emb_path_cfg = get_param(args, cfg, "index_emb_path", "data/prompt_index_embeddings.npy")
    embedding_model_cfg = get_param(args, cfg, "embed_model", None)
    if embedding_model_cfg is None:
        embedding_model_cfg = cfg.get("embedding_model")
    embed_model_name = str(
        embedding_model_cfg or "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    if not embed_model_name:
        raise ValueError("embed_model must be non-empty")
    max_input_tokens = int(get_param(args, cfg, "max_input_tokens", 512))
    max_new_tokens = int(get_param(args, cfg, "max_new_tokens", 192))
    generation_batch_size = int(get_param(args, cfg, "generation_batch_size", 1))
    if generation_batch_size < 1:
        raise ValueError("generation_batch_size must be >= 1")
    prompt_format = str(get_param(args, cfg, "prompt_format", "auto")).strip().lower()
    if prompt_format not in {"auto", "plain", "chat", "chat_template", *V2_SQLCTX_PROMPT_FORMATS}:
        raise ValueError(
            "prompt_format must be 'auto', 'plain', 'chat', 'chat_template', "
            "'qwen_sqlctx_chatml' (preferred), legacy aliases 'qwen_v2_sqlctx', "
            "'qwen_v2_sqlctx_full_chat', 'v2_prompt_completion_chatml', "
            "'llama32_instruct_sqlctx', 'llama32_v2_sqlctx', or "
            f"'{LLAMA32_NATIVE_CHAT_FORMAT}'"
        )
    resolved_prompt_format = resolve_prompt_format(llm, prompt_format)
    chat_template = get_param(args, cfg, "chat_template", None)
    if chat_template is not None:
        chat_template = str(chat_template)
    system_prompt_variant = str(get_param(args, cfg, "system_prompt_variant", "current")).strip()
    system_prompt_path_raw = get_param(args, cfg, "system_prompt_path", None)
    system_prompt_path = None
    if system_prompt_path_raw is not None:
        system_prompt_path = str(system_prompt_path_raw).strip()
        if not system_prompt_path:
            system_prompt_path = None
    testcases_path_cfg = str(get_param(args, cfg, "testcases_path", "data/testcases.jsonl")).strip()
    traincases_path_cfg = str(get_param(args, cfg, "traincases_path", "data/traincases.jsonl")).strip()
    retrieval_pool_path_cfg = str(
        get_param(args, cfg, "retrieval_pool_path", traincases_path_cfg)
    ).strip()
    retrieval_index_path_cfg = str(
        get_param(
            args,
            cfg,
            "retrieval_index_path",
            "data/retrieval_indexes/sql_create_context_no_spider_dev_overlap_minilm",
        )
    ).strip()
    retrieval_method = str(
        get_param(
            args,
            cfg,
            "retrieval_method",
            "sentence_transformer_faiss" if prompt_tuning == "dynamic_fewshot" else "static_seeded",
        )
    ).strip()
    retrieval_rerank_method = str(
        get_param(args, cfg, "retrieval_rerank_method", "none")
    ).strip().lower()
    retrieval_rerank_top_n = int(
        get_param(args, cfg, "retrieval_rerank_top_n", 5)
    )
    retrieval_structure_bonus_max = float(
        get_param(args, cfg, "retrieval_structure_bonus_max", 0.08)
    )
    fewshot_example_schema_mode = str(
        get_param(args, cfg, "fewshot_example_schema_mode", "full")
    ).strip().lower()
    fewshot_example_mode = str(
        get_param(args, cfg, "fewshot_example_mode", "schema_with_rules")
    ).strip().lower()
    fewshot_gate_enabled = bool(get_param(args, cfg, "fewshot_gate_enabled", False))
    fewshot_gate_threshold_raw = get_param(args, cfg, "fewshot_gate_threshold", None)
    fewshot_gate_threshold = (
        float(fewshot_gate_threshold_raw)
        if fewshot_gate_threshold_raw is not None
        else None
    )
    fewshot_gate_mode_raw = get_param(args, cfg, "fewshot_gate_mode", None)
    fewshot_gate_mode = (
        str(fewshot_gate_mode_raw).strip().lower()
        if fewshot_gate_mode_raw is not None
        else None
    )
    fewshot_rerank_top_n_raw = get_param(args, cfg, "fewshot_rerank_top_n", None)
    fewshot_rerank_top_n = (
        int(fewshot_rerank_top_n_raw)
        if fewshot_rerank_top_n_raw is not None
        else None
    )
    fewshot_gate_features_raw = get_param(args, cfg, "fewshot_gate_features", [])
    if isinstance(fewshot_gate_features_raw, str):
        fewshot_gate_features = [
            item.strip().lower()
            for item in fewshot_gate_features_raw.split(",")
            if item.strip()
        ]
    elif isinstance(fewshot_gate_features_raw, list):
        fewshot_gate_features = [
            str(item).strip().lower()
            for item in fewshot_gate_features_raw
            if str(item).strip()
        ]
    else:
        raise ValueError("fewshot_gate_features must be a list or comma-separated string")
    fewshot_gate_debug = bool(get_param(args, cfg, "fewshot_gate_debug", False))
    if not testcases_path_cfg:
        raise ValueError("testcases_path must be non-empty")
    if not traincases_path_cfg:
        raise ValueError("traincases_path must be non-empty")
    if prompt_tuning in {"static_fewshot", "dynamic_fewshot"} and not retrieval_pool_path_cfg:
        raise ValueError("retrieval_pool_path must be non-empty for Few-Shot modes")
    if prompt_tuning == "dynamic_fewshot" and not retrieval_index_path_cfg:
        raise ValueError("retrieval_index_path must be non-empty for dynamic_fewshot")
    if prompt_tuning == "dynamic_fewshot" and retrieval_method != "sentence_transformer_faiss":
        raise ValueError("dynamic_fewshot requires retrieval_method='sentence_transformer_faiss'")
    if retrieval_rerank_method not in {"none", "sqlaware_topk", "structure_topk_v2"}:
        raise ValueError(
            "retrieval_rerank_method must be 'none', 'sqlaware_topk', or 'structure_topk_v2'"
        )
    if prompt_tuning != "dynamic_fewshot" and retrieval_rerank_method != "none":
        raise ValueError("retrieval_rerank_method can only be used with dynamic_fewshot")
    if fewshot_example_schema_mode not in {"full", "sql_relevant_only"}:
        raise ValueError("fewshot_example_schema_mode must be 'full' or 'sql_relevant_only'")
    if prompt_tuning not in {"static_fewshot", "dynamic_fewshot"} and fewshot_example_schema_mode != "full":
        raise ValueError(
            "fewshot_example_schema_mode can only be changed for static_fewshot "
            "or dynamic_fewshot"
        )
    if fewshot_example_mode not in {"schema_with_rules", "question_sql_only"}:
        raise ValueError(
            "fewshot_example_mode must be 'schema_with_rules' or 'question_sql_only'"
        )
    if prompt_tuning not in {"static_fewshot", "dynamic_fewshot"} and fewshot_example_mode != "schema_with_rules":
        raise ValueError(
            "fewshot_example_mode can only be changed for static_fewshot "
            "or dynamic_fewshot"
        )
    if retrieval_rerank_top_n < 1:
        raise ValueError("retrieval_rerank_top_n must be >= 1")
    if fewshot_rerank_top_n is not None and fewshot_rerank_top_n < 1:
        raise ValueError("fewshot_rerank_top_n must be >= 1 when set")
    if fewshot_gate_threshold is not None and (
        fewshot_gate_threshold < -1.0 or fewshot_gate_threshold > 2.0
    ):
        raise ValueError("fewshot_gate_threshold must be between -1.0 and 2.0")
    if fewshot_gate_enabled:
        if prompt_tuning != "dynamic_fewshot":
            raise ValueError("fewshot_gate_enabled currently requires dynamic_fewshot")
        if fewshot_gate_mode not in {
            "similarity_only",
            "weighted_score",
            "rerank_similarity",
            "set_min_similarity",
        }:
            raise ValueError(
                "fewshot_gate_mode must be one of: similarity_only, weighted_score, "
                "rerank_similarity, set_min_similarity"
            )
        if fewshot_gate_mode in {
            "similarity_only",
            "weighted_score",
            "set_min_similarity",
        } and fewshot_gate_threshold is None:
            raise ValueError(
                "fewshot_gate_threshold is required for similarity_only, weighted_score, "
                "or set_min_similarity"
            )
        if fewshot_gate_mode != "weighted_score" and fewshot_gate_features:
            raise ValueError(
                "fewshot_gate_features are currently only supported with weighted_score"
            )
        unsupported_gate_features = sorted(
            set(fewshot_gate_features) - {"sqlaware_structure_bonus"}
        )
        if unsupported_gate_features:
            raise ValueError(
                "Unsupported fewshot_gate_features: "
                + ", ".join(unsupported_gate_features)
            )
    if prompt_tuning == "dynamic_fewshot" and retrieval_rerank_method in {
        "sqlaware_topk",
        "structure_topk_v2",
    }:
        if retrieval_rerank_top_n < k:
            raise ValueError("retrieval_rerank_top_n must be >= k")
        if retrieval_structure_bonus_max < 0 or retrieval_structure_bonus_max > 0.08:
            raise ValueError("retrieval_structure_bonus_max must be between 0 and 0.08")
    max_test_samples_raw = get_param(args, cfg, "max_test_samples", None)
    max_test_samples = int(max_test_samples_raw) if max_test_samples_raw is not None else None
    extractor_mode = str(get_param(args, cfg, "extractor_mode", "legacy")).strip().lower()
    if extractor_mode not in {"legacy", "robust_v2", "robust_v3", "sql_first_statement_only", "first_statement"}:
        raise ValueError(
            "extractor_mode must be 'legacy', 'robust_v2', 'robust_v3', "
            "'sql_first_statement_only', or 'first_statement'"
        )
    progress_log_every = int(get_param(args, cfg, "progress_log_every", 25))
    compute_perplexity = bool(get_param(args, cfg, "compute_perplexity", False))
    allow_overlap = bool(get_param(args, cfg, "allow_overlap", False))
    same_db_only = bool(get_param(args, cfg, "same_db_only", False))
    dummy_mode = bool(get_param(args, cfg, "dummy_mode", False))
    dummy_sql = str(get_param(args, cfg, "dummy_sql", "SELECT 1")).strip()
    if not dummy_sql:
        raise ValueError("dummy_sql must be non-empty when provided")
    if dummy_mode and compute_perplexity:
        logger.warning("compute_perplexity is ignored in dummy_mode (no model is loaded).")
        compute_perplexity = False

    if args.config:
        logger.info("Using evaluation config: %s", args.config)
    else:
        logger.info("No evaluation config provided. Using CLI/default parameters.")
    logger.info(
        "Final eval params: llm=%s, adapter=%s, prompt_tuning=%s, k=%s, "
        "max_input_tokens=%s, max_new_tokens=%s, generation_batch_size=%s, "
        "prompt_format=%s, max_test_samples=%s, "
        "compute_perplexity=%s, allow_overlap=%s, same_db_only=%s, extractor_mode=%s, "
        "testcases_path=%s, traincases_path=%s, progress_log_every=%s, dummy_mode=%s",
        llm,
        adapter,
        prompt_tuning,
        k,
        max_input_tokens,
        max_new_tokens,
        generation_batch_size,
        resolved_prompt_format,
        max_test_samples,
        compute_perplexity,
        allow_overlap,
        same_db_only,
        extractor_mode,
        testcases_path_cfg,
        traincases_path_cfg,
        progress_log_every,
        dummy_mode,
    )
    if prompt_tuning in {"static_fewshot", "dynamic_fewshot"}:
        logger.info(
            "Retrieval params: method=%s, pool_path=%s, index_path=%s, embedding_model=%s, "
            "rerank_method=%s, rerank_top_n=%s, structure_bonus_max=%s, "
            "fewshot_example_schema_mode=%s, fewshot_example_mode=%s",
            retrieval_method,
            retrieval_pool_path_cfg,
            retrieval_index_path_cfg,
            embed_model_name,
            retrieval_rerank_method,
            retrieval_rerank_top_n,
            retrieval_structure_bonus_max,
            fewshot_example_schema_mode,
            fewshot_example_mode,
        )
        if fewshot_gate_enabled:
            logger.info(
                "Few-shot gate params: enabled=%s, mode=%s, threshold=%s, "
                "rerank_top_n=%s, features=%s, debug=%s",
                fewshot_gate_enabled,
                fewshot_gate_mode,
                fewshot_gate_threshold,
                fewshot_rerank_top_n,
                fewshot_gate_features,
                fewshot_gate_debug,
            )

    project_root = Path(__file__).resolve().parents[1]
    expected_interpreter = project_root / ".venv_flash" / "bin" / "python"
    actual_interpreter = Path(sys.executable).absolute()
    if actual_interpreter != expected_interpreter:
        raise RuntimeError(
            "The dynamic-k3 extension must run with the authoritative interpreter: "
            f"expected={expected_interpreter}, actual={actual_interpreter}"
        )
    if not dummy_mode:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU readiness is required for a full dynamic-k3 run")
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Flash Attention must be importable for a full dynamic-k3 run") from exc
    data_dir = project_root / "data"
    results_dir_candidate = Path(results_dir_cfg)
    results_dir = (
        results_dir_candidate
        if results_dir_candidate.is_absolute()
        else project_root / results_dir_candidate
    ).resolve()
    if not results_dir.is_relative_to(project_root.resolve()):
        raise ValueError("results_dir must remain inside the project root")
    results_dir.mkdir(parents=True, exist_ok=True)
    resolved_system_prompt, system_prompt_source, resolved_system_prompt_path, system_prompt_hash = resolve_system_prompt(
        project_root=project_root,
        system_prompt_variant=system_prompt_variant,
        system_prompt_path=system_prompt_path,
    )
    if resolved_prompt_format in V2_PROMPT_FORMATS:
        sqlctx_default_variants = {"current", "strict_sql_only", "v2_sqlctx_default", "sqlctx_default"}
        system_prompt_variant_norm = system_prompt_variant.strip().lower()
        if system_prompt_path is None and system_prompt_variant_norm in sqlctx_default_variants:
            resolved_system_prompt = V2_SQLCTX_SYSTEM_PROMPT
            system_prompt_source = "qwen_sqlctx_chatml"
            resolved_system_prompt_path = None
            system_prompt_hash = _sha256_text(resolved_system_prompt)
            logger.info(
                "Prompt format %s uses the Qwen SQLCTX ChatML training system prompt "
                "for default-compatible system prompt variants.",
                resolved_prompt_format,
            )
        else:
            if system_prompt_source == "variant":
                system_prompt_source = f"qwen_sqlctx_chatml_variant:{system_prompt_variant_norm}"
            logger.info(
                "Prompt format %s uses resolved Qwen SQLCTX ChatML-compatible system prompt source: %s",
                resolved_prompt_format,
                system_prompt_source,
            )
    logger.info("System prompt variant: %s", system_prompt_variant)
    logger.info("System prompt path: %s", system_prompt_path)
    logger.info("System prompt resolved_source: %s", system_prompt_source)
    logger.info("System prompt resolved_path: %s", resolved_system_prompt_path)
    logger.info("System prompt sha256: %s", system_prompt_hash)
    index_json_path = Path(index_json_path_cfg)
    if not index_json_path.is_absolute():
        index_json_path = project_root / index_json_path
    index_emb_path = Path(index_emb_path_cfg)
    if not index_emb_path.is_absolute():
        index_emb_path = project_root / index_emb_path
    testcases_path = Path(testcases_path_cfg)
    if not testcases_path.is_absolute():
        testcases_path = project_root / testcases_path
    traincases_path = Path(traincases_path_cfg)
    if not traincases_path.is_absolute():
        traincases_path = project_root / traincases_path
    retrieval_pool_path = Path(retrieval_pool_path_cfg)
    if not retrieval_pool_path.is_absolute():
        retrieval_pool_path = project_root / retrieval_pool_path
    retrieval_index_path = Path(retrieval_index_path_cfg)
    if not retrieval_index_path.is_absolute():
        retrieval_index_path = project_root / retrieval_index_path

    schema_path = data_dir / "schema_prompt.txt"

    schema_txt_global = normalize_schema_text(schema_path.read_text(encoding="utf-8"))

    if not testcases_path.exists():
        raise FileNotFoundError(f"Missing testcases JSONL: {testcases_path}")

    testcases: list[dict[str, Any]] = []
    with testcases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                testcases.append(json.loads(line))

    total_loaded_testcases = len(testcases)
    if max_test_samples is not None:
        if max_test_samples < 1:
            raise ValueError("max_test_samples must be >= 1 or unset")
        testcases = testcases[:max_test_samples]
        logger.info(
            f"Loaded {total_loaded_testcases} testcases from {testcases_path}; "
            f"using first {len(testcases)} (max_test_samples={max_test_samples})"
        )
    else:
        logger.info(
            f"Loaded {total_loaded_testcases} testcases from {testcases_path}; "
            f"using all {len(testcases)} testcases"
        )

    q_overlap, s_overlap, both_overlap = train_test_overlap_stats(traincases_path, testcases)
    if not allow_overlap and (q_overlap > 0 or s_overlap > 0):
        raise RuntimeError(
            "Train/Test overlap detected. "
            f"question_overlap={q_overlap}, sql_overlap={s_overlap}, both_overlap={both_overlap}. "
            "Run src/00_prepare_spider_subset.py to generate Spider dataset."
        )

    index_items = None
    index_emb = None
    embedder = None
    fewshot_retriever = None
    leakage_guard = LeakageGuard.from_testcases_path(testcases_path)
    if prompt_tuning == "fewshot":
        logger.info("Few-shot retrieval: index_json_path=%s", index_json_path)
        logger.info("Few-shot retrieval: index_emb_path=%s", index_emb_path)
        logger.info("Few-shot retrieval: embed_model=%s", embed_model_name)
        logger.info("Few-shot retrieval: k=%s", k)
        if not index_json_path.exists() or not index_emb_path.exists():
            raise FileNotFoundError("Prompt index files not found. Run src/09_build_prompt_index.py first.")
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Legacy prompt_tuning='fewshot' requires numpy and sentence-transformers. "
                "Run src/check_embedding_retrieval_env.py before using Few-Shot modes."
            ) from exc
        index_items = json.loads(index_json_path.read_text(encoding="utf-8"))
        index_emb = np.load(index_emb_path).astype(np.float32)
        embedder = SentenceTransformer(embed_model_name)
    elif prompt_tuning == "static_fewshot":
        if not retrieval_pool_path.exists():
            raise FileNotFoundError(f"Missing retrieval_pool_path: {retrieval_pool_path}")
        retrieval_examples = load_retrieval_pool(retrieval_pool_path)
        if not retrieval_examples:
            raise RuntimeError(f"No usable retrieval examples found in {retrieval_pool_path}")
        fewshot_retriever = StaticFewShotRetriever(
            examples=retrieval_examples,
            k=k,
            seed=42,
            allow_overlap=allow_overlap,
            same_db_only=same_db_only,
            leakage_guard=leakage_guard,
            retrieval_pool_path=retrieval_pool_path,
        )
        logger.info(
            "Static few-shot ready: pool=%s, examples=%s, seed=42, k=%s",
            retrieval_pool_path,
            len(retrieval_examples),
            k,
        )
    elif prompt_tuning == "dynamic_fewshot":
        fewshot_retriever = FaissFewShotRetriever(
            index_dir=retrieval_index_path,
            embedding_model=embed_model_name,
            k=k,
            allow_overlap=allow_overlap,
            same_db_only=same_db_only,
            leakage_guard=leakage_guard,
            retrieval_pool_path=retrieval_pool_path,
            rerank_method=retrieval_rerank_method,
            rerank_top_n=retrieval_rerank_top_n,
            structure_bonus_max=retrieval_structure_bonus_max,
        )
        logger.info(
            "Dynamic few-shot ready: index=%s, embedding_model=%s, k=%s, "
            "rerank_method=%s, rerank_top_n=%s, structure_bonus_max=%s",
            retrieval_index_path,
            embed_model_name,
            k,
            retrieval_rerank_method,
            retrieval_rerank_top_n,
            retrieval_structure_bonus_max,
        )

    client = LLMClient(project_root)
    model_id = client.resolve_model_id(llm)
    # Pin the existing loader to the immutable revision for this process only.
    LLMClient.MODEL_REVISIONS[llm] = expected_model_revision
    model_snapshot_provenance = _local_model_snapshot_provenance(
        model_id,
        expected_model_revision,
    )
    if adapter.lower() == "base":
        adapter_provenance: dict[str, Any] = {
            "adapter": "base",
            "adapter_path": None,
            "adapter_model_sha256": None,
            "adapter_config_sha256": None,
        }
    else:
        adapter_dir = project_root / "adapters" / llm / adapter
        adapter_model_path = adapter_dir / "adapter_model.safetensors"
        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_model_path.is_file() or not adapter_config_path.is_file():
            raise RuntimeError(f"Incomplete adapter root: {adapter_dir}")
        adapter_provenance = {
            "adapter": adapter,
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_model_sha256": _sha256_file(adapter_model_path),
            "adapter_config_sha256": _sha256_file(adapter_config_path),
        }

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = client.get_tokenizer(llm)
    prompt_add_special_tokens = True
    generation_eos_token_ids: list[int] | None = None
    generation_pad_token_id: int | None = None
    native_assistant_prefix: str | None = None
    if resolved_prompt_format == LLAMA32_NATIVE_CHAT_FORMAT:
        prompt_add_special_tokens = False
        generation_pad_token_id = configure_llama32_padding(tokenizer)
        generation_eos_token_ids = llama32_generation_stop_token_ids(tokenizer)
        native_assistant_prefix = llama32_assistant_generation_prefix(tokenizer)
        logger.info(
            "Native Llama generation tokens: bos=%s, pad=%s, stops=%s, prefix=%r",
            tokenizer.bos_token_id,
            generation_pad_token_id,
            generation_eos_token_ids,
            native_assistant_prefix,
        )

    model = None
    if dummy_mode:
        logger.info("Dummy mode enabled: skipping model load and generation; using dummy_sql=%r", dummy_sql)
    else:
        logger.info("Loading model: %s (llm=%s, adapter=%s)", model_id, llm, adapter)
        model = client.get_model(llm, adapter)
        logger.info("MODEL DEVICE: %s", next(model.parameters()).device)

    code_provenance_paths = {
        "runner": Path(__file__).resolve(),
        "evaluator": Path(__file__).resolve(),
        "chat_formatting": project_root / "src" / "chat_formatting.py",
        "prompt_presets": project_root / "src" / "prompt_presets.py",
        "llama32_native_chat": project_root / "src" / "llama32_native_chat.py",
        "retrieval_utils": project_root / "src" / "retrieval_utils_dynamic_k3_v1.py",
        "structure_rerank_v2": project_root / "src" / "structure_rerank_v2.py",
    }
    resolved_run_config_path = None
    if args.config:
        resolved_run_config_path = Path(args.config)
        if not resolved_run_config_path.is_absolute():
            resolved_run_config_path = project_root / resolved_run_config_path
    tokenizer_backend = getattr(tokenizer, "backend_tokenizer", None)
    cuda_available = bool(torch.cuda.is_available())
    retrieval_artifact_sha256 = {
        name: _sha256_file(retrieval_index_path / name)
        for name in ("index.faiss", "metadata.jsonl", "manifest.json")
        if (retrieval_index_path / name).is_file()
    }
    run_provenance: dict[str, Any] = {
        "runner_variant": K3_RUNNER_VERSION,
        "sys_executable": sys.executable,
        "absolute_interpreter_path": str(Path(sys.executable).absolute()),
        "base_model_id": model_id,
        "base_model_revision": expected_model_revision,
        "model_snapshot": model_snapshot_provenance,
        "adapter_provenance": adapter_provenance,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_backend_sha256": (
            _sha256_text(tokenizer_backend.to_str()) if tokenizer_backend is not None else None
        ),
        "tokenizer_chat_template_sha256": (
            _sha256_text(str(tokenizer.chat_template))
            if getattr(tokenizer, "chat_template", None)
            else None
        ),
        "python": sys.version,
        "os": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "trl": _package_version("trl"),
        "accelerate": _package_version("accelerate"),
        "datasets": _package_version("datasets"),
        "flash_attn": _package_version("flash-attn"),
        "sentence_transformers": _package_version("sentence-transformers"),
        "faiss": _package_version("faiss-cpu") or _package_version("faiss-gpu"),
        "tokenizers": _package_version("tokenizers"),
        "cuda_available": cuda_available,
        "cuda_compiled_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "config_sha256": (
            _sha256_file(resolved_run_config_path)
            if resolved_run_config_path is not None and resolved_run_config_path.is_file()
            else None
        ),
        "testcases_sha256": _sha256_file(testcases_path),
        "retrieval_artifact_sha256": retrieval_artifact_sha256,
        "code_sha256": {
            name: _sha256_file(path)
            for name, path in code_provenance_paths.items()
            if path.is_file()
        },
    }

    conn_cache: dict[str, sqlite3.Connection] = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = f"{run_output_prefix}_{timestamp}"
    out_csv = results_dir / f"{output_stem}.csv"
    out_metadata_json = results_dir / f"{output_stem}_metadata.json"
    fewshot_active = prompt_tuning in {"fewshot", "static_fewshot", "dynamic_fewshot"}
    retrieval_trace_path = None
    if fewshot_active:
        retrieval_trace_dir = results_dir / "retrieval_traces"
        retrieval_trace_dir.mkdir(parents=True, exist_ok=True)
        retrieval_trace_path = retrieval_trace_dir / f"{output_stem}_retrieval_traces.jsonl"
    output_paths = [out_csv, out_metadata_json]
    if retrieval_trace_path is not None:
        output_paths.append(retrieval_trace_path)
    collisions = [str(path) for path in output_paths if path.exists()]
    if collisions:
        raise RuntimeError(f"Refusing to overwrite existing k3 outputs: {collisions}")

    total = pred_exec_ok = gold_exec_ok = exec_match_ok = 0
    string_exact_ok = normalized_exact_ok = 0
    char_acc_sum = token_acc_sum = 0.0
    ppl_sum = 0.0
    ppl_count = 0
    eval_start_wall_time = datetime.now(timezone.utc)
    token_metric_sums: dict[str, float] = {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "reasoning_tokens": 0.0,
        "generation_time_seconds": 0.0,
        "tokens_per_second": 0.0,
    }
    token_metric_counts: dict[str, int] = {key: 0 for key in token_metric_sums}
    retrieval_similarity_sum = 0.0
    retrieval_similarity_count = 0
    retrieval_similarity_min: float | None = None
    retrieval_similarity_max: float | None = None
    retrieval_used_examples_total = 0
    retrieval_filtered_total = 0
    retrieval_success_total = 0
    fewshot_gate_fewshot_total = 0
    fewshot_gate_zero_shot_total = 0
    run_config_path = args.config if args.config else ""
    run_metadata: dict[str, Any] = {
        "run_runner_variant": K3_RUNNER_VERSION,
        "run_output_prefix": run_output_prefix,
        "run_results_dir": str(results_dir),
        "run_llm": llm,
        "run_model_id": model_id,
        "run_model_revision": expected_model_revision,
        "run_adapter": adapter,
        "run_prompt_tuning": prompt_tuning,
        "run_k": k,
        "run_prompt_format": resolved_prompt_format,
        "run_system_prompt_variant": system_prompt_variant,
        "run_system_prompt_path": resolved_system_prompt_path or "",
        "run_system_prompt_sha256": system_prompt_hash,
        "run_max_input_tokens": max_input_tokens,
        "run_max_new_tokens": max_new_tokens,
        "run_generation_batch_size": generation_batch_size,
        "run_compute_perplexity": compute_perplexity,
        "run_allow_overlap": allow_overlap,
        "run_same_db_only": same_db_only,
        "run_extractor_mode": extractor_mode,
        "run_max_test_samples": max_test_samples if max_test_samples is not None else "",
        "run_config_path": run_config_path,
        "run_prompt_add_special_tokens": int(prompt_add_special_tokens),
        "run_generation_eos_token_ids": generation_eos_token_ids or [],
        "run_generation_pad_token_id": (
            generation_pad_token_id if generation_pad_token_id is not None else ""
        ),
        "run_tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "run_tokenizer_chat_template_sha256": (
            _sha256_text(str(tokenizer.chat_template))
            if getattr(tokenizer, "chat_template", None)
            else ""
        ),
    }
    if fewshot_active:
        run_metadata.update(
            {
                "run_retrieval_method": (
                    "legacy_sentence_transformer_numpy"
                    if prompt_tuning == "fewshot"
                    else retrieval_method
                ),
                "run_retrieval_pool_path": str(retrieval_pool_path),
                "run_retrieval_index_path": (
                    str(index_json_path)
                    if prompt_tuning == "fewshot"
                    else str(retrieval_index_path)
                    if prompt_tuning == "dynamic_fewshot"
                    else ""
                ),
                "run_embedding_model": embed_model_name,
                "run_retrieval_rerank_method": (
                    retrieval_rerank_method if prompt_tuning == "dynamic_fewshot" else ""
                ),
                "run_retrieval_rerank_top_n": (
                    retrieval_rerank_top_n if prompt_tuning == "dynamic_fewshot" else ""
                ),
                "run_retrieval_structure_bonus_max": (
                    retrieval_structure_bonus_max if prompt_tuning == "dynamic_fewshot" else ""
                ),
                "run_fewshot_example_schema_mode": fewshot_example_schema_mode,
                "run_fewshot_example_mode": fewshot_example_mode,
                "run_retrieval_trace_path": str(retrieval_trace_path or ""),
            }
        )
        if fewshot_gate_enabled:
            run_metadata.update(
                {
                    "run_fewshot_gate_enabled": int(fewshot_gate_enabled),
                    "run_fewshot_gate_mode": fewshot_gate_mode or "",
                    "run_fewshot_gate_threshold": (
                        fewshot_gate_threshold
                        if fewshot_gate_threshold is not None
                        else ""
                    ),
                    "run_fewshot_rerank_top_n": (
                        fewshot_rerank_top_n if fewshot_rerank_top_n is not None else ""
                    ),
                    "run_fewshot_gate_features": json.dumps(
                        fewshot_gate_features,
                        ensure_ascii=False,
                    ),
                    "run_fewshot_gate_debug": int(fewshot_gate_debug),
                }
            )

    retrieval_trace_file = (
        retrieval_trace_path.open("w", encoding="utf-8")
        if retrieval_trace_path is not None
        else None
    )

    with out_csv.open("w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "id",
            "db_id",
            "db_path",
            "question",
            "gold_sql",
            "pred_sql",
            "gold_ok",
            "pred_ok",
            "exec_match",
            "gold_error",
            "pred_error",
            "string_exact",
            "normalized_exact",
            "char_accuracy",
            "token_accuracy",
            "gold_perplexity",
            "raw_output",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "generation_time_seconds",
            "tokens_per_second",
            "run_runner_variant",
            "run_output_prefix",
            "run_results_dir",
            "run_llm",
            "run_model_id",
            "run_model_revision",
            "run_adapter",
            "run_prompt_tuning",
            "run_k",
            "run_prompt_format",
            "run_system_prompt_variant",
            "run_system_prompt_path",
            "run_system_prompt_sha256",
            "run_max_input_tokens",
            "run_max_new_tokens",
            "run_generation_batch_size",
            "run_compute_perplexity",
            "run_allow_overlap",
            "run_same_db_only",
            "run_extractor_mode",
            "run_max_test_samples",
            "run_config_path",
            "run_prompt_add_special_tokens",
            "run_generation_eos_token_ids",
            "run_generation_pad_token_id",
            "run_tokenizer_name_or_path",
            "run_tokenizer_chat_template_sha256",
        ]
        if fewshot_active:
            fieldnames.extend(
                [
                    "retrieved_ids",
                    "retrieved_scores",
                    "retrieved_db_ids",
                    "retrieval_method",
                    "retrieval_index_path",
                    "num_fewshot_examples",
                    "run_retrieval_method",
                    "run_retrieval_pool_path",
                    "run_retrieval_index_path",
                    "run_embedding_model",
                    "run_retrieval_rerank_method",
                    "run_retrieval_rerank_top_n",
                    "run_retrieval_structure_bonus_max",
                    "run_fewshot_example_schema_mode",
                    "run_fewshot_example_mode",
                    "run_retrieval_trace_path",
                ]
            )
            if fewshot_gate_enabled:
                fieldnames.extend(
                    [
                        "gate_enabled",
                        "gate_mode",
                        "gate_score",
                        "gate_threshold",
                        "gate_decision",
                        "gate_reason",
                        "retrieval_similarity",
                        "selected_example_id",
                        "number_of_retrieved_candidates",
                        "gate_similarity_scores",
                        "gate_similarity_min",
                        "gate_similarity_max",
                        "gate_similarity_mean",
                        "gate_score_semantics",
                        "run_fewshot_gate_enabled",
                        "run_fewshot_gate_mode",
                        "run_fewshot_gate_threshold",
                        "run_fewshot_rerank_top_n",
                        "run_fewshot_gate_features",
                        "run_fewshot_gate_debug",
                    ]
                )
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        eval_start_time = time.perf_counter()
        total_cases = len(testcases)
        prompt_audit_logged = False

        def build_case_record(tc: dict[str, Any], case_number: int) -> dict[str, Any]:
            nonlocal prompt_audit_logged
            nonlocal retrieval_similarity_sum, retrieval_similarity_count
            nonlocal retrieval_similarity_min, retrieval_similarity_max
            nonlocal retrieval_used_examples_total, retrieval_filtered_total, retrieval_success_total
            nonlocal fewshot_gate_fewshot_total, fewshot_gate_zero_shot_total
            qid = tc.get("id", f"Q{case_number:03d}")
            db_id = str(tc.get("db_id", "")).strip()
            if not db_id:
                raise ValueError(
                    "Missing db_id in testcase. Spider format requires db_id for each example."
                )

            db_path_value = str(tc.get("db_path", "")).strip()
            if not db_path_value:
                raise ValueError(
                    "Missing db_path in testcase. Spider format requires db_path for each example."
                )
            db_path_case = Path(db_path_value)
            if not db_path_case.is_absolute():
                db_path_case = project_root / db_path_case

            question = tc["question"]
            gold_sql = ensure_semicolon(tc["gold_sql"])
            schema_txt_case = normalize_schema_text(str(tc.get("schema_prompt", "")))
            if not schema_txt_case:
                schema_txt_case = schema_txt_global

            retrieval_selection: FewShotSelection | None = None
            gate_decision: FewShotGateDecision | None = None
            if prompt_tuning == "fewshot":
                demos = retrieve_examples(
                    question,
                    k,
                    embedder,
                    index_items,
                    index_emb,
                    exclude_id=qid,
                    exclude_question=question,
                    query_db_id=db_id,
                    same_db_only=same_db_only,
                )
                retrieval_selection = FewShotSelection(
                    examples=demos,
                    scores=[],
                    filtered_count=0,
                    filtered_reasons={},
                    retrieval_method="legacy_sentence_transformer_numpy",
                    retrieval_index_path=str(index_json_path),
                    retrieval_pool_path=str(traincases_path),
                    retrieval_success=len(demos) >= k,
                )
                prompt = build_prompt_fewshot(
                    schema_txt_case,
                    question,
                    demos,
                    llm,
                    tokenizer,
                    prompt_format=prompt_format,
                    chat_template=chat_template,
                    system_instruction=resolved_system_prompt,
                )
            elif prompt_tuning in {"static_fewshot", "dynamic_fewshot"}:
                if fewshot_retriever is None:
                    raise RuntimeError(f"Few-shot retriever not initialized for mode: {prompt_tuning}")
                selection_kwargs = {
                    "question": question,
                    "qid": str(qid),
                    "db_id": db_id,
                }
                if prompt_tuning == "dynamic_fewshot":
                    selection_kwargs["target_schema"] = schema_txt_case
                retrieval_selection = fewshot_retriever.select(**selection_kwargs)
                selected_ids = retrieval_selection.ids()
                if len(retrieval_selection.examples) != 3:
                    raise RuntimeError(
                        f"Dynamic k3 selection returned {len(retrieval_selection.examples)} "
                        f"demonstrations for case {qid}; exactly 3 are required"
                    )
                if len(set(selected_ids)) != 3:
                    raise RuntimeError(
                        f"Dynamic k3 selection contains duplicate demonstration IDs for case {qid}: "
                        f"{selected_ids}"
                    )
                if len(retrieval_selection.scores) != 3:
                    raise RuntimeError(
                        f"Dynamic k3 selection lacks one BGE score per demonstration for case {qid}"
                    )
                gate_decision = evaluate_fewshot_gate(
                    enabled=fewshot_gate_enabled,
                    mode=fewshot_gate_mode,
                    threshold=fewshot_gate_threshold,
                    features=fewshot_gate_features,
                    selection=retrieval_selection,
                    question=question,
                    debug_enabled=fewshot_gate_debug,
                )
                if gate_decision.enabled and gate_decision.decision == "zero_shot":
                    fewshot_gate_zero_shot_total += 1
                    prompt = build_prompt(
                        schema_txt_case,
                        question,
                        llm,
                        tokenizer,
                        prompt_format=prompt_format,
                        chat_template=chat_template,
                        system_instruction=resolved_system_prompt,
                    )
                else:
                    if gate_decision.enabled:
                        fewshot_gate_fewshot_total += 1
                    prompt = build_prompt_schema_fewshot(
                        schema_txt_case,
                        question,
                        retrieval_selection.examples,
                        llm,
                        tokenizer,
                        prompt_format=prompt_format,
                        chat_template=chat_template,
                        system_instruction=resolved_system_prompt,
                        example_schema_mode=fewshot_example_schema_mode,
                        example_mode=fewshot_example_mode,
                    )
            else:
                prompt = build_prompt(
                    schema_txt_case,
                    question,
                    llm,
                    tokenizer,
                    prompt_format=prompt_format,
                    chat_template=chat_template,
                    system_instruction=resolved_system_prompt,
                )

            retrieval_csv_values: dict[str, str | int] = {}
            if retrieval_selection is not None:
                retrieval_used_examples_total += len(retrieval_selection.examples)
                retrieval_filtered_total += retrieval_selection.filtered_count
                retrieval_success_total += int(retrieval_selection.retrieval_success)
                if retrieval_selection.scores:
                    retrieval_similarity_sum += sum(retrieval_selection.scores)
                    retrieval_similarity_count += len(retrieval_selection.scores)
                    score_min = min(retrieval_selection.scores)
                    score_max = max(retrieval_selection.scores)
                    retrieval_similarity_min = (
                        score_min
                        if retrieval_similarity_min is None
                        else min(retrieval_similarity_min, score_min)
                    )
                    retrieval_similarity_max = (
                        score_max
                        if retrieval_similarity_max is None
                        else max(retrieval_similarity_max, score_max)
                    )
                retrieval_csv_values = selection_to_csv_values(retrieval_selection)
                if fewshot_gate_enabled and gate_decision is not None:
                    retrieval_csv_values.update(gate_decision_to_csv_values(gate_decision))
                trace = selection_to_trace(
                    qid=str(qid),
                    db_id=db_id,
                    question=str(question),
                    selection=retrieval_selection,
                    prompt_char_length=len(prompt),
                )
                if fewshot_gate_enabled and gate_decision is not None:
                    trace.update(
                        {
                            "gate_enabled": gate_decision.enabled,
                            "gate_mode": gate_decision.mode,
                            "gate_score": gate_decision.score,
                            "gate_threshold": gate_decision.threshold,
                            "gate_decision": gate_decision.decision,
                            "gate_reason": gate_decision.reason,
                            "retrieval_similarity": gate_decision.retrieval_similarity,
                            "selected_example_id": gate_decision.selected_example_id,
                            "number_of_retrieved_candidates": gate_decision.number_of_retrieved_candidates,
                            "selected_original_bge_scores": list(
                                gate_decision.retrieval_similarities
                            ),
                            "selected_original_bge_score_min": (
                                gate_decision.retrieval_similarity_min
                            ),
                            "selected_original_bge_score_max": (
                                gate_decision.retrieval_similarity_max
                            ),
                            "selected_original_bge_score_mean": (
                                gate_decision.retrieval_similarity_mean
                            ),
                            "gate_score_semantics": gate_decision.score_semantics,
                        }
                    )
                    if fewshot_gate_debug:
                        trace["gate_debug"] = gate_decision.debug
                if retrieval_trace_file is not None:
                    retrieval_trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")

            if not prompt_audit_logged:
                prompt_tail = prompt[-200:]
                prompt_contains_think = bool(re.search(r"(?i)<think|</think>", prompt))
                prompt_contains_qwen_chatml = bool(IM_TOKEN_RE.search(prompt))
                prompt_ends_with_qwen_assistant = prompt.endswith(V2_SQLCTX_ASSISTANT_PREFIX)
                prompt_ends_with_llama_assistant = prompt.endswith(
                    LLAMA32_V2_SQLCTX_ASSISTANT_PREFIX
                )
                if resolved_prompt_format == LLAMA32_NATIVE_CHAT_FORMAT:
                    prompt_assistant_ok = bool(
                        native_assistant_prefix and prompt.endswith(native_assistant_prefix)
                    )
                    prompt_forbidden_tokens_found = prompt_contains_qwen_chatml
                elif resolved_prompt_format in LLAMA32_V2_PROMPT_FORMATS:
                    prompt_assistant_ok = prompt_ends_with_llama_assistant
                    prompt_forbidden_tokens_found = prompt_contains_qwen_chatml
                elif resolved_prompt_format in V2_PROMPT_FORMATS:
                    prompt_assistant_ok = prompt_ends_with_qwen_assistant
                    prompt_forbidden_tokens_found = False
                else:
                    prompt_assistant_ok = True
                    prompt_forbidden_tokens_found = False
                logger.info("Prompt audit: tail_last_200=%r", prompt_tail)
                logger.info("Prompt audit: contains_think=%s", prompt_contains_think)
                logger.info("Prompt audit: contains_qwen_chatml=%s", prompt_contains_qwen_chatml)
                logger.info(
                    "Prompt audit: ends_with_qwen_assistant_prefix=%s",
                    prompt_ends_with_qwen_assistant,
                )
                logger.info(
                    "Prompt audit: ends_with_llama_assistant_prefix=%s",
                    prompt_ends_with_llama_assistant,
                )
                logger.info("Prompt audit: assistant_prefix_ok=%s", prompt_assistant_ok)
                logger.info(
                    "Prompt audit: forbidden_prompt_tokens_found=%s",
                    prompt_forbidden_tokens_found,
                )
                prompt_audit_logged = True

            return {
                "qid": qid,
                "db_id": db_id,
                "db_path_value": db_path_value,
                "db_path_case": db_path_case,
                "question": question,
                "gold_sql": gold_sql,
                "prompt": prompt,
                "retrieval_csv_values": retrieval_csv_values,
            }

        def update_metric(name: str, value: int | float | None) -> None:
            if value is None:
                return
            token_metric_sums[name] += float(value)
            token_metric_counts[name] += 1

        def metric_csv_value(value: int | float | None) -> str | int:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6f}"
            return value

        def write_case_result(record: dict[str, Any], generation_result: GenerationResult) -> None:
            nonlocal total, pred_exec_ok, gold_exec_ok, exec_match_ok
            nonlocal string_exact_ok, normalized_exact_ok, char_acc_sum, token_acc_sum
            nonlocal ppl_sum, ppl_count

            total += 1
            decoded = generation_result.text
            qid = record["qid"]
            db_id = record["db_id"]
            db_path_value = record["db_path_value"]
            db_path_case = record["db_path_case"]
            question = record["question"]
            gold_sql = record["gold_sql"]
            prompt = record["prompt"]

            pred_sql = ensure_semicolon(extract_sql_by_mode(decoded, extractor_mode) or "")

            db_error: str | None = None
            cache_key = str(db_path_case)
            conn = conn_cache.get(cache_key)
            if conn is None:
                if not db_path_case.exists():
                    db_error = f"DB file not found: {db_path_case}"
                    logger.error("DB missing for id=%s db_id=%s: %s", qid, db_id or "n/a", db_path_case)
                else:
                    try:
                        db_uri = f"file:{db_path_case.resolve()}?mode=ro"
                        conn = sqlite3.connect(db_uri, uri=True)
                        conn.execute("PRAGMA foreign_keys = ON;")
                        conn_cache[cache_key] = conn
                        logger.info("Opened DB connection for db_id=%s path=%s", db_id or "n/a", db_path_case)
                    except Exception as e:
                        db_error = f"DB connect failed: {repr(e)}"
                        logger.error(
                            "DB connection failed for id=%s db_id=%s path=%s: %s",
                            qid,
                            db_id or "n/a",
                            db_path_case,
                            repr(e),
                        )

            if db_error is None and conn is not None:
                gold_res = run_sql(conn, gold_sql)
                pred_res = run_sql(conn, pred_sql) if pred_sql else ExecResult(False, None, "No SQL extracted")
            else:
                err = db_error or "DB unavailable"
                gold_res = ExecResult(False, None, err)
                if pred_sql:
                    pred_res = ExecResult(False, None, err)
                else:
                    pred_res = ExecResult(False, None, f"{err}; No SQL extracted")

            if gold_res.ok:
                gold_exec_ok += 1

            if pred_res.ok:
                pred_exec_ok += 1

            match = execution_match(pred_res, gold_res)
            if match:
                exec_match_ok += 1

            string_exact = int(pred_sql == gold_sql)
            normalized_exact = int(normalize_sql_for_exact(pred_sql) == normalize_sql_for_exact(gold_sql))
            c_acc = char_accuracy(pred_sql, gold_sql)
            t_acc = token_accuracy(pred_sql, gold_sql)
            string_exact_ok += string_exact
            normalized_exact_ok += normalized_exact
            char_acc_sum += c_acc
            token_acc_sum += t_acc

            gold_ppl = None
            if compute_perplexity:
                target_text = build_target_block(gold_sql)
                gold_ppl = compute_target_perplexity(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    target_text=target_text,
                    max_length=max_input_tokens,
                )
                if gold_ppl is not None:
                    ppl_sum += gold_ppl
                    ppl_count += 1

            update_metric("prompt_tokens", generation_result.prompt_tokens)
            update_metric("completion_tokens", generation_result.completion_tokens)
            update_metric("total_tokens", generation_result.total_tokens)
            update_metric("reasoning_tokens", generation_result.reasoning_tokens)
            update_metric("generation_time_seconds", generation_result.generation_time_seconds)
            update_metric("tokens_per_second", generation_result.tokens_per_second)

            writer.writerow(
                {
                    "id": qid,
                    "db_id": db_id,
                    "db_path": db_path_value,
                    "question": question,
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "gold_ok": int(gold_res.ok),
                    "pred_ok": int(pred_res.ok),
                    "exec_match": int(match),
                    "gold_error": gold_res.error or "",
                    "pred_error": pred_res.error or "",
                    "string_exact": string_exact,
                    "normalized_exact": normalized_exact,
                    "char_accuracy": f"{c_acc:.6f}",
                    "token_accuracy": f"{t_acc:.6f}",
                    "gold_perplexity": f"{gold_ppl:.6f}" if gold_ppl is not None else "",
                    "raw_output": decoded,
                    "prompt_tokens": metric_csv_value(generation_result.prompt_tokens),
                    "completion_tokens": metric_csv_value(generation_result.completion_tokens),
                    "total_tokens": metric_csv_value(generation_result.total_tokens),
                    "reasoning_tokens": metric_csv_value(generation_result.reasoning_tokens),
                    "generation_time_seconds": metric_csv_value(generation_result.generation_time_seconds),
                    "tokens_per_second": metric_csv_value(generation_result.tokens_per_second),
                    **(record["retrieval_csv_values"] if fewshot_active else {}),
                    **run_metadata,
                }
            )

            if progress_log_every > 0 and total_cases > 0:
                if (total % progress_log_every == 0) or (total == total_cases):
                    elapsed_sec = time.perf_counter() - eval_start_time
                    avg_sec = elapsed_sec / total
                    remaining = total_cases - total
                    eta_sec = avg_sec * remaining
                    pct = (total / total_cases) * 100.0
                    logger.info(
                        "Progress: %s/%s (%.1f%%) | elapsed=%s | eta=%s | avg=%.2fs/case",
                        total,
                        total_cases,
                        pct,
                        _format_duration_hms(elapsed_sec),
                        _format_duration_hms(eta_sec),
                        avg_sec,
                    )

        if generation_batch_size == 1:
            for tc in testcases:
                record = build_case_record(tc, total + 1)
                if dummy_mode:
                    generation_result = GenerationResult(
                        text=dummy_sql,
                        generation_time_seconds=0.0,
                    )
                else:
                    generation_result = _decode_single_generation_with_metrics(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=record["prompt"],
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                        add_special_tokens=prompt_add_special_tokens,
                        eos_token_ids=generation_eos_token_ids,
                        pad_token_id=generation_pad_token_id,
                    )
                write_case_result(record, generation_result)
        else:
            logger.info("Batch generation enabled: generation_batch_size=%s", generation_batch_size)
            for chunk in _iter_chunks(testcases, generation_batch_size):
                records = [
                    build_case_record(tc, total + idx + 1)
                    for idx, tc in enumerate(chunk)
                ]
                if dummy_mode:
                    generation_results = [
                        GenerationResult(text=dummy_sql, generation_time_seconds=0.0)
                        for _ in records
                    ]
                else:
                    generation_results = _decode_batch_generation_with_metrics(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=[record["prompt"] for record in records],
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                        add_special_tokens=prompt_add_special_tokens,
                        eos_token_ids=generation_eos_token_ids,
                        pad_token_id=generation_pad_token_id,
                    )
                for record, generation_result in zip(records, generation_results):
                    write_case_result(record, generation_result)

    if retrieval_trace_file is not None:
        retrieval_trace_file.close()

    for cache_conn in conn_cache.values():
        cache_conn.close()

    eval_end_wall_time = datetime.now(timezone.utc)
    run_metric_averages = {
        key: (
            token_metric_sums[key] / token_metric_counts[key]
            if token_metric_counts[key] > 0
            else None
        )
        for key in token_metric_sums
    }
    run_metric_totals = {
        key: (
            token_metric_sums[key]
            if token_metric_counts[key] > 0
            else None
        )
        for key in token_metric_sums
    }
    if total > 0:
        total_elapsed_sec = time.perf_counter() - eval_start_time
        logger.info(
            "Evaluation runtime: %s total (avg %.2fs/case)",
            _format_duration_hms(total_elapsed_sec),
            total_elapsed_sec / total,
        )
    else:
        total_elapsed_sec = 0.0

    run_summary_metadata = {
        **run_metadata,
        "start_time": eval_start_wall_time.isoformat(),
        "end_time": eval_end_wall_time.isoformat(),
        "duration_seconds": total_elapsed_sec,
        "duration_human_readable": _format_duration_hms(total_elapsed_sec),
        "total_testcases": total,
        "execution_success_rate": (pred_exec_ok / total) if total else None,
        "execution_match_accuracy": (exec_match_ok / total) if total else None,
        "string_exact_match": (string_exact_ok / total) if total else None,
        "normalized_exact_match": (normalized_exact_ok / total) if total else None,
        "char_accuracy_avg": (char_acc_sum / total) if total else None,
        "token_accuracy_avg": (token_acc_sum / total) if total else None,
        "avg_prompt_tokens": run_metric_averages["prompt_tokens"],
        "avg_completion_tokens": run_metric_averages["completion_tokens"],
        "avg_total_tokens": run_metric_averages["total_tokens"],
        "avg_reasoning_tokens": run_metric_averages["reasoning_tokens"],
        "avg_generation_time_seconds": run_metric_averages["generation_time_seconds"],
        "avg_tokens_per_second": run_metric_averages["tokens_per_second"],
        "sum_prompt_tokens": run_metric_totals["prompt_tokens"],
        "sum_completion_tokens": run_metric_totals["completion_tokens"],
        "sum_total_tokens": run_metric_totals["total_tokens"],
        "sum_reasoning_tokens": run_metric_totals["reasoning_tokens"],
        "sum_generation_time_seconds": run_metric_totals["generation_time_seconds"],
        "csv_path": str(out_csv),
        "provenance": run_provenance,
    }
    if fewshot_active:
        run_summary_metadata.update(
            {
                "retrieval_method": run_metadata.get("run_retrieval_method", ""),
                "retrieval_index_path": run_metadata.get("run_retrieval_index_path", ""),
                "retrieval_pool_path": run_metadata.get("run_retrieval_pool_path", ""),
                "retrieval_trace_path": str(retrieval_trace_path or ""),
                "retrieval_k": k,
                "retrieval_avg_similarity": (
                    retrieval_similarity_sum / retrieval_similarity_count
                    if retrieval_similarity_count
                    else None
                ),
                "retrieval_min_similarity": retrieval_similarity_min,
                "retrieval_max_similarity": retrieval_similarity_max,
                "retrieval_total_used_examples": retrieval_used_examples_total,
                "retrieval_avg_used_examples": (
                    retrieval_used_examples_total / total if total else None
                ),
                "retrieval_filtered_total": retrieval_filtered_total,
                "retrieval_success_rate": (
                    retrieval_success_total / total if total else None
                ),
            }
        )
        if fewshot_gate_enabled:
            run_summary_metadata.update(
                {
                    "fewshot_gate_enabled": True,
                    "fewshot_gate_mode": fewshot_gate_mode,
                    "fewshot_gate_threshold": fewshot_gate_threshold,
                    "fewshot_rerank_top_n": fewshot_rerank_top_n,
                    "fewshot_gate_features": fewshot_gate_features,
                    "fewshot_gate_debug": fewshot_gate_debug,
                    "fewshot_gate_fewshot_total": fewshot_gate_fewshot_total,
                    "fewshot_gate_zero_shot_total": fewshot_gate_zero_shot_total,
                    "fewshot_gate_fewshot_rate": (
                        fewshot_gate_fewshot_total / total if total else None
                    ),
                }
            )
    out_metadata_json.write_text(
        json.dumps(run_summary_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("=== SUMMARY ===")
    logger.info("LLM: %s", llm)
    logger.info("Adapter: %s", adapter)
    logger.info("Prompt mode: %s (k=%s)", prompt_tuning, k)
    logger.info(
        "Generation: max_input_tokens=%s, max_new_tokens=%s",
        max_input_tokens,
        max_new_tokens,
    )
    logger.info("[Classical ML-Style Metrics]")
    logger.info("String Exact Match: %.2f%%", (string_exact_ok / total) * 100.0)
    logger.info("Normalized Exact Match: %.2f%%", (normalized_exact_ok / total) * 100.0)
    logger.info("Char Accuracy (avg): %.2f%%", (char_acc_sum / total) * 100.0)
    logger.info("Token Accuracy (avg): %.2f%%", (token_acc_sum / total) * 100.0)
    if compute_perplexity:
        if ppl_count > 0:
            logger.info("Gold-Target Perplexity (avg): %.4f (n=%s)", (ppl_sum / ppl_count), ppl_count)
        else:
            logger.info("Gold-Target Perplexity (avg): n/a")
    logger.info("[Execution-Based Metrics]")
    logger.info("Total testcases: %s", total)
    logger.info("Execution Success Rate: %.2f%%", (pred_exec_ok / total) * 100.0)
    logger.info("Execution Match Accuracy: %.2f%%", (exec_match_ok / total) * 100.0)
    logger.info("[Generation Token/Runtime Metrics]")
    logger.info("Avg prompt tokens: %s", run_metric_averages["prompt_tokens"])
    logger.info("Avg completion tokens: %s", run_metric_averages["completion_tokens"])
    logger.info("Avg total tokens: %s", run_metric_averages["total_tokens"])
    logger.info("Avg reasoning tokens: %s", run_metric_averages["reasoning_tokens"])
    logger.info("Avg generation time seconds: %s", run_metric_averages["generation_time_seconds"])
    logger.info("Avg tokens per second: %s", run_metric_averages["tokens_per_second"])
    if fewshot_active:
        logger.info("[Retrieval Metrics]")
        logger.info("Retrieval method: %s", run_summary_metadata["retrieval_method"])
        logger.info("Retrieval index path: %s", run_summary_metadata["retrieval_index_path"])
        logger.info("Retrieval k: %s", k)
        logger.info("Avg similarity: %s", run_summary_metadata["retrieval_avg_similarity"])
        logger.info("Min similarity: %s", retrieval_similarity_min)
        logger.info("Max similarity: %s", retrieval_similarity_max)
        logger.info("Avg few-shot examples: %s", run_summary_metadata["retrieval_avg_used_examples"])
        logger.info("Filtered retrieval hits: %s", retrieval_filtered_total)
        logger.info("Retrieval success rate: %s", run_summary_metadata["retrieval_success_rate"])
        logger.info("Retrieval trace written to: %s", retrieval_trace_path)
        if fewshot_gate_enabled:
            logger.info("[Few-Shot Gate Metrics]")
            logger.info("Gate mode: %s", fewshot_gate_mode)
            logger.info("Gate threshold: %s", fewshot_gate_threshold)
            logger.info("Gate chose few-shot: %s", fewshot_gate_fewshot_total)
            logger.info("Gate chose zero-shot: %s", fewshot_gate_zero_shot_total)
    logger.info("CSV written to: %s", out_csv)
    logger.info("Run metadata written to: %s", out_metadata_json)


if __name__ == "__main__":
    import sys

    if "--selftest-extract-sql" in sys.argv:
        setup_logging("INFO")
        _run_extract_sql_selftests()
        logger.info("extract_sql selftests passed.")
    elif "--selftest-execution-match" in sys.argv:
        setup_logging("INFO")
        _run_execution_match_selftests()
        logger.info("execution_match selftests passed.")
    elif "--selftest-batch-slicing" in sys.argv:
        setup_logging("INFO")
        _run_batch_slicing_selftests()
        logger.info("batch slicing selftests passed.")
    else:
        main()
