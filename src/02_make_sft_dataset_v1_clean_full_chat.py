#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.config import get_param, load_config
except ModuleNotFoundError:
    from config import get_param, load_config

try:
    from src.logging_utils import setup_logging
except ModuleNotFoundError:
    from logging_utils import setup_logging

try:
    from src.prompt_presets import resolve_system_prompt
except ModuleNotFoundError:
    from prompt_presets import resolve_system_prompt


logger = logging.getLogger(__name__)

SQL_START_RE = re.compile(r"(?is)^\s*(select|with)\b")
THINK_BLOCK_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")
THINK_TAG_RE = re.compile(r"(?i)</?think\b[^>]*>")
QWEN_SQLCTX_CHATML_FORMAT = "qwen_sqlctx_chatml"
QWEN_SQLCTX_LEGACY_CHAT_FORMATS = {"qwen_v2_sqlctx", "qwen_v2_sqlctx_full_chat"}
QWEN_V2_CHAT_FORMAT = "qwen_v2_sqlctx_full_chat"
LLAMA32_V2_CHAT_FORMAT = "llama32_v2_sqlctx_full_chat"
LLAMA32_INSTRUCT_SQLCTX_FORMAT = "llama32_instruct_sqlctx"
LLAMA32_CHAT_FORMATS = {LLAMA32_V2_CHAT_FORMAT, LLAMA32_INSTRUCT_SQLCTX_FORMAT}
SUPPORTED_CHAT_FORMATS = {
    QWEN_SQLCTX_CHATML_FORMAT,
    *QWEN_SQLCTX_LEGACY_CHAT_FORMATS,
    *LLAMA32_CHAT_FORMATS,
}
QWEN_ASSISTANT_MARKER = "<|im_start|>assistant\n"
LLAMA32_ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

USER_PROMPT_TEMPLATE = """Database schema:
{schema}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Pipeline v1-clean full-ChatML SFT dataset for NL2SQL."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config path")
    parser.add_argument("--input_path", default=None, help="Overlap-clean source JSONL")
    parser.add_argument("--output_path", default=None, help="Output full-chat JSONL")
    parser.add_argument("--manifest_path", default=None, help="Output manifest JSON")
    parser.add_argument(
        "--system_prompt_variant",
        default=None,
        help="System prompt preset variant. Preferred default: sqlctx_default. Legacy alias: v2_sqlctx_default.",
    )
    parser.add_argument(
        "--system_prompt_path",
        default=None,
        help="Path to a text file containing the system prompt.",
    )
    parser.add_argument(
        "--chat_format",
        default=None,
        choices=sorted(SUPPORTED_CHAT_FORMATS),
        help=(
            "Chat serialization format. Preferred: qwen_sqlctx_chatml. "
            "Legacy aliases: qwen_v2_sqlctx, qwen_v2_sqlctx_full_chat. "
            "Use llama32_instruct_sqlctx for Llama 3.2 Instruct."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output/manifest files if they already exist.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_schema_text(schema_text: str) -> str:
    lines = schema_text.strip().splitlines()
    if lines and lines[0].strip().lower() == "database schema:":
        lines = lines[1:]
    return "\n".join(lines).strip()


def strip_think_tags(text: str) -> str:
    text = THINK_BLOCK_RE.sub(" ", text)
    text = THINK_TAG_RE.sub(" ", text)
    return text


def split_first_statement(sql_text: str) -> tuple[str, str, bool]:
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


def validate_completion(completion: str) -> None:
    if not SQL_START_RE.match(completion):
        raise ValueError("assistant SQL does not start with SELECT or WITH")
    if not completion.rstrip().endswith(";"):
        raise ValueError("assistant SQL does not end with semicolon")
    _first_stmt, remainder, had_semicolon = split_first_statement(completion)
    if not had_semicolon:
        raise ValueError("assistant SQL has no statement-ending semicolon")
    if remainder.strip():
        raise ValueError("assistant SQL contains more than one statement")
    if re.search(r"```|<think|</think>", completion, flags=re.IGNORECASE):
        raise ValueError("assistant SQL contains markdown or thinking tags")


def sanitize_completion(gold_sql: str) -> str:
    cleaned = strip_think_tags(gold_sql).strip()
    if not cleaned:
        raise ValueError("empty SQL target")
    match = re.search(r"(?is)\b(select|with)\b", cleaned)
    if match is None:
        raise ValueError("SQL target does not contain SELECT or WITH")
    cleaned = cleaned[match.start() :].strip()
    first_stmt, remainder, had_semicolon = split_first_statement(cleaned)
    if had_semicolon and remainder.strip():
        raise ValueError("SQL target contains text after first statement")
    completion = first_stmt.strip()
    if not completion.endswith(";"):
        completion += ";"
    validate_completion(completion)
    return completion


def build_user_prompt(schema_text: str, question: str) -> str:
    return USER_PROMPT_TEMPLATE.format(schema=schema_text.strip(), question=question.strip())


def assistant_marker_for_chat_format(chat_format: str) -> str:
    chat_format = normalize_chat_format(chat_format)
    if chat_format == QWEN_SQLCTX_CHATML_FORMAT:
        return QWEN_ASSISTANT_MARKER
    if chat_format in LLAMA32_CHAT_FORMATS:
        return LLAMA32_ASSISTANT_MARKER
    raise ValueError(f"Unsupported chat_format: {chat_format}")


def build_prompt_prefix(*, system_prompt: str, user_prompt: str, chat_format: str) -> str:
    chat_format = normalize_chat_format(chat_format)
    if chat_format == QWEN_SQLCTX_CHATML_FORMAT:
        return (
            "<|im_start|>system\n"
            + system_prompt.strip()
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + user_prompt.strip()
            + "<|im_end|>\n"
            + QWEN_ASSISTANT_MARKER
        )
    if chat_format in LLAMA32_CHAT_FORMATS:
        return (
            "<|begin_of_text|>"
            + "<|start_header_id|>system<|end_header_id|>\n\n"
            + system_prompt.strip()
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n\n"
            + user_prompt.strip()
            + "<|eot_id|>"
            + LLAMA32_ASSISTANT_MARKER
        )
    raise ValueError(f"Unsupported chat_format: {chat_format}")


def assistant_end_for_chat_format(chat_format: str) -> str:
    chat_format = normalize_chat_format(chat_format)
    if chat_format == QWEN_SQLCTX_CHATML_FORMAT:
        return "<|im_end|>\n"
    if chat_format in LLAMA32_CHAT_FORMATS:
        return "<|eot_id|>"
    raise ValueError(f"Unsupported chat_format: {chat_format}")


def build_full_chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    completion: str,
    chat_format: str,
) -> str:
    return (
        build_prompt_prefix(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_format=chat_format,
        )
        + completion
        + assistant_end_for_chat_format(chat_format)
    )


def normalize_for_leak_check(value: str) -> str:
    return " ".join(value.casefold().split())


def check_no_prompt_leakage(
    prompt_prefix: str,
    completion: str,
    *,
    assistant_marker: str,
) -> list[str]:
    errors: list[str] = []
    completion_norm = normalize_for_leak_check(completion)
    prompt_norm = normalize_for_leak_check(prompt_prefix)
    if completion_norm and completion_norm in prompt_norm:
        errors.append("completion appears in prompt prefix")
    if not prompt_prefix.endswith(assistant_marker):
        errors.append("prompt prefix does not end with assistant generation marker")
    tail = prompt_prefix.split(assistant_marker, 1)[-1].strip()
    if tail:
        errors.append("assistant prompt prefix contains assistant content")
    return errors


def load_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_manifest.json")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_chat_format(chat_format: str) -> str:
    normalized = str(chat_format).strip().lower()
    if normalized in {QWEN_SQLCTX_CHATML_FORMAT, *QWEN_SQLCTX_LEGACY_CHAT_FORMATS}:
        return QWEN_SQLCTX_CHATML_FORMAT
    if normalized in LLAMA32_CHAT_FORMATS:
        return normalized
    return normalized


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config) if args.config else {}
    project_root = Path(__file__).resolve().parents[1]

    input_path_raw = str(
        get_param(
            args,
            cfg,
            "input_path",
            "data/sql_create_context/train_no_spider_dev_overlap.jsonl",
        )
    )
    output_path_raw = str(
        get_param(
            args,
            cfg,
            "output_path",
            "data/sql_create_context/train_sft_qwen35_9b_full_chat_v1_clean_no_spider_dev_overlap.jsonl",
        )
    )
    system_prompt_variant = _optional_str(
        get_param(args, cfg, "system_prompt_variant", "sqlctx_default")
    )
    system_prompt_path = _optional_str(get_param(args, cfg, "system_prompt_path", None))
    chat_format = normalize_chat_format(get_param(args, cfg, "chat_format", QWEN_SQLCTX_CHATML_FORMAT))
    if chat_format not in SUPPORTED_CHAT_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_CHAT_FORMATS))
        raise ValueError(f"chat_format must be one of: {allowed}")
    max_samples = get_param(args, cfg, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1 or null")

    system_prompt, system_prompt_source, resolved_system_prompt_path, _prompt_hash = resolve_system_prompt(
        project_root=project_root,
        system_prompt_variant=system_prompt_variant or "sqlctx_default",
        system_prompt_path=system_prompt_path,
    )

    input_path = Path(input_path_raw)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_path = Path(output_path_raw)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    manifest_raw = get_param(args, cfg, "manifest_path", None)
    manifest_path = Path(str(manifest_raw)) if manifest_raw else build_manifest_path(output_path)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    for path in (output_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    prompt_template_hash = sha256_text(
        json.dumps(
            {
                "system_prompt": system_prompt,
                "user_prompt_template": USER_PROMPT_TEMPLATE,
                "chat_format": chat_format,
                "assistant_end": assistant_end_for_chat_format(chat_format),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    rows_in = load_rows(input_path, max_samples)
    rows_out: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    leakage_errors: list[dict[str, Any]] = []
    checked_examples = 0

    for index, obj in enumerate(rows_in, start=1):
        qid = str(obj.get("id", f"SFTV1CLEAN_{index:06d}"))
        try:
            question = str(obj["question"]).strip()
            schema = normalize_schema_text(str(obj.get("schema_prompt") or obj.get("context") or ""))
            if not question:
                raise ValueError("missing question")
            if not schema:
                raise ValueError("missing schema")
            completion = sanitize_completion(str(obj["gold_sql"]))
            user_prompt = build_user_prompt(schema, question)
            prompt_prefix = build_prompt_prefix(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_format=chat_format,
            )
            checked_examples += 1
            errors = check_no_prompt_leakage(
                prompt_prefix,
                completion,
                assistant_marker=assistant_marker_for_chat_format(chat_format),
            )
            if errors:
                leakage_errors.append({"id": qid, "errors": errors})
                continue
            text = build_full_chat_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion=completion,
                chat_format=chat_format,
            )
            rows_out.append({"id": qid, "text": text})
        except Exception as exc:
            dropped.append({"id": qid, "reason": str(exc)})

    manifest_base: dict[str, Any] = {
        "pipeline_version": "v1_clean_full_chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dataset_path": str(output_path),
        "dataset_format": "full_chat_text",
        "chat_format": chat_format,
        "system_prompt_source": system_prompt_source,
        "system_prompt_variant": system_prompt_variant,
        "system_prompt_path": resolved_system_prompt_path,
        "input_examples_read": len(rows_in),
        "checked_examples": checked_examples,
        "dropped_examples": len(dropped),
        "dropped_preview": dropped[:20],
        "input_sha256": sha256_file(input_path),
        "prompt_template_sha256": prompt_template_hash,
        "system_prompt_sha256": sha256_text(system_prompt),
        "max_samples": max_samples,
    }

    if leakage_errors:
        manifest = {
            **manifest_base,
            "written_examples": 0,
            "leakage_found": True,
            "leakage_count": len(leakage_errors),
            "leakage_examples": leakage_errors[:20],
            "output_sha256": None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Prompt leakage detected in {len(leakage_errors)} example(s). "
            f"Manifest written to {manifest_path}."
        )

    write_jsonl(output_path, rows_out)
    manifest = {
        **manifest_base,
        "written_examples": len(rows_out),
        "leakage_found": False,
        "leakage_count": 0,
        "output_sha256": sha256_file(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d v1-clean full-chat examples to %s", len(rows_out), output_path)
    logger.info("Dropped %d examples", len(dropped))
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
