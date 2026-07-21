#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

from chat_formatting import build_nl2sql_messages, render_messages

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

try:
    from src.llm_client import LLMClient
except ModuleNotFoundError:
    from llm_client import LLMClient


logger = logging.getLogger(__name__)
THINK_BLOCK_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")
THINK_TAG_RE = re.compile(r"(?i)</?think\b[^>]*>")
SQL_START_RE = re.compile(r"(?i)\b(?:select|with)\b")
ASSISTANT_IM_START_TO_SQL_RE = re.compile(r"(?is)(<\|im_start\|>assistant)\s*((?:select|with)\b)")
ASSISTANT_TAG_TO_SQL_RE = re.compile(r"(?is)(<\|assistant\|>)\s*((?:select|with)\b)")
LEGACY_TRAINING_SYSTEM_PROMPT = (
    "You translate natural language questions into SQLite SQL. "
    "Return only one SQL query. Start with SELECT or WITH, end with a semicolon. "
    "No explanation. No markdown."
)


def list_llms() -> list[str]:
    return LLMClient.list_llms()


def try_get_tokenizer(project_root: Path, llm_name: str):
    """
    Best-effort tokenizer loading.
    Priority:
    1) Existing project LLMClient path (keeps original behavior)
    2) Direct AutoTokenizer loading
    3) None (fallback to plain textual chat formatting)
    """
    model_id = LLMClient.resolve_model_id(llm_name)

    try:
        client = LLMClient(project_root)
        return client.get_tokenizer(llm_name)
    except Exception as exc:
        logger.warning("Tokenizer load via LLMClient failed (%s). Falling back.", exc)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:
        logger.warning("Tokenizer load via transformers failed (%s). Using text fallback.", exc)
        return None


def uses_chat_format(llm_name: str) -> bool:
    return llm_name in {"tinyllama_11b"}


def resolve_prompt_format(llm_name: str, prompt_format: str) -> str:
    fmt = prompt_format.strip().lower()
    if fmt in {"chat", "chat_template"}:
        return "chat_template"
    if fmt == "plain":
        return "plain"
    if fmt == "auto":
        return "chat_template" if uses_chat_format(llm_name) else "plain"
    raise ValueError("prompt_format must be 'auto', 'plain', 'chat', or 'chat_template'")


def normalize_schema_text(schema_txt: str) -> str:
    """
    Backward-compatible cleanup:
    old schema_prompt.txt versions started with 'Database schema:'.
    """
    lines = schema_txt.strip().splitlines()
    if lines and lines[0].strip().lower() == "database schema:":
        lines = lines[1:]
    return "\n".join(lines).strip()


def strip_think_tags(text: str) -> str:
    text = THINK_BLOCK_RE.sub(" ", text)
    text = THINK_TAG_RE.sub(" ", text)
    return text


def sanitize_sql_target(sql_text: str) -> str:
    cleaned = strip_think_tags(sql_text).strip()
    if not cleaned:
        return ""
    start_match = SQL_START_RE.search(cleaned)
    if start_match is None:
        return ""
    return cleaned[start_match.start() :].strip()


def clean_rendered_training_text(text: str) -> str:
    cleaned = strip_think_tags(text)
    # Ensure assistant content starts directly with SQL after role marker.
    cleaned = ASSISTANT_IM_START_TO_SQL_RE.sub(r"\1\n\2", cleaned)
    cleaned = ASSISTANT_TAG_TO_SQL_RE.sub(r"\1\n\2", cleaned)
    return cleaned


def build_user_prompt(schema_txt: str, question: str) -> str:
    return f"""
You are an assistant that translates natural language questions into SQLite SQL queries.

Database schema:
{schema_txt}

Rules:
- Use only the tables and columns from the schema.
- Output exactly ONE SQL query.
- The query must be a single read query (SELECT or WITH...SELECT).
- Start directly with SELECT or WITH.
- End with a semicolon.
- Do NOT explain anything.
- Do NOT add any text before or after the SQL.
- Do NOT use markdown.

Question:
{question}

SQL:
""".strip()


def build_target(gold_sql: str) -> str:
    gold_sql = sanitize_sql_target(gold_sql)
    if not gold_sql:
        raise ValueError("gold_sql is empty after sanitization")
    if not gold_sql.endswith(";"):
        gold_sql += ";"
    return gold_sql


def build_training_text(
    schema_txt: str,
    question: str,
    gold_sql: str,
    llm_name: str,
    tokenizer,
    prompt_format: str = "auto",
    chat_template: str | None = None,
    system_instruction: str = LEGACY_TRAINING_SYSTEM_PROMPT,
) -> str:
    user_prompt = build_user_prompt(schema_txt, question)
    assistant_answer = build_target(gold_sql)
    resolved_prompt_format = resolve_prompt_format(llm_name, prompt_format)

    # Keep the existing plain SFT style exactly: user prompt + newline + target SQL.
    if resolved_prompt_format == "plain":
        return user_prompt + "\n" + assistant_answer

    messages = build_nl2sql_messages(system_instruction=system_instruction, user_prompt=user_prompt)
    messages.append({"role": "assistant", "content": assistant_answer})
    rendered = render_messages(
        tokenizer=tokenizer,
        messages=messages,
        prompt_format=resolved_prompt_format,
        chat_template=chat_template,
        add_generation_prompt=False,
    )
    return clean_rendered_training_text(rendered)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT dataset in plain or chat-template format.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file for dataset parameters.",
    )
    p.add_argument(
        "--llm",
        type=str,
        default=None,
        choices=list_llms(),
        help="LLM key. Chat-template formatting is applied automatically for chat models.",
    )
    p.add_argument(
        "--prompt_format",
        type=str,
        default=None,
        choices=["auto", "plain", "chat", "chat_template"],
        help="Prompt serialization format: auto (default), plain, chat, or chat_template.",
    )
    p.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Optional custom tokenizer chat template (Jinja).",
    )
    p.add_argument(
        "--system_prompt_variant",
        type=str,
        default=None,
        help="Optional system prompt preset variant from prompt_presets.py.",
    )
    p.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Optional path to a text file containing the system prompt.",
    )
    p.add_argument(
        "--traincases_path",
        type=str,
        default=None,
        help="Path to input traincases JSONL (default: data/traincases.jsonl).",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output SFT JSONL (default: data/train_sft.jsonl).",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return p.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_training_system_prompt(
    *,
    project_root: Path,
    system_prompt_variant: str | None,
    system_prompt_path: str | None,
) -> tuple[str, str, str | None, str]:
    """
    Resolve the training system prompt while preserving legacy default behavior.
    """
    path_raw = (system_prompt_path or "").strip()
    variant_raw = (system_prompt_variant or "").strip()

    # If neither is set, preserve exact legacy default.
    if not path_raw and not variant_raw:
        return (
            LEGACY_TRAINING_SYSTEM_PROMPT,
            "legacy_default",
            None,
            _sha256_text(LEGACY_TRAINING_SYSTEM_PROMPT),
        )

    prompt_text, source, resolved_path, prompt_hash = resolve_system_prompt(
        project_root=project_root,
        system_prompt_variant=variant_raw or "current",
        system_prompt_path=path_raw or None,
    )
    return prompt_text, source, resolved_path, prompt_hash


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config) if args.config else {}
    project_root = Path(__file__).resolve().parents[1]

    llm = get_param(args, cfg, "llm", "llama32_1b")
    if llm not in list_llms():
        raise ValueError(f"Unknown llm '{llm}'. Available: {', '.join(list_llms())}")
    prompt_format = str(get_param(args, cfg, "prompt_format", "auto")).strip().lower()
    if prompt_format not in {"auto", "plain", "chat", "chat_template"}:
        raise ValueError("prompt_format must be 'auto', 'plain', 'chat', or 'chat_template'")
    chat_template = get_param(args, cfg, "chat_template", None)
    if chat_template is not None:
        chat_template = str(chat_template)
    system_prompt_variant_raw = get_param(args, cfg, "system_prompt_variant", None)
    system_prompt_variant = None
    if system_prompt_variant_raw is not None:
        system_prompt_variant = str(system_prompt_variant_raw).strip() or None
    system_prompt_path_raw = get_param(args, cfg, "system_prompt_path", None)
    system_prompt_path = None
    if system_prompt_path_raw is not None:
        system_prompt_path = str(system_prompt_path_raw).strip() or None
    resolved_prompt_format = resolve_prompt_format(llm, prompt_format)
    resolved_system_prompt, system_prompt_source, resolved_system_prompt_path, system_prompt_hash = (
        resolve_training_system_prompt(
            project_root=project_root,
            system_prompt_variant=system_prompt_variant,
            system_prompt_path=system_prompt_path,
        )
    )

    schema_path = project_root / "data" / "schema_prompt.txt"
    traincases_path_raw = str(get_param(args, cfg, "traincases_path", "data/traincases.jsonl"))
    output_path_raw = str(get_param(args, cfg, "output_path", "data/train_sft.jsonl"))
    traincases_path = Path(traincases_path_raw)
    if not traincases_path.is_absolute():
        traincases_path = project_root / traincases_path
    output_path = Path(output_path_raw)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if not traincases_path.exists():
        raise FileNotFoundError(
            f"Missing training cases file: {traincases_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_txt_global = normalize_schema_text(schema_path.read_text(encoding="utf-8"))
    lines = [l for l in traincases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tokenizer = try_get_tokenizer(project_root, llm)
    logger.info(
        "Building SFT dataset: traincases_path=%s, output_path=%s, llm=%s, prompt_format=%s (resolved_mode=%s)",
        traincases_path,
        output_path,
        llm,
        prompt_format,
        resolved_prompt_format,
    )
    logger.info("system_prompt_variant=%s", system_prompt_variant)
    logger.info("system_prompt_path=%s", system_prompt_path)
    logger.info("system_prompt_resolved_source=%s", system_prompt_source)
    logger.info("system_prompt_resolved_path=%s", resolved_system_prompt_path)
    logger.info("system_prompt_sha256=%s", system_prompt_hash)

    rows = []
    fallback_count = 0
    for i, line in enumerate(lines, start=1):
        obj = json.loads(line)
        question = obj["question"]
        gold_sql = obj["gold_sql"]
        schema_txt = normalize_schema_text(str(obj.get("schema_prompt", "")))
        if not schema_txt:
            schema_txt = schema_txt_global
            fallback_count += 1

        text = build_training_text(
            schema_txt=schema_txt,
            question=question,
            gold_sql=gold_sql,
            llm_name=llm,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            chat_template=chat_template,
            system_instruction=resolved_system_prompt,
        )

        rows.append({
            "id": obj.get("id", f"Q{i:03d}"),
            "text": text
        })

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d training samples to: %s", len(rows), output_path)
    logger.info("Schema source: per-example=%d, global-fallback=%d", len(rows) - fallback_count, fallback_count)


if __name__ == "__main__":
    main()
