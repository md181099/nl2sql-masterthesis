#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


# IMPORTANT:
# Keep this string exactly in sync with the previous default system prompt in
# src/06_batch_run.py to preserve backward-compatible behavior for "current".
CURRENT_SYSTEM_PROMPT = (
    "Return the SQL query only. Start directly with SELECT or WITH. End with a semicolon. "
    "Do not use markdown. Do not explain. "
    "Do not emit additional examples, prompt labels, or chat role tokens."
)

V2_SQLCTX_SYSTEM_PROMPT = (
    "You are an SQLite SQL generator. "
    "Return exactly one valid SQLite query and nothing else. "
    "Output SQL only. "
    "Do not explain. "
    "Do not reason. "
    "Do not use markdown. "
    "Do not use comments. "
    "Use only tables and columns from the provided schema. "
    "The query must start with SELECT or WITH and end with a semicolon."
)


SYSTEM_PROMPT_PRESETS: dict[str, str] = {
    "current": CURRENT_SYSTEM_PROMPT,
    "v2_sqlctx_default": V2_SQLCTX_SYSTEM_PROMPT,
    "v2_sqlctx_anti_overjoin": (
        "You are an SQLite SQL generator. "
        "Return exactly one valid SQLite query and nothing else. "
        "Output SQL only. "
        "Do not explain. "
        "Do not reason. "
        "Do not use markdown. "
        "Do not use comments. "
        "Use only tables and columns from the provided schema. "
        "Use only tables required by the question. "
        "Do not join tables unless their columns are required. "
        "If one table contains all required columns and filters, use only that table. "
        "The query must start with SELECT or WITH and end with a semicolon."
    ),
    "v2_sqlctx_minimal_sufficient": (
        "You are an SQLite SQL generator. "
        "Return exactly one valid SQLite query and nothing else. "
        "Output SQL only. "
        "Do not explain. "
        "Do not reason. "
        "Do not use markdown. "
        "Do not use comments. "
        "Use only tables and columns from the provided schema. "
        "Use all and only the tables required by the question. "
        "Do not join tables unless their columns or relationships are necessary. "
        "If one table contains all required columns and filters, use only that table. "
        "Use bridge tables only when they are necessary to connect required tables. "
        "Do not omit tables required for selected columns, filters, grouping, ordering, aggregation, or relationships. "
        "Prefer the simplest valid SQL query that preserves the exact question semantics. "
        "The query must start with SELECT or WITH and end with a semicolon."
    ),
    "concise_sql": (
        "You are a SQLite SQL generator. "
        "Return exactly one valid SQLite SQL query. "
        "Rules: "
        "Output SQL only. "
        "No explanations. "
        "No markdown. "
        "No reasoning. "
        "No comments. "
        "No <think> tags. "
        "Start directly with SELECT or WITH. "
        "End with semicolon."
    ),
    "strict_sql_only": (
        "You are an SQLite SQL generator. "
        "Return exactly one valid SQLite query and nothing else. "
        "Rules: "
        "Output SQL only. "
        "Do not explain. "
        "Do not reason. "
        "Do not describe the schema. "
        "Do not write natural language. "
        "Do not use markdown. "
        "Do not use code fences. "
        "Do not use comments. "
        "Do not output <think> or </think>. "
        "The first token must be SELECT or WITH. "
        "The final character must be semicolon. "
        "If unsure, still return the most likely valid SQLite query."
    ),
    "anti_hallucination": (
        "Use only tables and columns from the provided schema. "
        "Do not invent tables or columns. "
        "If unsure, prefer a simpler query using only clearly relevant schema elements. "
        "Return only executable SQLite SQL. "
        "Start with SELECT or WITH and end with a semicolon. "
        "No explanation. No markdown."
    ),
    "minimal_joins": (
        "Prefer the simplest correct SQL query. "
        "Do not add joins unless required by the question. "
        "Use joins only when selected columns or filters require multiple tables. "
        "Avoid unnecessary intermediate tables. "
        "Return exactly one executable SQLite query. "
        "Start with SELECT or WITH and end with a semicolon. "
        "No explanation. No markdown."
    ),
    "schema_strict": (
        "Strictly follow the provided schema. "
        "Use only listed tables and columns. "
        "Respect foreign keys when joining. "
        "Return exactly one valid SQLite SELECT or WITH query. "
        "Start with SELECT or WITH and end with a semicolon. "
        "No explanation. No markdown."
    ),
    "anti_hallucination_minimal_joins": (
        "You are an expert SQLite query generator. "
        "Use only tables and columns from the provided schema. "
        "Do not invent tables, columns, functions, or values not present in the schema or question. "
        "Prefer the simplest correct SQL query. "
        "Do not add joins unless the question requires columns or filters from multiple tables. "
        "When joins are necessary, use only relationships supported by the schema. "
        "Return exactly one executable SQLite query. "
        "Start with SELECT or WITH and end with a semicolon. "
        "No explanation, markdown, comments, or extra text."
    ),
    "single_table_first": (
        "You are an expert SQLite query generator. "
        "Use only tables and columns from the provided schema. "
        "Prefer a single-table query whenever the question can be answered from one table. "
        "Add joins only if the requested columns or filters require multiple tables. "
        "Do not invent tables or columns. "
        "Return exactly one executable SQLite query. "
        "Start with SELECT or WITH. "
        "No explanation. No markdown."
    ),
    "schema_grounded": (
        "You are an expert SQLite query generator. "
        "Ground every table and column in the provided schema. "
        "Before using a column, ensure it appears under the selected table in the schema. "
        "If a question can be answered with a direct column from one table, use that table directly. "
        "Use joins only for relationships needed to connect required columns. "
        "Return exactly one executable SQLite query. "
        "Start with SELECT or WITH. "
        "No explanation. No markdown."
    ),
    "simple_sql_bias": (
        "You are an expert SQLite query generator. "
        "Generate the shortest correct SQLite query for the question. "
        "Use only schema tables and columns. "
        "Avoid unnecessary joins, subqueries, aliases, and selected columns. "
        "Select only the columns asked for in the question. "
        "Use aggregation only when the question asks for counts, averages, sums, minimums, maximums, or rankings. "
        "Return exactly one executable SQL query. "
        "Start with SELECT or WITH. "
        "No explanation. No markdown."
    ),
    "balanced_sql_guard": (
        "You are an expert SQLite query generator. "
        "Use only the provided schema. Do not invent tables or columns. "
        "Select only the columns needed to answer the question. "
        "Prefer the simplest correct query. "
        "Use a single table when possible; add joins only when required by the question. "
        "Use aggregations, GROUP BY, ORDER BY, LIMIT, or subqueries only when the question requires them. "
        "Return exactly one executable SQLite query. "
        "Start with SELECT or WITH. "
        "No explanation. No markdown."
    ),
}

SYSTEM_PROMPT_PRESETS["sqlctx_default"] = SYSTEM_PROMPT_PRESETS["v2_sqlctx_default"]
SYSTEM_PROMPT_PRESETS["sqlctx_anti_overjoin"] = SYSTEM_PROMPT_PRESETS["v2_sqlctx_anti_overjoin"]
SYSTEM_PROMPT_PRESETS["sqlctx_minimal_sufficient"] = SYSTEM_PROMPT_PRESETS["v2_sqlctx_minimal_sufficient"]


def list_system_prompt_variants() -> list[str]:
    return sorted(SYSTEM_PROMPT_PRESETS.keys())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_system_prompt(
    *,
    project_root: Path,
    system_prompt_variant: str = "current",
    system_prompt_path: str | None = None,
) -> tuple[str, str, str | None, str]:
    """
    Resolve system prompt source for evaluation.

    Returns:
      (prompt_text, source, resolved_path, prompt_hash)
      source in {"variant", "path"}
    """
    path_raw = (system_prompt_path or "").strip()
    if path_raw:
        path = Path(path_raw)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"system_prompt_path not found: {path}")
        if not path.is_file():
            raise ValueError(f"system_prompt_path is not a file: {path}")
        prompt_text = path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"system_prompt_path is empty: {path}")
        return prompt_text, "path", str(path), _sha256_text(prompt_text)

    variant = system_prompt_variant.strip().lower()
    if not variant:
        raise ValueError("system_prompt_variant must be non-empty")
    if variant not in SYSTEM_PROMPT_PRESETS:
        allowed = ", ".join(list_system_prompt_variants())
        raise ValueError(
            f"Unknown system_prompt_variant '{system_prompt_variant}'. Allowed: {allowed}"
        )
    prompt_text = SYSTEM_PROMPT_PRESETS[variant]
    return prompt_text, "variant", None, _sha256_text(prompt_text)
