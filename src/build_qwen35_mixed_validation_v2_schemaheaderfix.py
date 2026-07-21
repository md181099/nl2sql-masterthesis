#!/usr/bin/env python3
"""Migrate frozen MixedVal2500 v1 to v2 by removing one redundant schema header.

The migration is deliberately narrower than a dataset rebuild: it reads the
frozen v1 JSONL in its existing order and permits exactly one textual change in
each row. It never re-samples or re-renders examples from source datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_DATASET = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42.jsonl"
)
PARENT_MANIFEST = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_manifest.json"
)
OUTPUT_DATASET = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl"
)
OUTPUT_MANIFEST = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_"
    "no_train_no_dev_overlap_seed42_schemaheaderfix_manifest.json"
)
OLD25K = Path(
    "data/sql_create_context/"
    "train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_"
    "mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
OLD25K_RAW = Path(
    "data/sql_create_context/"
    "train_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_"
    "25k_seed42_no_dev_overlap.jsonl"
)
SPIDER_TRAIN = Path("data/spider/spider_data/train_spider.json")
SPIDER_DEV = Path("data/testcases_spider_dev_full.jsonl")
RETRIEVAL_POOL = Path(
    "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
)
STATIC_DEMO = Path(
    "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
)
SQLCC_ONLY_VALIDATION = Path(
    "data/sql_create_context/"
    "val_sft_qwen35_full_chat_v1_clean_anti_overjoin_sqlcc_only_"
    "no_spider_no_train_overlap_2500_seed42.jsonl"
)

SYSTEM_START = "<|im_start|>system\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"
SCHEMA_HEADER = "Database schema:\n"
DOUBLE_SCHEMA_PREFIX = USER_START + SCHEMA_HEADER + SCHEMA_HEADER
SINGLE_SCHEMA_PREFIX = USER_START + SCHEMA_HEADER
RULES_SEPARATOR = "\n\nRules:\n"
QUESTION_SEPARATOR = "\n\nQuestion:\n"
SQL_SEPARATOR = "\n\nSQL:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write new artifacts atomically")
    parser.add_argument("--parent", default=str(PARENT_DATASET))
    parser.add_argument("--parent-manifest", default=str(PARENT_MANIFEST))
    parser.add_argument("--output", default=str(OUTPUT_DATASET))
    parser.add_argument("--manifest", default=str(OUTPUT_MANIFEST))
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_order(values: Iterable[str]) -> str:
    return sha256_text("\n".join(values))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            require(b"\x00" not in raw, f"Null byte at {path}:{line_number}")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"Invalid UTF-8 at {path}:{line_number}") from exc
            require(text.endswith("\n"), f"Missing line ending at {path}:{line_number}")
            value = json.loads(text)
            require(isinstance(value, dict), f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    chars = []
    for char in value:
        category = unicodedata.category(char)
        chars.append(" " if category.startswith(("P", "S", "Z")) else char)
    return " ".join("".join(chars).split())


def strip_sql_comments(value: str) -> str:
    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(value):
        char = value[index]
        if quote:
            output.append(char)
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    output.append(value[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if value.startswith("--", index):
            end = value.find("\n", index + 2)
            index = len(value) if end < 0 else end
            output.append(" ")
            continue
        if value.startswith("/*", index):
            end = value.find("*/", index + 2)
            require(end >= 0, "Unterminated SQL block comment")
            index = end + 2
            output.append(" ")
            continue
        output.append(char)
        index += 1
    require(quote is None, "Unterminated SQL quote")
    return "".join(output)


def sql_tokens(value: str) -> list[str]:
    value = strip_sql_comments(unicodedata.normalize("NFKC", str(value)))
    pattern = re.compile(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
        r"<=|>=|<>|!=|==|\|\||[(),;.*+/%<>=-]|[A-Za-z_][\w$]*|\d+(?:\.\d+)?|\S",
        flags=re.UNICODE,
    )
    return pattern.findall(value)


def normalize_sql(value: str) -> str:
    tokens = sql_tokens(value)
    while tokens and tokens[-1] == ";":
        tokens.pop()
    return " ".join(token.casefold() for token in tokens)


def schema_struct(schema: str) -> tuple[tuple[Any, ...], ...]:
    blocks = re.split(r"(?m)^Table:\s*", str(schema))[1:]
    parsed: list[tuple[Any, ...]] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        name = unicodedata.normalize("NFKC", lines[0]).casefold().strip(' `"[]')
        columns: tuple[str, ...] = ()
        primary: tuple[str, ...] = ()
        foreign: list[str] = []
        in_foreign = False
        for line in lines[1:]:
            stripped = line.strip()
            low = stripped.casefold()
            if low.startswith("columns:"):
                columns = tuple(
                    sorted(item.strip().casefold() for item in stripped.split(":", 1)[1].split(",") if item.strip())
                )
                in_foreign = False
            elif low.startswith("primary key:"):
                primary = tuple(
                    sorted(
                        item.strip().casefold()
                        for item in stripped.split(":", 1)[1].split(",")
                        if item.strip() and item.strip().casefold() != "none"
                    )
                )
                in_foreign = False
            elif low.startswith("foreign keys:"):
                in_foreign = True
            elif in_foreign and stripped:
                foreign.append(" ".join(low.split()))
        if name:
            parsed.append((name, columns, primary, tuple(sorted(foreign))))
    return tuple(sorted(parsed))


def schema_signature(schema: str) -> str:
    payload = json.dumps(schema_struct(schema), ensure_ascii=False, sort_keys=True)
    return sha256_text(payload)


def normalize_schema_semantics(schema: str) -> str:
    lines = schema.strip().splitlines()
    while lines and lines[0].strip().casefold() == "database schema:":
        lines = lines[1:]
    return "\n".join(lines).strip()


def parse_chatml(text: str) -> dict[str, Any]:
    require(text.startswith(SYSTEM_START), "Text does not start with system role")
    require(text.count(SYSTEM_START) == 1, "System role count is not one")
    require(text.count(USER_START) == 1, "User role count is not one")
    require(text.count(ASSISTANT_START) == 1, "Assistant role count is not one")
    require(text.count(END_MARKER) == 3, "ChatML end-marker count is not three")

    system_end = text.index(END_MARKER, len(SYSTEM_START))
    system = text[len(SYSTEM_START) : system_end]
    expected_user_start = system_end + len(END_MARKER) + 1
    require(text.startswith(USER_START, expected_user_start), "Unexpected system/user boundary")
    user_start = expected_user_start + len(USER_START)
    user_end = text.index(END_MARKER, user_start)
    user = text[user_start:user_end]
    expected_assistant_start = user_end + len(END_MARKER) + 1
    require(text.startswith(ASSISTANT_START, expected_assistant_start), "Unexpected user/assistant boundary")
    assistant_start = expected_assistant_start + len(ASSISTANT_START)
    assistant_end = text.index(END_MARKER, assistant_start)
    assistant = text[assistant_start:assistant_end]
    require(text[assistant_end + len(END_MARKER) :] == "\n", "Unexpected trailing text")

    require(user.startswith(SCHEMA_HEADER), "User content lacks schema header")
    schema_and_rest = user[len(SCHEMA_HEADER) :]
    require(schema_and_rest.count(RULES_SEPARATOR) == 1, "Rules separator count is not one")
    schema_raw, after_schema = schema_and_rest.split(RULES_SEPARATOR, 1)
    require(after_schema.count(QUESTION_SEPARATOR) == 1, "Question separator count is not one")
    rules, after_rules = after_schema.split(QUESTION_SEPARATOR, 1)
    require(after_rules.count(SQL_SEPARATOR) == 1, "SQL separator count is not one")
    question, sql_label_tail = after_rules.split(SQL_SEPARATOR, 1)
    require(sql_label_tail == "", "Unexpected text after SQL label")
    schema = normalize_schema_semantics(schema_raw)
    return {
        "system": system,
        "user": user,
        "assistant": assistant,
        "schema_raw": schema_raw,
        "schema": schema,
        "schema_signature": schema_signature(schema),
        "schema_struct": schema_struct(schema),
        "rules": rules,
        "question": question.strip(),
        "sql": assistant,
    }


def migrate_text(text: str) -> tuple[str, int]:
    require(text.count(DOUBLE_SCHEMA_PREFIX) == 1, "Expected exactly one redundant schema-header prefix")
    offset = text.index(DOUBLE_SCHEMA_PREFIX) + len(USER_START)
    migrated = text.replace(DOUBLE_SCHEMA_PREFIX, SINGLE_SCHEMA_PREFIX, 1)
    require(len(text) - len(migrated) == len(SCHEMA_HEADER), "Unexpected character-length delta")
    return migrated, offset


def prompt_audit(parsed: dict[str, Any], text: str) -> list[str]:
    errors: list[str] = []
    checks = {
        "schema_header_count": text.count(SCHEMA_HEADER) == 1,
        "rules_label_count": text.count("\nRules:\n") == 1,
        "question_label_count": text.count("\nQuestion:\n") == 1,
        "sql_label_count": text.count("\nSQL:") == 1,
        "system_role_count": text.count(SYSTEM_START) == 1,
        "user_role_count": text.count(USER_START) == 1,
        "assistant_role_count": text.count(ASSISTANT_START) == 1,
        "end_marker_count": text.count(END_MARKER) == 3,
        "no_think": "<think" not in text.casefold() and "</think" not in text.casefold(),
        "no_markdown_fence": "```" not in text,
        "schema_nonempty": bool(parsed["schema"]),
        "question_nonempty": bool(parsed["question"]),
        "sql_nonempty": bool(parsed["sql"]),
        "sql_starts_correctly": bool(re.match(r"(?is)^\s*(select|with)\b", parsed["sql"])),
        "sql_ends_semicolon": parsed["sql"].rstrip().endswith(";"),
        "no_null_byte": "\x00" not in text,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(name)

    table_names = [item[0] for item in parsed["schema_struct"]]
    if len(table_names) != len(set(table_names)):
        errors.append("duplicate_table_names")
    for table in parsed["schema_struct"]:
        foreign_keys = table[3]
        if len(foreign_keys) != len(set(foreign_keys)):
            errors.append("duplicate_foreign_keys")
            break
    return errors


def identity_from_parsed(row_id: str, parsed: dict[str, Any], source_id: str = "", db_id: str = "") -> dict[str, Any]:
    return {
        "id": row_id,
        "source_id": source_id or row_id,
        "question_norm": normalize_question(parsed["question"]),
        "sql_norm": normalize_sql(parsed["sql"]),
        "pair_norm": (normalize_question(parsed["question"]), normalize_sql(parsed["sql"])),
        "schema_signature": parsed["schema_signature"],
        "db_id": db_id,
    }


def identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[Any]]:
    keys = ("id", "source_id", "question_norm", "sql_norm", "pair_norm", "schema_signature", "db_id")
    output = {key: set() for key in keys}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                output[key].add(tuple(value) if isinstance(value, list) else value)
    return output


def overlap_counts(left: dict[str, set[Any]], right: dict[str, set[Any]]) -> dict[str, int]:
    return {key: len(left[key] & right[key]) for key in left}


def materialized_reference(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in load_jsonl(path):
        parsed = parse_chatml(str(row["text"]))
        output.append(identity_from_parsed(str(row["id"]), parsed))
    return output


def structured_reference(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in load_jsonl(path):
        question = str(row.get("question", ""))
        sql = str(row.get("gold_sql") or row.get("query") or "")
        schema = normalize_schema_semantics(str(row.get("schema_prompt") or ""))
        source_path = str(row.get("source_path") or "")
        source_idx = row.get("source_idx")
        source_id = (
            f"{source_path}#{int(source_idx)}"
            if source_path and source_idx is not None
            else str(row.get("id", ""))
        )
        output.append(
            {
                "id": str(row.get("id", "")),
                "source_id": source_id,
                "question_norm": normalize_question(question),
                "sql_norm": normalize_sql(sql),
                "pair_norm": (normalize_question(question), normalize_sql(sql)),
                "schema_signature": schema_signature(schema) if schema else "",
                "db_id": str(row.get("db_id") or ""),
            }
        )
    return output


def spider_train_reference(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    require(isinstance(rows, list), "Spider Train must be a JSON list")
    output = []
    for index, row in enumerate(rows):
        question = str(row.get("question", ""))
        sql = str(row.get("query") or row.get("gold_sql") or "")
        output.append(
            {
                "id": f"SPIDER_TRAIN_{index:06d}",
                "source_id": f"data/spider/spider_data/train_spider.json#{index}",
                "question_norm": normalize_question(question),
                "sql_norm": normalize_sql(sql),
                "pair_norm": (normalize_question(question), normalize_sql(sql)),
                "schema_signature": "",
                "db_id": str(row.get("db_id") or ""),
            }
        )
    return output


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    require(not path.exists(), f"Refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def main() -> None:
    args = parse_args()
    parent_path = resolve(args.parent)
    parent_manifest_path = resolve(args.parent_manifest)
    output_path = resolve(args.output)
    manifest_path = resolve(args.manifest)
    for path in (
        parent_path,
        parent_manifest_path,
        resolve(OLD25K),
        resolve(OLD25K_RAW),
        resolve(SPIDER_TRAIN),
        resolve(SPIDER_DEV),
        resolve(RETRIEVAL_POOL),
        resolve(STATIC_DEMO),
        resolve(SQLCC_ONLY_VALIDATION),
    ):
        require(path.is_file(), f"Required input missing: {path}")

    parent_sha = sha256_file(parent_path)
    require(parent_sha == "b146f8b0500204719dc17b3025f2bb6e7c8a59200ee29f66a9e3566627ddaf01", "Unexpected parent SHA256")
    parent_rows = load_jsonl(parent_path)
    parent_manifest = load_json(parent_manifest_path)
    require(len(parent_rows) == 2500, "Parent row count is not 2500")
    provenance_rows = parent_manifest.get("selected_provenance", [])
    require(len(provenance_rows) == 2500, "Parent provenance count is not 2500")
    provenance = {str(row["id"]): row for row in provenance_rows}
    require(len(provenance) == 2500, "Parent provenance IDs are not unique")

    output_rows: list[dict[str, Any]] = []
    v2_identities: list[dict[str, Any]] = []
    parent_ids: list[str] = []
    parent_questions: list[str] = []
    parent_sqls: list[str] = []
    parent_schemas: list[str] = []
    output_questions: list[str] = []
    output_sqls: list[str] = []
    output_schemas: list[str] = []
    diff_records: list[dict[str, Any]] = []
    prompt_errors: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for index, parent_row in enumerate(parent_rows):
        row_id = str(parent_row.get("id", ""))
        text = str(parent_row.get("text", ""))
        require(row_id and text, f"Missing id/text at parent row {index}")
        require(row_id in provenance, f"Missing provenance for {row_id}")
        before = parse_chatml(text)
        migrated, offset = migrate_text(text)
        after = parse_chatml(migrated)

        checks = {
            "id": row_id == str(parent_row["id"]),
            "system": before["system"] == after["system"],
            "rules": before["rules"] == after["rules"],
            "question": before["question"] == after["question"],
            "sql": before["sql"] == after["sql"],
            "schema_semantics": before["schema"] == after["schema"],
            "schema_struct": before["schema_struct"] == after["schema_struct"],
        }
        require(all(checks.values()), f"Disallowed semantic change at row {index}: {row_id}")
        expected = text[:offset] + text[offset + len(SCHEMA_HEADER) :]
        require(migrated == expected, f"Migration changed more than one header at row {index}: {row_id}")

        errors = prompt_audit(after, migrated)
        if errors:
            prompt_errors.append({"index": index, "id": row_id, "errors": errors})
        provenance_row = provenance[row_id]
        source = str(provenance_row["source_dataset"])
        source_id = f"{provenance_row['source_path']}#{provenance_row['source_idx']}"
        db_id = str(provenance_row.get("db_id") or "")
        source_counts[source] += 1
        identity = identity_from_parsed(row_id, after, source_id=source_id, db_id=db_id)
        require(identity["schema_signature"] == provenance_row["schema_signature"], f"Schema provenance mismatch for {row_id}")
        v2_identities.append(identity)
        output_rows.append({"id": row_id, "text": migrated})
        parent_ids.append(row_id)
        parent_questions.append(before["question"])
        parent_sqls.append(before["sql"])
        parent_schemas.append(before["schema"])
        output_questions.append(after["question"])
        output_sqls.append(after["sql"])
        output_schemas.append(after["schema"])
        diff_records.append(
            {
                "index": index,
                "id": row_id,
                "source": source,
                "source_id": source_id,
                "removed_offset": offset,
                "removed_text": SCHEMA_HEADER.rstrip("\n"),
                "character_delta": -len(SCHEMA_HEADER),
                "parent_text_sha256": sha256_text(text),
                "v2_text_sha256": sha256_text(migrated),
                "semantic_fields_equal": checks,
            }
        )

    require(not prompt_errors, f"Prompt audit failed for {len(prompt_errors)} rows")
    require(source_counts == Counter({"spider_train_others": 700, "sql_create_context": 1800}), f"Unexpected source counts: {source_counts}")
    require(parent_questions == output_questions, "Question order/content changed")
    require(parent_sqls == output_sqls, "SQL order/content changed")
    require(parent_schemas == output_schemas, "Schema semantics/order changed")

    v2_sets = identity_sets(v2_identities)
    internal_duplicates = {
        key: len(v2_identities) - len(values)
        for key, values in v2_sets.items()
        if key not in {"db_id", "schema_signature"}
    }
    require(internal_duplicates["id"] == 0, "Internal ID duplicates")
    require(internal_duplicates["source_id"] == 0, "Internal source-ID duplicates")
    require(internal_duplicates["question_norm"] == 0, "Internal question duplicates")
    require(internal_duplicates["sql_norm"] == 0, "Internal SQL duplicates")
    require(internal_duplicates["pair_norm"] == 0, "Internal pair duplicates")

    references = {
        "old25k": structured_reference(resolve(OLD25K_RAW)),
        "spider_train": spider_train_reference(resolve(SPIDER_TRAIN)),
        "spider_dev_1032": structured_reference(resolve(SPIDER_DEV)),
        "retrieval_pool_6960": structured_reference(resolve(RETRIEVAL_POOL)),
        "static_demo": structured_reference(resolve(STATIC_DEMO)),
        "sqlcc_only_validation": materialized_reference(resolve(SQLCC_ONLY_VALIDATION)),
    }
    leakage = {name: overlap_counts(v2_sets, identity_sets(rows)) for name, rows in references.items()}
    forbidden_keys = ("id", "source_id", "question_norm", "sql_norm", "pair_norm")
    for name in ("old25k", "spider_dev_1032", "retrieval_pool_6960", "static_demo", "sqlcc_only_validation"):
        require(all(leakage[name][key] == 0 for key in forbidden_keys), f"Forbidden leakage against {name}: {leakage[name]}")
    require(leakage["old25k"]["schema_signature"] == 0, "Exact schema overlap with old25k")

    diff_digest_payload = "\n".join(
        f"{item['index']}|{item['id']}|{item['parent_text_sha256']}|{item['v2_text_sha256']}|{item['removed_offset']}"
        for item in diff_records
    )
    script_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "dataset_version": "mixedval2500_v2_schemaheaderfix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_dataset_path": relative(parent_path),
        "parent_dataset_sha256": parent_sha,
        "parent_manifest_path": relative(parent_manifest_path),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "transformation_script": relative(script_path),
        "transformation_script_sha256": sha256_file(script_path),
        "transformation_description": "Remove exactly one redundant Database schema: line immediately after the user-role marker in every frozen v1 row; no reselection or rerendering.",
        "row_count": len(output_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "source_id_order_hash": hash_order(item["source_id"] for item in diff_records),
        "row_id_order_hash": hash_order(parent_ids),
        "question_order_hash": hash_order(output_questions),
        "sql_order_hash": hash_order(output_sqls),
        "schema_order_hash": hash_order(output_schemas),
        "output_path": relative(output_path),
        "output_sha256": None,
        "double_schema_header_before": sum(str(row["text"]).count(DOUBLE_SCHEMA_PREFIX) for row in parent_rows),
        "double_schema_header_after": sum(row["text"].count(DOUBLE_SCHEMA_PREFIX) for row in output_rows),
        "selection_changed": False,
        "source_ids_changed": False,
        "row_order_changed": False,
        "questions_changed": False,
        "sql_changed": False,
        "schema_semantics_changed": False,
        "prompt_format_changed": True,
        "allowed_change": {
            "removed_text": SCHEMA_HEADER.rstrip("\n"),
            "occurrences_per_row": 1,
            "character_delta_per_row": -len(SCHEMA_HEADER),
        },
        "structured_diff": {
            "rows_checked": len(diff_records),
            "rows_with_exactly_allowed_change": len(diff_records),
            "rows_with_disallowed_change": 0,
            "diff_records_sha256": sha256_text(diff_digest_payload),
            "first_10_records": diff_records[:10],
            "last_10_records": diff_records[-10:],
        },
        "prompt_audit": {
            "rows_checked": len(output_rows),
            "rows_failed": 0,
            "schema_header_exactly_once": len(output_rows),
            "roles_exactly_once": len(output_rows),
            "three_end_markers": len(output_rows),
            "empty_schema": 0,
            "empty_question": 0,
            "empty_sql": 0,
            "think_tags": 0,
            "markdown_fences": 0,
            "duplicate_table_names": 0,
            "duplicate_foreign_keys": 0,
            "invalid_utf8": 0,
            "null_bytes": 0,
        },
        "internal_duplicate_counts": internal_duplicates,
        "leakage_matrix": leakage,
        "reference_paths": {name: relative(path) for name, path in {
            "old25k": resolve(OLD25K),
            "old25k_raw": resolve(OLD25K_RAW),
            "spider_train": resolve(SPIDER_TRAIN),
            "spider_dev_1032": resolve(SPIDER_DEV),
            "retrieval_pool_6960": resolve(RETRIEVAL_POOL),
            "static_demo": resolve(STATIC_DEMO),
            "sqlcc_only_validation": resolve(SQLCC_ONLY_VALIDATION),
        }.items()},
        "reference_sha256": {name: sha256_file(path) for name, path in {
            "old25k": resolve(OLD25K),
            "old25k_raw": resolve(OLD25K_RAW),
            "spider_train": resolve(SPIDER_TRAIN),
            "spider_dev_1032": resolve(SPIDER_DEV),
            "retrieval_pool_6960": resolve(RETRIEVAL_POOL),
            "static_demo": resolve(STATIC_DEMO),
            "sqlcc_only_validation": resolve(SQLCC_ONLY_VALIDATION),
        }.items()},
        "acceptance": {
            "parent_sha256_verified": True,
            "same_2500_rows": True,
            "same_source_ids": True,
            "same_order": True,
            "same_questions": True,
            "same_sql": True,
            "same_schema_semantics": True,
            "only_redundant_header_removed": True,
            "prompt_audit_passed": True,
            "leakage_audit_passed": True,
            "all_passed": True,
        },
    }

    if args.write:
        write_jsonl_atomic(output_path, output_rows)
        manifest["output_sha256"] = sha256_file(output_path)
        write_json_atomic(manifest_path, manifest)
    else:
        serialized = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in output_rows)
        manifest["output_sha256"] = sha256_bytes(serialized)

    summary = {
        "status": "PASS",
        "write": args.write,
        "output": relative(output_path),
        "output_sha256": manifest["output_sha256"],
        "manifest": relative(manifest_path),
        "rows": len(output_rows),
        "source_counts": dict(source_counts),
        "double_schema_header_before": manifest["double_schema_header_before"],
        "double_schema_header_after": manifest["double_schema_header_after"],
        "disallowed_diffs": 0,
        "prompt_errors": 0,
        "leakage": leakage,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
