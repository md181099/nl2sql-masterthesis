#!/usr/bin/env python3
"""General post-hoc teacher-forced loss diagnostics for Qwen 3.5 LoRA runs.

`--validate-only` loads only the tokenizer and dataset. It validates masks,
packing, config provenance, output safety, and checkpoint discovery without
loading a language model or adapter. Actual forward passes require an explicit
`--checkpoint` or `--run-all` invocation after training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_MARKER = "<|im_start|>assistant\n"
ANSWER_END_MARKER = "<|im_end|>"
LLAMA32_NATIVE_CHAT_FORMAT = "llama32_instruct_native_chat"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def functional_tokenizer_hash(tokenizer: Any) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocab": sorted(tokenizer.get_vocab().items()),
        "added_vocab": sorted(tokenizer.get_added_vocab().items()),
        "special_tokens_map": {key: str(value) for key, value in tokenizer.special_tokens_map.items()},
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256_bytes(encoded)


def tokenizer_provenance_hash(tokenizer: Any, model_cfg: dict[str, Any]) -> str:
    payload = {
        "tokenizer_id": model_cfg["tokenizer_id"],
        "revision": model_cfg.get("revision"),
        "name_or_path": tokenizer.name_or_path,
        "functional_sha256": functional_tokenizer_hash(tokenizer),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def source_name(example_id: str, source_prefixes: dict[str, list[str]]) -> str:
    matches = [
        source
        for source, prefixes in source_prefixes.items()
        if any(example_id.startswith(prefix) for prefix in prefixes)
    ]
    require(len(matches) == 1, f"Could not uniquely map source for id={example_id!r}: {matches}")
    return matches[0]


def token_mask(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [int(offset_end > start and offset_start < end) for offset_start, offset_end in offsets]


def position_ids_from_seq_lengths(seq_lengths: list[int]) -> list[int]:
    result: list[int] = []
    for length in seq_lengths:
        result.extend(range(int(length)))
    return result


def quantile(values: list[int], fraction: float) -> float:
    require(bool(values), "Cannot calculate a quantile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def metric_result(total_loss: float, total_correct: int, total_tokens: int) -> dict[str, Any]:
    require(total_tokens > 0, "Cannot finalize an empty metric")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else float("inf"),
        "token_accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == 2, "Unsupported config schema_version")
    require(config.get("purpose") == "posthoc_loss_diagnostic_checkpoint_discovery", "Unexpected config purpose")
    for section in ("model", "validation", "adapter_discovery", "evaluation", "expected_lora"):
        require(isinstance(config.get(section), dict), f"Missing config section: {section}")
    validation_path = resolve_path(config["validation"]["path"])
    require(validation_path.is_file(), f"Validation dataset missing: {validation_path}")
    actual_validation_sha = sha256_file(validation_path)
    require(actual_validation_sha == config["validation"]["sha256"], "Validation SHA256 mismatch")
    result_dir = resolve_path(config["evaluation"]["result_dir"])
    if result_dir.exists():
        require(result_dir.is_dir(), f"Result path exists as a file: {result_dir}")
        existing = list(result_dir.glob("*.json"))
    else:
        existing = []
    return {
        "config_path": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "validation_path": relative(validation_path),
        "validation_sha256": actual_validation_sha,
        "adapter_root": config["adapter_discovery"]["adapter_root"],
        "result_dir": config["evaluation"]["result_dir"],
        "existing_result_jsons": [relative(path) for path in existing],
    }


def prepare_packed_validation(config: dict[str, Any]) -> tuple[Any, dict[str, Any], list[str], list[str], Any]:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import pack_dataset

    model_cfg = config["model"]
    validation_cfg = config["validation"]
    evaluation_cfg = config["evaluation"]
    validation_path = resolve_path(validation_cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["tokenizer_id"],
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    require(tokenizer.is_fast, "Fast tokenizer with offset mappings is required")
    require(tokenizer.eos_token is not None, "Tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_format = str(validation_cfg.get("chat_format", "qwen_sqlctx_chatml"))
    if chat_format == LLAMA32_NATIVE_CHAT_FORMAT:
        from llama32_native_chat import llama32_assistant_generation_prefix

        assistant_marker = llama32_assistant_generation_prefix(tokenizer)
        answer_end_marker = str(tokenizer.eos_token)
        add_special_tokens = False
    elif chat_format == "qwen_sqlctx_chatml":
        assistant_marker = ASSISTANT_MARKER
        answer_end_marker = ANSWER_END_MARKER
        add_special_tokens = True
    else:
        raise RuntimeError(f"Unsupported validation chat_format: {chat_format!r}")

    source_prefixes = validation_cfg["source_prefixes"]
    source_names = list(source_prefixes)
    source_codes = {source: index for index, source in enumerate(source_names)}
    columns: dict[str, list[list[int]]] = {
        "input_ids": [],
        "sql_mask": [],
        "assistant_completion_mask": [],
        "source_code": [],
        "example_index": [],
    }
    ids: list[str] = []
    sources: list[str] = []
    texts: list[str] = []
    raw_lengths: list[int] = []
    sql_lengths: list[int] = []
    completion_lengths: list[int] = []
    source_rows: Counter[str] = Counter()
    source_sql_tokens: Counter[str] = Counter()
    source_completion_tokens: Counter[str] = Counter()
    mask_rows: list[dict[str, Any]] = []

    with validation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example_id = row.get("id")
            text = row.get("text")
            require(isinstance(example_id, str) and example_id, f"Invalid id at line {line_number}")
            require(isinstance(text, str) and text, f"Invalid text at line {line_number}")
            require(text.count(assistant_marker) == 1, f"Assistant marker count != 1 for {example_id}")
            sql_start = text.index(assistant_marker) + len(assistant_marker)
            sql_end = text.find(answer_end_marker, sql_start)
            require(sql_end > sql_start, f"Missing or empty assistant SQL for {example_id}")
            completion_end = sql_end + len(answer_end_marker)
            tokenized_text = text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token
            encoded = tokenizer(
                tokenized_text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            sql_mask = token_mask(offsets, sql_start, sql_end)
            completion_mask = token_mask(offsets, sql_start, completion_end)
            require(sum(sql_mask) > 0, f"Empty SQL token mask for {example_id}")
            require(sum(completion_mask) > sum(sql_mask), f"End marker absent from completion mask: {example_id}")
            require(len(input_ids) <= int(evaluation_cfg["max_length"]), f"Over-length row: {example_id}")
            require(not any(mask and start < sql_start for mask, (start, _end) in zip(sql_mask, offsets)), f"Prompt token in SQL mask: {example_id}")
            require(not any(mask and end > sql_end for mask, (_start, end) in zip(sql_mask, offsets)), f"End marker/EOS in SQL mask: {example_id}")
            source = source_name(example_id, source_prefixes)
            example_index = len(ids)
            code = source_codes[source]
            columns["input_ids"].append(input_ids)
            columns["sql_mask"].append(sql_mask)
            columns["assistant_completion_mask"].append(completion_mask)
            columns["source_code"].append([code] * len(input_ids))
            columns["example_index"].append([example_index] * len(input_ids))
            ids.append(example_id)
            sources.append(source)
            texts.append(text)
            raw_lengths.append(len(input_ids))
            sql_lengths.append(sum(sql_mask))
            completion_lengths.append(sum(completion_mask))
            source_rows[source] += 1
            source_sql_tokens[source] += sum(sql_mask)
            source_completion_tokens[source] += sum(completion_mask)
            sql = text[sql_start:sql_end]
            mask_rows.append(
                {
                    "index": example_index,
                    "id": example_id,
                    "source": source,
                    "sql": sql,
                    "sql_start": sql_start,
                    "sql_end": sql_end,
                    "input_ids": input_ids,
                    "offsets": offsets,
                    "sql_mask": sql_mask,
                    "completion_mask": completion_mask,
                }
            )

    expected = validation_cfg.get("expected", {})
    if "rows" in expected:
        require(len(ids) == int(expected["rows"]), "Unexpected validation row count")
    if "source_rows" in expected:
        require(dict(source_rows) == expected["source_rows"], f"Unexpected source counts: {source_rows}")
    if "sql_tokens" in expected:
        require(sum(sql_lengths) == int(expected["sql_tokens"]), "Unexpected SQL token count")
    if "assistant_completion_tokens" in expected:
        require(sum(completion_lengths) == int(expected["assistant_completion_tokens"]), "Unexpected completion token count")
    if "max_tokens_with_eos" in expected:
        require(max(raw_lengths) == int(expected["max_tokens_with_eos"]), "Unexpected maximum token length")

    dataset = Dataset.from_dict(columns)
    if bool(evaluation_cfg.get("shuffle_before_packing", True)):
        dataset = dataset.shuffle(seed=int(evaluation_cfg["seed"]))
    packed = pack_dataset(
        dataset,
        seq_length=int(evaluation_cfg["max_length"]),
        strategy=evaluation_cfg["packing_strategy"],
    )
    if bool(evaluation_cfg.get("shuffle_after_packing", True)):
        packed = packed.shuffle(seed=int(evaluation_cfg["seed"]))
    packed_examples = sum(len(row["seq_lengths"]) for row in packed)
    packed_tokens = sum(len(row["input_ids"]) for row in packed)
    require(packed_examples == len(ids), "Packing lost or duplicated examples")
    require(packed_tokens == sum(raw_lengths), "Packing lost or duplicated tokens")
    require(sum(sum(row["sql_mask"]) for row in packed) == sum(sql_lengths), "Packing changed SQL masks")
    require(sum(sum(row["assistant_completion_mask"]) for row in packed) == sum(completion_lengths), "Packing changed completion masks")
    if "packed_sequences" in expected:
        require(len(packed) == int(expected["packed_sequences"]), "Unexpected packed sequence count")

    mask_audit = build_mask_audit(
        mask_rows,
        tokenizer,
        source_names,
        assistant_marker=assistant_marker,
        answer_end_marker=answer_end_marker,
        add_special_tokens=add_special_tokens,
    )
    stats = {
        "validation_path": relative(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "rows": len(ids),
        "source_rows": dict(source_rows),
        "raw_tokens_with_eos": sum(raw_lengths),
        "raw_length": {
            "min": min(raw_lengths),
            "mean": statistics.fmean(raw_lengths),
            "median": statistics.median(raw_lengths),
            "p95": quantile(raw_lengths, 0.95),
            "p99": quantile(raw_lengths, 0.99),
            "max": max(raw_lengths),
            "exactly_max_length": sum(length == int(evaluation_cfg["max_length"]) for length in raw_lengths),
            "over_max_length": 0,
        },
        "sql_tokens": sum(sql_lengths),
        "assistant_completion_tokens": sum(completion_lengths),
        "source_sql_tokens": dict(source_sql_tokens),
        "source_assistant_completion_tokens": dict(source_completion_tokens),
        "packed_sequences": len(packed),
        "packed_examples": packed_examples,
        "packed_tokens": packed_tokens,
        "max_packed_length": max(len(row["input_ids"]) for row in packed),
        "truncated_examples": 0,
        "tokenizer": {
            "id": model_cfg["tokenizer_id"],
            "revision": model_cfg.get("revision"),
            "class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "functional_sha256": functional_tokenizer_hash(tokenizer),
            "provenance_sha256": tokenizer_provenance_hash(tokenizer, model_cfg),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "mask_definition": {
            "sql": (
                "Assistant SQL span only; excludes assistant marker, the native answer-end marker, "
                "and any separately appended technical EOS."
            ),
            "assistant_completion": (
                "Assistant SQL plus the native answer-end marker; excludes the prompt and any "
                "separately appended technical EOS."
            ),
            "full_chat": "All causal targets except the first token of every packed document, matching full-chat labels.",
        },
        "chat_format": chat_format,
        "assistant_marker": assistant_marker,
        "answer_end_marker": answer_end_marker,
        "add_special_tokens": add_special_tokens,
        "mask_audit": mask_audit,
    }
    return packed, stats, ids, sources, tokenizer


def build_mask_audit(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    sources: list[str],
    *,
    assistant_marker: str,
    answer_end_marker: str,
    add_special_tokens: bool,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for source in sources:
        candidates = [row for row in rows if row["source"] == source]
        if not candidates:
            continue
        ranked: list[dict[str, Any]] = []
        ranked.extend(sorted(candidates, key=lambda row: (len(row["sql"]), row["id"]))[:4])
        ranked.extend(sorted(candidates, key=lambda row: (-len(row["sql"]), row["id"]))[:4])
        ranked.extend([row for row in candidates if re.search(r"(?i)\bjoin\b", row["sql"])][:4])
        ranked.extend([row for row in candidates if len(re.findall(r"(?i)\bselect\b", row["sql"])) > 1][:4])
        ranked.extend([row for row in candidates if row["sql"].rstrip().endswith(";")][:4])
        seen: set[int] = set()
        chosen: list[dict[str, Any]] = []
        for row in ranked + candidates:
            if row["index"] not in seen:
                seen.add(row["index"])
                chosen.append(row)
            if len(chosen) == min(20, len(candidates)):
                break
        selected.extend(chosen)

    summaries = []
    readable = []
    for row in selected:
        sql_offsets = [offset for offset, active in zip(row["offsets"], row["sql_mask"]) if active]
        require(sql_offsets and min(start for start, _end in sql_offsets) <= row["sql_start"], f"SQL mask start gap: {row['id']}")
        require(max(end for _start, end in sql_offsets) >= row["sql_end"], f"SQL mask end gap: {row['id']}")
        semicolon_present = ";" in row["sql"]
        semicolon_masked = any(
            active and row["sql_start"] <= start < row["sql_end"] and ";" in row["sql"][max(0, start-row["sql_start"]):max(0, end-row["sql_start"])]
            for active, (start, end) in zip(row["sql_mask"], row["offsets"])
        )
        if semicolon_present:
            require(semicolon_masked, f"Semicolon missing from SQL mask: {row['id']}")
        summaries.append(
            {
                "id": row["id"],
                "source": row["source"],
                "sql_tokens": sum(row["sql_mask"]),
                "completion_tokens": sum(row["completion_mask"]),
                "has_join": bool(re.search(r"(?i)\bjoin\b", row["sql"])),
                "has_subquery": len(re.findall(r"(?i)\bselect\b", row["sql"])) > 1,
                "semicolon_present_and_masked": semicolon_present and semicolon_masked,
                "assistant_marker_excluded": True,
                "answer_end_excluded_from_sql": True,
                "technical_eos_excluded_from_sql": True,
            }
        )
        if len(readable) < 4:
            token_rows = []
            for token_id, offset, sql_active, completion_active in zip(
                row["input_ids"], row["offsets"], row["sql_mask"], row["completion_mask"]
            ):
                if sql_active or completion_active or (offset[1] > row["sql_start"] - 20 and offset[0] < row["sql_end"] + len(answer_end_marker) + 5):
                    token_rows.append(
                        {
                            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                            "offset": list(offset),
                            "sql_mask": bool(sql_active),
                            "completion_mask": bool(completion_active),
                        }
                    )
            readable.append({"id": row["id"], "source": row["source"], "sql": row["sql"], "tokens": token_rows})

    synthetic_results = []
    for sql in ("SELECT 1;", "SELECT 1"):
        text = assistant_marker + sql + answer_end_marker
        tokenized_text = text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token
        encoded = tokenizer(
            tokenized_text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        start = len(assistant_marker)
        end = start + len(sql)
        mask = token_mask([tuple(pair) for pair in encoded["offset_mapping"]], start, end)
        require(sum(mask) > 0, f"Synthetic SQL mask empty for {sql!r}")
        synthetic_results.append({"sql": sql, "sql_tokens": sum(mask), "semicolon": sql.endswith(";")})

    return {
        "rows_checked": len(selected),
        "rows_by_source": dict(Counter(row["source"] for row in selected)),
        "empty_sql_masks": 0,
        "off_by_one_failures": 0,
        "actual_rows_without_semicolon": sum(not row["sql"].rstrip().endswith(";") for row in rows),
        "synthetic_with_and_without_semicolon": synthetic_results,
        "summaries": summaries,
        "readable_token_samples": readable,
    }


def validate_adapter_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    adapter = load_json(path)
    expected = config["expected_lora"]
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "task_type", "use_dora"):
        require(adapter.get(key) == expected[key], f"Unexpected {key} in {path}")
    require(sorted(adapter.get("target_modules", [])) == sorted(expected["target_modules"]), f"Unexpected target_modules in {path}")
    require(adapter.get("base_model_name_or_path") == config["model"]["model_id"], f"Wrong base model in {path}")
    return adapter


def discover_checkpoints(config: dict[str, Any]) -> dict[str, Any]:
    discovery = config["adapter_discovery"]
    root = resolve_path(discovery["adapter_root"])
    allow_missing = bool(discovery.get("allow_missing_adapter_root_before_training", False))
    if not root.exists():
        require(allow_missing, f"Adapter root missing: {root}")
        return {"adapter_root_exists": False, "unique_weights": [], "root_matches_checkpoint": None}
    require(root.is_dir(), f"Adapter root is not a directory: {root}")
    candidates: list[dict[str, Any]] = []
    checkpoint_root = root / discovery.get("checkpoint_subdir", "checkpoints")
    if checkpoint_root.exists():
        for path in sorted(checkpoint_root.glob("checkpoint-*"), key=lambda value: int(value.name.rsplit("-", 1)[1])):
            candidates.append({"path": path, "kind": "checkpoint"})
    if (root / "adapter_model.safetensors").is_file():
        candidates.append({"path": root, "kind": "root"})

    validated = []
    for candidate in candidates:
        path = candidate["path"]
        model_file = path / "adapter_model.safetensors"
        config_file = path / "adapter_config.json"
        require(model_file.is_file() and model_file.stat().st_size > 0, f"Missing adapter weights: {path}")
        require(config_file.is_file() and config_file.stat().st_size > 0, f"Missing adapter config: {path}")
        validate_adapter_config(config_file, config)
        state_file = path / "trainer_state.json"
        state = load_json(state_file) if state_file.is_file() else None
        step = state.get("global_step") if state else None
        if candidate["kind"] == "checkpoint":
            name_step = int(path.name.rsplit("-", 1)[1])
            require(step == name_step, f"Checkpoint name/state mismatch: {path}")
        validated.append(
            {
                "path": relative(path),
                "kind": candidate["kind"],
                "adapter_model_sha256": sha256_file(model_file),
                "adapter_config_sha256": sha256_file(config_file),
                "global_step": step,
                "trainer_state": state,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in validated:
        grouped.setdefault(item["adapter_model_sha256"], []).append(item)
    unique = []
    for model_hash, aliases in grouped.items():
        canonical = next((item for item in aliases if item["kind"] == "checkpoint"), aliases[0])
        state = canonical.pop("trainer_state")
        for alias in aliases:
            alias.pop("trainer_state", None)
        stored_eval_loss = None
        epoch = None
        if state:
            step = canonical["global_step"]
            matches = [entry for entry in state.get("log_history", []) if entry.get("step") == step and "eval_loss" in entry]
            require(len(matches) == 1, f"Missing unique eval loss for {canonical['path']}")
            stored_eval_loss = float(matches[0]["eval_loss"])
            epoch = float(matches[0].get("epoch"))
        unique.append(
            {
                **canonical,
                "epoch": epoch,
                "stored_trainer_eval_loss": stored_eval_loss,
                "aliases": sorted(item["path"] for item in aliases),
            }
        )
    unique.sort(key=lambda item: (item["global_step"] is None, item["global_step"] or 10**18))
    root_hashes = {item["adapter_model_sha256"] for item in validated if item["kind"] == "root"}
    checkpoint_hashes = {item["adapter_model_sha256"] for item in validated if item["kind"] == "checkpoint"}
    root_match = bool(root_hashes) and root_hashes <= checkpoint_hashes
    if root_hashes and bool(discovery.get("require_root_matches_checkpoint", True)):
        require(root_match, "Root adapter does not match any retained checkpoint")
    return {"adapter_root_exists": True, "unique_weights": unique, "root_matches_checkpoint": root_match}


def result_path(config: dict[str, Any], checkpoint: dict[str, Any]) -> Path:
    model_label = config["model"]["label"]
    dataset_label = config["validation"]["label"]
    step_label = f"step{int(checkpoint['global_step']):06d}" if checkpoint["global_step"] is not None else "root"
    name = (
        f"{model_label}_{dataset_label}_{step_label}_"
        f"adapter-{checkpoint['adapter_model_sha256'][:12]}_"
        f"data-{config['validation']['sha256'][:12]}.json"
    )
    return resolve_path(config["evaluation"]["result_dir"]) / name


def evaluate_checkpoint(
    config_path: Path,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    packed: Any,
    validation_stats: dict[str, Any],
    example_ids: list[str],
    example_sources: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import peft
    import transformers
    import trl
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    require(torch.cuda.is_available(), "CUDA is required for an actual loss run; no model was loaded")
    checkpoint_path = resolve_path(checkpoint["path"])
    output_path = result_path(config, checkpoint)
    require(not output_path.exists(), f"Refusing to overwrite existing result: {output_path}")
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    dtype_name = eval_cfg["torch_dtype"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    torch.manual_seed(int(eval_cfg["seed"]))
    torch.cuda.manual_seed_all(int(eval_cfg["seed"]))

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=eval_cfg["attn_implementation"],
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    metric_names = ("sql", "assistant_completion", "full_chat")
    totals = {name: {"loss": 0.0, "correct": 0, "tokens": 0, "pack_means": []} for name in metric_names}
    source_names = list(config["validation"]["source_prefixes"])
    source_codes = {index: source for index, source in enumerate(source_names)}
    source_totals = {
        source: {name: {"loss": 0.0, "correct": 0, "tokens": 0} for name in metric_names}
        for source in source_names
    }
    per_example_loss = torch.zeros(len(example_ids), dtype=torch.float64)
    per_example_correct = torch.zeros(len(example_ids), dtype=torch.int64)
    per_example_tokens = torch.zeros(len(example_ids), dtype=torch.int64)
    chunk_size = int(eval_cfg.get("logit_chunk_size", 128))

    with torch.inference_mode():
        for packed_index, row in enumerate(packed):
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=input_device)
            position_ids = torch.tensor([position_ids_from_seq_lengths(row["seq_lengths"])], dtype=torch.long, device=input_device)
            masks = {
                "sql": torch.tensor(row["sql_mask"][1:], dtype=torch.bool, device=input_device),
                "assistant_completion": torch.tensor(row["assistant_completion_mask"][1:], dtype=torch.bool, device=input_device),
                "full_chat": position_ids[0, 1:] != 0,
            }
            row_sources = torch.tensor(row["source_code"][1:], dtype=torch.long, device=input_device)
            example_indices = torch.tensor(row["example_index"][1:], dtype=torch.long, device=input_device)
            labels = input_ids[:, 1:]
            logits = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[:, :-1, :]
            pack_sums = {name: 0.0 for name in metric_names}
            pack_counts = {name: 0 for name in metric_names}
            for start in range(0, logits.shape[1], chunk_size):
                end = min(start + chunk_size, logits.shape[1])
                chunk_logits = logits[0, start:end].float()
                chunk_labels = labels[0, start:end]
                losses = functional.cross_entropy(chunk_logits, chunk_labels, reduction="none")
                correct = chunk_logits.argmax(dim=-1).eq(chunk_labels)
                for name, full_mask in masks.items():
                    selected = full_mask[start:end]
                    count = int(selected.sum().item())
                    if not count:
                        continue
                    selected_losses = losses[selected]
                    selected_correct = correct[selected]
                    loss_sum = float(selected_losses.sum().item())
                    correct_sum = int(selected_correct.sum().item())
                    totals[name]["loss"] += loss_sum
                    totals[name]["correct"] += correct_sum
                    totals[name]["tokens"] += count
                    pack_sums[name] += loss_sum
                    pack_counts[name] += count
                    selected_source_codes = row_sources[start:end][selected]
                    for code, source in source_codes.items():
                        source_mask = selected_source_codes == code
                        source_count = int(source_mask.sum().item())
                        if source_count:
                            source_totals[source][name]["loss"] += float(selected_losses[source_mask].sum().item())
                            source_totals[source][name]["correct"] += int(selected_correct[source_mask].sum().item())
                            source_totals[source][name]["tokens"] += source_count

                sql_selected = masks["sql"][start:end]
                if sql_selected.any():
                    indices = example_indices[start:end][sql_selected].cpu()
                    per_example_loss.index_add_(0, indices, losses[sql_selected].double().cpu())
                    per_example_correct.index_add_(0, indices, correct[sql_selected].long().cpu())
                    per_example_tokens.index_add_(0, indices, torch.ones_like(indices))
            for name in metric_names:
                require(pack_counts[name] > 0, f"Pack {packed_index} has no {name} targets")
                totals[name]["pack_means"].append(pack_sums[name] / pack_counts[name])
            del logits, input_ids, position_ids
            if (packed_index + 1) % int(eval_cfg.get("progress_every", 25)) == 0 or packed_index + 1 == len(packed):
                print(f"[{checkpoint['global_step']}] {packed_index + 1}/{len(packed)} packs", flush=True)

    require(int((per_example_tokens == 0).sum().item()) == 0, "At least one example has no SQL targets")
    per_example_losses = (per_example_loss / per_example_tokens).numpy()
    hardest = np.argsort(per_example_losses)[-20:][::-1]
    metrics = {
        "eval_full_chat_pack_macro_loss": statistics.fmean(totals["full_chat"]["pack_means"]),
        "eval_full_chat_micro": metric_result(totals["full_chat"]["loss"], totals["full_chat"]["correct"], totals["full_chat"]["tokens"]),
        "eval_assistant_completion_micro": metric_result(totals["assistant_completion"]["loss"], totals["assistant_completion"]["correct"], totals["assistant_completion"]["tokens"]),
        "eval_sql_micro": metric_result(totals["sql"]["loss"], totals["sql"]["correct"], totals["sql"]["tokens"]),
        "by_source_micro": {
            source: {
                name: metric_result(values["loss"], values["correct"], values["tokens"])
                for name, values in source_totals[source].items()
            }
            for source in source_names
        },
        "eval_sql_per_example_loss_summary": {
            "min": float(np.min(per_example_losses)),
            "mean": float(np.mean(per_example_losses)),
            "median": float(np.median(per_example_losses)),
            "p95": float(np.quantile(per_example_losses, 0.95)),
            "max": float(np.max(per_example_losses)),
        },
        "hardest_20_sql_examples": [
            {
                "id": example_ids[int(index)],
                "source": example_sources[int(index)],
                "loss": float(per_example_losses[int(index)]),
                "tokens": int(per_example_tokens[int(index)].item()),
                "token_accuracy": float(per_example_correct[int(index)].item() / per_example_tokens[int(index)].item()),
            }
            for index in hardest
        ],
    }
    if checkpoint["stored_trainer_eval_loss"] is not None:
        metrics["stored_trainer_eval_loss"] = checkpoint["stored_trainer_eval_loss"]
        metrics["pack_macro_minus_stored_trainer_eval_loss"] = metrics["eval_full_chat_pack_macro_loss"] - checkpoint["stored_trainer_eval_loss"]

    result = {
        "status": "complete",
        "diagnostic_only": True,
        "checkpoint_selection_changed": False,
        "generation_performed": False,
        "sql_execution_performed": False,
        "model": model_cfg,
        "checkpoint": checkpoint,
        "validation": validation_stats,
        "config_path": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "evaluator_path": relative(Path(__file__)),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "metric_definition": {
            "checkpoint_selection": "diagnostic only; primary training selection remains MixedVal-v2 Full-Chat eval_loss",
            "full_chat_pack_macro": "Unweighted mean of each packed sequence's mean causal NLL; trainer-compatible at eval batch size 1.",
            "full_chat_micro": "Token-weighted causal NLL over all non-document-start targets.",
            "sql_micro": "Token-weighted causal NLL over assistant SQL only.",
            "by_source": "Token-weighted micro metrics within each source; no source macro is used for selection.",
        },
        "metrics": metrics,
        "runtime": {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "seconds": time.perf_counter() - start_perf,
            "device": str(input_device),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "torch_dtype": dtype_name,
            "attn_implementation": eval_cfg["attn_implementation"],
            "batch_size": 1,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"Temporary output collision: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps({"status": "complete", "output": relative(output_path)}, indent=2))
    return result


def select_checkpoint(discovery: dict[str, Any], selector: str) -> dict[str, Any]:
    choices = discovery["unique_weights"]
    matches = [
        item
        for item in choices
        if selector in {item["path"], Path(item["path"]).name, str(item.get("global_step")), item["adapter_model_sha256"], item["adapter_model_sha256"][:12]}
    ]
    require(len(matches) == 1, f"Checkpoint selector matched {len(matches)} entries: {selector!r}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint path/name, step, or adapter hash")
    parser.add_argument("--run-all", action="store_true", help="Evaluate every unique discovered weight set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    require(config_path.is_file(), f"Config missing: {config_path}")
    config = load_json(config_path)
    config_summary = validate_config(config_path, config)
    packed, validation_stats, example_ids, example_sources, _tokenizer = prepare_packed_validation(config)
    discovery = discover_checkpoints(config)
    summary = {
        "status": "PASS",
        "config": config_summary,
        "validation": validation_stats,
        "checkpoint_discovery": discovery,
        "model_loaded": False,
        "adapter_loaded": False,
        "cuda_used": False,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    require(not (args.checkpoint and args.run_all), "Use either --checkpoint or --run-all")
    require(args.checkpoint or args.run_all, "Specify --checkpoint or --run-all for actual evaluation")
    require(discovery["adapter_root_exists"], "Adapter root does not exist yet")
    checkpoints = discovery["unique_weights"] if args.run_all else [select_checkpoint(discovery, args.checkpoint)]
    require(checkpoints, "No unique checkpoints discovered")
    for checkpoint in checkpoints:
        evaluate_checkpoint(config_path, config, checkpoint, packed, validation_stats, example_ids, example_sources)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
