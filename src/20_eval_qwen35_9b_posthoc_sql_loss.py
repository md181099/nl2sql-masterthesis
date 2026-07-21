#!/usr/bin/env python3
"""Post-hoc teacher-forced SQL-loss diagnostics for frozen LoRA checkpoints.

This script never generates SQL and never executes SQL. It evaluates one named
checkpoint from a run-matrix config and writes one new, immutable JSON result.
Use --validate-only to verify all inputs, masks, and packing without loading a
language model or touching CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_MARKER = "<|im_start|>assistant\n"
ANSWER_END_MARKER = "<|im_end|>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_by_id(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    matches = [run for run in config["runs"] if run.get("id") == run_id]
    require(len(matches) == 1, f"Expected exactly one run with id={run_id!r}; found {len(matches)}")
    return matches[0]


def validate_checkpoint(run: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    checkpoint = resolve_path(run["adapter_checkpoint"])
    require(checkpoint.is_dir(), f"Checkpoint directory missing: {checkpoint}")
    model_file = checkpoint / "adapter_model.safetensors"
    adapter_config_file = checkpoint / "adapter_config.json"
    trainer_state_file = checkpoint / "trainer_state.json"
    for path in (model_file, adapter_config_file, trainer_state_file):
        require(path.is_file() and path.stat().st_size > 0, f"Required checkpoint file missing/empty: {path}")

    model_sha = sha256_file(model_file)
    config_sha = sha256_file(adapter_config_file)
    require(model_sha == run["expected_adapter_model_sha256"], f"Adapter SHA256 mismatch: {checkpoint}")
    require(config_sha == run["expected_adapter_config_sha256"], f"Adapter config SHA256 mismatch: {checkpoint}")

    adapter_config = load_json(adapter_config_file)
    expected_lora = common["expected_lora"]
    for key in ("r", "lora_alpha", "lora_dropout", "task_type"):
        require(adapter_config.get(key) == expected_lora[key], f"Unexpected {key} in {adapter_config_file}")
    require(
        sorted(adapter_config.get("target_modules", [])) == sorted(expected_lora["target_modules"]),
        f"Unexpected target_modules in {adapter_config_file}",
    )
    require(adapter_config.get("bias") == expected_lora["bias"], f"Unexpected bias in {adapter_config_file}")
    require(adapter_config.get("use_dora") is expected_lora["use_dora"], f"Unexpected use_dora in {adapter_config_file}")
    require(
        adapter_config.get("base_model_name_or_path") == common["model_id"],
        f"Unexpected base model in {adapter_config_file}",
    )

    state = load_json(trainer_state_file)
    require(state.get("global_step") == run["global_step"], f"Unexpected global_step in {trainer_state_file}")
    eval_entries = [entry for entry in state.get("log_history", []) if "eval_loss" in entry]
    matching_eval = [entry for entry in eval_entries if entry.get("step") == run["global_step"]]
    require(len(matching_eval) == 1, f"Missing unique eval entry for {run['id']}")
    stored_loss = float(matching_eval[0]["eval_loss"])
    require(
        abs(stored_loss - float(run["stored_official_full_chat_eval_loss"])) < 1e-12,
        f"Stored eval_loss differs from config provenance for {run['id']}",
    )

    output_path = resolve_path(run["output_path"])
    return {
        "id": run["id"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "adapter_model_sha256": model_sha,
        "adapter_config_sha256": config_sha,
        "global_step": state["global_step"],
        "epoch": run["epoch"],
        "stored_official_full_chat_eval_loss": stored_loss,
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_exists": output_path.exists(),
    }


def source_name(example_id: str) -> str:
    if example_id.startswith("SPIDER_TRAIN_OTHERS_"):
        return "spider_train_others"
    if example_id.startswith("SCC_"):
        return "sql_create_context"
    raise ValueError(f"Unknown validation source for id={example_id!r}")


def token_mask(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [int(offset_end > start and offset_start < end) for offset_start, offset_end in offsets]


def prepare_packed_validation(config: dict[str, Any]) -> tuple[Any, dict[str, Any], list[str], list[str]]:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import pack_dataset

    common = config["common"]
    validation_path = resolve_path(common["validation_path"])
    require(validation_path.is_file(), f"Validation dataset missing: {validation_path}")
    validation_sha = sha256_file(validation_path)
    require(validation_sha == common["validation_sha256"], "Frozen validation SHA256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        common["model_id"],
        revision=common.get("model_revision"),
        local_files_only=bool(common.get("local_files_only", True)),
    )
    require(tokenizer.is_fast, "A fast tokenizer with offset mappings is required")
    require(tokenizer.eos_token is not None, "Tokenizer has no EOS token")

    columns: dict[str, list[list[int]]] = {
        "input_ids": [],
        "sql_mask": [],
        "assistant_completion_mask": [],
        "source_code": [],
        "example_index": [],
    }
    ids: list[str] = []
    sources: list[str] = []
    raw_lengths: list[int] = []
    sql_lengths: list[int] = []
    completion_lengths: list[int] = []
    source_rows: Counter[str] = Counter()
    source_sql_tokens: Counter[str] = Counter()
    source_completion_tokens: Counter[str] = Counter()

    with validation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example_id = row.get("id")
            text = row.get("text")
            require(isinstance(example_id, str) and example_id, f"Invalid id at line {line_number}")
            require(isinstance(text, str) and text, f"Invalid text at line {line_number}")
            require(text.count(ASSISTANT_MARKER) == 1, f"Assistant marker count != 1 for {example_id}")

            sql_start = text.index(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
            sql_end = text.find(ANSWER_END_MARKER, sql_start)
            require(sql_end > sql_start, f"Missing or empty assistant SQL for {example_id}")
            completion_end = sql_end + len(ANSWER_END_MARKER)

            # Match TRL 1.4.0: append tokenizer EOS to non-conversational text when needed.
            tokenized_text = text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token
            encoded = tokenizer(
                tokenized_text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            sql_mask = token_mask(offsets, sql_start, sql_end)
            completion_mask = token_mask(offsets, sql_start, completion_end)
            require(sum(sql_mask) > 0, f"Empty SQL token mask for {example_id}")
            require(sum(completion_mask) > sum(sql_mask), f"Answer-end token not represented for {example_id}")
            require(len(input_ids) <= int(common["max_length"]), f"Over-length validation row: {example_id}")

            source = source_name(example_id)
            source_code = 0 if source == "spider_train_others" else 1
            example_index = len(ids)
            columns["input_ids"].append(input_ids)
            columns["sql_mask"].append(sql_mask)
            columns["assistant_completion_mask"].append(completion_mask)
            columns["source_code"].append([source_code] * len(input_ids))
            columns["example_index"].append([example_index] * len(input_ids))
            ids.append(example_id)
            sources.append(source)
            raw_lengths.append(len(input_ids))
            sql_lengths.append(sum(sql_mask))
            completion_lengths.append(sum(completion_mask))
            source_rows[source] += 1
            source_sql_tokens[source] += sum(sql_mask)
            source_completion_tokens[source] += sum(completion_mask)

    expected = common["expected_validation"]
    require(len(ids) == expected["rows"], f"Expected {expected['rows']} validation rows; found {len(ids)}")
    require(dict(source_rows) == expected["source_rows"], f"Unexpected validation source counts: {source_rows}")
    require(sum(sql_lengths) == expected["sql_tokens"], "Unexpected pure-SQL token count")
    require(sum(completion_lengths) == expected["assistant_completion_tokens"], "Unexpected completion token count")
    require(max(raw_lengths) == expected["max_tokens_with_eos"], "Unexpected maximum validation length")

    packed = pack_dataset(
        Dataset.from_dict(columns),
        seq_length=int(common["max_length"]),
        strategy=common["packing_strategy"],
    )
    packed_tokens = sum(len(row) for row in packed["input_ids"])
    packed_sql_tokens = sum(sum(row) for row in packed["sql_mask"])
    packed_completion_tokens = sum(sum(row) for row in packed["assistant_completion_mask"])
    packed_examples = sum(len(lengths) for lengths in packed["seq_lengths"])
    require(len(packed) == expected["packed_sequences"], f"Unexpected packed sequence count: {len(packed)}")
    require(packed_examples == len(ids), "Packing lost or duplicated examples")
    require(packed_tokens == sum(raw_lengths), "Packing lost or duplicated tokens")
    require(packed_sql_tokens == sum(sql_lengths), "Packing changed SQL masks")
    require(packed_completion_tokens == sum(completion_lengths), "Packing changed completion masks")

    stats = {
        "validation_path": str(validation_path.relative_to(PROJECT_ROOT)),
        "validation_sha256": validation_sha,
        "rows": len(ids),
        "source_rows": dict(source_rows),
        "raw_tokens_with_eos": sum(raw_lengths),
        "raw_length_min": min(raw_lengths),
        "raw_length_mean": statistics.fmean(raw_lengths),
        "raw_length_max": max(raw_lengths),
        "sql_tokens": sum(sql_lengths),
        "assistant_completion_tokens": sum(completion_lengths),
        "source_sql_tokens": dict(source_sql_tokens),
        "source_assistant_completion_tokens": dict(source_completion_tokens),
        "packed_sequences": len(packed),
        "packed_examples": packed_examples,
        "packed_tokens": packed_tokens,
        "max_packed_length": max(len(row) for row in packed["input_ids"]),
        "truncated_examples": 0,
        "mask_definition": {
            "eval_sql_loss": "Assistant SQL statement tokens only; excludes <|im_end|> and technical EOS.",
            "eval_assistant_completion_loss": "Assistant SQL plus its first <|im_end|>; excludes technical packing EOS.",
        },
    }
    return packed, stats, ids, sources


def position_ids_from_seq_lengths(seq_lengths: list[int]) -> list[int]:
    result: list[int] = []
    for length in seq_lengths:
        result.extend(range(length))
    return result


def metric_result(total_loss: float, total_correct: int, total_tokens: int) -> dict[str, Any]:
    require(total_tokens > 0, "Cannot finalize an empty metric")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else float("inf"),
        "token_accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }


def evaluate_run(
    config_path: Path,
    config: dict[str, Any],
    run: dict[str, Any],
    packed: Any,
    validation_stats: dict[str, Any],
    example_ids: list[str],
    example_sources: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import transformers
    import trl
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    require(torch.cuda.is_available(), "CUDA is required for the 9B diagnostic; no model was loaded")
    common = config["common"]
    checkpoint = resolve_path(run["adapter_checkpoint"])
    output_path = resolve_path(run["output_path"])
    require(not output_path.exists(), f"Refusing to overwrite existing result: {output_path}")

    dtype_name = common["torch_dtype"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    torch.manual_seed(int(common["seed"]))
    torch.cuda.manual_seed_all(int(common["seed"]))

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        common["model_id"],
        revision=common.get("model_revision"),
        local_files_only=bool(common.get("local_files_only", True)),
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=common["attn_implementation"],
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint), is_trainable=False)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    metric_names = ("sql", "assistant_completion", "full_chat")
    totals = {name: {"loss": 0.0, "correct": 0, "tokens": 0, "pack_means": []} for name in metric_names}
    source_totals = {
        source: {"loss": 0.0, "correct": 0, "tokens": 0}
        for source in ("spider_train_others", "sql_create_context")
    }
    per_example_loss = torch.zeros(len(example_ids), dtype=torch.float64)
    per_example_correct = torch.zeros(len(example_ids), dtype=torch.int64)
    per_example_tokens = torch.zeros(len(example_ids), dtype=torch.int64)
    chunk_size = int(common.get("logit_chunk_size", 128))

    with torch.inference_mode():
        for packed_index, row in enumerate(packed):
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=input_device)
            position_ids = torch.tensor(
                [position_ids_from_seq_lengths(row["seq_lengths"])], dtype=torch.long, device=input_device
            )
            sql_mask = torch.tensor(row["sql_mask"][1:], dtype=torch.bool, device=input_device)
            completion_mask = torch.tensor(
                row["assistant_completion_mask"][1:], dtype=torch.bool, device=input_device
            )
            full_mask = position_ids[0, 1:] != 0
            source_codes = torch.tensor(row["source_code"][1:], dtype=torch.long, device=input_device)
            example_indices = torch.tensor(row["example_index"][1:], dtype=torch.long, device=input_device)
            labels = input_ids[:, 1:]

            logits = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[:, :-1, :]
            pack_sums = {name: 0.0 for name in metric_names}
            pack_counts = {name: 0 for name in metric_names}
            masks = {"sql": sql_mask, "assistant_completion": completion_mask, "full_chat": full_mask}

            for start in range(0, logits.shape[1], chunk_size):
                end = min(start + chunk_size, logits.shape[1])
                chunk_logits = logits[0, start:end].float()
                chunk_labels = labels[0, start:end]
                losses = functional.cross_entropy(chunk_logits, chunk_labels, reduction="none")
                predictions = chunk_logits.argmax(dim=-1)
                correct = predictions.eq(chunk_labels)

                for name, mask in masks.items():
                    selected = mask[start:end]
                    count = int(selected.sum().item())
                    if count:
                        loss_sum = float(losses[selected].sum().item())
                        correct_sum = int(correct[selected].sum().item())
                        totals[name]["loss"] += loss_sum
                        totals[name]["correct"] += correct_sum
                        totals[name]["tokens"] += count
                        pack_sums[name] += loss_sum
                        pack_counts[name] += count

                sql_selected = sql_mask[start:end]
                if sql_selected.any():
                    sql_losses_cpu = losses[sql_selected].double().cpu()
                    sql_correct_cpu = correct[sql_selected].long().cpu()
                    sql_examples_cpu = example_indices[start:end][sql_selected].cpu()
                    per_example_loss.index_add_(0, sql_examples_cpu, sql_losses_cpu)
                    per_example_correct.index_add_(0, sql_examples_cpu, sql_correct_cpu)
                    per_example_tokens.index_add_(0, sql_examples_cpu, torch.ones_like(sql_examples_cpu))
                    selected_sources = source_codes[start:end][sql_selected]
                    for code, source in ((0, "spider_train_others"), (1, "sql_create_context")):
                        source_mask = selected_sources == code
                        source_count = int(source_mask.sum().item())
                        if source_count:
                            source_totals[source]["loss"] += float(losses[sql_selected][source_mask].sum().item())
                            source_totals[source]["correct"] += int(correct[sql_selected][source_mask].sum().item())
                            source_totals[source]["tokens"] += source_count

            for name in metric_names:
                require(pack_counts[name] > 0, f"Packed sequence {packed_index} has no {name} targets")
                totals[name]["pack_means"].append(pack_sums[name] / pack_counts[name])
            del logits, input_ids, position_ids
            if (packed_index + 1) % int(common.get("progress_every", 25)) == 0 or packed_index + 1 == len(packed):
                print(f"[{run['id']}] {packed_index + 1}/{len(packed)} packed sequences", flush=True)

    require(int((per_example_tokens == 0).sum().item()) == 0, "At least one example has no SQL target tokens")
    per_example_losses = (per_example_loss / per_example_tokens).numpy()
    hardest_indices = np.argsort(per_example_losses)[-20:][::-1]
    full_chat_pack_macro = statistics.fmean(totals["full_chat"]["pack_means"])
    sql_result = metric_result(totals["sql"]["loss"], totals["sql"]["correct"], totals["sql"]["tokens"])
    completion_result = metric_result(
        totals["assistant_completion"]["loss"],
        totals["assistant_completion"]["correct"],
        totals["assistant_completion"]["tokens"],
    )
    full_chat_result = metric_result(
        totals["full_chat"]["loss"], totals["full_chat"]["correct"], totals["full_chat"]["tokens"]
    )
    finished = datetime.now(timezone.utc)

    result = {
        "status": "complete",
        "diagnostic_only": True,
        "official_checkpoint_selection_changed": False,
        "generation_performed": False,
        "sql_execution_performed": False,
        "run_id": run["id"],
        "learning_rate": run["learning_rate"],
        "epoch": run["epoch"],
        "global_step": run["global_step"],
        "model_id": common["model_id"],
        "model_revision": common.get("model_revision"),
        "adapter_checkpoint": run["adapter_checkpoint"],
        "adapter_model_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "validation": validation_stats,
        "metric_definition": {
            "primary": "eval_sql_loss",
            "aggregation": "micro-average negative log-likelihood over all selected target tokens",
            "checkpoint_selection": "diagnostic only; the historical Full-Chat eval_loss selection remains official",
        },
        "metrics": {
            "eval_sql_loss": sql_result["loss"],
            "eval_sql_perplexity": sql_result["perplexity"],
            "eval_sql_token_accuracy": sql_result["token_accuracy"],
            "eval_sql_tokens": sql_result["tokens"],
            "eval_assistant_completion_loss": completion_result["loss"],
            "eval_assistant_completion_token_accuracy": completion_result["token_accuracy"],
            "eval_assistant_completion_tokens": completion_result["tokens"],
            "eval_full_chat_micro_loss": full_chat_result["loss"],
            "eval_full_chat_micro_token_accuracy": full_chat_result["token_accuracy"],
            "eval_full_chat_tokens": full_chat_result["tokens"],
            "eval_full_chat_pack_macro_loss": full_chat_pack_macro,
            "stored_official_full_chat_eval_loss": run["stored_official_full_chat_eval_loss"],
            "pack_macro_minus_stored_eval_loss": full_chat_pack_macro - run["stored_official_full_chat_eval_loss"],
            "eval_sql_by_source": {
                source: metric_result(values["loss"], values["correct"], values["tokens"])
                for source, values in source_totals.items()
            },
            "eval_sql_per_example_loss_summary": {
                "min": float(np.min(per_example_losses)),
                "mean": float(np.mean(per_example_losses)),
                "median": float(np.median(per_example_losses)),
                "p95": float(np.quantile(per_example_losses, 0.95)),
                "max": float(np.max(per_example_losses)),
            },
            "hardest_20_examples_by_sql_loss": [
                {
                    "id": example_ids[int(index)],
                    "source": example_sources[int(index)],
                    "sql_loss": float(per_example_losses[int(index)]),
                    "sql_tokens": int(per_example_tokens[int(index)].item()),
                    "sql_token_accuracy": float(
                        per_example_correct[int(index)].item() / per_example_tokens[int(index)].item()
                    ),
                }
                for index in hardest_indices
            ],
        },
        "runtime": {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "seconds": time.perf_counter() - start_perf,
            "device": str(input_device),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "torch_dtype": dtype_name,
            "attn_implementation": common["attn_implementation"],
            "batch_size": 1,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    require(not temporary_path.exists(), f"Unexpected temporary-file collision: {temporary_path}")
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    print(json.dumps({"status": "complete", "run_id": run["id"], "output": str(output_path)}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Run-matrix JSON config")
    parser.add_argument("--run-id", help="One run id from the config")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all files, masks, and BFD packing without loading a model or CUDA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    require(config_path.is_file(), f"Config missing: {config_path}")
    config = load_json(config_path)
    require(config.get("schema_version") == 1, "Unsupported config schema_version")
    require(config.get("purpose") == "posthoc_sql_loss_diagnostic", "Unexpected config purpose")
    require(isinstance(config.get("runs"), list) and len(config["runs"]) == 6, "Expected exactly six runs")
    ids = [run.get("id") for run in config["runs"]]
    require(len(set(ids)) == len(ids), "Duplicate run IDs")

    validated_runs = [validate_checkpoint(run, config["common"]) for run in config["runs"]]
    packed, validation_stats, example_ids, example_sources = prepare_packed_validation(config)
    validation_summary = {
        "status": "pass",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "runs": validated_runs,
        "validation": validation_stats,
        "model_loaded": False,
        "cuda_used": False,
    }
    if args.validate_only:
        print(json.dumps(validation_summary, indent=2, sort_keys=True))
        return

    require(args.run_id is not None, "--run-id is required unless --validate-only is used")
    selected_run = run_by_id(config, args.run_id)
    evaluate_run(
        config_path,
        config,
        selected_run,
        packed,
        validation_stats,
        example_ids,
        example_sources,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
