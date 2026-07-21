#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llama32_native_chat import (  # noqa: E402
    LLAMA32_3B_INSTRUCT_MODEL_ID,
    LLAMA32_3B_INSTRUCT_REVISION,
    llama32_assistant_generation_prefix,
)
from prompt_presets import resolve_system_prompt  # noqa: E402


OUTPUT = PROJECT_ROOT / "audits/derived/llama32_3b_instruct_prompt_smoke_20260714.json"
TESTCASES = PROJECT_ROOT / "data/testcases_spider_dev_full.jsonl"
METADATA = PROJECT_ROOT / (
    "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
)
STATIC = PROJECT_ROOT / (
    "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl"
)
TOP1_TRACE = PROJECT_ROOT / (
    "results/retrieval_traces/run_base_20260712_143438_retrieval_traces.jsonl"
)
STRUCTURE_TRACE = PROJECT_ROOT / (
    "results/retrieval_traces/run_base_20260712_160614_retrieval_traces.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    remainder = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * remainder


def load_runner() -> Any:
    path = SRC_DIR / "06_batch_run.py"
    spec = importlib.util.spec_from_file_location("batch_run_llama_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prompt-smoke output: {output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA32_3B_INSTRUCT_MODEL_ID,
        revision=LLAMA32_3B_INSTRUCT_REVISION,
        local_files_only=True,
    )
    runner = load_runner()
    system_prompt, source, _path, system_hash = resolve_system_prompt(
        project_root=PROJECT_ROOT,
        system_prompt_variant="sqlctx_anti_overjoin",
        system_prompt_path=None,
    )
    tests = load_jsonl(TESTCASES)
    metadata = {row["id"]: row for row in load_jsonl(METADATA)}
    static_rows = load_jsonl(STATIC)
    if len(static_rows) != 1 or static_rows[0].get("id") != "SPIDER_TRAIN_001657":
        raise RuntimeError("Unexpected materialized static demonstration")
    top1 = {row["id"]: row for row in load_jsonl(TOP1_TRACE)}
    structure = {row["id"]: row for row in load_jsonl(STRUCTURE_TRACE)}
    if not (len(tests) == len(top1) == len(structure) == 1032):
        raise RuntimeError("Prompt-smoke inputs do not contain exactly 1032 aligned cases")

    conditions = {
        "zero_shot": (None, None),
        "dynamic_top1": ("top1", None),
        "dynamic_top1_gate070": ("top1", 0.70),
        "dynamic_top1_gate085": ("top1", 0.85),
        "static_seed42": ("static", None),
        "structure_top10_v2": ("structure", None),
        "structure_top10_v2_gate070": ("structure", 0.70),
        "structure_top10_v2_gate085": ("structure", 0.85),
    }
    assistant_prefix = llama32_assistant_generation_prefix(tokenizer)
    result: dict[str, Any] = {
        "status": "PASS",
        "model_id": LLAMA32_3B_INSTRUCT_MODEL_ID,
        "model_revision": LLAMA32_3B_INSTRUCT_REVISION,
        "system_prompt_source": source,
        "system_prompt_sha256": system_hash,
        "testcases": str(TESTCASES.relative_to(PROJECT_ROOT)),
        "top1_trace": str(TOP1_TRACE.relative_to(PROJECT_ROOT)),
        "structure_trace": str(STRUCTURE_TRACE.relative_to(PROJECT_ROOT)),
        "retrieval_model_loaded": False,
        "language_model_loaded": False,
        "conditions": {},
    }

    for condition, (selection_type, threshold) in conditions.items():
        lengths: list[int] = []
        fewshot_count = 0
        zero_count = 0
        empty_demos = 0
        multiple_demos = 0
        missing_schemas = 0
        missing_questions = 0
        missing_prefix = 0
        qwen_tokens = 0
        think_tokens = 0
        invalid = 0

        for testcase in tests:
            qid = str(testcase["id"])
            schema = runner.normalize_schema_text(str(testcase.get("schema_prompt", "")))
            question = str(testcase.get("question", ""))
            if not schema:
                missing_schemas += 1
            if not question:
                missing_questions += 1

            demos: list[dict[str, Any]] = []
            if selection_type == "static":
                demos = static_rows
            elif selection_type in {"top1", "structure"}:
                trace = top1[qid] if selection_type == "top1" else structure[qid]
                score = float(trace["retrieved_scores"][0])
                if threshold is None or score >= threshold:
                    demo_id = str(trace["retrieved_ids"][0])
                    demos = [metadata[demo_id]]

            if demos:
                fewshot_count += 1
                empty_demos += int(any(not str(demo.get("question", "")).strip() for demo in demos))
                multiple_demos += int(len(demos) != 1)
                prompt = runner.build_prompt_schema_fewshot(
                    schema,
                    question,
                    demos,
                    "llama32_3b_instruct",
                    tokenizer,
                    prompt_format="llama32_instruct_native_chat",
                    system_instruction=system_prompt,
                    example_schema_mode="full",
                    example_mode="schema_with_rules",
                )
            else:
                zero_count += 1
                prompt = runner.build_prompt(
                    schema,
                    question,
                    "llama32_3b_instruct",
                    tokenizer,
                    prompt_format="llama32_instruct_native_chat",
                    system_instruction=system_prompt,
                )

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            lengths.append(len(prompt_ids))
            missing_prefix += int(not prompt.endswith(assistant_prefix))
            qwen_tokens += int("<|im_start|>" in prompt or "<|im_end|>" in prompt)
            think_tokens += int("<think" in prompt.lower())
            invalid += int(
                not schema
                or not question
                or not prompt.endswith(assistant_prefix)
                or "<|im_start|>" in prompt
                or "<|im_end|>" in prompt
                or "<think" in prompt.lower()
                or len(prompt_ids) > 2048
                or len(demos) > 1
            )

        condition_stats = {
            "cases": len(tests),
            "few_shot": fewshot_count,
            "zero_shot": zero_count,
            "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths),
            "mean_tokens": statistics.fmean(lengths),
            "median_tokens": statistics.median(lengths),
            "p95_tokens": quantile(lengths, 0.95),
            "p99_tokens": quantile(lengths, 0.99),
            "over_2048": sum(length > 2048 for length in lengths),
            "exactly_2048": sum(length == 2048 for length in lengths),
            "truncations": 0,
            "empty_demos": empty_demos,
            "more_than_one_demo": multiple_demos,
            "missing_schemas": missing_schemas,
            "missing_questions": missing_questions,
            "missing_assistant_prefix": missing_prefix,
            "qwen_special_token_prompts": qwen_tokens,
            "think_token_prompts": think_tokens,
            "invalid_prompts": invalid,
        }
        result["conditions"][condition] = condition_stats
        if condition_stats["over_2048"] or condition_stats["invalid_prompts"]:
            result["status"] = "FAIL"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
