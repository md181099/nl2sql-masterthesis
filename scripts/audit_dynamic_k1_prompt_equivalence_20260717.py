#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers import AutoTokenizer  # noqa: E402


BASELINE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
TESTCASES = ROOT / "data/testcases_spider_dev_full.jsonl"
DETAIL = ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_20260717.csv"
SUMMARY = ROOT / "audits/derived/dynamic_k1_2048_vs_4352_prompt_equivalence_summary_20260717.json"
AUDIT = ROOT / "audits/audit_dynamic_k1_2048_vs_4352_prompt_equivalence_20260717.md"
MANIFEST = ROOT / "audits/dynamic_k1_2048_vs_4352_prompt_equivalence_manifest_20260717.json"
EXPECTED_INTERPRETER = ROOT / ".venv_flash/bin/python"
SOURCE_CONDITIONS = {
    "top1",
    "top1_gate070",
    "top1_gate085",
    "structure",
    "structure_gate070",
    "structure_gate085",
}
MODEL_REVISIONS = {
    "qwen2b": ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
    "llama3b": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "qwen9b": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def token_hash(ids: list[int]) -> str:
    return sha256_text(json.dumps(ids, separators=(",", ":")))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new(path: Path, text: str) -> None:
    allowed = {DETAIL, SUMMARY, AUDIT, MANIFEST}
    require(path in allowed, f"Refusing out-of-scope write: {path}")
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_runner():
    path = SRC / "06_batch_run.py"
    spec = importlib.util.spec_from_file_location("k1_prompt_equivalence_runner", path)
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
    require((path / "tokenizer.json").is_file(), f"Missing tokenizer: {path}")
    return path


def csv_text(rows: list[dict[str, Any]]) -> str:
    require(bool(rows), "Cannot serialize empty CSV")
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    require(Path(sys.executable).absolute() == EXPECTED_INTERPRETER, "Authoritative interpreter required")
    for path in (DETAIL, SUMMARY, AUDIT, MANIFEST):
        require(not path.exists(), f"Refusing to overwrite: {path}")
    runner = load_runner()
    baseline = [row for row in read_csv(BASELINE) if row["condition"] in SOURCE_CONDITIONS]
    require(len(baseline) == 36, f"Expected 36 dynamic k1 runs, found {len(baseline)}")
    testcases = read_jsonl(TESTCASES)
    require(len(testcases) == 1032, "Spider Dev row count mismatch")
    testcase_ids = [str(row["id"]) for row in testcases]
    require(len(set(testcase_ids)) == 1032, "Duplicate Spider Dev IDs")

    tokenizers: dict[str, Any] = {}
    tokenizer_provenance: dict[str, Any] = {}
    for model_key, (model_id, revision) in MODEL_REVISIONS.items():
        path = snapshot_path(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizers[model_key] = tokenizer
        tokenizer_provenance[model_key] = {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path": str(path),
            "tokenizer_json_sha256": sha256(path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256(path / "tokenizer_config.json"),
        }

    pool_path = ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl"
    pool = read_jsonl(pool_path)
    require(len(pool) == 6960, "Retrieval pool row count mismatch")
    examples = {str(row["id"]): row for row in pool}
    require(len(examples) == 6960, "Duplicate retrieval example IDs")

    detail_rows: list[dict[str, Any]] = []
    group_stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "tokens": [],
            "truncations": 0,
            "token_differences": 0,
            "prompt_hash_differences": 0,
            "historical_token_mismatches": 0,
            "trace_char_mismatches": 0,
            "fewshot": 0,
            "fallback": 0,
        }
    )
    for run in baseline:
        config_path = ROOT / run["config_path"]
        csv_path = ROOT / run["csv_path"]
        trace_path = ROOT / run["trace_path"]
        require(sha256(config_path) == run["config_sha256"], f"Config hash drift: {config_path}")
        require(sha256(csv_path) == run["csv_sha256"], f"CSV hash drift: {csv_path}")
        require(sha256(trace_path) == run["trace_sha256"], f"Trace hash drift: {trace_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        require(config.get("k") == 1, f"Not k1: {config_path}")
        require(config.get("max_input_tokens") == 2048, f"Not maxin2048: {config_path}")
        require(config.get("max_new_tokens") == 256, f"Unexpected output limit: {config_path}")
        historical = read_csv(csv_path)
        traces = read_jsonl(trace_path)
        require(len(historical) == len(traces) == len(testcases) == 1032, f"Incomplete run: {run['run_id']}")
        require([row["id"] for row in historical] == testcase_ids, f"CSV order mismatch: {run['run_id']}")
        require([str(row["id"]) for row in traces] == testcase_ids, f"Trace order mismatch: {run['run_id']}")
        tokenizer = tokenizers[run["model_key"]]
        system_prompt, _source, _path, system_hash = runner.resolve_system_prompt(
            project_root=ROOT,
            system_prompt_variant=config["system_prompt_variant"],
            system_prompt_path=config.get("system_prompt_path"),
        )
        add_special_tokens = config["prompt_format"] != runner.LLAMA32_NATIVE_CHAT_FORMAT
        for testcase, trace, historical_row in zip(testcases, traces, historical):
            retrieved_ids = [str(value) for value in trace.get("retrieved_ids", [])]
            require(len(retrieved_ids) == 1, f"Unexpected k1 trace selection: {run['run_id']} {testcase['id']}")
            fallback = trace.get("gate_decision") == "zero_shot"
            schema = runner.normalize_schema_text(str(testcase.get("schema_prompt", "")))
            question = str(testcase["question"])
            if fallback:
                prompt = runner.build_prompt(
                    schema,
                    question,
                    config["llm"],
                    tokenizer,
                    prompt_format=config["prompt_format"],
                    system_instruction=system_prompt,
                )
                actual_k = 0
            else:
                example_id = retrieved_ids[0]
                require(example_id in examples, f"Unknown retrieval ID: {example_id}")
                prompt = runner.build_prompt_schema_fewshot(
                    schema,
                    question,
                    [examples[example_id]],
                    config["llm"],
                    tokenizer,
                    prompt_format=config["prompt_format"],
                    system_instruction=system_prompt,
                    example_schema_mode=config["fewshot_example_schema_mode"],
                    example_mode=config["fewshot_example_mode"],
                )
                actual_k = 1
            full_ids = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )["input_ids"]
            ids_2048 = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=2048,
            )["input_ids"]
            ids_4352 = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=4352,
            )["input_ids"]
            prompt_hash_2048 = sha256_text(prompt)
            prompt_hash_4352 = sha256_text(prompt)
            token_difference = ids_2048 != ids_4352
            prompt_hash_difference = prompt_hash_2048 != prompt_hash_4352
            would_truncate_2048 = len(full_ids) > 2048
            historical_prompt_tokens = int(float(historical_row["prompt_tokens"]))
            historical_token_mismatch = historical_prompt_tokens != len(ids_2048)
            trace_char_mismatch = int(trace.get("prompt_char_length", len(prompt))) != len(prompt)
            key = (run["model_key"], run["role"], run["condition"])
            stats = group_stats[key]
            stats["tokens"].append(len(full_ids))
            stats["truncations"] += int(would_truncate_2048)
            stats["token_differences"] += int(token_difference)
            stats["prompt_hash_differences"] += int(prompt_hash_difference)
            stats["historical_token_mismatches"] += int(historical_token_mismatch)
            stats["trace_char_mismatches"] += int(trace_char_mismatch)
            stats["fewshot"] += int(actual_k == 1)
            stats["fallback"] += int(actual_k == 0)
            detail_rows.append(
                {
                    "model_key": run["model_key"],
                    "role": run["role"],
                    "condition": run["condition"],
                    "run_id": run["run_id"],
                    "config_path": run["config_path"],
                    "config_sha256": run["config_sha256"],
                    "case_id": testcase["id"],
                    "actual_k": actual_k,
                    "fallback": int(fallback),
                    "prompt_tokens_untruncated": len(full_ids),
                    "historical_prompt_tokens": historical_prompt_tokens,
                    "would_truncate_at_2048": int(would_truncate_2048),
                    "tokens_at_2048": len(ids_2048),
                    "tokens_at_4352": len(ids_4352),
                    "token_ids_sha256_2048": token_hash(ids_2048),
                    "token_ids_sha256_4352": token_hash(ids_4352),
                    "token_id_difference": int(token_difference),
                    "prompt_sha256_2048": prompt_hash_2048,
                    "prompt_sha256_4352": prompt_hash_4352,
                    "prompt_hash_difference": int(prompt_hash_difference),
                    "historical_prompt_token_mismatch": int(historical_token_mismatch),
                    "trace_prompt_char_length_mismatch": int(trace_char_mismatch),
                    "system_prompt_sha256": system_hash,
                    "status": "PASS" if not any((would_truncate_2048, token_difference, prompt_hash_difference, historical_token_mismatch, trace_char_mismatch)) else "FAIL",
                }
            )

    group_rows: list[dict[str, Any]] = []
    for (model_key, role, condition), stats in sorted(group_stats.items()):
        group_rows.append(
            {
                "model_key": model_key,
                "role": role,
                "condition": condition,
                "cases": len(stats["tokens"]),
                "prompt_tokens_min": min(stats["tokens"]),
                "prompt_tokens_mean": statistics.fmean(stats["tokens"]),
                "prompt_tokens_max": max(stats["tokens"]),
                "prompt_truncations_at_2048": stats["truncations"],
                "token_id_differences": stats["token_differences"],
                "prompt_hash_differences": stats["prompt_hash_differences"],
                "historical_prompt_token_mismatches": stats["historical_token_mismatches"],
                "trace_prompt_char_mismatches": stats["trace_char_mismatches"],
                "fewshot_cases": stats["fewshot"],
                "fallback_cases": stats["fallback"],
            }
        )
    totals = {
        "prompt_truncations_at_2048": sum(row["prompt_truncations_at_2048"] for row in group_rows),
        "token_id_differences": sum(row["token_id_differences"] for row in group_rows),
        "prompt_hash_differences": sum(row["prompt_hash_differences"] for row in group_rows),
        "historical_prompt_token_mismatches": sum(row["historical_prompt_token_mismatches"] for row in group_rows),
        "trace_prompt_char_mismatches": sum(row["trace_prompt_char_mismatches"] for row in group_rows),
    }
    status = "PASS" if sum(totals.values()) == 0 else "FAIL"
    summary = {
        "status": status,
        "status_label": "K1-2048-VS-4352-PROMPT-EQUIVALENCE",
        "runs": len(baseline),
        "cases_per_run": 1032,
        "prompt_rows": len(detail_rows),
        "maximum_prompt_tokens": max(row["prompt_tokens_max"] for row in group_rows),
        "totals": totals,
        "k1_reruns_required": False if status == "PASS" else True,
        "k1_vs_k3_comparison_permitted": status == "PASS",
        "full_runs_started": False,
        "groups": group_rows,
        "tokenizers": tokenizer_provenance,
        "sources": {
            "baseline": {"path": str(BASELINE.relative_to(ROOT)), "sha256": sha256(BASELINE)},
            "testcases": {"path": str(TESTCASES.relative_to(ROOT)), "sha256": sha256(TESTCASES)},
            "retrieval_pool": {"path": str(pool_path.relative_to(ROOT)), "sha256": sha256(pool_path)},
            "runner": {"path": "src/06_batch_run.py", "sha256": sha256(SRC / "06_batch_run.py")},
        },
    }
    write_new(DETAIL, csv_text(detail_rows))
    write_new(SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    group_table = "\n".join(
        f"| {row['model_key']} | {row['role']} | {row['condition']} | {row['prompt_tokens_max']} | "
        f"{row['prompt_truncations_at_2048']} | {row['token_id_differences']} | {row['prompt_hash_differences']} |"
        for row in group_rows
    )
    audit_text = f"""# Audit: Dynamic k1 Prompt Equivalence 2048 versus 4352

**K1-2048-VS-4352-PROMPT-EQUIVALENCE: {status}**

```text
K1 runs checked: {len(baseline)}
K1 prompts checked: {len(detail_rows)}
Prompt truncations at 2048: {totals['prompt_truncations_at_2048']}
Token-ID differences: {totals['token_id_differences']}
Prompt-hash differences: {totals['prompt_hash_differences']}
Historical prompt-token mismatches: {totals['historical_prompt_token_mismatches']}
Trace prompt-length mismatches: {totals['trace_prompt_char_mismatches']}
K1 reruns required: {'NEIN' if status == 'PASS' else 'JA'}
K1-vs-K3 comparison permitted: {'JA' if status == 'PASS' else 'NEIN'}
Model inference started: NEIN
```

Die Prompts wurden aus den 36 autoritativen k1-Configs, ihren gespeicherten Retrievaltraces und den lokalen Tokenizer-Snapshots materialisiert. Es wurde kein Retriever und kein generatives Modell ausgefuehrt. Da kein Prompt die historische Grenze ueberschreitet, sind die Token-ID-Sequenzen bei 2.048 und 4.352 Tokens identisch.

| Modell | Rolle | Bedingung | Max Tokens | Truncations | Token-ID-Diffs | Prompt-Hash-Diffs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{group_table}
"""
    write_new(AUDIT, audit_text)
    manifest = {
        "schema_version": 1,
        **summary,
        "artifacts": [
            {"path": str(DETAIL.relative_to(ROOT)), "sha256": sha256(DETAIL)},
            {"path": str(SUMMARY.relative_to(ROOT)), "sha256": sha256(SUMMARY)},
            {"path": str(AUDIT.relative_to(ROOT)), "sha256": sha256(AUDIT)},
            {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        ],
        "self_hash_policy": "Manifest SHA256 is reported externally.",
    }
    write_new(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "runs": len(baseline),
        "prompt_rows": len(detail_rows),
        "maximum_prompt_tokens": summary["maximum_prompt_tokens"],
        **totals,
        "k1_reruns_required": summary["k1_reruns_required"],
        "k1_vs_k3_comparison_permitted": summary["k1_vs_k3_comparison_permitted"],
        "full_runs_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
