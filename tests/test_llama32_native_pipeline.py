from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama32_native_chat import (  # noqa: E402
    LLAMA32_3B_INSTRUCT_MODEL_ID,
    LLAMA32_3B_INSTRUCT_REVISION,
    llama32_assistant_generation_prefix,
    llama32_generation_stop_token_ids,
    render_llama32_native_chat,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    path = SRC / "06_batch_run.py"
    spec = importlib.util.spec_from_file_location("batch_run_llama_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Llama32NativePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA32_3B_INSTRUCT_MODEL_ID,
            revision=LLAMA32_3B_INSTRUCT_REVISION,
            local_files_only=True,
        )
        cls.runner = load_runner()

    def test_native_chat_roundtrip_and_special_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "SELECT 1;"},
        ]
        rendered = render_llama32_native_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=False,
        )
        direct = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            date_string="26 Jul 2024",
        )["input_ids"]
        roundtrip = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        self.assertEqual(direct, roundtrip)
        self.assertEqual(roundtrip.count(self.tokenizer.bos_token_id), 1)
        self.assertEqual(rendered.count("<|eot_id|>"), 3)
        self.assertNotIn("<|im_start|>", rendered)
        self.assertNotIn("<|im_end|>", rendered)
        self.assertNotIn("<think", rendered.lower())

    def test_native_evaluation_generation_prefix_and_stop_ids(self) -> None:
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
        ]
        rendered = render_llama32_native_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )
        prefix = llama32_assistant_generation_prefix(self.tokenizer)
        self.assertTrue(rendered.endswith(prefix))
        self.assertFalse(rendered.endswith("<|eot_id|>"))
        self.assertEqual(llama32_generation_stop_token_ids(self.tokenizer), [128001, 128008, 128009])

    def test_materialized_dataset_matches_native_trainer_tokenization(self) -> None:
        path = ROOT / (
            "data/sql_create_context/"
            "train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl"
        )
        with path.open(encoding="utf-8") as handle:
            row = json.loads(next(handle))
        direct = self.tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            **row["chat_template_kwargs"],
        )["input_ids"]
        rendered = self.tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        self.assertEqual(direct, rendered)

    def test_full_chat_collator_labels_all_non_boundary_tokens(self) -> None:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        rows = []
        path = ROOT / (
            "data/sql_create_context/"
            "train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl"
        )
        with path.open(encoding="utf-8") as handle:
            for _ in range(2):
                row = json.loads(next(handle))
                rows.append(
                    self.tokenizer.apply_chat_template(
                        row["messages"],
                        tokenize=True,
                        return_dict=True,
                        **row["chat_template_kwargs"],
                    )["input_ids"]
                )
        packed = {"input_ids": rows[0] + rows[1], "seq_lengths": [len(rows[0]), len(rows[1])]}
        collator = DataCollatorForLanguageModeling(
            pad_token_id=128009,
            max_length=2048,
            completion_only_loss=False,
            padding_free=True,
        )
        batch = collator([packed])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        self.assertEqual(int(labels[0]), -100)
        self.assertEqual(int(labels[len(rows[0])]), -100)
        trainable = labels != -100
        self.assertTrue((labels[trainable] == input_ids[trainable]).all().item())
        self.assertGreater(int(trainable.sum()), 0)

    def test_zero_and_fewshot_prompt_semantics(self) -> None:
        testcase = json.loads((ROOT / "data/testcases_spider_dev_full.jsonl").read_text(encoding="utf-8").splitlines()[0])
        demo = json.loads(
            (ROOT / "data/fewshot_static/static_fewshot_k1_full_schema_seed42_spider_train_no_dev_overlap.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        schema = self.runner.normalize_schema_text(testcase["schema_prompt"])
        zero = self.runner.build_prompt(
            schema,
            testcase["question"],
            "llama32_3b_instruct",
            self.tokenizer,
            prompt_format="llama32_instruct_native_chat",
            system_instruction="SYSTEM",
        )
        few = self.runner.build_prompt_schema_fewshot(
            schema,
            testcase["question"],
            [demo],
            "llama32_3b_instruct",
            self.tokenizer,
            prompt_format="llama32_instruct_native_chat",
            system_instruction="SYSTEM",
        )
        self.assertNotIn("Example 1", zero)
        self.assertEqual(few.count("Example 1"), 1)
        self.assertIn(testcase["question"], zero)
        self.assertIn(testcase["question"], few)
        self.assertNotIn("<|im_start|>", zero + few)

    def test_prompt_smoke_all_conditions_pass(self) -> None:
        path = ROOT / "audits/derived/llama32_3b_instruct_prompt_smoke_20260714.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["conditions"]), 8)
        for stats in result["conditions"].values():
            self.assertEqual(stats["cases"], 1032)
            self.assertEqual(stats["over_2048"], 0)
            self.assertEqual(stats["invalid_prompts"], 0)

    def test_qwen_prompt_and_retrieval_regression(self) -> None:
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "USER"},
        ]
        expected = (
            "<|im_start|>system\nSYSTEM<|im_end|>\n"
            "<|im_start|>user\nUSER<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(
            self.runner._render_qwen_sqlctx_chatml_messages(messages, add_generation_prompt=True),
            expected,
        )
        self.assertEqual(
            sha256(ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss"),
            "62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc",
        )
        self.assertEqual(
            sha256(ROOT / "results/retrieval_traces/run_base_20260712_143438_retrieval_traces.jsonl"),
            "e79c0297ad3a0a94e00da96af175693e8dfd450b0cfbd9761c2e46d3efafa049",
        )
        self.assertEqual(
            sha256(ROOT / "configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json"),
            "0bfce20d1e97f0b42b61d3db67679e3feef46b94a58a09147e4a5fb82240815e",
        )


if __name__ == "__main__":
    unittest.main()
