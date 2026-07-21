from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_runner():
    path = SRC / "06_batch_run_dynamic_k3_v1.py"
    spec = importlib.util.spec_from_file_location("dynamic_k3_runner_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def selection(scores: list[float]):
    examples = [
        {"id": f"D{idx}", "question": f"q{idx}", "schema_prompt": "Table: t\nColumns: x", "gold_sql": "SELECT x FROM t;"}
        for idx in range(1, 4)
    ]
    return RUNNER.FewShotSelection(
        examples=examples,
        scores=scores,
        filtered_count=0,
        filtered_reasons={},
        retrieval_method="test",
        retrieval_index_path="index",
        retrieval_pool_path="pool",
        retrieval_success=True,
    )


class DynamicK3GateTests(unittest.TestCase):
    def test_set_gate_uses_minimum_and_is_binary(self) -> None:
        selected = selection([0.91, 0.82, 0.74])
        accepted = RUNNER.evaluate_fewshot_gate(
            enabled=True,
            mode="set_min_similarity",
            threshold=0.70,
            features=[],
            selection=selected,
            question="q",
            debug_enabled=True,
        )
        rejected = RUNNER.evaluate_fewshot_gate(
            enabled=True,
            mode="set_min_similarity",
            threshold=0.85,
            features=[],
            selection=selected,
            question="q",
            debug_enabled=True,
        )
        self.assertEqual(accepted.decision, "fewshot")
        self.assertEqual(rejected.decision, "zero_shot")
        self.assertEqual(accepted.score, 0.74)
        self.assertEqual(accepted.number_of_retrieved_candidates, 3)
        self.assertEqual(
            accepted.score_semantics,
            "minimum_original_bge_similarity_of_selected_set",
        )

    def test_set_gate_rejects_missing_or_nonfinite_scores(self) -> None:
        with self.assertRaises(RuntimeError):
            RUNNER.evaluate_fewshot_gate(
                enabled=True,
                mode="set_min_similarity",
                threshold=0.70,
                features=[],
                selection=selection([0.9, 0.8]),
                question="q",
                debug_enabled=True,
            )
        with self.assertRaises(RuntimeError):
            RUNNER.evaluate_fewshot_gate(
                enabled=True,
                mode="set_min_similarity",
                threshold=0.70,
                features=[],
                selection=selection([0.9, 0.8, math.nan]),
                question="q",
                debug_enabled=True,
            )

    def test_prompt_renders_three_full_schema_demos_in_order(self) -> None:
        prompt = RUNNER.build_prompt_schema_fewshot(
            "Table: target\nColumns: id",
            "target question",
            selection([0.9, 0.8, 0.7]).examples,
            "qwen35_2b_base",
            tokenizer=None,
            prompt_format="qwen_sqlctx_chatml",
            system_instruction="system",
            example_schema_mode="full",
            example_mode="schema_with_rules",
        )
        positions = [prompt.index(f"Example {idx}") for idx in range(1, 4)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(prompt.count("Database schema:"), 4)
        self.assertTrue(prompt.endswith(RUNNER.V2_SQLCTX_ASSISTANT_PREFIX))


class DynamicK3RetrieverTests(unittest.TestCase):
    def test_distinct_candidate_selection_is_stable(self) -> None:
        retriever = object.__new__(RUNNER.FaissFewShotRetriever)
        retriever.k = 3
        ranked = [
            {"id": "A"},
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
        ]
        selected = retriever._take_distinct_candidates(ranked)
        self.assertEqual([item["id"] for item in selected], ["A", "B", "C"])


class DynamicK3ConfigTests(unittest.TestCase):
    def test_config_matrix_has_36_valid_rows(self) -> None:
        path = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({row["new_k3_config"] for row in rows}), 36)
        for row in rows:
            config = json.loads((ROOT / row["new_k3_config"]).read_text(encoding="utf-8"))
            self.assertEqual(config["k"], 3)
            self.assertEqual(config["max_input_tokens"], 4352)
            self.assertEqual(config["max_new_tokens"], 256)
            self.assertEqual(config["results_dir"], "results/k3_extension_20260717")
            self.assertIn("k3", config["run_output_prefix"])
            self.assertIn("maxin4352", config["run_output_prefix"])
            self.assertIn("maxin4352", row["new_k3_config"])
            if "gate" in row["condition"]:
                self.assertEqual(config["fewshot_gate_mode"], "set_min_similarity")
                self.assertTrue(config["fewshot_gate_enabled"])


if __name__ == "__main__":
    unittest.main()
