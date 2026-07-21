#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from structure_rerank_v2 import (
    METHOD_NAME as STRUCTURE_RERANK_METHOD,
    structure_rerank_adjustment,
)


def ensure_semicolon(sql: str) -> str:
    sql = str(sql).strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql


def normalize_question_for_retrieval(question: str) -> str:
    return " ".join(str(question).strip().lower().split())


def normalize_sql_for_retrieval(sql: str) -> str:
    return " ".join(ensure_semicolon(str(sql)).strip().lower().split())


def question_sql_signature(question: str, sql: str) -> tuple[str, str]:
    return (
        normalize_question_for_retrieval(question),
        normalize_sql_for_retrieval(sql),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolved_path_string(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path.absolute())


def _assistant_message_sql(obj: dict[str, Any]) -> str:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() == "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return ""


def _extract_schema(obj: dict[str, Any]) -> str:
    for key in ("schema_prompt", "context", "db_schema", "schema", "database_schema"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() != "user":
                continue
            content = str(message.get("content", ""))
            marker = "Context:"
            if marker in content:
                after = content.split(marker, 1)[1]
                if "\n\nQuestion:" in after:
                    after = after.split("\n\nQuestion:", 1)[0]
                if after.strip():
                    return after.strip()
    return ""


def extract_retrieval_example(
    obj: dict[str, Any],
    *,
    row_idx: int,
    source_path: Path,
) -> dict[str, Any] | None:
    question = str(obj.get("question", "")).strip()
    gold_sql = str(obj.get("gold_sql") or obj.get("sql") or obj.get("query") or "").strip()
    if not gold_sql:
        gold_sql = _assistant_message_sql(obj)
    schema_prompt = _extract_schema(obj)
    if not question or not gold_sql or not schema_prompt:
        return None

    source_path_str = _resolved_path_string(source_path)
    example_id = str(obj.get("id") or f"RET_{row_idx:06d}").strip()
    row_hash = str(obj.get("row_hash") or "").strip()
    if not row_hash:
        row_hash = "sha256:" + sha256_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True)
        )

    return {
        "id": example_id,
        "question": question,
        "gold_sql": ensure_semicolon(gold_sql),
        "schema_prompt": schema_prompt,
        "db_id": str(obj.get("db_id", "")).strip(),
        "source_path": source_path_str,
        "source_idx": int(obj.get("source_idx", row_idx - 1))
        if str(obj.get("source_idx", "")).strip().isdigit()
        else row_idx - 1,
        "row_hash": row_hash,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_retrieval_pool(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            example = extract_retrieval_example(
                obj,
                row_idx=row_idx,
                source_path=path,
            )
            if example is not None:
                examples.append(example)
    return examples


@dataclass
class LeakageGuard:
    testcases_path: Path | None
    testcase_ids: set[str]
    testcase_questions: set[str]
    testcase_sqls: set[str]
    question_sql_pairs: set[tuple[str, str]]
    resolved_testcase_path: str | None

    @classmethod
    def from_testcases_path(cls, testcases_path: Path | None) -> "LeakageGuard":
        if testcases_path is None or not testcases_path.exists():
            return cls(
                testcases_path=testcases_path,
                testcase_ids=set(),
                testcase_questions=set(),
                testcase_sqls=set(),
                question_sql_pairs=set(),
                resolved_testcase_path=_resolved_path_string(testcases_path)
                if testcases_path is not None
                else None,
            )
        rows = load_jsonl(testcases_path)
        testcase_ids = {
            str(row.get("id", "")).strip()
            for row in rows
            if str(row.get("id", "")).strip()
        }
        testcase_questions = {
            normalize_question_for_retrieval(str(row.get("question", "")))
            for row in rows
            if str(row.get("question", "")).strip()
        }
        testcase_sqls = {
            normalize_sql_for_retrieval(
                str(row.get("gold_sql") or row.get("sql") or row.get("query") or "")
            )
            for row in rows
            if str(row.get("gold_sql") or row.get("sql") or row.get("query") or "").strip()
        }
        question_sql_pairs = {
            question_sql_signature(
                str(row.get("question", "")),
                str(row.get("gold_sql") or row.get("sql") or row.get("query") or ""),
            )
            for row in rows
            if str(row.get("question", "")).strip()
            and str(row.get("gold_sql") or row.get("sql") or row.get("query") or "").strip()
        }
        return cls(
            testcases_path=testcases_path,
            testcase_ids=testcase_ids,
            testcase_questions=testcase_questions,
            testcase_sqls=testcase_sqls,
            question_sql_pairs=question_sql_pairs,
            resolved_testcase_path=_resolved_path_string(testcases_path),
        )

    def leakage_reasons(self, example: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        example_id = str(example.get("id", "")).strip()
        if example_id and example_id in self.testcase_ids:
            reasons.append("testcase_id")
        question_norm = normalize_question_for_retrieval(str(example.get("question", "")))
        if question_norm and question_norm in self.testcase_questions:
            reasons.append("question")
        sql_norm = normalize_sql_for_retrieval(str(example.get("gold_sql", "")))
        if sql_norm and sql_norm in self.testcase_sqls:
            reasons.append("sql")
        pair = question_sql_signature(
            str(example.get("question", "")),
            str(example.get("gold_sql", "")),
        )
        if pair in self.question_sql_pairs:
            reasons.append("question_sql_pair")
        source_path = str(example.get("source_path", "")).strip()
        if (
            source_path
            and self.resolved_testcase_path
            and source_path == self.resolved_testcase_path
        ):
            reasons.append("source_path_is_testcases")
        return reasons


@dataclass
class FewShotSelection:
    examples: list[dict[str, Any]]
    scores: list[float]
    filtered_count: int
    filtered_reasons: dict[str, int]
    retrieval_method: str
    retrieval_index_path: str
    retrieval_pool_path: str
    retrieval_success: bool

    def ids(self) -> list[str]:
        return [str(example.get("id", "")) for example in self.examples]

    def db_ids(self) -> list[str]:
        return [str(example.get("db_id", "")) for example in self.examples]


def _filter_reason(
    example: dict[str, Any],
    *,
    target_id: str | None,
    target_question: str | None,
    target_db_id: str | None,
    same_db_only: bool,
    allow_overlap: bool,
    leakage_guard: LeakageGuard,
) -> str | None:
    example_id = str(example.get("id", "")).strip()
    if target_id and example_id and example_id == str(target_id):
        return "same_id_as_target"

    if target_question:
        example_question = normalize_question_for_retrieval(str(example.get("question", "")))
        target_question_norm = normalize_question_for_retrieval(target_question)
        if example_question and example_question == target_question_norm:
            return "same_question_as_target"

    if same_db_only:
        query_db_id = str(target_db_id or "").strip()
        example_db_id = str(example.get("db_id", "")).strip()
        if query_db_id and example_db_id != query_db_id:
            return "db_mismatch"

    if not allow_overlap:
        reasons = leakage_guard.leakage_reasons(example)
        if reasons:
            return "leakage_" + "+".join(reasons)

    return None


SQLAWARE_RERANK_METHOD = "sqlaware_topk"
SQLAWARE_STRUCTURE_BONUS_MAX = 0.08


def _question_sqlaware_hints(question: str) -> set[str]:
    q = normalize_question_for_retrieval(question)
    hints: set[str] = set()
    if re.search(r"\b(how many|number of|count|counts|total number)\b", q):
        hints.add("count")
    if re.search(r"\b(average|avg|mean)\b", q):
        hints.add("avg")
    if re.search(r"\b(sum|summed)\b", q) or (
        re.search(r"\btotal\b", q) and not re.search(r"\btotal number\b", q)
    ):
        hints.add("sum")
    if re.search(r"\b(maximum|max|highest|largest|most)\b", q):
        hints.add("max")
        hints.add("order_extreme")
    if re.search(r"\b(minimum|min|lowest|smallest|least|fewest)\b", q):
        hints.add("min")
        hints.add("order_extreme")
    if re.search(r"\b(for each|per|each)\b", q):
        hints.add("group_by")
    if re.search(r"\b(distinct|different|unique)\b", q):
        hints.add("distinct")
    if re.search(r"\b(not|without|except)\b", q):
        hints.add("negation")
    return hints


def _strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", str(sql))


def _candidate_sql_features(sql: str) -> tuple[set[str], int]:
    low = _strip_sql_literals(sql).lower()
    features: set[str] = set()
    for func in ("count", "sum", "avg", "min", "max"):
        if re.search(rf"\b{func}\s*\(", low):
            features.add(func)
    if re.search(r"\bgroup\s+by\b", low):
        features.add("group_by")
    if re.search(r"\bhaving\b", low):
        features.add("having")
    if re.search(r"\border\s+by\b", low):
        features.add("order_by")
    if re.search(r"\blimit\b", low):
        features.add("limit")
    if "order_by" in features and "limit" in features:
        features.add("order_by_limit")
    if re.search(r"\bdistinct\b", low):
        features.add("distinct")
    join_count = len(re.findall(r"\bjoin\b", low))
    if join_count >= 1:
        features.add("join")
    if join_count >= 2:
        features.add("multi_join")
    if len(re.findall(r"\bselect\b", low)) >= 2:
        features.add("nested_select")
    if re.search(r"\bexists\s*\(", low):
        features.add("exists")
    if re.search(r"\bnot\s+in\s*\(", low):
        features.add("not_in")
    if re.search(r"\bin\s*\(", low):
        features.add("in")
    for set_op in ("union", "intersect", "except"):
        if re.search(rf"\b{set_op}\b", low):
            features.add(set_op)
    return features, join_count


def sqlaware_structure_bonus(
    question: str,
    candidate_sql: str,
    *,
    max_bonus: float = SQLAWARE_STRUCTURE_BONUS_MAX,
) -> tuple[float, dict[str, Any]]:
    hints = _question_sqlaware_hints(question)
    features, join_count = _candidate_sql_features(candidate_sql)
    matches: list[dict[str, Any]] = []
    raw_bonus = 0.0

    def add(label: str, condition: bool, weight: float) -> None:
        nonlocal raw_bonus
        if condition:
            raw_bonus += weight
            matches.append({"feature": label, "weight": weight})

    add("COUNT", "count" in hints and "count" in features, 0.018)
    add("SUM", "sum" in hints and "sum" in features, 0.018)
    add("AVG", "avg" in hints and "avg" in features, 0.018)
    add("MIN", "min" in hints and "min" in features, 0.018)
    add("MAX", "max" in hints and "max" in features, 0.018)
    add("GROUP BY", "group_by" in hints and "group_by" in features, 0.016)
    add("DISTINCT", "distinct" in hints and "distinct" in features, 0.012)
    add(
        "ORDER BY + LIMIT",
        "order_extreme" in hints and "order_by_limit" in features,
        0.012,
    )
    add(
        "NEGATION",
        "negation" in hints
        and bool({"not_in", "except", "exists"} & features),
        0.012,
    )

    bonus_cap = max(0.0, min(float(max_bonus), SQLAWARE_STRUCTURE_BONUS_MAX))
    bonus = min(raw_bonus, bonus_cap)
    return bonus, {
        "question_hints": sorted(hints),
        "candidate_sql_features": sorted(features),
        "candidate_join_count": join_count,
        "matches": matches,
        "raw_bonus": raw_bonus,
        "max_bonus": bonus_cap,
    }


class StaticFewShotRetriever:
    def __init__(
        self,
        *,
        examples: list[dict[str, Any]],
        k: int,
        seed: int,
        allow_overlap: bool,
        same_db_only: bool,
        leakage_guard: LeakageGuard,
        retrieval_pool_path: Path,
    ) -> None:
        self.examples = list(examples)
        self.k = max(0, int(k))
        self.seed = int(seed)
        self.allow_overlap = bool(allow_overlap)
        self.same_db_only = bool(same_db_only)
        self.leakage_guard = leakage_guard
        self.retrieval_pool_path = str(retrieval_pool_path)
        self._order = list(range(len(self.examples)))
        random.Random(self.seed).shuffle(self._order)

    def select(
        self,
        *,
        question: str,
        qid: str,
        db_id: str,
    ) -> FewShotSelection:
        selected: list[dict[str, Any]] = []
        filtered_reasons: Counter[str] = Counter()
        for idx in self._order:
            example = self.examples[idx]
            reason = _filter_reason(
                example,
                target_id=qid,
                target_question=question,
                target_db_id=db_id,
                same_db_only=self.same_db_only,
                allow_overlap=self.allow_overlap,
                leakage_guard=self.leakage_guard,
            )
            if reason is not None:
                filtered_reasons[reason] += 1
                continue
            selected.append(example)
            if len(selected) >= self.k:
                break

        return FewShotSelection(
            examples=selected,
            scores=[],
            filtered_count=sum(filtered_reasons.values()),
            filtered_reasons=dict(filtered_reasons),
            retrieval_method="static_seeded",
            retrieval_index_path="",
            retrieval_pool_path=self.retrieval_pool_path,
            retrieval_success=len(selected) >= self.k,
        )


class FaissFewShotRetriever:
    def __init__(
        self,
        *,
        index_dir: Path,
        embedding_model: str,
        k: int,
        allow_overlap: bool,
        same_db_only: bool,
        leakage_guard: LeakageGuard,
        retrieval_pool_path: Path | None = None,
        rerank_method: str = "none",
        rerank_top_n: int = 5,
        structure_bonus_max: float = SQLAWARE_STRUCTURE_BONUS_MAX,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.k = max(0, int(k))
        self.allow_overlap = bool(allow_overlap)
        self.same_db_only = bool(same_db_only)
        self.leakage_guard = leakage_guard
        self.retrieval_pool_path = str(retrieval_pool_path or "")
        self.rerank_method = str(rerank_method or "none").strip().lower()
        if self.rerank_method not in {"none", SQLAWARE_RERANK_METHOD, STRUCTURE_RERANK_METHOD}:
            raise ValueError(
                f"Unsupported rerank_method={rerank_method!r}; "
                f"expected 'none', '{SQLAWARE_RERANK_METHOD}', or '{STRUCTURE_RERANK_METHOD}'"
            )
        self.rerank_top_n = max(1, int(rerank_top_n))
        self.structure_bonus_max = float(structure_bonus_max)
        if self.structure_bonus_max < 0 or self.structure_bonus_max > SQLAWARE_STRUCTURE_BONUS_MAX:
            raise ValueError(
                "structure_bonus_max must be between 0 and "
                f"{SQLAWARE_STRUCTURE_BONUS_MAX}"
            )

        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "faiss is required for dynamic_fewshot. Run "
                "src/check_embedding_retrieval_env.py and install faiss-cpu/faiss-gpu if needed."
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "sentence-transformers is required for dynamic_fewshot. Run "
                "src/check_embedding_retrieval_env.py before starting an eval run."
            ) from exc

        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.jsonl"
        manifest_path = index_dir / "manifest.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing retrieval metadata: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.examples = load_jsonl(metadata_path)
        if self.index.ntotal != len(self.examples):
            raise ValueError(
                f"FAISS index/metadata size mismatch: index.ntotal={self.index.ntotal}, "
                f"metadata_rows={len(self.examples)}"
            )
        self.manifest: dict[str, Any] = {}
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.retrieval_pool_path = str(
                self.manifest.get("source_path") or self.retrieval_pool_path
            )
        self.normalize = bool(self.manifest.get("normalize", True))
        self.query_prefix = str(self.manifest.get("query_prefix", ""))
        self.apply_query_prefix_to_queries = bool(
            self.manifest.get("apply_query_prefix_to_queries", bool(self.query_prefix))
        )
        try:
            self.embedder = SentenceTransformer(embedding_model)
        except Exception:
            self.embedder = SentenceTransformer(embedding_model, local_files_only=True)

    def _query_embedding_text(self, question: str) -> str:
        if (
            self.apply_query_prefix_to_queries
            and self.query_prefix
            and not str(question).startswith(self.query_prefix)
        ):
            return self.query_prefix + str(question)
        return str(question)

    def _search(self, question: str, search_k: int) -> tuple[list[int], list[float]]:
        import numpy as np

        query_emb = self.embedder.encode(
            [self._query_embedding_text(question)],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, indices = self.index.search(query_emb, search_k)
        return [int(i) for i in indices[0]], [float(s) for s in scores[0]]

    def preview_sqlaware_rerank(
        self,
        *,
        question: str,
        qid: str,
        db_id: str,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        search_k = min(int(self.index.ntotal), max(1, int(top_n or self.rerank_top_n)))
        indices, scores = self._search(question, search_k)
        filtered_reasons: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []

        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            example = self.examples[idx]
            reason = _filter_reason(
                example,
                target_id=qid,
                target_question=question,
                target_db_id=db_id,
                same_db_only=self.same_db_only,
                allow_overlap=self.allow_overlap,
                leakage_guard=self.leakage_guard,
            )
            if reason is not None:
                filtered_reasons[reason] += 1
                bonus = 0.0
                bonus_details: dict[str, Any] = {}
            else:
                bonus, bonus_details = sqlaware_structure_bonus(
                    question,
                    str(example.get("gold_sql", "")),
                    max_bonus=self.structure_bonus_max,
                )
            candidates.append(
                {
                    "rank": rank,
                    "index_position": idx,
                    "id": str(example.get("id", "")),
                    "db_id": str(example.get("db_id", "")),
                    "question": str(example.get("question", "")),
                    "gold_sql": str(example.get("gold_sql", "")),
                    "example": example,
                    "bge_similarity": score,
                    "structure_bonus": bonus,
                    "final_score": score + bonus,
                    "bonus_details": bonus_details,
                    "filtered_reason": reason,
                    "selected": False,
                    "selection_rank": None,
                }
            )

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate["filtered_reason"] is None
        ]
        ranked_candidates = sorted(
            valid_candidates,
            key=lambda candidate: (
                -float(candidate["final_score"]),
                -float(candidate["bge_similarity"]),
                int(candidate["rank"]),
                str(candidate["id"]),
            ),
        )
        selected_candidates = self._take_distinct_candidates(ranked_candidates)
        selected_positions = {
            int(candidate["index_position"]): rank
            for rank, candidate in enumerate(selected_candidates, start=1)
        }
        for candidate in candidates:
            selection_rank = selected_positions.get(int(candidate["index_position"]))
            if selection_rank is not None:
                candidate["selected"] = True
                candidate["selection_rank"] = selection_rank

        return {
            "candidates": candidates,
            "selected_candidates": selected_candidates,
            "filtered_reasons": dict(filtered_reasons),
        }

    def preview_structure_rerank(
        self,
        *,
        question: str,
        qid: str,
        db_id: str,
        target_schema: str,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        search_k = min(int(self.index.ntotal), max(1, int(top_n or self.rerank_top_n)))
        indices, scores = self._search(question, search_k)
        filtered_reasons: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []

        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            example = self.examples[idx]
            reason = _filter_reason(
                example,
                target_id=qid,
                target_question=question,
                target_db_id=db_id,
                same_db_only=self.same_db_only,
                allow_overlap=self.allow_overlap,
                leakage_guard=self.leakage_guard,
            )
            if reason is None:
                adjustment, details = structure_rerank_adjustment(
                    question=question,
                    target_schema=target_schema,
                    candidate_sql=str(example.get("gold_sql", "")),
                    candidate_schema=str(example.get("schema_prompt", "")),
                    max_adjustment=self.structure_bonus_max,
                )
            else:
                filtered_reasons[reason] += 1
                adjustment, details = 0.0, {}
            candidates.append(
                {
                    "rank": rank,
                    "index_position": idx,
                    "id": str(example.get("id", "")),
                    "db_id": str(example.get("db_id", "")),
                    "question": str(example.get("question", "")),
                    "gold_sql": str(example.get("gold_sql", "")),
                    "example": example,
                    "bge_similarity": score,
                    "structure_adjustment": adjustment,
                    "final_score": score + adjustment,
                    "structure_details": details,
                    "filtered_reason": reason,
                    "selected": False,
                    "selection_rank": None,
                }
            )

        valid_candidates = [item for item in candidates if item["filtered_reason"] is None]
        ranked_candidates = sorted(
            valid_candidates,
            key=lambda item: (
                -float(item["final_score"]),
                -float(item["bge_similarity"]),
                int(item["rank"]),
                str(item["id"]),
            ),
        )
        selected_candidates = self._take_distinct_candidates(ranked_candidates)
        selected_positions = {
            int(item["index_position"]): rank
            for rank, item in enumerate(selected_candidates, start=1)
        }
        for candidate in candidates:
            selection_rank = selected_positions.get(int(candidate["index_position"]))
            if selection_rank is not None:
                candidate["selected"] = True
                candidate["selection_rank"] = selection_rank

        return {
            "candidates": candidates,
            "selected_candidates": selected_candidates,
            "filtered_reasons": dict(filtered_reasons),
        }

    def _take_distinct_candidates(
        self,
        ranked_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in ranked_candidates:
            example_id = str(candidate.get("id", "")).strip()
            if example_id and example_id in seen_ids:
                continue
            if example_id:
                seen_ids.add(example_id)
            selected.append(candidate)
            if len(selected) >= self.k:
                break
        return selected

    def select(
        self,
        *,
        question: str,
        qid: str,
        db_id: str,
        target_schema: str = "",
    ) -> FewShotSelection:
        total = int(self.index.ntotal)
        if self.k == 0 or total == 0:
            return FewShotSelection(
                examples=[],
                scores=[],
                filtered_count=0,
                filtered_reasons={},
                retrieval_method="sentence_transformer_faiss",
                retrieval_index_path=str(self.index_dir),
                retrieval_pool_path=self.retrieval_pool_path,
                retrieval_success=True,
            )

        if self.rerank_method == SQLAWARE_RERANK_METHOD:
            preview = self.preview_sqlaware_rerank(
                question=question,
                qid=qid,
                db_id=db_id,
                top_n=self.rerank_top_n,
            )
            selected_candidates = preview["selected_candidates"]
            selected = [candidate["example"] for candidate in selected_candidates]
            selected_scores = [
                float(candidate["bge_similarity"]) for candidate in selected_candidates
            ]
            return FewShotSelection(
                examples=selected,
                scores=selected_scores,
                filtered_count=sum(preview["filtered_reasons"].values()),
                filtered_reasons=preview["filtered_reasons"],
                retrieval_method=f"sentence_transformer_faiss_sqlaware_top{self.rerank_top_n}",
                retrieval_index_path=str(self.index_dir),
                retrieval_pool_path=self.retrieval_pool_path,
                retrieval_success=len(selected) >= self.k,
            )

        if self.rerank_method == STRUCTURE_RERANK_METHOD:
            preview = self.preview_structure_rerank(
                question=question,
                qid=qid,
                db_id=db_id,
                target_schema=target_schema,
                top_n=self.rerank_top_n,
            )
            selected_candidates = preview["selected_candidates"]
            selected = [candidate["example"] for candidate in selected_candidates]
            selected_scores = [
                float(candidate["bge_similarity"]) for candidate in selected_candidates
            ]
            return FewShotSelection(
                examples=selected,
                scores=selected_scores,
                filtered_count=sum(preview["filtered_reasons"].values()),
                filtered_reasons=preview["filtered_reasons"],
                retrieval_method=f"sentence_transformer_faiss_{STRUCTURE_RERANK_METHOD}_top{self.rerank_top_n}",
                retrieval_index_path=str(self.index_dir),
                retrieval_pool_path=self.retrieval_pool_path,
                retrieval_success=len(selected) >= self.k,
            )

        selected: list[dict[str, Any]] = []
        selected_scores: list[float] = []
        selected_ids: set[str] = set()
        filtered_reasons: Counter[str] = Counter()
        search_k = total if self.same_db_only else min(total, max(self.k * 20, self.k + 50))

        while True:
            indices, scores = self._search(question, search_k)
            selected.clear()
            selected_scores.clear()
            selected_ids.clear()
            filtered_reasons.clear()
            for idx, score in zip(indices, scores):
                if idx < 0:
                    continue
                example = self.examples[idx]
                reason = _filter_reason(
                    example,
                    target_id=qid,
                    target_question=question,
                    target_db_id=db_id,
                    same_db_only=self.same_db_only,
                    allow_overlap=self.allow_overlap,
                    leakage_guard=self.leakage_guard,
                )
                if reason is not None:
                    filtered_reasons[reason] += 1
                    continue
                example_id = str(example.get("id", "")).strip()
                if example_id and example_id in selected_ids:
                    filtered_reasons["duplicate_example_id"] += 1
                    continue
                selected.append(example)
                selected_scores.append(score)
                if example_id:
                    selected_ids.add(example_id)
                if len(selected) >= self.k:
                    break

            if len(selected) >= self.k or search_k >= total:
                break
            search_k = min(total, max(search_k * 2, search_k + self.k + 50))

        return FewShotSelection(
            examples=selected,
            scores=selected_scores,
            filtered_count=sum(filtered_reasons.values()),
            filtered_reasons=dict(filtered_reasons),
            retrieval_method="sentence_transformer_faiss",
            retrieval_index_path=str(self.index_dir),
            retrieval_pool_path=self.retrieval_pool_path,
            retrieval_success=len(selected) >= self.k,
        )


def selection_to_trace(
    *,
    qid: str,
    db_id: str,
    question: str,
    selection: FewShotSelection,
    prompt_char_length: int,
) -> dict[str, Any]:
    leakage_status = "pass"
    if selection.filtered_reasons:
        leak_keys = [key for key in selection.filtered_reasons if key.startswith("leakage_")]
        if leak_keys:
            leakage_status = "filtered_leakage"
    return {
        "id": qid,
        "db_id": db_id,
        "question": question,
        "retrieval_method": selection.retrieval_method,
        "retrieval_index_path": selection.retrieval_index_path,
        "retrieval_pool_path": selection.retrieval_pool_path,
        "retrieved_ids": selection.ids(),
        "retrieved_scores": selection.scores,
        "retrieved_db_ids": selection.db_ids(),
        "num_fewshot_examples": len(selection.examples),
        "filtered_count": selection.filtered_count,
        "filtered_reasons": selection.filtered_reasons,
        "retrieval_success": selection.retrieval_success,
        "prompt_char_length": prompt_char_length,
        "leakage_status": leakage_status,
    }


def selection_to_csv_values(selection: FewShotSelection) -> dict[str, str | int]:
    return {
        "retrieved_ids": json.dumps(selection.ids(), ensure_ascii=False),
        "retrieved_scores": json.dumps(
            [round(score, 6) for score in selection.scores],
            ensure_ascii=False,
        ),
        "retrieved_db_ids": json.dumps(selection.db_ids(), ensure_ascii=False),
        "retrieval_method": selection.retrieval_method,
        "retrieval_index_path": selection.retrieval_index_path,
        "num_fewshot_examples": len(selection.examples),
    }
