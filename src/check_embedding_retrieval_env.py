#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _add(results: list[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail))


def _load_sentence_transformer_offline(model_name: str) -> tuple[Any | None, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        return None, f"import failed: {repr(exc)}"

    try:
        return SentenceTransformer(model_name, local_files_only=True), "loaded from local cache"
    except TypeError:
        try:
            return SentenceTransformer(model_name), "loaded with offline environment"
        except Exception as exc:
            return None, f"offline load failed: {repr(exc)}"
    except Exception as exc:
        return None, f"offline load failed: {repr(exc)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight check for SentenceTransformer + FAISS retrieval without model downloads."
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model id or local path to test from cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[CheckResult] = []

    _add(results, "python_executable", True, sys.executable)

    torch_ok = _module_available("torch")
    torch_detail = _version("torch")
    if torch_ok:
        try:
            import torch  # type: ignore

            torch_detail += f", cuda_available={torch.cuda.is_available()}"
        except Exception as exc:
            torch_ok = False
            torch_detail += f", import failed: {repr(exc)}"
    _add(results, "torch", torch_ok, torch_detail)

    numpy_ok = _module_available("numpy")
    _add(results, "numpy", numpy_ok, _version("numpy"))

    sklearn_ok = _module_available("sklearn")
    _add(results, "sklearn", sklearn_ok, _version("scikit-learn"))

    st_ok = _module_available("sentence_transformers")
    _add(results, "sentence_transformers", st_ok, _version("sentence-transformers"))

    faiss_ok = _module_available("faiss")
    faiss_detail = _version("faiss-cpu")
    if faiss_detail == "not installed":
        faiss_detail = f"faiss-cpu={faiss_detail}, faiss-gpu={_version('faiss-gpu')}"
    _add(results, "faiss", faiss_ok, faiss_detail)

    model = None
    if st_ok:
        model, detail = _load_sentence_transformer_offline(str(args.embedding_model))
        _add(results, "embedding_model_cache", model is not None, detail)
    else:
        _add(results, "embedding_model_cache", False, "skipped because sentence_transformers is missing")

    embedding_ok = False
    embedding_dim = "n/a"
    if model is not None:
        try:
            embedding = model.encode(
                ["How many singers are there?"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding_dim = str(int(embedding.shape[1]))
            embedding_ok = True
        except Exception as exc:
            embedding_dim = f"test embedding failed: {repr(exc)}"
    _add(results, "test_embedding", embedding_ok, embedding_dim)

    faiss_mini_ok = False
    faiss_mini_detail = "skipped"
    if faiss_ok and numpy_ok:
        try:
            import faiss  # type: ignore
            import numpy as np

            vectors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            index = faiss.IndexFlatIP(3)
            index.add(vectors)
            scores, indices = index.search(vectors[:1], 3)
            faiss_mini_ok = indices.shape == (1, 3) and scores.shape == (1, 3)
            faiss_mini_detail = f"indices={indices[0].tolist()}, scores={scores[0].tolist()}"
        except Exception as exc:
            faiss_mini_detail = f"mini-index failed: {repr(exc)}"
    elif not faiss_ok:
        faiss_mini_detail = "skipped because faiss is missing"
    elif not numpy_ok:
        faiss_mini_detail = "skipped because numpy is missing"
    _add(results, "faiss_mini_index", faiss_mini_ok, faiss_mini_detail)

    ready = all(
        result.ok
        for result in results
        if result.name
        in {
            "torch",
            "numpy",
            "sklearn",
            "sentence_transformers",
            "faiss",
            "embedding_model_cache",
            "test_embedding",
            "faiss_mini_index",
        }
    )

    print("Embedding retrieval preflight")
    print(f"Embedding model: {args.embedding_model}")
    for result in results:
        status = "OK" if result.ok else "MISSING/FAIL"
        print(f"- {result.name}: {status} | {result.detail}")
    print(f"STATUS: {'READY' if ready else 'NOT READY'}")
    if not ready:
        print("Installationsvorschlag, erst nach Rückfrage ausführen:")
        print("  python3 -m pip install sentence-transformers faiss-cpu numpy scikit-learn torch")
        print("Hinweis: Das Embedding-Modell muss lokal im Hugging-Face-Cache liegen oder bewusst heruntergeladen werden.")


if __name__ == "__main__":
    main()
