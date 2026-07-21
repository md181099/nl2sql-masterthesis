#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


logger = logging.getLogger(__name__)


class LLMClient:
    """
    Simple registry + loader for multiple open-source LLMs (Hugging Face models).

    Goal:
      - One place to declare available LLMs.
      - Batch scripts can select an LLM via CLI (e.g., --llm llama32_1b)
      - Optional PEFT adapters (LoRA/Prefix/...) can be loaded per LLM.

    Adapter convention:
      - adapter='base' loads the raw HF model.
      - otherwise loads PEFT adapter from:
          <project_root>/adapters/<llm_name>/<adapter>/
        Example:
          adapters/llama32_1b/lora_sql
          adapters/llama32_1b/prefix_sql
    """

    # Add/remove models here. Keys are the CLI names.
    REGISTRY: Dict[str, str] = {
        "llama32_1b": "meta-llama/Llama-3.2-1B",
        "llama32_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "tinyllama_11b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen25_15b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen35_2b_base": "Qwen/Qwen3.5-2B-Base",
        "qwen35_9b_base": "Qwen/Qwen3.5-9B-Base",
        # Optional examples (uncomment if needed):
        # "tinyllama_11b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "qwen25_05b": "Qwen/Qwen2.5-0.5B",
    }

    MODEL_REVISIONS: Dict[str, str] = {
        "llama32_3b_instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root

    @classmethod
    def list_llms(cls) -> List[str]:
        return sorted(cls.REGISTRY.keys())

    @classmethod
    def list_registered_llms(cls) -> List[str]:
        """Alias for callers that expect an explicit registry-oriented method name."""
        return cls.list_llms()

    @classmethod
    def resolve_model_id(cls, llm_name: str) -> str:
        if llm_name not in cls.REGISTRY:
            logger.error("Unknown llm '%s'. Available: %s", llm_name, ", ".join(cls.list_llms()))
            raise ValueError(
                f"Unknown llm '{llm_name}'. Available: {', '.join(cls.list_llms())}"
            )
        return cls.REGISTRY[llm_name]

    def get_model_id(self, llm_name: str) -> str:
        """
        Convenience/compatibility wrapper for scripts that expect get_model_id().
        Always returns the Hugging Face repo id string (e.g. 'meta-llama/Llama-3.2-1B').
        """
        return self.resolve_model_id(llm_name)

    @classmethod
    def resolve_model_revision(cls, llm_name: str) -> str | None:
        cls.resolve_model_id(llm_name)
        return cls.MODEL_REVISIONS.get(llm_name)

    def get_tokenizer(self, llm_name: str) -> AutoTokenizer:
        model_id = self.resolve_model_id(llm_name)
        revision = self.resolve_model_revision(llm_name)
        logger.info("Loading tokenizer for llm=%s (%s), revision=%s", llm_name, model_id, revision or "default")
        load_kwargs = {"revision": revision} if revision else {}
        tok = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        # Some Llama tokenizers don't define pad_token; make padding stable.
        if tok.pad_token is None and tok.eos_token is not None:
            logger.debug("Tokenizer has no pad token. Using eos token as pad token.")
            tok.pad_token = tok.eos_token
        return tok

    def _dtype(self):
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def get_base_model(
        self,
        llm_name: str,
        *,
        attn_implementation: str | None = None,
    ) -> AutoModelForCausalLM:
        """Load the base (non-adapted) model for training scripts."""
        model_id = self.resolve_model_id(llm_name)
        revision = self.resolve_model_revision(llm_name)
        dtype = self._dtype()
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": dtype,
        }
        if revision:
            load_kwargs["revision"] = revision
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        logger.info(
            "Loading base model for llm=%s (%s), revision=%s, torch_dtype=%s, attn_implementation=%s",
            llm_name,
            model_id,
            revision or "default",
            dtype,
            attn_implementation or "default",
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        return model

    def get_model(self, llm_name: str, adapter: str = "base") -> AutoModelForCausalLM:
        """
        Load model with optional PEFT adapter.
        """
        base_model = self.get_base_model(llm_name)

        if adapter.lower() == "base":
            logger.info("Using base model without adapter (llm=%s)", llm_name)
            base_model.eval()
            return base_model

        adapter_dir = self.project_root / "adapters" / llm_name / adapter
        if not adapter_dir.exists():
            logger.error("Adapter directory not found: %s", adapter_dir)
            raise FileNotFoundError(
                f"Adapter directory not found: {adapter_dir}\n"
                "Expected layout: adapters/<llm_name>/<adapter> (e.g. adapters/llama32_1b/lora_sql)"
            )

        logger.info("Loading adapter '%s' from %s", adapter, adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()
        return model
