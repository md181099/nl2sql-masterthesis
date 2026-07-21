#!/usr/bin/env python3
from __future__ import annotations

"""Native Llama 3.2 Instruct chat serialization helpers.

The official tokenizer template includes a date preamble.  A fixed date keeps
training and evaluation prompts byte-reproducible while still using the native
template implementation rather than a hand-written serialization.
"""

from typing import Any


LLAMA32_NATIVE_CHAT_FORMAT = "llama32_instruct_native_chat"
LLAMA32_3B_INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA32_3B_INSTRUCT_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
LLAMA32_NATIVE_TEMPLATE_DATE = "26 Jul 2024"


def render_llama32_native_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported Llama chat role: {role!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"Llama chat message content must be a string: {role!r}")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        date_string=LLAMA32_NATIVE_TEMPLATE_DATE,
    )


def llama32_native_template_kwargs() -> dict[str, str]:
    return {"date_string": LLAMA32_NATIVE_TEMPLATE_DATE}


def llama32_assistant_generation_prefix(tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
    ]
    without_generation = render_llama32_native_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    with_generation = render_llama32_native_chat(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    if not with_generation.startswith(without_generation):
        raise RuntimeError("Native Llama chat template has a non-prefix generation form")
    prefix = with_generation[len(without_generation) :]
    if not prefix:
        raise RuntimeError("Native Llama chat template produced an empty generation prefix")
    return prefix


def llama32_generation_stop_token_ids(tokenizer: Any) -> list[int]:
    token_names = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    stop_ids: list[int] = []
    for token_name in token_names:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id is None or token_id == unknown_id:
            continue
        token_id = int(token_id)
        if token_id not in stop_ids:
            stop_ids.append(token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and int(eos_token_id) not in stop_ids:
        stop_ids.append(int(eos_token_id))
    if not stop_ids:
        raise RuntimeError("No native Llama generation stop token IDs were resolved")
    return stop_ids


def configure_llama32_padding(tokenizer: Any) -> int:
    """Use the existing EOT/EOS token for padding without resizing embeddings."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Llama tokenizer has neither PAD nor EOS/EOT token")
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def tokenize_rendered_llama32_chat(tokenizer: Any, text: str, **kwargs: Any) -> Any:
    """Tokenize a rendered native prompt without adding a second BOS token."""
    return tokenizer(text, add_special_tokens=False, **kwargs)
