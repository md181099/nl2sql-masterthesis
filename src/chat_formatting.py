#!/usr/bin/env python3
from __future__ import annotations


def build_nl2sql_messages(
    system_instruction: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]


def messages_to_plain_text(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    contents = [str(m.get("content", "")) for m in messages if str(m.get("content", ""))]
    if not contents:
        return ""

    user_messages = [str(m.get("content", "")) for m in messages if str(m.get("role", "")) == "user"]
    assistant_messages = [str(m.get("content", "")) for m in messages if str(m.get("role", "")) == "assistant"]

    if add_generation_prompt:
        if user_messages:
            return user_messages[-1]
        return "\n\n".join(contents)

    if user_messages:
        base = user_messages[-1]
        if assistant_messages:
            return base + "\n" + assistant_messages[-1]
        return base

    return "\n\n".join(contents)


def _chat_fallback_text(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    role_to_content = {str(m.get("role", "")): str(m.get("content", "")) for m in messages}
    system_content = role_to_content.get("system", "")
    user_content = role_to_content.get("user", "")
    assistant_content = role_to_content.get("assistant", "")

    if add_generation_prompt:
        if system_content and user_content:
            return system_content + "\n\n" + user_content
    else:
        if system_content and user_content and assistant_content:
            return system_content + "\n\n" + user_content + "\n" + assistant_content

    contents = [str(m.get("content", "")) for m in messages if str(m.get("content", ""))]
    return "\n\n".join(contents)


def render_messages(
    tokenizer,
    messages: list[dict[str, str]],
    prompt_format: str,
    chat_template: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    fmt_raw = str(prompt_format).strip().lower()
    explicit_chat_template = fmt_raw in {"chat", "chat_template"}
    fmt = fmt_raw
    if fmt == "chat":
        fmt = "chat_template"
    elif fmt == "auto":
        fmt = "chat_template" if hasattr(tokenizer, "apply_chat_template") else "plain"

    if fmt == "plain":
        return messages_to_plain_text(messages, add_generation_prompt=add_generation_prompt)

    if fmt != "chat_template":
        raise ValueError("prompt_format must be 'plain', 'chat_template', 'chat', or 'auto'")

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    if not hasattr(tokenizer, "apply_chat_template"):
        if explicit_chat_template:
            raise RuntimeError(
                "prompt_format='chat_template' was requested, but tokenizer has no apply_chat_template()."
            )
        return _chat_fallback_text(messages, add_generation_prompt=add_generation_prompt)

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        if explicit_chat_template:
            raise RuntimeError(
                "prompt_format='chat_template' failed in tokenizer.apply_chat_template()."
            ) from exc

    return _chat_fallback_text(messages, add_generation_prompt=add_generation_prompt)
