#!/usr/bin/env python3
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return data


def get_param(
    args: Namespace,
    config: Mapping[str, Any],
    cli_name: str,
    default: Any,
    config_name: str | None = None,
) -> Any:
    cli_value = getattr(args, cli_name, None)
    if cli_value is not None:
        return cli_value
    key = config_name or cli_name
    if key in config and config[key] is not None:
        return config[key]
    return default


def get_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be an object")
    return section
