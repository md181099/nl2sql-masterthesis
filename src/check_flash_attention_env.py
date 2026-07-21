#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from typing import Any


def _module_version(module_name: str) -> tuple[bool, str | None, str | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, getattr(module, "__version__", None), None


def main() -> int:
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }

    torch_ok, torch_version, torch_error = _module_version("torch")
    report["torch"] = {"installed": torch_ok, "version": torch_version, "error": torch_error}
    cuda_available = False

    if torch_ok:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        report["torch"].update(
            {
                "cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": cuda_available,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "gpu_name_0": torch.cuda.get_device_name(0) if cuda_available else None,
                "device_capability_0": (
                    list(torch.cuda.get_device_capability(0)) if cuda_available else None
                ),
                "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else None,
            }
        )

    for module_name in ("transformers", "trl"):
        ok, version, error = _module_version(module_name)
        report[module_name] = {"installed": ok, "version": version, "error": error}

    flash_ok, flash_version, flash_error = _module_version("flash_attn")
    report["flash_attn"] = {
        "installed": flash_ok,
        "version": flash_version,
        "error": flash_error,
    }

    try:
        from transformers.utils import is_flash_attn_2_available

        flash_attention_2_available = bool(is_flash_attn_2_available())
        fa2_error = None
    except Exception as exc:
        flash_attention_2_available = False
        fa2_error = f"{type(exc).__name__}: {exc}"

    report["flash_attention_2"] = {
        "transformers_available": flash_attention_2_available,
        "error": fa2_error,
    }

    ready = bool(torch_ok and cuda_available and flash_ok and flash_attention_2_available)
    report["status"] = "READY" if ready else "NOT READY"
    report["reason"] = (
        "flash_attention_2 is importable and CUDA is visible."
        if ready
        else "Need CUDA visibility and a compatible flash-attn installation."
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(report["status"])
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
