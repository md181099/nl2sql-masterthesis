#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _module_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "installed": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "installed": True,
        "version": getattr(module, "__version__", None),
        "error": None,
    }


def _dist_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_project_sft_v2_module() -> Any | None:
    script_path = Path(__file__).resolve().parent / "07_lora_finetune_sft_v2.py"
    if not script_path.exists():
        return None
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("project_sft_v2", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight check for Transformers kernels-community FlashAttention2 without training."
    )
    parser.add_argument(
        "--repo",
        default="kernels-community/flash-attn2",
        help="Kernel repo id to inspect. Default: kernels-community/flash-attn2.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional hub revision/commit for a kernel load test.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Kernel version to resolve when --revision is not set. Default: 1.",
    )
    parser.add_argument(
        "--try-load-kernel",
        action="store_true",
        help="Actually call transformers.integrations.hub_kernels.get_kernel(). May need network/cache.",
    )
    args = parser.parse_args()
    if args.revision is not None and args.version is not None:
        parser.error("--revision and --version are mutually exclusive.")
    if args.revision is None and args.version is None:
        args.version = 1

    requested = args.repo if args.revision is None else f"{args.repo}@{args.revision}"
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "repo": args.repo,
        "requested_revision": args.revision,
        "requested_version": args.version,
        "requested_attn_implementation": requested,
        "try_load_kernel": args.try_load_kernel,
    }

    torch_status = _module_status("torch")
    report["torch"] = torch_status
    cuda_available = False
    if torch_status["installed"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_status.update(
            {
                "cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": cuda_available,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "gpu_name_0": torch.cuda.get_device_name(0) if cuda_available else None,
            }
        )

    report["transformers"] = _module_status("transformers")
    report["trl"] = _module_status("trl")
    report["flash_attn"] = _module_status("flash_attn")

    kernels_status = _module_status("kernels")
    kernels_status["distribution_version"] = _dist_version("kernels")
    report["kernels"] = kernels_status

    resolved_revision = args.revision
    if kernels_status["installed"]:
        try:
            from kernels._versions import resolve_version_spec_as_ref

            if resolved_revision is None and args.version is not None:
                ref = resolve_version_spec_as_ref(args.repo, args.version)
                resolved_revision = ref.target_commit
                report["kernel_version_resolution"] = {
                    "ok": True,
                    "ref": ref.ref,
                    "target_commit": ref.target_commit,
                }
        except Exception as exc:
            report["kernel_version_resolution"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    pinned_attn_implementation = args.repo if resolved_revision is None else f"{args.repo}@{resolved_revision}"
    report["pinned_attn_implementation"] = pinned_attn_implementation

    try:
        from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
        from transformers.utils.import_utils import KERNELS_MIN_VERSION, is_kernels_available

        report["transformers_kernels"] = {
            "min_kernels_version": KERNELS_MIN_VERSION,
            "is_kernels_available": bool(is_kernels_available()),
            "flash_attention_2_fallback": FLASH_ATTN_KERNEL_FALLBACK.get("flash_attention_2"),
        }
    except Exception as exc:
        report["transformers_kernels"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        from trl.trainer.sft_trainer import FLASH_ATTENTION_VARIANTS

        variants = sorted(FLASH_ATTENTION_VARIANTS)
        report["trl_packing_warning"] = {
            "supported_variants": variants,
            "direct_kernel_suppresses_warning": args.repo in FLASH_ATTENTION_VARIANTS,
            "pinned_kernel_suppresses_warning_vanilla_trl": pinned_attn_implementation in FLASH_ATTENTION_VARIANTS,
        }
    except Exception as exc:
        report["trl_packing_warning"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        project_sft_v2 = _load_project_sft_v2_module()
        if project_sft_v2 is None:
            raise RuntimeError("Could not load project SFT v2 module.")
        accepts_pinned = bool(project_sft_v2._is_flash_attention_implementation(pinned_attn_implementation))
        report["project_trainer_support"] = {
            "accepts_pinned_attn_implementation": accepts_pinned,
            "registers_pinned_trl_variant": accepts_pinned and pinned_attn_implementation.startswith("kernels-community/"),
        }
    except Exception as exc:
        report["project_trainer_support"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }

    if kernels_status["installed"]:
        try:
            from kernels.utils import get_kernel_variants, has_kernel

            variant_kwargs = {"revision": resolved_revision} if resolved_revision is not None else {"version": args.version}
            has_build = bool(has_kernel(args.repo, **variant_kwargs))
            decisions = get_kernel_variants(args.repo, **variant_kwargs)
            report["kernel_build"] = {
                "has_kernel": has_build,
                "top_variants": [
                    {
                        "decision": type(item).__name__,
                        "variant": getattr(getattr(item, "variant", None), "variant_str", None),
                        "reason": getattr(item, "reason", None),
                    }
                    for item in decisions[:10]
                ],
            }
        except Exception as exc:
            report["kernel_build"] = {
                "has_kernel": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if args.try_load_kernel:
        try:
            from transformers.integrations.hub_kernels import get_kernel

            kernel = get_kernel(args.repo, revision=resolved_revision)
            report["kernel_load"] = {
                "ok": True,
                "module": getattr(kernel, "__name__", str(kernel)),
                "has_flash_attn_varlen_func": hasattr(kernel, "flash_attn_varlen_func"),
                "has_flash_attn_func": hasattr(kernel, "flash_attn_func"),
            }
        except Exception as exc:
            report["kernel_load"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    ready = bool(
        cuda_available
        and report.get("transformers_kernels", {}).get("is_kernels_available")
        and report.get("kernel_version_resolution", {"ok": resolved_revision is not None}).get("ok")
        and report.get("kernel_build", {}).get("has_kernel")
        and report.get("project_trainer_support", {}).get("accepts_pinned_attn_implementation")
    )
    report["status"] = "READY" if ready else "NOT READY"
    report["reason"] = (
        "CUDA is visible, kernels is available, the pinned kernel resolves, and the project trainer accepts it."
        if ready
        else "Need CUDA visibility, kernels availability, a resolvable kernel version/revision, and a matching build."
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(report["status"])
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
