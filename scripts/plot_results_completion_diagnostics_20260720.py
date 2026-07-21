#!/usr/bin/env python3
"""Create the thesis figure for Qwen-2B completion diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json"
SOURCE_SHA256 = "e4ca268c6c5d08733bcd22268c75fb41386907b2b915753bf85f0e0fe3222853"
PDF_OUT = ROOT / "figures/results_completion_diagnostics.pdf"
PNG_OUT = ROOT / "figures/results_completion_diagnostics.png"

LABELS = (
    "Limit bei\n256 Tokens",
    "erneut Limit bei\n512 Tokens",
    "Repetitions-\nregel",
    "Terminierung\nvor 512 Tokens",
    "neue Execution\nMatches",
)
COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7")
EXPECTED_VALUES = (2215, 2215, 2215, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_values() -> tuple[int, int, int, int, int]:
    actual_hash = sha256(SOURCE)
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(f"Source hash mismatch: {actual_hash}")

    with SOURCE.open(encoding="utf-8") as handle:
        data = json.load(handle)
    capped = data["capped_case_summary"]
    values = (
        sum(int(item["capped_cases"]) for item in capped.values()),
        sum(int(item["capped_again_512"]) for item in capped.values()),
        sum(int(item["continued_repetition"]) for item in capped.values()),
        sum(int(item["terminated_before_512"]) for item in capped.values()),
        sum(int(item["newly_correct"]) for item in capped.values()),
    )
    if values != EXPECTED_VALUES:
        raise RuntimeError(f"Authoritative completion values changed: {values}")
    return values


def format_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def main() -> None:
    for output in (PDF_OUT, PNG_OUT):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {output}")

    values = load_values()
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.65))
    x = np.arange(len(LABELS))
    bars = ax.bar(
        x,
        values,
        width=0.66,
        color=COLORS,
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )
    for bar, value in zip(bars, values):
        y = value + 55 if value else 45
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format_integer(value),
            ha="center",
            va="bottom",
            fontsize=10.0,
        )

    ax.set_ylabel("Beobachtungen")
    ax.set_xlabel("Qwen 3.5 2B Base, acht Bedingungen", labelpad=10)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 2500)
    ax.set_yticks(range(0, 2501, 500))
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors="#333333", length=3)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.96, bottom=0.25)

    metadata = {
        "Title": "Qwen 2B: Completionlimit- und Repetitionsdiagnostik",
        "Subject": "Autoritative Max-new-tokens-Sensitivitätsanalyse über acht Base-Bedingungen",
    }
    fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(PNG_OUT, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
