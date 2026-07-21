#!/usr/bin/env python3
"""Create the thesis figure for broad exploratory error families."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audits/derived/conservative_error_analysis_broad_families_20260716.csv"
SOURCE_SHA256 = "c25622fbd2290b55ccd8af7d598e1e6e6811de71fd729fa844f5ec8ca78d01f7"
PDF_OUT = ROOT / "figures/results_error_analysis_broad_families.pdf"
PNG_OUT = ROOT / "figures/results_error_analysis_broad_families.png"

MODEL_ORDER = ("qwen2b", "llama3b", "qwen9b")
MODEL_LABELS = {
    "qwen2b": "(a) Qwen 3.5 2B",
    "llama3b": "(b) Llama 3.2 3B Instruct",
    "qwen9b": "(c) Qwen 3.5 9B",
}
ROLE_ORDER = ("base", "lora_v2")
FAMILY_ORDER = (
    "Output/Kontrolle",
    "Syntax/Ausführung",
    "Schema/Projektion",
    "Querystruktur/-logik",
    "Ergebnisabweichung",
    "Unklar/heuristisch",
)
BASE_COLOR = "#0072B2"
LORA_COLOR = "#D55E00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_values() -> dict[tuple[str, str, str], float]:
    actual_hash = sha256(SOURCE)
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(f"Source hash mismatch: {actual_hash}")

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["model_key"], row["role"], row["broad_family"])
        if key in values:
            raise RuntimeError(f"Duplicate source row: {key}")
        values[key] = float(row["rate_per_1032"]) * 100.0

    expected = {
        (model, role, family)
        for model in MODEL_ORDER
        for role in ROLE_ORDER
        for family in FAMILY_ORDER
    }
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise RuntimeError(f"Unexpected source matrix; missing={missing}, extra={extra}")
    return values


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
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.75), sharex=True, sharey=True)
    y = np.arange(len(FAMILY_ORDER))
    width = 0.37

    for index, (ax, model_key) in enumerate(zip(axes, MODEL_ORDER)):
        base_values = [values[(model_key, "base", family)] for family in FAMILY_ORDER]
        lora_values = [values[(model_key, "lora_v2", family)] for family in FAMILY_ORDER]
        ax.barh(
            y - width / 2,
            base_values,
            height=width,
            color=BASE_COLOR,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
        ax.barh(
            y + width / 2,
            lora_values,
            height=width,
            color=LORA_COLOR,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )

        ax.set_title(MODEL_LABELS[model_key], fontsize=10.5, pad=7)
        ax.set_xlim(0, 60)
        ax.set_xticks(range(0, 61, 10))
        ax.set_yticks(y, FAMILY_ORDER)
        ax.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.75, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")
        ax.tick_params(axis="x", colors="#333333", length=3)
        ax.tick_params(axis="y", colors="#333333", length=0, pad=5, labelsize=8.5)
        if index > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

    axes[0].invert_yaxis()
    legend_handles = (
        Patch(facecolor=BASE_COLOR, edgecolor="#333333", label="Ausgangsmodell"),
        Patch(facecolor=LORA_COLOR, edgecolor="#333333", label="LoRA v2"),
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.0,
    )
    fig.text(
        0.5,
        0.012,
        "Explorative Multi-Label-Kategorien; Familien sind nicht disjunkt.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    fig.supxlabel("Anteil an 1.032 Fällen in %", x=0.59, y=0.075, fontsize=11.5)
    fig.subplots_adjust(left=0.18, right=0.995, top=0.79, bottom=0.20, wspace=0.15)

    metadata = {
        "Title": "Breite explorative Fehlerfamilien nach Modellrolle",
        "Subject": "Autoritative konservative Zero-Shot-Fehleranalyse",
    }
    fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(PNG_OUT, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
