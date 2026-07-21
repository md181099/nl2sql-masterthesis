#!/usr/bin/env python3
"""Create the thesis figure for authoritative zero-shot EMA results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audits/derived/cross_model_zero_shot_comparison_20260716.csv"
PDF_OUT = ROOT / "figures/results_zero_shot_ema_base_vs_lora.pdf"
PNG_OUT = ROOT / "figures/results_zero_shot_ema_base_vs_lora.png"

MODEL_ORDER = ("qwen2b", "llama3b", "qwen9b")
MODEL_LABELS = {
    "qwen2b": "(a) Qwen 3.5 2B",
    "llama3b": "(b) Llama 3.2 3B Instruct",
    "qwen9b": "(c) Qwen 3.5 9B",
}
EXPECTED_COUNTS = {
    "qwen2b": (464, 615),
    "llama3b": (568, 630),
    "qwen9b": (711, 769),
}

BASE_COLOR = "#0072B2"
LORA_COLOR = "#D55E00"


def format_decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def load_authoritative_values() -> dict[str, dict[str, float]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected = {
        row["model_key"]: row
        for row in rows
        if row.get("condition") == "zero_shot" and row.get("model_key") in MODEL_ORDER
    }
    if set(selected) != set(MODEL_ORDER):
        raise RuntimeError("Authoritative zero-shot rows are incomplete or ambiguous")

    values: dict[str, dict[str, float]] = {}
    for model_key in MODEL_ORDER:
        row = selected[model_key]
        if int(row["n"]) != 1032:
            raise RuntimeError(f"Unexpected case count for {model_key}: {row['n']}")
        counts = (int(row["a_correct"]), int(row["b_correct"]))
        if counts != EXPECTED_COUNTS[model_key]:
            raise RuntimeError(f"Authoritative count mismatch for {model_key}: {counts}")

        base = float(row["a_ema"]) * 100.0
        lora = float(row["b_ema"]) * 100.0
        delta = float(row["delta_pp"])
        if abs((lora - base) - delta) > 1e-10:
            raise RuntimeError(f"Stored EMA delta is inconsistent for {model_key}")
        values[model_key] = {"base": base, "lora": lora, "delta": delta}
    return values


def main() -> None:
    for output in (PDF_OUT, PNG_OUT):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {output}")

    values = load_authoritative_values()
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

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.65), sharey=True)
    colors = (BASE_COLOR, LORA_COLOR)

    for index, (ax, model_key) in enumerate(zip(axes, MODEL_ORDER)):
        model_values = values[model_key]
        heights = (model_values["base"], model_values["lora"])
        bars = ax.bar(
            (0, 1),
            heights,
            width=0.66,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )

        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.25,
                f"{format_decimal(value)} %",
                ha="center",
                va="bottom",
                fontsize=10.0,
            )

        ax.text(
            0.5,
            0.955,
            f"$\Delta$ = +{format_decimal(model_values['delta'])} PP",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.7,
            color="#333333",
        )
        ax.set_xlabel(MODEL_LABELS[model_key], labelpad=10)
        ax.set_xticks([])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 90)
        ax.set_yticks(range(0, 91, 10))
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")
        ax.tick_params(axis="y", colors="#333333", length=3)
        if index > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)

    axes[0].set_ylabel("EMA in %")
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
    fig.subplots_adjust(left=0.08, right=0.995, top=0.82, bottom=0.21, wspace=0.16)

    metadata = {
        "Title": "Zero-shot EMA: Ausgangsmodelle und LoRA v2",
        "Subject": "Autoritative Zero-shot-Ergebnisse der 48-Run-Hauptmatrix",
    }
    fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(PNG_OUT, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
