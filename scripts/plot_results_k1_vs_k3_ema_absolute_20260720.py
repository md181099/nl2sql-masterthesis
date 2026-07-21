#!/usr/bin/env python3
"""Plot absolute k=1 and k=3 EMA values from the authoritative pair table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audits/derived/k1_vs_k3_paired_statistics_20260718.csv"
OUTPUT_PDF = ROOT / "figures/results_k1_vs_k3_ema_absolute.pdf"
OUTPUT_PNG = ROOT / "figures/results_k1_vs_k3_ema_absolute.png"

MODEL_ORDER = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B",
    "llama3b": "Llama 3.2 3B Instruct",
    "qwen9b": "Qwen 3.5 9B",
}
ROLE_ORDER = ["base", "lora_v2"]
ROLE_LABELS = {"base": "Ausgangsmodell", "lora_v2": "LoRA v2"}
CONDITIONS = [
    "top3",
    "top3_gate070",
    "top3_gate085",
    "structure_top3",
    "structure_top3_gate070",
    "structure_top3_gate085",
]
TICK_LABELS = [
    "Top",
    "Top G\n0,70",
    "Top G\n0,85",
    "Struct.",
    "Struct. G\n0,70",
    "Struct. G\n0,85",
]
SERIES = ["k1", "k3"]
SERIES_LABELS = {"k1": "k=1", "k3": "k=3"}
COLORS = {"k1": "#0072B2", "k3": "#D55E00"}


def read_and_validate() -> dict[tuple[str, str, str], tuple[float, float]]:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 36:
        raise RuntimeError(f"Expected 36 authoritative pairs, found {len(rows)}")

    values: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in rows:
        if row["family"] != "K1_VS_K3_DEMONSTRATION_COUNT_FAMILY":
            raise RuntimeError(f"Unexpected family in {row['pair_id']}")
        if row["comparability_status"] != "MATCHED_WITH_LIMITATIONS":
            raise RuntimeError(f"Unexpected comparability status in {row['pair_id']}")
        if int(row["cases"]) != 1032:
            raise RuntimeError(f"Unexpected case count in {row['pair_id']}")

        key = (row["model_key"], row["model_role"], row["condition"])
        if key in values:
            raise RuntimeError(f"Duplicate pair key: {key}")

        k1 = float(row["k1_ema"]) * 100.0
        k3 = float(row["k3_ema"]) * 100.0
        if round(k1 * 10.32) != int(row["k1_correct"]):
            raise RuntimeError(f"k1 EMA/count mismatch in {row['pair_id']}")
        if round(k3 * 10.32) != int(row["k3_correct"]):
            raise RuntimeError(f"k3 EMA/count mismatch in {row['pair_id']}")
        values[key] = (k1, k3)

    expected = {
        (model, role, condition)
        for model in MODEL_ORDER
        for role in ROLE_ORDER
        for condition in CONDITIONS
    }
    if set(values) != expected:
        raise RuntimeError("Pair matrix does not match 3 models x 2 roles x 6 conditions")
    return values


def german(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#333333")


def create_plot(values: dict[tuple[str, str, str], tuple[float, float]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.37

    for column, model in enumerate(MODEL_ORDER):
        axes[0, column].set_title(MODEL_LABELS[model], pad=8, fontweight="semibold")
        for row_index, role in enumerate(ROLE_ORDER):
            ax = axes[row_index, column]
            style_axis(ax)
            for series_index, series in enumerate(SERIES):
                offset = (-0.5 if series_index == 0 else 0.5) * width
                value_index = 0 if series == "k1" else 1
                heights = [values[(model, role, condition)][value_index]
                           for condition in CONDITIONS]
                bars = ax.bar(
                    x + offset,
                    heights,
                    width,
                    color=COLORS[series],
                    edgecolor="#333333",
                    linewidth=0.55,
                    zorder=3,
                )
                for bar, value in zip(bars, heights):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.7,
                        german(value),
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6.3,
                        color="#111111",
                    )
            ax.set_ylim(0, 82)
            ax.set_yticks(np.arange(0, 81, 10))
            ax.set_xticks(x, TICK_LABELS)
            ax.tick_params(axis="x", length=0, pad=3)

    fig.supylabel("EMA in %", x=0.014, fontsize=10.5)
    fig.text(0.005, 0.675, ROLE_LABELS["base"], rotation=90,
             ha="center", va="center", fontsize=9.5, fontweight="semibold")
    fig.text(0.005, 0.255, ROLE_LABELS["lora_v2"], rotation=90,
             ha="center", va="center", fontsize=9.5, fontweight="semibold")

    handles = [
        Patch(facecolor=COLORS[series], edgecolor="#333333",
              label=SERIES_LABELS[series])
        for series in SERIES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=2.6,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.012,
        "Deskriptive Darstellung; k3 mit erweitertem Kontextbudget, "
        "Qwen-9B-k3 zus\u00e4tzlich mit abweichender Timeoutpolicy.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.12,
                        top=0.89, hspace=0.30, wspace=0.15)

    for path in (OUTPUT_PDF, OUTPUT_PNG):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "Title": "Absolute EMA values for k=1 and k=3",
        "Author": "NL2SQL Testbench",
        "Subject": "Descriptive comparison with methodological limitations",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", metadata=metadata)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    create_plot(read_and_validate())


if __name__ == "__main__":
    main()
