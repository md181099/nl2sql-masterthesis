#!/usr/bin/env python3
"""Create thesis figures from authoritative prompting/retrieval result tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv"
K1_K3_SOURCE = ROOT / "audits/derived/k1_vs_k3_paired_statistics_20260718.csv"
MAIN_PDF = ROOT / "figures/results_prompting_retrieval_ema_main.pdf"
MAIN_PNG = ROOT / "figures/results_prompting_retrieval_ema_main.png"
DELTA_PDF = ROOT / "figures/results_k1_vs_k3_ema_delta.pdf"
DELTA_PNG = ROOT / "figures/results_k1_vs_k3_ema_delta.png"

MODEL_ORDER = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {
    "qwen2b": "(a) Qwen 3.5 2B",
    "llama3b": "(b) Llama 3.2 3B Instruct",
    "qwen9b": "(c) Qwen 3.5 9B",
}
ROLE_ORDER = ["base", "lora_v2"]
ROLE_LABELS = {"base": "Ausgangsmodell", "lora_v2": "LoRA v2"}
COLORS = {"base": "#0072B2", "lora_v2": "#D55E00"}

MAIN_CONDITIONS = [
    "zero_shot",
    "static_seed42",
    "top1",
    "top1_gate070",
    "top1_gate085",
    "structure",
    "structure_gate070",
    "structure_gate085",
]
MAIN_TICK_LABELS = [
    "Zero",
    "Static",
    "Top-1",
    "Top-1\nG 0,70",
    "Top-1\nG 0,85",
    "Struct.",
    "Struct.\nG 0,70",
    "Struct.\nG 0,85",
]

K3_CONDITIONS = [
    "top3",
    "top3_gate070",
    "top3_gate085",
    "structure_top3",
    "structure_top3_gate070",
    "structure_top3_gate085",
]
K3_TICK_LABELS = [
    "Top",
    "Top G\n0,70",
    "Top G\n0,85",
    "Struct.",
    "Struct. G\n0,70",
    "Struct. G\n0,85",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def german(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def validate_and_index_main() -> dict[tuple[str, str, str], float]:
    rows = read_csv(MAIN_SOURCE)
    if len(rows) != 48:
        raise RuntimeError(f"Expected 48 authoritative main rows, found {len(rows)}")

    indexed: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["model_key"], row["role"], row["condition"])
        if key in indexed:
            raise RuntimeError(f"Duplicate main result key: {key}")
        indexed[key] = float(row["ema"]) * 100.0

    expected = {
        (model, role, condition)
        for model in MODEL_ORDER
        for role in ROLE_ORDER
        for condition in MAIN_CONDITIONS
    }
    if set(indexed) != expected:
        raise RuntimeError("Authoritative main result matrix does not match 3 x 2 x 8")
    return indexed


def validate_and_index_delta() -> dict[tuple[str, str, str], float]:
    rows = read_csv(K1_K3_SOURCE)
    if len(rows) != 36:
        raise RuntimeError(f"Expected 36 authoritative k1-k3 rows, found {len(rows)}")

    indexed: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["family"] != "K1_VS_K3_DEMONSTRATION_COUNT_FAMILY":
            raise RuntimeError(f"Unexpected statistical family: {row['family']}")
        if row["comparability_status"] != "MATCHED_WITH_LIMITATIONS":
            raise RuntimeError(f"Unexpected comparability status: {row['pair_id']}")
        key = (row["model_key"], row["model_role"], row["condition"])
        if key in indexed:
            raise RuntimeError(f"Duplicate k1-k3 result key: {key}")
        indexed[key] = float(row["delta_pp"])

    expected = {
        (model, role, condition)
        for model in MODEL_ORDER
        for role in ROLE_ORDER
        for condition in K3_CONDITIONS
    }
    if set(indexed) != expected:
        raise RuntimeError("Authoritative k1-k3 matrix does not match 3 x 2 x 6")
    return indexed


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#333333")


def common_legend(fig: plt.Figure) -> None:
    handles = [
        Patch(facecolor=COLORS[role], edgecolor="#333333", label=ROLE_LABELS[role])
        for role in ROLE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=2.4,
        handlelength=1.6,
    )


def save(fig: plt.Figure, pdf_path: Path, png_path: Path) -> None:
    for path in (pdf_path, png_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Prompting and retrieval results",
        "Author": "NL2SQL Testbench",
        "Subject": "Authoritative Spider Dev EMA results",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_main(values: dict[tuple[str, str, str], float]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.25), sharey=True)
    x = np.arange(len(MAIN_CONDITIONS))
    width = 0.37

    for ax, model in zip(axes, MODEL_ORDER):
        style_axis(ax)
        for offset_index, role in enumerate(ROLE_ORDER):
            offset = (-0.5 if offset_index == 0 else 0.5) * width
            heights = [values[(model, role, condition)] for condition in MAIN_CONDITIONS]
            bars = ax.bar(
                x + offset,
                heights,
                width,
                color=COLORS[role],
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
            for bar, value in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.8,
                    german(value),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.2,
                    color="#111111",
                )
        ax.set_ylim(0, 85)
        ax.set_yticks(np.arange(0, 81, 10))
        ax.set_xticks(x, MAIN_TICK_LABELS)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.text(0.5, -0.25, MODEL_LABELS[model], transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5)

    axes[0].set_ylabel("EMA in %")
    common_legend(fig)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.27, top=0.84, wspace=0.16)
    save(fig, MAIN_PDF, MAIN_PNG)


def plot_delta(values: dict[tuple[str, str, str], float]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.25), sharey=True)
    x = np.arange(len(K3_CONDITIONS))
    width = 0.37

    for ax, model in zip(axes, MODEL_ORDER):
        style_axis(ax)
        ax.axhline(0, color="#333333", linewidth=0.9, zorder=2)
        for offset_index, role in enumerate(ROLE_ORDER):
            offset = (-0.5 if offset_index == 0 else 0.5) * width
            heights = [values[(model, role, condition)] for condition in K3_CONDITIONS]
            bars = ax.bar(
                x + offset,
                heights,
                width,
                color=COLORS[role],
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
            for bar, value in zip(bars, heights):
                label = f"{value:+.2f}".replace(".", ",")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.08 if value >= 0 else -0.08),
                    label,
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    rotation=90,
                    fontsize=6.4,
                    color="#111111",
                )
        ax.set_ylim(-3.8, 1.5)
        ax.set_yticks(np.arange(-3, 2, 1))
        ax.set_xticks(x, K3_TICK_LABELS)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.text(0.5, -0.25, MODEL_LABELS[model], transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5)

    axes[0].set_ylabel("EMA(k=3) - EMA(k=1) in PP")
    common_legend(fig)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.84, wspace=0.16)
    save(fig, DELTA_PDF, DELTA_PNG)


def main() -> None:
    plot_main(validate_and_index_main())
    plot_delta(validate_and_index_delta())


if __name__ == "__main__":
    main()
