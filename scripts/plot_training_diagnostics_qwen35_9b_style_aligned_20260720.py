#!/usr/bin/env python3
"""Create the thesis-style Qwen 3.5 9B training diagnostic figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_ROOT = (
    ROOT
    / "results/diagnostics/qwen35_v2_training_curves/qwen35_9b_r8_alpha16"
)
TRAIN_SOURCE = DIAGNOSTICS_ROOT / "tables/training_loss_points.csv"
EVAL_SOURCE = DIAGNOSTICS_ROOT / "tables/training_diagnostics_wide.csv"
PLOT_CONFIG = ROOT / "configs/plot_qwen35_9b_v2_training_diagnostics.json"
TRAINER_STATE = (
    ROOT
    / "adapters/qwen35_9b_base/"
    "lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_"
    "schemaheaderfix_evalstop_maxlen2048_epochs5/checkpoints/"
    "checkpoint-1506/trainer_state.json"
)
PDF_OUT = ROOT / "figures/training_diagnostics_qwen35_9b_fullchat_loss.pdf"
PNG_OUT = ROOT / "figures/training_diagnostics_qwen35_9b_fullchat_loss.png"

EXPECTED_HASHES = {
    TRAIN_SOURCE: "4c4f8553091d1fa0981965d0284a9d1177890e67cf73ade6982e469e0bb44091",
    EVAL_SOURCE: "8d1f4d4a9e23fb43eea8f3d886464eb036c419df741b29de4ae9f271db503516",
    PLOT_CONFIG: "c5d5ca88f02511743772e0fd45f04c5367ed55ae6dc852badd29969894207620",
    TRAINER_STATE: "e1e553a9236143f91e84891268a04e40271d7be90c0cfc4372641ed66deb84d5",
}
EXPECTED_EVAL = (
    (1.0, 502, 0.4077516198158264),
    (2.0, 1004, 0.42377933859825134),
    (3.0, 1506, 0.4483620524406433),
)
RUN_NAME = "Qwen 3.5 9B r8/alpha16 v2"
BLUE = "#0072B2"
ORANGE = "#D55E00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_and_validate() -> tuple[list[int], list[float], list[float], list[int], list[float]]:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        require(actual == expected, f"SHA256 mismatch for {path}: {actual}")

    with PLOT_CONFIG.open(encoding="utf-8") as handle:
        config = json.load(handle)
    smoothing = config.get("smoothing", {})
    require(smoothing.get("method") == "ema", "Unexpected smoothing method")
    require(
        math.isclose(float(smoothing.get("alpha")), 0.15, rel_tol=0.0, abs_tol=1e-15),
        "Unexpected smoothing alpha",
    )
    require(smoothing.get("cross_epoch_boundaries") is False, "Unexpected cross-epoch smoothing")
    require(smoothing.get("replace_raw_values") is False, "Raw values must remain unchanged")

    train_rows = load_csv(TRAIN_SOURCE)
    require(len(train_rows) == 150, f"Expected 150 training points, found {len(train_rows)}")
    require({row["run_name"] for row in train_rows} == {RUN_NAME}, "Unexpected training run")
    train_steps = [int(row["global_step"]) for row in train_rows]
    require(train_steps == sorted(set(train_steps)), "Training steps are not unique and ordered")
    raw_loss = [float(row["training_loss_raw"]) for row in train_rows]
    smooth_loss = [float(row["training_loss_smoothed"]) for row in train_rows]
    require(all(math.isfinite(value) for value in raw_loss + smooth_loss), "Non-finite loss value")

    eval_rows = sorted(load_csv(EVAL_SOURCE), key=lambda row: int(row["global_step"]))
    require(len(eval_rows) == 3, f"Expected 3 validation rows, found {len(eval_rows)}")
    require({row["run_name"] for row in eval_rows} == {RUN_NAME}, "Unexpected validation run")
    for row, (epoch, step, loss) in zip(eval_rows, EXPECTED_EVAL):
        require(math.isclose(float(row["epoch"]), epoch, abs_tol=1e-12), "Epoch mismatch")
        require(int(row["global_step"]) == step, "Checkpoint step mismatch")
        require(
            math.isclose(float(row["official_eval_loss"]), loss, rel_tol=0.0, abs_tol=1e-15),
            "Official eval_loss mismatch",
        )
    best_rows = [row for row in eval_rows if row["is_best_checkpoint"].lower() == "true"]
    require(len(best_rows) == 1 and int(best_rows[0]["global_step"]) == 502, "Best checkpoint mismatch")

    with TRAINER_STATE.open(encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    require(int(trainer_state["global_step"]) == 1506, "Unexpected final training step")
    require(math.isclose(float(trainer_state["epoch"]), 3.0, abs_tol=1e-12), "Unexpected final epoch")
    require(
        Path(trainer_state["best_model_checkpoint"]).name == "checkpoint-502",
        "Trainer best checkpoint mismatch",
    )
    require(
        math.isclose(
            float(trainer_state["best_metric"]),
            EXPECTED_EVAL[0][2],
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "Trainer best metric mismatch",
    )
    state_eval = [entry for entry in trainer_state["log_history"] if "eval_loss" in entry]
    require(len(state_eval) == 3, "Trainer state does not contain three validation points")
    for entry, (_, step, loss) in zip(state_eval, EXPECTED_EVAL):
        require(int(entry["step"]) == step, "Trainer-state validation step mismatch")
        require(
            math.isclose(float(entry["eval_loss"]), loss, rel_tol=0.0, abs_tol=1e-15),
            "Trainer-state validation loss mismatch",
        )

    eval_steps = [step for _, step, _ in EXPECTED_EVAL]
    eval_loss = [loss for _, _, loss in EXPECTED_EVAL]
    return train_steps, raw_loss, smooth_loss, eval_steps, eval_loss


def create_plot() -> None:
    for output in (PDF_OUT, PNG_OUT):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    train_steps, raw_loss, smooth_loss, eval_steps, eval_loss = load_and_validate()
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
    ax.plot(
        train_steps,
        raw_loss,
        color=BLUE,
        linewidth=0.8,
        alpha=0.32,
        label="Trainingsverlust",
        zorder=2,
    )
    ax.plot(
        train_steps,
        smooth_loss,
        color=BLUE,
        linewidth=2.0,
        label="Geglätteter Trainingsverlust",
        zorder=3,
    )
    ax.plot(
        eval_steps,
        eval_loss,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=5.2,
        markeredgecolor="#333333",
        markeredgewidth=0.45,
        label="Full-Chat-Validierungsverlust",
        zorder=5,
    )

    best_step, best_loss = eval_steps[0], eval_loss[0]
    ax.scatter(
        [best_step],
        [best_loss],
        marker="*",
        s=155,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.0,
        zorder=7,
    )
    ax.annotate(
        "Bester Checkpoint: 502",
        xy=(best_step, best_loss),
        xytext=(610, 0.68),
        ha="left",
        va="center",
        fontsize=9.5,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
        zorder=8,
    )

    final_step = eval_steps[-1]
    ax.axvline(final_step, color="#666666", linewidth=1.0, linestyle=":", zorder=1)
    ax.text(
        final_step - 19,
        1.18,
        "Early Stopping",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9.5,
        color="#444444",
    )

    ax.set_xlabel("Trainingsschritt", labelpad=7)
    ax.set_ylabel("Verlust")
    ax.set_xlim(0, 1545)
    ax.set_ylim(0, 2.28)
    ax.set_xticks(range(0, 1501, 250))
    ax.set_yticks([value / 4 for value in range(0, 10)])
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="x", colors="#333333", length=3)
    ax.tick_params(axis="y", colors="#333333", length=3)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.82, bottom=0.18)

    metadata = {
        "Title": "Qwen 3.5 9B LoRA v2: Training und Full-Chat-Validierung",
        "Subject": "Training loss and authoritative MixedVal2500-v2 Full-Chat eval_loss",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        PDF_OUT,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    fig.savefig(
        PNG_OUT,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    create_plot()
