#!/usr/bin/env python3
"""Create deterministic publication plots from joined diagnostics tables only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FORMATS = {"png", "pdf", "svg"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing input table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"Temporary path collision: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def import_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen35_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (7.2, 4.4),
            "savefig.bbox": "tight",
            "svg.hashsalt": "qwen35-v2-training-diagnostics",
        }
    )
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class Caption:
    filename: str
    title: str
    data: str
    axes: str
    best_checkpoint: str
    interpretation: str
    limitation: str
    suggestion: str


def labels(language: str) -> dict[str, str]:
    if language == "en":
        return {
            "step": "Global step",
            "epoch": "Epoch",
            "loss": "Loss",
            "accuracy": "SQL token accuracy (%)",
            "perplexity": "SQL perplexity",
            "gap": "Difference",
            "raw_train": "Raw training loss",
            "smooth_train": "Smoothed training loss (EMA)",
            "eval": "Full-Chat eval loss",
            "best": "Official best checkpoint",
        }
    return {
        "step": "Globaler Step",
        "epoch": "Epoche",
        "loss": "Loss",
        "accuracy": "SQL-Token-Accuracy (%)",
        "perplexity": "SQL-Perplexity",
        "gap": "Differenz",
        "raw_train": "Roher Training Loss",
        "smooth_train": "Geglaetteter Training Loss (EMA)",
        "eval": "Full-Chat Eval Loss",
        "best": "Offizieller Best-Checkpoint",
    }


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row["run_name"], []).append(row)
    for values in result.values():
        values.sort(key=lambda row: number(row.get("global_step")) or 0.0)
    return result


def values(rows: list[dict[str, str]], x: str, y: str, scale: float = 1.0) -> tuple[list[float], list[float]]:
    points = []
    for row in rows:
        x_value = number(row.get(x))
        y_value = number(row.get(y))
        if x_value is not None and y_value is not None:
            points.append((x_value, y_value * scale))
    points.sort()
    return [item[0] for item in points], [item[1] for item in points]


def values_with_gaps(rows: list[dict[str, str]], x: str, y: str, scale: float = 1.0) -> tuple[list[float], list[float]]:
    points = []
    for row in rows:
        x_value = number(row.get(x))
        if x_value is None:
            continue
        y_value = number(row.get(y))
        points.append((x_value, math.nan if y_value is None else y_value * scale))
    points.sort()
    return [item[0] for item in points], [item[1] for item in points]


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if boolean(row.get("is_best_checkpoint"))]
    require(len(selected) == 1, f"Expected exactly one official best checkpoint, got {len(selected)}")
    return selected[0]


def style_axis(ax: Any, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(frameon=False)


def line(ax: Any, xs: list[float], ys: list[float], label: str, index: int, *, linewidth: float = 1.8, alpha: float = 1.0) -> None:
    if xs:
        ax.plot(
            xs,
            ys,
            label=label,
            color=COLORS[index % len(COLORS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=4.2,
            linewidth=linewidth,
            alpha=alpha,
        )


def mark_best(ax: Any, row: dict[str, str], x_field: str, y_field: str, label: str) -> None:
    x_value = number(row.get(x_field))
    y_value = number(row.get(y_field))
    if x_value is not None and y_value is not None:
        ax.scatter([x_value], [y_value], marker="*", s=145, facecolor="white", edgecolor="black", linewidth=1.2, zorder=10, label=label)


def render_figure(fig: Any, base_name: str, config: dict[str, Any], generated: list[Path]) -> None:
    formats = config["image_formats"]
    dpi = int(config["dpi"])
    output_dir = resolve_path(config["output_dir"])
    targets = [output_dir / "plots" / fmt / f"{base_name}.{fmt}" for fmt in formats]
    for target in targets:
        require(not target.exists(), f"Refusing to overwrite plot: {target}")
    for target, fmt in zip(targets, formats):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        metadata = {"Creator": "qwen35-v2-training-diagnostics", "CreationDate": None, "ModDate": None}
        if fmt == "svg":
            metadata = {"Creator": "qwen35-v2-training-diagnostics", "Date": None}
        elif fmt == "png":
            metadata = {"Software": "qwen35-v2-training-diagnostics"}
        try:
            fig.savefig(temporary, format=fmt, dpi=dpi if fmt == "png" else None, metadata=metadata)
            require(temporary.is_file() and temporary.stat().st_size > 0, f"Empty plot output: {temporary}")
            os.link(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        generated.append(target)


def add_single_plots(
    config: dict[str, Any],
    wide: list[dict[str, str]],
    train: list[dict[str, str]],
    plt: Any,
    optional_metric_anomalies: dict[str, Any],
) -> tuple[list[Path], list[Caption], list[str]]:
    text = labels(config["language"])
    generated: list[Path] = []
    captions: list[Caption] = []
    skipped: list[str] = []
    runs = grouped(wide)
    train_runs = grouped(train)
    require(len(runs) == 1, "Single-run plot config must contain exactly one run")
    run_name, rows = next(iter(runs.items()))
    train_rows = train_runs.get(run_name, [])
    official_best = best_row(rows)

    fig, ax = plt.subplots(constrained_layout=True)
    raw_x, raw_y = values(train_rows, "global_step", "training_loss_raw")
    smooth_x, smooth_y = values(train_rows, "global_step", "training_loss_smoothed")
    eval_x, eval_y = values(rows, "global_step", "official_eval_loss")
    require(raw_x and eval_x, "Training/eval loss data missing")
    ax.plot(raw_x, raw_y, color=COLORS[0], linewidth=0.8, alpha=0.35, label=text["raw_train"])
    ax.plot(smooth_x, smooth_y, color=COLORS[0], linewidth=2.0, linestyle="-", label=text["smooth_train"])
    line(ax, eval_x, eval_y, text["eval"], 1)
    mark_best(ax, official_best, "global_step", "official_eval_loss", text["best"])
    last_checkpoint = max(rows, key=lambda row: number(row.get("global_step")) or -1.0)
    max_epochs = float(config["runs"][0]["hyperparameters"]["max_epochs"])
    last_epoch = number(last_checkpoint.get("epoch"))
    if last_epoch is not None and last_epoch + 1e-9 < max_epochs:
        ax.axvline(
            float(last_checkpoint["global_step"]),
            color="#666666",
            linewidth=1.0,
            linestyle=":",
            label=("Early Stop" if config["language"] == "en" else "Early Stop"),
        )
    ax.set_title("Training Loss und Full-Chat Validation Loss" if config["language"] == "de" else "Training and Full-Chat Validation Loss")
    style_axis(ax, text["step"], text["loss"])
    render_figure(fig, "01_training_and_fullchat_validation_loss", config, generated)
    plt.close(fig)
    captions.append(
        Caption(
            "01_training_and_fullchat_validation_loss",
            "Training Loss und Full-Chat Validation Loss",
            "Trainer-Log-History und offizieller MixedVal-v2 eval_loss.",
            "x: globaler Step; y: nicht normalisierter Loss.",
            f"Step {official_best['global_step']} (Epoche {official_best['epoch']}).",
            "Zeigt den deskriptiven Verlauf von Optimierung und Validation.",
            "Training und Evaluation nutzen dieselbe Full-Chat-Zielfunktion, aber unterschiedliche Daten und Aggregation.",
            "Verlauf des rohen und geglaetteten Training Loss sowie des Full-Chat Validation Loss; der offizielle Best-Checkpoint ist markiert.",
        )
    )

    posthoc_complete = all(number(row.get("eval_sql_loss")) is not None for row in rows)
    if not posthoc_complete:
        skipped.extend([
            "02_validation_loss_components_by_epoch",
            "03_sql_loss_by_validation_source",
            "04_sql_token_accuracy_by_validation_source",
            "05_sql_source_generalization_gap",
            "06_sql_perplexity_by_epoch",
        ])
        return generated, captions, skipped

    def multi_plot(base: str, title: str, ylabel: str, specs: list[tuple[str, str, float]], caption_limit: str, *, zero_line: bool = False) -> None:
        fig_local, ax_local = plt.subplots(constrained_layout=True)
        for index, (field, label_value, scale) in enumerate(specs):
            xs, ys = values(rows, "epoch", field, scale)
            line(ax_local, xs, ys, label_value, index)
        if zero_line:
            ax_local.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
        ax_local.set_title(title)
        style_axis(ax_local, text["epoch"], ylabel)
        render_figure(fig_local, base, config, generated)
        plt.close(fig_local)
        captions.append(Caption(base, title, "Zusammengefuehrte Trainer- und Post-hoc-Metriken.", f"x: Epoche; y: {ylabel}.", f"Offiziell: Step {official_best['global_step']}.", "Deskriptiver Vergleich gemessener Metrikverlaeufe.", caption_limit, f"{title} ueber die gespeicherten Epochencheckpoints."))

    multi_plot(
        "02_validation_loss_components_by_epoch",
        "Validation-Loss-Komponenten nach Epoche",
        text["loss"],
        [
            ("eval_fullchat_pack_macro_loss", "Full-Chat Pack-Makro", 1.0),
            ("eval_fullchat_token_micro_loss", "Full-Chat Token-Mikro", 1.0),
            ("eval_assistant_completion_loss", "Assistant-Completion", 1.0),
            ("eval_sql_loss", "SQL-only", 1.0),
        ],
        "Die Lossarten bewerten verschiedene Tokenmengen und sind nicht austauschbar.",
    )
    multi_plot(
        "03_sql_loss_by_validation_source",
        "SQL-only Loss nach Validation-Quelle",
        text["loss"],
        [
            ("eval_sql_loss", "Gesamt", 1.0),
            ("sqlcc_sql_loss", "SQLCC", 1.0),
            ("train_others_sql_loss", "train_others", 1.0),
        ],
        "Unterschiede sind deskriptiv und belegen allein keine Kausalitaet oder Spider-EMA.",
    )
    multi_plot(
        "04_sql_token_accuracy_by_validation_source",
        "SQL-Token-Accuracy nach Validation-Quelle",
        text["accuracy"],
        [
            ("eval_sql_token_accuracy", "Gesamt", 100.0),
            ("sqlcc_sql_token_accuracy", "SQLCC", 100.0),
            ("train_others_sql_token_accuracy", "train_others", 100.0),
        ],
        "Token Accuracy misst teacher-forced Next-Token-Treffer und ist nicht mit EMA gleichzusetzen.",
    )
    multi_plot(
        "05_sql_source_generalization_gap",
        "SQLCC-train_others Loss-Gap",
        text["gap"],
        [("sql_source_loss_gap", "train_others Loss - SQLCC Loss", 1.0)],
        "Der Gap ist eine vorab definierte, rein deskriptive Differenz.",
        zero_line=True,
    )
    # Add a zero reference to the last generated gap plot through a separate figure.
    fig, ax = plt.subplots(constrained_layout=True)
    xs, ys = values(rows, "epoch", "sql_source_accuracy_gap", 100.0)
    line(ax, xs, ys, "SQLCC Accuracy - train_others Accuracy", 0)
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_title("SQLCC-train_others Accuracy-Gap")
    style_axis(ax, text["epoch"], "Prozentpunkte")
    render_figure(fig, "05b_sql_source_accuracy_gap", config, generated)
    plt.close(fig)
    captions.append(Caption("05b_sql_source_accuracy_gap", "SQLCC-train_others Accuracy-Gap", "Post-hoc Quellenmetriken.", "x: Epoche; y: Prozentpunkte.", f"Offiziell: Step {official_best['global_step']}.", "Positiv bedeutet numerisch hoehere SQLCC-Accuracy.", "Rein deskriptiv; keine Signifikanz- oder Kausalaussage.", "Differenz der SQL-Token-Accuracy zwischen SQLCC und train_others."))
    multi_plot(
        "06_sql_perplexity_by_epoch",
        "SQL-Perplexity nach Epoche",
        text["perplexity"],
        [("eval_sql_perplexity", "SQL-Perplexity", 1.0)],
        "Perplexity ist exp(SQL-only Loss) und keine generative Ausfuehrungsmetrik.",
    )

    grad_x, grad_y = values_with_gaps(train_rows, "global_step", "grad_norm")
    valid_grad_count = sum(math.isfinite(value) for value in grad_y)
    grad_anomaly = optional_metric_anomalies.get("grad_norm", {})
    grad_nonfinite_count = int(grad_anomaly.get("nonfinite_count", 0) or 0)
    if valid_grad_count:
        fig, ax = plt.subplots(constrained_layout=True)
        line(ax, grad_x, grad_y, "Gradient Norm", 0, linewidth=1.2, alpha=0.85)
        ax.set_title("Gradient Norm ueber das Training")
        style_axis(ax, text["step"], "Gradient Norm")
        render_figure(fig, "07_gradient_norm_over_training", config, generated)
        plt.close(fig)
        anomaly_note = (
            " Im Trainer-State lag bei Step 10 eine nicht-endliche optionale Gradientennorm vor. "
            "Dieser einzelne Messpunkt wurde nicht imputiert und nicht geplottet. Training Loss, Evaluation Loss, "
            "Checkpointauswahl und Adaptergewichte waren davon nicht betroffen."
            if grad_nonfinite_count == 1 and grad_anomaly.get("affected_steps") == [10]
            else (
                f" {grad_nonfinite_count} nicht-endliche optionale Gradientennorm-Messpunkte wurden nicht imputiert und nicht geplottet."
                if grad_nonfinite_count
                else ""
            )
        )
        captions.append(Caption("07_gradient_norm_over_training", "Gradient Norm ueber das Training", "Trainer-Log-History.", "x: Step; y: Gradient Norm.", "Nicht checkpointselektierend.", "Technische Stabilitaetsdiagnose.", "Nicht als Qualitaetsmetrik interpretieren.", "Verlauf der geloggten Gradient Norm." + anomaly_note))
    else:
        skipped.append("07_gradient_norm_over_training")
    skipped.append("08_learning_rate_over_training (constant scheduler; geringe inhaltliche Aussage)")
    return generated, captions, skipped


def add_comparison_plots(config: dict[str, Any], wide: list[dict[str, str]], plt: Any) -> tuple[list[Path], list[Caption], list[str]]:
    text = labels(config["language"])
    generated: list[Path] = []
    captions: list[Caption] = []
    skipped: list[str] = []
    runs = grouped(wide)
    require(len(runs) >= 2, "Comparison config needs at least two runs")
    require(all(all(number(row.get("eval_sql_loss")) is not None for row in rows) for rows in runs.values()), "Comparison requires complete post-hoc diagnostics")

    def compare(base: str, title: str, field: str, ylabel: str, scale: float = 1.0) -> None:
        fig, ax = plt.subplots(constrained_layout=True)
        for index, (run_name, rows) in enumerate(sorted(runs.items())):
            xs, ys = values(rows, "epoch", field, scale)
            line(ax, xs, ys, run_name, index)
            selected = best_row(rows)
            x_value = number(selected.get("epoch"))
            y_value = number(selected.get(field))
            if x_value is not None and y_value is not None:
                ax.scatter([x_value], [y_value * scale], marker="*", s=120, facecolor="white", edgecolor=COLORS[index], linewidth=1.2, zorder=10)
        ax.set_title(title)
        style_axis(ax, text["epoch"], ylabel)
        render_figure(fig, base, config, generated)
        plt.close(fig)
        captions.append(Caption(base, title, "Methodisch kontrollierte MixedVal-v2-Laeufe.", f"x: Epoche; y: {ylabel}.", "Je Modell ist der Trainer-Best-Checkpoint markiert.", "Deskriptiver Modellgroessenvergleich.", "Loss/Accuracy ersetzen keine generative Spider-EMA.", f"{title}; Sterne markieren die offiziell nach Full-Chat eval_loss ausgewaehlten Checkpoints."))

    compare("11_2b_vs_9b_fullchat_eval_loss", "2B vs. 9B: Full-Chat Eval Loss", "official_eval_loss", text["loss"])
    compare("12_2b_vs_9b_sql_only_loss", "2B vs. 9B: SQL-only Loss", "eval_sql_loss", text["loss"])
    compare("13_2b_vs_9b_train_others_sql_loss", "2B vs. 9B: train_others SQL Loss", "train_others_sql_loss", text["loss"])
    compare("14_2b_vs_9b_sql_token_accuracy", "2B vs. 9B: SQL-Token-Accuracy", "eval_sql_token_accuracy", text["accuracy"], 100.0)

    summaries = [
        ("15a_2b_vs_9b_best_fullchat_loss", "Best-Checkpoint: Full-Chat Eval Loss", "official_eval_loss", text["loss"], 1.0),
        ("15b_2b_vs_9b_best_sql_loss", "Best-Checkpoint: SQL-only Loss", "eval_sql_loss", text["loss"], 1.0),
        ("15c_2b_vs_9b_best_train_others_sql_loss", "Best-Checkpoint: train_others SQL Loss", "train_others_sql_loss", text["loss"], 1.0),
        ("15d_2b_vs_9b_best_sql_token_accuracy", "Best-Checkpoint: SQL-Token-Accuracy", "eval_sql_token_accuracy", text["accuracy"], 100.0),
    ]
    for base, title, field, ylabel, scale in summaries:
        fig, ax = plt.subplots(constrained_layout=True)
        names = []
        ys = []
        colors = []
        for index, (run_name, rows) in enumerate(sorted(runs.items())):
            selected = best_row(rows)
            value = number(selected.get(field))
            require(value is not None, f"Missing {field} at best checkpoint for {run_name}")
            names.append(run_name)
            ys.append(value * scale)
            colors.append(COLORS[index])
        positions = list(range(len(names)))
        ax.bar(positions, ys, color=colors, edgecolor="black", linewidth=0.7, hatch=["", "//", "xx", ".."][: len(names)])
        ax.set_xticks(positions, names, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        render_figure(fig, base, config, generated)
        plt.close(fig)
        captions.append(Caption(base, title, "Nur offizielle Trainer-Best-Checkpoints.", f"x: Modell; y: {ylabel}.", "Je Lauf genau ein offizieller Best-Checkpoint.", "Kompakte deskriptive Gegenueberstellung.", "Keine gemeinsame Achse fuer Loss und Accuracy; keine EMA-Aussage.", f"{title} der offiziell ausgewaehlten v2-Checkpoints."))
    return generated, captions, skipped


def captions_markdown(captions: list[Caption], skipped: list[str], status: str) -> str:
    lines = ["# Plot Captions and Interpretation Notes", "", f"Pipeline status: `{status}`", ""]
    for item in captions:
        lines.extend(
            [
                f"## {item.filename}",
                "",
                f"- Titel: {item.title}",
                f"- Verwendete Daten: {item.data}",
                f"- Achsen: {item.axes}",
                f"- Best-Checkpoint: {item.best_checkpoint}",
                f"- Zulaessige Interpretation: {item.interpretation}",
                f"- Einschraenkung: {item.limitation}",
                f"- Bildunterschrift: {item.suggestion}",
                "",
            ]
        )
    if skipped:
        lines.extend(["## Nicht erzeugte Plots", ""] + [f"- {item}" for item in skipped] + [""])
    return "\n".join(lines)


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == 1, "Unsupported config schema_version")
    require(
        config.get("purpose") in {"qwen35_v2_training_diagnostics", "model_training_diagnostics"},
        "Unexpected config purpose",
    )
    require(config.get("overwrite") is False, "overwrite must be false")
    require(config.get("language") in {"de", "en"}, "Invalid language")
    formats = config.get("image_formats")
    require(isinstance(formats, list) and formats and set(formats) <= ALLOWED_FORMATS, "Invalid image formats")
    require(int(config.get("dpi", 0)) >= 300, "PNG DPI must be >= 300")
    output_dir = resolve_path(config["output_dir"])
    tables_manifest = output_dir / "manifests" / "training_diagnostics_manifest.json"
    return {
        "config_path": project_path(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir": project_path(output_dir),
        "tables_ready": tables_manifest.is_file(),
        "matplotlib_available": True,
        "formats": formats,
        "dpi": int(config["dpi"]),
    }


def generate(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    static = validate_config(config_path, config)
    output_dir = resolve_path(config["output_dir"])
    table_manifest_path = output_dir / "manifests" / "training_diagnostics_manifest.json"
    table_manifest = load_json(table_manifest_path)
    require(table_manifest.get("config_sha256") == sha256_file(config_path), "Table manifest belongs to another config")
    wide_path = output_dir / "tables" / "training_diagnostics_wide.csv"
    train_path = output_dir / "tables" / "training_loss_points.csv"
    wide = read_csv(wide_path)
    train = read_csv(train_path)
    require(wide, "Wide diagnostics table is empty")
    plt = import_pyplot()
    if config.get("mode") == "comparison":
        generated, captions, skipped = add_comparison_plots(config, wide, plt)
    else:
        generated, captions, skipped = add_single_plots(
            config,
            wide,
            train,
            plt,
            table_manifest.get("optional_metric_anomalies", {}),
        )
    status = table_manifest["status"]
    if status == "TRAINING_ONLY":
        message = "Post-hoc SQL diagnostics not available yet. Run the configured post-hoc evaluator first."
    elif status == "INCOMPLETE":
        message = "Post-hoc SQL diagnostics are incomplete; only non-post-hoc plots are valid."
    else:
        message = None
    captions_path = output_dir / "captions" / "plot_captions.md"
    atomic_write(captions_path, captions_markdown(captions, skipped, status).encode("utf-8"))
    generated.append(captions_path)
    plot_hashes = {project_path(path): sha256_file(path) for path in generated}
    plot_manifest_path = output_dir / "manifests" / "plot_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_only": bool(config.get("test_only", False)),
        "config_path": project_path(config_path),
        "config_sha256": sha256_file(config_path),
        "table_manifest_path": project_path(table_manifest_path),
        "table_manifest_sha256": sha256_file(table_manifest_path),
        "plotter_path": project_path(Path(__file__)),
        "plotter_sha256": sha256_file(Path(__file__)),
        "image_formats": config["image_formats"],
        "dpi": int(config["dpi"]),
        "language": config["language"],
        "generated_hashes": dict(sorted(plot_hashes.items())),
        "optional_metric_anomalies": table_manifest.get("optional_metric_anomalies", {}),
        "skipped": skipped,
        "no_interpolation": True,
        "no_dual_y_axes": True,
        "message": message,
    }
    atomic_write(plot_manifest_path, (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    generated.append(plot_manifest_path)
    return {
        "status": status,
        "config": static,
        "generated": [project_path(path) for path in generated],
        "skipped": skipped,
        "message": message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate plotting dependencies and config only.")
    parser.add_argument("--language", choices=("de", "en"), help="Override language without changing input data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    require(config_path.is_file(), f"Config missing: {config_path}")
    config = load_json(config_path)
    if args.language:
        config = dict(config)
        config["language"] = args.language
    if args.preflight:
        import_pyplot()
        result = validate_config(config_path, config)
        result["status"] = "PASS"
        result["plots_written"] = False
    else:
        result = generate(config_path, config)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
