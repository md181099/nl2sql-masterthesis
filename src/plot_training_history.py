#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from src.training_history_utils import (
        HISTORY_COLUMNS,
        central_plot_dir,
        coerce_float,
        coerce_int,
        history_paths,
        load_history_from_path,
        materialize_history_for_adapter,
        run_slug_from_adapter_dir,
        write_history_files,
    )
except ModuleNotFoundError:
    from training_history_utils import (
        HISTORY_COLUMNS,
        central_plot_dir,
        coerce_float,
        coerce_int,
        history_paths,
        load_history_from_path,
        materialize_history_for_adapter,
        run_slug_from_adapter_dir,
        write_history_files,
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    ylabel: str
    single_filename: str
    comparison_filename: str
    xlabel: str = "Training step"
    x_column: str = "step"


METRIC_SPECS: dict[str, MetricSpec] = {
    "loss": MetricSpec(
        column="loss",
        title="Training Loss over Steps",
        ylabel="Training loss",
        single_filename="training_loss_over_steps.png",
        comparison_filename="loss_comparison.png",
    ),
    "eval_loss": MetricSpec(
        column="eval_loss",
        title="Eval Loss over Epochs",
        ylabel="Eval loss",
        single_filename="eval_loss_over_epochs.png",
        comparison_filename="eval_loss_comparison.png",
        xlabel="Epoch",
        x_column="epoch",
    ),
    "grad_norm": MetricSpec(
        column="grad_norm",
        title="Gradient Norm over Steps",
        ylabel="Gradient norm",
        single_filename="grad_norm_over_steps.png",
        comparison_filename="grad_norm_comparison.png",
    ),
    "mean_token_accuracy": MetricSpec(
        column="mean_token_accuracy",
        title="Mean Token Accuracy over Steps",
        ylabel="Mean token accuracy",
        single_filename="mean_token_accuracy_over_steps.png",
        comparison_filename="mean_token_accuracy_comparison.png",
    ),
    "entropy": MetricSpec(
        column="entropy",
        title="Entropy over Steps",
        ylabel="Entropy",
        single_filename="entropy_over_steps.png",
        comparison_filename="entropy_comparison.png",
    ),
    "learning_rate": MetricSpec(
        column="learning_rate",
        title="Learning Rate over Steps",
        ylabel="Learning rate",
        single_filename="learning_rate_over_steps.png",
        comparison_filename="learning_rate_comparison.png",
    ),
    "num_tokens": MetricSpec(
        column="num_tokens",
        title="Tokens Processed over Steps",
        ylabel="Tokens processed",
        single_filename="tokens_processed_over_steps.png",
        comparison_filename="tokens_processed_comparison.png",
    ),
}

SINGLE_RUN_METRICS = [
    "loss",
    "eval_loss",
    "train_eval_loss",
    "grad_norm",
    "mean_token_accuracy",
    "entropy",
    "learning_rate",
    "num_tokens",
]
COMPARISON_METRICS = ["loss", "eval_loss", "mean_token_accuracy", "learning_rate", "grad_norm"]


@dataclass
class RunHistory:
    path: Path
    label: str
    rows: list[dict[str, str]]


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it or use the history CSV/JSONL files directly."
        ) from exc
    return plt


def _default_label(path: Path, project_root: Path) -> str:
    if path.is_dir():
        try:
            rel = path.resolve().relative_to((project_root / "adapters").resolve())
            if len(rel.parts) >= 2:
                return rel.parts[1]
        except ValueError:
            pass
        return path.name
    if path.name.endswith("_training_history.csv"):
        return path.name.removesuffix("_training_history.csv")
    if path.name.endswith("_training_history.jsonl"):
        return path.name.removesuffix("_training_history.jsonl")
    return path.stem


def _series(
    rows: Iterable[dict[str, str]],
    metric: str,
    *,
    x_column: str = "step",
) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x_value = coerce_float(row.get(x_column))
        value = coerce_float(row.get(metric))
        if x_value is None or value is None:
            continue
        points.append((x_value, value))
    points.sort(key=lambda item: item[0])
    return [point[0] for point in points], [point[1] for point in points]


def _plot_metric(
    runs: list[RunHistory],
    *,
    metric: str,
    out_file: Path,
    title: str,
    ylabel: str,
    dpi: int,
) -> Path | None:
    plt = _import_pyplot()
    plotted = 0
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for run in runs:
        xs, ys = _series(run.rows, metric, x_column=METRIC_SPECS[metric].x_column)
        if not xs:
            logger.warning("No values for metric '%s' in %s", metric, run.path)
            continue
        ax.plot(xs, ys, linewidth=1.8, label=run.label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel(METRIC_SPECS[metric].xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.ticklabel_format(axis="x", style="plain")
    if metric in {"learning_rate", "num_tokens"}:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if plotted > 1:
        ax.legend(frameon=False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _plot_train_eval_loss(
    histories: list[RunHistory],
    *,
    out_file: Path,
    dpi: int,
    title_prefix: str | None = None,
) -> Path | None:
    plt = _import_pyplot()
    plotted = 0
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for run in histories:
        train_xs, train_ys = _series(run.rows, "loss")
        eval_xs, eval_ys = _series(run.rows, "eval_loss")
        if train_xs:
            ax.plot(train_xs, train_ys, linewidth=1.8, label=f"{run.label} train")
            plotted += 1
        if eval_xs:
            ax.plot(eval_xs, eval_ys, marker="o", linewidth=1.8, label=f"{run.label} eval")
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    title = "Training and Eval Loss over Steps"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(frameon=False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def load_run_histories(
    run_paths: list[Path],
    *,
    labels: list[str] | None,
    project_root: Path,
) -> list[RunHistory]:
    histories: list[RunHistory] = []
    for index, path in enumerate(run_paths):
        rows = load_history_from_path(path)
        label = labels[index] if labels else _default_label(path, project_root)
        histories.append(RunHistory(path=path, label=label, rows=rows))
    return histories


def plot_histories(
    histories: list[RunHistory],
    *,
    out_dir: Path,
    metrics: list[str],
    comparison: bool,
    dpi: int = 300,
    title_prefix: str | None = None,
) -> list[Path]:
    if not histories:
        raise ValueError("No runs provided.")
    if not any(history.rows for history in histories):
        raise ValueError("No training history rows found.")

    generated: list[Path] = []
    for metric in metrics:
        if metric == "train_eval_loss":
            plotted = _plot_train_eval_loss(
                histories,
                out_file=out_dir / "train_eval_loss_over_steps.png",
                dpi=dpi,
                title_prefix=title_prefix,
            )
            if plotted is not None:
                generated.append(plotted)
            continue
        spec = METRIC_SPECS[metric]
        filename = spec.comparison_filename if comparison else spec.single_filename
        title = spec.title
        if comparison:
            title = title.replace(" over Steps", " Comparison")
        if title_prefix:
            title = f"{title_prefix}: {title}"
        out_file = out_dir / filename
        plotted = _plot_metric(
            histories,
            metric=metric,
            out_file=out_file,
            title=title,
            ylabel=spec.ylabel,
            dpi=dpi,
        )
        if plotted is not None:
            generated.append(plotted)
    return generated


def generate_plots_for_run(
    run_path: Path,
    *,
    out_dir: Path | None = None,
    label: str | None = None,
    dpi: int = 300,
    project_root: Path | None = None,
) -> list[Path]:
    project_root = project_root or Path(__file__).resolve().parents[1]
    run_path = run_path.resolve()
    out_dir = out_dir or run_path / "plots"
    labels = [label] if label else None
    histories = load_run_histories([run_path], labels=labels, project_root=project_root)
    return plot_histories(
        histories,
        out_dir=out_dir,
        metrics=SINGLE_RUN_METRICS,
        comparison=False,
        dpi=dpi,
    )


def generate_standard_single_run_outputs(
    adapter_dir: Path,
    *,
    project_root: Path | None = None,
    dpi: int = 300,
) -> list[Path]:
    project_root = project_root or Path(__file__).resolve().parents[1]
    adapter_dir = adapter_dir.resolve()
    generated: list[Path] = []
    generated.extend(
        generate_plots_for_run(
            adapter_dir,
            out_dir=adapter_dir / "plots",
            project_root=project_root,
            dpi=dpi,
        )
    )
    generated.extend(
        generate_plots_for_run(
            adapter_dir,
            out_dir=central_plot_dir(project_root, adapter_dir),
            project_root=project_root,
            dpi=dpi,
        )
    )
    return generated


def _materialize_if_requested(
    *,
    run_path: Path,
    rows: list[dict[str, str]],
    project_root: Path,
    history_out_dir: Path | None,
    no_materialize: bool,
) -> list[Path]:
    if no_materialize or not rows:
        return []

    created: list[Path] = []
    if history_out_dir is not None:
        csv_path, jsonl_path, _ = materialize_history_for_adapter(
            rows,
            adapter_dir=history_out_dir,
            project_root=project_root,
        )
        return [csv_path, jsonl_path]

    if run_path.is_dir():
        csv_path, jsonl_path = history_paths(run_path)
        if not csv_path.exists() and not jsonl_path.exists():
            materialize_history_for_adapter(rows, adapter_dir=run_path, project_root=project_root)
            created.extend([csv_path, jsonl_path])
        return created

    if run_path.suffix.lower() in {".log", ".txt", ".out", ".json"}:
        csv_path = run_path.with_name(f"{run_path.stem}_training_history.csv")
        jsonl_path = run_path.with_name(f"{run_path.stem}_training_history.jsonl")
        write_history_files(rows, csv_path, jsonl_path)
        created.extend([csv_path, jsonl_path])
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready plots from LoRA training_history CSV/JSONL files, "
            "adapter directories, trainer_state.json files, or legacy training logs."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Adapter dirs, training_history CSV/JSONL files, trainer_state.json files, or legacy logs.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels. Must match --runs length.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for PNG plots.",
    )
    parser.add_argument(
        "--history-out-dir",
        default=None,
        help="For a single legacy log, write training_history.csv/jsonl into this adapter directory.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root used for central results paths. Defaults to this repository root.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Do not write training_history.csv/jsonl when parsing legacy logs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only load/materialize history files; do not create PNG plots.",
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Optional prefix added to plot titles.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    run_paths = [Path(run).resolve() for run in args.runs]

    if args.labels is not None and len(args.labels) not in {0, len(run_paths)}:
        raise SystemExit("--labels must be omitted or provide exactly one label per run.")
    labels = args.labels if args.labels else None
    history_out_dir = Path(args.history_out_dir).resolve() if args.history_out_dir else None
    if history_out_dir is not None and len(run_paths) != 1:
        raise SystemExit("--history-out-dir can only be used with exactly one --runs input.")

    histories = load_run_histories(run_paths, labels=labels, project_root=project_root)
    for history in histories:
        logger.info("Loaded %d history rows from %s", len(history.rows), history.path)
        created = _materialize_if_requested(
            run_path=history.path,
            rows=history.rows,
            project_root=project_root,
            history_out_dir=history_out_dir,
            no_materialize=args.no_materialize,
        )
        for path in created:
            logger.info("Wrote %s", path)

    if args.no_plots:
        logger.info("Skipping PNG generation because --no-plots was set.")
        return

    comparison = len(histories) > 1
    metrics = COMPARISON_METRICS if comparison else SINGLE_RUN_METRICS
    generated = plot_histories(
        histories,
        out_dir=Path(args.out).resolve(),
        metrics=metrics,
        comparison=comparison,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )
    for path in generated:
        logger.info("Wrote %s", path)

    if not generated:
        raise SystemExit("No plots were generated. Check that the requested metrics exist.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
