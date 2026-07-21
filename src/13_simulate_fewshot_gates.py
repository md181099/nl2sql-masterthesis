#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_bool_int(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    return 0


def _as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _exact_mcnemar_p(left_only: int, right_only: int) -> float | None:
    n = left_only + right_only
    if n == 0:
        return None
    k = min(left_only, right_only)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _first_similarity(row: dict[str, Any], trace: dict[str, Any] | None) -> float | None:
    trace_scores = _json_list(trace.get("retrieved_scores") if trace else None)
    if trace_scores:
        return _as_float(trace_scores[0])
    row_scores = _json_list(row.get("retrieved_scores"))
    if row_scores:
        return _as_float(row_scores[0])
    return _as_float(row.get("retrieval_similarity"))


def _first_retrieved_id(row: dict[str, Any], trace: dict[str, Any] | None) -> str:
    trace_ids = _json_list(trace.get("retrieved_ids") if trace else None)
    if trace_ids:
        return str(trace_ids[0])
    row_ids = _json_list(row.get("retrieved_ids"))
    if row_ids:
        return str(row_ids[0])
    return ""


def _metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_as_bool_int(row.get(key, 0)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_cases(
    zero_rows: dict[str, dict[str, Any]],
    fewshot_rows: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for qid in sorted(set(zero_rows) & set(fewshot_rows)):
        z = zero_rows[qid]
        f = fewshot_rows[qid]
        trace = traces.get(qid)
        z_ok = _as_bool_int(z.get("exec_match"))
        f_ok = _as_bool_int(f.get("exec_match"))
        if z_ok and f_ok:
            group = "both_correct"
        elif z_ok and not f_ok:
            group = "fewshot_regression"
        elif not z_ok and f_ok:
            group = "fewshot_rescue"
        else:
            group = "both_wrong"
        cases.append(
            {
                "id": qid,
                "db_id": z.get("db_id", ""),
                "question": z.get("question", ""),
                "gold_sql": z.get("gold_sql", ""),
                "zero_pred_sql": z.get("pred_sql", ""),
                "fewshot_pred_sql": f.get("pred_sql", ""),
                "zero_exec_match": z_ok,
                "fewshot_exec_match": f_ok,
                "zero_pred_ok": _as_bool_int(z.get("pred_ok")),
                "fewshot_pred_ok": _as_bool_int(f.get("pred_ok")),
                "similarity": _first_similarity(f, trace),
                "retrieved_example_id": _first_retrieved_id(f, trace),
                "case_group": group,
            }
        )
    return cases


def _simulate_gate(
    cases: list[dict[str, Any]],
    *,
    name: str,
    threshold: float | None,
    always_fewshot: bool = False,
    never_fewshot: bool = False,
    oracle: bool = False,
) -> dict[str, Any]:
    fewshot_cases = 0
    exec_match = 0
    pred_ok = 0
    selected_rescues = 0
    selected_regressions = 0
    zero_correct_gate_wrong = 0
    zero_wrong_gate_correct = 0
    for case in cases:
        z_ok = int(case["zero_exec_match"])
        f_ok = int(case["fewshot_exec_match"])
        z_pred_ok = int(case["zero_pred_ok"])
        f_pred_ok = int(case["fewshot_pred_ok"])
        if oracle:
            use_fewshot = (not z_ok) and bool(f_ok)
        elif always_fewshot:
            use_fewshot = True
        elif never_fewshot:
            use_fewshot = False
        else:
            sim = _as_float(case.get("similarity"))
            use_fewshot = sim is not None and threshold is not None and sim >= threshold
        if use_fewshot:
            fewshot_cases += 1
            chosen_ok = f_ok
            chosen_pred_ok = f_pred_ok
            if (not z_ok) and f_ok:
                selected_rescues += 1
            if z_ok and not f_ok:
                selected_regressions += 1
        else:
            chosen_ok = z_ok
            chosen_pred_ok = z_pred_ok
        exec_match += chosen_ok
        pred_ok += chosen_pred_ok
        if z_ok and not chosen_ok:
            zero_correct_gate_wrong += 1
        if (not z_ok) and chosen_ok:
            zero_wrong_gate_correct += 1
    total = len(cases)
    return {
        "rule": name,
        "threshold": "" if threshold is None else threshold,
        "fewshot_cases": fewshot_cases,
        "fewshot_rate": fewshot_cases / total if total else 0.0,
        "ema": exec_match / total if total else 0.0,
        "esr": pred_ok / total if total else 0.0,
        "delta_vs_zero": (exec_match - sum(int(c["zero_exec_match"]) for c in cases)) / total
        if total
        else 0.0,
        "selected_rescues": selected_rescues,
        "selected_regressions": selected_regressions,
        "mcnemar_vs_zero_p": _exact_mcnemar_p(
            zero_correct_gate_wrong,
            zero_wrong_gate_correct,
        ),
    }


def run_simulation(
    *,
    zero_csv: Path,
    fewshot_csv: Path,
    retrieval_trace: Path,
    output_prefix: Path,
    title: str,
    thresholds: list[float],
    dry_run: bool,
) -> dict[str, Any]:
    zero_list = _read_csv(zero_csv)
    fewshot_list = _read_csv(fewshot_csv)
    trace_list = _read_jsonl(retrieval_trace)
    zero_rows = {str(row.get("id", "")): row for row in zero_list}
    fewshot_rows = {str(row.get("id", "")): row for row in fewshot_list}
    traces = {str(row.get("id", "")): row for row in trace_list}
    cases = _pairwise_cases(zero_rows, fewshot_rows, traces)
    groups = Counter(case["case_group"] for case in cases)
    total = len(cases)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_rows.append(
            _simulate_gate(
                cases,
                name=f"similarity >= {threshold:.2f}",
                threshold=threshold,
            )
        )
    threshold_rows.append(
        _simulate_gate(cases, name="always Few-Shot", threshold=None, always_fewshot=True)
    )
    threshold_rows.append(
        _simulate_gate(cases, name="never Few-Shot", threshold=None, never_fewshot=True)
    )
    oracle_row = _simulate_gate(cases, name="oracle_best_per_case", threshold=None, oracle=True)

    rescue_regression_rows = [
        case
        for case in cases
        if case["case_group"] in {"fewshot_rescue", "fewshot_regression"}
    ]
    per_db_rows: list[dict[str, Any]] = []
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_db[str(case.get("db_id", ""))].append(case)
    for db_id, db_cases in sorted(by_db.items()):
        z_ema = _mean([float(case["zero_exec_match"]) for case in db_cases]) or 0.0
        f_ema = _mean([float(case["fewshot_exec_match"]) for case in db_cases]) or 0.0
        per_db_rows.append(
            {
                "db_id": db_id,
                "n": len(db_cases),
                "zero_ema": z_ema,
                "fewshot_ema": f_ema,
                "delta_fewshot_minus_zero": f_ema - z_ema,
                "fewshot_rescues": sum(
                    1 for case in db_cases if case["case_group"] == "fewshot_rescue"
                ),
                "fewshot_regressions": sum(
                    1 for case in db_cases if case["case_group"] == "fewshot_regression"
                ),
            }
        )

    zero_ema = _metric(zero_list, "exec_match")
    zero_esr = _metric(zero_list, "pred_ok")
    fewshot_ema = _metric(fewshot_list, "exec_match")
    fewshot_esr = _metric(fewshot_list, "pred_ok")
    mcnemar_p = _exact_mcnemar_p(
        groups.get("fewshot_regression", 0),
        groups.get("fewshot_rescue", 0),
    )

    if not dry_run:
        _write_csv(output_prefix.with_name(output_prefix.name + "_similarity_thresholds.csv"), threshold_rows)
        _write_csv(output_prefix.with_name(output_prefix.name + "_pairwise_cases.csv"), cases)
        _write_csv(
            output_prefix.with_name(output_prefix.name + "_rescue_regression_cases.csv"),
            rescue_regression_rows,
        )
        _write_csv(output_prefix.with_name(output_prefix.name + "_per_database.csv"), per_db_rows)
        md = [
            f"# {title}",
            "",
            "Offline-Simulation: Keine Modellgeneration. Few-Shot- und Zero-Shot-Ausgaben werden pro Testfall virtuell kombiniert.",
            "",
            "## Inputs",
            "",
            f"- Zero-Shot CSV: `{zero_csv}`",
            f"- Few-Shot CSV: `{fewshot_csv}`",
            f"- Retrieval Trace: `{retrieval_trace}`",
            "",
            "## Baselines",
            "",
            "| Strategie | EMA | ESR |",
            "|---|---:|---:|",
            f"| Zero-Shot | {zero_ema:.4f} | {zero_esr:.4f} |",
            f"| Always Few-Shot | {fewshot_ema:.4f} | {fewshot_esr:.4f} |",
            "",
            "## Pairwise",
            "",
            f"- both correct: {groups.get('both_correct', 0)}",
            f"- Zero-Shot correct / Few-Shot wrong: {groups.get('fewshot_regression', 0)}",
            f"- Zero-Shot wrong / Few-Shot correct: {groups.get('fewshot_rescue', 0)}",
            f"- both wrong: {groups.get('both_wrong', 0)}",
            f"- Netto Few-Shot: {groups.get('fewshot_rescue', 0) - groups.get('fewshot_regression', 0)}",
            f"- McNemar p: {mcnemar_p}",
            "",
            "## Similarity Gates",
            "",
            "| Regel | Few-Shot Fälle | EMA | ESR | Delta vs Zero | Rescues | Regressions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in threshold_rows:
            md.append(
                f"| {row['rule']} | {row['fewshot_cases']} | {row['ema']:.4f} | "
                f"{row['esr']:.4f} | {row['delta_vs_zero']:.4f} | "
                f"{row['selected_rescues']} | {row['selected_regressions']} |"
            )
        md.extend(
            [
                "",
                "## Oracle Upper Bound",
                "",
                "Dieses Gate ist nicht für echte Evaluation zulässig, weil es Korrektheit kennt.",
                "",
                f"- Oracle EMA: {oracle_row['ema']:.4f}",
                f"- Oracle ESR: {oracle_row['esr']:.4f}",
                f"- Few-Shot-Fälle: {oracle_row['fewshot_cases']}",
                f"- Delta vs Zero: {oracle_row['delta_vs_zero']:.4f}",
                "",
            ]
        )
        output_prefix.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    return {
        "total_cases": total,
        "zero_ema": zero_ema,
        "zero_esr": zero_esr,
        "fewshot_ema": fewshot_ema,
        "fewshot_esr": fewshot_esr,
        "groups": dict(groups),
        "mcnemar_p": mcnemar_p,
        "best_similarity_or_baseline": max(threshold_rows, key=lambda row: row["ema"]),
        "oracle": oracle_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline simulation of adaptive Few-Shot gates from existing result files."
    )
    parser.add_argument("--zero-shot-csv", required=True, type=Path)
    parser.add_argument("--fewshot-csv", required=True, type=Path)
    parser.add_argument("--retrieval-trace", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--title", default="Few-Shot Gate Simulation")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated similarity thresholds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    thresholds = [
        float(item.strip())
        for item in str(args.thresholds).split(",")
        if item.strip()
    ]
    summary = run_simulation(
        zero_csv=args.zero_shot_csv,
        fewshot_csv=args.fewshot_csv,
        retrieval_trace=args.retrieval_trace,
        output_prefix=args.output_prefix,
        title=args.title,
        thresholds=thresholds,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
