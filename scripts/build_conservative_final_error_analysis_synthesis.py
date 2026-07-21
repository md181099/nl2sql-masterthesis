#!/usr/bin/env python3
"""Build the additive, conservative final error-analysis synthesis.

This script only reads frozen analysis artifacts. It validates their SHA256
provenance, aggregates existing labels, and writes new thesis-facing tables,
plots, an audit, and a manifest. It performs no model, tokenizer, retriever,
parser, generation, evaluation, or training work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen35_conservative_error_analysis_mpl")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATE = "20260716"
N = 1032
STATUS = "PASS MIT METHODISCHEN EINSCHRAENKUNGEN"

MODEL_ORDER = ("qwen2b", "llama3b", "qwen9b")
MODEL_LABELS = {
    "qwen2b": "Qwen 3.5 2B",
    "llama3b": "Llama 3.2 3B Instruct",
    "qwen9b": "Qwen 3.5 9B",
}
ROLE_LABELS = {"base": "Ausgangsmodell", "lora_v2": "LoRA v2"}

SOURCES = {
    "audits/audit_cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_20260716.md": "4304813c8b5fc6a87c62291b2c6c4ff90b747d43dfb217bb07fe4db6d2513b74",
    "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json": "24b4dec07d2d4981b42ce22e1295d27b0ccd9cbcc10666a422118b267fd14e37",
    "audits/audit_cross_model_zero_shot_error_analysis_20260716.md": "1c9cdedf3f5fe699ab3f4485678b09acb3cf61dca6407e483e0c6549f26991ee",
    "audits/cross_model_zero_shot_error_analysis_manifest_20260716.json": "a5cde941ca9d44c2a29bf19cf0c2ffdb0053732c97270ef228a0b908083a7327",
    "audits/error_analysis_codebook_20260716.md": "0e53cb8f346cbacc04aa8c7751981807afdf02d6d0a36979af11a8ab3931ef07",
    "audits/derived/cross_model_zero_shot_error_analysis_cases_20260716.csv": "aee58459c12729ea831207f26e70ff9048dc72fd6cbb00c7022bb4b484c727ee",
    "audits/derived/cross_model_zero_shot_error_labels_long_20260716.csv": "2116869f36f06d13d2ca2b8d18f491db0da0d09d88ebabd2880e2c37972485cf",
    "audits/derived/cross_model_zero_shot_error_transition_summary_20260716.csv": "d9f0f71dac67762a219caaafb4b8353a9462136648678f3eab043f06f25dafbc",
    "audits/derived/cross_model_zero_shot_error_category_summary_20260716.csv": "6f81c9f11a24303ad5004531da583ad5981898217308b8acfe9fb3e7b34017b6",
    "audits/derived/cross_model_zero_shot_error_category_statistics_20260716.csv": "f9986b69e4b243114faa36e0d82007487969f6b4b6b1611b4c4a7ff286f068ab",
    "audits/derived/cross_model_zero_shot_alternative_valid_sql_20260716.csv": "aa9efc5522a09392beaa47d5e889b9e27ba85140c2329a012c322fa2393b7681",
    "audits/derived/cross_model_fewshot_harm_error_analysis_20260716.csv": "052dd0efd5035c319d068350be4490e24b873b6722634723d9d807c8c240fcdc",
    "scripts/analyze_cross_model_zero_shot_error_taxonomy.py": "2c01f8e175a9325da3770a98e92b2d573846cdf1ebef303657da9ae950388ea4",
    "audits/audit_local_sql_ast_preflight_and_ai_error_preannotation_20260716.md": "9dd28282904a8242f622f1b65b1a6e30b2c9188c0279fd13f77d611770479843",
    "audits/local_sql_ast_preflight_and_ai_error_preannotation_manifest_20260716.json": "60da5ad7d1b5fb74439f0c83ab4f67f6f00d6232907e69cb0437d8ac44a82272",
    "audits/derived/cross_model_zero_shot_error_ai_preannotation_20260716.csv": "5fc6c476197b22aa9514ccff29d4e85d62d0466050c0b6e683bd887b3155bd87",
    "audits/derived/cross_model_zero_shot_error_human_review_priority_20260716.csv": "755985694d1720d2789b5a6798f5f88b5cb000c86d2f2079b0fc4aa644c2ed02",
    "audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260716.md": "94511610d0433e1e43cdb26ce0da22531530eb7da7a0e48ad3a7c0c22457fe1e",
    "audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json": "e4ca268c6c5d08733bcd22268c75fb41386907b2b915753bf85f0e0fe3222853",
    "audits/derived/qwen35_2b_base_maxnew256_vs_512_summary_20260716.csv": "c656475f316b7a70aa2dbdf46247f20abe8f952e02ce7f8c2439fd0d7f222d7c",
    "audits/derived/qwen35_2b_base_maxnew256_vs_512_capped_case_analysis_20260716.csv": "2a006f3471707044aab9e921e93ed1ae8499dba656b9698024c8f4261432b83b",
    "audits/derived/qwen35_2b_base_maxnew256_vs_512_repetition_analysis_20260716.csv": "67b7df4fc3cb5bdea128383d95b7ebfd8a5b5c8d1899004cdc60fc3e62c2bac9",
}

OUT = {
    "audit": "audits/audit_conservative_final_error_analysis_synthesis_20260716.md",
    "manifest": "audits/conservative_final_error_analysis_synthesis_manifest_20260716.json",
    "transitions": "audits/derived/conservative_error_analysis_transition_table_20260716.csv",
    "types": "audits/derived/conservative_error_analysis_repair_regression_types_20260716.csv",
    "families": "audits/derived/conservative_error_analysis_broad_families_20260716.csv",
    "evidence": "audits/derived/conservative_error_analysis_evidence_levels_20260716.csv",
    "labels": "audits/derived/conservative_error_analysis_concrete_e1_e2_labels_20260716.csv",
    "examples": "audits/derived/conservative_error_analysis_qualitative_examples_20260716.csv",
    "tables": "audits/derived/conservative_error_analysis_thesis_ready_tables_20260716.md",
    "text": "audits/derived/conservative_error_analysis_thesis_ready_text_20260716.md",
    "statements": "audits/derived/conservative_error_analysis_supported_statements_20260716.md",
    "limitations": "audits/derived/conservative_error_analysis_limitations_20260716.md",
}

PLOTS = {
    "transitions": "audits/plots/conservative_error_analysis_transitions_20260716",
    "types": "audits/plots/conservative_error_analysis_repair_regression_types_20260716",
    "families": "audits/plots/conservative_error_analysis_broad_families_20260716",
    "evidence": "audits/plots/conservative_error_analysis_evidence_levels_20260716",
    "completion": "audits/plots/conservative_error_analysis_completion_diagnostics_20260716",
}

META_LABELS = {
    "COMPLEX_MULTI_COMPONENT_ERROR",
    "HEURISTIC_ONLY",
    "MANUAL_REVIEW_REQUIRED",
    "UNCLASSIFIED",
    "PARSER_UNAVAILABLE",
    "PARSER_DISAGREEMENT",
}

BROAD_FAMILY = {
    "OUTPUT_CONTROL": "Output/Kontrolle",
    "EXECUTION_SYNTAX": "Syntax/Ausführung",
    "SCHEMA_LINKING": "Schema/Projektion",
    "PROJECTION": "Schema/Projektion",
    "AGGREGATION": "Querystruktur/-logik",
    "JOIN": "Querystruktur/-logik",
    "FILTER": "Querystruktur/-logik",
    "GROUPING": "Querystruktur/-logik",
    "ORDER_LIMIT": "Querystruktur/-logik",
    "SUBQUERY_SET": "Querystruktur/-logik",
    "RESULT_CARDINALITY": "Ergebnisabweichung",
    "UNCLASSIFIED_REVIEW": "Unklar/heuristisch",
}
BROAD_ORDER = (
    "Output/Kontrolle",
    "Syntax/Ausführung",
    "Schema/Projektion",
    "Querystruktur/-logik",
    "Ergebnisabweichung",
    "Unklar/heuristisch",
)

# Fixed, balanced illustration set from the existing 180-case package. Every
# non-T4 example must retain at least one E1/E2 concrete label. The set is not
# used for prevalence estimates.
EXAMPLE_SELECTION = {
    "repair": {
        "qwen2b": ("SPIDER_DEV_000263", "SPIDER_DEV_000614"),
        "llama3b": ("SPIDER_DEV_000011", "SPIDER_DEV_000584"),
        "qwen9b": ("SPIDER_DEV_000071",),
    },
    "regression": {
        "qwen2b": ("SPIDER_DEV_000966",),
        "llama3b": ("SPIDER_DEV_000482", "SPIDER_DEV_000404"),
        "qwen9b": ("SPIDER_DEV_000791", "SPIDER_DEV_000440"),
    },
    "persistent": {
        "qwen2b": ("SPIDER_DEV_000579",),
        "llama3b": ("SPIDER_DEV_000903",),
        "qwen9b": ("SPIDER_DEV_000388",),
    },
    "alternative_valid": {
        "qwen2b": ("SPIDER_DEV_000658",),
        "qwen9b": ("SPIDER_DEV_000809",),
    },
}

SUPPORTED_STATEMENTS = [
    "LoRA reparierte in allen drei Modelllinien mehr Zero-Shot-Fälle, als es verschlechterte.",
    "Die Nettoverbesserungen betrugen 151 Fälle bei Qwen 3.5 2B, 62 bei Llama 3.2 3B Instruct und 58 bei Qwen 3.5 9B.",
    "Ein wesentlicher Teil der Reparaturen betraf bereits ausführbare, aber auf der vorhandenen Datenbankinstanz ergebnisfalsche SQL-Abfragen.",
    "Verbesserte Ausführbarkeit allein erklärt den beobachteten LoRA-Gewinn deshalb nicht.",
    "Trotz positiver Nettoeffekte entstanden in allen Modelllinien neue Regressionen.",
    "Qwen 3.5 9B wies nach LoRA mit 180 Fällen weniger persistente Zero-Shot-Fehler auf als Qwen 3.5 2B (327) und Llama 3.2 3B Instruct (312).",
    "Die regelbasierte Multi-Label-Analyse deutet häufig auf Änderungen in Schema-, Projektions- und Querystrukturkomponenten hin; diese Zuordnungen sind explorativ.",
    "Der Qwen-2B-LoRA-Gewinn ist mit besserer Ausgabe- und Terminierungskontrolle vereinbar, wird dadurch aber nicht vollständig erklärt.",
    "Die 512-Token-Sensitivität stützt nicht die Annahme, dass das Qwen-2B-Verhalten lediglich durch ein knappes 256-Token-Budget verursacht wurde.",
    "Unterschiedliche, auf der vorhandenen Datenbankinstanz jeweils execution-korrekte SQLs zeigen Grenzen textbasierter Exaktheitsmetriken.",
    "Few-Shot-Verschlechterungen waren explorativ mit zusätzlichen Tabellen, Joins und struktureller Komplexität assoziiert.",
    "Die Error Analysis beschreibt beobachtete Muster und erlaubt keine kausale Interpretation von Modellgröße oder Architektur.",
]

UNSUPPORTED_STATEMENTS = [
    "Die automatische Taxonomie ist menschlich validiert.",
    "Die KI-Vorannotation ist eine manuelle Prüfung oder ein Goldstandard.",
    "Die Fehlerkategorien bilden kausale und disjunkte Fehlerursachen ab.",
    "LoRA behebt alle Schema- oder Joinfehler.",
    "Jede automatische Labelzuweisung erklärt kausal das falsche Ergebnis.",
    "Qwen 3.5 9B macht ausschließlich wegen seiner Parameterzahl weniger Fehler.",
    "Repetition erklärt den gesamten Qwen-2B-LoRA-Gewinn.",
    "Few-Shot-Demonstrationen verursachen nachweislich die beobachteten Fehler.",
    "Unterschiedliche SQLs mit demselben Ergebnis sind auf allen möglichen Datenbankinstanzen logisch äquivalent.",
    "Die 180 geschichteten Fälle sind eine repräsentative Prävalenzstichprobe.",
    "Meta-Labels wie HEURISTIC_ONLY sind fachliche SQL-Fehler.",
    "Nicht signifikante oder unklare Befunde beweisen Gleichheit.",
]

LIMITATIONS = [
    "Kein lokaler AST-Parser war verfügbar.",
    "Die Analyse verwendet den deterministischen Clause-Fallback project-local-sqlite-clause-fallback 1.0.0.",
    "Die feingranularen Labels wurden nicht vollständig menschlich validiert.",
    "Mehrere Fehlerlabels können derselben Prediction zugeordnet sein.",
    "Labelzahlen sind deshalb keine disjunkten Fallzahlen und ihre Summen können Fallzahlen überschreiten.",
    "Komponentenunterschiede sind nicht zwingend die kausale Fehlerursache.",
    "Execution Match gilt nur für die vorhandene Datenbankinstanz.",
    "Gold-SQL ist nicht die einzige mögliche korrekte Formulierung.",
    "Die KI-Vorannotation wurde nicht als Goldstandard oder Prävalenzgrundlage verwendet.",
    "Die geschichtete 180-Fall-Stichprobe wurde nicht vollständig menschlich codiert.",
    "Die Few-Shot-Schadensanalyse ist explorativ und nicht kausal.",
    "Cross-Model-Vergleiche bleiben aufgrund unterschiedlicher Modellfamilien Klasse B beziehungsweise B+.",
    "Autoritative Spider-Difficulty-Labels waren nicht verfügbar.",
    "Semantische Qualität und Outputkontrolle lassen sich mit den vorhandenen Artefakten nicht vollständig kausal trennen.",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"Unexpected boolean value: {value!r}")


def labels(value: str) -> set[str]:
    return {item for item in value.split(";") if item}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_csv(path: str, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    target = ROOT / path
    require(not target.exists(), f"Refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        require(bool(rows), f"Cannot infer CSV fields for empty output: {path}")
        fields = rows[0].keys()
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str, content: str) -> None:
    target = ROOT / path
    require(not target.exists(), f"Refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def f_pct(value: float) -> str:
    return f"{value:.2f} %".replace(".", ",")


def md_escape(value: Any, limit: int | None = None) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    if limit and len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    align = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(md_escape(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, align, *body])


def save_plot(base: str) -> None:
    target = ROOT / base
    target.parent.mkdir(parents=True, exist_ok=True)
    png = target.with_suffix(".png")
    pdf = target.with_suffix(".pdf")
    refresh = os.environ.get("CONSERVATIVE_SYNTHESIS_REFRESH_PLOTS") == "1"
    if refresh:
        require(png.is_file() and pdf.is_file(), f"Refresh requested for missing plot pair: {base}")
    else:
        require(not png.exists(), f"Refusing to overwrite existing plot: {png}")
        require(not pdf.exists(), f"Refusing to overwrite existing plot: {pdf}")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def source_integrity() -> list[dict[str, Any]]:
    rows = []
    for relative, expected in SOURCES.items():
        path = ROOT / relative
        require(path.is_file(), f"Missing authoritative source: {relative}")
        actual = sha256(path)
        require(actual == expected, f"SHA256 mismatch for {relative}: {actual} != {expected}")
        rows.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual, "status": "PASS"})
    return rows


def derive_transitions(raw: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        "qwen2b": (241, 99, 142, 90, 38, 52, 327, 374),
        "llama3b": (152, 64, 88, 90, 39, 51, 312, 478),
        "qwen9b": (141, 11, 130, 83, 17, 66, 180, 628),
    }
    by_model = {row["model_key"]: row for row in raw}
    require(set(by_model) == set(MODEL_ORDER), "Unexpected transition model set")
    transitions: list[dict[str, Any]] = []
    types: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row = by_model[model]
        values = tuple(
            int(row[key])
            for key in (
                "T1_repairs",
                "T1a_nonexec_to_correct",
                "T1b_execwrong_to_correct",
                "T2_regressions",
                "T2a_correct_to_nonexec",
                "T2b_correct_to_execwrong",
                "T3_persistent_errors",
                "T4_stable_correct",
            )
        )
        require(values == expected[model], f"Frozen transition mismatch for {model}: {values}")
        repairs, t1a, t1b, regressions, t2a, t2b, persistent, stable = values
        total = int(row["total"])
        require(total == N and repairs + regressions + persistent + stable == N, f"Transition total mismatch for {model}")
        require(t1a + t1b == repairs and t2a + t2b == regressions, f"Transition subgroup mismatch for {model}")
        transitions.append(
            {
                "model_line": MODEL_LABELS[model],
                "model_key": model,
                "repairs": repairs,
                "regressions": regressions,
                "net_improvement": repairs - regressions,
                "repair_to_regression_ratio": repairs / regressions,
                "persistent_errors": persistent,
                "stable_correct": stable,
                "total": total,
                "repair_rate_per_1032": repairs / N,
                "regression_rate_per_1032": regressions / N,
                "persistent_rate_per_1032": persistent / N,
                "stable_correct_rate_per_1032": stable / N,
            }
        )
        types.append(
            {
                "model_line": MODEL_LABELS[model],
                "model_key": model,
                "technical_repairs_nonexec_to_correct": t1a,
                "result_based_repairs_execwrong_to_correct": t1b,
                "technical_regressions_correct_to_nonexec": t2a,
                "result_based_regressions_correct_to_execwrong": t2b,
                "technical_share_of_repairs": t1a / repairs,
                "result_based_share_of_repairs": t1b / repairs,
                "technical_share_of_regressions": t2a / regressions,
                "result_based_share_of_regressions": t2b / regressions,
            }
        )
    return transitions, types


def derive_families(
    long_rows: list[dict[str, str]], case_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    family_cases: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in long_rows:
        broad = BROAD_FAMILY.get(row["error_family"])
        require(broad is not None, f"Unmapped error family: {row['error_family']}")
        family_cases[(row["model_key"], row["role"], broad)].add(row["case_id"])

    incorrect: dict[tuple[str, str], int] = {}
    for model in MODEL_ORDER:
        relevant = [row for row in case_rows if row["model_key"] == model]
        require(len(relevant) == N, f"Expected {N} cases for {model}")
        incorrect[(model, "base")] = sum(not as_bool(row["starting_execution_match"]) for row in relevant)
        incorrect[(model, "lora_v2")] = sum(not as_bool(row["lora_execution_match"]) for row in relevant)

    result = []
    for model in MODEL_ORDER:
        for role in ("base", "lora_v2"):
            for broad in BROAD_ORDER:
                count = len(family_cases[(model, role, broad)])
                result.append(
                    {
                        "model_line": MODEL_LABELS[model],
                        "model_key": model,
                        "role": role,
                        "role_label": ROLE_LABELS[role],
                        "broad_family": broad,
                        "unique_case_count": count,
                        "rate_per_1032": count / N,
                        "incorrect_prediction_count": incorrect[(model, role)],
                        "share_of_incorrect_predictions": count / incorrect[(model, role)],
                        "counting_note": "Unique cases within family; families overlap because classification is multi-label.",
                        "inference_class": "EXPLORATIVE RULE-BASED MULTI-LABEL DIAGNOSTIC",
                    }
                )
    return result


def derive_evidence(long_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = Counter((row["model_key"], row["role"], row["evidence_level"]) for row in long_rows)
    result = []
    for model in MODEL_ORDER:
        for role in ("base", "lora_v2"):
            total = sum(counts[(model, role, level)] for level in ("E1", "E2", "E3", "E4"))
            for level in ("E1", "E2", "E3", "E4"):
                count = counts[(model, role, level)]
                result.append(
                    {
                        "model_line": MODEL_LABELS[model],
                        "model_key": model,
                        "role": role,
                        "role_label": ROLE_LABELS[role],
                        "evidence_level": level,
                        "label_assignments": count,
                        "share_of_role_label_assignments": count / total,
                        "role_label_assignments_total": total,
                    }
                )
    require(sum(row["label_assignments"] for row in result) == len(long_rows), "Evidence aggregation mismatch")
    return result


def derive_concrete_labels(
    long_rows: list[dict[str, str]], case_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    direct: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    family: dict[str, str] = {}
    for row in long_rows:
        family[row["error_label"]] = BROAD_FAMILY[row["error_family"]]
        if row["evidence_level"] in {"E1", "E2"}:
            direct[(row["model_key"], row["role"], row["case_id"], row["error_label"])].add(row["evidence_level"])

    result = []
    for model in MODEL_ORDER:
        for action, transition, role, column in (
            ("repair", "T1", "base", "repaired_labels"),
            ("regression", "T2", "lora_v2", "introduced_labels"),
        ):
            counter: Counter[str] = Counter()
            evidence_by_label: dict[str, set[str]] = defaultdict(set)
            for row in case_rows:
                if row["model_key"] != model or row["transition_group"] != transition:
                    continue
                for label in labels(row[column]):
                    ev = direct.get((model, role, row["case_id"], label), set())
                    if label in META_LABELS or not ev:
                        continue
                    counter[label] += 1
                    evidence_by_label[label].update(ev)
            for rank, (label, count) in enumerate(counter.most_common(10), 1):
                result.append(
                    {
                        "model_line": MODEL_LABELS[model],
                        "model_key": model,
                        "outcome_role": action,
                        "transition_group": transition,
                        "rank": rank,
                        "error_label": label,
                        "broad_family": family[label],
                        "unique_outcome_cases_with_label": count,
                        "evidence_levels": ";".join(sorted(evidence_by_label[label])),
                        "meta_label_excluded": False,
                        "interpretation_note": "Multi-label count within outcome cases; not a disjoint or causal error count.",
                    }
                )
    return result


def example_rows(
    case_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    pre_rows: list[dict[str, str]],
    alternative_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    case_index = {(row["model_key"], row["case_id"]): row for row in case_rows}
    pre_index = {(row["model_line"], row["case_id"]): row for row in pre_rows}
    alt_index = {(row["model_key"], row["case_id"]): row for row in alternative_rows}
    long_index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in long_rows:
        long_index[(row["model_key"], row["role"], row["case_id"])].append(row)

    output = []
    transition_map = {"repair": "T1", "regression": "T2", "persistent": "T3", "alternative_valid": "T4"}
    role_map = {"repair": "base", "regression": "lora_v2", "persistent": "base"}
    for example_type, by_model in EXAMPLE_SELECTION.items():
        for model, case_ids in by_model.items():
            for case_id in case_ids:
                case = case_index[(model, case_id)]
                require(case["transition_group"] == transition_map[example_type], f"Example transition mismatch: {model} {case_id}")
                pre = pre_index[(MODEL_LABELS[model], case_id)]
                if example_type == "alternative_valid":
                    alt = alt_index[(model, case_id)]
                    require(alt["formulation_class"] == "structurally_different_execution_equivalent", f"Alternative example mismatch: {case_id}")
                    direct_labels = ["ALTERNATIVE_VALID_FORMULATION"]
                    evidence_levels = "E1"
                    interpretation = "Beide SQL-Formulierungen execution-matchen auf der vorhandenen Datenbankinstanz, unterscheiden sich aber nach der eingefrorenen Strukturklassifikation."
                else:
                    role = role_map[example_type]
                    entries = long_index[(model, role, case_id)]
                    direct_labels = sorted(
                        {
                            row["error_label"]
                            for row in entries
                            if row["evidence_level"] in {"E1", "E2"} and row["error_label"] not in META_LABELS
                        }
                    )
                    require(direct_labels, f"Example lacks E1/E2 concrete evidence: {model} {case_id}")
                    evidence_levels = ";".join(sorted({row["evidence_level"] for row in entries if row["error_label"] in direct_labels}))
                    if example_type == "repair":
                        interpretation = "Das Ausgangsmodell verfehlte Execution Match; die LoRA-Prediction war korrekt. Die Labels illustrieren beobachtete Ausgangsabweichungen."
                    elif example_type == "regression":
                        interpretation = "Das Ausgangsmodell war korrekt; die LoRA-Prediction verfehlte Execution Match. Die Labels illustrieren neu beobachtete LoRA-Abweichungen."
                    else:
                        interpretation = "Beide Predictions verfehlten Execution Match. Die Labels illustrieren fortbestehende oder verwandte Komponentenabweichungen."
                output.append(
                    {
                        "example_type": example_type,
                        "model_line": MODEL_LABELS[model],
                        "model_key": model,
                        "transition": case["transition_group"],
                        "transition_subgroup": case["transition_subgroup"],
                        "case_id": case_id,
                        "db_id": case["db_id"],
                        "question": case["question"],
                        "relevant_schema_excerpt": pre["relevant_schema_excerpt"],
                        "gold_sql": case["gold_sql"],
                        "starting_model_sql": case["starting_pred_sql"],
                        "lora_sql": case["lora_pred_sql"],
                        "starting_execution_status": f"success={case['starting_execution_success']}; match={case['starting_execution_match']}",
                        "lora_execution_status": f"success={case['lora_execution_success']}; match={case['lora_execution_match']}",
                        "exploratory_error_labels": ";".join(direct_labels[:6]),
                        "evidence_levels": evidence_levels,
                        "short_interpretation": interpretation,
                        "methodological_note": "Illustrative case only; no prevalence or causal inference. Execution equivalence is limited to the current database instance.",
                    }
                )
    require(len(output) == 15, f"Expected 15 examples, got {len(output)}")
    require(Counter(row["example_type"] for row in output) == Counter({"repair": 5, "regression": 5, "persistent": 3, "alternative_valid": 2}), "Example type distribution mismatch")
    require(Counter(row["model_key"] for row in output) == Counter({"qwen2b": 5, "llama3b": 5, "qwen9b": 5}), "Example model balance mismatch")
    return output


def alternative_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    expected = {"qwen2b": (374, 299), "llama3b": (478, 282), "qwen9b": (628, 419)}
    for model in MODEL_ORDER:
        subset = [row for row in rows if row["model_key"] == model]
        different = sum(row["formulation_class"] == "structurally_different_execution_equivalent" for row in subset)
        require((len(subset), different) == expected[model], f"Alternative SQL mismatch for {model}")
        result.append(
            {
                "model_line": MODEL_LABELS[model],
                "stable_correct_pairs": len(subset),
                "different_sql_both_execution_match": different,
                "share": different / len(subset),
            }
        )
    require(sum(row["stable_correct_pairs"] for row in result) == 1480, "Alternative SQL T4 total mismatch")
    require(sum(row["different_sql_both_execution_match"] for row in result) == 1000, "Alternative SQL different total mismatch")
    return result


def fewshot_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    unique = {
        model: {(row["condition"], row["case_id"]) for row in rows if row["model_key"] == model}
        for model in ("qwen2b", "llama3b")
    }
    require(len(unique["qwen2b"]) == 210, "Qwen 2B few-shot harm count mismatch")
    require(len(unique["llama3b"]) == 166, "Llama 3B few-shot harm count mismatch")
    top = {}
    for model in ("qwen2b", "llama3b"):
        counter = Counter(row["error_label"] for row in rows if row["model_key"] == model)
        top[model] = [{"label": label, "assignments": count} for label, count in counter.most_common(6)]
    return {
        "qwen2b_unique_condition_cases": 210,
        "llama3b_unique_condition_cases": 166,
        "qwen2b_top_patterns": top["qwen2b"],
        "llama3b_top_patterns": top["llama3b"],
        "causal_interpretation_allowed": False,
    }


def make_plots(
    transitions: list[dict[str, Any]],
    types: list[dict[str, Any]],
    families: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    q2_manifest: dict[str, Any],
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    model_names = [row["model_line"] for row in transitions]
    x = np.arange(len(model_names))

    plt.figure(figsize=(9.2, 5.5))
    bottoms = np.zeros(len(model_names))
    parts = [
        ("repairs", "Reparaturen", "#2a9d8f"),
        ("regressions", "Regressionen", "#e76f51"),
        ("persistent_errors", "Persistent falsch", "#f4a261"),
        ("stable_correct", "Stabil korrekt", "#457b9d"),
    ]
    for key, label, color in parts:
        values = np.array([row[key] for row in transitions])
        plt.bar(x, values, bottom=bottoms, label=label, color=color, width=0.66)
        for i, value in enumerate(values):
            if value >= 70:
                plt.text(i, bottoms[i] + value / 2, str(int(value)), ha="center", va="center", fontsize=8)
        bottoms += values
    plt.xticks(x, model_names)
    plt.ylabel("Fälle (n = 1.032 je Modelllinie)")
    plt.title("Gepaarte Zero-Shot-Fallübergänge durch LoRA")
    plt.legend(ncol=2, frameon=False)
    plt.ylim(0, 1080)
    plt.grid(axis="y", alpha=0.2)
    save_plot(PLOTS["transitions"])

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    keys = [
        ("technical_repairs_nonexec_to_correct", "Nicht ausführbar → korrekt", "#2a9d8f"),
        ("result_based_repairs_execwrong_to_correct", "Ausführbar falsch → korrekt", "#457b9d"),
        ("technical_regressions_correct_to_nonexec", "Korrekt → nicht ausführbar", "#e76f51"),
        ("result_based_regressions_correct_to_execwrong", "Korrekt → ausführbar falsch", "#f4a261"),
    ]
    width = 0.19
    for j, (key, label, color) in enumerate(keys):
        values = [row[key] for row in types]
        pos = x + (j - 1.5) * width
        ax.bar(pos, values, width=width, label=label, color=color)
        for px, value in zip(pos, values):
            ax.text(px, value + 2, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, model_names)
    ax.set_ylabel("Fälle")
    ax.set_ylim(0, 160)
    ax.grid(axis="y", alpha=0.2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle("Technische und ergebnisbezogene Reparaturen und Regressionen", y=0.98)
    fig.legend(handles, legend_labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_plot(PLOTS["types"])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), sharey=True)
    fam_x = np.arange(len(BROAD_ORDER))
    for ax, model in zip(axes, MODEL_ORDER):
        for offset, role, label, color in ((-0.19, "base", "Ausgangsmodell", "#457b9d"), (0.19, "lora_v2", "LoRA v2", "#e76f51")):
            values = [
                100 * next(row["rate_per_1032"] for row in families if row["model_key"] == model and row["role"] == role and row["broad_family"] == family)
                for family in BROAD_ORDER
            ]
            ax.bar(fam_x + offset, values, width=0.38, label=label, color=color)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xticks(fam_x, [name.replace("/", "/\n") for name in BROAD_ORDER], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Anteil an 1.032 Fällen (%)")
    fig.suptitle("Breite explorative Fehlerfamilien nach Modellrolle")
    fig.text(0.5, 0.005, "Familien sind Multi-Label-Kategorien und daher nicht disjunkt.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_plot(PLOTS["families"])

    plt.figure(figsize=(10.0, 5.5))
    labels_x = []
    direct_values = []
    heuristic_values = []
    for model in MODEL_ORDER:
        for role in ("base", "lora_v2"):
            subset = [row for row in evidence if row["model_key"] == model and row["role"] == role]
            total = sum(row["label_assignments"] for row in subset)
            direct = sum(row["label_assignments"] for row in subset if row["evidence_level"] in {"E1", "E2"})
            short_model = {"qwen2b": "Qwen 3.5 2B", "llama3b": "Llama 3.2 3B", "qwen9b": "Qwen 3.5 9B"}[model]
            short_role = "Base" if role == "base" else "LoRA v2"
            labels_x.append(f"{short_model}\n{short_role}")
            direct_values.append(100 * direct / total)
            heuristic_values.append(100 * (total - direct) / total)
    xx = np.arange(len(labels_x))
    plt.bar(xx, direct_values, label="E1/E2", color="#2a9d8f")
    plt.bar(xx, heuristic_values, bottom=direct_values, label="E3/E4", color="#f4a261")
    for i, value in enumerate(direct_values):
        plt.text(i, value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=8)
        plt.text(i, value + heuristic_values[i] / 2, f"{heuristic_values[i]:.1f}%", ha="center", va="center", fontsize=8)
    plt.xticks(xx, labels_x, fontsize=8)
    plt.ylabel("Anteil der Labelzuweisungen (%)")
    plt.ylim(0, 112)
    plt.title("Direkte/technische versus heuristische Evidenzstufen")
    plt.legend(frameon=False, ncol=2, loc="upper center")
    plt.grid(axis="y", alpha=0.2)
    save_plot(PLOTS["evidence"])

    capped = q2_manifest["capped_case_summary"]
    capped_256 = sum(item["capped_cases"] for item in capped.values())
    capped_512 = sum(item["capped_again_512"] for item in capped.values())
    repetition = sum(item["continued_repetition"] for item in capped.values())
    terminated = sum(item["terminated_before_512"] for item in capped.values())
    new_correct = sum(item["newly_correct"] for item in capped.values())
    require((capped_256, capped_512, repetition, terminated, new_correct) == (2215, 2215, 2215, 0, 0), "Qwen 2B completion aggregate mismatch")
    names = ["Limit @256", "Erneut Limit @512", "Repetitionsregel", "Terminiert <512", "Neue Matches"]
    values = [capped_256, capped_512, repetition, terminated, new_correct]
    colors = ["#457b9d", "#e76f51", "#f4a261", "#2a9d8f", "#6a4c93"]
    plt.figure(figsize=(9.4, 5.2))
    bars = plt.bar(np.arange(len(names)), values, color=colors, width=0.65)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,}".replace(",", "."), ha="center", fontsize=9)
    plt.xticks(np.arange(len(names)), names)
    plt.ylabel("Beobachtungen über acht Qwen-2B-Base-Bedingungen")
    plt.title("Qwen 2B: Completionlimit- und Repetitionsdiagnostik")
    plt.ylim(0, 2450)
    plt.grid(axis="y", alpha=0.2)
    save_plot(PLOTS["completion"])


def build_tables_md(
    transitions: list[dict[str, Any]],
    types: list[dict[str, Any]],
    families: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    concrete: list[dict[str, Any]],
    alternatives: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = ["# Konservative Error Analysis: thesis-taugliche Tabellen", ""]
    lines += ["## Tabelle A: Outcome-Transitionen", "", md_table(
        [{**row, "ratio": f"{row['repair_to_regression_ratio']:.2f}"} for row in transitions],
        [("model_line", "Modelllinie"), ("repairs", "Reparaturen"), ("regressions", "Regressionen"), ("net_improvement", "Netto"), ("persistent_errors", "Persistent falsch"), ("stable_correct", "Stabil korrekt"), ("ratio", "Repair/Regression")],
    ), ""]
    lines += ["## Tabelle B: Reparatur- und Regressionstypen", "", md_table(
        types,
        [("model_line", "Modelllinie"), ("technical_repairs_nonexec_to_correct", "Nicht ausführbar → korrekt"), ("result_based_repairs_execwrong_to_correct", "Ausführbar falsch → korrekt"), ("technical_regressions_correct_to_nonexec", "Korrekt → nicht ausführbar"), ("result_based_regressions_correct_to_execwrong", "Korrekt → ausführbar falsch")],
    ), ""]
    family_md = []
    for row in families:
        family_md.append({**row, "rate": f_pct(100 * row["rate_per_1032"]), "incorrect_share": f_pct(100 * row["share_of_incorrect_predictions"])})
    lines += ["## Tabelle C: Breite explorative Fehlerfamilien", "", md_table(
        family_md,
        [("model_line", "Modelllinie"), ("role_label", "Rolle"), ("broad_family", "Familie"), ("unique_case_count", "Eindeutige Fälle"), ("rate", "Rate je 1.032"), ("incorrect_share", "Anteil falscher Predictions")],
    ), "", "Mehrfachzuweisungen sind möglich; Familien summieren sich daher nicht zu einer disjunkten Gesamtzahl.", ""]
    for action, title in (("repair", "Tabelle D: Häufigste konkrete E1/E2-Reparaturlabels"), ("regression", "Tabelle E: Häufigste konkrete E1/E2-Regressionslabels")):
        subset = [row for row in concrete if row["outcome_role"] == action]
        lines += [f"## {title}", "", md_table(
            subset,
            [("model_line", "Modelllinie"), ("rank", "Rang"), ("error_label", "Konkretes Label"), ("broad_family", "Breite Familie"), ("unique_outcome_cases_with_label", "Outcome-Fälle"), ("evidence_levels", "Evidenz")],
        ), "", "Die Zählung ist Multi-Label und keine kausale oder disjunkte Fehlerzählung.", ""]
    evidence_md = [{**row, "share": f_pct(100 * row["share_of_role_label_assignments"])} for row in evidence]
    lines += ["## Tabelle F: Evidenzstufen", "", md_table(
        evidence_md,
        [("model_line", "Modelllinie"), ("role_label", "Rolle"), ("evidence_level", "Evidenz"), ("label_assignments", "Labelzuweisungen"), ("share", "Anteil")],
    ), ""]
    alt_md = [{**row, "share_text": f_pct(100 * row["share"])} for row in alternatives]
    alt_md.append({"model_line": "Gesamt", "stable_correct_pairs": 1480, "different_sql_both_execution_match": 1000, "share_text": f_pct(1000 / 1480 * 100)})
    lines += ["## Tabelle G: Alternative valide SQL", "", md_table(
        alt_md,
        [("model_line", "Modelllinie"), ("stable_correct_pairs", "Stabil korrekte Paare"), ("different_sql_both_execution_match", "Unterschiedliche SQL"), ("share_text", "Anteil")],
    ), ""]
    compact = [
        {**row, "question_short": md_escape(row["question"], 90), "labels_short": md_escape(row["exploratory_error_labels"], 85)}
        for row in examples
    ]
    lines += ["## Tabelle H: Qualitative Beispiele", "", md_table(
        compact,
        [("example_type", "Typ"), ("model_line", "Modelllinie"), ("case_id", "Case-ID"), ("db_id", "DB"), ("question_short", "Frage"), ("labels_short", "E1/E2-Label bzw. Alternative")],
    ), "", f"Die vollständigen 15 Fallfelder einschließlich Schemaauszug und SQL stehen in `{OUT['examples']}`. Die Auswahl illustriert Befunde; sie ist keine Prävalenzstichprobe.", ""]
    return "\n".join(lines)


def build_thesis_text(
    transitions: list[dict[str, Any]],
    types: list[dict[str, Any]],
    alternatives: list[dict[str, Any]],
    fewshot: dict[str, Any],
    examples: list[dict[str, Any]],
) -> str:
    by_model = {row["model_key"]: row for row in transitions}
    type_by_model = {row["model_key"]: row for row in types}
    lines = [
        "# Thesis-taugliche Ergebnisformulierung: konservative Error Analysis",
        "",
        "## 1. Ziel und Methodik der Error Analysis",
        "",
        "Zur Untersuchung der Fehlerprofilveränderungen wurden sechs gepaarte Zero-Shot-Runs aus drei Modelllinien ausgewertet. Die Analyse umfasst 3.096 gepaarte Modelllinienfälle beziehungsweise 6.192 Predictions. Execution Match auf der vorhandenen Spider-Datenbankinstanz bildet das autoritative Korrektheitskriterium. Ergänzend wurde die eingefrorene deterministische Multi-Label-Diagnostik `project-local-sqlite-clause-fallback 1.0.0` verwendet. Deren feingranulare Komponentenlabels sind explorativ und nicht menschlich validiert.",
        "",
        "## 2. Fallübergänge durch LoRA",
        "",
    ]
    for model in MODEL_ORDER:
        row = by_model[model]
        lines.append(
            f"Bei {MODEL_LABELS[model]} standen {row['repairs']} Reparaturen {row['regressions']} Regressionen gegenüber. Daraus ergibt sich eine Nettoverbesserung von {row['net_improvement']} Fällen; {row['persistent_errors']} Fälle blieben falsch und {row['stable_correct']} Fälle blieben korrekt."
        )
    lines += ["", "## 3. Ausführbarkeit und Ergebnisrichtigkeit", ""]
    for model in MODEL_ORDER:
        row = type_by_model[model]
        lines.append(
            f"Für {MODEL_LABELS[model]} entfielen {row['technical_repairs_nonexec_to_correct']} Reparaturen auf den operationalen Übergang nicht ausführbar zu korrekt und {row['result_based_repairs_execwrong_to_correct']} auf ausführbar, aber ergebnisfalsch zu korrekt. Dem standen {row['technical_regressions_correct_to_nonexec']} technische und {row['result_based_regressions_correct_to_execwrong']} ergebnisbezogene Regressionen gegenüber."
        )
    lines += [
        "",
        "Die zweite Gruppe darf nicht als ausschließlich semantischer Fehler verstanden werden: Sie bezeichnet nur, dass die SQL technisch ausführbar war, auf der vorhandenen Datenbankinstanz jedoch kein korrektes Ergebnis erzeugte. Da in allen Modelllinien ein großer Anteil der Reparaturen in diese Gruppe fällt, erklärt verbesserte Ausführbarkeit allein den LoRA-Gewinn nicht.",
        "",
        "## 4. Exploratives Fehlerprofil",
        "",
        "Die breite, regelbasierte Aggregation deutet vor allem auf Veränderungen in Schema- und Projektionskomponenten sowie in Join-, Filter-, Aggregations- und weiterer Querylogik hin. E1/E2-Zuweisungen werden als direkte beziehungsweise technisch reproduzierbare Evidenz ausgewiesen; E3/E4 bleiben heuristische Hinweise. Wegen Multi-Label-Zuweisungen dürfen Kategorien weder addiert noch als disjunkte oder kausale Fehlerursachen interpretiert werden.",
        "",
        "## 5. Ausgabe- und Terminierungsverhalten",
        "",
        "In der Qwen-2B-Base-Sensitivitätsanalyse erreichten 2.215 Beobachtungen aus acht Bedingungen das 256-Token-Limit. Alle 2.215 erreichten auch bei 512 Tokens erneut das Limit, keine terminierte regulär vor 512 Tokens, keine erzeugte ein zusätzliches Execution Match und alle erfüllten die eingefrorene Repetitionsregel. Im primären Zero-Shot-Run waren davon lediglich 24 Fälle enthalten. Zugleich waren 142 der 241 Zero-Shot-Reparaturen bereits zuvor ausführbar, aber ergebnisfalsch. Der LoRA-Gewinn ist daher mit besserer Ausgabe- und Terminierungskontrolle vereinbar, kann jedoch nicht vollständig dadurch erklärt werden.",
        "",
        "## 6. Alternative valide SQL-Formulierungen",
        "",
        f"Von 1.480 stabil korrekten Paaren verwendeten 1.000 ({f_pct(1000 / 1480 * 100)}) nach der eingefrorenen Strukturklassifikation unterschiedliche SQL-Formulierungen, während beide Predictions auf der vorhandenen Datenbankinstanz execution-matchten. Dies verdeutlicht Grenzen von String Exact Match und Normalized Exact Match. Daraus folgt keine Garantie logischer Äquivalenz auf anderen Datenbankinstanzen.",
        "",
        "## 7. Few-Shot-Schadensmuster",
        "",
        f"Die vorhandene explorative Few-Shot-Schadensanalyse umfasst {fewshot['qwen2b_unique_condition_cases']} Qwen-2B- und {fewshot['llama3b_unique_condition_cases']} Llama-3B-Bedingung-Fall-Kombinationen, die unter Zero Shot korrekt und unter mindestens einer untersuchten Few-Shot-Bedingung falsch waren. Häufig beobachtete Muster umfassten zusätzliche Tabellen, zusätzliche beziehungsweise übermäßige Joins und heuristische Ähnlichkeit zur Demo-Struktur. Diese Muster sind Assoziationen; eine kausale Wirkung der Demonstration ist nicht nachgewiesen.",
        "",
        "## 8. Methodische Einschränkungen",
        "",
        "Es stand kein lokaler AST-Parser zur Verfügung. Die feingranularen Labels wurden nicht vollständig menschlich validiert, und die additive KI-Vorannotation wurde nicht als Goldstandard oder Prävalenzgrundlage verwendet. Komponentenunterschiede können koexistieren und müssen nicht die kausale Fehlerursache darstellen. Execution Match gilt nur für die vorhandene Datenbankinstanz. Cross-Model-Vergleiche bleiben aufgrund unterschiedlicher Modellfamilien methodisch Klasse B beziehungsweise B+.",
        "",
        "## Empfohlene Unterforschungsfrage",
        "",
        "> Welche Veränderungen des Fehlerprofils sowie des Ausgabe- und Terminierungsverhaltens lassen sich durch eine explorative regelbasierte Analyse der Ausgangs- und LoRA-feinabgestimmten Modelle beobachten?",
        "",
        "Die Ergebnisse beantworten diese Frage durch gepaarte Netto-Reparaturen und Regressionen, die operationale Trennung technischer und ergebnisbezogener Übergänge, die Qwen-2B-Terminierungsdiagnostik, explorative breite Fehlerfamilien, alternative valide SQL-Formulierungen sowie aggregierte Few-Shot-Schadensmuster.",
        "",
        "## Qualitative Illustration",
        "",
        f"Es wurden {len(examples)} ausgewogene Beispiele (je fünf pro Modelllinie) aus dem bestehenden Reviewpaket ausgewählt: fünf Reparaturen, fünf Regressionen, drei persistente Fehler und zwei alternative valide Formulierungen. Die vollständigen Fallangaben stehen in `{OUT['examples']}` und dienen ausschließlich der Illustration.",
    ]
    return "\n".join(lines)


def statements_md() -> str:
    lines = ["# Wissenschaftlich gestützte und nicht gestützte Aussagen", "", "## Gestützte thesis-taugliche Aussagen", ""]
    lines.extend(f"{i}. {text}" for i, text in enumerate(SUPPORTED_STATEMENTS, 1))
    lines += ["", "## Nicht gestützte Aussagen", ""]
    lines.extend(f"{i}. {text}" for i, text in enumerate(UNSUPPORTED_STATEMENTS, 1))
    lines += ["", "Alle Aussagen gelten für die eingefrorenen Runs und Artefakte dieses Projekts; die explorativen Labelbefunde sind nicht kausal."]
    return "\n".join(lines)


def limitations_md() -> str:
    lines = ["# Methodische Einschränkungen der konservativen Error Analysis", ""]
    lines.extend(f"{i}. {text}" for i, text in enumerate(LIMITATIONS, 1))
    lines += [
        "",
        "## Entscheidung zur menschlichen Review",
        "",
        "Für diese ausdrücklich explorative und konservative Thesis-Fassung müssen die 180 Fälle nicht vollständig manuell nachcodiert werden. Eine vollständige menschliche Prüfung wäre jedoch erforderlich, sobald die Taxonomie als menschlich validiert, als Goldstandard oder zur belastbaren Prävalenzschätzung bezeichnet werden soll.",
    ]
    return "\n".join(lines)


def audit_md(
    source_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    types: list[dict[str, Any]],
    families: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    concrete: list[dict[str, Any]],
    alternatives: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    fewshot: dict[str, Any],
    ai_summary: dict[str, Any],
    zero_runs: list[dict[str, Any]],
) -> str:
    trans_md = []
    for row in transitions:
        trans_md.append({**row, "ratio": f"{row['repair_to_regression_ratio']:.2f}", "repair_rate": f_pct(100 * row["repair_rate_per_1032"]), "regression_rate": f_pct(100 * row["regression_rate_per_1032"])})
    type_md = []
    for row in types:
        type_md.append({**row, "result_repair_share": f_pct(100 * row["result_based_share_of_repairs"]), "result_regression_share": f_pct(100 * row["result_based_share_of_regressions"])})
    overall_evidence = Counter()
    for row in evidence:
        overall_evidence[row["evidence_level"]] += row["label_assignments"]
    alt_total = sum(row["different_sql_both_execution_match"] for row in alternatives)
    family_excerpt = [
        {**row, "rate": f_pct(100 * row["rate_per_1032"])}
        for row in families
    ]
    source_table = md_table(source_rows, [("path", "Artefakt"), ("actual_sha256", "SHA256"), ("status", "Status")])
    lines = [
        "# Audit: Konservative finale Synthese der modellübergreifenden Error Analysis",
        "",
        f"**CONSERVATIVE-FINAL-ERROR-ANALYSIS-SYNTHESIS: {STATUS}**",
        "",
        "## 1. Scope und wissenschaftliche Einordnung",
        "",
        "Die Synthese ist rein additiv und basiert ausschließlich auf eingefrorenen Analyseartefakten. Sie ersetzt weder den Cross-Model-Abschlussaudit noch die Error Analysis v1. Es wurden keine Modelle, Adapter, Tokenizer, Retriever oder Parser geladen und keine Evaluation, Generation, Vollklassifikation oder menschliche Nachcodierung durchgeführt.",
        "",
        "Die Error Analysis ist eine explorative, deterministische, regelbasierte Multi-Label-Diagnostik. Direkt belastbar sind Execution Match, Execution Success und die gepaarten Übergänge. Technische Diagnosen wie SQLite-Fehler, leere SQLs, Completionlimits und Repetition sind reproduzierbar. Komponentenlabels zu Schema, Projektion und Querylogik bleiben explorativ.",
        "",
        "## 2. Source Integrity",
        "",
        "**SOURCE-INTEGRITY: PASS**",
        "",
        source_table,
        "",
        "Die beiden verbindlich vorgegebenen Manifest-Hashes stimmen exakt: Cross-Model `24b4dec…e37` und Parser/KI `60da5ad…2272`. Sämtliche weiteren verwendeten Hashes entsprechen den eingefrorenen Manifesten. Bestehende Dateien wurden nicht verändert.",
        "",
        "## 3. Datenbasis",
        "",
        f"Ausgewertet wurden {len(zero_runs)} autoritative Zero-Shot-Runs, 3.096 gepaarte Modelllinienfälle und 6.192 Predictions. Jede Modelllinie umfasst exakt 1.032 Spider-Dev-Fälle.",
        "",
        "## 4. Outcome-Transitionen",
        "",
        md_table(trans_md, [("model_line", "Modelllinie"), ("repairs", "Reparaturen"), ("regressions", "Regressionen"), ("net_improvement", "Netto"), ("persistent_errors", "Persistent"), ("stable_correct", "Stabil korrekt"), ("ratio", "Repair/Regression"), ("repair_rate", "Repair-Rate"), ("regression_rate", "Regression-Rate")]),
        "",
        "In allen drei Linien überstiegen Reparaturen die Regressionen. Der positive Nettoeffekt schließt neue Fehler nicht aus: Qwen 2B und Llama 3B weisen jeweils 90, Qwen 9B 83 Regressionen auf.",
        "",
        "## 5. Technische und ergebnisbezogene Übergänge",
        "",
        md_table(type_md, [("model_line", "Modelllinie"), ("technical_repairs_nonexec_to_correct", "Nicht ausführbar → korrekt"), ("result_based_repairs_execwrong_to_correct", "Ausführbar falsch → korrekt"), ("result_repair_share", "Anteil Ergebnis-Reparaturen"), ("technical_regressions_correct_to_nonexec", "Korrekt → nicht ausführbar"), ("result_based_regressions_correct_to_execwrong", "Korrekt → ausführbar falsch"), ("result_regression_share", "Anteil Ergebnis-Regressionen")]),
        "",
        "Die Bezeichnung ergebnisbezogen ist operational: Die SQL war ausführbar, lieferte auf der vorhandenen Instanz aber kein korrektes Ergebnis. Sie impliziert keinen ausschließlich semantischen Einzelgrund.",
        "",
        "## 6. Breite explorative Fehlerfamilien",
        "",
        md_table(family_excerpt, [("model_line", "Modelllinie"), ("role_label", "Rolle"), ("broad_family", "Familie"), ("unique_case_count", "Eindeutige Fälle"), ("rate", "Rate je 1.032")]),
        "",
        "Die Familien sind Multi-Label-Gruppen. Ein Fall kann mehreren Familien angehören; Summen über Familien dürfen daher größer als die Zahl falscher Predictions sein. Die Tabelle ist explorativ, nicht kausal.",
        "",
        "## 7. Konkrete E1/E2-Labels",
        "",
        "Fachliche Ranglisten verwenden ausschließlich konkrete E1/E2-Labels innerhalb von T1-Reparaturen beziehungsweise T2-Regressionen. Ausgeschlossen wurden `COMPLEX_MULTI_COMPONENT_ERROR`, `HEURISTIC_ONLY`, `MANUAL_REVIEW_REQUIRED`, `UNCLASSIFIED`, `PARSER_UNAVAILABLE` und `PARSER_DISAGREEMENT`.",
        "",
        f"Die vollständigen Ranglisten mit {len(concrete)} Zeilen stehen in `{OUT['labels']}`. Die Werte sind Multi-Label-Zählungen innerhalb der jeweiligen Outcome-Fälle und keine disjunkten Ursachen.",
        "",
        "## 8. Evidenzstufen",
        "",
        md_table([{"level": level, "count": overall_evidence[level], "share": f_pct(100 * overall_evidence[level] / sum(overall_evidence.values()))} for level in ("E1", "E2", "E3", "E4")], [("level", "Evidenz"), ("count", "Labelzuweisungen"), ("share", "Anteil")]),
        "",
        "Konkrete thesis-taugliche Komponentenbefunde stützen sich primär auf E1/E2. E3/E4 werden nur als heuristische Hinweise formuliert.",
        "",
        "## 9. Qwen-2B-Ausgabe- und Terminierungsverhalten",
        "",
        "Über acht Qwen-2B-Base-Bedingungen lagen 2.215 Completionlimit-Beobachtungen bei 256 Tokens vor. Alle 2.215 erreichten erneut 512 Tokens, keine terminierte regulär vor 512, kein zusätzlicher Execution Match entstand und alle erfüllten die Repetitionsregel. Im primären Zero-Shot-Run waren nur 24 Limitfälle enthalten; außerdem waren 142 von 241 Zero-Shot-Reparaturen bereits ausführbar, aber ergebnisfalsch. Outputkontrolle ist daher eine kompatible Teil-, aber keine vollständige Erklärung des LoRA-Gewinns.",
        "",
        "## 10. Alternative valide SQL-Formulierungen",
        "",
        f"{alt_total} von 1.480 stabil korrekten Paaren ({f_pct(100 * alt_total / 1480)}) verwendeten nach der eingefrorenen Strukturklassifikation unterschiedliche SQLs, während beide auf der vorhandenen Instanz execution-matchten. Das belegt Grenzen textbasierter Metriken, aber keine universelle logische Äquivalenz.",
        "",
        "## 11. Few-Shot-Schadensmuster",
        "",
        f"Die vorhandene Analyse umfasst {fewshot['qwen2b_unique_condition_cases']} Qwen-2B- und {fewshot['llama3b_unique_condition_cases']} Llama-3B-Bedingung-Fall-Kombinationen. Häufige explorative Muster waren zusätzliche Tabellen, Over-Joining und Demo-Strukturähnlichkeit. Es ist keine kausale Interpretation zulässig.",
        "",
        "## 12. KI-Vorannotation und menschliche Validierung",
        "",
        f"Die additive KI-Vorannotation enthält {ai_summary['cases']} Fälle: {ai_summary['reviewed']} `REVIEWED`, {ai_summary['ambiguous']} `AMBIGUOUS`, {ai_summary['high']} HIGH-, {ai_summary['medium']} MEDIUM- und {ai_summary['low']} LOW-Confidence sowie {ai_summary['high_priority']} Fälle mit hoher menschlicher Reviewpriorität. Sie wurde nicht zur Prävalenzneuberechnung verwendet und ist kein Goldstandard.",
        "",
        "Für diese konservative, ausdrücklich explorative Thesis-Fassung ist keine vollständige manuelle Bearbeitung der 180 Fälle erforderlich. Eine solche wäre erforderlich, falls die Taxonomie als menschlich validiert, Goldstandard oder belastbare Prävalenzklassifikation bezeichnet werden soll.",
        "",
        "## 13. Qualitative Beispiele",
        "",
        f"Aus dem bestehenden Paket wurden exakt {len(examples)} klare Illustrationen gewählt: fünf Reparaturen, fünf Regressionen, drei persistente Fehler und zwei alternative valide Formulierungen; jede Modelllinie ist fünfmal vertreten. Alle nicht-alternativen Beispiele besitzen mindestens ein konkretes E1/E2-Label. Vollständige Angaben stehen in `{OUT['examples']}`.",
        "",
        "## 14. Forschungsfrage",
        "",
        "> Welche Veränderungen des Fehlerprofils sowie des Ausgabe- und Terminierungsverhaltens lassen sich durch eine explorative regelbasierte Analyse der Ausgangs- und LoRA-feinabgestimmten Modelle beobachten?",
        "",
        "Antwort: LoRA führte in allen drei Modelllinien zu positiven Netto-Reparaturen, zugleich aber zu neuen Regressionen. Die Gewinne umfassen sowohl technische als auch bereits ausführbare, ergebnisfalsche Ausgangsfälle. Breite regelbasierte Familien deuten auf Änderungen in Schema-, Projektions- und Querystrukturkomponenten hin. Bei Qwen 2B spricht die Tokensensitivität für persistente Repetition statt bloß verspäteter Terminierung. Alternative execution-korrekte Formulierungen begrenzen textbasierte Exaktheitsmetriken; Few-Shot-Schadensmuster bleiben explorative Assoziationen.",
        "",
        "## 15. Methodische Einschränkungen",
        "",
    ]
    lines.extend(f"{i}. {text}" for i, text in enumerate(LIMITATIONS, 1))
    lines += [
        "",
        "## 16. Freigabe",
        "",
        f"**CONSERVATIVE-FINAL-ERROR-ANALYSIS-SYNTHESIS: {STATUS}**",
        "",
        "Die direkten Outcome-Aussagen sind belastbar; technische Diagnosen sind reproduzierbar; Komponenten- und Few-Shot-Labels bleiben explorativ. Experimente fehlen für diese Synthese nicht. Ein Rerun ist nicht erforderlich.",
        "",
        "**EXPERIMENTE FEHLEND: NEIN**  ",
        "**RERUN ERFORDERLICH: NEIN**  ",
        "**BESTEHENDE DATEIEN VERÄNDERT: NEIN**",
    ]
    return "\n".join(lines)


def output_file_records(include_audit: bool = True) -> list[dict[str, Any]]:
    paths = [value for key, value in OUT.items() if key != "manifest" and (include_audit or key != "audit")]
    for base in PLOTS.values():
        paths.extend([base + ".png", base + ".pdf"])
    paths.append("scripts/build_conservative_final_error_analysis_synthesis.py")
    result = []
    for relative in paths:
        path = ROOT / relative
        require(path.is_file(), f"Expected generated file missing: {relative}")
        result.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    return result


def main() -> None:
    # The script file itself is additive, but every data/report/plot target must
    # be absent so no prior output can be overwritten.
    for relative in OUT.values():
        require(not (ROOT / relative).exists(), f"Target already exists: {relative}")
    for base in PLOTS.values():
        require(not (ROOT / (base + ".png")).exists(), f"Target already exists: {base}.png")
        require(not (ROOT / (base + ".pdf")).exists(), f"Target already exists: {base}.pdf")

    sources = source_integrity()
    cases = read_csv("audits/derived/cross_model_zero_shot_error_analysis_cases_20260716.csv")
    long_rows = read_csv("audits/derived/cross_model_zero_shot_error_labels_long_20260716.csv")
    transition_raw = read_csv("audits/derived/cross_model_zero_shot_error_transition_summary_20260716.csv")
    alternatives_raw = read_csv("audits/derived/cross_model_zero_shot_alternative_valid_sql_20260716.csv")
    fewshot_raw = read_csv("audits/derived/cross_model_fewshot_harm_error_analysis_20260716.csv")
    pre_rows = read_csv("audits/derived/cross_model_zero_shot_error_ai_preannotation_20260716.csv")
    priority_rows = read_csv("audits/derived/cross_model_zero_shot_error_human_review_priority_20260716.csv")
    error_manifest = read_json("audits/cross_model_zero_shot_error_analysis_manifest_20260716.json")
    parser_manifest = read_json("audits/local_sql_ast_preflight_and_ai_error_preannotation_manifest_20260716.json")
    cross_manifest = read_json("audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json")
    q2_manifest = read_json("audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json")

    require(len(cases) == 3096, f"Expected 3,096 paired cases, got {len(cases)}")
    require(len(long_rows) == 31279, f"Expected 31,279 label assignments, got {len(long_rows)}")
    require(len(pre_rows) == 180 and len(priority_rows) == 180, "AI preannotation/review-priority size mismatch")
    require(error_manifest["paired_model_line_cases"] == 3096 and error_manifest["predictions"] == 6192, "Error manifest scope mismatch")
    require(parser_manifest["parser_decision"]["ast_sensitivity_conducted"] is False, "Unexpected AST sensitivity result")
    require(q2_manifest["official_mainline_max_new_tokens"] == 256 and q2_manifest["sensitivity_max_new_tokens"] == 512, "Qwen 2B sensitivity scope mismatch")

    transitions, types = derive_transitions(transition_raw)
    families = derive_families(long_rows, cases)
    evidence = derive_evidence(long_rows)
    concrete = derive_concrete_labels(long_rows, cases)
    alternatives = alternative_summary(alternatives_raw)
    fewshot = fewshot_summary(fewshot_raw)
    examples = example_rows(cases, long_rows, pre_rows, alternatives_raw)

    ai = parser_manifest["ai_assisted_preannotation"]
    ai_summary = {
        "cases": ai["cases"],
        "reviewed": ai["status_distribution"]["REVIEWED"],
        "ambiguous": ai["status_distribution"]["AMBIGUOUS"],
        "high": ai["confidence_distribution"]["HIGH"],
        "medium": ai["confidence_distribution"]["MEDIUM"],
        "low": ai["confidence_distribution"]["LOW"],
        "high_priority": ai["review_priority_distribution"]["HIGH"],
        "human_validated": ai["human_validated"],
        "gold_standard": ai["gold_standard"],
    }
    require(ai_summary == {"cases": 180, "reviewed": 31, "ambiguous": 149, "high": 1, "medium": 30, "low": 149, "high_priority": 164, "human_validated": False, "gold_standard": False}, "AI preannotation summary mismatch")

    zero_runs = [row for row in cross_manifest["runs"] if row["condition"] == "zero_shot" and row["model"] in MODEL_ORDER and row["role"] in {"base", "lora_v2"}]
    require(len(zero_runs) == 6, f"Expected six zero-shot runs, got {len(zero_runs)}")
    require(all(row["case_count"] == N and row["unique_case_count"] == N for row in zero_runs), "Zero-shot run completeness mismatch")

    write_csv(OUT["transitions"], transitions)
    write_csv(OUT["types"], types)
    write_csv(OUT["families"], families)
    write_csv(OUT["evidence"], evidence)
    write_csv(OUT["labels"], concrete)
    write_csv(OUT["examples"], examples)

    make_plots(transitions, types, families, evidence, q2_manifest)
    write_text(OUT["tables"], build_tables_md(transitions, types, families, evidence, concrete, alternatives, examples))
    write_text(OUT["text"], build_thesis_text(transitions, types, alternatives, fewshot, examples))
    write_text(OUT["statements"], statements_md())
    write_text(OUT["limitations"], limitations_md())

    # Recheck every frozen source after all additive output generation.
    source_integrity()
    write_text(OUT["audit"], audit_md(sources, transitions, types, families, evidence, concrete, alternatives, examples, fewshot, ai_summary, zero_runs))

    new_files = output_file_records(include_audit=True)
    manifest = {
        "audit_status": STATUS,
        "date": DATE,
        "classification": "conservative thesis-ready synthesis of frozen cross-model error analysis",
        "read_only_against_existing_artifacts": True,
        "existing_files_modified": False,
        "source_integrity": "PASS",
        "source_artifacts": sources,
        "source_manifests": {
            "cross_model": SOURCES["audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json"],
            "error_analysis": SOURCES["audits/cross_model_zero_shot_error_analysis_manifest_20260716.json"],
            "parser_ai": SOURCES["audits/local_sql_ast_preflight_and_ai_error_preannotation_manifest_20260716.json"],
            "qwen2b_sensitivity": SOURCES["audits/qwen35_2b_base_maxnew256_vs_512_sensitivity_manifest_20260716.json"],
        },
        "analysis_script": {
            "path": "scripts/build_conservative_final_error_analysis_synthesis.py",
            "sha256": sha256(ROOT / "scripts/build_conservative_final_error_analysis_synthesis.py"),
        },
        "scope": {"model_lines": 3, "zero_shot_runs": 6, "paired_cases": 3096, "predictions": 6192, "cases_per_model_line": 1032},
        "six_zero_shot_runs": [
            {
                "model": row["model"],
                "role": row["role"],
                "run_id": row["run_id"],
                "csv_path": row["csv_path"],
                "csv_sha256": row["csv_sha256"],
                "config_path": row["config_path"],
                "config_sha256": row["config_sha256"],
                "cases": row["case_count"],
            }
            for row in zero_runs
        ],
        "transitions": transitions,
        "repair_regression_types": types,
        "broad_error_families": {
            "mapping": BROAD_FAMILY,
            "counts": families,
            "multi_label": True,
            "inference_class": "EXPLORATIVE",
        },
        "meta_labels": {"excluded_from_technical_rankings": sorted(META_LABELS), "included_only_as_analysis_limit": True},
        "evidence_levels": {"counts": evidence, "primary_for_concrete_claims": ["E1", "E2"], "heuristic_only": ["E3", "E4"]},
        "concrete_e1_e2_rankings": {"rows": len(concrete), "top_n_per_model_and_outcome": 10, "outcomes": ["T1 repair", "T2 regression"]},
        "qualitative_selection": {
            "rules": [
                "existing 180-case package only",
                "five examples per model line",
                "five repairs, five regressions, three persistent errors, two alternative-valid formulations",
                "non-T4 examples require at least one concrete E1/E2 label",
                "illustration only; no prevalence inference",
            ],
            "selected": [{"model_key": row["model_key"], "case_id": row["case_id"], "example_type": row["example_type"], "evidence_levels": row["evidence_levels"]} for row in examples],
        },
        "qwen2b_sensitivity_integration": {
            "completion_limit_observations_256": 2215,
            "reached_512_again": 2215,
            "terminated_before_512": 0,
            "new_execution_matches": 0,
            "repetition_rule": 2215,
            "zero_shot_limit_observations": 24,
            "official_mainline_remains_max_new_tokens": 256,
            "interpretation": "Output control is a compatible partial explanation, not a complete explanation of zero-shot LoRA gain.",
        },
        "alternative_valid_sql": {"stable_correct_pairs": 1480, "different_sql_both_execution_match": 1000, "share": 1000 / 1480, "by_model": alternatives, "equivalence_scope": "current database instance only"},
        "fewshot_harm": fewshot,
        "ai_preannotation": {**ai_summary, "used_for_prevalence": False, "human_review_needed_for_exploratory_thesis_version": False, "human_review_needed_to_claim_validated_taxonomy": True},
        "methodological_limitations": LIMITATIONS,
        "supported_thesis_statements": SUPPORTED_STATEMENTS,
        "unsupported_statements": UNSUPPORTED_STATEMENTS,
        "experiments_missing": False,
        "rerun_required": False,
        "new_files": new_files,
        "manifest_self_hash": None,
        "manifest_self_hash_note": "Computed externally after write to avoid recursive self-hash.",
    }
    write_text(OUT["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps({"status": STATUS, "source_integrity": "PASS", "new_files_excluding_manifest": len(new_files), "manifest": OUT["manifest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
