#!/usr/bin/env python3
"""Read-only project-completion evidence matrix for the final NL2SQL study.

The module intentionally has no write path.  The complete 24-run analysis imports
``build_gap_rows`` and writes its returned rows to new, additive audit artifacts.
Running this file directly prints the matrix as JSON.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def present(path: str) -> bool:
    return (ROOT / path).is_file()


def evidence(paths: list[str]) -> str:
    return "; ".join(f"{path} ({'present' if present(path) else 'missing'})" for path in paths)


def row(
    area: str,
    requirement: str,
    artifacts: list[str],
    status: str,
    mandatory: bool,
    recommended: bool,
    optional: bool,
    no_longer_required: bool,
    next_step: str,
    priority: str,
) -> dict[str, Any]:
    return {
        "area": area,
        "requirement": requirement,
        "existing_artifact": "; ".join(artifacts),
        "status": status,
        "evidence": evidence(artifacts) if artifacts else "No frozen artifact identified",
        "mandatory_before_thesis_completion": mandatory,
        "recommended": recommended,
        "optional": optional,
        "no_longer_required": no_longer_required,
        "priority": priority,
        "concrete_next_step": next_step,
    }


def build_gap_rows() -> list[dict[str, Any]]:
    cross_audit = "audits/audit_cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_20260716.md"
    cross_manifest = "audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json"
    conservative = "audits/audit_conservative_final_error_analysis_synthesis_20260716.md"
    current_audit = "audits/audit_complete_24_lora_run_error_analysis_and_project_gap_review_20260716.md"
    rq_matrix = "audits/derived/final_research_questions_results_matrix_20260716.md"
    rows: list[dict[str, Any]] = []

    def add(area: str, requirement: str, artifacts: list[str], status: str, next_step: str = "None", priority: str = "complete") -> None:
        rows.append(row(
            area, requirement, artifacts, status,
            status == "ZWINGEND OFFEN", status == "EMPFOHLEN OFFEN",
            status == "OPTIONAL", status == "NICHT MEHR ERFORDERLICH",
            next_step, priority,
        ))

    # A. Research design
    add("A. Forschungsdesign", "Finale Hauptforschungsfrage und Unterforschungsfragen", [rq_matrix, cross_audit], "ABGESCHLOSSEN")
    add("A. Forschungsdesign", "Hypothesen und Operationalisierung", [rq_matrix, cross_audit], "ABGESCHLOSSEN")
    add("A. Forschungsdesign", "Forschungsfragen-Ergebnis-Matrix", [rq_matrix], "ABGESCHLOSSEN")
    add("A. Forschungsdesign", "Abgrenzung konfirmatorisch versus explorativ", [cross_audit, conservative, current_audit], "ABGESCHLOSSEN")

    # B. Data
    add("B. Datensatze", "Trainingsdatensatz old25k und MixedVal2500-v2", ["reports/analysis_validation_dataset_and_sql_loss_strategy_qwen35_2b_9b_20260712.md", "audits/audit_llama32_3b_instruct_lora_v2_training_posthoc_and_release_20260714.md"], "ABGESCHLOSSEN")
    add("B. Datensatze", "Spider Dev, Leakagefreiheit und Datensatzhashes", [cross_manifest, "audits/audit_cross_model_zero_shot_error_analysis_20260716.md"], "ABGESCHLOSSEN")
    add("B. Datensatze", "Testset-Pfaddrift dokumentiert", ["audits/audit_cross_model_zero_shot_error_analysis_20260716.md", current_audit], "ABGESCHLOSSEN")
    add("B. Datensatze", "Spider Test als nicht verwendetes, ungelabeltes finales Benchmarkset dokumentieren", [], "ZWINGEND OFFEN", "Im Methodikkapitel explizit festhalten, dass ausschließlich Spider Dev mit verfügbaren Goldlabels evaluiert wurde.", "high")

    # C. Training
    add("C. Training", "Drei finale LoRA-v2-Trainingslinien", [cross_manifest, "audits/audit_qwen35_9b_v2_training_posthoc_plots_completion_20260713.md", "audits/audit_llama32_3b_instruct_lora_v2_training_posthoc_and_release_20260714.md"], "ABGESCHLOSSEN")
    add("C. Training", "Hyperparameter, Seed, Early Stopping und Checkpointauswahl", [cross_manifest, "audits/audit_qwen35_9b_v2_training_posthoc_plots_completion_20260713.md"], "ABGESCHLOSSEN")
    add("C. Training", "Trainingslogs, Adapterhashes, Hardware und Laufzeiten", [cross_manifest, "audits/audit_qwen35_9b_v2_training_posthoc_plots_completion_20260713.md", "audits/audit_llama32_3b_instruct_lora_v2_training_posthoc_and_release_20260714.md"], "ABGESCHLOSSEN")

    # D. Evaluation
    add("D. Evaluation", "48 Hauptlaufe: 24 Ausgangsmodelle und 24 LoRA-v2", [cross_manifest, "audits/derived/cross_model_complete_48_run_results_20260716.csv"], "ABGESCHLOSSEN")
    add("D. Evaluation", "EMA, ESR und Textmetriken reproduziert", [cross_audit, cross_manifest], "ABGESCHLOSSEN")
    add("D. Evaluation", "Retrievaltraces, Gate-Semantik, Static-Demo und Structure", [cross_manifest], "ABGESCHLOSSEN")
    add("D. Evaluation", "Tokenlimits und Qwen-2B-Sensitivitat", ["audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260716.md"], "ABGESCHLOSSEN")

    # E. Statistics
    add("E. Statistik", "McNemar, Holm, Bootstrap und Effektgroßen", [cross_audit, cross_manifest], "ABGESCHLOSSEN")
    add("E. Statistik", "Vergleichbarkeitsklassen und multiple Testfamilien", [cross_audit, conservative], "ABGESCHLOSSEN")

    # F. Error analysis
    add("F. Error Analysis", "Zero-Shot Ausgangsmodell-versus-LoRA", ["audits/audit_cross_model_zero_shot_error_analysis_20260716.md"], "ABGESCHLOSSEN")
    add("F. Error Analysis", "Vollstandige 24-LoRA-Run-Analyse", [current_audit], "ABGESCHLOSSEN")
    add("F. Error Analysis", "Qwen-2B-Terminierung und Few-Shot-Schaden", ["audits/audit_qwen35_2b_base_maxnew256_vs_512_sensitivity_20260716.md", "audits/audit_conservative_final_error_analysis_synthesis_20260716.md"], "ABGESCHLOSSEN")
    add("F. Error Analysis", "Qualitative Beispiele und methodische Einschrankungen", ["audits/derived/complete_24_lora_run_qualitative_examples_20260716.csv", current_audit], "ABGESCHLOSSEN")
    add("F. Error Analysis", "Vollstandige menschliche Annotation", [], "NICHT MEHR ERFORDERLICH", "Als Limitation benennen; keine neue Vollannotation fur den definierten Abschlussumfang starten.", "none")
    add("F. Error Analysis", "AST-Parser-Nachinstallation", ["audits/audit_local_sql_ast_preflight_and_ai_error_preannotation_20260716.md"], "NICHT MEHR ERFORDERLICH", "Den dokumentierten Clause-Fallback und seine Grenzen berichten.", "none")

    # G. Reproducibility
    add("G. Reproduzierbarkeit", "Finaler Git-Commit, sauberer Git-Status und Git-Tag", [], "ZWINGEND OFFEN", "Arbeitsbaum prufen, finale Artefakte committen und einen unveranderlichen Abgabe-Tag setzen.", "high")
    add("G. Reproduzierbarkeit", "Python-, Paket-, CUDA-, GPU- und Betriebssystem-Freeze", [], "ZWINGEND OFFEN", "Versionen und Hardware in einem finalen Reproduzierbarkeitsmanifest einfrieren.", "high")
    add("G. Reproduzierbarkeit", "Snapshots, Adapter-, Daten-, Index-, Config- und Run-Hashes", [cross_manifest], "ABGESCHLOSSEN")
    add("G. Reproduzierbarkeit", "Read-only-Reproduktionsskripte", ["scripts/analyze_cross_model_complete_8x8_synthesis.py", "scripts/analyze_cross_model_zero_shot_error_taxonomy.py", "scripts/analyze_complete_24_lora_run_error_profiles.py"], "ABGESCHLOSSEN")
    add("G. Reproduzierbarkeit", "Externe Backup- und Abgabeversion", [], "ZWINGEND OFFEN", "Projekt, finale PDF und Hashmanifeste auf mindestens zwei getrennte Speicherorte sichern.", "high")

    # H. Thesis artifacts
    for requirement in [
        "Theorie- und Forschungsstand", "Methodikkapitel", "Ergebniskapitel", "Diskussion und Limitationen",
        "Fazit und Abstract", "Tabellen-, Abbildungs- und Abkurzungsverzeichnis", "Anhang", "Literaturverzeichnis und Zitierprufung",
    ]:
        add("H. Thesis-Artefakte", requirement, ["audits/derived/complete_24_lora_run_error_analysis_thesis_ready_text_20260716.md"], "ZWINGEND OFFEN", "Thesis-ready Evidenz in den finalen Fließtext integrieren und redaktionell prufen.", "high")

    # I. Formal final checks
    for requirement in [
        "Zahlenkonsistenz, Prozent/Prozentpunkte und Rundung", "Abbildungsqualitat und Tabellenreferenzen",
        "Quellen- und Hochschulformatprufung", "PDF-Sichtprufung", "Backup und finale Abgabeversion",
    ]:
        add("I. Formale Abschlusskontrolle", requirement, [], "ZWINGEND OFFEN", "Nach Fertigstellung der Thesis einen dokumentierten finalen Konsistenz- und PDF-Check durchfuhren.", "high")

    # Explicitly closed experimental temptations.
    for requirement in [
        "Weitere Modellruns", "Neue Gate-Schwellen", "Weitere Tokenlimits", "Neue Retrievalpools",
        "Erneute Trainings", "Zusatzliche Spider-Dev-Optimierung",
    ]:
        add("I. Formale Abschlusskontrolle", requirement, [cross_manifest], "NICHT MEHR ERFORDERLICH", "Keine weiteren Spider-Dev-Experimente starten.", "none")
    return rows


def environment_snapshot() -> dict[str, Any]:
    """Collect read-only, non-network environment facts for the final manifest."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # pragma: no cover - defensive audit path
            return f"unavailable: {exc!r}"

    return {
        "python": run("python3", "--version"),
        "git_status": run("git", "status", "--short"),
        "git_commit": run("git", "rev-parse", "HEAD"),
        "os_release": run("uname", "-a"),
        "nvidia_smi": run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    }


def main() -> None:
    print(json.dumps({"rows": build_gap_rows(), "environment": environment_snapshot()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
