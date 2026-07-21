#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits/audit_dynamic_fewshot_k3_implementation_and_preflight_20260717.md"
MANIFEST = ROOT / "audits/dynamic_fewshot_k3_implementation_and_preflight_manifest_20260717.json"
DOCS = ROOT / "docs/final_project_documentation_20260717_k3_extension_v2"
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
DEPENDENCIES = ROOT / "audits/derived/dynamic_k3_implementation_dependencies_20260717.csv"
PROMPTS = ROOT / "audits/derived/dynamic_k3_prompt_preflight_20260717.csv"
SUMMARY = ROOT / "audits/derived/dynamic_k3_prompt_preflight_summary_20260717.json"
RETRIEVAL = ROOT / "audits/derived/dynamic_k3_retrieval_selection_validation_20260717.csv"
VALIDATION = ROOT / "audits/derived/dynamic_k3_preflight_validation_20260717_v2.json"
IDENTITY = ROOT / "audits/derived/dynamic_k3_k1_reference_identity_20260717_v2.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, text: str) -> None:
    require(not path.exists(), f"Refusing to overwrite existing additive file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_group_table(groups: list[dict[str, Any]]) -> str:
    lines = [
        "| Modell | Rolle | Bedingung | Min | Mittel | Median | p95 | p99 | Max | Truncations | k=3 | k=0 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        lines.append(
            "| {model_key} | {role} | {condition} | {minimum} | {mean:.2f} | "
            "{median:.1f} | {p95:.2f} | {p99:.2f} | {maximum} | "
            "{prompt_truncations} | {fewshot_cases} | {fallback_cases} |".format(**row)
        )
    return "\n".join(lines)


def csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    require(bool(rows), "Cannot serialize empty CSV")
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    require(not AUDIT.exists(), f"Refusing to overwrite: {AUDIT}")
    require(not MANIFEST.exists(), f"Refusing to overwrite: {MANIFEST}")
    require(not DOCS.exists(), f"Refusing to overwrite documentation directory: {DOCS}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    matrix = read_csv(MATRIX)
    dependencies = read_csv(DEPENDENCIES)
    prompt_rows = read_csv(PROMPTS)
    retrieval_rows = read_csv(RETRIEVAL)
    identity_rows = read_csv(IDENTITY)
    require(summary["status"] == "BLOCKED-BY-PROMPT-TRUNCATION", "Unexpected summary status")
    require(validation["status"] == "BLOCKED-BY-PROMPT-TRUNCATION", "Unexpected validation status")
    require(len(matrix) == 36, "Expected 36 configs")
    require(len(prompt_rows) == 37152, "Expected 37152 prompt rows")
    require(len(retrieval_rows) == 1032, "Expected 1032 retrieval rows")
    require(all(row["status"] == "PASS" for row in identity_rows), "k1 reference identity failed")

    affected_case_ids = sorted({row["case_id"] for row in prompt_rows if row["would_truncate"] == "1"})
    actual_k_counts = Counter(row["actual_k"] for row in prompt_rows)
    maximum = max(int(row["prompt_tokens"]) for row in prompt_rows)
    gate_counts: dict[str, tuple[int, int]] = {}
    for condition in (
        "top3_gate070",
        "top3_gate085",
        "structure_top3_gate070",
        "structure_top3_gate085",
    ):
        rows = [
            row for row in prompt_rows
            if row["model_key"] == "qwen2b" and row["role"] == "base" and row["condition"] == condition
        ]
        gate_counts[condition] = (
            sum(row["actual_k"] == "3" for row in rows),
            sum(row["actual_k"] == "0" for row in rows),
        )

    dependency_lines = [
        "| Komponente | Bestehendes Verhalten | k3-Aenderung | Additiver Pfad | Risiko |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in dependencies:
        dependency_lines.append(
            f"| {row['component']} | {row['k1_behavior']} | {row['required_k3_change']} | "
            f"`{row['additive_implementation_path']}` | {row['risk']} |"
        )

    audit_text = f"""# Audit: Dynamic-Few-Shot-k=3-Implementierung und Feasibility-Preflight

**DYNAMIC-FEWSHOT-K3-EXTENSION: BLOCKED-BY-PREFLIGHT**

```text
DYNAMIC-K3-IMPLEMENTATION-PREFLIGHT: BLOCKED-BY-PROMPT-TRUNCATION
Existing files modified: NEIN
New implementation created: JA
36 configs created: JA (36/36)
Full runs started: NEIN
Affected config-case rows: {summary['total_prompt_truncations']}
Affected unique Spider-Dev cases: {len(affected_case_ids)}
Required methodological decision: max_input_tokens/Demodarstellung/k1-k3-Kontrollmatrix
```

Datum: 2026-07-17. Der Writer-Check war leer. Es wurden keine bestehenden Dateien veraendert, keine generative Evaluation, kein Training und keine Modellinferenz gestartet. BGE und Tokenizer wurden ausschliesslich offline aus vorhandenen lokalen Caches fuer Retrieval- und Promptmaterialisierung geladen.

## Entscheidung

Die harte Stop-Regel ist ausgeloest: {summary['total_prompt_truncations']} von {summary['prompt_rows']} Config-Fall-Prompts wuerden bei `max_input_tokens=2048` truncieren. Der laengste Prompt umfasst {maximum} Tokens. Deshalb wurden die 36 Vollruns nicht freigegeben und nicht gestartet. Die 48 bestehenden Runs bleiben unveraendert die **PRE-K3 AUTHORITATIVE BASELINE**; eine 84-Run-Analyse existiert nicht.

## Audit der k=1-Abhaengigkeiten

{chr(10).join(dependency_lines)}

Die bestehende Auswahl- und Promptlogik verarbeitet bereits Listen mit mehr als einer Demonstration. Nicht kompatibel war die k=1-Gatefunktion, weil sie nur den ersten BGE-Score thresholdete. Die additive Runner-Version implementiert daher `set_min_similarity`, erzwingt genau drei unterschiedliche Demos und persistiert Score-Minimum, -Maximum, -Mittelwert sowie alle Einzelscores.

## Additive Implementierung

- Runner: `src/06_batch_run_dynamic_k3_v1.py`
- Retriever: `src/retrieval_utils_dynamic_k3_v1.py`
- Configgenerator: `scripts/prepare_dynamic_k3_extension_20260717.py`
- Promptpreflight: `scripts/preflight_dynamic_k3_extension_20260717.py`
- Nachvalidator: `scripts/validate_dynamic_k3_preflight_20260717_v2.py`
- Tests: `tests/test_dynamic_k3_v1.py` (5/5 PASS)

Der Runner pinnt die drei autoritativen Modellrevisionen, verlangt den absoluten `.venv_flash`-Interpreter, verweigert Vollruns ohne CUDA/Flash Attention, verwendet kollisionsfreie k3-Prefixe und persistiert Runtime-, Modell-, Adapter-, Testset- und Retrievalprovenienz. Die bestehende `structure_rerank_v2`-Heuristik wurde nicht geaendert.

## Configmatrix

36/36 Configs wurden aus den 36 autoritativen k=1-Zeilen von `audits/derived/cross_model_complete_48_run_results_20260716.csv` erzeugt. Zulaessige technische Aenderungen sind `k: 1 -> 3`, bei Gates `similarity_only -> set_min_similarity`, immutable Modellrevision und additive Outputidentitaet. Nicht beabsichtigte Diffs: 0.

## Retrievalvalidierung

- Pool/Index: 6.960 Beispiele, `BAAI/bge-large-en-v1.5`
- Testfaelle: 1.032
- Top-3 mit drei unterschiedlichen IDs: 1.032/1.032
- Structure-Top-3 mit drei unterschiedlichen IDs: 1.032/1.032
- Leakage-/Zieloverlap: 0
- Unerwartetes `actual_k` 1 oder 2: 0
- k=1-Referenzidentitaet: 36/36 Runs, jeweils 1.032/1.032 erste Demo-IDs und BGE-Scores innerhalb 1e-5
- Base-/LoRA-Prompt-Hashabweichungen: 0
- Gate-Fallback-Prompt-Hashabweichungen: 0

## Gateverteilungen

| Bedingung | k=3 akzeptiert | Zero-Shot-Fallback |
| --- | ---: | ---: |
| Top-3 Gate 0.70 | {gate_counts['top3_gate070'][0]} | {gate_counts['top3_gate070'][1]} |
| Top-3 Gate 0.85 | {gate_counts['top3_gate085'][0]} | {gate_counts['top3_gate085'][1]} |
| Structure Top-3 Gate 0.70 | {gate_counts['structure_top3_gate070'][0]} | {gate_counts['structure_top3_gate070'][1]} |
| Structure Top-3 Gate 0.85 | {gate_counts['structure_top3_gate085'][0]} | {gate_counts['structure_top3_gate085'][1]} |

Die Verteilungen sind modell- und rollenunabhaengig, weil Retrieval und Set-Gate vor der Modellgeneration identisch sind. `actual_k` ueber alle 37.152 Zeilen: k=3 {actual_k_counts['3']}, k=0 {actual_k_counts['0']}.

## Promptlaengen

{markdown_group_table(summary['groups'])}

Ein gemeinsames Inputlimit fuer 100 % truncationfreie Abdeckung muesste mindestens {maximum} Tokens zulassen. Das ist eine Feasibility-Grenze, keine selbststaendige Empfehlung fuer einen neuen Hauptlinienwert.

## Methodische Optionen

1. 2.048 Tokens mit Truncation. Dies verletzt die vorab definierte Stop-Regel und ist fuer die geplante Hauptmatrix nicht freigegeben.
2. Hoeheres Inputlimit, mindestens entsprechend dem jeweiligen Gruppenmaximum beziehungsweise {maximum} fuer eine gemeinsame Matrix.
3. Kompaktere Demonstrationen. Dies waere eine neue Promptmethodik und benoetigt einen eigenen Preflight.
4. Kontrollierte neue k1/k3-Vergleichsmatrix mit identischem hoeheren Limit. Dies ist die sauberste direkte k-Effekt-Pruefung, aber ein neues Studiendesign.

Keine Option wurde automatisch umgesetzt. Vor weiteren Evaluationslaeufen ist eine explizite methodische Entscheidung erforderlich.

## Fehlgeschlagene additive Validator-Erstversion

`scripts/validate_dynamic_k3_preflight_20260717.py` brach nach dem Schreiben eines Teiloutputs wegen einer Bool-Konvertierung fuer leere Fallback-Demolisten ab. Bestehende Artefakte waren nicht betroffen. Die Datei und ihr Teiloutput bleiben unangetastet und sind durch die additive `_v2`-Version ersetzt; die korrigierte Validierung besteht vollstaendig.

## Nicht erzeugte Ergebnisartefakte

Keine Run-IDs, Prediction-CSVs, Ergebniskennzahlen, McNemar-/Holm-Tests, Bootstrapintervalle, Fehleranalysen oder Ergebnisplots wurden erzeugt. Ein aktualisiertes autoritatives Ergebnisregister wurde bewusst nicht erstellt.

## Abschluss

```text
Existing files modified: NEIN
Authoritative environment: .venv_flash
New configs: 36/36
Retrieval preflight: PASS
Prompt validity: PASS
Prompt truncations: {summary['total_prompt_truncations']}
Full runs: 0/36
Full runs started: NEIN
Statistics complete: NICHT ANWENDBAR
Error analysis complete: NICHT ANWENDBAR
Documentation v2: FEASIBILITY ADDENDUM COMPLETE
Rerun required: NEIN
New methodological decision required: JA
```
"""
    write_new(AUDIT, audit_text)

    docs_files: dict[str, str] = {
        "README.md": """# Dynamic-Few-Shot-k=3-Erweiterung v2\n\nDieses Verzeichnis ist ein additives Feasibility- und Implementierungsaddendum. Der Preflight wurde wegen realer Prompttruncation bei 2.048 Tokens blockiert. Es wurden keine der 36 Evaluationen gestartet und keine k=3-Ergebnisse erzeugt. Die bestehende Dokumentation vom 2026-07-16 und ihre 48-Run-Ergebnisregister bleiben unveraendert autoritativ.\n""",
        "MASTERARBEIT_COMPLETE_PROJECT_DOCUMENTATION_K3_EXTENSION_V2.md": f"""# Projektabschlussdokumentation: k=3-Erweiterungsaddendum\n\nStatus: **BLOCKED-BY-PROMPT-TRUNCATION**. Nach Abschluss der 48-Run-Hauptlinie wurde additiv eine kontrollierte Dynamic-Few-Shot-k=3-Erweiterung vorbereitet. 36 Configs und eine versionierte Implementierung liegen vor; 2.530/37.152 Promptzeilen ueberschreiten 2.048 Tokens, maximal {maximum}. Daher existieren keine k=3-Evaluationsergebnisse.\n\nHistorische Hauptlinie: 48 Runs, unveraendert autoritativ. Vorgesehene erweiterte Linie: 84 Runs, nicht realisiert. Vor Fortsetzung ist eine methodische Entscheidung zum gemeinsamen Inputlimit oder zur Demonstrationsdarstellung erforderlich.\n""",
        "EXECUTIVE_PROJECT_SUMMARY_K3_EXTENSION_V2.md": f"""# Executive Summary: k=3-Feasibility\n\nDie k=3-Erweiterung ist technisch implementiert und retrievalseitig valide: 36/36 Configs, 1.032/1.032 eindeutige Top-3- und Structure-Top-3-Auswahlen, keine Leakage, keine variablen Gate-k-Werte. Sie ist bei `max_input_tokens=2048` nicht ausfuehrungsfreigegeben, weil {summary['total_prompt_truncations']} Promptzeilen truncieren wuerden. Keine Performanceaussage zu k=3 ist zulaessig.\n""",
        "DYNAMIC_K1_VS_K3_METHODS_AND_RESULTS.md": """# Dynamic k=1 versus k=3: Methoden und Ergebnisstatus\n\n## Methode\n\nk=3 verwendet drei unterschiedliche Demonstrationen in finaler Rankingreihenfolge. Gates akzeptieren das gesamte Set nur bei `min(score_1, score_2, score_3) >= threshold`; andernfalls folgt vollstaendiges Zero Shot. Structure verwendet unveraendert BGE Top-10 und `structure_topk_v2`.\n\n## Ergebnisstatus\n\nEs liegen keine k=3-Modellresultate vor. Der Promptpreflight wurde durch Truncation blockiert. Entsprechend wurden keine EMA-/ESR-, k1-vs.-k3- oder Signifikanzwerte berechnet.\n""",
        "UPDATED_RESEARCH_QUESTIONS_RESULTS_MATRIX_K3_V2.md": """# Forschungsfragenmatrix: k=3-Erweiterungsstatus\n\n| Zusatzfrage | Evidenzstatus | Zulaessige Aussage |\n| --- | --- | --- |\n| Ist Dynamic k=3 bei 2.048 Tokens vollstaendig auswertbar? | Beantwortet | Nein, der vorab definierte truncationfreie Preflight ist blockiert. |\n| Verbessert k=3 die EMA gegenueber k=1 oder Zero Shot? | Nicht untersucht | Keine Aussage ohne Vollruns. |\n| Veraendert Structure Top-3 den Retrievaleffekt? | Nicht untersucht | Nur Retrievalauswahl und Promptlaenge sind validiert. |\n| Wirkt das Set-Gate bei 0.70/0.85 binaer? | Technisch beantwortet | Ja, im Preflight treten ausschliesslich k=3 und k=0 auf. |\n""",
        "K3_EXTENSION_RESULTS_REGISTRY_NOT_CREATED.md": """# Ergebnisregister bewusst nicht erstellt\n\n`UPDATED_AUTHORITATIVE_RESULTS_REGISTRY_K3_V2.csv` wurde nicht erzeugt. Der Promptpreflight blockierte alle Vollruns; daher existieren keine k=3-Ergebnisse, die autoritativ registriert werden koennten. Das bestehende 48-Run-Ergebnisregister bleibt unveraendert.\n""",
        "UPDATED_THESIS_CHAPTER_WRITING_BLUEPRINT_K3_V2.md": """# Thesis-Blueprint: k=3-Feasibility-Addendum\n\n1. Beschreibe k=3 als nachtraeglich vorbereitete kontrollierte Erweiterung.\n2. Definiere Top-3-, Structure-Top-3- und Set-Minimum-Gate-Semantik.\n3. Berichte den truncationfreien Preflight und die harte Stop-Regel.\n4. Stelle Promptlaengen und Gate-Akzeptanz als Feasibility-, nicht als Leistungsresultate dar.\n5. Begruende, warum die 48-Run-Hauptlinie unveraendert bleibt.\n6. Fuehre hoehere Limits, kompaktere Demos und eine limitkontrollierte k1/k3-Matrix nur als zukuenftige Optionen auf.\n""",
        "K3_EXTENSION_TIMELINE_AND_DECISION_ADDENDUM.md": """# Timeline- und Entscheidungsaddendum\n\n- 2026-07-17: Aktive Writer ausgeschlossen.\n- 2026-07-17: k=1-Runner, Retriever, Gates, Structure-Reranking und Prompts auditiert.\n- 2026-07-17: additive k=3-Runner-/Retriever-Version und 36 Configs erzeugt.\n- 2026-07-17: offline Retrieval- und Promptpreflight ueber 37.152 Zeilen abgeschlossen.\n- 2026-07-17: Harte Stop-Regel wegen Prompttruncation ausgeloest.\n- 2026-07-17: Entscheidung: keine Vollruns, keine Ergebnisanalyse, Feasibility-Addendum statt Ergebnisupdate.\n""",
        "K3_EXTENSION_LIMITATIONS_AND_COMPARABILITY.md": """# Limitationen und Vergleichbarkeit\n\nFull-Schema-k=3 vergroessert die Prompts fallabhaengig stark. Ein Lauf mit stiller Truncation waere nicht sauber mit der truncationfreien k=1-Hauptlinie vergleichbar. Eine Erhoehung des Inputlimits nur fuer k=3 vermischt k-Effekt und Kontextlimit-Effekt. Eine kompaktere Demoform waere ebenfalls eine Methodenveraenderung. Fuer eine direkte k1/k3-Aussage ist daher eine kontrollierte Matrix mit identischem Promptformat und identischem ausreichend hohem Limit erforderlich.\n\nDie Gatebedingungen sind zudem deutlich selektiver, weil das Minimum dreier Scores thresholded wird. Diese Semantik ist beabsichtigt, aber nicht numerisch mit der Akzeptanzrate eines k=1-Topscore-Gates gleichzusetzen.\n""",
    }
    for name, text in docs_files.items():
        write_new(DOCS / name, text)

    artifact_rows = [
        {
            "artifact": row["new_k3_config"],
            "sha256": row["config_sha256"],
            "classification": "PREPARED_NOT_EVALUATED_CONFIG",
            "authoritative_for": "k3 feasibility only",
        }
        for row in matrix
    ]
    artifact_rows.extend(
        {
            "artifact": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "classification": classification,
            "authoritative_for": purpose,
        }
        for path, classification, purpose in (
            (AUDIT, "AUTHORITATIVE_FEASIBILITY_AUDIT", "k3 preflight decision"),
            (PROMPTS, "AUTHORITATIVE_FEASIBILITY_DATA", "prompt lengths and truncation"),
            (RETRIEVAL, "AUTHORITATIVE_FEASIBILITY_DATA", "retrieval selection validation"),
            (SUMMARY, "AUTHORITATIVE_FEASIBILITY_SUMMARY", "group statistics"),
            (VALIDATION, "AUTHORITATIVE_FEASIBILITY_VALIDATION", "independent integrity checks"),
        )
    )
    write_new(
        DOCS / "UPDATED_AUTHORITATIVE_ARTIFACT_REGISTRY_K3_V2.csv",
        csv_text(artifact_rows),
    )
    claims = [
        {
            "claim": "The k3 extension is technically prepared.",
            "evidence": "36 configs, additive runner/retriever, 5 passing tests",
            "status": "SUPPORTED",
            "scope": "implementation only",
        },
        {
            "claim": "The k3 extension is truncation-free at 2048 tokens.",
            "evidence": f"{summary['total_prompt_truncations']} truncating config-case rows",
            "status": "REFUTED",
            "scope": "prompt feasibility",
        },
        {
            "claim": "k3 improves or harms NL2SQL accuracy.",
            "evidence": "No full runs were started.",
            "status": "NOT_EVALUATED",
            "scope": "performance",
        },
        {
            "claim": "Set gates are binary k3/k0 and leakage-free in preflight.",
            "evidence": "37152 prompt rows; unexpected k=0; leakage=0",
            "status": "SUPPORTED",
            "scope": "preflight mechanics",
        },
    ]
    write_new(DOCS / "UPDATED_CLAIMS_EVIDENCE_SOURCE_MATRIX_K3_V2.csv", csv_text(claims))
    table_register = [
        {
            "id": "K3-F1",
            "title": "Prompt-token distributions by model role and k3 condition",
            "source": str(PROMPTS.relative_to(ROOT)),
            "status": "AVAILABLE_FEASIBILITY_TABLE",
        },
        {
            "id": "K3-F2",
            "title": "Gate acceptance and zero-shot fallback counts",
            "source": str(SUMMARY.relative_to(ROOT)),
            "status": "AVAILABLE_FEASIBILITY_TABLE",
        },
        {
            "id": "K3-R1",
            "title": "k1-vs-k3 performance table",
            "source": "none",
            "status": "NOT_CREATED_NO_FULL_RUNS",
        },
    ]
    write_new(DOCS / "UPDATED_THESIS_TABLE_AND_FIGURE_REGISTER_K3_V2.csv", csv_text(table_register))

    doc_payloads = [path for path in sorted(DOCS.iterdir()) if path.is_file()]
    docs_manifest = {
        "status": "BLOCKED-BY-PROMPT-TRUNCATION",
        "documentation_type": "feasibility_and_implementation_addendum",
        "pre_k3_baseline_runs": 48,
        "k3_full_runs": 0,
        "extended_84_run_analysis_exists": False,
        "updated_results_registry_created": False,
        "files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in doc_payloads
        ],
    }
    docs_manifest_path = DOCS / "PROJECT_DOCUMENTATION_K3_EXTENSION_MANIFEST.json"
    write_new(docs_manifest_path, json.dumps(docs_manifest, ensure_ascii=False, indent=2) + "\n")

    known_new_paths = [
        ROOT / "src/06_batch_run_dynamic_k3_v1.py",
        ROOT / "src/retrieval_utils_dynamic_k3_v1.py",
        ROOT / "scripts/prepare_dynamic_k3_extension_20260717.py",
        ROOT / "scripts/preflight_dynamic_k3_extension_20260717.py",
        ROOT / "scripts/validate_dynamic_k3_preflight_20260717.py",
        ROOT / "scripts/validate_dynamic_k3_preflight_20260717_v2.py",
        ROOT / "scripts/finalize_dynamic_k3_feasibility_20260717.py",
        ROOT / "tests/test_dynamic_k3_v1.py",
        MATRIX,
        DEPENDENCIES,
        PROMPTS,
        SUMMARY,
        RETRIEVAL,
        ROOT / "audits/derived/dynamic_k3_k1_reference_identity_20260717.csv",
        IDENTITY,
        VALIDATION,
        AUDIT,
        *[ROOT / row["new_k3_config"] for row in matrix],
        *[path for path in sorted(DOCS.iterdir()) if path.is_file()],
    ]
    for path in known_new_paths:
        require(path.is_file(), f"Missing new artifact: {path}")
    source_paths = [
        ROOT / "src/06_batch_run.py",
        ROOT / "src/retrieval_utils.py",
        ROOT / "src/structure_rerank_v2.py",
        ROOT / "audits/derived/cross_model_complete_48_run_results_20260716.csv",
        ROOT / "audits/addendum_python_environment_authoritative_freeze_release_20260716.md",
        ROOT / "data/testcases_spider_dev_full.jsonl",
        ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss",
        ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl",
        ROOT / "data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/manifest.json",
    ]
    manifest = {
        "schema_version": 1,
        "date": "2026-07-17",
        "status": "BLOCKED-BY-PROMPT-TRUNCATION",
        "existing_files_modified": False,
        "authoritative_environment": "/home/ec2-user/nl2sql_testbench/.venv_flash/bin/python",
        "implementation_created": True,
        "configs_created": 36,
        "retrieval_preflight": "PASS",
        "prompt_validity": "PASS",
        "prompt_truncations": summary["total_prompt_truncations"],
        "affected_unique_cases": len(affected_case_ids),
        "maximum_prompt_tokens": maximum,
        "full_runs_released": False,
        "full_runs_started": False,
        "full_run_ids": [],
        "result_files": [],
        "statistics_created": False,
        "error_analysis_created": False,
        "pre_k3_authoritative_baseline_runs": 48,
        "extended_analysis_run_count": 48,
        "retrieval_validation": validation,
        "gate_acceptance_per_model_role": {
            key: {"k3": value[0], "zero_shot": value[1]}
            for key, value in gate_counts.items()
        },
        "prompt_group_statistics": summary["groups"],
        "config_matrix": [
            {
                "path": row["new_k3_config"],
                "sha256": row["config_sha256"],
                "reference": row["reference_k1_config"],
                "reference_sha256": row["reference_config_sha256"],
                "model_key": row["model_key"],
                "role": row["role"],
                "condition": row["condition"],
                "changed_fields": json.loads(row["changed_fields"]),
            }
            for row in matrix
        ],
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in source_paths
        ],
        "new_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in known_new_paths
        ],
        "superseded_additive_validator": {
            "path": "scripts/validate_dynamic_k3_preflight_20260717.py",
            "reason": "TypeError while counting duplicate IDs for empty fallback demo lists",
            "replacement": "scripts/validate_dynamic_k3_preflight_20260717_v2.py",
            "existing_project_artifacts_affected": False,
        },
        "documentation_directory": str(DOCS.relative_to(ROOT)),
        "updated_results_registry_created": False,
        "methodological_decision_required": True,
        "rerun_required": False,
        "self_hash_policy": "Manifest SHA256 is reported externally after serialization.",
    }
    write_new(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "audit": str(AUDIT.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "docs": str(DOCS.relative_to(ROOT)),
        "new_artifacts_registered": len(known_new_paths),
        "full_runs_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
