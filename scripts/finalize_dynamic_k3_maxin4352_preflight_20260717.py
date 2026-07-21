#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "audits/derived/dynamic_k3_config_matrix_20260717.csv"
CHANGES = ROOT / "audits/derived/dynamic_k3_maxin4352_config_changes_20260717.csv"
PROMPTS = ROOT / "audits/derived/dynamic_k3_prompt_preflight_20260717.csv"
RETRIEVAL = ROOT / "audits/derived/dynamic_k3_retrieval_selection_validation_20260717.csv"
SUMMARY = ROOT / "audits/derived/dynamic_k3_prompt_preflight_summary_20260717.json"
CONTEXT = ROOT / "audits/derived/dynamic_k3_model_context_capacity_20260717.json"
IDENTITY = ROOT / "audits/derived/dynamic_k3_k1_reference_identity_20260717_v2.csv"
VALIDATION = ROOT / "audits/derived/dynamic_k3_preflight_validation_20260717_v2.json"
DEPENDENCIES = ROOT / "audits/derived/dynamic_k3_implementation_dependencies_20260717.csv"
AUDIT = ROOT / "audits/audit_dynamic_fewshot_k3_implementation_and_preflight_20260717.md"
MANIFEST = ROOT / "audits/dynamic_fewshot_k3_implementation_and_preflight_manifest_20260717.json"
DOC_DIR = ROOT / "docs/final_project_documentation_20260717_k3_extension_v2"

DOC_NAMES = (
    "README.md",
    "MASTERARBEIT_COMPLETE_PROJECT_DOCUMENTATION_K3_EXTENSION_V2.md",
    "EXECUTIVE_PROJECT_SUMMARY_K3_EXTENSION_V2.md",
    "DYNAMIC_K1_VS_K3_METHODS_AND_RESULTS.md",
    "UPDATED_RESEARCH_QUESTIONS_RESULTS_MATRIX_K3_V2.md",
    "K3_EXTENSION_RESULTS_REGISTRY_NOT_CREATED.md",
    "UPDATED_THESIS_CHAPTER_WRITING_BLUEPRINT_K3_V2.md",
    "K3_EXTENSION_TIMELINE_AND_DECISION_ADDENDUM.md",
    "K3_EXTENSION_LIMITATIONS_AND_COMPARABILITY.md",
    "UPDATED_AUTHORITATIVE_ARTIFACT_REGISTRY_K3_V2.csv",
    "UPDATED_CLAIMS_EVIDENCE_SOURCE_MATRIX_K3_V2.csv",
    "UPDATED_THESIS_TABLE_AND_FIGURE_REGISTER_K3_V2.csv",
    "PROJECT_DOCUMENTATION_K3_EXTENSION_MANIFEST.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(rows: list[dict[str, Any]]) -> str:
    require(bool(rows), "Cannot serialize an empty CSV")
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_scoped(path: Path, text: str) -> None:
    allowed = {AUDIT.resolve(), MANIFEST.resolve()}
    allowed.update((DOC_DIR / name).resolve() for name in DOC_NAMES)
    require(path.resolve() in allowed, f"Refusing out-of-scope write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".maxin4352.tmp")
    require(not temporary.exists(), f"Temporary output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def artifact(path: Path) -> dict[str, str]:
    require(path.is_file(), f"Missing artifact: {path}")
    return {"path": relative(path), "sha256": sha256(path)}


def main() -> None:
    prior_audit_sha = sha256(AUDIT) if AUDIT.is_file() else None
    prior_manifest_sha = sha256(MANIFEST) if MANIFEST.is_file() else None
    matrix = read_csv(MATRIX)
    changes = read_csv(CHANGES)
    identity = read_csv(IDENTITY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    context = json.loads(CONTEXT.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    require(len(matrix) == len(changes) == len(identity) == 36, "Expected 36 configs and references")
    require(summary["status"] == validation["status"] == "PASS", "Preflight is not PASS")
    require(summary["total_prompt_truncations"] == 0, "Prompt truncations remain")
    require(summary["prompt_rows"] == 37152, "Prompt row count mismatch")
    require(summary["maximum_prompt_tokens"] == 4269, "Unexpected maximum prompt length")
    require(context["status"] == "PASS", "Context capacity check failed")
    require(not validation["full_runs_started"], "Full runs were unexpectedly started")
    require(not (ROOT / "results/k3_extension_20260717").exists(), "K3 result directory exists")
    require(all(row["status"] == "PASS" for row in changes), "Config migration failure")
    require(all(row["status"] == "PASS" for row in identity), "K1 identity failure")
    require(all("maxin4352" in row["new_k3_config"] for row in matrix), "Old k3 config name remains")

    gate_rows: list[dict[str, Any]] = []
    for condition in (
        "top3_gate070",
        "top3_gate085",
        "structure_top3_gate070",
        "structure_top3_gate085",
    ):
        matches = [row for row in validation["gate_distributions"] if row["condition"] == condition]
        require(len(matches) == 6 and all(row["status"] == "PASS" for row in matches), f"Gate mismatch: {condition}")
        first = matches[0]
        gate_rows.append(
            {
                "condition": condition,
                "k3": first["actual_k3"],
                "zero_shot": first["actual_k0"],
                "status": "PASS",
            }
        )

    config_table = "\n".join(
        f"| {row['model_key']} | {row['role']} | {row['condition']} | `{row['new_k3_config']}` | `{row['config_sha256']}` |"
        for row in matrix
    )
    group_table = "\n".join(
        "| {model_key} | {role} | {condition} | {minimum} | {mean:.2f} | {median:.1f} | "
        "{p95:.2f} | {p99:.2f} | {maximum} | {prompt_truncations} | {fewshot_cases} | {fallback_cases} |".format(**row)
        for row in summary["groups"]
    )
    gate_table = "\n".join(
        f"| {row['condition']} | {row['k3']} | {row['zero_shot']} | {row['status']} |"
        for row in gate_rows
    )
    context_table = "\n".join(
        f"| {row['model_key']} | `{row['revision']}` | {row['resolved_context_capacity']} | "
        f"{row['required_total_tokens']} | {'PASS' if row['capacity_sufficient'] else 'FAIL'} |"
        for row in context["models"]
    )

    audit_text = f"""# Audit: Dynamic-Few-Shot-k=3 maxin4352 Preflight

**DYNAMIC-K3-MAXIN4352-PREFLIGHT: PASS**

```text
Existing pre-k3 project files modified: NEIN
K3 configs renamed: 36/36
K3 configs updated: 36/36
max_input_tokens: 4352
max_new_tokens: 256
Retrieval index unchanged: JA
Retrieval pool unchanged: JA
K1 first-demo reference identity: PASS
Prompt rows checked: 37152
Maximum prompt tokens: 4269
Prompt truncations: 0
Unexpected actual_k values: 0
Leakage: 0
Gate distributions unchanged: JA
Model context capacity sufficient: JA
Full runs started: NEIN
Ready for 36 full runs: JA
```

## Methodische Aenderung

Der urspruengliche k=3-Preflight mit `max_input_tokens=2048` wurde durch die methodisch beschlossene Anpassung auf 4352 Tokens ersetzt. Die Configdateien wurden eindeutig mit `maxin4352` benannt. Die autoritative 48-Run-Baseline und alle vor-k3-Artefakte blieben unveraendert.

Gegenueber den bereits vorbereiteten k=3-Configs wurden ausschliesslich `max_input_tokens: 2048 -> 4352` und die kollisionsfreie Outputidentitaet geaendert. `max_new_tokens=256`, Modelle, Revisionen, Adapter, Prompts, Testset, Retrieval, Gates, Structure-Reranking und Generationseinstellungen blieben unveraendert.

## Runner

`src/06_batch_run_dynamic_k3_v1.py` liest `max_input_tokens` aus der Config. Es war keine Runneraenderung erforderlich. Spaetere Runs persistieren `run_max_input_tokens=4352` und `run_max_new_tokens=256` in CSV und Metadaten.

## Retrieval und Identitaet

- Index: `data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15`
- Pool: 6.960 Spider-Train-Beispiele
- Embedding: `BAAI/bge-large-en-v1.5`
- Neuer Index: NEIN
- Demo-/Scoreidentitaet gegen vorherigen k3-Preflight: 1.032/1.032 Top-3 und 1.032/1.032 Structure-Top-3
- Prompt-/Gateidentitaet gegen vorherigen k3-Preflight: 37.152/37.152
- k=1-Erstdemo-ID und BGE-Score: 36/36 Referenzruns mit jeweils 1.032/1.032 PASS
- Leakage: 0
- Doppelte Demos: 0

## Gateverteilungen pro Modellrolle

| Bedingung | k=3 | Zero Shot | Status |
| --- | ---: | ---: | --- |
{gate_table}

## Promptpreflight

| Modell | Rolle | Bedingung | Min | Mittel | Median | p95 | p99 | Max | Truncations | k=3 | k=0 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{group_table}

Alle 37.152 Zeilen verwenden `max_input_tokens=4352`. Die maximale beobachtete Promptlaenge ist 4.269 Tokens; der verbleibende Puffer betraegt 83 Tokens. Zielanfrage, Schema und Assistant-Praefix sind in allen Faellen valide.

## Modellkontext

| Modell | Revision | Lokale Kontextkapazitaet | Benoetigt | Status |
| --- | --- | ---: | ---: | --- |
{context_table}

Der Runner begrenzt den Input separat auf 4.352 Tokens und uebergibt `max_new_tokens=256` an `generate`. Das erforderliche Gesamtbudget von 4.608 Tokens liegt bei allen drei lokalen Modellkonfigurationen deutlich unter der deklarierten Kapazitaet.

## Configs und SHA256

| Modell | Rolle | Bedingung | Config | SHA256 |
| --- | --- | --- | --- | --- |
{config_table}

## Freigabegrenze

Der Preflight gibt die 36 Configs technisch frei. In diesem Auftrag wurde keine generative Evaluation gestartet. Es existieren weiterhin keine k=3-Ergebnis-CSVs, Run-IDs, Leistungsmetriken oder statistischen k1-vs.-k3-Analysen.
"""

    artifact_rows = [
        {
            "artifact": row["new_k3_config"],
            "sha256": row["config_sha256"],
            "status": "PREFLIGHT_PASS_NOT_EVALUATED",
            "notes": "max_input_tokens=4352; max_new_tokens=256",
        }
        for row in matrix
    ]
    docs: dict[str, str] = {
        "README.md": """# Dynamic-k3 Extension v2

Status: `DYNAMIC-K3-MAXIN4352-PREFLIGHT: PASS`.

Die 36 k=3-Configs sind mit `max_input_tokens=4352` truncationfrei vorgeprueft und fuer eine spaetere sequenzielle Evaluation technisch freigegeben. In diesem Auftrag wurden keine Vollruns gestartet. Die bestehende 48-Run-Baseline bleibt autoritativ.
""",
        "MASTERARBEIT_COMPLETE_PROJECT_DOCUMENTATION_K3_EXTENSION_V2.md": """# Projektdokumentation: Dynamic-k3-Erweiterung v2

Die nachtraegliche k=3-Erweiterung umfasst 36 vorbereitete Configs. Der urspruengliche 2.048er Feasibility-Preflight war durch Prompttruncation blockiert. Nach der methodisch beschlossenen, kontrollierten Anpassung auf 4.352 Inputtokens besteht der vollstaendige Preflight mit 0/37.152 Truncations. Die Vollruns und Leistungsanalysen stehen noch aus.

Die historische 48-Run-Hauptlinie bleibt unveraendert autoritativ. Kuenftige k=3-Ergebnisse bilden erst nach vollstaendiger Ausfuehrung und Validierung eine additive 84-Run-Analyse.
""",
        "EXECUTIVE_PROJECT_SUMMARY_K3_EXTENSION_V2.md": """# Executive Summary: k=3 maxin4352

36/36 Configs, Retrieval, Set-Gates, Structure-Reranking, Promptintegritaet und Modellkontextkapazitaet bestehen den Preflight. Die maximale Promptlaenge betraegt 4.269 bei einem Limit von 4.352 Tokens. Keine Evaluation wurde gestartet; daher liegen keine k=3-Leistungswerte vor.
""",
        "DYNAMIC_K1_VS_K3_METHODS_AND_RESULTS.md": """# Dynamic k=1 versus k=3

Die k=3-Methode verwendet drei Full-Schema-Demonstrationen. Gates akzeptieren binaer k=3 oder fallen vollstaendig auf k=0 zurueck; der Set-Score ist das Minimum der drei originalen BGE-Scores. Structure Top-3 wird aus BGE Top-10 mit der unveraenderten Structure-v2-Heuristik gewaehlt.

Der k=3-Preflight ist bei 4.352 Inputtokens bestanden. Noch liegen keine Modellresultate vor. Aussagen zu EMA, ESR, k1-vs.-k3-Effekten oder Signifikanz sind daher nicht zulaessig.
""",
        "UPDATED_RESEARCH_QUESTIONS_RESULTS_MATRIX_K3_V2.md": """# Research Questions Matrix: k=3 Addendum

| Frage | Status | Evidenz |
| --- | --- | --- |
| Ist k=3 technisch truncationfrei bei 4.352 Tokens? | PASS | 0/37.152 Truncations; Maximum 4.269. |
| Bleiben Retrieval und Gates unveraendert? | PASS | Demo-, Score-, Prompt- und Gateverteilungsidentitaet. |
| Verbessert k=3 die NL2SQL-Leistung? | Nicht untersucht | Keine Vollruns oder Ergebniswerte. |
""",
        "K3_EXTENSION_RESULTS_REGISTRY_NOT_CREATED.md": """# Kein k=3-Ergebnisregister

Ein aktualisiertes autoritatives Ergebnisregister wurde nicht erzeugt. Der maxin4352-Preflight ist bestanden, die 36 generativen Vollruns wurden in diesem Auftrag jedoch nicht gestartet. Das bestehende 48-Run-Ergebnisregister bleibt autoritativ.
""",
        "UPDATED_THESIS_CHAPTER_WRITING_BLUEPRINT_K3_V2.md": """# Thesis Blueprint: k=3 Addendum

1. Die 48-Run-Matrix als autoritative Hauptlinie beschreiben.
2. Die k=3-Erweiterung und den binaeren Set-Gate definieren.
3. Die Aenderung von 2.048 auf 4.352 Inputtokens transparent als methodische Entscheidung dokumentieren.
4. Den bestandenen Feasibility-Preflight berichten.
5. Leistungsresultate erst nach allen 36 vollstaendigen und validierten Runs ergaenzen.
""",
        "K3_EXTENSION_TIMELINE_AND_DECISION_ADDENDUM.md": """# Timeline and Decision Addendum

- 2026-07-17: Additive k=3-Implementierung und 36 Configs erstellt.
- 2026-07-17: Preflight bei 2.048 Tokens wegen 2.530/37.152 Truncations blockiert.
- 2026-07-17: Gemeinsames Inputlimit methodisch auf 4.352 Tokens festgelegt.
- 2026-07-17: 36 Configs eindeutig umbenannt und ausschliesslich Limit sowie Outputidentitaet angepasst.
- 2026-07-17: Vollstaendiger maxin4352-Preflight mit 0 Truncations bestanden.
- 2026-07-17: Keine generativen Vollruns gestartet.
""",
        "K3_EXTENSION_LIMITATIONS_AND_COMPARABILITY.md": """# Limitations and Comparability

Das hoehere Inputlimit unterscheidet die geplante k=3-Erweiterung von der bestehenden k=1-Hauptlinie mit 2.048 Tokens. Ein direkter k1-vs.-k3-Effekt kann deshalb durch das unterschiedliche zulaessige Inputbudget mitbeeinflusst werden. Die 48-Run-Baseline wird nicht ersetzt. K3-Ergebnisse sind nach ihrer spaeteren Erzeugung als additive Erweiterung zu berichten.
""",
        "UPDATED_AUTHORITATIVE_ARTIFACT_REGISTRY_K3_V2.csv": csv_text(artifact_rows),
        "UPDATED_CLAIMS_EVIDENCE_SOURCE_MATRIX_K3_V2.csv": csv_text(
            [
                {
                    "claim": "The maxin4352 k3 prompt preflight is truncation-free.",
                    "status": "SUPPORTED",
                    "evidence": relative(PROMPTS),
                },
                {
                    "claim": "Retrieval selections and gate decisions are unchanged from the initial k3 preflight.",
                    "status": "SUPPORTED",
                    "evidence": relative(SUMMARY),
                },
                {
                    "claim": "K3 improves NL2SQL performance.",
                    "status": "NOT_TESTED",
                    "evidence": "No full k3 runs",
                },
            ]
        ),
        "UPDATED_THESIS_TABLE_AND_FIGURE_REGISTER_K3_V2.csv": csv_text(
            [
                {
                    "item": "K3 maxin4352 prompt-length table",
                    "status": "AVAILABLE",
                    "source": relative(SUMMARY),
                },
                {
                    "item": "K3 gate-distribution table",
                    "status": "AVAILABLE",
                    "source": relative(VALIDATION),
                },
                {
                    "item": "K3 performance table",
                    "status": "NOT_AVAILABLE",
                    "source": "Full runs not started",
                },
            ]
        ),
    }
    for name, content in docs.items():
        write_scoped(DOC_DIR / name, content if content.endswith("\n") else content + "\n")

    doc_manifest_entries = [artifact(DOC_DIR / name) for name in docs]
    doc_manifest = {
        "schema_version": 2,
        "status": "PREFLIGHT_PASS_FULL_RUNS_NOT_STARTED",
        "max_input_tokens": 4352,
        "max_new_tokens": 256,
        "pre_k3_authoritative_baseline_runs": 48,
        "k3_configs": 36,
        "k3_full_runs_started": False,
        "files": doc_manifest_entries,
    }
    write_scoped(
        DOC_DIR / "PROJECT_DOCUMENTATION_K3_EXTENSION_MANIFEST.json",
        json.dumps(doc_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    write_scoped(AUDIT, audit_text)

    current_paths = [
        ROOT / "src/06_batch_run_dynamic_k3_v1.py",
        ROOT / "src/retrieval_utils_dynamic_k3_v1.py",
        ROOT / "scripts/prepare_dynamic_k3_extension_20260717.py",
        ROOT / "scripts/migrate_dynamic_k3_maxin4352_20260717.py",
        ROOT / "scripts/preflight_dynamic_k3_extension_20260717.py",
        ROOT / "scripts/validate_dynamic_k3_preflight_20260717_v2.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_dynamic_k3_v1.py",
        MATRIX,
        CHANGES,
        PROMPTS,
        RETRIEVAL,
        SUMMARY,
        CONTEXT,
        IDENTITY,
        VALIDATION,
        DEPENDENCIES,
        AUDIT,
        *[ROOT / row["new_k3_config"] for row in matrix],
        *[DOC_DIR / name for name in DOC_NAMES],
    ]
    manifest = {
        "schema_version": 2,
        "date": "2026-07-17",
        "status": "PASS",
        "status_label": "DYNAMIC-K3-MAXIN4352-PREFLIGHT",
        "existing_pre_k3_project_files_modified": False,
        "original_maxin2048_preflight": {
            "status": "BLOCKED-BY-PROMPT-TRUNCATION",
            "prompt_truncations": 2530,
            "maximum_prompt_tokens": 4269,
            "replacement_decision": "max_input_tokens increased to 4352 for the additive k3 extension",
            "prior_audit_sha256": prior_audit_sha,
            "prior_manifest_sha256": prior_manifest_sha,
        },
        "config_migration": {
            "renamed": 36,
            "updated": 36,
            "max_input_tokens": 4352,
            "max_new_tokens": 256,
            "allowed_changes": ["max_input_tokens", "run_output_prefix", "config filename"],
            "unintended_changes": 0,
            "change_report": artifact(CHANGES),
        },
        "runner": {
            "path": "src/06_batch_run_dynamic_k3_v1.py",
            "modified_for_maxin4352": False,
            "config_driven_max_input_tokens": True,
            "metadata_fields": ["run_max_input_tokens", "run_max_new_tokens"],
        },
        "preflight": {
            "configs": 36,
            "testcases": 1032,
            "prompt_rows": 37152,
            "configured_max_input_tokens": 4352,
            "configured_max_new_tokens": 256,
            "maximum_prompt_tokens": 4269,
            "prompt_truncations": 0,
            "invalid_prompts": 0,
            "unexpected_actual_k": 0,
            "leakage": 0,
            "full_runs_started": False,
            "ready_for_36_full_runs": True,
        },
        "retrieval": summary["retrieval"],
        "prior_prompt_identity": summary["prior_maxin2048_prompt_identity"],
        "gate_distributions_per_model_role": gate_rows,
        "model_context_capacity": context,
        "k1_reference_identity": validation["k1_reference_identity"],
        "validation_failures": validation["hard_failures_excluding_truncation"],
        "configs": [artifact(ROOT / row["new_k3_config"]) for row in matrix],
        "current_artifacts": [artifact(path) for path in current_paths],
        "result_artifacts": [],
        "full_run_ids": [],
        "pre_k3_authoritative_baseline_runs": 48,
        "updated_result_registry_created": False,
        "self_hash_policy": "Manifest SHA256 is reported externally after serialization.",
    }
    write_scoped(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "audit": relative(AUDIT),
                "manifest": relative(MANIFEST),
                "configs": 36,
                "prompt_rows": 37152,
                "prompt_truncations": 0,
                "maximum_prompt_tokens": 4269,
                "full_runs_started": False,
                "ready_for_36_full_runs": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
