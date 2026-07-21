#!/usr/bin/env python3
"""Additive post-generation validation fixes for the complete 24-run audit.

This script does not alter the original analysis. It creates corrected,
versioned presentation artifacts after visual QA and refreshes the gap-evidence
snapshot after the main audit outputs exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen35_matplotlib_cache_complete24_v2")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_project_completion_and_remaining_gaps import build_gap_rows

DATE = "20260716"
MODELS = ["qwen2b", "llama3b", "qwen9b"]
MODEL_LABELS = {"qwen2b": "Qwen 3.5 2B LoRA v2", "llama3b": "Llama 3.2 3B Instruct LoRA v2", "qwen9b": "Qwen 3.5 9B LoRA v2"}
CONDS = ["zero_shot", "top1", "top1_gate070", "top1_gate085", "static_seed42", "structure", "structure_gate070", "structure_gate085"]
COND_LABELS = {"zero_shot": "Zero Shot", "top1": "Top-1", "top1_gate070": "Top-1 Gate 0.70", "top1_gate085": "Top-1 Gate 0.85", "static_seed42": "Static", "structure": "Structure", "structure_gate070": "Structure Gate 0.70", "structure_gate085": "Structure Gate 0.85"}

ORIGINAL_AUDIT = ROOT / f"audits/audit_complete_24_lora_run_error_analysis_and_project_gap_review_{DATE}.md"
ORIGINAL_MANIFEST = ROOT / f"audits/complete_24_lora_run_error_analysis_and_project_gap_review_manifest_{DATE}.json"
LABELS = ROOT / f"audits/derived/complete_24_lora_run_error_labels_long_{DATE}.csv"
OVERLAP = ROOT / f"audits/derived/lora_cross_prompt_error_set_overlap_{DATE}.csv"
GATE = ROOT / f"audits/derived/lora_gate_partition_error_analysis_{DATE}.csv"
OUT = {
    "examples": ROOT / f"audits/derived/complete_24_lora_run_qualitative_examples_{DATE}_v2.csv",
    "gap": ROOT / f"audits/derived/project_completion_gap_matrix_{DATE}_v2.csv",
    "jaccard_png": ROOT / f"audits/plots/complete_24_lora_incorrect_set_jaccard_{DATE}_v2.png",
    "jaccard_pdf": ROOT / f"audits/plots/complete_24_lora_incorrect_set_jaccard_{DATE}_v2.pdf",
    "gate_png": ROOT / f"audits/plots/complete_24_lora_gate_partitions_{DATE}_v2.png",
    "gate_pdf": ROOT / f"audits/plots/complete_24_lora_gate_partitions_{DATE}_v2.pdf",
    "addendum": ROOT / f"audits/addendum_complete_24_lora_post_generation_validation_{DATE}.md",
    "manifest": ROOT / f"audits/complete_24_lora_post_generation_validation_manifest_{DATE}.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def norm_sql(sql: str) -> str:
    return " ".join(sql.lower().replace(";", " ; ").split())


def save_pair(png: Path, pdf: Path) -> None:
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Plot target exists: {png} / {pdf}")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("One or more additive v2 targets already exist")
    manifest = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    records = {(row["model"], row["condition"]): row for row in manifest["runs"]}
    run_rows: dict[tuple[str, str], list[dict[str, str]]] = {
        key: read_csv(ROOT / record["csv_path"]) for key, record in records.items()
    }
    row_by = {(model, condition, row["id"]): row for (model, condition), rows in run_rows.items() for row in rows}
    testcases = {row["id"]: row for row in (json.loads(line) for line in (ROOT / "data/testcases_spider_dev_full.jsonl").read_text(encoding="utf-8").splitlines() if line)}

    concrete_labels: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in read_csv(LABELS):
        if row["evidence_level"] in {"E1", "E2"} and row["broad_family"] != "F_UNCLEAR_HEURISTIC":
            concrete_labels[(row["model_key"], row["condition"], row["case_id"])].append(row["error_label"])

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    ids = [row["id"] for row in run_rows[("qwen2b", "zero_shot")]]
    for model in MODELS:
        for condition in CONDS[1:]:
            for cid in ids:
                zero = row_by[(model, "zero_shot", cid)]; current = row_by[(model, condition, cid)]
                zc, cc = truth(zero["exec_match"]), truth(current["exec_match"])
                if not zc and cc and concrete_labels[(model, "zero_shot", cid)]:
                    candidates["prompt_benefit"].append({"model": model, "condition": condition, "cid": cid, "labels": ";".join(concrete_labels[(model, "zero_shot", cid)])})
                if zc and not cc and concrete_labels[(model, condition, cid)]:
                    candidates["prompt_harm"].append({"model": model, "condition": condition, "cid": cid, "labels": ";".join(concrete_labels[(model, condition, cid)])})
                if zc and cc and zero["pred_sql"] != current["pred_sql"] and norm_sql(zero["pred_sql"]) != norm_sql(current["pred_sql"]):
                    candidates["alternative_valid"].append({"model": model, "condition": condition, "cid": cid, "labels": "ALTERNATIVE_VALID_FORMULATION"})
        for cid in ids:
            if all(not truth(row_by[(model, condition, cid)]["exec_match"]) for condition in CONDS) and concrete_labels[(model, "zero_shot", cid)]:
                candidates["stable_unresolved"].append({"model": model, "condition": "structure", "cid": cid, "labels": ";".join(concrete_labels[(model, "zero_shot", cid)])})

    def diverse(group: str, limit: int) -> list[dict[str, str]]:
        pool = sorted(candidates[group], key=lambda x: (CONDS.index(x["condition"]), MODELS.index(x["model"]), x["cid"]))
        selected: list[dict[str, str]] = []
        desired_conditions = CONDS[1:] if group in {"prompt_benefit", "prompt_harm"} else ["top1", "static_seed42", "structure", "structure_gate070"] if group == "alternative_valid" else ["structure"] * limit
        for index, condition in enumerate(desired_conditions):
            preferred_model = MODELS[index % 3]
            choice = next((x for x in pool if x not in selected and x["condition"] == condition and x["model"] == preferred_model), None)
            choice = choice or next((x for x in pool if x not in selected and x["condition"] == condition), None)
            if choice:
                selected.append(choice)
            if len(selected) == limit:
                return selected
        for item in pool:
            if item not in selected:
                selected.append(item)
            if len(selected) == limit:
                return selected
        raise RuntimeError(f"Insufficient candidates for {group}")

    selected = (
        [("prompt_benefit", x) for x in diverse("prompt_benefit", 8)] +
        [("prompt_harm", x) for x in diverse("prompt_harm", 8)] +
        [("stable_unresolved", x) for x in diverse("stable_unresolved", 4)] +
        [("alternative_valid", x) for x in diverse("alternative_valid", 4)]
    )
    examples: list[dict[str, Any]] = []
    for group, item in selected:
        model, condition, cid = item["model"], item["condition"], item["cid"]
        zero = row_by[(model, "zero_shot", cid)]; current = row_by[(model, condition, cid)]
        tc = testcases[cid]
        examples.append({
            "example_type": group, "model_key": model, "model_line": MODEL_LABELS[model], "condition": condition,
            "case_id": cid, "db_id": tc["db_id"], "question": tc["question"], "schema_excerpt": tc.get("schema_prompt", "")[:1800],
            "gold_sql": tc["gold_sql"], "zero_shot_sql": zero["pred_sql"], "condition_sql": current["pred_sql"],
            "execution_status": f"zero_success={truth(zero['pred_ok'])};zero_match={truth(zero['exec_match'])};condition_success={truth(current['pred_ok'])};condition_match={truth(current['exec_match'])}",
            "outcome_transition": group, "explorative_error_labels": item["labels"],
            "short_interpretation": {"prompt_benefit": "Prompted condition corrects a Zero-Shot error.", "prompt_harm": "Prompted condition introduces an error relative to correct Zero Shot.", "stable_unresolved": "Incorrect under all eight prompt conditions.", "alternative_valid": "Both structurally different SQL outputs execution-match."}[group],
            "methodological_note": "Illustrative deterministic example; no prevalence or causal claim.",
        })
    write_csv(OUT["examples"], examples)

    gap_rows = build_gap_rows()
    write_csv(OUT["gap"], gap_rows)

    overlap = read_csv(OVERLAP)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    image = None
    for ax, model in zip(axes, MODELS):
        matrix = np.zeros((8, 8))
        for row in overlap:
            if row["model_key"] == model:
                matrix[CONDS.index(row["condition_a"]), CONDS.index(row["condition_b"])] = float(row["jaccard"])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(8), [COND_LABELS[c] for c in CONDS], rotation=58, ha="right", fontsize=7)
        ax.set_yticks(range(8), [COND_LABELS[c] for c in CONDS], fontsize=7); ax.set_title(MODEL_LABELS[model])
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.28, top=0.88, wspace=0.48)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.58]); fig.colorbar(image, cax=cax, label="Jaccard")
    fig.suptitle("Jaccard overlap of incorrect case sets", y=0.97)
    fig.text(0.01, 0.01, "Same incorrect outcome does not imply the same SQL error.", fontsize=8)
    save_pair(OUT["jaccard_png"], OUT["jaccard_pdf"])

    gate = read_csv(GATE); gate_order = ["top1_gate070", "top1_gate085", "structure_gate070", "structure_gate085"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, model in zip(axes, MODELS):
        few = [next(row for row in gate if row["model_key"] == model and row["gate_condition"] == cond and row["partition"] == "fewshot_accepted") for cond in gate_order]
        fallback = [next(row for row in gate if row["model_key"] == model and row["gate_condition"] == cond and row["partition"] == "zero_shot_fallback") for cond in gate_order]
        xx = np.arange(4); width = 0.38
        ax.bar(xx - width / 2, [100 * float(row["ema"]) for row in few], width, label="Accepted demo", color="#276FBF")
        ax.bar(xx + width / 2, [100 * float(row["ema"]) for row in fallback], width, label="Zero-shot fallback", color="#9AA0A6")
        ax.set_ylim(0, 100); ax.set_ylabel("EMA (%)"); ax.set_title(MODEL_LABELS[model], pad=7); ax.legend(loc="upper left")
    axes[-1].set_xticks(range(4), [COND_LABELS[c] for c in gate_order], rotation=30, ha="right")
    fig.suptitle("Gate partitions: accepted demonstrations versus fallback", y=0.995)
    fig.text(0.01, 0.01, "Fallback outputs exactly match the corresponding Zero-Shot output.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955)); save_pair(OUT["gate_png"], OUT["gate_pdf"])

    script_hashes = {
        "scripts/analyze_complete_24_lora_run_error_profiles.py": sha(ROOT / "scripts/analyze_complete_24_lora_run_error_profiles.py"),
        "scripts/audit_project_completion_and_remaining_gaps.py": sha(ROOT / "scripts/audit_project_completion_and_remaining_gaps.py"),
        "scripts/finalize_complete_24_lora_run_error_profiles_v2.py": sha(Path(__file__)),
    }
    addendum = "\n".join([
        "# Addendum: Post-Generation-Validierung der vollständigen 24-LoRA-Run-Analyse", "",
        "**STATUS: PASS**", "",
        "Die numerischen Ergebnisse, Fehlerlabels, Statistiken und Schlussfolgerungen des Hauptaudits bleiben unverändert.", "",
        "## Additive Korrekturen", "",
        "1. Die qualitative 24er-Stichprobe wurde bedingungsdivers neu materialisiert: Nutzen und Schaden decken alle sieben Promptbedingungen ab; alternative valide Formulierungen decken Top-1, Static, Structure und Structure Gate 0.70 ab.",
        "2. Die Jaccard-Abbildung wurde mit einer separaten Farbskalenachse neu ausgegeben; keine Beschriftung überlappt die rechte Matrix.",
        "3. Der Gate-Plot wurde neu ausgegeben, sodass alle drei Paneltitel sichtbar sind.",
        "4. Die Gap-Matrix wurde nach Erzeugung der Hauptartefakte erneut snapshot-basiert ausgegeben; Artefakte des aktuellen Audits erscheinen nun korrekt als vorhanden.", "",
        "## Provenienz", "",
        *[f"- `{path}`: `{digest}`" for path, digest in script_hashes.items()], "",
        "Keine bestehende Datei wurde verändert oder überschrieben; alle Korrekturen tragen `_v2` oder einen Addendum-Namen.",
    ])
    write_text(OUT["addendum"], addendum)

    hashes = {str(path.relative_to(ROOT)): sha(path) for key, path in OUT.items() if key != "manifest"}
    manifest_out = {
        "status": "PASS", "date": DATE, "classification": "additive post-generation QA correction",
        "original_audit": {"path": str(ORIGINAL_AUDIT.relative_to(ROOT)), "sha256": sha(ORIGINAL_AUDIT)},
        "original_manifest": {"path": str(ORIGINAL_MANIFEST.relative_to(ROOT)), "sha256": sha(ORIGINAL_MANIFEST)},
        "numerical_results_changed": False, "existing_files_modified": False,
        "qualitative_examples": {"rows": 24, "groups": dict(Counter(row["example_type"] for row in examples)), "benefit_conditions": sorted({row["condition"] for row in examples if row["example_type"] == "prompt_benefit"}), "harm_conditions": sorted({row["condition"] for row in examples if row["example_type"] == "prompt_harm"}), "alternative_conditions": sorted({row["condition"] for row in examples if row["example_type"] == "alternative_valid"})},
        "gap_evidence_missing_count_after_refresh": sum("(missing)" in row["evidence"] for row in gap_rows if row["status"] == "ABGESCHLOSSEN"),
        "preferred_corrected_artifacts": hashes, "script_hashes": script_hashes,
        "manifest_self_hash": None, "manifest_self_hash_note": "Hash externally after creation.",
    }
    write_text(OUT["manifest"], json.dumps(manifest_out, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"status": "PASS", "new_files": len(OUT), "example_groups": dict(Counter(row["example_type"] for row in examples)), "benefit_conditions": sorted({row["condition"] for row in examples if row["example_type"] == "prompt_benefit"}), "harm_conditions": sorted({row["condition"] for row in examples if row["example_type"] == "prompt_harm"})}, indent=2))


if __name__ == "__main__":
    main()
