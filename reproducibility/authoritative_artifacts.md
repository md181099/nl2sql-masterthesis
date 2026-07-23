# Autoritative Forschungsartefakte

Dieses Dokument kennzeichnet die für die Hauptauswertung der Masterarbeit maßgeblichen Trainings-, Evaluations- und Ergebnisartefakte.

Historische Entwicklungsruns, explorative Adaptervarianten und nicht vollständig vergleichbare Evaluationen sind nicht Bestandteil der autoritativen Hauptauswertung, sofern sie nicht ausdrücklich als explorativ gekennzeichnet werden.

## 1. Untersuchte Modelllinien

Die Hauptauswertung umfasst drei Modelllinien:

1. Qwen 3.5 2B Base und zugehöriger LoRA-v2-Adapter
2. Llama 3.2 3B Instruct und zugehöriger LoRA-v2-Adapter
3. Qwen 3.5 9B Base und zugehöriger LoRA-v2-Adapter

Für jede Modelllinie werden das jeweilige Ausgangsmodell und der finale LoRA-v2-Adapter untersucht.

## 2. Finale Trainingskonfigurationen

Die autoritativen Trainingsconfigs sind:

- `configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json`
- `configs/train_lora_llama32_3b_instruct_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json`
- `configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json`

Gemeinsame zentrale Trainingsparameter:

- LoRA-Rang: `r=8`
- LoRA-Alpha: `16`
- LoRA-Dropout: `0.05`
- Zielmodule: geeignete lineare Module
- Trainingsverfahren: Full-Chat-Supervised-Fine-Tuning
- maximale Sequenzlänge: 2.048 Tokens
- Lernrate: `1e-4`
- Scheduler: konstant
- Trainings-Batchgröße: 2
- Gradient Accumulation: 4
- effektive Batchgröße: 8
- Seed: 42
- maximale Epochenzahl: 5
- Early Stopping: Patience 2, Threshold 0.001
- Checkpointauswahl anhand von MixedVal2500-v2
- Spider Dev nicht für Checkpointauswahl verwendet

Die final verwendeten Best-Checkpoints stammen jeweils aus Epoche 1:

- Qwen 3.5 2B: `checkpoint-502`
- Llama 3.2 3B Instruct: `checkpoint-509`
- Qwen 3.5 9B: `checkpoint-502`

Für die Evaluation wurden die finalen Root-Adapter verwendet.

## 3. Trainings- und Validierungsdaten

Autoritativer Trainingsdatensatz:

- Gesamtumfang: 25.000 Beispiele
- Spider Train: 6.960 Beispiele
- SQL Create Context: 18.040 Beispiele

Autoritative Validation:

- MixedVal2500-v2
- Gesamtumfang: 2.500 Beispiele
- `train_others`: 700 Beispiele
- SQL Create Context: 1.800 Beispiele

Autoritative Evaluation:

- Spider Dev
- 1.032 Fälle
- keine Verwendung für Training oder Checkpointauswahl

## 4. Retrievalkonfiguration

Finaler Retrievalindex:

- Name: `spider_train_no_dev_overlap_bge_large_en_v15`
- Poolgröße: 6.960 Beispiele
- Embeddingmodell: `BAAI/bge-large-en-v1.5`
- Indexverfahren: FAISS `IndexFlatIP`
- kein direkter Spider-Dev-Overlap

Untersuchte Retrieval- und Promptingbedingungen:

- Zero Shot
- Static Few Shot
- Dynamic Top-1
- Dynamic Gate 0.70
- Dynamic Gate 0.85
- Structure-Reranking
- Structure-Reranking mit Gate 0.70
- Structure-Reranking mit Gate 0.85
- additive k=3-Erweiterungen

## 5. Autoritative Evaluationsmatrix

Die Hauptauswertung umfasst:

- 48 autoritative Hauptläufe
- 36 autoritative k=3-Erweiterungsläufe
- insgesamt 84 autoritative Evaluationsläufe

Die maßgebliche Run-Matrix ist:

- `audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv`

Diese Datei legt die autoritativen Modellrollen, Bedingungen, Configs und Run-Zuordnungen fest.

## 6. Zentrale Ergebnis- und Trainingsartefakte

Folgende Dateien gehören zur autoritativen Ergebnisbasis:

- `audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv`
- `audits/derived/cross_model_complete_48_run_results_20260716.csv`
- `audits/derived/k3_authoritative_run_metrics_20260718.csv`
- `audits/derived/k1_k3_authoritative_pair_mapping_20260718.csv`
- `audits/derived/lora_v2_authoritative_training_matrix_20260718.csv`
- `audits/derived/lora_v2_training_artifact_inventory_20260718.csv`
- `audits/derived/k3_final_analysis_manifest_20260718.json`

Diese Artefakte dokumentieren:

- die vollständige 48-Run-Hauptmatrix
- die k=3-Erweiterung
- die Zuordnung zwischen k=1- und k=3-Läufen
- die finalen LoRA-v2-Trainingsläufe
- die finalen Adapter- und Checkpointartefakte
- die autoritativen Metriken und Run-Metadaten

## 7. Primäre Evaluationsmetriken

Primäre Metrik:

- Execution Match Accuracy

Ergänzende Metriken:

- Execution Success Rate
- String Exact Match
- Normalized Exact Match
- Character Accuracy
- Token Accuracy
- Laufzeit
- Prompt- und Completion-Tokenanzahl
- Retrievalsimilarity

Statistische Auswertungen umfassen unter anderem:

- McNemar-Tests für gepaarte Erfolgsunterschiede
- Bootstrap-Konfidenzintervalle
- Holm-Korrektur für multiple Vergleiche

## 8. Explorative Artefakte

Nicht alle im Projekt vorhandenen Adapter, Checkpoints und Ergebnisdateien gehören zur Hauptauswertung.

Explorative Artefakte umfassen insbesondere:

- frühere Completion-only-Adapter
- alternative LoRA-Ränge und Alpha-Werte
- alternative Lernraten
- frühere Trainingsdatenmischungen
- andere Sequenzlängen
- spätere Epochen oder alternative Checkpoints
- Teilmengen- und Smoke-Test-Evaluationen
- nicht vollständig vergleichbare Entwicklungsruns

Solche Artefakte dürfen nur als explorative Entwicklungsbefunde interpretiert werden.

## 9. Reproduzierbarkeitsumgebung

Die autoritative technische Umgebung ist unter folgendem Verzeichnis dokumentiert:

- `reproducibility/final_freeze_20260721/`

Zentrale Umgebungseigenschaften:

- Python 3.11.15
- PyTorch 2.12.1 mit CUDA 13.0
- Transformers 5.9.0
- PEFT 0.18.1
- TRL 1.4.0
- Accelerate 1.12.0
- Datasets 4.8.5
- Sentence Transformers 5.2.2
- FAISS 1.14.3
- FlashAttention 2.8.3.post1
- GPU: NVIDIA L40S
- Betriebssystem: Amazon Linux 2023

Der vollständige Environment-Freeze und die Hostinformationen dienen als Evidenz für die tatsächlich verwendete Umgebung.

## 10. Abgrenzung

Für die Reproduktion der Hauptbefunde sollen ausschließlich die in diesem Dokument genannten Trainingsconfigs, Run-Matrizen, Ergebnisdateien und finalen Adapter verwendet werden.

Andere im Repository vorhandene historische oder explorative Artefakte sind nicht ohne zusätzliche methodische Prüfung mit der autoritativen Hauptlinie gleichzusetzen.
