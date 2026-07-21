# LLM NL2SQL Thesis Project

Masterarbeitsprojekt zur Evaluation großer Sprachmodelle für die Erzeugung von SQL-Abfragen aus natürlichsprachlichen Fragen. Verglichen werden drei Modelllinien als Ausgangsmodell und als LoRA-v2-Variante auf dem Spider-Benchmark.

Der Versuchsaufbau umfasst Zero-Shot- und Few-Shot-Prompting, semantisches Retrieval, strukturbasiertes Reranking sowie eine ausführungsbasierte Evaluation gegen SQLite. Die autoritative Ergebnisbasis besteht aus 48 Hauptläufen und einer additiven Erweiterung mit 36 `k=3`-Läufen.

## Prerequisites

### Linux GPU Host / AWS EC2

Die finale Umgebung wurde unter Linux auf einer NVIDIA L40S mit CUDA 13.0 ausgeführt. Training und vollständige Evaluation benötigen eine CUDA-fähige NVIDIA-GPU sowie ausreichend GPU-, Arbeits- und Plattenspeicher.

Der dokumentierte Host verwendet den Pfad `/home/ec2-user/nl2sql_testbench`. AWS-spezifische Bereitstellung, Netzwerk- und Zugriffsverwaltung sind nicht Bestandteil dieses Repositories und müssen extern eingerichtet werden.

### Hugging Face Account

Je nach lokalem Cache werden benötigt:

- ein Hugging-Face-Zugriffstoken außerhalb des Projektordners,
- gegebenenfalls Zugriff auf gated Model-Repositories,
- die drei verwendeten Modell-Snapshots,
- das Retrievalmodell `BAAI/bge-large-en-v1.5`.

Die vollständigen Basismodellgewichte liegen nicht im Projektverzeichnis.

### Spider and SQLite

Erforderlich sind:

- Spider Train,
- Spider Dev,
- `tables.json`,
- die Spider-SQLite-Datenbanken unter `data/spider/spider_data/database/`.

Die Evaluation verwendet lokale SQLite-Dateien und benötigt keinen separaten Datenbankserver.

### Python Environment

Die autoritative Umgebung ist:

```text
Python 3.11.15
/home/ec2-user/nl2sql_testbench/.venv_flash
```

Der datierte Paketfreeze liegt unter:

```text
reproducibility/final_freeze_20260721/pip-freeze-venv-flash-20260721.txt
```

## Setup

Projektverzeichnis öffnen und Umgebung aktivieren:

```bash
cd /home/ec2-user/nl2sql_testbench
source .venv_flash/bin/activate
```

Python und Paketkonsistenz prüfen:

```bash
which python
python --version
python -m pip --version
python -m pip check
```

CUDA und GPU prüfen:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

FlashAttention prüfen:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

Zentrale Datenartefakte prüfen:

```bash
test -f data/testcases_spider_dev_full.jsonl
test -f data/spider/spider_data/tables.json
test -f data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss
wc -l data/testcases_spider_dev_full.jsonl
```

Vorhandene Modell-Snapshots prüfen:

```bash
test -d /home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
test -d /home/ec2-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
test -d /home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404
```

> `requirements.txt` ist historisch. `requirements_current.txt` dokumentiert die zweite Umgebung `.venv`. Für die finale Umgebung `.venv_flash` ist ausschließlich der datierte Freeze unter `reproducibility/final_freeze_20260721/` maßgeblich.

## Structure

| Verzeichnis | Funktion |
|---|---|
| `src/` | Kernimplementierung für Datenaufbereitung, Training, Retrieval und Evaluation |
| `scripts/` | Statistik-, Fehleranalyse- und Plotprogramme |
| `configs/` | Trainings- und Evaluationskonfigurationen |
| `data/` | Spider, SQL Create Context, SFT-Dateien, SQLite-Datenbanken und Retrievalindizes |
| `adapters/` | LoRA-Adapter, Checkpoints und Trainer-States |
| `results/` | Evaluationsresultate und Trainingsdiagnostik |
| `logs/` | Lauf- und Gruppenprotokolle |
| `audits/` | technische und methodische Nachweise sowie Runinventare |
| `figures/` | thesisgeeignete PDF- und PNG-Abbildungen |
| `reproducibility/` | datierte Umgebungs- und Freeze-Artefakte |

Autoritative Läufe werden über Config, Modellrolle, Ergebnis-CSV, Metadata-Summary und gegebenenfalls Retrievaltrace zugeordnet. Das zentrale Register ist:

```text
audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv
```

## Used Data

### Spider

Der finale Trainingsbestand enthält alle 6.960 Spider-Train-Fälle. Für die Evaluation wird der autoritative Bestand `data/testcases_spider_dev_full.jsonl` mit 1.032 Spider-Dev-Fällen verwendet.

Die zugehörigen Schemata stammen aus `tables.json`; die Abfragen werden gegen die lokalen Spider-SQLite-Datenbanken ausgeführt. Spider Dev wird ausschließlich für die Ergebnisevaluation verwendet.

### SQL Create Context

Zusätzliche Trainingsbeispiele stammen aus `philschmid/sql-create-context-copy`. Der finale Trainingsbestand enthält 18.040 ausgewählte Fälle dieser Quelle.

### Training Dataset

Der finale Trainingsdatensatz umfasst 25.000 Beispiele:

```text
6.960 Spider Train + 18.040 SQL Create Context
```

Die Auswahl und Mischung sind mit Seed 42 deterministisch materialisiert. Qwen 2B und Qwen 9B verwenden byteidentische SFT-Dateien; Llama verwendet dieselben fachlichen Fälle in nativer Chatserialisierung.

### Validation Dataset

MixedVal2500-v2 umfasst 2.500 Beispiele:

```text
700 train_others + 1.800 SQL Create Context
```

MixedVal2500-v2 wird für die epochische Validierung, Early Stopping und Checkpointauswahl verwendet. Spider Dev wird nicht zur Modellauswahl herangezogen.

### Retrieval Pool

Der autoritative Retrievalpool enthält ausschließlich die 6.960 Spider-Train-Fälle und keine freigegebenen direkten Spider-Dev-Overlaps.

```text
data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15
```

## Used Models

| Modell | Modell-ID | Rolle | Zweck |
|---|---|---|---|
| Qwen 3.5 2B Base | `Qwen/Qwen3.5-2B-Base` | Base und LoRA v2 | kleine Qwen-Modelllinie |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Instruct-Basis und LoRA v2 | alternative Modellfamilie |
| Qwen 3.5 9B Base | `Qwen/Qwen3.5-9B-Base` | Base und LoRA v2 | größere Qwen-Modelllinie |
| BGE Large EN v1.5 | `BAAI/bge-large-en-v1.5` | Retriever | semantische Demonstrationsauswahl |

Die finalen Adapter liegen unter `adapters/`. Ihre eindeutige Zuordnung erfolgt über die drei finalen Trainingsconfigs und die Trainingsmetadaten.

## Used Training Techniques

### Parameter-Efficient Fine-Tuning

Die Ausgangsgewichte der Sprachmodelle bleiben eingefroren. Trainiert werden nur zusätzliche Adapterparameter, wodurch Speicher- und Rechenbedarf gegenüber einem vollständigen Fine-Tuning reduziert werden.

### LoRA

Alle drei finalen Modelllinien verwenden:

```text
r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = all-linear
Quantisierung = keine
```

Die konkreten linearen Zielmodule werden durch PEFT passend zur jeweiligen Modellarchitektur aufgelöst.

### Full-Chat Supervised Fine-Tuning

Trainiert wird mit Full-Chat-Loss auf materialisierten System-, User- und Assistant-Nachrichten. Die Daten werden mit BFD-Packing und einer maximalen Sequenzlänge von 2.048 Tokens verarbeitet.

### Checkpoint Selection

Maximal fünf Epochen sind konfiguriert; Early Stopping beendete alle finalen Trainings nach drei Epochen. Der jeweils beste Checkpoint stammt aus Epoche 1 und wird anhand des niedrigsten Full-Chat-`eval_loss` auf MixedVal2500-v2 wiederhergestellt.

Spider Dev wurde nicht zur Checkpointauswahl verwendet.

### Prompting and Retrieval

Die Hauptuntersuchung umfasst Zero Shot, ein statisches Seed-42-Beispiel, Dynamic Top-1, Similarity-Gates bei 0,70 und 0,85 sowie strukturbasiertes Reranking der semantischen Top-10.

Die additive `k=3`-Erweiterung verwendet die entsprechenden dynamischen Bedingungen mit drei Demonstrationen. Nicht erfüllte Gates führen vollständig auf Zero Shot zurück. Die Gold-SQL des Zielfalls wird beim Reranking nicht verwendet.

## Evaluation Metrics Hints

- **Execution Match Accuracy (EMA):** Primärmetrik. Gold- und Prediction-SQL müssen auf der vorhandenen SQLite-Instanz dieselbe normalisierte Ergebnisrelation liefern.
- **Execution Success Rate (ESR):** Anteil der Vorhersagen, die ohne Ausführungsfehler ausgeführt werden können. Ausführbarkeit bedeutet nicht automatisch inhaltliche Korrektheit.
- **String Exact Match:** Direkte textuelle Übereinstimmung von Referenz- und Prediction-SQL.
- **Normalized Exact Match:** Übereinstimmung nach der projektinternen SQL-Normalisierung.
- **Character Accuracy:** Zeichenbasierte Ähnlichkeit zwischen Referenz und Vorhersage.
- **Token Accuracy:** Tokenbasierte Ähnlichkeit zwischen Referenz und Vorhersage.
- **Laufzeit und Tokenanzahl:** Gespeichert werden Generationszeit sowie Prompt-, Completion- und Gesamttokens.
- **Retrieval Similarity:** Ähnlichkeitswert des normalisierten BGE-Embeddingraums; keine kalibrierte Wahrscheinlichkeit.

Gepaarte Ergebnisvergleiche verwenden die bereits festgelegten McNemar-, Holm- und Bootstrapverfahren. Die vollständige statistische Methodik ist in den Audits dokumentiert.

## Showcase || Pipeline Hints

### 00 – Environment and Model Setup

- `.venv_flash` aktivieren.
- Python, CUDA, GPU und FlashAttention prüfen.
- Basismodell-Snapshots, Adapter, Spider-Daten und Retrievalindex prüfen.

### 01 – Prepare SQL Create Context

`src/01_prepare_sqlcreatecontext_dataset.py` importiert und materialisiert `philschmid/sql-create-context-copy` mit Seed 42.

```bash
.venv_flash/bin/python src/01_prepare_sqlcreatecontext_dataset.py
```

Der Schritt kann einen Dataset-Download auslösen und ist nur erforderlich, wenn die materialisierten Quelldateien fehlen.

### 02 – Prepare Spider Dev

`src/00_prepare_spider_subset.py` erzeugt den projektinternen Spider-Dev-Bestand mit stabilen Case-IDs, Schema und SQLite-Pfad.

```bash
.venv_flash/bin/python src/00_prepare_spider_subset.py \
  --out_train /tmp/nl2sql_traincases_unused.jsonl \
  --out_test /tmp/nl2sql_testcases_unused.jsonl \
  --out_test_full data/testcases_spider_dev_full.jsonl \
  --manifest data/spider/subset_manifest_full_dev.json
```

Dieser Builder ist für eine frische Rekonstruktionskopie gedacht. Der vorhandene autoritative Testbestand darf nicht überschrieben werden.

### 03 – Build Training Dataset

`src/04_build_spider_sqlcc_complexity_mix.py` wählt 18.040 SQLCC-Fälle aus und mischt sie deterministisch mit 6.960 Spider-Train-Fällen.

```bash
.venv_flash/bin/python src/04_build_spider_sqlcc_complexity_mix.py
```

Das Ergebnis umfasst 25.000 Beispiele und ein zugehöriges Manifest.

### 04 – Build SFT and Validation Data

Qwen-SFT-Dateien werden mit `src/02_make_sft_dataset_v1_clean_full_chat.py` erzeugt. Die erforderlichen Ein- und Ausgabepfade sind über dessen CLI und die vorhandenen Manifeste dokumentiert:

```bash
.venv_flash/bin/python src/02_make_sft_dataset_v1_clean_full_chat.py --help
```

MixedVal2500-v2 erzeugen:

```bash
.venv_flash/bin/python src/build_qwen35_mixed_validation_trainothers700_sqlcc1800.py --write
.venv_flash/bin/python src/build_qwen35_mixed_validation_v2_schemaheaderfix.py --write
```

Native Llama-Serialisierung erzeugen oder prüfen:

```bash
.venv_flash/bin/python scripts/prepare_llama32_3b_native_datasets.py --create
.venv_flash/bin/python scripts/prepare_llama32_3b_native_datasets.py --verify-only
```

### 05 – Build or Check Retrieval Index

Offizieller Indexpfad:

```text
data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15
```

Vorhandenen Index prüfen:

```bash
test -f data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss
wc -l data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/metadata.jsonl
```

Index in einer frischen Rekonstruktionskopie neu erzeugen:

```bash
.venv_flash/bin/python src/08_build_spider_train_dynamic_fewshot_index.py
```

Der Neuaufbau lädt BGE und berechnet Embeddings. Er darf den vorhandenen autoritativen Index nicht überschreiben.

### 06 – Train Models

Autoritativer Trainingseintrittspunkt:

```text
src/07_lora_finetune_sft_v1_clean.py
```

Qwen 2B:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_flash/bin/python3 src/07_lora_finetune_sft_v1_clean.py --config configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json
```

Llama 3B:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_flash/bin/python3 src/07_lora_finetune_sft_v1_clean.py --config configs/train_lora_llama32_3b_instruct_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json
```

Qwen 9B:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_flash/bin/python3 src/07_lora_finetune_sft_v1_clean.py --config configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json
```

Diese Befehle starten vollständige GPU-Trainings und dürfen nicht auf vorhandene finale Adapterroots schreiben.

### 07 – Evaluate Models

Runner:

```text
src/06_batch_run.py
src/06_batch_run_dynamic_k3_v1.py
src/06_batch_run_dynamic_k3_sqltimeout_v3.py
```

Alle 84 autoritativen Config-/Runzuordnungen stehen in:

```text
audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv
```

Beispiel für einen vollständigen Zero-Shot-Lauf:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_flash/bin/python3 src/06_batch_run.py --config configs/eval_qwen35_2b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_zero_shot_maxinput2048_full_aliasnames.json
```

Jeder autoritative Vollrun enthält 1.032 Fälle. Teilresultate werden nicht mit späteren Läufen kombiniert.

### 08 – Analyze and Visualize Results

Zentrale Statistikskripte:

```text
scripts/analyze_cross_model_complete_8x8_synthesis.py
scripts/analyze_k3_final_results_and_k1_vs_k3_statistics_20260718.py
```

Zentrale Fehleranalyseskripte:

```text
scripts/analyze_cross_model_zero_shot_error_taxonomy.py
scripts/analyze_complete_24_lora_run_error_profiles.py
scripts/analyze_k3_error_analysis_extension_20260718.py
```

Thesisgeeignete Plotprogramme liegen unter `scripts/`; erzeugte Abbildungen liegen unter `figures/`. Die Analyseprogramme dürfen nur mit freien additiven Zielpfaden ausgeführt werden.

Ausführliche technische, methodische und statistische Nachweise befinden sich unter `audits/` und `reproducibility/`.

## Important Notes

- Autoritative Daten, Adapter, Ergebnisse, Summaries und Traces niemals überschreiben.
- Training, vollständige Evaluation und Indexaufbau sind GPU- beziehungsweise rechenintensiv.
- Spider Dev ausschließlich zur Evaluation, nicht zur Checkpointauswahl verwenden.
- Basismodellgewichte liegen außerhalb des Projektverzeichnisses und müssen gegebenenfalls erneut bereitgestellt werden.
- Keine AWS-Schlüssel, Hugging-Face-Tokens, SSH-Schlüssel oder andere Secrets im Projekt speichern.
- Historische Configs und Adapter nicht mit den finalen LoRA-v2- und Runinventaren vermischen.
- Vollständige Nachweise, Hashes und Einschränkungen stehen unter `audits/` und `reproducibility/`.
- Das Projekt dient wissenschaftlichen Zwecken und besitzt keine produktive Softwaregarantie.
