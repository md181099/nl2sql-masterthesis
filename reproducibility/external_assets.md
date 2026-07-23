# Externe Assets

Dieses Repository enthält den produktiven Quellcode, die relevanten Konfigurationen, Tests, Reproduzierbarkeitsinformationen sowie zentrale Ergebnis- und Run-Manifeste der Masterarbeit.

Große Basismodelle, vollständige Datensätze, SQLite-Datenbanken, Retrievalindizes und LoRA-Adaptergewichte werden nicht vollständig über das GitHub-Repository bereitgestellt. Die folgenden externen Assets werden für Training oder Evaluation benötigt.

## 1. Basismodelle

### Qwen 3.5 2B Base

- Hugging-Face-ID: `Qwen/Qwen3.5-2B-Base`
- verwendete Revision: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`
- Rolle: Basismodell der 2B-Modelllinie
- Verwendung: Base-Evaluation und Ausgangsmodell für LoRA-Fine-Tuning

### Llama 3.2 3B Instruct

- Hugging-Face-ID: `meta-llama/Llama-3.2-3B-Instruct`
- verwendete Revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Rolle: Ausgangsmodell der 3B-Modelllinie
- Verwendung: Evaluation des Ausgangsmodells und LoRA-Fine-Tuning
- Hinweis: Der Zugriff kann die Annahme der Nutzungsbedingungen auf Hugging Face erfordern.

### Qwen 3.5 9B Base

- Hugging-Face-ID: `Qwen/Qwen3.5-9B-Base`
- verwendete Revision: `68c46c4b3498877f3ef123c856ecfde50c39f404`
- Rolle: Basismodell der 9B-Modelllinie
- Verwendung: Base-Evaluation und Ausgangsmodell für LoRA-Fine-Tuning

## 2. Retrievalmodell

Für das semantische Few-Shot-Retrieval wurde folgendes Embeddingmodell verwendet:

- Hugging-Face-ID: `BAAI/bge-large-en-v1.5`
- Retrievalverfahren: FAISS `IndexFlatIP`
- finaler Retrievalpool: 6.960 Spider-Train-Beispiele
- lokaler Indexname: `spider_train_no_dev_overlap_bge_large_en_v15`
- Status der exakten Hugging-Face-Revision: noch in der finalen Release-Dokumentation zu ergänzen

Der Retrievalindex ist nicht vollständig im GitHub-Repository enthalten. Er kann anhand der bereitgestellten Retrievalskripte und des dokumentierten Spider-Train-Pools neu erzeugt werden.

## 3. Spider

Für die Evaluation wurde Spider Dev mit 1.032 Fällen verwendet.

Verwendung im Projekt:

- Spider Train: Bestandteil des LoRA-Trainingsdatensatzes und des Retrievalpools
- Spider Dev: ausschließlich für die finale Evaluation
- Spider Dev wurde nicht zur Checkpointauswahl verwendet
- SQLite-Datenbanken werden zur Execution-Match-Evaluation benötigt

Nicht im GitHub-Repository enthalten:

- vollständige Spider-SQLite-Datenbanken
- das ursprüngliche Spider-Archiv
- große lokal erzeugte Zwischenartefakte

Die genaue Downloadquelle, Archivversion und der Archivhash sollen vor dem finalen Release ergänzt werden.

## 4. SQL Create Context

Zusätzlich zu Spider Train wurde der Datensatz SQL Create Context für das LoRA-Fine-Tuning und die Validation verwendet.

- Hugging-Face-Dataset: `philschmid/sql-create-context-copy`
- Verwendung im Training: 18.040 Beispiele
- Verwendung in MixedVal2500-v2: 1.800 Beispiele
- Status der exakten Dataset-Revision und des Quelldateihashs: noch in der finalen Release-Dokumentation zu ergänzen

## 5. Finale Trainings- und Validierungsdaten

Die finale Trainingsmischung umfasst 25.000 Beispiele:

- 6.960 Beispiele aus Spider Train
- 18.040 Beispiele aus SQL Create Context

Die finale Validation MixedVal2500-v2 umfasst 2.500 Beispiele:

- 700 Beispiele aus `train_others`
- 1.800 Beispiele aus SQL Create Context

Spider Dev mit 1.032 Fällen wurde ausschließlich für die Ergebnisevaluation verwendet.

Kleine Manifeste und ausgewählte aufbereitete Dateien können im Repository enthalten sein. Große erzeugte JSONL-Dateien werden nur dann versioniert, wenn dies technisch und lizenzrechtlich vertretbar ist.

## 6. Finale LoRA-Adapter

Die drei finalen LoRA-v2-Adapter sollen separat über Hugging Face bereitgestellt werden:

- Qwen 3.5 2B LoRA-v2
- Llama 3.2 3B Instruct LoRA-v2
- Qwen 3.5 9B LoRA-v2

Die endgültigen Hugging-Face-Repository-IDs und Commitrevisionen werden nach dem Upload ergänzt.

Bekannte SHA-256-Hashes der Adaptergewichte:

| Modelllinie | SHA-256 von `adapter_model.safetensors` |
|---|---|
| Qwen 3.5 2B | `6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3` |
| Llama 3.2 3B Instruct | `fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d` |
| Qwen 3.5 9B | `dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0` |

## 7. Nicht über GitHub bereitgestellte große Artefakte

Bewusst nicht Bestandteil des normalen GitHub-Repositories sind insbesondere:

- vollständige virtuelle Umgebungen
- Hugging-Face-Modellcaches
- historische Adapter und Checkpoints
- Optimizer-Zustände
- vollständige Spider-Datenbanken
- Spider-Archive
- große FAISS-Indizes
- große Rohresultate
- vollständige Retrievaltraces
- temporäre Logs und Caches

Diese Trennung reduziert die Repositorygröße und trennt den reproduzierbaren Codebestand von großen externen Forschungsartefakten.

## 8. Noch zu ergänzende Release-Angaben

Vor dem finalen Release sind folgende Angaben zu vervollständigen:

- exakte Revision von `BAAI/bge-large-en-v1.5`
- genaue Spider-Downloadquelle und Archivhash
- exakte Revision von `philschmid/sql-create-context-copy`
- Hashes der verwendeten Quelldateien
- Hugging-Face-IDs und Commitrevisionen der drei finalen Adapter
- gegebenenfalls Download- oder Buildweg des finalen Retrievalindex
