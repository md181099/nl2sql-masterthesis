# Data Provenance and Licenses

Dieses Dokument beschreibt Herkunft, Lizenzstatus und projektspezifische Verarbeitung der im Repository enthaltenen oder für die Reproduktion benötigten Datensätze.

Es ersetzt nicht die Lizenztexte und Nutzungsbedingungen der jeweiligen Originalquellen. Bei Abweichungen sind die Angaben der ursprünglichen Datenanbieter maßgeblich.

## 1. Spider 1.0

### Herkunft

- Datensatz: Spider 1.0
- Herausgeber: Yale LILY / Spider-Projekt
- Offizielle Projektseite: https://yale-lily.github.io/spider
- Wissenschaftliche Referenz: Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018

Der offizielle Download befindet sich auf der Projektseite im Abschnitt **Getting Started** unter **Spider Dataset**.

### Lizenz

Die offizielle Spider-Projektseite weist den Datensatz als unter **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** veröffentlicht aus:

https://creativecommons.org/licenses/by-sa/4.0/

Die Bereitstellung und Weitergabe Spider-abgeleiteter Artefakte erfolgt unter Beachtung der Attributions- und Share-Alike-Anforderungen der Originalquelle.

### Im Repository enthaltene Spider-abgeleitete Artefakte

Unter anderem:

```text
data/testcases_spider_dev_full.jsonl
data/spider/spider_data/tables.json
data/fewshot_static/
data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/
```

Der Retrievalpool `metadata.jsonl` enthält 6.960 Beispiele aus Spider Train. Das Testset `data/testcases_spider_dev_full.jsonl` materialisiert die 1.032 Fälle von Spider Dev in der projektspezifischen Evaluationsstruktur.

### Projektspezifische Verarbeitung

Die enthaltenen Artefakte sind teilweise nicht unveränderte Kopien der ursprünglichen Spider-Dateien. Zu den projektspezifischen Verarbeitungsschritten gehören insbesondere:

- Materialisierung stabiler Case-IDs,
- Ergänzung relationaler Schemabeschreibungen,
- Umwandlung in Full-Chat- beziehungsweise Evaluationsformate,
- Ausschluss direkter Spider-Dev-Überlappungen aus Trainings- und Retrievalbeständen,
- Aufbau eines BGE-/FAISS-Retrievalindex,
- Erstellung statischer Few-Shot-Demonstrationen,
- Erstellung von Manifesten, Hashes und Retrievalaudits.

Die inhaltlichen Fragen, SQL-Abfragen und Schemadaten stammen weiterhin aus Spider und unterliegen den Bedingungen der ursprünglichen Spider-Lizenz.

### Nicht im Repository enthalten

Die vollständigen Spider-SQLite-Datenbanken und das ursprüngliche Spider-Archiv sind aufgrund ihres Umfangs nicht enthalten. Sie müssen separat über die offizielle Spider-Projektseite bezogen und unter folgendem Pfad abgelegt werden:

```text
data/spider/spider_data/database/
```

## 2. SQL Create Context

### Herkunft

- Verwendete Quelle: `philschmid/sql-create-context-copy`
- Hugging-Face-Dataset: https://huggingface.co/datasets/philschmid/sql-create-context-copy
- Angegebene Ursprungsquelle: `b-mc2/sql-create-context`
- Inhaltliche Grundlage laut Dataset Card: unter anderem Spider und WikiSQL
- Verwendete Quelldatei: `sql_create_context_v4.json`

### Lizenz

Die Dataset Card von `philschmid/sql-create-context-copy` weist den Datensatz als unter **Creative Commons Attribution 4.0 International (CC BY 4.0)** veröffentlicht aus:

https://creativecommons.org/licenses/by/4.0/

Zusätzlich können für Bestandteile, die aus Spider oder WikiSQL abgeleitet wurden, die Lizenz- und Attributionsbedingungen der jeweiligen Ursprungsdatensätze relevant bleiben.

### Im Repository enthaltene abgeleitete Artefakte

Teile der finalen Trainings- und Validierungsdateien enthalten ausgewählte und transformierte SQL-Create-Context-Beispiele, unter anderem:

```text
data/sql_create_context/train_sft_qwen35_2b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl

data/sql_create_context/train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl

data/sql_create_context/train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl

data/sql_create_context/val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl

data/sql_create_context/val_sft_llama32_3b_instruct_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl
```

### Projektspezifische Verarbeitung

Zu den Verarbeitungsschritten gehören insbesondere:

- deterministische Auswahl mit Seed 42,
- Mischung mit Spider-Train-Beispielen,
- Komplexitätsanreicherung und Schemaharmonisierung,
- Materialisierung als Full-Chat-SFT-Datensatz,
- modellfamilienabhängige Chat-Templates,
- Aufbau der MixedVal2500-v2-Validierungsbestände,
- Ausschluss direkter Train- und Spider-Dev-Überlappungen,
- Korrektur beziehungsweise Vereinheitlichung von Schema-Headern.

Die Transformationen erzeugen projektspezifische Dateiformate und Zusammenstellungen. Sie übertragen jedoch nicht das Eigentum an den zugrunde liegenden Beispielen und ersetzen nicht die Lizenzbedingungen der Originalquellen.

## 3. Gemischte und abgeleitete Datensätze

Mehrere Dateien kombinieren Inhalte aus Spider und SQL Create Context. Für solche gemischten Artefakte sind die Herkunft und Lizenzanforderungen aller enthaltenen Ausgangsquellen zu berücksichtigen.

Insbesondere gilt:

- Spider-abgeleitete Bestandteile: CC BY-SA 4.0 gemäß offizieller Spider-Projektseite,
- SQL-Create-Context-Bestandteile: CC BY 4.0 gemäß Dataset Card,
- projektspezifischer Code, Konfigurationen, Manifeste und Auswertungen sind von den Lizenzen der enthaltenen Fremddaten zu unterscheiden.

Aus Vorsichtsgründen sollten gemischte Dateien nicht so behandelt werden, als seien sämtliche Bestandteile ausschließlich unter einer einzigen, weniger restriktiven Lizenz verfügbar.

## 4. Attribution

Bei wissenschaftlicher oder sonstiger Weiterverwendung sind mindestens die ursprünglichen Datensatzanbieter sowie die einschlägigen wissenschaftlichen Veröffentlichungen zu nennen.

Empfohlene Attribution:

- Spider: Tao Yu et al., EMNLP 2018, offizielle Spider-Projektseite.
- SQL Create Context: `b-mc2/sql-create-context` und die verwendete Kopie `philschmid/sql-create-context-copy`.
- Dieses Repository für die projektspezifische Selektion, Transformation, Retrievalaufbereitung und Evaluationsstruktur.

## 5. Keine Rechtsberatung

Diese Dokumentation dient der transparenten wissenschaftlichen Herkunfts- und Lizenzkennzeichnung. Sie stellt keine Rechtsberatung dar. Nutzer sind selbst dafür verantwortlich, die aktuellen Lizenztexte und Nutzungsbedingungen der Originalquellen zu prüfen.
