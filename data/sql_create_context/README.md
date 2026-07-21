# Final Adapter Datasets

Dieses Verzeichnis enthält ausschließlich die materialisierten Trainings- und Validierungsdateien, die für die drei veröffentlichten finalen LoRA-v2-Adapter verwendet wurden.

Spider-SQLite-Datenbanken, vollständige Rohdatensätze, Retrievalindizes und sonstige Zwischenartefakte sind nicht Bestandteil dieses Verzeichnisses.

## Dateien

| Datei | Modelllinie | Verwendung | Fälle | Größe | SHA-256 |
|---|---|---|---:|---:|---|
| `train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl` | Llama 3.2 3B Instruct | Training | 25000 | 70.17 MiB | `14f151ba086d183a139579871762992f699a195bc9300214b544947f8d73edb8` |
| `train_sft_qwen35_2b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl` | Qwen 3.5 2B Base | Training | 25000 | 32.99 MiB | `c4b72a87d175b79895081a83f525997b71a230fd9088a7f8c59c40673fa0a40d` |
| `train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl` | Qwen 3.5 9B Base | Training | 25000 | 32.99 MiB | `c4b72a87d175b79895081a83f525997b71a230fd9088a7f8c59c40673fa0a40d` |
| `val_sft_llama32_3b_instruct_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl` | Llama 3.2 3B Instruct | Validation | 2500 | 7.49 MiB | `917b0f16a45dc86ead42fea508595860b8700d6fb2c019c8125c3170d032bfb0` |
| `val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl` | Qwen 3.5 2B und 9B | Validation | 2500 | 3.54 MiB | `711b23a6dfca40234a33e9aca66506eb33df197f69b6f466fd875854bdb89c08` |

## Zusammensetzung

Die Trainingsdateien enthalten jeweils 25.000 fachliche Fälle:

```text
6.960 Spider Train + 18.040 SQL Create Context
```

Die Validierungsdateien enthalten jeweils 2.500 Fälle:

```text
700 train_others + 1.800 SQL Create Context
```

Spider Dev wurde nicht für Training, Early Stopping oder Checkpointauswahl verwendet.

Qwen 3.5 2B und Qwen 3.5 9B verwenden byteidentische Trainingsdaten. Für Llama 3.2 3B werden dieselben fachlichen Fälle in einer modellspezifischen nativen Chatserialisierung bereitgestellt.

## Datenformat

- `train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl`: JSON-Schlüssel des ersten Datensatzfalls: `chat_template_kwargs, id, messages, text`
- `train_sft_qwen35_2b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl`: JSON-Schlüssel des ersten Datensatzfalls: `id, text`
- `train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl`: JSON-Schlüssel des ersten Datensatzfalls: `id, text`
- `val_sft_llama32_3b_instruct_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl`: JSON-Schlüssel des ersten Datensatzfalls: `chat_template_kwargs, id, messages, text`
- `val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl`: JSON-Schlüssel des ersten Datensatzfalls: `id, text`

## Provenienz

Die materialisierten Dateien enthalten abgeleitete Beispiele aus Spider und `philschmid/sql-create-context-copy`. Nutzerinnen und Nutzer müssen die jeweiligen Bedingungen und Lizenzen der zugrunde liegenden Quellen beachten.

Die Dateihashes dienen zur Überprüfung, dass die veröffentlichten Dateien dem eingefrorenen Trainingsstand entsprechen.
