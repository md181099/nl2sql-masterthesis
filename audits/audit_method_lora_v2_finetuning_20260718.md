# Read-only-Methodenaudit: LoRA-v2-Fine-Tuning

**Datum:** 18.07.2026  
**Projektwurzel:** `/home/ec2-user/nl2sql_testbench`  
**Zweck:** belastbare Methodengrundlage fuer Abschnitt 5.6.3 der Masterarbeit  
**Arbeitsmodus:** read-only gegen alle bestehenden Projektartefakte; ausschließlich additive Ausgabe dieses Audits und der zugehoerigen Derived-Dateien

## 1. Executive Summary

Die drei finalen LoRA-v2-Adapter sind eindeutig identifiziert. Sie wurden mit demselben fachlichen Trainingsdesign trainiert: 25.000 Full-Chat-Beispiele, eine vom Training getrennte MixedVal2500-v2-Validierung, `r=8`, `alpha=16`, LoRA-Dropout 0,05, `all-linear`, FP16, FlashAttention2, BFD-Packing bis 2.048 Tokens, effektive Batchgroesse 8 und ein maximales Budget von fuenf Epochen. Early Stopping beendete alle drei Laeufe nach der dritten Epoche; jeweils wurde der Zustand nach Epoche 1 als bester Checkpoint anhand des gepackten Full-Chat-Validierungsverlusts ausgewaehlt.

Entgegen einer moeglichen Fehlinterpretation des Namens `v2` verwendete die finale Linie **keinen Completion-only-Loss**. Der tatsaechlich ausgefuehrte Einstiegspunkt war `src/07_lora_finetune_sft_v1_clean.py`; dieser trainierte System-, Nutzer-/Schema-/Frage- und Assistant-/SQL-Tokens gemeinsam. Nur Paddingpositionen und die Startpositionen gepackter Dokumente erhielten das Label `-100`. Das importierte Modul `src/07_lora_finetune_sft_v2.py` lieferte Hilfsfunktionen, war aber nicht der ausgefuehrte Completion-only-Hauptpfad.

Die Trainings- und Validierungsdaten enthalten keine Spider-Dev-Faelle. Direkte projektseitige ID-, Question-, SQL- und Question-SQL-Paarpruefungen fanden weder Train-Validation- noch Train/Validation-Spider-Dev-Ueberschneidungen. Diese Aussage gilt fuer die kontrollierten Projektdateien und ist kein Nachweis gegen unbekannte Vortrainingskontamination.

Die Adapter- und Checkpointprovenienz ist bestaetigt. Dokumentationswarnungen verbleiben vor allem fuer die beiden Qwen-Linien: Die Modellrevision und `sys.executable` wurden nicht direkt in deren Trainingsmetadaten persistiert. Die Snapshots wurden deshalb aus zeitnahen Preflights, lokalem Cache und autoritativen Evaluationsmanifesten rekonstruiert. Außerdem existiert kein separater persistierter Konsolenlog der drei Laeufe. Diese Luecken beeintraechtigen die Identitaet der Adapter nicht, muessen aber als Provenienzgrenze ausgewiesen werden.

**Gesamtentscheidung:** Die Methode ist fuer Abschnitt 5.6.3 verwendbar, sofern die Checkpointauswahl als projektbezogener Befund und nicht als allgemeine Optimalitaetsaussage beschrieben wird und die Qwen-Provenienzrekonstruktion transparent bleibt.

## 2. Autoritative Adapter und ausgeschlossene Varianten

### 2.1 Autoritative Adapter

| Modelllinie | Ausgangsmodell | Revision/Snapshot | Autoritativer Adapterroot | Root-Gewicht-SHA256 | Status |
|---|---|---|---|---|---|
| Qwen 3.5 2B | `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | `adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5` | `6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3` | `AUTHORITATIVE_WITH_DOCUMENTATION_WARNINGS` |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | `adapters/llama32_3b_instruct/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5` | `fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d` | `AUTHORITATIVE` |
| Qwen 3.5 9B | `Qwen/Qwen3.5-9B-Base` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | `adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5` | `dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0` | `AUTHORITATIVE_WITH_DOCUMENTATION_WARNINGS` |

In allen drei Wurzeln sind `adapter_model.safetensors` und `adapter_config.json` vorhanden. Die Root-Gewichte sind byteidentisch zu den Gewichten des jeweils ausgewaehlten Best-Checkpoints. Die Root-Verzeichnisse repraesentieren daher nicht den zuletzt trainierten Zustand nach Epoche 3, sondern den durch `load_best_model_at_end` ausgewaehlten Zustand nach Epoche 1.

### 2.2 Tatsaechlich verwendete Trainingsconfigs

| Modelllinie | Config | SHA256 |
|---|---|---|
| Qwen 3.5 2B | `configs/train_lora_qwen35_2b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json` | `020662e3158f1d848e0c55976197b57a211b4e92fa96ebc6aca5dd453542b327` |
| Llama 3.2 3B Instruct | `configs/train_lora_llama32_3b_instruct_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json` | `81001f1a2b6287d589412bab658d50290fa1ba33c9e5557d44b9ff04a1c4282b` |
| Qwen 3.5 9B | `configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json` | `0bfce20d1e97f0b42b61d3db67679e3feef46b94a58a09147e4a5fb82240815e` |

### 2.3 Explizite Ausschlussliste

Nicht in die Beschreibung der finalen LoRA-v2-Hauptlinie eingehen duerfen:

- alle `lora_v1_*`-Adapter und zugehoerigen Configs,
- `lora_clean_split_*`-Experimente,
- `testmode_*`, Smoke- und Teilmengentrainings,
- archivierte 20k-, Full-Source-, Grounding- und Relevance-Experimente,
- historische Trainings mit `max_length=1024`,
- Completion-only- und No-Packing-Ablationen,
- Qwen-9B-Linien mit `r=4/alpha=8`, alternativer Lernrate oder anderer Validation,
- Zwei- und Vier-Epochen-Fortsetzungen sowie nicht ausgewaehlte Checkpoints,
- historische SQL-Loss-Nachanalysen als angebliche Checkpointauswahlgrundlage.

Die vollstaendige, gruppierte Ausschlussliste steht in `audits/derived/lora_v2_training_artifact_inventory_20260718.csv`. Kein historischer Adapter wurde anhand seines Dateinamens als Ersatz fuer einen finalen v2-Adapter zugelassen.

## 3. Trainings- und Validierungsdaten

### 3.1 Training

| Modelllinie | Dataset | SHA256 | Faelle | Zusammensetzung |
|---|---|---|---:|---|
| Qwen 3.5 2B | `data/sql_create_context/train_sft_qwen35_2b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl` | `c4b72a87d175b79895081a83f525997b71a230fd9088a7f8c59c40673fa0a40d` | 25.000 | 6.960 Spider Train + 18.040 SQL Create Context |
| Llama 3.2 3B | `data/sql_create_context/train_sft_llama32_3b_instruct_full_chat_v2_old25k_no_dev_overlap_seed42.jsonl` | `14f151ba086d183a139579871762992f699a195bc9300214b544947f8d73edb8` | 25.000 | dieselben 6.960 + 18.040 semantischen Faelle, Llama-nativ serialisiert |
| Qwen 3.5 9B | `data/sql_create_context/train_sft_qwen35_9b_base_full_chat_v1_clean_anti_overjoin_mix_spider_train_sqlcc_spider_schema_harmonized_complexity_enriched_25k_seed42_no_dev_overlap.jsonl` | `c4b72a87d175b79895081a83f525997b71a230fd9088a7f8c59c40673fa0a40d` | 25.000 | 6.960 Spider Train + 18.040 SQL Create Context |

Die beiden Qwen-Dateien sind byteidentisch. Die Llama-Datei enthaelt dieselben semantischen Faelle in derselben fachlichen Zusammensetzung, aber eine modellnative Serialisierung. Der Datenseed ist 42.

### 3.2 Validierung

| Modelllinie | Dataset | SHA256 | Faelle | Zusammensetzung |
|---|---|---|---:|---|
| Qwen 2B/9B | `data/sql_create_context/val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl` | `711b23a6dfca40234a33e9aca66506eb33df197f69b6f466fd875854bdb89c08` | 2.500 | 700 `train_others` + 1.800 SQL Create Context |
| Llama 3B | `data/sql_create_context/val_sft_llama32_3b_instruct_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl` | `917b0f16a45dc86ead42fea508595860b8700d6fb2c019c8125c3170d032bfb0` | 2.500 | dieselben semantischen 700 + 1.800 Faelle, Llama-nativ serialisiert |

MixedVal2500-v2 korrigiert gegenueber der historischen Variante den doppelten Schemaheader. Die Checkpointauswahl verwendete ausschließlich diesen Validierungsbestand, nicht Spider Dev.

### 3.3 Ueberschneidungspruefungen

Die vorhandenen Manifeste und Overlap-Audits weisen fuer die kontrollierten Projektdateien aus:

- Train gegen Validation: keine direkten ID-, Question-, SQL- oder Question-SQL-Paar-Ueberschneidungen,
- Train gegen Spider Dev: keine entsprechenden direkten Ueberschneidungen,
- Validation gegen Spider Dev: keine entsprechenden direkten Ueberschneidungen,
- Spider Dev wurde weder trainiert noch zur Checkpointauswahl verwendet,
- `data/testcases.jsonl` war keine Trainings- oder Validierungsgrundlage.

Diese harten Vergleiche erfassen keine unbekannte Vortrainingskontamination oder semantische Paraphrasen jenseits der implementierten Pruefregeln.

## 4. Promptformat und Tokenisierung

### 4.1 Gemeinsame fachliche Struktur

Der Systemprompt lautet exakt:

```text
You are an SQLite SQL generator. Return exactly one valid SQLite query and nothing else. Output SQL only. Do not explain. Do not reason. Do not use markdown. Do not use comments. Use only tables and columns from the provided schema. Use only tables required by the question. Do not join tables unless their columns are required. If one table contains all required columns and filters, use only that table. The query must start with SELECT or WITH and end with a semicolon.
```

Der Nutzerinhalt folgt diesem Muster:

```text
Database schema:
{schema}

Rules:
- Use only the tables and columns from the schema.
- Output exactly ONE SQLite read query.
- Start directly with SELECT or WITH.
- End with a semicolon.
- Do NOT explain anything.
- Do NOT use markdown.

Question:
{question}

SQL:
```

Die Referenz-SQL steht als Assistant-Inhalt hinter dem Nutzerblock. Sie wird auf genau ein `SELECT`- oder `WITH`-Statement mit abschließendem Semikolon ausgerichtet.

### 4.2 Qwen-Serialisierung

Qwen 2B und Qwen 9B nutzen das projektspezifische Format `qwen_sqlctx_chatml` mit manuellen System-, User- und Assistant-ChatML-Bloecken. Der Assistant-Block beginnt unmittelbar vor der Referenz-SQL und endet mit `<|im_end|>`. Es wird kein `<think>`-Block eingefuegt.

### 4.3 Llama-Serialisierung

Llama 3B nutzt `llama32_instruct_native_chat` und das native Chattemplate der festgelegten Modellrevision. Die Reihenfolge ist ebenfalls System, User, Assistant. Die Serialisierung verwendet einmalig BOS und beendet Rollen mit `<|eot_id|>`; der feste Datumswert des nativen Templates ist in Training und Evaluation konsistent. Die fachliche Nutzlast ist zu Qwen gleich, die Special-Token-Serialisierung modellabhaengig.

### 4.4 Sequenzlaengen und Truncation

| Modellformat | maximale Trainingstokens vor Packing | maximale Validierungstokens | Faelle ueber 2.048 |
|---|---:|---:|---:|
| Qwen ChatML | 1.842 | 852 | 0 |
| Llama native chat | 1.829 | 848 | 0 |

Damit wurde kein einzelnes Trainings- oder Validierungsbeispiel am Limit von 2.048 Tokens gekuerzt.

## 5. Label- und Losslogik

Die finale v2-Linie verwendet **Full-Chat-Loss**:

- `completion_only_loss=false`,
- `assistant_only_loss=false`,
- Systemtokens gehen in den Loss ein,
- Nutzertokens einschließlich Schema, Regeln und Frage gehen in den Loss ein,
- Assistanttokens einschließlich Referenz-SQL gehen in den Loss ein,
- trainierbare Labels entsprechen an diesen Positionen den Input-Token-IDs,
- Paddingpositionen erhalten `-100`,
- die erste Position jedes gepackten Dokuments erhaelt ebenfalls `-100`, damit kein kuenstliches Vorhersageziel ueber die Dokumentgrenze entsteht.

Der Einstiegspunkt prueft die Labels stichprobenartig und bricht bei unerwarteten `-100`-Positionen außerhalb von Padding oder gepackten Sequenzstarts ab. Eine Beschreibung als Completion-only- oder SQL-only-Loss waere falsch. Das Ziel ist zwar SQL-Generierung, der optimierte Tokenloss umfasst aber den vollstaendigen Chat.

## 6. Packing und Sequenzbehandlung

Alle drei Linien verwenden:

- `packing=true`,
- `packing_strategy=bfd`,
- `max_length=2048`,
- FlashAttention2.

Die Laufzeitmetadaten melden:

| Modellformat | rohe Trainingsfaelle | gepackte Trainingssequenzen | rohe Validation | gepackte Validation |
|---|---:|---:|---:|---:|
| Qwen | 25.000 | 4.011 | 2.500 | 444 |
| Llama | 25.000 | 4.066 | 2.500 | 442 |

In TRL 1.4.0 aktiviert BFD-Packing automatisch den paddingfreien Datenpfad. `seq_lengths` bleiben erhalten, Positions-IDs beginnen an jeder Dokumentgrenze wieder bei null und die erste Position jedes Dokuments wird im Loss maskiert. Der verwendete FlashAttention-Pfad ist dadurch dokumentbewusst; gepackte Dokumente werden nicht als eine fachlich zusammenhaengende Unterhaltung behandelt. Dieser Befund ist aus der tatsaechlichen Codefolge und der installierten Bibliotheksversion rekonstruiert. Da keine konkreten Trainingsbatch-Tensoren persistiert wurden, ist er als `IMPLEMENTATION-INFERRED`, nicht als unabhaengig numerisch replayter Laufzeitbeweis zu kennzeichnen.

Die serialisierten Einzelbeispiele enthalten bereits Rollenendmarker (`<|im_end|>` beziehungsweise `<|eot_id|>`), bevor sie gepackt werden.

## 7. LoRA-Konfiguration

### 7.1 Gemeinsame Parameter

| Parameter | Wert |
|---|---|
| Rang `r` | 8 |
| `lora_alpha` | 16 |
| effektive Standardskalierung `alpha/r` | 2,0 |
| Dropout | 0,05 |
| Bias | `none` |
| Task-Type | `CAUSAL_LM` |
| Initialisierung | PEFT-Standard, `init_lora_weights=true` |
| RS-LoRA | nein |
| DoRA | nein |
| QLoRA/Quantisierung | nein |
| angeforderte Zielmodule | `all-linear` |
| Ausgangsgewichte | eingefroren |

`all-linear` schließt die linearen Transformationsmodule des Modells ein; Embeddings und der Output-Head erscheinen nicht in den gespeicherten Adaptertensors. Die konkreten Modulnamen wurden direkt aus Config, Adapterconfig und Safetensors-Header rekonstruiert, ohne ein Modell oder einen Adapter zu laden.

### 7.2 Modellabhaengige Zielmodule

| Modelllinie | eindeutige Zielmodulnamen | adressierte Modulinstanzen | Adaptertensors |
|---|---|---:|---:|
| Qwen 3.5 2B | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj` | 186 | 372 |
| Llama 3.2 3B | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 196 | 392 |
| Qwen 3.5 9B | wie Qwen 2B | 248 | 496 |

Die zusaetzlichen Qwen-Projektionen folgen aus der Qwen-Architektur. Es waere falsch, fuer alle drei Modelllinien dieselbe aufgeloeste Zielmodulliste zu behaupten.

### 7.3 Trainierbare Parameter

| Modelllinie | trainierbar | Gesamtparameter mit Adapter | Anteil |
|---|---:|---:|---:|
| Qwen 3.5 2B | 8.409.600 | 1.890.234.688 | 0,444897 % |
| Llama 3.2 3B | 12.156.928 | 3.224.906.752 | 0,376970 % |
| Qwen 3.5 9B | 21.639.168 | 8.975.442.432 | 0,241093 % |

Die trainierbaren Parameter gehoeren ausschließlich zu den LoRA-Adaptern; die Ausgangsgewichte blieben eingefroren.

## 8. Optimierung und Batching

| Parameter | Wert fuer alle drei Linien |
|---|---|
| Optimizer | `adamw_torch_fused` |
| Lernrate | `1e-4` |
| Scheduler | `constant` |
| konfigurierte Warmup-Ratio | 0,03 |
| effektive Warmup-Rampe | keine unter dem konstanten Scheduler |
| Weight Decay | 0 |
| Gradient Clipping | maximale Norm 0,3 |
| Adam-Betas / Epsilon | 0,9 / 0,999 / `1e-8` |
| Train-Batch pro Device | 2 |
| Eval-Batch pro Device | 1 |
| Gradient Accumulation | 4 |
| World Size | 1 |
| effektive globale Train-Batchgroesse | `2 x 4 x 1 = 8` |
| Trainingsseed | 42 |
| separater Data-Seed | nicht persistiert; Trainer-Seed 42 steuert Sampling/Shuffle |
| Gradient Checkpointing | aktiviert |
| Praezision | FP16, nicht BF16 |
| TF32 | nicht explizit konfiguriert |
| Flash Attention | angefordert und effektiv `flash_attention_2` |
| Dataloader Worker | 0 |
| Logging | alle 10 Schritte |
| Evaluation / Speichern | je Epoche |
| Checkpointlimit | 5 |

Die Konfiguration enthaelt `warmup_ratio=0.03`; der effektive Scheduler ist jedoch `constant` und die Trainingshistorien protokollieren bereits beim ersten Logpunkt `1e-4`. Fuer die Methodik ist deshalb zwischen konfigurierter Warmup-Ratio und fehlender effektiver Warmup-Rampe zu unterscheiden.

## 9. Trainingsdauer

| Modelllinie | konfigurierte Maximalepochen | tatsaechlich abgeschlossen | Schritte/Epoche | letzter Schritt | Dauer |
|---|---:|---:|---:|---:|---:|
| Qwen 3.5 2B | 5 | 3 | 502 | 1.506 | 28.890,45 s (ca. 8:01:30) |
| Llama 3.2 3B | 5 | 3 | 509 | 1.527 | 6.374,44 s (ca. 1:46:14) |
| Qwen 3.5 9B | 5 | 3 | 502 | 1.506 | 41.273,62 s (ca. 11:27:54) |

Die Laeufe wurden nicht nur eine Epoche ausgefuehrt. Sie liefen drei Epochen und stoppten dann aufgrund der Early-Stopping-Regel. Autoritativ ist jeweils der zurueckgeladene Best-Checkpoint aus Epoche 1.

## 10. Validierung und Checkpointauswahl

| Modelllinie | Eval-Loss Epoche 1 | Epoche 2 | Epoche 3 | Best-Checkpoint |
|---|---:|---:|---:|---|
| Qwen 3.5 2B | 0,4983387 | 0,5043698 | 0,5256951 | `checkpoint-502` |
| Llama 3.2 3B | 0,4808335 | 0,5036232 | 0,5086325 | `checkpoint-509` |
| Qwen 3.5 9B | 0,4077516 | 0,4237793 | 0,4483621 | `checkpoint-502` |

Auswahlregel:

- `load_best_model_at_end=true`,
- Zielmetrik: `eval_loss`, kleiner ist besser,
- Evaluation und Speichern am Ende jeder Epoche,
- Early-Stopping-Geduld 2,
- Schwelle 0,001,
- Root-Adapter nach Trainingsende entspricht dem Best-Checkpoint.

Der Validierungsverlust ist der Loss ueber gepackte Full-Chat-Sequenzen von MixedVal2500-v2. Eine historische nachtraegliche SQL-Loss-Analyse war **nicht** die Auswahlmetrik. Zulaessig ist die Aussage, dass in diesem Versuch die Epoche-1-Zustaende den niedrigsten beobachteten Validierungsverlust hatten. Nicht zulaessig ist die Verallgemeinerung, eine Epoche sei fuer LoRA oder diese Modelle generell optimal.

Beim Qwen-9B-Lauf trat am Schritt 10 einmal `grad_norm=NaN` auf. Der Lauf setzte sich regulaer fort; Lossverlauf, drei Epochen, Checkpointauswahl, Root-/Checkpoint-Identitaet und Adapterhashes sind konsistent. Der Befund bleibt als technische Warnung dokumentiert, ist aber kein Nachweis eines beschaedigten Adapters.

## 11. Unterschiede zwischen den Modelllinien

### Ueber alle drei Linien identisch

- fachliche System-/User-/SQL-Aufgabe,
- 25.000 semantische Trainingsfaelle und 2.500 semantische Validierungsfaelle,
- Full-Chat-Loss,
- BFD-Packing und 2.048 Tokens,
- `r=8`, `alpha=16`, Dropout 0,05, Bias `none`, `all-linear`,
- Optimizer, Lernrate, Scheduler, Batchdesign, Seed und Praezision,
- fuenf konfigurierte und drei tatsaechlich ausgefuehrte Epochen,
- Checkpointwahl nach MixedVal2500-v2-`eval_loss`.

### Nur innerhalb der Qwen-Linien identisch

- byteidentische Trainings- und Validierungsdateien,
- manuelle Qwen-ChatML-Serialisierung,
- aufgeloeste zusaetzliche Qwen-Projektionsmodule,
- 4.011 Trainings- und 444 Validierungspacks,
- 502 Schritte je Epoche,
- Best-Checkpoint 502.

### Modellabhaengig

- Model-ID und Revision,
- Chattemplate und Special Tokens,
- aufgeloeste Zielmodulnamen und Instanzzahl,
- trainierbare Parameterzahl und relativer Anteil,
- Packingausbeute und Schritte je Epoche,
- Laufzeit und Best-Eval-Loss.

### Nicht direkt persistiert

- Qwen-Modellrevision in `training_metadata.json` und `adapter_config.json`,
- Qwen-`sys.executable` und Pythonversion als EENV-1-Beleg,
- historischer NVIDIA-Treiberversionsstring,
- konkrete Batch-Tensoren fuer einen unabhaengigen Replay der Packinggrenzen.

## 12. Konsistenz zwischen Training und Evaluation

| Aspekt | Befund |
|---|---|
| Systemprompt | fachlich und textuell konsistent |
| Schema-/Frage-Struktur | konsistent |
| Assistant-Beginn | konsistent, modellabhaengig serialisiert |
| SQL-only-/Semikolonregeln | konsistent |
| Chattemplate | Qwen ChatML bzw. Llama native chat jeweils konsistent |
| Gold-SQL | nur im supervised Assistant-Ziel des Trainings; nicht im Evaluationsprompt |
| Few Shot | Evaluation kann Demos hinzufuegen; Training besteht aus einzelnen Zielkonversationen |
| Sequenzlimit | Training und pre-k3-Hauptauswertung 2.048; additive k3-Auswertung 4.352 Inputtokens |

Die finalen Adapter wurden somit auf derselben fachlichen Frage-Schema-SQL-Aufgabe trainiert, die spaeter evaluiert wurde. Die historische Qwen-Fehlerquelle eines unerwuenschten Thinking-Templates ist nicht Teil der finalen Linie. Die k3-Erweiterung ist hinsichtlich ihres hoeheren Inputlimits separat zu begrenzen; sie aendert nicht die hier rekonstruierte Trainingsmethode.

## 13. Reproduzierbarkeit

### 13.1 Python- und Hardwareumgebung

Der bestehende Environment-Provenienzaudit klassifiziert alle drei Trainingslinien als `USED_VENV_FLASH` mit Evidenzstufe EENV-2 und hoher Konfidenz. Die relevante Umgebung ist:

```text
/home/ec2-user/nl2sql_testbench/.venv_flash/bin/python
Python 3.11.15
PyTorch 2.12.1+cu130
CUDA 13.0
Transformers 5.9.0
PEFT 0.18.1
TRL 1.4.0
Datasets 4.8.5
Flash Attention 2.8.3.post1
GPU NVIDIA L40S
```

Die Pythonversion und Modellrevision sind beim Llama-Lauf direkt persistiert. Fuer die Qwen-Laeufe fehlen direkte `sys.executable`-Felder; ihre Umgebungszuordnung stuetzt sich auf Paket-/Flash-Attention-/Artefaktevidenz. Daher darf fuer die Qwen-Linien nicht faelschlich EENV-1 behauptet werden.

### 13.2 Skriptprovenienz

- Tatsaechlicher Einstiegspunkt: `src/07_lora_finetune_sft_v1_clean.py`.
- Aktueller SHA256: `37d038ef7189d5b3868fa2abf5fa90868e1b07fd477aa4edd1c3d3684955eaf7`.
- Direkt fuer Llama persistierter Laufhash: derselbe Wert.
- Fuer die beiden Qwen-Laeufe dokumentierter historischer Laufhash: `f34e6fc2d8dca457925ebc51a3e7c2e412977cd83780fe0f9fd1a805cc0e3c64`.
- Importiertes Hilfsmodul: `src/07_lora_finetune_sft_v2.py`, aktueller SHA256 `13d42f2f767a0bca55d77b89d32b1d89ce2dcbe3000289a7ac68f4bda30f65d0`.

Der fruehere formale Entrypoint-Einwand im Erratum wurde durch das funktionale Addendum geklaert: Der full-chat Einstiegspunkt nutzte die erlaubten v2-Hilfsfunktionen, ohne den Completion-only-Hauptpfad auszufuehren. Die aktuelle Datei darf dennoch nicht rueckwirkend als byteidentischer Qwen-Laufcode ausgegeben werden.

### 13.3 Tragende Hashes

Config-, Dataset-, Adapter-, Metadata- und Trainer-State-Hashes sind vollstaendig im maschinenlesbaren JSON und im Artefaktinventar dokumentiert. Ein separater Konsolenlog wurde fuer keinen der drei Trainingslaeufe gefunden; `training_metadata.json`, `training_history.*` und `trainer_state.json` bilden deshalb gemeinsam den Laufzeitnachweis.

## 14. Methodische Grenzen

1. Es liegt je Modelllinie ein autoritativer Trainingslauf mit Seed 42 vor. Die Varianz ueber unabhaengige Trainingsseeds wurde nicht bestimmt.
2. Checkpointauswahl und Early Stopping beruhen auf einer projektspezifischen MixedVal2500-v2-Stichprobe und Full-Chat-Loss, nicht direkt auf Spider-Dev-EMA.
3. Der niedrigste Validierungsverlust nach Epoche 1 belegt keine allgemeine Optimalitaet einer Epoche.
4. Full-Chat-Loss optimiert auch Prompttokens. Er darf nicht als reines SQL-Completion-Training beschrieben werden.
5. Die Qwen-Snapshot- und Environmentprovenienz ist stark trianguliert, aber nicht vollstaendig direkt in den Trainingsmetadaten persistiert.
6. `all-linear` ergibt architekturabhaengige Zielmodule; LoRA ist daher nicht tensorweise identisch ueber Modellfamilien.
7. Direkte Overlapfreiheit in den kontrollierten Projektdateien schließt unbekannte Vortrainingskontamination nicht aus.
8. Die Qwen-9B-`grad_norm=NaN`-Warnung am fruehen Logpunkt ist zu dokumentieren, ohne daraus eine nicht belegte Wirkung auf die Ergebnisse abzuleiten.
9. Die k3-Evaluation nutzt ein hoeheres Inputlimit als Training und pre-k3-Hauptauswertung; sie ist eine additive Erweiterung, keine vollstaendig homogene Fortsetzung.

## 15. Thesisfertige Methodenzusammenfassung

Fuer jede der drei Modelllinien wurde ein separater LoRA-Adapter auf dem jeweiligen Ausgangsmodell trainiert. Der Trainingsbestand umfasste 25.000 Frage-Schema-SQL-Beispiele, davon 6.960 Faelle aus Spider Train und 18.040 Faelle aus SQL Create Context. Zur Validierung wurde MixedVal2500-v2 mit 2.500 vom Training getrennten Beispielen verwendet. Spider Dev floss weder in das Training noch in die Auswahl des finalen Checkpoints ein. Direkte projektseitige Ueberschneidungspruefungen fanden zwischen Training, Validierung und Spider Dev keine identischen IDs, Fragen, SQL-Abfragen oder Frage-SQL-Paare.

Die Beispiele wurden als vollstaendige System-User-Assistant-Konversationen serialisiert. Der Systemprompt legte eine einzelne, erklaerungsfreie SQLite-Abfrage fest; der Nutzerblock enthielt das vollstaendige Schema, die Frage und SQL-Ausgaberegeln, waehrend die Referenzabfrage den Assistant-Inhalt bildete. Qwen verwendete das projektspezifische ChatML-Format, Llama das native Chattemplate der festgelegten Modellrevision. Trainiert wurde mit Full-Chat-Loss: Neben der SQL-Antwort gingen auch System- und Nutzertokens in die Verlustberechnung ein. Lediglich Paddingpositionen und die Startpositionen gepackter Dokumente wurden maskiert.

Zur effizienten Sequenznutzung wurden die Beispiele mit der BFD-Strategie bis zu einer maximalen Sequenzlaenge von 2.048 Tokens gepackt. Kein einzelnes Trainings- oder Validierungsbeispiel ueberschritt dieses Limit. Dokumentgrenzen blieben im dokumentbewussten FlashAttention-Pfad ueber gespeicherte Sequenzlaengen und zurueckgesetzte Positions-IDs erhalten.

Alle Adapter nutzten LoRA mit Rang 8, `alpha=16`, Dropout 0,05, ohne Bias und mit eingefrorenen Ausgangsgewichten. Die effektive Standardskalierung betrug damit `alpha/r=2`. Als Ziel wurde `all-linear` gewaehlt, wodurch sich die konkreten Projektionsmodule architekturabhaengig unterschieden: Die Qwen-Modelle enthielten zusaetzliche lineare Projektionen, waehrend Llama die ueblichen Attention- und MLP-Projektionen adressierte. Eine Quantisierung beziehungsweise QLoRA, DoRA oder RS-LoRA wurde nicht verwendet.

Die Optimierung erfolgte mit fused AdamW, einer konstanten Lernrate von `1e-4`, ohne Weight Decay, mit maximaler Gradientennorm 0,3, FP16, Gradient Checkpointing und FlashAttention2. Bei einer Batchgroesse von zwei Beispielen pro Device, vier Akkumulationsschritten und einer GPU ergab sich eine effektive globale Batchgroesse von acht. Der Trainingsseed betrug 42.

Die Konfiguration sah maximal fuenf Epochen vor. Nach jeder Epoche wurden Validierungsverlust und Checkpoint gespeichert. Early Stopping mit einer Geduld von zwei Epochen beendete alle drei Laeufe nach Epoche 3. In allen Modelllinien hatte der Checkpoint nach Epoche 1 den niedrigsten gepackten Full-Chat-Validierungsverlust und wurde als finaler Adapterzustand geladen. Dieser Befund beschreibt die konkrete Konfiguration und ist nicht als allgemeine Aussage ueber eine optimale Epochenzahl zu verstehen.

Die fachliche Promptstruktur stimmt zwischen Training und spaeterer Evaluation ueberein. Modellabhaengige Unterschiede bestehen in Chattemplate, Special Tokens, aufgeloesten Zielmodulen, Parameterzahl und Packingausbeute. Die LoRA-Primarliteratur kann das allgemeine Prinzip eingefrorener Ausgangsgewichte und niedrig-rangiger Updates begruenden. Datensaetze, Hyperparameter, Zielmodule, Lossgestaltung, Packing und Checkpointauswahl sind dagegen projektspezifische Festlegungen und muessen durch die hier dokumentierten Projektartefakte belegt werden.

## 16. Evidenzmatrix

| Aussage | Primaere Evidenz | Status | Fuer Methodik geeignet |
|---|---|---|---|
| Genau drei finale Adapterroots | Adaptergewichte, Adapterconfigs, Evaluationsmanifeste | `PROJECT-VERIFIED` | ja |
| Qwen-Snapshots | Preflight, Cache, Evaluationsmanifeste | `AUDIT-VERIFIED` | ja, mit Provenienzhinweis |
| Llama-Revision | Trainingsmetadaten und Datasetmanifest | `PROJECT-VERIFIED` | ja |
| 25.000 = 6.960 + 18.040 | Datensaetze und Mainline-Manifeste | `AUDIT-VERIFIED` | ja |
| MixedVal2500-v2 = 700 + 1.800 | Validierungsdateien und Manifeste | `AUDIT-VERIFIED` | ja |
| Kein Spider Dev in Training/Checkpointwahl | Datasetpfade, IDs und Overlap-Audits | `AUDIT-VERIFIED` | ja, auf Projektdateien begrenzen |
| System-/Nutzer-/Assistant-Struktur | Datasettext und Builder | `PROJECT-VERIFIED` | ja |
| Full-Chat statt Completion-only | Config, Einstiegspunkt, Laufzeit-Labelpruefung | `PROJECT-VERIFIED` | ja |
| BFD-Packing, 2.048, keine Truncation | Config, Metadata, Tokenpreflight | `AUDIT-VERIFIED` | ja |
| Dokumentbewusste Attentiongrenzen | TRL-1.4.0-Code, `seq_lengths`, FlashAttention-Metadaten | `IMPLEMENTATION-INFERRED` | ja, vorsichtig formulieren |
| `r=8`, `alpha=16`, Dropout 0,05 | Config und Adapterconfig | `PROJECT-VERIFIED` | ja |
| konkrete Zielmodule | Adapterconfig und Safetensors-Header | `PROJECT-VERIFIED` | ja |
| Ausgangsgewichte eingefroren | PEFT-LoRA-Konfiguration und Trainable-Parameter | `IMPLEMENTATION-INFERRED` | ja |
| effektive Batchgroesse 8 | Laufzeitmetadaten und `2 x 4 x 1` | `PROJECT-VERIFIED` | ja |
| drei Epochen, Best-Checkpoint Epoche 1 | finaler Trainer-State und Metadata | `PROJECT-VERIFIED` | ja |
| Auswahl nach MixedVal-Full-Chat-`eval_loss` | Trainer-State und Config | `PROJECT-VERIFIED` | ja |
| historische `.venv_flash`-Nutzung | Environment-Provenienzaudit | `AUDIT-VERIFIED` (EENV-2) | ja, mit Evidenzstufe |
| Qwen-`sys.executable` | nicht direkt persistiert | `UNRESOLVED` | nicht als EENV-1 behaupten |
| Eine Epoche sei allgemein optimal | keine tragende Evidenz | `UNRESOLVED`/nicht gestuetzt | nein |

## 17. Erzeugte Artefakte und SHA256

| Artefakt | SHA256 |
|---|---|
| `audits/derived/method_lora_v2_finetuning_20260718.json` | `7435214f9e8bbf640c9d3322276289fbf45e8d2199dbb65f830461bdbdd6e0ef` |
| `audits/derived/lora_v2_authoritative_training_matrix_20260718.csv` | `59a8f154034d155d2936ce1e36c921d2b3c33190cf8e6340c7a871e8d5a13b4c` |
| `audits/derived/lora_v2_training_artifact_inventory_20260718.csv` | `0d4b6cc8d32538c3d176651f5c122ef4e54389b83cbaee0b173d1631325f4e52` |
| `audits/audit_method_lora_v2_finetuning_20260718.md` | Selbsthash wird nach dem Schreiben extern berichtet, um einen zirkulaeren In-Datei-Hash zu vermeiden. |

## 18. Offene beziehungsweise nicht eindeutig aufloesbare Punkte

1. Der absolute Qwen-Interpreterpfad und die Qwen-Pythonversion sind nicht direkt in den historischen Trainingsmetadaten gespeichert; `.venv_flash` ist EENV-2 mit hoher, nicht EENV-1 mit direkter Evidenz.
2. Die Qwen-Modellrevisionen sind nicht direkt in `training_metadata.json` oder `adapter_config.json` persistiert; die angegebenen Snapshots sind konsistent und auditgestuetzt rekonstruiert.
3. Fuer keinen der drei Laeufe wurde ein separater Konsolenlog gefunden. Trainer-State, Training-History und Metadata sind vorhanden und konsistent.
4. Die aktuelle Qwen-Trainerdatei ist nicht byteidentisch zum fuer die Qwen-Laeufe dokumentierten historischen Hash; der historische Hash bleibt deshalb separat ausgewiesen.
5. Der konkrete NVIDIA-Treiberversionsstring des historischen Trainingshosts ist nicht persistiert.
6. Die dokumentbewusste Attentiontrennung beim BFD-Packing ist aus Bibliothekscode und Laufzeitmetadaten rekonstruiert, wurde aber nicht durch persistierte Batch-/Attention-Tensoren numerisch replayt.

## Abschlussstatus

```text
Autoritativer Qwen-2B-LoRA-v2-Adapter:
adapters/qwen35_2b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5

Autoritativer Llama-3B-LoRA-v2-Adapter:
adapters/llama32_3b_instruct/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5

Autoritativer Qwen-9B-LoRA-v2-Adapter:
adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5

Ausgangsmodelle und Revisionen:
- Qwen/Qwen3.5-2B-Base @ b1485b2fa6dfa1287294f269f5fb618e03d52d7c
- meta-llama/Llama-3.2-3B-Instruct @ 0cb88a4f764b7a12671c53f0838cd831a0843b95
- Qwen/Qwen3.5-9B-Base @ 68c46c4b3498877f3ef123c856ecfde50c39f404

Training: 25.000 Faelle = 6.960 Spider Train + 18.040 SQL Create Context
Validation: MixedVal2500-v2, 2.500 Faelle = 700 train_others + 1.800 SQL Create Context
Spider Dev im Training verwendet: NEIN
Spider Dev fuer Checkpointauswahl verwendet: NEIN
Prompttemplate: Qwen qwen_sqlctx_chatml; Llama llama32_instruct_native_chat
Loss: Full-Chat; nicht Completion-only
Packing aktiviert: JA, BFD
Maximale Sequenzlaenge: 2.048
r: 8
alpha: 16
Dropout: 0,05
Bias: none
Zielmodule: all-linear; Qwen 12 und Llama 7 aufgeloeste Modulnamen
Trainierbare Parameter: 8.409.600 / 12.156.928 / 21.639.168
Lernrate: 1e-4
Optimizer: adamw_torch_fused
Scheduler: constant
Warmup: ratio 0,03 konfiguriert; keine effektive Rampe
Batchgroessen: Train 2, Eval 1
Gradient Accumulation: 4
Effektive Batchgroesse: 8
Epochen: maximal 5, tatsaechlich 3
Finaler Trainingsschritt: 1.506 / 1.527 / 1.506
Auswahlregel: kleinstes MixedVal2500-v2 Full-Chat-eval_loss
Best-Checkpoint: 502 / 509 / 502 (jeweils Epoche 1)
Seed: 42
Praezision: FP16
Gradient Checkpointing: JA
Training-Evaluation-Promptkonsistenz: PASS MIT DEKLARIERTEN MODELL- UND K3-UNTERSCHIEDEN
Geaenderte Bestandsdateien: keine
Training gestartet: nein
Evaluation gestartet: nein
Modellinferenz gestartet: nein
```

`LORA-V2-METHOD-AUDIT: PASS MIT WARNUNGEN`

`LORA-V2-AUTHORITATIVE-ADAPTERS: CONFIRMED_WITH_WARNINGS`

`LORA-V2-TRAINING-RECONSTRUCTION: PASS MIT WARNUNGEN`

`THESIS-SECTION-5.6.3: READY_WITH_LIMITATIONS`
