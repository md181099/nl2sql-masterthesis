# Methodenaudit: Dynamic Retrieval, Structure-Reranking und Similarity-Gates

Datum: 18.07.2026  
Projektwurzel: `/home/ec2-user/nl2sql_testbench`  
Gegenstand: fachliche Rekonstruktion für Abschnitt 5.6.2  
Arbeitsmodus: read-only hinsichtlich aller Bestandsartefakte; ausschließlich die in Abschnitt 15 aufgeführten Auditdateien wurden neu angelegt.

## 1. Executive Summary

Die autoritative Retrievalmethodik ist vollständig rekonstruierbar. Alle Dynamic-Bedingungen verwenden denselben Index mit 6.960 eindeutigen Spider-Train-Beispielen und `BAAI/bge-large-en-v1.5`. Sowohl die Zielanfrage als auch die Pooldokumente bestehen für das Embedding ausschließlich aus der natürlichsprachlichen Frage, jeweils mit dem Präfix `Represent this sentence for searching relevant passages: `. Frage- und Pool-Embeddings werden auf Einheitslänge normalisiert. Die Suche erfolgt mit `faiss.IndexFlatIP`; das innere Produkt entspricht deshalb der Kosinusähnlichkeit.

Die zentrale wissenschaftliche Klärung lautet:

> **Die Gold-SQL des Zielfalls wird für das Structure-Reranking nicht verwendet.**

Die erwartete Zielstruktur wird projektspezifisch und heuristisch aus zwei Inputs geschätzt:

1. lexikalische Muster in der natürlichen Zielfrage, etwa für `COUNT`, Mittelwert, Summe, Extremwert, Gruppierung, Distinktheit, Negation und Unterabfragen;
2. lexikalische Nennungen von Tabellen- und informativen Spaltennamen aus dem Zielschema in der Zielfrage, aus denen bei mindestens zwei betroffenen Tabellen ein Join-Bucket geschätzt wird.

Nur für die Spider-Train-Demonstrationskandidaten wird deren bekannte SQL gelesen, um strukturelle Kandidatenmerkmale zu bestimmen. Die zehn semantisch besten Kandidaten werden nach `originaler BGE-Score + signiertes Structure-Adjustment` neu sortiert. Das Adjustment liegt technisch zwischen `-0,04` und höchstens `+0,08`. Danach werden bei k=1 der erste beziehungsweise bei k=3 die ersten drei unterschiedlichen Kandidaten übernommen.

Die Similarity-Gates greifen **nach** Auswahl beziehungsweise Reranking und **vor** dem Promptbau. Bei k=1 wird der ursprüngliche BGE-Score der final ausgewählten Demo geprüft. Bei k=3 ist der Gatewert das Minimum der drei ursprünglichen BGE-Scores. Unterschreitet dieser Wert die Schwelle 0,70 oder 0,85, wird das gesamte Demonstrationsset verworfen und der reguläre Zero-Shot-Prompt erzeugt; eine Reduktion auf k=1 oder k=2 findet nicht statt.

Die fallweise Vollprüfung umfasst 72 autoritative Tracefiles und 74.304 gespeicherte Tracezeilen. Auswahlidentität über alle sechs Modellrollen, Gatearithmetik, Schwellenentscheidungen, Demo-Eindeutigkeit und Leakage-Status sind konsistent. Die Methodik ist für Abschnitt 5.6.2 freigegeben. Zwei Dokumentationsgrenzen bleiben: finale Runtraces enthalten nicht die komplette Top-10-Rerankliste mit Adjustments, und sie speichern die effektive Demoanzahl nach dem Gate nicht als eigenes `actual_k`-Feld.

## 2. Prüfgegenstand und Evidenzstatus

Geprüft wurden zwölf Bedingungen für sechs Modellrollen:

- Dynamic Top-1, Gate 0,70 und Gate 0,85;
- Structure Top-1, Gate 0,70 und Gate 0,85;
- Dynamic Top-3, Gate 0,70 und Gate 0,85;
- Structure Top-3, Gate 0,70 und Gate 0,85.

Die 36 k=1-/k=3-Paare in `audits/derived/k1_k3_authoritative_pair_mapping_20260718.csv` verweisen auf 36 autoritative k=1- und 36 autoritative k=3-Configs. Historische, ersetzte und partielle Runs sind nicht Bestandteil dieses Methodenaudits.

Die Statuslabels bedeuten:

| Status | Bedeutung |
|---|---|
| `PROJECT-VERIFIED` | direkt durch autoritativen Code und persistierte Projektartefakte belegt |
| `AUDIT-VERIFIED` | read-only aus eingefrorenen Traces, Manifesten oder bestehenden Audits reproduziert |
| `CONFIG-ONLY` | nur aus einem Configfeld ableitbar, ohne stärkere Laufzeitevidenz |
| `IMPLEMENTATION-INFERRED` | folgt aus der Codefolge, ist aber nicht als eigenes Laufzeitfeld persistiert |
| `UNRESOLVED` | aus den vorhandenen Artefakten nicht eindeutig auflösbar |

## 3. Autoritative Dateien

### 3.1 Implementierung

| Datei | Rolle | SHA256 | Status |
|---|---|---|---|
| `src/retrieval_utils.py` | k=1-FAISS-Retrieval, Filter, Structure-Reranking, Tracefelder | `b9ea76ae6c181988a66a886de72a96517a923df6cf748f2517cf1f1644c6e103` | `AUTHORITATIVE` |
| `src/retrieval_utils_dynamic_k3_v1.py` | k=3-Retrieval, eindeutige IDs, k=3-Tie-Breaking | `cb27aada2fbcca47e5c34c42fbe72083747a513d6c58e9867438a56441de5190` | `AUTHORITATIVE` |
| `src/structure_rerank_v2.py` | gemeinsame Merkmals- und Adjustmentlogik | `5cdbd48cd35cb252d778cef0ed2fba54ecb314a9988d6bdc1e8f1f098b6c603c` | `AUTHORITATIVE` |
| `src/06_batch_run.py` | k=1-Gate, Fallback und Promptdispatch | `a37286649920f4224999b5184e6117ea31f24968ad2c353ff338397c99a7a3c9` | `AUTHORITATIVE` |
| `src/06_batch_run_dynamic_k3_v1.py` | k=3-Set-Min-Gate, Fallback und Promptdispatch | `fca9840e3a736659c5bd1b6100d287c6aaca00193b2cef5f24c1050b5f7bffe5` | `AUTHORITATIVE` |
| `src/06_batch_run_dynamic_k3_sqltimeout_v3.py` | Qwen-9B-k=3-Variante mit gleicher Retrieval-/Gate-/Promptlogik | `330138f724cfb25ba9b77e8dabb48dc95731638357a51356bf578923e0fc9fd2` | `AUTHORITATIVE` |
| `src/08_build_spider_train_dynamic_fewshot_index.py` | Bau des 6.960er BGE-/FAISS-Index | `c5724b43c541788f7dcd06a172e7b3351aa2b1a0f343cd03194ac63dc1a61ca1` | `AUTHORITATIVE` |
| `src/prompt_presets.py` | Auflösung des Systemprompts | `7f285e6b93788bd573ef6cc6d10de71793bfcafe751e1a83df17f808fd5409b6` | `SUPPORTING` |

Die in den Runmetadaten persistierten Hashes für `retrieval_utils`, `structure_rerank_v2` und die jeweilige Runnergeneration stimmen mit diesen Dateien überein. Der frühere Runnerhash `a087...` in Vorbereitungsmanifeste ist historischer Preflightzustand; die autoritativen späteren Laufmetadaten weisen den aktuellen k=1-Runnerhash `a372...` aus.

### 3.2 Index und Register

| Datei | Rolle | SHA256 | Status |
|---|---|---|---|
| `data/retrieval_indexes/spider_train_no_dev_overlap_bge_large_en_v15/index.faiss` | autoritativer Vektorindex | `62a0a55a286934d334498ab01eee032407b9ec42c9915f587564a7cf89aa9cfc` | `AUTHORITATIVE` |
| `.../metadata.jsonl` | 6.960 Retrievalbeispiele | `05058698f782806dd706040da9a9197345246c20df4d93429d719b79565cda55` | `AUTHORITATIVE` |
| `.../manifest.json` | Index-, Embedding-, Pool- und Overlapprovenienz | `93eea8b31a9f98e5f175380118fb02007df2c0424aaad2810e90519ade07f86a` | `AUTHORITATIVE` |
| `audits/derived/k1_k3_authoritative_pair_mapping_20260718.csv` | exakte 72-Config-/Tracezuordnung | `aebcca28e4ed5e561f2c492cb97790de945e9fc71d41f3e93f2487d5497dc159` | `AUTHORITATIVE` |
| `audits/derived/dynamic_k3_retrieval_selection_validation_20260717.csv` | detaillierte ausgewählte k=3-Scores und Adjustments | `f54360619f750e44728ce1e69918f779d29894b528356e6da2f18ec8ea2aaf74` | `SUPPORTING` |
| `audits/derived/k3_gate_and_fallback_distribution_20260718.csv` | finale k=3-Gate- und Identitätsprüfung | `cbfe01b1648d1631d8ce6ac83f50f5b65a89a51376834e72f1458e754bbb2a33` | `AUTHORITATIVE` |

Die Pfade `spider_train6960_plus_trainothers103_mixedval_disjoint_official_bge_large_en_v15` und `spider_train6960_plus_trainothers935_no_dev_no_mixedval_overlap_bge_large_en_v15` sind historische beziehungsweise andere Versuchsvarianten. Sie werden ebenso ausgeschlossen wie alte 7.063er- oder 7.895er-Configs. Der im Modul noch vorhandene Modus `sqlaware_topk` ist nicht die autoritative Structure-Methode; die freigegebenen Configs verwenden ausschließlich `structure_topk_v2`.

## 4. Retrieval-Pool und Index

Der read-only geöffnete FAISS-Index hat folgende Eigenschaften:

| Merkmal | Befund | Status |
|---|---|---|
| Klasse | `IndexFlatIP` | `AUDIT-VERIFIED` |
| Vektoren | 6.960 | `AUDIT-VERIFIED` |
| Dimension | 1.024 | `AUDIT-VERIFIED` |
| trainiert | `true` | `AUDIT-VERIFIED` |
| Metadatenzeilen | 6.960 | `AUDIT-VERIFIED` |
| eindeutige IDs | 6.960 | `AUDIT-VERIFIED` |
| Quelle | 6.960-mal `spider_train` / `train_spider` | `AUDIT-VERIFIED` |
| vollständige Frage/SQL/Schemafelder | 6.960/6.960 | `AUDIT-VERIFIED` |

Der Indexbuilder startete laut Manifest mit 7.000 Spider-Train-Zeilen. Er entfernte 31 doppelte normalisierte Frage-SQL-Paare und neun Zeilen mit direkter Dev-Frage- oder Dev-SQL-Überschneidung; zwei dieser neun waren vollständige Frage-SQL-Paarüberlappungen. Das Resultat sind 6.960 Zeilen.

Das Manifest berichtet für Spider Dev:

| direkte Überlappung | Treffer |
|---|---:|
| ID | 0 |
| Frage | 0 |
| SQL | 0 |
| Frage-SQL-Paar | 0 |

Dies belegt die kontrollierten direkten Prüfungen, nicht die Abwesenheit paraphrastischer Überschneidungen oder unbekannter Vortrainingskontamination.

## 5. Query- und Dokumentrepräsentation

### 5.1 Zielanfrage

`FaissFewShotRetriever._search()` erhält die natürliche Frage. `_query_embedding_text()` setzt bei aktivem Manifestflag das Präfix

```text
Represent this sentence for searching relevant passages: 
```

vor die Frage. Weder Zielschema noch `db_id` noch Gold-SQL werden in den Retrievalquery aufgenommen. Befund: `PROJECT-VERIFIED`.

### 5.2 Pooldokumente

Der Indexbuilder setzt `embedding_text` auf die Spider-Train-Frage. Das Präfix wird laut gespeichertem Manifest sowohl auf Queries als auch auf Dokumente angewandt. `embedding_text == question` gilt in den gespeicherten Metadaten 6.960/6.960-mal; das Präfix wird erst beim Encoding ergänzt. SQL und Schema bleiben Metadaten für Prompt und Reranking, sind aber kein Bestandteil des BGE-Dokumentembeddings. Befund: `PROJECT-VERIFIED` und `AUDIT-VERIFIED`.

### 5.3 Normalisierung und Similarity

Pool- und Queryencoding verwenden `normalize_embeddings=True`. Dies erzeugt Einheitsvektoren. `IndexFlatIP` liefert innere Produkte; für Einheitsvektoren entsprechen diese der Kosinusähnlichkeit. Höhere Werte werden zuerst geliefert. Die Scores sind **keine kalibrierten Wahrscheinlichkeiten**.

## 6. Top-k-Auswahl

### 6.1 Semantisches k=1

Bei `same_db_only=false` beginnt die Suche nicht mit nur einem Treffer, sondern mit `max(k*20, k+50) = 51` Kandidaten. Die Kandidaten werden in der von FAISS gelieferten absteigenden Scorefolge geprüft. Verbotene Kandidaten werden übersprungen; der erste gültige Kandidat wird übernommen. Reicht das Fenster nicht, wird es vergrößert. Der k=1-Code besitzt keinen zusätzlichen ID-Duplikatfilter, aber der autoritative Pool enthält 6.960 eindeutige IDs.

### 6.2 Semantisches k=3

Bei k=3 beginnt das Fenster mit 60 Kandidaten. Zusätzlich führt der k=3-Retriever `selected_ids` und verwirft eine bereits ausgewählte Beispiel-ID. Genau drei unterschiedliche gültige Beispiele werden verlangt; der Runner bricht andernfalls vor der Generierung dieses Falls ab. In allen autoritativen k=3-Traces liegen drei verschiedene IDs vor.

### 6.3 Filter

Die Filterreihenfolge lautet:

1. gleiche ID wie Zielfall;
2. exakt gleiche normalisierte Frage;
3. optional andere Datenbank bei `same_db_only=true`;
4. direkte Leakage-Treffer gegen ID, Frage, SQL oder Frage-SQL-Paar, wenn `allow_overlap=false`.

Alle 72 autoritativen Configs setzen `same_db_only=false` und `allow_overlap=false`. Demos dürfen damit aus anderen Datenbanken stammen, direkte Dev-Überlappungen sind ausgeschlossen.

### 6.4 Reihenfolge im Prompt

Der Promptbuilder iteriert die ausgewählte Liste unverändert mit `enumerate(demos, start=1)`. Daher gilt:

- semantische Auswahl: absteigende ursprüngliche BGE-Reihenfolge;
- Structure-Auswahl: finale Rerankingreihenfolge;
- keine Umkehrung vor dem Prompt;
- bei k=3 erscheinen `Example 1`, `Example 2`, `Example 3` genau in dieser Reihenfolge.

Die Demonstrationen enthalten im autoritativen Modus jeweils Full Schema, Regeln, Frage und Gold-SQL (`fewshot_example_schema_mode=full`, `fewshot_example_mode=schema_with_rules`).

## 7. Strukturbasiertes Reranking

### 7.1 Pipeline und Gold-SQL-Abgrenzung

Die autoritative Methode ist `structure_topk_v2`. Sie ruft zunächst exakt die semantischen Top-10 ab und ordnet **nur diese zehn Kandidaten** neu. Es findet keine zweite Suche über den Gesamtpool statt.

Der Funktionsaufruf lautet semantisch:

```text
structure_rerank_adjustment(
    question=Zielfrage,
    target_schema=Zielschema,
    candidate_sql=Gold-SQL der Spider-Train-Demo,
    candidate_schema=Schema der Spider-Train-Demo
)
```

Eine Ziel-Gold-SQL wird nicht übergeben und an keiner Stelle innerhalb von `structure_rerank_v2.py` gelesen. Antwort auf die Kernfrage:

```text
ZIEL-GOLD-SQL FÜR RERANKING VERWENDET: NEIN
```

### 7.2 Heuristische Zielstruktur aus der Frage

Die Zielfrage wird Unicode-normalisiert, kleingeschrieben und auf Wortzeichen reduziert. Folgende Muster erzeugen Zielhints:

| Hint | tatsächlich erkannte Ausdrücke |
|---|---|
| `count` | `how many`, `number of`, `count`, `counts`, `total number` |
| `avg` | `average`, `avg`, `mean` |
| `sum` | `sum`, `summed`; außerdem `total`, sofern nicht `total number` |
| `max` + `order_extreme` | `maximum`, `max`, `highest`, `largest`, `greatest` |
| `min` + `order_extreme` | `minimum`, `min`, `lowest`, `smallest` |
| `order_extreme` | `most`, `least`, `fewest` |
| `group_by` | `for each`, `per`, `each`, `grouped by`, `by every` |
| `distinct` | `distinct`, `different`, `unique` |
| `negation` + `nested_select` | `without`, `never`, `not`, `no`, `except` |
| `nested_select` | `more/less than the average`, `above/below average` |

Das englische Wort `top` ist kein eigenes Hint. Generische Vergleichsoperatoren, `WHERE`, `HAVING`, `UNION` oder `INTERSECT` werden nicht als eigenständige Zielhints erkannt.

### 7.3 Heuristische Joinstruktur aus Frage und Zielschema

Das Zielschema wird anhand der Zeilen `Table:` und `Columns:` geparst. Für Tabellen werden einfache englische Pluralvarianten erzeugt. Ein Tabellenname zählt als erwähnt, wenn er als vollständige Wortfolge in der Zielfrage vorkommt. Vollständige Spaltennamen können ebenfalls eine Tabelle markieren, sofern sie mindestens fünf Zeichen lang, nicht trivial und als komplette Wortfolge in der Frage vorhanden sind.

Aus den markierten Tabellen entsteht:

- mindestens drei Tabellen: `target_join_bucket=2`;
- genau zwei Tabellen: `target_join_bucket=1`;
- null oder eine Tabelle: `target_join_bucket=None`.

Wichtig: Die Heuristik erzeugt **keinen expliziten Bucket 0** für sicher joinfreie Ziele. Bei weniger als zwei erkannten Tabellen wird die Joinkomponente schlicht nicht bewertet. Fremdschlüssel werden gezählt und in Diagnosedetails gespeichert, beeinflussen aber den Score nicht.

### 7.4 Kandidatenmerkmale aus Demonstrations-SQL

Vor der Merkmalsuche werden einfache ein- und doppelt zitierte SQL-Literale entfernt. Anschließend werden erkannt:

- `count`, `sum`, `avg`, `min`, `max`;
- `group_by`, `having`, `order_by`, `limit`, `distinct`;
- `exists`, `not_in`, `order_by_limit`;
- `join`, `multi_join` und Joinanzahl;
- `nested_select` bei mindestens zwei `SELECT`-Vorkommen;
- `union`, `intersect`, `except`.

Nicht alle erkannten Merkmale haben eine eigenständige Gewichtung. Insbesondere `having`, `union` und `intersect` werden gespeichert, aber nicht direkt bepunktet; `except` kann über die Negationskomponente wirken. `order_by` und `limit` wirken für Extremwertfragen nur gemeinsam als `order_by_limit`.

### 7.5 Adjustmentformel

| Komponente | Match | erwartetes Merkmal fehlt |
|---|---:|---:|
| je erforderlicher Aggregator `count/sum/avg/min/max` | +0,016 | -0,018 |
| `group_by` | +0,014 | -0,014 |
| `distinct` | +0,010 | -0,010 |
| Extremwert mit `ORDER BY` + `LIMIT` | +0,014 | -0,014 |
| Negation durch `NOT IN`, `EXISTS`, `EXCEPT` oder Unterabfrage | +0,012 | -0,012 |
| erwartete Unterabfrage vorhanden | +0,010 | kein separater Missing-Abzug |
| Join-Bucket identisch | +0,026 | n. a. |
| Join-Bucket Abstand 1 | n. a. | -0,014 |
| Join-Bucket Abstand 2 | n. a. | -0,026 |
| Demo-Schema länger als 6.000 Zeichen | n. a. | -0,008 |

Der Rohwert wird auf

```text
[-0,04; min(retrieval_structure_bonus_max, 0,08)]
```

begrenzt. In allen autoritativen Structure-Configs ist `retrieval_structure_bonus_max=0.08`. Trotz des Configbegriffs „bonus“ ist der Wert signiert und kann die semantische Similarity senken.

Der finale Score ist:

```text
final_score = original_bge_similarity + structure_adjustment
```

### 7.6 Rangfolge und Auswahl

Für k=1 wird sortiert nach:

1. höherem `final_score`;
2. höherem ursprünglichen BGE-Score;
3. besserem ursprünglichen BGE-Rang.

Für k=3 folgt als vierter Schlüssel die stabile Beispiel-ID. Danach werden die ersten drei unterschiedlichen IDs übernommen. Die final ausgewählten Kandidaten werden in genau dieser Reihenfolge in den Prompt geschrieben.

Der k=3-Preflight bestätigt für 1.032 Fälle:

| Prüfung | Ergebnis |
|---|---:|
| drei unterschiedliche semantische Demos | 1.032/1.032 |
| drei unterschiedliche Structure-Demos | 1.032/1.032 |
| Structure-Set unterscheidet sich von semantischem Top-3 | 480/1.032 |
| erste Structure-Demo unterscheidet sich von semantischem Top-1 | 210/1.032 |
| ausgewählte Slots | 3.096 |
| Slots mit nicht null Adjustment | 2.329 |
| beobachtetes ausgewähltes Adjustment | -0,040 bis +0,064 |

Das beobachtete Maximum +0,064 ändert die implementierte Obergrenze +0,08 nicht.

### 7.7 Fachlicher Pseudocode

```text
1. Präfixierte Zielfrage mit BGE einbetten.
2. Semantische Top-10 im normalisierten FAISS-IP-Index abrufen.
3. Verbotene direkte Overlaps filtern.
4. Zielhints aus Wörtern der Zielfrage bestimmen.
5. Mögliche Ziel-Joinkomplexität aus Frage-Zielschema-Namensübereinstimmungen schätzen.
6. Strukturmerkmale aus der bekannten SQL jedes Spider-Train-Kandidaten bestimmen.
7. Signiertes, begrenztes Structure-Adjustment berechnen.
8. Adjustment zum ursprünglichen BGE-Score addieren.
9. Top-10 nach finalem Score und den Tie-Breakern neu sortieren.
10. Erstes beziehungsweise erste drei unterschiedliche Beispiele auswählen.
```

## 8. Similarity-Gates

### 8.1 Pipelineposition

Die Codefolge lautet:

```text
semantisches Retrieval
-> optionales Structure-Reranking
-> finale Auswahl von k=1 oder k=3
-> Gateentscheidung
-> Few-Shot- oder Zero-Shot-Promptbau
```

Das Gate beeinflusst damit nicht die Kandidatenrangfolge. Es entscheidet nur, ob die bereits final ausgewählten Demos in den Prompt gelangen.

### 8.2 k=1

Die autoritativen k=1-Gates verwenden `similarity_only`. `FewShotSelection.scores[0]` ist bei semantischem Top-1 dessen BGE-Score. Beim Structure-Pfad wird nach dem Reranking ausdrücklich wieder der **ursprüngliche BGE-Score** der ausgewählten Demo in `FewShotSelection.scores` übernommen, nicht `final_score` und nicht das Adjustment.

Die Regel lautet:

```text
fewshot genau dann, wenn selected_original_bge_score >= threshold
```

### 8.3 k=3

Die autoritativen k=3-Gates verwenden `set_min_similarity`. Es muss zu jeder der drei Demos genau ein endlicher ursprünglicher BGE-Score vorliegen. Der Set-Score ist:

```text
gate_score = min(score_1, score_2, score_3)
```

Die Regel lautet:

```text
fewshot genau dann, wenn min(originale BGE-Scores der drei Demos) >= threshold
```

Sobald der niedrigste Score die Schwelle unterschreitet, wird das **gesamte Set** verworfen. Es gibt keine Einzelprüfung, keinen Austausch und keine Reduktion auf ein oder zwei Demos.

### 8.4 Schwellen und Verteilungen

Die Schwellen 0,70 und 0,85 sind projektspezifische Festlegungen. Pro Modellrolle gelten:

| Bedingung | Few Shot | Zero Shot |
|---|---:|---:|
| Top-1 Gate 0,70 | 634 | 398 |
| Top-1 Gate 0,85 | 57 | 975 |
| Structure Top-1 Gate 0,70 | 613 | 419 |
| Structure Top-1 Gate 0,85 | 57 | 975 |
| Top-3 Gate 0,70 | 480 | 552 |
| Top-3 Gate 0,85 | 7 | 1.025 |
| Structure Top-3 Gate 0,70 | 450 | 582 |
| Structure Top-3 Gate 0,85 | 6 | 1.026 |

Gate 0,85 bei k=3 ist daher fast vollständig eine Zero-Shot-Fallbackbedingung und darf nicht als durchgängige Three-Shot-Bedingung bezeichnet werden.

## 9. Zero-Shot-Fallback und Tracefelder

Bei `gate_decision == "zero_shot"` ruft der Runner dieselbe Funktion `build_prompt()` auf, die auch die reguläre Zero-Shot-Bedingung verwendet. Inputs sind Zielschema, Zielfrage, Modell-/Tokenizerbezug, Promptformat, Chattemplate und derselbe aufgelöste Systemprompt. Die ausgewählten Demos werden nicht übergeben. Bestehende Gate-Identitätsaudits bestätigen zusätzlich die Output- und SQL-Identität zu den jeweiligen Zero-Shot- beziehungsweise ungated Referenzen.

Die Retrievaltraces behalten trotz Fallback die zuvor ausgewählten IDs und Scores. Das ist für die Gateprüfung nützlich, bedeutet aber:

- `num_fewshot_examples` ist die **vor dem Gate ausgewählte** Anzahl und bleibt in Gate-Runs immer 1 beziehungsweise 3;
- ein separates `actual_k` existiert in den finalen Runtraces und Prediction-CSVs nicht;
- die effektive Demoanzahl ist deterministisch `0` bei `gate_decision=zero_shot`, sonst 1 beziehungsweise 3;
- der Prompt-Preflight materialisiert diese abgeleitete Größe ausdrücklich als `actual_k`.

Dies ist eine Dokumentationswarnung, kein Methodenfehler.

## 10. Unterschiede zwischen k=1 und k=3

| Aspekt | k=1 | k=3 |
|---|---|---|
| angeforderte Demos | 1 | 3 unterschiedliche IDs |
| semantisches Startfenster | 51 | 60 |
| Structure-Kandidaten | 10 | 10 |
| Structure-Auswahl | erster Rerankkandidat | erste drei unterschiedlichen Rerankkandidaten |
| Gate-Modus | `similarity_only` | `set_min_similarity` |
| Gate-Score | BGE-Score der einen Demo | Minimum der drei BGE-Scores |
| Fallback | k=0 | k=0; niemals k=1/2 |
| finaler Tie-Break | final, BGE, Ausgangsrang | final, BGE, Ausgangsrang, ID |
| Inputlimit | 2.048 | 4.352 |
| Promptreihenfolge | Auswahlreihenfolge | Auswahlreihenfolge |

Die erste ausgewählte Demo ist zwischen den methodisch korrespondierenden k=1- und k=3-Pfaden in 37.152/37.152 Paar-Fall-Kombinationen identisch. Das stützt die kontrollierte Erweiterung des bestehenden k=1-Auswahlpfads. Der k=1-/k=3-Wirkungsvergleich bleibt dennoch wegen Demonstrationszahl und Inputlimit eine Erweiterungsanalyse, kein isolierter Ein-Faktor-Kausaleffekt.

## 11. Tracevalidierung

Die Prüfung nutzte ausschließlich vorhandene Traces; kein Retriever wurde aufgerufen.

| Prüfung | Ergebnis |
|---|---:|
| autoritative Tracefiles | 72 |
| Tracezeilen | 74.304 |
| Files mit genau 1.032 Fällen | 72/72 |
| identische Fallreihenfolge je k=1-/k=3-Paar | 36/36 |
| Base-LoRA-ID-Identität je k=1-Gruppe | 18/18 |
| Base-LoRA-ID-Identität je k=3-Gruppe | 18/18 |
| Auswahl-ID-Identität über alle sechs Modellrollen | alle zwölf Methodenbedingungen |
| k=1-/k=3-Erstdemoidentität | 37.152/37.152 |
| Gate-Tracefiles | 48 |
| Gatezeilen | 49.536 |
| Gate-Score mathematisch korrekt | 49.536/49.536 |
| Schwellenentscheidung korrekt | 49.536/49.536 |
| Gateauswahl identisch zum jeweiligen ungated Retrieval | 1.032/1.032 pro Gate-Run |
| Tracezeilen mit Leakage-Status ungleich `pass` | 0 |
| Tracezeilen mit doppelter Demo-ID | 0 |

Die Prüfung deckt je Modelllinie Base und LoRA, k=1 und k=3, semantisch und Structure, ungated und gated sowie Few-Shot- und Fallbackfälle vollständig ab; sie ist stärker als eine Stichprobe.

### 11.1 In finalen Runtraces gespeichert

- Ziel-ID, `db_id`, Frage;
- Retrievalmethode, Index- und Poolpfad;
- ausgewählte Demo-IDs, ursprüngliche BGE-Scores und Demo-`db_id`s;
- vor dem Gate ausgewählte Anzahl;
- Filterzahlen und Filtergründe;
- Retrievalerfolg, Promptzeichenlänge und Leakage-Status;
- bei Gates: Modus, Score, Schwelle, Entscheidung, Grund sowie Score-Minimum, -Maximum und -Mittelwert bei k=3.

### 11.2 Nicht in finalen Runtraces gespeichert

- vollständige semantische Top-10-Kandidatenliste;
- per-Kandidat-Structure-Adjustment;
- per-Kandidat-`final_score`;
- explizites effektives `actual_k` nach dem Gate.

Die ausgewählten k=3-Adjustments, finalen Scores und ursprünglichen Ränge liegen im freigegebenen Preflightartefakt `dynamic_k3_retrieval_selection_validation_20260717.csv`. Die vollständige Rechenlogik ist durch den gehashten Code belegt. Für maximale Reproduzierbarkeit sollte die Thesis diese Evidenztrennung offen nennen, nicht so formulieren, als enthielten die finalen Runtraces sämtliche Top-10-Zwischenwerte.

## 12. Aussage-Evidenz-Matrix

| Aussage | Technischer Befund | Evidenzdatei | Status | Für Methodik geeignet |
|---|---|---|---|---|
| Query enthält nur die Frage | Präfix + Frage; kein Schema, keine DB-ID, keine SQL | `src/retrieval_utils.py:561` | `PROJECT-VERIFIED` | JA |
| Poolembedding enthält nur die Frage | 6.960-mal `embedding_text == question` | Indexbuilder + Metadaten | `AUDIT-VERIFIED` | JA |
| Similarity ist normalisiertes IP | beide Seiten normalisiert, `IndexFlatIP` | Indexbuilder + Manifest | `PROJECT-VERIFIED` | JA |
| Pool besteht nur aus Spider Train | 6.960/6.960 | Metadaten + Manifest | `AUDIT-VERIFIED` | JA |
| Top-3 verwendet drei eindeutige IDs | expliziter ID-Filter; 74.304 Traces ohne Duplikat | k=3-Retriever + Traces | `AUDIT-VERIFIED` | JA |
| Structure verwendet BGE Top-10 | `rerank_top_n=10` in 36 Structure-Configs | Configregister + Retriever | `PROJECT-VERIFIED` | JA |
| Ziel-Gold-SQL wird nicht verwendet | Funktionssignatur erhält nur Frage und Zielschema | `structure_rerank_v2.py` + Aufruf | `PROJECT-VERIFIED` | JA |
| Zielstruktur ist heuristisch | Fragehints + Schema-Namensmatches | `structure_rerank_v2.py:30`, `:117` | `PROJECT-VERIFIED` | JA, als projektspezifisch |
| Kandidatenstruktur stammt aus Demo-SQL | Featureparser liest bekannte Train-SQL | `structure_rerank_v2.py:58` | `PROJECT-VERIFIED` | JA |
| Adjustment ist signiert und begrenzt | Clamp -0,04 bis +0,08 | `structure_rerank_v2.py:214` | `PROJECT-VERIFIED` | JA |
| Gate prüft originalen BGE-Score | Selection übernimmt `bge_similarity` | Retriever + Runner | `PROJECT-VERIFIED` | JA |
| k=3-Gate prüft Set-Minimum | `min(raw_scores)` | k=3-Runner | `PROJECT-VERIFIED` | JA |
| Fallback verwirft das gesamte Set | regulärer Zero-Shot-Promptbuilder | Runner | `PROJECT-VERIFIED` | JA |
| Retrieval ist modellrollenunabhängig | IDs über sechs Rollen identisch | 72 Traces | `AUDIT-VERIFIED` | JA |
| effektives `actual_k` ist im Runtrace gespeichert | kein solches Rohfeld | 72 Traces | `AUDIT-VERIFIED` | NEIN; nur Ableitung beschreiben |
| komplette Top-10-Rerankwerte stehen im Runtrace | nicht persistiert | 72 Traces | `AUDIT-VERIFIED` | NEIN |

Die vollständige maschinenlesbare Matrix mit 31 Befunden steht in `audits/derived/method_dynamic_retrieval_structure_gates_evidence_20260718.csv`.

## 13. Methodische Grenzen

1. Die Zielstrukturheuristik ist lexikalisch. Sie ist weder vollständiger SQL-Parser noch gelerntes Intentmodell und kann implizite Strukturen übersehen.
2. Ein Join-Bucket wird nur bei mindestens zwei erkannten Schemaentitäten gebildet. Eine einzelne erkannte Tabelle bedeutet nicht automatisch „kein Join“.
3. Exakte vollständige Tabellen-/Spaltenphrasen sind sprachlich restriktiv; Synonyme und Paraphrasen werden nicht systematisch auf Schemaentitäten abgebildet.
4. Einige Kandidatenmerkmale werden zwar erkannt, aber nicht separat gewichtet. Eine Aufzählung als „bewertete Strukturmerkmale“ wäre daher zu weitgehend.
5. Das Adjustment ist projektspezifisch und signiert. Der Begriff „Bonus“ darf nicht den Eindruck erwecken, dass es ausschließlich positive Zuschläge gibt.
6. Similarity 0,70 oder 0,85 ist ein Schwellenwert auf unkalibrierten Embeddingscores, keine Erfolgswahrscheinlichkeit.
7. Gate 0,85 bei k=3 besteht zu über 99 Prozent aus Zero-Shot-Fallbacks.
8. Die finalen Traces speichern nicht sämtliche Rerankingzwischenwerte. Die Rekonstruktion kombiniert gehashten Code, Configs, ausgewählte Preflightwerte und finale Auswahltraces.
9. `num_fewshot_examples` bezeichnet in Gate-Traces die vor dem Gate ausgewählten Kandidaten. Die effektive Demoanzahl muss aus `gate_decision` abgeleitet werden.
10. k=3 verwendet ein höheres Inputlimit und einen zusätzlichen stabilen ID-Tie-Break. Dies ist bei k=1-/k=3-Vergleichen transparent zu berichten.

## 14. Literaturabgrenzung

In diesem Audit wurde keine externe Literatur gesucht.

Mit der verifizierten BGE-Quelle `baai2023bgelargeenv15` können später das verwendete Embeddingmodell sowie die allgemeine Idee dichter Repräsentationen und Ähnlichkeitssuche im Embeddingraum eingeordnet werden. Die konkrete lokale Präfixanwendung auf beide Seiten, Poolgröße, Normalisierung und FAISS-Ausführung sind durch Projektartefakte zu belegen.

Mit `gao2024texttosql` kann fachlich eingeordnet werden, dass Demonstrationsauswahl neben semantischer Frageähnlichkeit auch SQL-Strukturähnlichkeit berücksichtigen kann. Die Projektmethode ist jedoch keine ungeprüft aus dieser Quelle übernommene Implementierung.

Ausschließlich projektspezifisch und durch Projektartefakte zu belegen sind:

- Poolgröße 6.960;
- BGE Top-10 als Kandidatenmenge;
- konkrete Frage- und Schemaheuristiken;
- Gewichtung und Grenzen `-0,04` / `+0,08`;
- Gatewerte 0,70 und 0,85;
- Gate nach dem Reranking;
- Minimum-der-drei-Semantik;
- vollständiger Zero-Shot-Fallback;
- Verwendung der ursprünglichen BGE-Scores nach dem Reranking.

Ausdrückliche Abgrenzung:

> **Weder die BGE-Quelle noch DAIL-SQL dürfen als Beleg für die konkrete Gate-, Gewichtungs-, Bonus- oder Fallbacklogik dieses Projekts verwendet werden.**

## 15. Thesisfertige Methodenzusammenfassung

Die folgenden Aussagen sind als fachliche Grundlage für Abschnitt 5.6.2 freigegeben; sie sollten in der Thesis sinngemäß und mit Projektverweisen formuliert werden.

### Retrieval-Pool

`AUDIT-VERIFIED`: Als Demonstrationspool diente ein eingefrorener Index aus 6.960 eindeutigen Spider-Train-Beispielen. Direkte Übereinstimmungen mit Spider Dev hinsichtlich ID, normalisierter Frage, normalisierter SQL und Frage-SQL-Paar wurden beim Indexbau ausgeschlossen und im Manifest mit jeweils null Treffern bestätigt.

### Einbettung und semantische Suche

`PROJECT-VERIFIED`: Eingebettet wurde ausschließlich die natürlichsprachliche Frage. Für Ziel- und Poolfragen wurde dasselbe BGE-Präfix verwendet. Die normalisierten 1.024-dimensionalen Vektoren wurden mit einem `IndexFlatIP` durchsucht, sodass der gespeicherte innere Produktscore einer Kosinusähnlichkeit entspricht.

### Top-1 und Top-3

`PROJECT-VERIFIED`: Die semantische Auswahl übernahm die höchstbewerteten gültigen Beispiele in absteigender Similarityreihenfolge. k=1 verwendete ein Beispiel; k=3 verlangte drei unterschiedliche Beispiel-IDs. Die Reihenfolge wurde im Prompt beibehalten.

### Strukturbasiertes Reranking

`PROJECT-VERIFIED`: Für Structure-Bedingungen wurden zunächst die zehn semantisch ähnlichsten Kandidaten bestimmt. Die erwartete Zielstruktur wurde ohne Gold-SQL heuristisch aus Schlüsselwörtern der Zielfrage und aus in der Frage erwähnten Tabellen- beziehungsweise Spaltennamen des Zielschema geschätzt. Demgegenüber wurden Strukturmerkmale der Demonstrationen aus deren bekannten Spider-Train-SQLs extrahiert. Ein signiertes, auf `[-0,04; +0,08]` begrenztes Adjustment wurde zum ursprünglichen BGE-Score addiert; anschließend wurden die Kandidaten neu sortiert und die ersten ein beziehungsweise drei Beispiele gewählt.

### Similarity-Gates

`PROJECT-VERIFIED`: Die Gates wurden nach der finalen Auswahl angewandt. Bei k=1 wurde der ursprüngliche BGE-Score des ausgewählten Beispiels mit der Schwelle verglichen. Bei k=3 musste der niedrigste ursprüngliche BGE-Score des gesamten Dreiersets mindestens die Schwelle erreichen. Andernfalls wurde das vollständige Set verworfen.

### Zero-Shot-Fallback

`PROJECT-VERIFIED`: Bei abgelehntem Gate erzeugte der Runner den regulären Zero-Shot-Prompt aus Zielschema und Zielfrage mit demselben Systemprompt und Promptformat. Es wurden weder eine noch zwei Demonstrationen beibehalten.

### Methodische Grenzen

`AUDIT-VERIFIED`: Die Similarity ist nicht probabilistisch kalibriert. Die Structure-Methode ist eine projektspezifische Heuristik und keine vollständige Rekonstruktion der Ziel-SQL-Struktur. Besonders Gate 0,85 bei k=3 ist aufgrund von nur sieben beziehungsweise sechs akzeptierten Fällen überwiegend als Zero-Shot-Fallbackbedingung zu interpretieren.

## 16. Offene beziehungsweise nicht vollständig persistierte Punkte

Es verbleibt kein methodischer Punkt, der Abschnitt 5.6.2 blockiert. Folgende Evidenzgrenzen müssen jedoch genannt werden:

1. Die komplette Top-10-Liste mitsamt Adjustment und `final_score` ist nicht in jedem finalen Runtrace gespeichert. Sie ist über Code und den k=3-Preflight für die final ausgewählten Beispiele nachvollziehbar.
2. Das effektive `actual_k` ist in finalen Runtraces kein eigenes Feld. Es wird aus Gateentscheidung und konfiguriertem k bestimmt; im Prompt-Preflight liegt es explizit vor.
3. k=1 besitzt keinen vierten ID-Tie-Break nach dem ursprünglichen BGE-Rang. Da der Rang bereits eindeutig ist und alle Traces identisch sind, entsteht keine beobachtete Ambiguität.

## 17. Erzeugte Artefakte

| Datei | SHA256 |
|---|---|
| `audits/derived/method_dynamic_retrieval_structure_gates_evidence_20260718.csv` | `d0bb6e8d5faa3f87b1a5d24de767eb74ea46d22dcd955c57169a49efb8719c10` |
| `audits/derived/method_dynamic_retrieval_structure_gates_20260718.json` | `e964b61d7099fc02a17642bb2e1c0acb80d38adc39aff80e70268c67af65b4eb` |
| `audits/audit_method_dynamic_retrieval_structure_gates_20260718.md` | nicht rekursiv im Dokument einbettbar; finaler Hash wird extern ausgegeben |

## 18. Abschlussstatus

```text
DYNAMIC-RETRIEVAL-METHOD-AUDIT: PASS MIT WARNUNGEN
STRUCTURE-RERANKING-RECONSTRUCTION: PASS
SIMILARITY-GATE-RECONSTRUCTION: PASS
THESIS-SECTION-5.6.2: READY_WITH_LIMITATIONS
```

Die Warnungen betreffen ausschließlich die Granularität der persistierten Zwischenwerte und die Benennung des effektiven `actual_k`; sie stellen weder die Auswahl- noch die Gate- oder Fallbacksemantik infrage.
