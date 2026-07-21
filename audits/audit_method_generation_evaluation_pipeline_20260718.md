# Methodenaudit: Generations- und Evaluationsablauf

Datum: 18.07.2026  
Projektwurzel: `/home/ec2-user/nl2sql_testbench`  
Zweck: belastbare Rekonstruktion fuer Abschnitt 5.6.4 der Masterarbeit  
Arbeitsmodus: ausschliesslich lesend gegen Bestandsartefakte; nur dieser Audit und die drei zugehoerigen Derived-Dateien wurden additiv erzeugt.

## 1. Executive Summary

Der Generations- und Evaluationsablauf ist fuer die 48-Run-Hauptuntersuchung und die 36-Run-k3-Erweiterung hinreichend rekonstruierbar. Alle 84 autoritativen Runs enthalten 1.032 eindeutige Spider-Dev-Faelle in derselben Reihenfolge; insgesamt wurden 86.688 Fallzeilen, 84 Metadaten-Summaries und 80.496 Tracezeilen aus 78 Retrievalruns statisch geprueft. Sechs Zero-Shot-Runs benoetigen keinen Retrievaltrace. Es bestehen keine fehlenden oder doppelten Fall-IDs und keine Summaryabweichungen oberhalb der durch sechsstellige CSV-Rundung erklaerbaren Toleranz.

Der gemeinsame Kernablauf lautet:

1. Der Fall liefert `id`, `db_id`, `db_path`, Frage, Full-Schema-Text und Gold-SQL.
2. Der Runner baut einen modellabhaengig serialisierten Qwen- beziehungsweise Llama-Chatprompt; bei Few Shot stehen die Demonstrationen vor dem Zielfall.
3. Der Prompt wird mit dem konfigurierten Eingabelimit tokenisiert und mit `max_new_tokens=256`, `do_sample=False`, Batchgroesse 1 und Cache generiert.
4. Nur der neu generierte Tokensuffix wird dekodiert und als `raw_output` gespeichert.
5. `sql_first_statement_only` extrahiert den ersten akzeptierten, top-level beginnenden `SELECT`- oder CTE-`WITH`-Kandidaten und begrenzt ihn am ersten top-level Semikolon.
6. Gold- und Prediction-SQL werden in dieser Reihenfolge gegen dieselbe, per `mode=ro` geoeffnete SQLite-Verbindung ausgefuehrt.
7. EMA ist genau dann eins, wenn beide Ausfuehrungen erfolgreich sind und die nach Zeilenreihenfolge normalisierten Tupellisten mit Python-Gleichheit uebereinstimmen. ESR entspricht `pred_ok`.
8. Fehler, leere Extraktionen und in v3 abgefangene Timeouts verbleiben als Nullwerte im Nenner von 1.032 Faellen.

Vier Einschränkungen muessen im Methodentext sichtbar bleiben:

- Die Hauptuntersuchung nutzt in 46 Runs 2.048 Eingabetokens; zwei fruehe Base-Zero-Shot-Runs nutzen 1.536, ohne dass dieses Limit bei maximal 736 Prompttokens band. Die k3-Erweiterung nutzt 4.352.
- 48 Haupt- und 24 k3-Runs besitzen keinen expliziten SQL-Statement-Timeout. Zwoelf Qwen-9B-k3-v3-Runs verwenden 900 Sekunden je Gold- und Prediction-Statement.
- Bei 27 aelteren Hauptlaeufen wurde der Runtime-Runner-Hash nicht in den Runmetadaten persistiert. Configs, Fallartefakte und unabhaengige Rescoringaudits sind konsistent, die Provenienzluecke bleibt jedoch eine Dokumentationswarnung.
- Der spaetere Cross-Model-Syntheseskripttext nennt einmal `data/testcases.jsonl`, speichert dort aber den Hash und die 1.032er-Reihenfolge von `data/testcases_spider_dev_full.jsonl`. Alle 84 autoritativen Configs und Ergebnisse verwenden nachweislich den Full-Testbestand.

**Auditentscheidung:** Die Methodik ist fuer Abschnitt 5.6.4 freigegeben, sofern die gemischte Timeoutpolicy, die unterschiedlichen Eingabelimits und die genannten Provenienzgrenzen berichtet werden.

## 2. Autoritative Dateien und ausgeschlossene Varianten

| Datei/Artefakt | SHA256 | Rolle | Status |
|---|---|---|---|
| `src/06_batch_run.py` | `a37286649920f4224999b5184e6117ea31f24968ad2c353ff338397c99a7a3c9` | Haupt-Runner; Prompt, Generierung, Extraktion, untimed SQLite, Aggregation | `AUTHORITATIVE` fuer den aktuellen/21-fach persistierten Stand; fuer 27 aeltere Runs methodisch auditiert |
| `src/06_batch_run_dynamic_k3_v1.py` | `fca9840e3a736659c5bd1b6100d287c6aaca00193b2cef5f24c1050b5f7bffe5` | 24 k3-Runs Qwen 2B/Llama 3B ohne Statement-Timeout | `AUTHORITATIVE` |
| `src/06_batch_run_dynamic_k3_sqltimeout_v3.py` | `330138f724cfb25ba9b77e8dabb48dc95731638357a51356bf578923e0fc9fd2` | 12 Qwen-9B-k3-Runs mit 900-s-Guard | `AUTHORITATIVE` |
| `src/llm_client.py` | `e9cc2aea6952e164391f7ff67cdb7af5d41ad3864d3da8c44d4045a9adbd304b` | Modell-/Tokenizerregistry, Precision und Adapterladung | `SUPPORTING` |
| `src/prompt_presets.py` | `7f285e6b93788bd573ef6cc6d10de71793bfcafe751e1a83df17f808fd5409b6` | Systemprompt | `AUTHORITATIVE` |
| `src/llama32_native_chat.py` | `e784497c10728b13d195dbeb987340b84b3c9be535c6d03f8f0e8271d6653ef4` | natives Llama-Template, Stop-/PAD-Tokens | `AUTHORITATIVE` |
| `src/chat_formatting.py` | `83175264867a7e5f4331aff1def756d941577d3db603b520937a6f8a11358cbb` | Nachrichtenstruktur | `SUPPORTING` |
| `audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json` | `24b4dec07d2d4981b42ce22e1295d27b0ccd9cbcc10666a422118b267fd14e37` | Register der 48 Hauptlaeufe | `AUTHORITATIVE_WITH_PATH_WARNING` |
| `audits/derived/k3_all_runs_completion_inventory_after_repair_20260718.csv` | `adf17bd367049ba86a66b649f474e8d6744af9279e0961605bf668643ebe87e3` | Register der 36 k3-Laeufe | `AUTHORITATIVE` |
| `audits/audit_k3_matrix_completion_after_qwen9b_base_top3_v3_20260718.md` | `748965e7ced25ddced2cf49dc60765c2536d7f2d584737ed823cc872e02a1a35` | k3-Abschluss- und Timeoutaudit | `AUTHORITATIVE` |
| `scripts/analyze_cross_model_complete_8x8_synthesis.py` | `95dbda1932ec957bba9fa54c3ddbb8963ded1d7d974ced492e5199d3fd6b6475` | statistische Weiterverarbeitung Hauptmatrix | `SUPPORTING_WITH_PATH_DRIFT` |
| `scripts/analyze_k3_final_results_and_k1_vs_k3_statistics_20260718.py` | `ad2e0fa50ed4efd56de2ac982ceb7ab18efa69559d5f704d4d9b6355fae7fa5c` | k1-k3-Paarung und separate Statistikfamilie | `AUTHORITATIVE` |

Die 84 einzelnen Config-, CSV-, Metadata- und Tracepfade samt Hashes stehen in `audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv`.

Ausgeschlossen sind:

- `data/testcases.jsonl` mit 200 Zeilen und SHA256 `f7372af363054e95851e969159b98636817c77362003247d7c4a4876047dfed4`;
- der unbounded Qwen-9B-Base-Top-3-Teilrun `..._20260718_011245` mit 479 CSV-Zeilen;
- der v2-Timeout-Teilrun `...sqltimeout900_20260718_085006` mit 482 CSV-Zeilen;
- Smoke-, historische Adapter- und sonstige nicht in den beiden autoritativen Registern enthaltene Runs.

Der v2-Teilrun belegt eine wichtige Ablaufeigenschaft: Die Fallzeile wird erst **nach** SQL-Ausfuehrung und Metrikbildung geschrieben. Der dort korrekt ausgeloeste Timeout wurde wegen eines anschliessenden `nonlocal`-Fehlers nicht als Fallzeile persistiert. Dieser Runner und seine Teilresultate sind nicht Teil der 84 Runs.

## 3. Testset und Fallreihenfolge

| Merkmal | Befund |
|---|---|
| Pfad | `data/testcases_spider_dev_full.jsonl` |
| SHA256 | `6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce` |
| Faelle | 1.032 |
| eindeutige IDs | 1.032 |
| erste/letzte ID | `SPIDER_DEV_000000` / `SPIDER_DEV_001033` |
| Datenbanken | 20 `db_id`-Werte und 20 vorhandene SQLite-Dateien |
| Pflichtfelder | ID, `db_id`, `db_path`, Frage, Schema und Gold-SQL in 1.032/1.032 Faellen nicht leer |
| Reihenfolge | 84/84 Ergebnis-CSVs identisch zum Testset |
| fehlende/doppelte IDs | 0 / 0 |

Der Runner liest die JSONL-Reihenfolge unveraendert ein. `max_test_samples=None` bedeutet, dass keine Teilmenge gebildet wird. Die 84 autoritativen Runs enthalten daher jeden Fall genau einmal.

**Nennerregel:** `total` wird am Beginn von `write_case_result()` erhoeht, also bevor Extraktion und SQLite-Ausfuehrung bewertet werden. Leere SQL-Extraktionen, SQLite-Ausnahmen und gespeicherte Timeouts bleiben im Nenner. Ein ungefangener Fehler innerhalb von `model.generate()` wuerde hingegen vor `write_case_result()` den Lauf abbrechen und keine Fallzeile erzeugen. In den 84 vollstaendigen Runs trat kein solcher unpersistierter Generierungsfehler auf. Das Erreichen von 256 Completiontokens ist eine regulaer gespeicherte Generation und bleibt im Nenner.

## 4. Promptbau und Tokenisierung

### 4.1 Systemprompt

Alle 84 Runs speichern Variante `sqlctx_anti_overjoin` und SHA256 `d6dd5afc21336e4b44114511a1084e88758692cfb9ad1c24b9ca14e2f30a147e`:

> You are an SQLite SQL generator. Return exactly one valid SQLite query and nothing else. Output SQL only. Do not explain. Do not reason. Do not use markdown. Do not use comments. Use only tables and columns from the provided schema. Use only tables required by the question. Do not join tables unless their columns are required. If one table contains all required columns and filters, use only that table. The query must start with SELECT or WITH and end with a semicolon.

Der Prompt ist eine Ausgaberegel, kein technischer SQL-Sicherheitsmechanismus.

### 4.2 Zielfall

Der Zero-Shot-Nutzerinhalt ordnet die Bestandteile wie folgt:

1. `Database schema:` mit dem fallspezifischen `schema_prompt`,
2. Regeln zur Schemaausschliesslichkeit und SQL-only-Ausgabe,
3. `Question:` mit der natuerlichsprachlichen Frage,
4. abschliessendes `SQL:`.

Fehlt ausnahmsweise das fallspezifische Schema, sieht der Runner einen globalen Schemafallback vor. Im autoritativen Testset ist `schema_prompt` jedoch in 1.032/1.032 Faellen vorhanden.

### 4.3 Demonstrationen und Gate-Fallback

`build_prompt_schema_fewshot()` setzt jede Demonstration in persistierter Retrievalreihenfolge vor den Zielfall. Die autoritative Full-Schema-Darstellung umfasst je Beispiel Schema, dieselben SQL-Regeln, Frage und Gold-SQL. Danach folgen Zielfallschema, Regeln, Frage und `SQL:`. Bei einer Gateentscheidung `zero_shot` ruft der Runner den regulaeren `build_prompt()`-Pfad auf; es werden nicht ein oder zwei Beispiele behalten.

### 4.4 Modellabhaengige Serialisierung

| Modelllinie | Format | Assistant-Beginn | Special-Token-Behandlung |
|---|---|---|---|
| Qwen 3.5 2B/9B | `qwen_sqlctx_chatml` | `<|im_start|>assistant\n` | manuelles ChatML; Tokenizer danach mit `add_special_tokens=True`; kein `<think>`-Prefix |
| Llama 3.2 3B Instruct | `llama32_instruct_native_chat` | vom nativen Template erzeugter Assistant-Header | `apply_chat_template` mit festem Datum `26 Jul 2024`; anschliessend `add_special_tokens=False`, um ein doppeltes BOS zu vermeiden |

Llama verwendet PAD-ID `128009` und die Stop-IDs `[128001, 128008, 128009]`. Qwen uebergibt keine laufseitige EOS-/PAD-Ueberschreibung an `generate()`; Modell- und Tokenizerkonfiguration gelten. Bei 27 aelteren Qwen-Hauptlaeufen wurden diese Runtimefelder noch nicht einzeln persistiert, die Promptformate und Systemprompthashes jedoch schon.

## 5. Eingabelimits und Truncation

| Runfamilie | Konfiguriertes Limit | Runs | beobachtetes Maximum | aktive Truncationen |
|---|---:|---:|---:|---:|
| Hauptanalyse, fruehe Base Zero Shot | 1.536 | 2 | 736 | 0 |
| Hauptanalyse, uebrige Runs | 2.048 | 46 | 2.045 | 0 |
| k3-Erweiterung | 4.352 | 36 | 4.269 | 0 |

Der produktive Decoder ruft den Tokenizer mit `truncation=True` und `max_length=max_input_tokens` auf. Die Fall-CSV speichert die **nach** Tokenisierung erhaltene Promptlaenge, nicht die ungekappte Vorlaenge oder einen eigenen Truncationstatus. Die Nullbefunde stammen deshalb zusaetzlich aus den autoritativen Prompt-Preflights:

- k1-2048-vs.-4352-Aequivalenzmanifest, SHA256 `8db4114eeb52eba506a28d367933576ef3baebfd6b28772c4dbf6778d490f35c`: 37.152 Promptmaterialisierungen, Maximum 2.045, Truncationen bei 2.048 = 0, Token-ID- und Prompthashdifferenzen = 0;
- k3-Preflight, SHA256 `106378e71855b23f50cd2332d6fc9a9482bc1b019699f078387807b7a73bc89c`: 37.152 Promptzeilen, Maximum 4.269, Truncationen = 0.

Die Tokenizerconfigs setzen `truncation_side` nicht explizit. Die rechtsseitige Bibliotheksvorgabe ist daher nur implementation-inferiert; da kein autoritativer Prompt das Limit erreichte, hatte diese Vorgabe keine praktische Wirkung. `max_length` wird nicht als Generationslimit uebergeben; die Ausgabe wird allein durch `max_new_tokens` begrenzt.

## 6. Generationsparameter

| Parameter | Implementierter Wert/Status |
|---|---|
| Batchgroesse | 1 in 84/84 Runs |
| `max_new_tokens` | 256 in 84/84 Runs |
| `do_sample` | `False`, im Runner fest gesetzt |
| `num_beams` | nicht uebergeben; effektiver Standard 1, daher kein Beam Search |
| `temperature`, `top_p`, `top_k`, `typical_p` | nicht uebergeben; Samplingwerte sind bei `do_sample=False` wirkungslos |
| `repetition_penalty` | nicht uebergeben |
| `min_new_tokens` | nicht uebergeben |
| `use_cache` | `True` |
| eigene `StoppingCriteria` | keine |
| Generationseed | keiner im Evaluationsrunner gesetzt |
| deterministische Backendflags | nicht explizit gesetzt |
| Precision | Loaderpolicy: CUDA `float16`, sonst `float32` |
| Device | `device_map="auto"`; Eingabetensoren nach `model.device` |

Der lokale Llama-`generation_config` enthaelt Samplingwerte, aber der Runner ueberschreibt `do_sample` mit `False`. Die Generierung ist damit greedy und ohne stochastische Samplingvarianz. Dies ist keine Zusicherung plattformuebergreifender Bitidentitaet, weil kein gesonderter Generationseed und keine strikten deterministischen CUDA-Flags gesetzt werden.

`time.perf_counter()` misst nur den Zeitraum um `model.generate()`. `generation_time_seconds` ist daher **Generationszeit**, nicht End-to-End-Fallzeit: Promptbau, Retrieval, Extraktion und SQLite-Ausfuehrung liegen ausserhalb dieses Timers. Die Metadata-`duration_seconds` beschreibt dagegen die gesamte Runlaufzeit.

Nach `generate()` wird der Promptprefix anhand seiner Tokenbreite abgeschnitten. Nur der neue Suffix wird mit `skip_special_tokens=True` dekodiert. `raw_output` enthaelt somit die Modellausgabe, nicht den Eingabeprompt.

## 7. Rohoutput und SQL-Extraktion

Alle 84 Configs verwenden `sql_first_statement_only`. Die Extraktion ist textbasiert und verwendet weder Gold-SQL noch Datenbankresultate.

### 7.1 Fachlicher Pseudocode

```text
input = dekodierter generierter Tokensuffix
normalisiere Zeilenumbrueche
entferne bekannte Think-/ChatML-/Chat-Tags, Markdown-Fences und fuehrende Labels
finde top-level SELECT- und CTE-artige WITH-Starts ausserhalb von Quotes/Klammern
fuer jeden Start:
    schneide am ersten top-level Semikolon ausserhalb von Quotes/Klammern
    pruefe strukturelle Balance
    akzeptiere nur SELECT oder CTE-WITH mit enthaltenem SELECT
    verwerfe harte Fragmentmuster; markiere weiche Textkontamination
waehle den fruehesten vollstaendigen, balancierten, sauberen Kandidaten
sonst waehle den fruehesten vollstaendigen, balancierten Kandidaten
sonst waehle einen fruehesten balancierten, unkontaminierten Kandidaten
und ergaenze dessen fehlendes Semikolon
wenn kein Kandidat existiert: pred_sql = "" und pred_error = "No SQL extracted"
```

### 7.2 Exakte Auswirkungen

- Single- und Double-Quotes sowie verdoppelte Quote-Escapes werden beim Finden des Statementendes beruecksichtigt.
- Semikolons innerhalb von Strings oder Klammern beenden das Statement nicht.
- Prose vor dem ersten akzeptierten Top-level-Kandidaten wird ignoriert.
- Inhalt nach dem ersten akzeptierten Statement wird ignoriert. Mehrere Statements werden daher auf das erste reduziert, nicht gemeinsam ausgefuehrt.
- Fehlende Semikola werden nur bei einem balancierten akzeptierten Kandidaten ergaenzt.
- Markdown-Fences und bekannte Modell-/Rollenmarker werden entfernt. Es existiert jedoch kein vollstaendiger SQL-Kommentarparser; Kommentare innerhalb eines bereits begonnenen Kandidaten koennen erhalten bleiben.
- `SELECT` und CTE-artiges `WITH ... SELECT` sind die einzigen primaeren Starttypen. `EXPLAIN`, `PRAGMA`, `ATTACH`, `DETACH`, DDL, DML und Transaktionsbefehle werden nicht als primaerer Kandidat ausgegeben. Falls vor einem spaeteren `SELECT` solcher Text steht, startet der extrahierte Kandidat erst beim `SELECT`.
- Ein separates Feld `extraction_status` wird nicht geschrieben. Der Status ist ueber leeres `pred_sql` und `pred_error` rekonstruierbar.

Die drei Ebenen sind daher strikt zu unterscheiden:

1. `raw_output`: dekodierter generierter Text;
2. `pred_sql`: extrahierter und gegebenenfalls um ein Semikolon ergaenzter erster Kandidat;
3. ausgefuehrte Prediction-SQL: `pred_sql`, sofern nicht leer; andernfalls keine SQLite-Ausfuehrung.

## 8. Read-only-Pruefung

Die Read-only-Absicherung ist eine Kombination aus zwei Mechanismen:

1. **heuristische syntaktische Begrenzung:** Der Extraktor liefert nur einen `SELECT`- oder CTE-artigen `WITH`-Kandidaten mit enthaltenem `SELECT`;
2. **technische SQLite-Begrenzung:** Die Datenbank wird als URI `file:<aufgeloester Pfad>?mode=ro` mit `uri=True` geoeffnet.

Der zweite Mechanismus ist der eigentliche Schreibschutz-Backstop. Nicht verwendet werden:

- `PRAGMA query_only=ON`,
- `sqlite3.Connection.set_authorizer`,
- Datenbankkopien,
- Prozessisolation fuer jede Abfrage,
- ein AST-basierter Statementtyp-Validator.

`PRAGMA foreign_keys=ON` wird gesetzt, ist aber kein Schreibschutz. Benutzerdefinierte SQLite-Funktionen werden nicht registriert. Die Sicherheit darf deshalb nicht aus dem Systemprompt abgeleitet und nicht als mehrschichtige AST-Sandbox beschrieben werden. Korrekt ist: *Ausgaben wurden auf SELECT/CTE-WITH gefiltert und anschliessend gegen eine im SQLite-URI-Modus read-only geoeffnete Datenbank ausgefuehrt.*

## 9. SQLite-Ausfuehrung

Pro Fall wird `db_path` aus dem Testdatensatz aufgeloest. Der Runner cached je Datenbankpfad eine SQLite-Verbindung und verwendet sie fuer alle Faelle derselben Datenbank innerhalb eines Runs. Gold- und Prediction-SQL laufen nacheinander in derselben Verbindung:

1. Cursor erzeugen,
2. `execute(sql)`,
3. `fetchall()`,
4. bei Erfolg `ExecResult(ok=True, rows=..., error=None)`,
5. bei jeder Python-/SQLite-Ausnahme `ExecResult(ok=False, rows=None, error=repr(exception))`.

Die Gold-SQL wird somit in jedem Run fuer jeden Fall erneut ausgefuehrt. Es gibt keine Retrylogik, keinen expliziten Rollback und keinen gesonderten Cursor-Close; die read-only Verbindungen werden nach dem Run geschlossen. Resultatzeilen und Spaltenbeschreibungen werden nicht in den Ergebnis-CSVs persistiert.

Referenz- und Predictionfehler werden als getrennte `gold_ok`-/`pred_ok`-Flags und Errorstrings gespeichert. Eine leere Extraktion erzeugt ohne SQLite-Aufruf `ExecResult(False, None, "No SQL extracted")`.

## 10. Fehler- und Timeoutbehandlung

### 10.1 Haupt-Runner und 24 fruehe k3-Runs

`run_sql()` faengt SQLite-Ausnahmen pro Statement ab, besitzt aber keinen Deadline- oder Progress-Guard. Ein regulaerer Syntax-, Schema- oder Ausfuehrungsfehler wird gespeichert und der naechste Fall bearbeitet. Eine pathologisch lange SQL kann den Prozess dagegen unbegrenzt blockieren. Der historische unbounded Qwen-9B-Top-3-Teilrun demonstriert genau dieses Risiko und ist ausgeschlossen.

### 10.2 Zwoelf Qwen-9B-k3-v3-Runs

V3 setzt vor **jedem** Gold- und Prediction-Statement eine eigene Deadline von 900 Sekunden. `sqlite3.set_progress_handler()` ruft nach jeweils 10.000 SQLite-VM-Schritten einen Callback auf; nach Ablauf liefert dieser 1, SQLite unterbricht mit `OperationalError('interrupted')`, und der Runner speichert `SQLExecutionTimeout(...)`. Im `finally`-Block wird der Progresshandler geloescht, sodass die Verbindung weiterverwendet und der Run fortgesetzt werden kann.

| Run | Case | DB | Gold-Timeout | Prediction-Timeout | gespeicherter Status |
|---|---|---|---:|---:|---|
| Qwen-9B Base Top-3 v3 `..._180657` | `SPIDER_DEV_000484` | `wta_1` | 0 | 1 | `pred_ok=0`, `exec_match=0` |
| Qwen-9B Base Structure-Top-3 v3 `..._102308` | `SPIDER_DEV_000484` | `wta_1` | 0 | 1 | `pred_ok=0`, `exec_match=0` |

Beide Timeouts verbleiben im 1.032er-Nenner. Gold-Timeouts traten in den zwoelf v3-Runs nicht auf. Fuer die anderen 72 Runs kann nicht behauptet werden, dass dieselbe 900-s-Regel gegolten habe.

## 11. Ergebnisvergleich

### 11.1 EMA-Fallregel

`execution_match(pred, gold)` liefert nur dann `True`, wenn beide `ExecResult.ok=True` sind. Danach werden die Zeilenlisten sortiert. Der Sortierschluessel je Zellenwert ist `(Typname, repr(Wert))`; die Werte selbst werden nicht veraendert. Anschliessend werden die Listen von Tupeln mit Python-Gleichheit verglichen.

Folgen:

- Zeilenreihenfolge ist irrelevant.
- Duplikate bleiben erhalten und sind zaehlerrelevant.
- Spaltenreihenfolge bleibt relevant.
- Spaltennamen werden nicht verglichen.
- Es gibt keine explizite Typnormalisierung und keine Floattoleranz. Python-Gleichheit kann jedoch numerische Cross-Type-Gleichheit wie `1 == 1.0` zulassen.
- `NULL` bleibt `None`, BLOB bleibt `bytes`, Text bleibt Text.
- Zwei leere Ergebnislisten vergleichen gleich, auch wenn die nicht persistierte Spaltenstruktur unterschiedlich waere.
- Unterschiedliche nichtleere Zeilen- oder Spaltenanzahlen fuehren zur Ungleichheit.
- Alternative SQL kann EMA=1 erhalten, wenn sie auf der konkreten SQLite-Instanz dieselben Ergebnistupel produziert.
- Fehlerhafte Gold- oder Predictionausfuehrung fuehrt zu EMA=0.

### 11.2 ESR und weitere Fallmetriken

ESR entspricht `pred_ok`: Auch eine erfolgreich ausgefuehrte Abfrage mit leerer Ergebnismenge gilt als ausfuehrbar. Leere Extraktionen, SQLite-Fehler und Timeouts sind nicht ausfuehrbar.

String EM vergleicht `pred_sql == gold_sql`. Normalized EM tokenisiert beide SQLs per Regex, schreibt Tokens klein und vergleicht die Tokenfolgen. Char Accuracy und Token Accuracy zaehlen positionsgleiche Zeichen beziehungsweise SQL-Tokens und teilen durch die groessere der beiden Laengen. Dies sind dieselben projektseitigen Definitionen, auf die Abschnitt 5.4.1 verweist; hier wird keine neue fachliche Metrikbegruendung vorgenommen.

## 12. Gespeicherte Fallfelder

| Feldgruppe | Direkt gespeichert | Nicht direkt gespeichert / nur ableitbar |
|---|---|---|
| Identitaet | `id`, `db_id`, `db_path`, `question` | keine separate Quelle-/Splitspalte |
| SQL | `gold_sql`, `pred_sql`, `raw_output` | normalisierte SQL-Zeichenkette; Extraktionsstatus |
| Ausfuehrung | `gold_ok`, `pred_ok`, `exec_match`, `gold_error`, `pred_error` | Gold-/Prediction-Resultatzeilen; Spaltennamen; generische strukturierte Fehlerklasse |
| Stringmetriken | `string_exact`, `normalized_exact`, `char_accuracy`, `token_accuracy` | keine weitere Parsermetrik |
| Generation | Prompt-, Completion-, Gesamt-Tokens, Generationszeit, Tokens/s | ungekappte Vorlaenge; End-to-End-Fallzeit; aktiver Stopgrund |
| Reasoning/PPL | `reasoning_tokens` leer; `gold_perplexity` leer | Reasoningsegment; Perplexity, da `compute_perplexity=false` |
| Retrieval | IDs, Scores, DB-IDs, Methode, `num_fewshot_examples`; Gatefelder soweit aktiv | vollstaendige Kandidaten-/Rerankingdetails stehen im Trace |
| Runprovenienz | Modell, Adapter, Promptformat, Limits, Configpfad und weitere Runfelder | Runtime-Runner-Hash fehlt in 27 aelteren Hauptmetadaten |
| Timeout | Errorstring je Fall; v3-Summaryzaehler | eigenes boolesches Timeoutfeld fehlt |

Die zentrale Ergebnisdatei enthaelt daher genug Information fuer die gespeicherten EMA-/ESR-/Stringmetriken und gepaarte Fallvergleiche, aber nicht fuer eine vollstaendig neue Resultatsemantikpruefung ohne erneute SQL-Ausfuehrung.

## 13. Aggregation und Summary-Konsistenz

Der gemeinsame Nenner ist `total`. EMA, ESR, String EM und Normalized EM sind Zaehler geteilt durch `total`; Char und Token Accuracy sind arithmetische Mittel aller Fallwerte. Prompt-, Completion-, Gesamt-Token- und Generationszeitmittel werden aus den nichtleeren `GenerationResult`-Feldern gebildet. Retrievalsimilarity aggregiert die persistierten ausgewaehlten Scores; Gate- und Fallbackhaeufigkeiten werden separat gezaehlt.

Fallfloats werden im CSV auf sechs Dezimalstellen formatiert, waehrend die Metadata-Summary die ungerundeten In-Memory-Summen verwendet. Die statische Reproduktion tolerierte deshalb `1.1e-6` fuer Mittelwerte. Zaehler und Anteilsmetriken stimmen exakt.

| Integritaetscheck | Ergebnis |
|---|---:|
| autoritative Runs | 84/84 |
| Fallzeilen | 86.688/86.688 |
| Runs mit 1.032 eindeutigen IDs | 84/84 |
| identische Fallreihenfolge | 84/84 |
| fehlende / doppelte Fallzeilen | 0 / 0 |
| nichtleere Raw Outputs | 86.688/86.688 |
| leere extrahierte SQLs | 174 |
| konsistente Summaries | 84/84 |
| Summarymismatches | 0 |
| Retrievalruns / vollstaendige Traces | 78/78 |
| Tracezeilen | 80.496 |
| Prompttokens ueber konfiguriertem Limit | 0 |
| Completion-Limitfaelle Haupt/k3 | 2.234 / 1.765 |
| Gold-/Prediction-Timeouts | 0 / 2 |

Die Pruefung war rein statisch: CSV- und JSON-Felder wurden neu aggregiert; keine SQL-Abfrage wurde ausgefuehrt.

## 14. Hauptanalyse versus k3-Erweiterung

| Merkmal | 48-Run-Hauptanalyse | 36-Run-k3-Erweiterung |
|---|---|---|
| Demonstrationsanzahl | Zero Shot 0; Static/Dynamic k=1 | Dynamic k=3 oder vollstaendiger Gate-Fallback k=0 |
| Eingabelimit | 46 x 2.048; 2 x 1.536 | 36 x 4.352 |
| Ausgabelimit | 256 | 256 |
| Decoding | greedy, `do_sample=False` | identisch |
| Batchgroesse | 1 | 1 |
| SQL-Extraktion | `sql_first_statement_only` | identisch |
| SQLite-Ausfuehrung | read-only URI; kein expliziter Statement-Timeout | 24 ohne Timeout; 12 Qwen-9B-v3 mit 900 s je Statement |
| Nenner | 1.032 | 1.032 |
| Metriken | identische Fall- und Aggregationslogik | identisch; v3 zaehlt Timeoutstatus zusaetzlich |
| statistische Familie | autoritative Hauptfamilien | separate `K1_VS_K3_DEMONSTRATION_COUNT_FAMILY` |

Die Formulierung `84 ausgewertete Laeufe` ist korrekt. Die Formulierung `vollstaendig homogene 84-Run-Matrix` ist wegen Eingabelimit und Timeoutpolicy nicht korrekt.

## 15. Uebergang zur statistischen Auswertung

Die inferenzstatistische Weiterverarbeitung verwendet die binare Fallspalte `exec_match`. Vor Paarvergleichen werden 1.032 IDs und ihre Reihenfolge geprueft; die k1-k3-Analyse erzeugt zusaetzlich `by_id`-Mappen und verweigert abweichende Reihenfolgen. Es werden keine fehlenden Werte imputiert oder Fallzeilen ausgeschlossen, weil alle Paare vollstaendig sind.

Die Hauptanalyse behaelt ihre bereits definierten Holm-Familien. Der k1-k3-Vergleich verwendet die neue getrennte Familie `K1_VS_K3_DEMONSTRATION_COUNT_FAMILY`. Die statistischen Verfahren selbst gehoeren in Abschnitt 5.4.2.

Dokumentationswarnung: `scripts/analyze_cross_model_complete_8x8_synthesis.py` schreibt in seinem spaeter erzeugten Manifest an einer Stelle `data/testcases.jsonl`, kombiniert den Pfad aber mit SHA256 `6ce959...`, `N=1032` und der Fallreihenfolge des Full-Testsets. Die 48 Configs, CSVs und Metadaten belegen `data/testcases_spider_dev_full.jsonl`; die Zeile ist eine Manifest-Pfaddrift, keine 200-Fall-Auswertung.

## 16. Methodische Grenzen

1. Greedy Decoding liefert pro Run keine Samplingverteilung; das Design misst keine Generationsvarianz.
2. `max_new_tokens=256` begrenzt die Ausgabe. Ein Limitfall ist regulaer enthalten, kann aber eine unvollstaendige oder repetitive SQL-Ausgabe darstellen.
3. Der First-Statement-Extraktor ist eine projektspezifische Heuristik und kein vollstaendiger SQL-Parser.
4. Der SQLite-Schreibschutz basiert technisch auf URI-`mode=ro`; `PRAGMA query_only` und ein Authorizer fehlen.
5. EMA vergleicht Ergebnisse auf einer konkreten SQLite-Instanz. Resultatspaltennamen und -metadaten werden nicht beruecksichtigt; zwei leere Ergebnismengen gelten als gleich.
6. Es gibt keine explizite Floattoleranz oder vollstaendige Datentypnormalisierung.
7. Fehler und gespeicherte Timeouts bleiben im Nenner. Ein ungefangener Generierungs- oder Prozessfehler wuerde dagegen einen Lauf unvollstaendig machen; solche Runs wurden aus der autoritativen Matrix ausgeschlossen.
8. Die k3-Matrix hat eine gemischte Execution-Policy: 24 Runs ohne und 12 Runs mit 900-s-Statement-Timeout.
9. k3 besitzt ein hoeheres zulaessiges Eingabelimit. Der Aequivalenzcheck stuetzt k1-k3-Vergleiche, isoliert aber keinen reinen Kausaleffekt der Demonstrationszahl.
10. Die Fall-CSV speichert keine ungekappte Promptlaenge, keinen Stopgrund, keine Resultatzeilen und keine Spaltenmetadaten.
11. `generation_time_seconds` ist reine Generationszeit; End-to-End-Laufzeit steht nur in Runmetadaten.
12. Tokenzahlen sind innerhalb einer Modell-/Tokenizerlinie direkt interpretierbar; modelluebergreifend sind sie nicht vollstaendig tokenizerunabhaengig vergleichbar.
13. Bei 27 aelteren Hauptlaeufen fehlt der direkt persistierte Runtime-Runner-Hash. Die Methodik wird durch Configfelder, Ergebnisfelder, spaetere gehashte Runnerstaende und unabhaengige Rescoringaudits gestuetzt, nicht durch einen lueckenlosen per-Run-Codehash.

## 17. Thesisfertige Methodenzusammenfassung

### 17.1 Promptaufbau und Tokenisierung

Jeder Evaluationsfall enthielt eine natuerlichsprachliche Frage, die vollstaendige textuelle Darstellung des zugehoerigen Datenbankschemas, die Datenbankkennung und eine Referenzabfrage. Der systemseitige Prompt forderte genau eine SQLite-Abfrage ohne Erlaeuterungen und beschraenkte die Ausgabe auf die im Schema enthaltenen Tabellen und Spalten. Bei Qwen wurden System- und Nutzerinhalt manuell im projektspezifischen SQLCTX-ChatML-Format serialisiert und unmittelbar mit dem Assistant-Beginn abgeschlossen. Fuer Llama 3.2 3B Instruct wurde das native Chattemplate mit einem festen Datumswert verwendet. Few-Shot-Prompts stellten die ausgewaehlten Demonstrationen in Retrievalreihenfolge vor den Zielfall; bei abgelehntem Similarity-Gate wurde der regulaere Zero-Shot-Prompt erzeugt.

### 17.2 Eingabe- und Ausgabelimits

Die Hauptuntersuchung verwendete grundsaetzlich ein Eingabelimit von 2.048 Tokens. Zwei fruehe Base-Zero-Shot-Laeufe waren mit 1.536 Tokens konfiguriert; ihr beobachtetes Promptmaximum von 736 Tokens lag deutlich darunter. Die k3-Erweiterung verwendete 4.352 Eingabetokens, um drei Full-Schema-Demonstrationen ohne Kuerzung aufzunehmen. Vorab- und Abschlusspruefungen ergaben weder in der Hauptuntersuchung noch in der k3-Erweiterung aktive Prompttruncationen. Das Ausgabelimit betrug in allen 84 Laeufen 256 neue Tokens.

### 17.3 Deterministische Generierung

Die Generierung erfolgte fallweise mit Batchgroesse eins und greedy Decoding. Der Evaluationsrunner setzte `do_sample=False`, verwendete keinen Beam Search und aktivierte den Generationscache. Nur die nach dem Prompt erzeugten Tokens wurden dekodiert; Special Tokens wurden beim Dekodieren ausgelassen. Die protokollierte Generationszeit umfasst ausschliesslich den Aufruf der Modellgenerierung und nicht Retrieval oder SQL-Ausfuehrung.

### 17.4 SQL-Extraktion

Aus der rohen Modellausgabe wurde mit dem projektspezifischen Modus `sql_first_statement_only` der erste akzeptierte SQL-Kandidat extrahiert. Der Extraktor entfernte bekannte Modell-, Rollen- und Markdownmarker, suchte ausserhalb von Zeichenketten und Klammern nach einem top-level beginnenden `SELECT` oder einer CTE-artigen `WITH`-Abfrage und begrenzte den Kandidaten am ersten top-level Semikolon. Nachfolgender Text beziehungsweise weitere Statements wurden nicht ausgefuehrt. Bei einem balancierten Kandidaten ohne Semikolon wurde dieses ergaenzt; andernfalls blieb die extrahierte SQL leer.

### 17.5 Read-only-SQL-Pruefung

Die Ausfuehrung wurde zweifach begrenzt: Der Extraktor akzeptierte nur lesende `SELECT`- beziehungsweise CTE-`WITH`-Kandidaten, und SQLite oeffnete jede Datenbank mit dem URI-Parameter `mode=ro`. Damit beruhte der technische Schreibschutz auf der read-only SQLite-Verbindung und nicht allein auf der Formulierung des Prompts.

### 17.6 SQLite-Ausfuehrung und Fehlerbehandlung

Fuer jeden Fall wurde zuerst die Referenzabfrage und anschliessend die extrahierte Modellabfrage gegen dieselbe read-only Verbindung der zugehoerigen Spider-Datenbank ausgefuehrt. Erfolgreiche Abfragen wurden mit `fetchall()` vollstaendig eingelesen; Ausnahmen wurden als nicht erfolgreiche Ausfuehrungen samt Fehlertext gespeichert. Leere Extraktionen wurden ohne SQLite-Aufruf als nicht ausfuehrbar behandelt. Die 48 Hauptlaeufe und 24 k3-Laeufe der kleineren Modelllinien besassen keinen expliziten Statement-Timeout. Die zwoelf Qwen-9B-k3-v3-Laeufe begrenzten dagegen jedes Gold- und Prediction-Statement mittels SQLite-Progresshandler auf 900 Sekunden. Zwei Predictionabfragen erreichten dieses Limit und wurden regulaer als nicht ausfuehrbar und nicht korrekt gespeichert.

### 17.7 Ergebnisvergleich

Fuer den fallweisen Execution Match mussten Referenz- und Predictionabfrage erfolgreich ausgefuehrt werden. Anschliessend wurden ihre Ergebnistupel zeilenreihenfolgeunabhaengig sortiert und exakt verglichen; Duplikate und Spaltenreihenfolge blieben dabei erhalten. Die Execution Success Rate erfasste, ob die Predictionabfrage ohne Fehler ausgefuehrt werden konnte. Die aus den Fallresultaten aggregierten Kennzahlen entsprechen den in Abschnitt 5.4.1 definierten Evaluationsmetriken. Die statistische Auswertung gepaarter Bedingungen folgt Abschnitt 5.4.2.

### 17.8 Aggregation der 1.032 Faelle

Alle autoritativen Auswertungen verwendeten dieselben 1.032 Spider-Dev-Faelle. Leere Extraktionen, SQL-Fehler und gespeicherte Timeouts wurden nicht ausgeschlossen, sondern mit negativem Ausfuehrungs- und Matchstatus im gemeinsamen Nenner belassen. Neben den Erfolgs- und Matchwerten protokollierte der Runner die rohe Ausgabe, extrahierte SQL, Tokenzahlen und Generationszeit sowie bei Few-Shot-Bedingungen die Retrieval- und Gateinformationen. Die Run-Summary wurde aus den fallweisen Zaehlern und Summen gebildet.

### 17.9 Unterschiede der k3-Erweiterung

Die k3-Erweiterung uebernahm Promptformat, Systemprompt, greedy Decoding, Batchgroesse, Ausgabelimit, Extraktor und Metrikaggregation aus der Hauptuntersuchung. Sie unterschied sich durch drei statt einer Demonstration, ein auf 4.352 Tokens erweitertes Eingabebudget und bei den zwoelf Qwen-9B-v3-Laeufen durch den 900-Sekunden-Timeout. Die 48 Hauptlaeufe und 36 k3-Laeufe bilden deshalb 84 ausgewertete Laeufe, jedoch keine vollstaendig homogene 84-Run-Matrix.

### 17.10 Methodische Grenzen

Die Auswertung bildet deterministische Einzelgenerationen unter einem begrenzten Ausgabebudget ab. Die Extraktion auf das erste Statement und der instanzgebundene SQLite-Ergebnisvergleich sind projektspezifische Operationalisierungen. Zudem muessen die gemischte Timeoutpolicy der k3-Matrix, das hoehere k3-Eingabelimit und die fehlende Persistierung vollstaendiger Resultatzeilen bei der Interpretation beruecksichtigt werden.

## 18. Evidenzmatrix

Die vollstaendige 50-zeilige Evidenzmatrix steht in `audits/derived/generation_evaluation_pipeline_evidence_20260718.csv`. Zentrale Aussagen:

| Aussage | Technischer Befund | Evidenz | Status | Fuer Methodik geeignet |
|---|---|---|---|---|
| Testset | Full-JSONL, SHA `6ce959...`, 1.032 IDs | Configs, Fall-CSV, statischer Hashcheck | `PROJECT-VERIFIED` | ja |
| Systemprompt | Variante und SHA in 84/84 Runs gleich | `prompt_presets.py`, Metadaten | `AUDIT-VERIFIED` | ja |
| Promptserialisierung | Qwen manuelles ChatML; Llama natives Template | Runner und Llama-Helfer | `PROJECT-VERIFIED` | ja |
| Eingabelimits | 2 x 1.536, 46 x 2.048, 36 x 4.352 | 84er Runmatrix | `AUDIT-VERIFIED` | ja, differenziert |
| Truncation | Tokenizer kann truncieren; in autoritativen Prompts 0 aktiv | Runner plus Promptaudits | `AUDIT-VERIFIED` | ja |
| Generierung | greedy, Batch 1, 256, Cache | `model.generate`-Code und Configs | `PROJECT-VERIFIED` | ja |
| SQL-Extraktion | erster top-level SELECT/WITH-Kandidat | Runner Zeilen 760-1162 | `PROJECT-VERIFIED` | ja, als Heuristik |
| Read-only | SELECT/WITH-Filter plus SQLite `mode=ro` | Runner Zeilen 3352-3380 | `PROJECT-VERIFIED` | ja |
| `query_only`/Authorizer | nicht vorhanden | Codeaudit | `PROJECT-VERIFIED` | als Grenze |
| SQLite-Reihenfolge | Gold vor Prediction, gleiche Connection | Runner | `PROJECT-VERIFIED` | ja |
| EMA | beide ok; sortierte Tupellisten gleich | `execution_match` | `PROJECT-VERIFIED` | ja |
| ESR | `pred_ok / total` | Fall- und Summarycode | `PROJECT-VERIFIED` | ja |
| Fehler im Nenner | ja, sobald `write_case_result` erreicht ist | Runner | `PROJECT-VERIFIED` | ja, mit Generierungsfehler-Caveat |
| Timeoutpolicy | 72 ohne expliziten Guard, 12 mit 900 s | Runnerprovenienz und Configs | `AUDIT-VERIFIED` | ja, zwingend getrennt |
| 84er Konsistenz | 84/84 Summaries und Reihenfolgen konsistent | statische Reaggregation | `AUDIT-VERIFIED` | ja |
| Statistikuebergang | `exec_match`, Paarung ueber ID/Reihenfolge | Analyseskripte | `PROJECT-VERIFIED` | ja |
| Runtime-Runnerhash | in 27 aelteren Hauptmetadaten nicht persistiert | Runmatrix | `UNRESOLVED_PER_RUN` | als Warnung |

## 19. Erzeugte Artefakte

| Datei | SHA256 |
|---|---|
| `audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv` | `7d6107d942ea0f2890354cb426eebf929597dcb36f5c5108df1abce0b2c921dc` |
| `audits/derived/generation_evaluation_pipeline_evidence_20260718.csv` | `95d8f87dc852746c8b738cc2dff19570bca34577f9813c8c44878064e16980cd` |
| `audits/derived/method_generation_evaluation_pipeline_20260718.json` | `c83ff22ecd2c907bcc160a1402527f3e69329a67fd0dd989d8265db3873bb9e2` |
| `audits/audit_method_generation_evaluation_pipeline_20260718.md` | nach Erstellung extern zu hashen; kein selbstreferenzieller Hash im Dokument |

## 20. Offene beziehungsweise nicht eindeutig aufloesbare Punkte

1. Der exakte Runtime-Runner-Hash ist fuer 27 aeltere Hauptlaeufe nicht direkt persistiert. Ihre Config-, Prompt-, Extraktor- und Ergebnisfelder sowie spaetere unabhaengige Rescorings sind konsistent; eine lueckenlose per-Run-Codeprovenienz kann dennoch nicht rueckwirkend hergestellt werden.
2. Die tatsaechliche Tokenizer-`truncation_side` ist in den lokalen Tokenizerconfigs nicht eingetragen. Sie war wegen null aktiver Truncationen folgenlos.
3. Qwen-EOS-/PAD-Overrides fehlen in aelteren Metadaten; der Runner uebergibt keine explizite Ueberschreibung, sodass die Modell-/Tokenizerkonfiguration massgeblich ist.
4. Fetched Resultatzeilen, Spaltennamen und SQLite-`description` wurden nicht gespeichert. Der statische Abschlussaudit kann deshalb die vorhandenen Matchflags und Summaries reproduzieren, aber den Resultatvergleich nicht ohne verbotene SQL-Neuausfuehrung erneut bilden.
5. Ein unhandled Fehler innerhalb der Modellgenerierung waere kein regulärer negativer Fall, sondern wuerde den Run abbrechen. Die Vollstaendigkeitsregel der autoritativen Matrix verhindert, dass ein solcher Teilrun ausgewertet wird.

## 21. Abschlussblock

```text
Autoritativer Testsetpfad: data/testcases_spider_dev_full.jsonl
Testset-SHA256: 6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce
Fallzahl: 1032
Eindeutige Case-IDs: 1032
Fallreihenfolge ueber Runs identisch: JA
Prompttemplate Qwen: qwen_sqlctx_chatml
Prompttemplate Llama: llama32_instruct_native_chat
Systemprompt identisch: JA
Hauptanalyse-Eingabelimit: 2048; zwei nichtbindende Base-Zero-Shot-Runs 1536
k3-Eingabelimit: 4352
Ausgabelimit: 256 neue Tokens
Promptueberschreitungen: 0
Truncationsfaelle: 0
Batchgroesse: 1
Decodingverfahren: greedy
Sampling aktiviert: NEIN
Beam Search aktiviert: NEIN
Generationseed: keiner gesetzt
SQL-Extraktionsregel: sql_first_statement_only
Erstes Statement: erster akzeptierter top-level SELECT-/CTE-WITH-Kandidat bis zum ersten top-level Semikolon ausserhalb von Quotes und Klammern
Fehlendes Semikolon: bei balanciertem akzeptiertem Kandidaten ergaenzt
Erlaubte Statementtypen: SELECT und CTE-artiges WITH mit SELECT
Read-only-Schutzmechanismus: Extraktionsfilter plus SQLite-URI mode=ro; kein query_only und kein Authorizer
Referenz-SQL-Ausfuehrung: je Fall und Run zuerst, per fetchall
Vorhersage-SQL-Ausfuehrung: danach in derselben read-only Connection, sofern extrahiert
EMA-Vergleichsregel: beide ok und zeilenreihenfolgeunabhaengig sortierte Tupellisten mit Python-Gleichheit identisch
ESR-Ausfuehrbarkeitsregel: pred_ok / 1032
Fehler im Nenner: JA, sofern als Fallresultat gespeichert
Timeouts im Nenner: JA
k3-Timeoutpolicy: 24 ohne expliziten Statement-Timeout; 12 Qwen-9B-v3 mit 900 s pro Gold-/Prediction-Statement
Anzahl autoritativer Hauptlaeufe: 48
Anzahl autoritativer k3-Laeufe: 36
Vollstaendige Runs: 84/84
Fallzeilen insgesamt: 86688
Fehlende/doppelte Faelle: 0/0
Summary-Konsistenz: 84/84
Retrievaltrace-Konsistenz: 78/78 Retrievalruns; 6 Zero-Shot-Runs ohne Trace
Statistische Paarung ueber case_id: JA
Haupt- und k3-Holm-Familien getrennt: JA
Ungeloeste Punkte: 27 historische Runtime-Runnerhashes; nicht persistierte Resultatzeilen/Spaltenmetadaten; folgenlose Truncation-side-Provenienz
Geaenderte Bestandsdateien: keine
Training gestartet: nein
Evaluation gestartet: nein
Modellinferenz gestartet: nein
Retrieval gestartet: nein
SQL-Ausfuehrung neu gestartet: nein

GENERATION-EVALUATION-METHOD-AUDIT: PASS MIT WARNUNGEN
SQL-EXTRACTION-RECONSTRUCTION: PASS MIT WARNUNGEN
SQLITE-EXECUTION-RECONSTRUCTION: PASS MIT WARNUNGEN
84-RUN-RESULT-CONSISTENCY: PASS
THESIS-SECTION-5.6.4: READY_WITH_LIMITATIONS
```
