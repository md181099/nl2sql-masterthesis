# Methodenaudit 5.7: Methodische Herausforderungen und Reproduzierbarkeit

Auditlabel: 2026-07-19  
Ausfuehrt am: 2026-07-18, UTC  
Projektwurzel: `/home/ec2-user/nl2sql_testbench`

## Executive Summary

```text
REPRODUCIBILITY-METHOD-AUDIT: PASS MIT WARNUNGEN
AUTHORITATIVE-RUN-PROVENANCE: PASS MIT WARNUNGEN
PROJECT-FREEZE-STATUS: PARTIAL
REPRODUCIBILITY-GAPS: DOCUMENTED
THESIS-SECTION-5.7: READY_WITH_LIMITATIONS
```

Die drei autoritativen LoRA-v2-Trainingslinien, die 48-Runs-Hauptuntersuchung und die additive 36-Run-k3-Erweiterung sind durch persistierte Configs, Daten-, Adapter-, Ergebnis- und Traceartefakte sowie gestufte Provenienzaudits eindeutig rekonstruierbar. Alle 84 Evaluationsconfigs, Ergebnis-CSVs und Metadaten sowie alle 78 erforderlichen Retrievaltraces wurden in diesem Audit erneut gegen ihre registrierten SHA256-Werte geprueft. Es bestehen 84 konsistente Summaries, 86.688 vollstaendige Fallzeilen, keine fehlenden oder doppelten Faelle und keine Prompttruncation.

Die methodische Rechenlinie ist damit fuer Kapitel 5.7 belastbar dokumentierbar. Die verbleibenden Warnungen betreffen historische Laufzeitprovenienz, nicht die Zuordnung der autoritativen Ergebnisse: Bei 27 aelteren Hauptlaeufen wurde der Runtime-Runner-Hash nicht direkt in den Metadaten persistiert; die beiden Qwen-Modellrevisionen und deren historische Interpreterpfade sind stark trianguliert, aber nicht vollstaendig direkte Laufzeitevidenz; fuer die drei Trainingslaeufe fehlen separate Konsolenlogs.

Der **finale Projektfreeze ist dagegen nicht abgeschlossen**. `.venv_flash` ist als autoritative Freeze-Umgebung technisch freigegeben, doch dies ist kein Dateisystemfreeze. Es existieren weder ein aktueller vollstaendiger Projektsnapshot noch ein aktuelles rekursives SHA256-Manifest, ein finales schreibgeschuetztes Archiv, zwei unabhaengige externe Backups oder ein Restore-Test. Ein lesbares Same-Host-Archiv vom 12.05.2026 ist historisch und liegt zeitlich vor den autoritativen LoRA-v2- und k3-Artefakten. Die Freeze-Checkliste ist weiterhin unausgefuehrt.

## 1. Untersuchungsumfang und Arbeitsmodus

Geprueft wurden:

- drei autoritative LoRA-v2-Trainingslinien,
- 48 autoritative Hauptlaeufe mit acht Bedingungen je Modellrolle,
- 36 autoritative k3-Erweiterungslaeufe mit sechs Bedingungen je Modellrolle,
- Trainings- und Evaluationsconfigs, Adapter, Trainer-States und Laufmetadaten,
- Testset, Trainings- und Validierungsdaten sowie Retrievalindex und -traces,
- Statistik-, Fehleranalyse-, Environment-, Abschluss- und Freeze-Audits,
- physisch vorhandene Manifeste, Archive, Git-Metadaten und Dateiberechtigungen.

Vor Beginn war kein relevanter Trainings-, Evaluations- oder Analysewriter aktiv. Es wurde kein Training, keine Evaluation, keine Modell- oder Adapterinferenz, kein Retrieval, keine Embeddingberechnung und keine SQL-Ausfuehrung gestartet. Es wurden kein Snapshot, Archiv, Backup oder Git-Repository erzeugt. Bestehende Dateien und Berechtigungen blieben unveraendert; neu sind ausschliesslich die fuenf in Abschnitt 21 aufgefuehrten Auditdateien.

Technischer Hinweis: Die Ausfuehrungsumgebung aktualisiert waehrend der Sitzung automatisch den Zeitstempel des bereits vorhandenen leeren Workspace-Steuermarkers `.codex`. Inhalt, Groesse (0 Byte) und SHA256 `e3b0c442...b855` blieben unveraendert. Dies ist keine fachliche Projektdatei und keine Inhaltsaenderung eines Forschungsartefakts.

## 2. Vorhandene Reproduzierbarkeitsartefakte

Die vollstaendige zentrale Artefaktliste steht in `audits/derived/reproducibility_artifact_inventory_20260719.csv`. Der Bestand laesst sich in vier Ebenen gliedern:

| Ebene | Beispiele | Abdeckung | Status |
|---|---|---:|---|
| Ausfuehrungsdefinition | 3 Trainingsconfigs, 84 Evaluationsconfigs, Runner und Prompt-/Retrievalmodule | fachlich vollstaendig | `IMPLEMENTED_WITH_WARNINGS` |
| Laufartefakte | Adapter, Trainer-States, 84 CSVs, 84 Metadaten, 78 Traces | vollstaendig fuer autoritative Runs | `IMPLEMENTED` |
| Teilmanifeste und Audits | Trainings-, Modell-, Index-, Run-, Statistik- und Environmentmanifeste | zentrale Bereiche vollstaendig, projektweit verteilt | `IMPLEMENTED_WITH_WARNINGS` |
| Finaler Freeze | Gesamtsnapshot, aktuelles Rekursivmanifest, Archiv, Schreibschutz, externe Backups, Restore-Test | nicht umgesetzt | `PLANNED_NOT_IMPLEMENTED` |

Das 39.912.646 Byte grosse `COMPLETE_ARTIFACT_INVENTORY.csv` enthaelt 94.212 gehashte regulaere Dateien des Basisscans vom 16.07.2026. Es ist ein wertvoller historischer Integritaetsnachweis, aber kein finales aktuelles Manifest: Das Dokumentationsverzeichnis war aus dem Basisscope ausgeschlossen, und spaetere k3-, Literatur- und Methodenaudits entstanden erst am 17. und 18.07. Vor Erzeugung der vorliegenden Additivdateien enthielt der Projektbaum 94.707 regulaere Dateien.

## 3. Config- und Runprovenienz

### 3.1 Training

Fuer jede der drei LoRA-v2-Linien liegen eine eindeutige Config, Trainings- und Validierungsdatasets, Adapterconfig, Adaptergewichte, Trainingsmetadaten und ein abschliessender Trainer-State vor. Die erneut geprueften Config- und Adapterhashes sind:

| Modelllinie | Trainingsconfig-SHA256 | Adapter-SHA256 | Modellrevision | Revisionsevidenz |
|---|---|---|---|---|
| Qwen 3.5 2B | `020662e3158f1d848e0c55976197b57a211b4e92fa96ebc6aca5dd453542b327` | `6b92f120365d127d0c51a4c532953207d65cff611ac08cb7d573880be18223f3` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | trianguliert, hohe Konfidenz |
| Llama 3.2 3B Instruct | `81001f1a2b6287d589412bab658d50290fa1ba33c9e5557d44b9ff04a1c4282b` | `fcd4241f7a2e8e0388f13f0dd9517486cbee43fc3169c983a54e7b716c0e502d` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | direkt persistiert |
| Qwen 3.5 9B | `0bfce20d1e97f0b42b61d3db67679e3feef46b94a58a09147e4a5fb82240815e` | `dddf120df0703be5b9106ba17a628f2a9664e6ab5d1cc3ec1311c0a4a2b000f0` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | trianguliert, hohe Konfidenz |

Die Root-Adapter entsprechen laut Training- und Adapteraudit jeweils dem ausgewaehlten Zustand aus Epoche 1: Checkpoint 502 fuer beide Qwen-Linien und Checkpoint 509 fuer Llama. Die abgeschlossenen Trainer-States reichen bis Schritt 1.506, 1.527 beziehungsweise 1.506 und dokumentieren auch die spaeteren, nicht ausgewaehlten Epochen. Der historische Qwen-Einstiegspunkt hatte SHA256 `f34e6fc2...e3c64`; der heutige Dateistand besitzt `37d038ef...5eaf7`. Nur Llama persistiert den heutigen Stand direkt. Dieser Unterschied wird nicht rueckwirkend umgeschrieben.

### 3.2 Evaluation

Die maschinenlesbare Runmatrix weist fuer alle 84 Runs Config, Config-Hash, CSV, Metadata, Trace, Modell-/Adapteridentitaet, Runnerprovenienz und Fallintegritaet aus. Die aktuelle read-only Hashpruefung ergab:

- Evaluationsconfigs: 84/84 vorhanden, 84/84 SHA256-Matches,
- Ergebnis-CSVs: 84/84 vorhanden, 84/84 SHA256-Matches,
- Metadaten: 84/84 vorhanden, 84/84 SHA256-Matches,
- erforderliche Retrievaltraces: 78/78 vorhanden, 78/78 SHA256-Matches.

In 57/84 Runs wurde ein Runtime-Runner-Hash persistiert. Bei 27 aelteren Hauptlaeufen fehlt er. Deren methodische Zuordnung bleibt durch Configs, Ergebnisfelder, Testset- und Promptidentitaet, spaetere Reproduktions- und Rescoringaudits sowie gehashte Runnerstaende stark gestuetzt, ist jedoch keine bitgenaue direkte Executable-Provenienz.

## 4. Seeds und Determinismus

| Prozess | Persistierte Festlegung | Reichweite | Grenze |
|---|---|---|---|
| LoRA-Training | Seed 42 | alle drei Linien | nur ein Trainingsseed je Linie; kein separater `data_seed` |
| Static Few Shot | Seed 42 | Demo `SPIDER_TRAIN_001657` in allen sechs Rollen | deterministische Ressource, keine Seedvariabilitaet untersucht |
| Dynamisches Retrieval | gehashter Index und deterministische Sortierung | alle 78 Retrievalruns | k3 besitzt expliziten Stable-ID-Tie-Break; k1 nicht gleich stark expliziert |
| Generation | Greedy, `do_sample=false`, Batch 1 | alle 84 Runs | kein Generationseed und keine expliziten deterministischen CUDA-Backendflags |
| Hauptstatistik | Seed 20260716, 10.000 Bootstrapresamples | getrennte Hauptfamilien | Bibliotheks-/Plattformidentitaet nicht garantiert |
| k1-k3-Statistik | Seed 20260716, 10.000 Bootstrapresamples | eigene 36er-Holm-Familie | nicht mit Hauptfamilien vermischt |

Greedy Decoding beseitigt Samplingvarianz der Generierung, belegt aber allein keine plattformuebergreifende Bitidentitaet. Ebenso stellt Seed 42 keine vollstaendige Reproduzierbarkeitsgarantie dar. Die Thesis darf daher von kontrollierten Seeds und deterministischem Decoding sprechen, nicht von universell bitidentischer Wiederholbarkeit.

## 5. Autoritative Ein- und Ausschlussregeln

Ein Evaluationsrun wurde nur freigegeben, wenn er folgende Kriterien erfuellte:

1. 1.032 Fallzeilen und 1.032 eindeutige autoritative Case-IDs,
2. keine fehlenden, doppelten oder unerwarteten IDs,
3. exakte Fallreihenfolge des autoritativen Testsets,
4. eindeutige Config-, Modell-, Adapter- und Testsetzuordnung,
5. vollstaendige Metadaten und eine aus den Fallzeilen reproduzierbare Summary,
6. vollstaendiger geordneter Retrievaltrace bei Retrievalbedingungen,
7. keine ungeklärte Prompttruncation,
8. vollstaendige Generierung des gesamten Runs,
9. Fehler, leere Extraktionen und Timeouts bleiben im 1.032er-Nenner,
10. keine Vermischung mit historischen Methoden, Adaptern, Testsets oder Teilruns.

| Kriterium | 48-Run-Hauptmatrix | 36-Run-k3-Erweiterung | Status |
|---|---:|---:|---|
| 1.032 Zeilen | 48/48 | 36/36 | `IMPLEMENTED` |
| 1.032 eindeutige IDs | 48/48 | 36/36 | `IMPLEMENTED` |
| autoritative Reihenfolge | 48/48 | 36/36 | `IMPLEMENTED` |
| Summary konsistent | 48/48 | 36/36 | `IMPLEMENTED` |
| erforderlicher Trace vollstaendig | 42/42 | 36/36 | `IMPLEMENTED` |
| Prompttruncation | 0 | 0 | `IMPLEMENTED` |
| fehlende/duplizierte Fallzeilen | 0/0 | 0/0 | `IMPLEMENTED` |

Die vollstaendige Regelmatrix steht in `audits/derived/authoritative_run_acceptance_rules_20260719.csv`.

## 6. Historische und ausgeschlossene Runs

Folgende Gruppen duerfen nicht in die finale LoRA-v2- oder 84-Run-Auswertung eingehen:

- LoRA-v1-Adapter und alternative Rang-, Lernraten-, Epochen- oder Validierungslinien,
- historische 1.024-Token-Trainings sowie Completion-only- und No-Packing-Ablationen,
- Testmode-, Smoke- und Teilmengenlaeufe,
- nicht ausgewaehlte Checkpoints und nachtraegliche SQL-Loss-Analysen als Auswahlgrundlage,
- Retrievalpools mit 7.063 oder 7.895 Beispielen sowie historische `sqlaware_topk`-Pfade,
- der 200-Fall-Bestand `data/testcases.jsonl` als vermeintlicher Volltest,
- die zwei historischen Qwen-9B-Base-Top-3-Teilresultate mit 479 und 482 gespeicherten Zeilen.

Die beiden k3-Teilresultate wurden nicht kombiniert, nicht fortgesetzt und nicht in den vollstaendigen Ersatzrun integriert. Ihre Hashes blieben beim spaeteren Matrixabschluss unveraendert. Autoritative Inventare weisen genau einen freigegebenen Run je Bedingung aus.

## 7. Methodische Herausforderungen

### 7.1 Qwen-Train-Eval-Promptmismatch

Historische Qwen-Linien erzeugten unerwuenschte `<think>`-Anteile beziehungsweise verwendeten einen unpassenden Assistant-Start. Ursache war eine nicht zur SQL-only-Aufgabe passende Chattemplate-/Prefixkombination. Die finale Linie verwendet das projektspezifische `qwen_sqlctx_chatml`, den Systemprompt `sqlctx_anti_overjoin` und einen expliziten Assistant-Beginn. Der LoRA-Trainingsaudit bestaetigt fuer Qwen dieselbe fachliche System-User-Assistant-Struktur in Training und Evaluation und keinen eingefuegten `<think>`-Block. Historische Mismatch-Runs bleiben ausgeschlossen.

**Methodische Konsequenz:** Die finale Hauptlinie ist promptseitig konsistent; die Entwicklungsgeschichte darf nicht als Eigenschaft der finalen Adapter dargestellt werden.

### 7.2 Testset-Pfaddrift

Der autoritative Evaluationsbestand umfasst 1.032 Faelle und den SHA256 `6ce959230b7b6c3b564a7bdc8a4cb904a6dd62e78f245569489c218dcf1bdcce`. Die historische Datei `data/testcases.jsonl` umfasst dagegen 200 Faelle und SHA256 `f7372af3...ed4`. Alle 84 autoritativen Runartefakte enthalten die 1.032er Fallidentitaet und -reihenfolge.

Eine spaetere Cross-Model-Synthese nennt in einem Manifesttext irrtuemlich den 200er-Pfad, persistiert dort aber zugleich den vollen Testsethash und 1.032 Faelle. Der Fehler ist eine dokumentierte Pfaddrift, keine tatsaechliche Datensatzverwechslung. Fuer die Thesis ist der Hash zusammen mit Fallzahl und Case-ID-Reihenfolge massgeblich.

### 7.3 Sequenzlaengen und Prompttruncation

Historische 1.024-Token-Trainingslinien sind ausgeschlossen. Die Hauptlinie verwendet im Regelfall 2.048 Eingabetokens; zwei autoritative Base-Zero-Shot-Laeufe wurden mit 1.536 konfiguriert, erreichten aber hoechstens 736 Prompttokens und waren daher nicht limitgebunden. Das Promptmaximum der Hauptmatrix betraegt 2.045 Tokens. Die k3-Erweiterung wurde nach einem blockierenden 2.048er-Preflight methodisch auf 4.352 Eingabetokens festgelegt; ihr Maximum betraegt 4.269 Tokens.

In den autoritativen 84 Runs traten null Promptueberschreitungen und null Truncationen auf. Der k1-Aequivalenzaudit bestaetigt, dass die historischen k1-Prompts bei hypothetisch 4.352 Tokens weder Text-, Hash- noch Token-ID-Unterschiede zeigen. Gleichwohl misst k1 gegen k3 den Unterschied der Demonstrationszahl unter unterschiedlichen zulaessigen Kontextbudgets und nicht den isolierten Kausaleffekt einer einzigen Variablen.

### 7.4 SQL-Timeoutproblem

Beim urspruenglichen Qwen-9B-Base-Top-3-k3-Lauf blockierte eine pathologisch teure Prediction fuer `SPIDER_DEV_000484` in `wta_1`. Der v1-Runner besass keine Statementdeadline. Im ersten Timeoutversuch unterbrach der SQLite-Progress-Handler die Prediction nach 900 Sekunden korrekt; anschliessend brach der Runner wegen einer fehlenden `nonlocal`-Deklaration fuer den Timeoutzaehler ab. Die betroffene Ergebniszeile wurde noch nicht geschrieben.

Die v3-Implementierung verwendet `sqlite3.set_progress_handler` mit monotoner Deadline und 900 Sekunden **pro Gold- und separat pro Prediction-Statement**. Timeouts werden als nicht ausfuehrbar und nicht ergebnisgleich gespeichert, bleiben im Nenner und der Run setzt mit dem naechsten Fall fort. In der abgeschlossenen Matrix sind zwei Prediction-Timeouts gespeichert, beide bei `SPIDER_DEV_000484`: Qwen-9B Base Top-3 und Structure Top-3. Gold-Timeouts traten nicht auf.

**Verbleibende Grenze:** Die Timeoutpolicy ist innerhalb der k3-Matrix gemischt. Die 24 Qwen-2B-/Llama-k3-Laeufe besitzen keine explizite Statementdeadline, die zwoelf Qwen-9B-v3-Laeufe verwenden 900 Sekunden. Diese Policy darf nicht als einheitlich beschrieben werden.

### 7.5 Validatorpfadfehler

Nach elf vollstaendigen Qwen-9B-v3-Laeufen erwartete der Gruppenvalidator faelschlich:

```text
logs/k3_extension_20260718_sqltimeout900_v3
```

Tatsaechlich wurden Logs unter folgendem Pfad geschrieben:

```text
logs/k3_extension_20260718_sqltimeout900v3
```

Die Reparatur bestand aus genau dieser Ein-Zeilen-Pfadkorrektur. Runner, Config und Timeoutwert blieben unveraendert. Danach wurde ausschliesslich der fehlende Qwen-9B-Base-Top-3-v3-Vollrun unter neuer Identitaet ausgefuehrt. Kein anderer Run wurde wiederholt; historische Teilresultate blieben unveraendert.

### 7.6 Adapter- und Trainingsprovenienz

Der Llama-Snapshot ist direkt in Trainingsmetadaten und Datasetmanifest persistiert. Bei Qwen 2B und 9B wurden die Revisionen aus zeitnahen Preflights, den einzigen passenden lokalen Cache-Snapshots, `refs/main`, Evaluationsmanifesten und Dateihashes rekonstruiert. Diese Triangulation ist konsistent und hoch vertrauenswuerdig, aber nicht gleichwertig mit einem direkt im historischen Trainingsrecord gespeicherten Revisionsfeld.

Die drei Trainingslinien entsprechen mit hoher Konfidenz `.venv_flash` (EENV-2): PyTorch 2.12.1, CUDA 13.0, FlashAttention2 und NVIDIA L40S passen eindeutig zu dieser Umgebung und nicht zu `.venv`. Bei den Qwen-Linien fehlen jedoch `sys.executable` und Pythonversion als direkte historische Laufzeitfelder. Separate Konsolenlogs wurden fuer keinen der drei Trainingslaeufe gefunden; Trainingsmetadaten, History und Trainer-State tragen den Verlauf.

### 7.7 Runnerprovenienz aelterer Evaluationen

27 aeltere Hauptlaeufe speichern keinen Runtime-Runner-Hash. Diese Laeufe wurden nicht allein aufgrund heutiger Dateien freigegeben. Massgeblich sind die Kombination aus persistierter Config, Fallfeldern, Prompt-/Testsetidentitaet, vollstaendigen Ergebnissen und Traces, spaeteren Metrik- und SQLite-Rescoringaudits sowie bekannten gehashten Runnerstaenden. Die fachliche Vergleichbarkeit ist stark belegt; eine bitgenaue historische Runnerrekonstruktion bleibt fuer diese 27 Laeufe begrenzt.

## 8. Hauptmatrix versus k3-Erweiterung

| Merkmal | 48-Run-Hauptanalyse | 36-Run-k3-Erweiterung |
|---|---|---|
| Rolle im Projekt | `PRE-K3 AUTHORITATIVE BASELINE` | additive Erweiterungsanalyse |
| Bedingungen je Rolle | 8 | 6 |
| Demonstrationen | 0, static k1, dynamic k1 | dynamic k3 oder Gate-Fallback k0 |
| Eingabelimit | 46 x 2.048; 2 x nichtbindend 1.536 | 36 x 4.352 |
| Prompttruncationen | 0 | 0 |
| Ausgabelimit | 256 | 256 |
| Decoding | greedy, Batch 1 | greedy, Batch 1 |
| SQL-Timeout | kein explizites Statementtimeout | 24 ohne; 12 Qwen9 mit 900 s |
| Nenner | 1.032 je Run | 1.032 je Run |
| statistische Familien | eingefrorene Hauptfamilien | separate k1-k3-Familie |

Analyseskripte und Manifeste halten beide Bloecke getrennt. Die 36 k1-k3-Paare sind ueber dieselben 1.032 Case-IDs gepaart; die Holm-Familie `K1_VS_K3_DEMONSTRATION_COUNT_FAMILY` veraendert keine Hauptfamilie. Zulaessig ist: „Insgesamt wurden 84 autoritative Laeufe ausgewertet.“ Nicht zulaessig ist: „Es wurde eine vollstaendig homogene Matrix aus 84 Laeufen ausgewertet.“

## 9. Daten- und Retrievalabgrenzung

Die Trainingsmischung umfasst 25.000 Faelle: 6.960 Spider-Train-Faelle und 18.040 SQL-Create-Context-Faelle. MixedVal2500-v2 umfasst 700 `train_others`- und 1.800 SQL-Create-Context-Faelle. Spider Dev wurde weder fuer Training noch fuer die Checkpointauswahl verwendet.

Fuer Train-Validation, Train-Dev und Validation-Dev wurden direkte ID-, Frage-, SQL- und Frage-SQL-Paarueberschneidungen mit Ergebnis null dokumentiert. Der autoritative Retrievalpool umfasst ausschliesslich 6.960 eindeutige Spider-Train-Beispiele; seine direkten ID-, Frage-, SQL- und Paarueberschneidungen mit Spider Dev sind ebenfalls null. Der Index nutzt `BAAI/bge-large-en-v1.5` und `IndexFlatIP`; Index, Metadata und Manifest sind gehasht.

Diese Nachweise gelten fuer kontrollierte Projektartefakte. Sie schliessen unbekannte Vortrainingskontamination oder semantische/paraphrastische Ueberschneidungen nicht aus. Die allgemeine Validitaetseinordnung gehoert in Abschnitt 5.4.4.

## 10. Hashes, Manifeste und Dateiintegritaet

| Komponente | Nachweis | Status |
|---|---|---|
| Modellrevisionen | Llama direkt; Qwen ueber Cache-, Config- und Audittriangulation | `IMPLEMENTED_WITH_WARNINGS` |
| drei Adapter | Gewicht-, Config-, Metadata- und Trainer-State-Hashes | `IMPLEMENTED` |
| Trainings-/Validierungsdaten | vier Serialisierungsdateien mit SHA256 | `IMPLEMENTED` |
| Testset | Pfad, 1.032 IDs und SHA256 | `IMPLEMENTED` |
| Retrievalindex | Index-, Metadata- und Manifest-SHA256 | `IMPLEMENTED` |
| Evaluationsconfigs | 84/84 aktuelle Hashmatches | `IMPLEMENTED` |
| CSV/Metadata/Trace | 84/84, 84/84, 78/78 aktuelle Hashmatches | `IMPLEMENTED` |
| Runner | aktuelle/ausgefuehrte Hashes fuer 57 Runs direkt; 27 trianguliert | `IMPLEMENTED_WITH_WARNINGS` |
| Statistik-/Fehleranalysen | versionierte Skripte und Manifeste | `IMPLEMENTED` |
| rekursives Projektmanifest | historischer 16.07.-Basisscan, kein aktueller finaler Scan | `PARTIALLY_IMPLEMENTED` |

Es existiert **kein aktuelles vollstaendiges rekursives SHA256-Manifest des finalen Projektstands**. Die vielen fachlichen Teilmanifeste duerfen nicht mit einem vollstaendigen Projektfreeze verwechselt werden.

## 11. Git- und Versionsstatus

Die Projektwurzel enthaelt ein leeres Verzeichnis `.git` mit Modus 555. Es enthaelt weder `HEAD`, Config, Objekte noch Refs. `git status` meldet „not a git repository“, und es wurden keine weiteren `.git`-Unterverzeichnisse gefunden. Es bestehen keine Commit-Historie, Tags oder Releases.

Der Gitstatus lautet daher **NEIN**, nicht „vorhanden, aber schreibgeschuetzt“. Skriptstaende muessen ueber Dateihashes, Laufmetadaten und Manifeste rekonstruiert werden. In diesem Audit wurde Git nicht initialisiert.

## 12. Software- und Hardwareprovenienz

`.venv_flash` ist als kuenftige autoritative Freeze-Umgebung freigegeben. Der GPU-Host-Nachweis dokumentiert:

| Komponente | Version/Status | Evidenzklasse |
|---|---|---|
| Python | 3.11.15 | autoritativer Environmentnachweis |
| PyTorch / CUDA | 2.12.1+cu130 / 13.0 | Trainingsfingerprint und GPU-Host |
| GPU | NVIDIA L40S | Trainingsfingerprint und GPU-Host |
| Transformers | 5.9.0 | Environmentaudit |
| PEFT / TRL / Accelerate | 0.18.1 / 1.4.0 / 1.12.0 | Environmentaudit |
| Sentence Transformers / FAISS | 5.2.2 / 1.14.3 | Environmentaudit |
| Flash Attention | 2.8.3.post1 | GPU-Host und Training |
| Betriebssystem aktuell | Amazon Linux 2023.11.20260505 | `CURRENT_ENVIRONMENT_ONLY` |
| Kernel aktuell | 6.1.168-203.330.amzn2023.x86_64 | `CURRENT_ENVIRONMENT_ONLY` |

Historisch sind alle drei Trainingslinien `.venv_flash` mit EENV-2/HIGH zugeordnet. Fuer zwei der 48 Hauptlaeufe besteht starke `.venv_flash`-Startsequenzevidenz; fuer 46 wurde der Prozessinterpreter nicht direkt persistiert. Es gibt keine positive Evidenz fuer `.venv` oder System-Python in einem autoritativen Evaluationsrun. Primaere Endanalysen liefen nachweislich in `.venv_flash`; drei reine Standardbibliothek-Hilfsschritte liefen dokumentiert mit System-Python 3.9.

## 13. Speicherung und Nachpruefbarkeit

Je autoritativem Run stehen Fall-CSV und Metadaten/Summary bereit. Gespeichert sind Raw Output, extrahierte SQL, Ausfuehrungsstatus und Fehler, EMA/ESR-Felder, Textmetriken, Prompt-/Completiontokens und Generationszeit. Retrievalruns besitzen zusaetzlich ausgewählte Demo-IDs, Scores, Gateentscheidung und weitere Tracefelder.

Nicht vollstaendig persistiert sind:

- abgerufene SQLite-Ergebniszeilen und Spaltenmetadaten,
- Runtime-Runner-Hashes in 27 aelteren Hauptlaeufen,
- die vollstaendige Top-10-Reranking-Zwischentabelle in finalen Traces,
- `actual_k` als direktes Feld in einzelnen finalen Gate-Traces; es ist dort aus der Gateentscheidung ableitbar,
- separate Trainingskonsolenlogs.

Rein statisch reproduzierbar sind Fallintegritaet, gespeicherte Metrikaggregation, Summaries, gepaarte Statistik, Gateverteilungen, ausgewaehlte Retrievalidentitaet und grosse Teile der Fehlertransitionen. Eine unabhaengige EMA-/ESR-Neubewertung benoetigt read-only SQLite-Ausfuehrung, weil Ergebniszeilen nicht gespeichert wurden. Einzelne SQLite-aware Fehlerlabels erfordern ebenfalls diagnostische read-only SQL-Ausfuehrung. Neue Rohvorhersagen wuerden Modellinferenz erfordern; eine unabhaengige vollstaendige Retrievalrangliste wuerde Retrieval ausfuehren. Nichts davon wurde in diesem Audit gestartet.

## 14. Fehlgeschlagene Runs und Wiederholungsregeln

Die nachgewiesene Projektregel lautet:

1. Unvollstaendige Runs gehen nicht in autoritative Ergebnisanalysen ein.
2. Teilresultate werden nicht mit spaeteren Fortsetzungen kombiniert.
3. Nach einem methodischen Fix wird nur der erforderliche vollstaendige Run neu ausgefuehrt.
4. Bereits vollstaendige Runs werden ohne methodischen Grund nicht wiederholt.
5. Historische Artefakte bleiben mit eigenen Hashes unveraendert erhalten.
6. Ersatzruns erhalten eine neue Run-ID, eigene Metadaten und kollisionsfreie Outputs.
7. Das autoritative Inventar weist genau einen Run je Bedingung aus.

Diese Regeln wurden beim Qwen-9B-k3-Timeout- und Validatorproblem eingehalten: Die beiden Teilruns blieben ausgeschlossen, genau ein neuer Vollrun wurde erzeugt, und die Hashes der vorherigen 35 Vollruns blieben unveraendert.

## 15. Status des finalen Projektfreezes

| Massnahme | Physische Evidenz | Status | Noch erforderlich |
|---|---|---|---|
| autoritative Runliste | 84er-Runmatrix und k3-Inventar | `IMPLEMENTED` | in finalen Snapshot aufnehmen |
| Environmentfreigabe | GPU-/CUDA-/FlashAttention-/pip-check-Addendum | `IMPLEMENTED` | Paketliste beilegen |
| historisches Dateiinventar | 94.212 gehashte Dateien vom 16.07. | `PARTIALLY_IMPLEMENTED` | aktuellen finalen Scope neu hashen |
| aktuelles rekursives Manifest | nicht gefunden | `PLANNED_NOT_IMPLEMENTED` | nach Abschluss aller Thesisartefakte erzeugen |
| datierter Gesamtsnapshot | nicht gefunden | `PLANNED_NOT_IMPLEMENTED` | ausserhalb des Quellbaums erzeugen |
| finales Archiv plus Hash | nicht gefunden | `PLANNED_NOT_IMPLEMENTED` | Snapshot archivieren und Gesamt-SHA256 bilden |
| historisches Same-Host-Archiv | 12.05.2026, 925.518.767 Byte, 3.185 Eintraege, lesbar | `PARTIALLY_IMPLEMENTED` | nicht als finalen Freeze verwenden |
| Schreibschutz | 94.707/94.707 Quelldateien vor diesem Audit beschreibbar | `PLANNED_NOT_IMPLEMENTED` | erst nach erfolgreicher Verifikation setzen |
| externe Backups | keine unabhaengige Kopie physisch nachgewiesen | `PLANNED_NOT_IMPLEMENTED` | mindestens zwei Ziele |
| Transferhashpruefung | kein Zielprotokoll gefunden | `PLANNED_NOT_IMPLEMENTED` | Hash am Ziel neu berechnen |
| Restore-Test | kein Protokoll gefunden | `PLANNED_NOT_IMPLEMENTED` | dokumentierten Wiederherstellungstest durchfuehren |
| finale Thesis-PDF | noch nicht Teil dieses Scopes | `PLANNED_NOT_IMPLEMENTED` | PDF und Hash in Freeze aufnehmen |

Das Archiv `/home/ec2-user/nl2sql_testbench_full_backup.tar.gz` ist syntaktisch lesbar und besitzt den in diesem Audit berechneten SHA256 `8629586c66162e653c19750bda2afe83e78b7c315197d93f4c797995156a0b7f`. Es ist dennoch **kein finaler Backupnachweis**: Es liegt auf demselben Host, ist vom 12.05.2026 und enthaelt den spaeteren autoritativen Projektstand nicht. Die Zahl nachgewiesener externer Backups bleibt null.

## 16. Methodische Grenzen

1. Kein funktionsfaehiges Git-Repository und keine commitbasierte Historie.
2. 27 aeltere Hauptlaeufe ohne direkt persistierten Runtime-Runner-Hash.
3. Qwen-Snapshot- und historische Interpreterprovenienz teilweise indirekt.
4. Keine separaten Konsolenlogs der drei autoritativen Trainingslaeufe.
5. Ein Trainingsseed je Modelllinie; keine Seedvarianz der Adapterleistung.
6. Greedy Decoding ohne Samplingvarianz, aber auch ohne explizite Backend-Bitdeterminismusgarantie.
7. Gemischte SQL-Timeoutpolicy innerhalb der k3-Erweiterung.
8. Hoeheres Eingabelimit der k3-Erweiterung.
9. SQLite-Ergebniszeilen und Spaltenmetadaten nicht persistiert.
10. Unvollstaendige Reranking-Zwischenprotokollierung in finalen Traces.
11. Unbekannte Vortrainingskontamination nicht kontrollierbar.
12. Finaler Gesamtsnapshot, aktuelles Manifest, Schreibschutz, Backups und Restore-Test noch offen.

Diese Grenzen belegen nicht automatisch eine Ergebnisverfaelschung. Sie begrenzen vor allem die bitgenaue historische Wiederholung, die Homogenitaet der Erweiterungsmatrix und den aktuellen Schutz des Projektstands gegen Verlust oder spaetere Veraenderung.

## 17. Thesisfertige Formulierungsgrundlage

Die Durchfuehrung der Untersuchung wurde weitgehend konfigurationsgesteuert dokumentiert. Fuer die drei Fine-Tuning-Linien sowie fuer alle Evaluationsbedingungen liegen persistierte Konfigurationen vor; zentrale Datensaetze, Adapter, Testfaelle, Retrievalartefakte, Vorhersagen und Metadaten wurden durch kryptografische Hashwerte identifiziert. Als autoritativ galten nur Evaluationslaeufe, die alle 1.032 vorgesehenen Faelle in identischer Reihenfolge enthielten, keine fehlenden oder doppelten Fallkennungen aufwiesen, eindeutig einer Modell-, Adapter- und Konfigurationsversion zugeordnet werden konnten und konsistente Metadaten sowie, soweit erforderlich, vollstaendige Retrievaltraces besassen.

Die 48 Laeufe der Hauptuntersuchung und die 36 Laeufe der spaeteren k3-Erweiterung erfuellen diese Integritaetskriterien. Fehlerhafte oder unvollstaendige Laeufe wurden nicht mit spaeteren Fortsetzungen zusammengefuehrt, sondern unveraendert als historische Artefakte erhalten und aus den autoritativen Inventaren ausgeschlossen. Nach methodischen Korrekturen wurde nur der jeweils erforderliche Vollrun unter einer neuen Identitaet ausgefuehrt. Fehlerhafte SQL-Ausfuehrungen, leere Extraktionen und Timeouts blieben als negative Fallresultate im gemeinsamen Nenner erhalten.

Im Projektverlauf mussten mehrere methodische Probleme kontrolliert werden. Dazu gehoerten ein historischer Prompt- und Assistant-Prefix-Mismatch bei Qwen, die Koexistenz eines veralteten Teiltestsets, ungeeignete historische Sequenzlaengen sowie eine pathologisch lange SQL-Ausfuehrung in der k3-Erweiterung. Die finale Promptlinie wurde zwischen Training und Evaluation vereinheitlicht, die Testsetidentitaet ueber Fallzahl und Hashwert abgesichert und die k3-Promptlaenge vor der Ausfuehrung vollstaendig vorgeprueft. Fuer die Qwen-9B-k3-Laeufe wurde ein fallweise abgefangener Statement-Timeout eingefuehrt; ein anschliessender Validatorpfadfehler wurde durch eine isolierte Pfadkorrektur behoben. Die daraus resultierenden Einschraenkungen der internen und externen Validitaet werden in Abschnitt 5.4.4 eingeordnet.

Die Hauptuntersuchung und die k3-Erweiterung werden getrennt ausgewiesen. Sie unterscheiden sich in der Demonstrationszahl, dem zulaessigen Eingabebudget und teilweise in der SQL-Timeoutpolicy. Entsprechend wurden auch die statistischen Vergleichsfamilien getrennt gehalten. Insgesamt wurden 84 autoritative Laeufe ausgewertet, jedoch keine vollstaendig homogene 84-Run-Matrix.

Die historische Provenienz ist fuer die wissenschaftliche Zuordnung der Ergebnisse ausreichend, weist aber dokumentierte Grenzen auf. Insbesondere wurden bei einem Teil der aelteren Evaluationen die damals ausgefuehrte Runnerdatei und der Prozessinterpreter nicht direkt persistiert; fuer zwei Modelllinien musste die verwendete Modellrevision aus konsistenten Cache-, Config- und Auditnachweisen rekonstruiert werden. Die autoritative Python-Umgebung ist technisch freigegeben. Ein vollstaendiger finaler Dateisystemfreeze mit aktuellem Rekursivmanifest, schreibgeschuetztem Archiv, unabhaengigen Backups und Wiederherstellungstest ist dagegen noch nicht umgesetzt und darf erst nach physischer Verifikation als abgeschlossen bezeichnet werden.

Technische Details zu Dateien, Verzeichnissen, Bibliotheken und Ausfuehrungsbefehlen sollten in Kapitel 6 beziehungsweise im Reproduzierbarkeitsanhang dokumentiert werden.

## 18. Evidenzmatrix

| Aussage | Technischer Befund | Evidenzdatei | Status | Fuer 5.7 geeignet |
|---|---|---|---|---:|
| 48 Hauptlaeufe vollstaendig | 48 x 1.032, keine fehlenden/duplizierten IDs | `generation_evaluation_authoritative_run_matrix_20260718.csv` | `PROJECT-VERIFIED` | ja |
| 36 k3-Laeufe vollstaendig | 36 x 1.032 | `k3_all_runs_completion_inventory_after_repair_20260718.csv` | `AUDIT-VERIFIED` | ja |
| 84 Summaries konsistent | 84/84 | `audit_method_generation_evaluation_pipeline_20260718.md` | `AUDIT-VERIFIED` | ja |
| 78 erforderliche Traces vollstaendig | 80.496/80.496 Zeilen | `method_generation_evaluation_pipeline_20260718.json` | `AUDIT-VERIFIED` | ja |
| Testsethash bestaetigt | SHA256 `6ce959...cce` | Testset und 84er-Runmatrix | `PROJECT-VERIFIED` | ja |
| identische Fallreihenfolge | 84/84 | 84er-Runmatrix | `AUDIT-VERIFIED` | ja |
| keine fehlenden/doppelten Faelle | 0/0 | 84er-Runmatrix | `AUDIT-VERIFIED` | ja |
| autoritative Adapter bestaetigt | 3/3, Hashes und Checkpointauswahl | `method_lora_v2_finetuning_20260718.json` | `AUDIT-VERIFIED` | ja |
| Trainings-/Validierungsdaten bestaetigt | 25.000/2.500 je Linie; Hashes vorhanden | LoRA-Trainingsmatrix | `AUDIT-VERIFIED` | ja |
| Retrievalpool bestaetigt | 6.960, Index/Metadata/Manifest gehasht | Retrievalmethodenaudit | `AUDIT-VERIFIED` | ja |
| Configs vorhanden | 3/3 Training, 84/84 Evaluation | dieses Audit | `PROJECT-VERIFIED` | ja |
| historische Runs ausgeschlossen | v1, 1024, Ablationen, Smoke, Teilruns | Trainingsinventar und k3-Audit | `AUDIT-VERIFIED` | ja |
| Teilruns nicht kombiniert | zwei Qwen9-Teilruns unveraendert | k3-Post-Repair-Audit | `AUDIT-VERIFIED` | ja |
| nach Validatorfix genau ein Run | Qwen9 Base Top3 v3 | k3-Post-Repair-Audit | `AUDIT-VERIFIED` | ja |
| gemischte Timeoutpolicy | 24 k3 ohne, 12 Qwen9 mit 900 s | Generationaudit | `AUDIT-VERIFIED` | ja |
| aeltere Runnerhashes teilweise nicht persistiert | 27/84 fehlen | 84er-Runmatrix | `AUDIT-VERIFIED` | ja |
| Gitstatus | leeres `.git`, kein Repository | physische Pruefung | `PROJECT-VERIFIED` | ja |
| Snapshotstatus | kein finaler Projektsnapshot | physische Pruefung und Freeze-Plan | `PLANNED_NOT_IMPLEMENTED` | ja |
| Manifeststatus | historischer Basisscan, kein aktuelles finales Manifest | Inventar und physische Pruefung | `PARTIALLY_IMPLEMENTED` | ja |
| Archivstatus | nur historisches Same-Host-Archiv | physische Pruefung | `PARTIALLY_IMPLEMENTED` | ja |
| Backupstatus | null unabhaengige externe Backups nachgewiesen | physische Pruefung | `PLANNED_NOT_IMPLEMENTED` | ja |
| Restore-Teststatus | kein Protokoll | Freeze-Checkliste und physische Pruefung | `PLANNED_NOT_IMPLEMENTED` | ja |

## 19. Offene Punkte

Zwingend offen fuer den **operativen Freeze**, nicht fuer einen weiteren Modellrun:

1. alle finalen Thesis- und Auditdateien abschliessen,
2. einen datierten vollstaendigen Snapshot ausserhalb des Quellbaums erzeugen,
3. ein aktuelles rekursives SHA256-Manifest einschliesslich Symlinkzielen erzeugen,
4. das Archiv hashen und nach Verifikation schreibschuetzen,
5. mindestens zwei unabhaengige externe Kopien erstellen und am Ziel hashen,
6. einen dokumentierten Restore-Test durchfuehren,
7. die finale Thesis-PDF mit Hash und Abgabezeitpunkt ergaenzen.

Nicht offen sind fehlende Trainings- oder Evaluationsruns. Es ist kein Rerun erforderlich.

## 20. Abschlussblock

```text
Git-Repository vorhanden: NEIN
Vollstaendiger Dateisystemsnapshot vorhanden: NEIN
Vollstaendiges aktuelles rekursives SHA256-Manifest vorhanden: NEIN
Schreibgeschuetztes Gesamtarchiv vorhanden: NEIN
Externe Backups nachgewiesen: 0
Wiederherstellungstest nachgewiesen: NEIN
Autoritative Trainingsconfigs vollstaendig: JA, 3/3
Autoritative Evaluationsconfigs vollstaendig: JA, 84/84
Modellrevisionen vollstaendig direkt persistiert: NEIN
Adapterhashes vorhanden: JA, 3/3
Trainingsdatasethashes vorhanden: JA
Validierungsdatasethashes vorhanden: JA
Testsethash vorhanden: JA
Retrievalindexhash beziehungsweise Manifest vorhanden: JA
Hauptlaeufe vollstaendig: 48/48
k3-Laeufe vollstaendig: 36/36
Konsistente Summaries: 84/84
Vollstaendige erforderliche Retrievaltraces: 78/78
Fehlende Fallzeilen: 0
Doppelte Fallzeilen: 0
Prompttruncationen in autoritativen Runs: 0
Unvollstaendige historische Runs ausgeschlossen: JA
Teilruns kombiniert: NEIN
Nach Validatorfix neu ausgefuehrte Runs: 1
Andere Runs dabei wiederholt: NEIN
Haupt- und k3-Matrix getrennt behandelt: JA
Statistische Familien getrennt: JA
Gemischte Timeoutpolicy dokumentiert: JA
Aeltere Runtime-Runner-Hashes vollstaendig persistiert: NEIN, 27 fehlen
Historische Trainingsumgebung direkt belegt: teilweise
Separate Trainingskonsolenlogs vorhanden: 0/3
Verbleibende Reproduzierbarkeitsluecken: dokumentiert
Nur geplante, noch nicht umgesetzte Massnahmen: aktueller Snapshot, aktuelles Rekursivmanifest, finales Archiv, Schreibschutz, zwei externe Backups, Transferhashpruefung, Restore-Test, finaler PDF-Hash
Geaenderte Bestandsdateien: keine
Training gestartet: nein
Evaluation gestartet: nein
Modellinferenz gestartet: nein
Retrieval gestartet: nein
SQL-Ausfuehrung gestartet: nein
Snapshot erstellt: nein
Backup erstellt: nein
Git initialisiert: nein

REPRODUCIBILITY-METHOD-AUDIT: PASS MIT WARNUNGEN
AUTHORITATIVE-RUN-PROVENANCE: PASS MIT WARNUNGEN
PROJECT-FREEZE-STATUS: PARTIAL
REPRODUCIBILITY-GAPS: DOCUMENTED
THESIS-SECTION-5.7: READY_WITH_LIMITATIONS
```

## 21. Erzeugte Artefakte und SHA256

| Artefakt | SHA256 |
|---|---|
| `audits/derived/method_challenges_reproducibility_20260719.json` | `b97458b808693ce01db49172fa805baacd58899ac900e00adccbb7adad3c5eb4` |
| `audits/derived/reproducibility_artifact_inventory_20260719.csv` | `59f538bf55212c2e28f7aeb1022fd11c80c389aad088db0123891078b3b92749` |
| `audits/derived/reproducibility_measures_status_20260719.csv` | `43883a1baef4aa5f47b3b43f02afa08b4178b0af369f49bae355cacf58029f67` |
| `audits/derived/authoritative_run_acceptance_rules_20260719.csv` | `ce200a91f4d6ba8e1f5edb17350c6a3c813e90d44eeeddf237b6c43d6c796350` |
| `audits/audit_method_challenges_reproducibility_20260719.md` | siehe externe Abschlusspruefung; Selbsthash nicht in die gehashte Datei eingebettet |
