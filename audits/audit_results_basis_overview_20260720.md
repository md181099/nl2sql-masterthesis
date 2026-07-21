# Ergebnisaudit: Ueberblick ueber die Ergebnisbasis

Datum: 20.07.2026  
Projektwurzel: `/home/ec2-user/nl2sql_testbench`  
Zweck: belastbare Grundlage fuer Kapitel 7.1  
Arbeitsmodus: Bestandsartefakte ausschliesslich read-only geprueft; keine Evaluation, Inferenz, SQL-Ausfuehrung oder neue Ergebnisanalyse gestartet.

## 1. Executive Summary

Die autoritative Ergebnisbasis umfasst drei Modelllinien mit jeweils einer Ausgangs- und einer LoRA-v2-Rolle:

1. Qwen 3.5 2B,
2. Llama 3.2 3B Instruct,
3. Qwen 3.5 9B.

Damit liegen sechs Modellrollen vor. Die urspruengliche Hauptuntersuchung bildet eine vollstaendige Matrix aus sechs Rollen und acht Bedingungen, also 48 Runs. Die spaeter hinzugefuegte k3-Erweiterung umfasst dieselben sechs Rollen und sechs dynamische Bedingungen, also weitere 36 Runs. Insgesamt sind 84 autoritative Runs mit jeweils 1.032 Spider-Dev-Faellen vorhanden. Dies entspricht 86.688 Fallzeilen.

Alle 84 Runs besitzen eine Ergebnis-CSV und eine konsistente Metadata-Summary. Fuer die 78 retrievalbasierten Runs liegen vollstaendige Retrievaltraces mit insgesamt 80.496 Tracezeilen vor. Die sechs Zero-Shot-Runs benoetigen methodisch keinen Retrievaltrace. Fehlende oder doppelte Fall-IDs wurden in den autoritativen Runs nicht festgestellt.

Die beiden Versuchsblocke sind getrennt zu berichten. Die 48-Run-Hauptmatrix bleibt die `PRE-K3 AUTHORITATIVE BASELINE`; die 36 k3-Runs sind eine additive Erweiterungsanalyse. Zusammen duerfen sie als 84 ausgewertete autoritative Runs bezeichnet werden, nicht jedoch als vollstaendig homogene 84-Run-Matrix. Die k3-Erweiterung verwendet ein hoeheres Eingabelimit und besitzt fuer die zwoelf Qwen-9B-v3-Runs eine abweichende Statement-Timeoutpolicy. Diese Unterschiede betreffen die Vergleichbarkeit, nicht die Vollstaendigkeit der gespeicherten Ergebnisbasis.

Dieser Audit beschreibt ausschliesslich Umfang, Struktur, Felder und Freigabestatus. Konkrete Leistungswerte, Signifikanztests und inhaltliche Ergebnisinterpretationen sind nicht Gegenstand von Kapitel 7.1 und werden hier nicht berichtet.

## 2. Autoritative Inventare und Manifeste

| Pfad | SHA256 | Rolle | Status |
|---|---|---|---|
| `audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv` | `7d6107d942ea0f2890354cb426eebf929597dcb36f5c5108df1abce0b2c921dc` | gemeinsames Register aller 48+36 autoritativen Runs mit Config-, Ergebnis-, Metadata- und Tracezuordnung | `AUTHORITATIVE` |
| `audits/cross_model_qwen2b_llama3b_qwen9b_complete_synthesis_manifest_20260716.json` | `24b4dec07d2d4981b42ce22e1295d27b0ccd9cbcc10666a422118b267fd14e37` | Abschlussmanifest der 48-Run-Hauptmatrix | `AUTHORITATIVE_WITH_PATH_WARNING` |
| `audits/derived/cross_model_complete_48_run_results_20260716.csv` | `f051867b5d7ce599d4e8a3a9ffb45448c9d9e715a106fee0696199ea94c7f7d2` | deskriptives Ergebnisregister der Hauptmatrix | `AUTHORITATIVE` |
| `audits/derived/k3_all_runs_completion_inventory_after_repair_20260718.csv` | `adf17bd367049ba86a66b649f474e8d6744af9279e0961605bf668643ebe87e3` | Abschlussinventar der 36 k3-Runs | `AUTHORITATIVE` |
| `audits/audit_k3_matrix_completion_after_qwen9b_base_top3_v3_20260718.md` | `748965e7ced25ddced2cf49dc60765c2536d7f2d584737ed823cc872e02a1a35` | k3-Vollstaendigkeits-, Timeout- und Ausschlussaudit | `AUTHORITATIVE` |
| `audits/derived/k3_authoritative_run_metrics_20260718.csv` | `e6f74ac6b6430e3264047887245211e5361b7f2b30bbc12777da148dd4754190` | deskriptives Ergebnisregister der k3-Erweiterung | `AUTHORITATIVE` |
| `audits/derived/k3_final_analysis_manifest_20260718.json` | `7a23ddd2e16def54a54d53eaede7527eabf75722b3ad8916a41cd963581e97e3` | Provenienz der k3-Ergebnis-, Paar-, Effizienz- und Fehleranalyseartefakte | `AUTHORITATIVE` |
| `audits/derived/method_generation_evaluation_pipeline_20260718.json` | `c83ff22ecd2c907bcc160a1402527f3e69329a67fd0dd989d8265db3873bb9e2` | maschinenlesbarer Integritaets- und Feldnachweis fuer alle 84 Runs | `AUTHORITATIVE_SUPPORTING` |
| `audits/audit_method_generation_evaluation_pipeline_20260718.md` | `e1634980d41470fbc18a1ba4759aec10ac1b73198724b05a1d6f898da9a085bc` | Rekonstruktion der Ergebnisfelder, Summaries und Tracevollstaendigkeit | `AUTHORITATIVE_AUDIT` |

Die 84er-Runmatrix ist das primaere Register fuer die Aufnahme in die Ergebnisbasis. Dateinamen allein entscheiden nicht ueber Autoritaet. Jeder Registereintrag bindet Modelllinie, Rolle und Bedingung an eine konkrete Config, Ergebnis-CSV, Metadata-Summary und gegebenenfalls einen Retrievaltrace samt SHA256 und Integritaetsstatus.

## 3. Umfang der Ergebnisbasis

| Versuchsblock | Modellrollen | Bedingungen je Rolle | Runs | Faelle je Run | Fallzeilen |
|---|---:|---:|---:|---:|---:|
| 48-Run-Hauptmatrix | 6 | 8 | 48 | 1.032 | 49.536 |
| additive k3-Erweiterung | 6 | 6 | 36 | 1.032 | 37.152 |
| **Gesamt** | **6** | **8 plus 6 blockbezogen** | **84** | **1.032** | **86.688** |

Die sechs Modellrollen sind:

| Modelllinie | Ausgangsrolle | LoRA-Rolle | Runs je Rolle |
|---|---|---|---:|
| Qwen 3.5 2B | Base | LoRA v2 | 8 Haupt- plus 6 k3-Runs |
| Llama 3.2 3B Instruct | Instruct-Ausgangsmodell | LoRA v2 | 8 Haupt- plus 6 k3-Runs |
| Qwen 3.5 9B | Base | LoRA v2 | 8 Haupt- plus 6 k3-Runs |

Die Bezeichnung „Base“ ist fuer Qwen woertlich passend. Bei Llama bezeichnet die Ausgangsrolle das nicht projektspezifisch feinabgestimmte **Instruct-Ausgangsmodell**, nicht ein Pretraining-Base-Modell.

## 4. Bedingungen der beiden Versuchsblocke

| Nr. | 48-Run-Hauptmatrix | Demonstrationen/Fallback | 36-Run-k3-Erweiterung | Demonstrationen/Fallback |
|---:|---|---|---|---|
| 1 | Zero Shot | k=0 | nicht separat enthalten | - |
| 2 | Static Few Shot | statisch k=1, Seed 42 | nicht separat enthalten | - |
| 3 | Dynamic Top-1 | dynamisch k=1 | Dynamic Top-3 | dynamisch k=3 |
| 4 | Dynamic Top-1 Gate 0.70 | k=1 oder Zero-Shot-Fallback | Dynamic Top-3 Gate 0.70 | gesamtes Set k=3 oder Zero-Shot-Fallback |
| 5 | Dynamic Top-1 Gate 0.85 | k=1 oder Zero-Shot-Fallback | Dynamic Top-3 Gate 0.85 | gesamtes Set k=3 oder Zero-Shot-Fallback |
| 6 | Structure Top-1 | Structure-Reranking, k=1 | Structure Top-3 | Structure-Reranking, k=3 |
| 7 | Structure Top-1 Gate 0.70 | k=1 oder Zero-Shot-Fallback | Structure Top-3 Gate 0.70 | gesamtes Set k=3 oder Zero-Shot-Fallback |
| 8 | Structure Top-1 Gate 0.85 | k=1 oder Zero-Shot-Fallback | Structure Top-3 Gate 0.85 | gesamtes Set k=3 oder Zero-Shot-Fallback |

Die Hauptmatrix enthaelt jede ihrer acht Bedingungen genau sechsmal, einmal je Modellrolle. Die k3-Erweiterung enthaelt jede ihrer sechs Bedingungen ebenfalls genau sechsmal. Zero Shot und Static Few Shot wurden fuer k3 nicht erneut ausgefuehrt; sie bleiben Bestandteil der 48-Run-Hauptmatrix.

Gatebedingungen duerfen nicht pauschal als durchgaengige Few-Shot-Runs beschrieben werden. Ihre Fallartefakte enthalten je nach Gateentscheidung Few Shot oder den vorgesehenen Zero-Shot-Fallback. Die Ergebnisbasis speichert diese Entscheidung fallweise.

## 5. Vollstaendigkeit und Integritaetsstatus

| Pruefpunkt | Hauptmatrix | k3-Erweiterung | Gesamtstatus |
|---|---:|---:|---|
| autoritative Runs | 48/48 | 36/36 | 84/84 |
| Fallzeilen | 49.536/49.536 | 37.152/37.152 | 86.688/86.688 |
| eindeutige Case-IDs je Run | 1.032 | 1.032 | bestaetigt |
| fehlende Fall-IDs | 0 | 0 | 0 |
| doppelte Fall-IDs | 0 | 0 | 0 |
| Metadata-Summaries | 48/48 | 36/36 | 84/84 |
| konsistente Summaries | 48/48 | 36/36 | 84/84 |
| erforderliche Retrievaltraces | 42/42 | 36/36 | 78/78 |
| Retrievaltracezeilen | 43.344 | 37.152 | 80.496 |

In der gemeinsamen Runmatrix besitzen 72 Runs den Status `COMPLETE_VALID`. Die zwoelf Qwen-9B-k3-v3-Runs besitzen `COMPLETE_WITH_WARNING`, sind aber vollstaendig und autoritativ. Die Warnung dokumentiert die innerhalb der k3-Matrix gemischte SQL-Statement-Timeoutpolicy; sie bezeichnet keine fehlenden Faelle, Summaries oder Traces.

## 6. Verfuegbare Metriken und Artefakte

| Ebene | Verfuegbare Inhalte | Abdeckung/Einordnung |
|---|---|---|
| Fallidentitaet | `id`, `db_id`, Datenbankpfad, Frage und Referenz-SQL | 86.688 Fallzeilen |
| Modellausgabe | rohe Completion in `raw_output`, extrahierte Prediction-SQL | 84/84 Runs |
| Ausfuehrungsstatus | Gold-/Prediction-Ausfuehrbarkeit, Fehlertexte und `exec_match` | Grundlage fuer ESR und EMA |
| Primaermetrik | Execution Match Accuracy (EMA) | fallweises `exec_match`, je Run aggregiert |
| weitere Anteilsmetriken | ESR, String Exact Match, projektinternes Normalized Exact Match | Fallfelder und Summary |
| diagnostische Textmetriken | Char Accuracy und Token Accuracy | Fallfelder und Summary-Mittelwerte |
| Effizienz | Prompt-, Completion- und Gesamttokens; Generationszeit und Tokens pro Sekunde | Fallfelder und Summary; Laufzeit ist Generationszeit, nicht End-to-End-Fallzeit |
| Retrieval | Demo-IDs, Similarity-Scores, Retrievalmethode, Indexpfad und ausgewaehlte Beispiele | 78 Retrievalruns |
| Gates | Schwelle, Gate-Score, Entscheidung, Grund, Fallback und bei k3 Score-Setstatistiken | Gatebedingungen |
| Run-Summary | Fallzahl, aggregierte Metriken, Token-/Zeitwerte, Laufzeit und Provenienz | 84/84 konsistent |
| Retrievaltrace | fallweise Auswahl-, Score-, Reranking- und Gateprovenienz | 78/78 erforderlich |
| Analyseartefakte | getrennte Haupt-, k3-, k1-k3-, Effizienz- und Fehlerdateien | versioniert und manifestiert; in Kapitel 7.1 nur als Bestand nennen |

Die Ergebnisbasis enthaelt somit sowohl die primaere EMA als auch die vorgesehenen ergaenzenden Ausfuehrungs-, Text-, Effizienz- und Retrievalinformationen. Kapitel 7.1 sollte diese Verfuegbarkeit beschreiben, aber weder Rangfolgen noch Unterschiede zwischen Modellen oder Bedingungen vorwegnehmen.

## 7. Trennung der Versuchsblocke

| Merkmal | 48-Run-Hauptmatrix | 36-Run-k3-Erweiterung |
|---|---|---|
| Projektstatus | `PRE-K3 AUTHORITATIVE BASELINE` | additive Erweiterungsanalyse |
| Bedingungen | Zero, Static, sechs dynamische k1-Bedingungen | sechs dynamische k3-Bedingungen |
| regulaeres Eingabelimit | ueberwiegend 2.048; zwei nicht limitgebundene Zero-Shot-Runs mit 1.536 | 4.352 |
| Prompttruncationen | 0 | 0 |
| SQL-Statement-Timeout | kein explizites Statement-Timeout | 24 ohne; 12 Qwen-9B-v3 mit 900 Sekunden |
| statistische Einordnung | eingefrorene Hauptfamilien | separate k1-k3-Familie |

Zulaessig ist die Formulierung: „Insgesamt wurden 84 autoritative Runs ausgewertet.“ Nicht zulaessig ist ohne Einschraenkung: „Es wurde eine vollstaendig homogene Matrix aus 84 Runs ausgewertet.“

## 8. Ausgeschlossene historische und partielle Runs

Nicht zur autoritativen Ergebnisbasis gehoeren:

- Smoke-, Testmode-, Stichproben- und Teilmengenruns;
- Ergebnisse historischer LoRA-v1- oder nicht final ausgewaehlter Adaptervarianten;
- historische Prompt-/Chattemplate-Mismatch-Runs;
- Sensitivitaets- und Ablationsruns ausserhalb der 48+36-Matrizen;
- Runs auf `data/testcases.jsonl` mit 200 Faellen statt des autoritativen 1.032er-Bestands;
- Runs mit historischen Retrievalpools oder nicht freigegebenen Retrievalmodi;
- der unbounded Qwen-9B-Base-Top-3-Teilrun mit 479 gespeicherten Zeilen;
- der fehlerhafte v2-Timeout-Teilrun derselben Bedingung mit 482 gespeicherten Zeilen;
- weitere Ergebnisse, die nicht eindeutig in der autoritativen 84er-Runmatrix registriert sind.

Die beiden historischen k3-Teilruns und ihre Traces blieben unveraendert. Sie wurden weder fortgesetzt noch miteinander oder mit dem spaeteren Vollrun kombiniert. Nach dem isolierten Validatorfix wurde nur der fehlende vollstaendige Qwen-9B-Base-Top-3-v3-Run unter neuer Identitaet ausgefuehrt.

## 9. Grenzen fuer Kapitel 7.1

1. Die 84 Runs sind vollstaendig, aber nicht methodisch vollstaendig homogen.
2. Die zwoelf Qwen-9B-k3-v3-Runs besitzen eine andere Timeoutpolicy als die uebrigen Runs.
3. k1 und k3 verwenden unterschiedliche zulaessige Eingabelimits; ein separater Aequivalenzaudit bestaetigt fuer k1 zwar keine aktive Truncation oder Promptveraenderung.
4. Gatebedingungen enthalten fallweise Few Shot oder Zero-Shot-Fallback und muessen entsprechend bezeichnet werden.
5. Retrievalsimilarities sind Scores, keine kalibrierten Wahrscheinlichkeiten.
6. Tokenwerte verschiedener Modelltokenizer sind nicht als vollstaendig tokenizerunabhaengige Einheiten zu behandeln.
7. Historische Runnerprovenienz ist fuer 27 aeltere Hauptlaeufe teilweise trianguliert; dies betrifft die Reproduzierbarkeitsdokumentation, nicht die Vollstaendigkeit ihrer gespeicherten Ergebniszeilen.

## 10. Thesisfertige Formulierungsgrundlage fuer Kapitel 7.1

Die Ergebnisbasis umfasst drei Modelllinien, die jeweils als Ausgangsmodell und als LoRA-v2-adaptierte Variante untersucht wurden. Daraus ergeben sich sechs Modellrollen. Die autoritative Hauptuntersuchung kombiniert diese Rollen mit acht Prompting- und Retrievalbedingungen und umfasst 48 vollstaendige Runs. Neben Zero Shot und einer statischen One-Shot-Bedingung enthaelt sie sechs dynamische Bedingungen mit direkter beziehungsweise strukturorientierter Top-1-Auswahl und Similarity-Gates von 0,70 und 0,85.

Ergaenzend wurde eine separate k3-Analyse durchgefuehrt. Sie umfasst fuer dieselben sechs Modellrollen jeweils sechs dynamische Top-3- beziehungsweise Structure-Top-3-Bedingungen, wiederum ungefiltert sowie mit den beiden Gateschwellen. Diese Erweiterung steuert 36 weitere vollstaendige Runs bei. Insgesamt liegen damit 84 autoritative Runs mit jeweils 1.032 Spider-Dev-Faellen und insgesamt 86.688 fallweisen Ergebniszeilen vor. Jeder Run besitzt eine Metadata-Summary; fuer alle 78 retrievalbasierten Runs ist ausserdem ein vollstaendiger Retrievaltrace vorhanden.

Die Fallartefakte enthalten die rohe Modellausgabe, die extrahierte SQL-Abfrage, Ausfuehrungs- und Matchstatus, die primaere Execution Match Accuracy sowie ergaenzende Ausfuehrungs-, Text-, Token-, Laufzeit- und Retrievalfelder. Die 48-Run-Hauptmatrix und die nachtraegliche 36-Run-k3-Erweiterung werden in den folgenden Ergebnisabschnitten getrennt ausgewiesen. Sie bilden zusammen 84 ausgewertete Runs, jedoch aufgrund unterschiedlicher Demonstrationszahlen, Eingabelimits und der gemischten Timeoutpolicy keine vollstaendig homogene Versuchsmatrix.

## 11. Vorschlag fuer eine LaTeX-Tabelle

Empfohlen wird eine einzige kompakte Uebersicht mit den Spalten:

```text
Versuchsblock | Modelllinien | Rollen | Bedingungen je Rolle |
Runs | Faelle je Run | Fallzeilen | Summaries | Retrievaltraces
```

Die zwei Datenzeilen sollten `48-Run-Hauptmatrix` und `36-Run-k3-Erweiterung` lauten; eine dritte Summenzeile weist 84 Runs und 86.688 Fallzeilen aus. Die Bedingungsnamen koennen in einer Tabellenanmerkung oder unmittelbar danach als kompakte Liste stehen. Konkrete Metrikwerte und Signifikanzkennzeichnungen gehoeren nicht in diese Uebersicht.

## 12. Abschlussstatus

Drei Modelllinien bestaetigt: JA  
Sechs Modellrollen bestaetigt: JA  
48+36-Run-Struktur bestaetigt: JA  
84 Runs und 86.688 Fallzeilen bestaetigt: JA  
Summaries und erforderliche Traces vollstaendig: JA, 84/84 Summaries und 78/78 Traces  
Metrikfelder bestaetigt: JA  
Offene Punkte: keine Vollstaendigkeitsluecke; fuer die Darstellung bleiben Blocktrennung, unterschiedliche Eingabelimits, gemischte k3-Timeoutpolicy und Gate-Fallbacksemantik zu berichten  
Veraenderte Bestandsdateien: keine  
Evaluation gestartet: nein  
Inferenz gestartet: nein  
SQL-Ausfuehrung gestartet: nein  
Neue Analyse gestartet: nein

`RESULTS-BASIS-AUDIT: PASS MIT WARNUNGEN`

`THESIS-SECTION-7.1: READY_WITH_LIMITATIONS`
