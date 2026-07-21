# Read-only-Audit der Python-Umgebungen und Requirements vor dem Projekt-Freeze

Datum: 2026-07-21 (UTC)  
Projektwurzel: `/home/ec2-user/nl2sql_testbench`  
Auditstatus: **PASS MIT WARNUNGEN**

## 1. Kurzfazit

Die beiden Requirements-Dateien dokumentieren unterschiedliche Paketstaende:

- `requirements.txt` ist der aeltere, kuerzere Stand. Er ist fuer die heutige Projektsoftware unvollstaendig und enthaelt mehrere veraltete Pins.
- `requirements_current.txt` ist bytegleich zum aktuellen `pip freeze` von `.venv` (SHA256 jeweils `8b761073384117511baf4e560406dd2e471bed5b23ee30217612b8c17259f061`). Sie dokumentiert `.venv` daher exakt, aber **nicht** die autoritative GPU-/FlashAttention-Umgebung.
- `.venv_flash` ist durch den bestehenden Freeze-Release-Audit als autoritative Freeze-Umgebung festgelegt. Ihr aktueller `pip freeze` besitzt weiterhin exakt den bereits am 2026-07-16 protokollierten SHA256 `443575e5959f60e6abe508fb8d42749ed50ef5fdc19955984bfca8b336c627cd`.
- Keine der beiden vorhandenen Requirements-Dateien bildet `.venv_flash` vollstaendig ab. Gegenueber `requirements_current.txt` besitzt `.venv_flash` vier abweichende Versionsstaende und 20 zusaetzliche funktionale Pakete; insbesondere fehlen `flash-attn`, `einops` und der CUDA-13-Stack in der Datei.
- Beide Umgebungen bestehen `pip check`. Es wurden keine VCS-, lokalen Pfad- oder Editable-Installationen festgestellt.

**Entscheidung:** `.venv_flash` kann als projektseitig festgelegte autoritative Freeze-Umgebung bestaetigt werden. Diese Entscheidung ist jedoch nicht gleichbedeutend mit einem direkten Interpreter-Nachweis fuer jeden historischen Lauf: Alle 36 k3-Metadaten speichern `.venv_flash/bin/python` direkt; bei den drei Trainingslinien ist die Zuordnung insgesamt hoch plausibel, aber fuer beide Qwen-Trainings nur trianguliert; bei 46 der 48 historischen Hauptlaeufe fehlt ein direkt persistierter Prozessinterpreter. Diese Provenienzgrenzen bleiben bestehen.

Fuer den finalen Freeze sind beide vorhandenen Requirements-Dateien unveraendert als historische Artefakte zu erhalten. Zusaetzlich sollte ein datierter Freeze **aus `.venv_flash`** erzeugt und zusammen mit Interpreter-, pip-, CUDA-/GPU- und Paketmetadaten archiviert werden. Ein separater Snapshot von `.venv` ist sinnvoll, darf aber nicht als autoritativer GPU-Freeze bezeichnet werden.

## 2. Inventar

### 2.1 Requirements-Dateien

| Datei | Groesse | letzter Aenderungszeitpunkt (mtime, UTC) | SHA256 | Eintraege | Einordnung |
|---|---:|---|---|---:|---|
| `requirements.txt` | 983 Byte | 2026-06-10 12:12:52.374640796 | `ecf04a33712e400d435de3a0bd6079e8a82a6be00a803189c3e645bdb73b8afe` | 58 | aelterer, unvollstaendiger Stand |
| `requirements_current.txt` | 1.766 Byte | 2026-06-23 19:40:22.135041310 | `8b761073384117511baf4e560406dd2e471bed5b23ee30217612b8c17259f061` | 93 | exakter `.venv`-Freeze, nicht `.venv_flash` |

Die Zeitangaben sind Dateisystem-Metadaten. Sie belegen den letzten gespeicherten Aenderungszeitpunkt, nicht den Zeitpunkt einer bestimmten Installation oder eines Projektlaufs.

### 2.2 Virtuelle Umgebungen

| Merkmal | `.venv_flash` | `.venv` |
|---|---|---|
| Interpreter | `/home/ec2-user/nl2sql_testbench/.venv_flash/bin/python` | `/home/ec2-user/nl2sql_testbench/.venv/bin/python` |
| aufgeloestes Basisprogramm | `/usr/bin/python3.11` | `/usr/bin/python3.11` |
| Python | 3.11.15 | 3.11.15 |
| pip | 26.1.2 | 26.1.1 |
| `include-system-site-packages` | `false` | `false` |
| `pyvenv.cfg` mtime | 2026-06-23 19:39:19 UTC | 2026-05-12 07:49:29 UTC |
| `pip freeze`-Zeilen | 113 | 93 |
| `pip list`-Distributionen | 116 | 96 |
| `pip freeze` SHA256 | `443575e5959f60e6abe508fb8d42749ed50ef5fdc19955984bfca8b336c627cd` | `8b761073384117511baf4e560406dd2e471bed5b23ee30217612b8c17259f061` |
| `pip check` | `No broken requirements found.` | `No broken requirements found.` |

Die Differenz zwischen `pip list` und `pip freeze` besteht in beiden Umgebungen aus den Bootstrap-Paketen `pip`, `setuptools` und `wheel`, die `pip freeze` standardmaessig nicht ausgibt.

## 3. Vergleich `requirements.txt` und `requirements_current.txt`

### 3.1 Vollstaendiger Differenzbefund

Beide Dateien enthalten eindeutige Paketnamen. Es existieren keine doppelten Namen, keine widerspruechlichen Mehrfacheintraege und keine VCS-, URL-, lokalen Pfad- oder Editable-Eintraege.

Nur in `requirements.txt`:

- `setuptools==80.10.2`

Nur in `requirements_current.txt` (36 Pakete):

`annotated-doc`, `contourpy`, `cuda-bindings`, `cuda-pathfinder`, `cycler`, `et-xmlfile`, `faiss-cpu`, `fonttools`, `greenlet`, `kiwisolver`, `markdown-it-py`, `mdurl`, `ninja`, `nvidia-cublas-cu12`, `nvidia-cuda-cupti-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-cufile-cu12`, `nvidia-curand-cu12`, `nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusparselt-cu12`, `nvidia-nccl-cu12`, `nvidia-nvjitlink-cu12`, `nvidia-nvshmem-cu12`, `nvidia-nvtx-cu12`, `openpyxl`, `pillow`, `pygments`, `pyparsing`, `rich`, `triton`, `trl`, `typer`.

Gemeinsame Pakete mit unterschiedlicher Spezifikation:

| Paket | `requirements.txt` | `requirements_current.txt` |
|---|---|---|
| `datasets` | `==4.5.0` | `==4.8.5` |
| `hf-xet` | `==1.2.0` | `==1.5.0` |
| `huggingface-hub` | `==1.3.7` | `==1.14.0` |
| `matplotlib` | `>=3.9` | `==3.10.9` |
| `transformers` | `==5.0.0` | `==5.9.0` |

Alle anderen 52 gemeinsamen Eintraege sind spezifikationsgleich.

### 3.2 Bewertung

`requirements.txt` ist wahrscheinlich veraltet und fuer den finalen Stand unvollstaendig. Dies folgt nicht nur aus dem Datum, sondern aus konkreten technischen Abweichungen: Es fehlen unter anderem `faiss-cpu` fuer den FAISS-Index, `trl` fuer die autoritative Trainingspipeline und zahlreiche beim Projektstand installierte Abhaengigkeiten; zudem weichen zentrale Hugging-Face-Pins ab.

`requirements_current.txt` ist inhaltlich ein reproduzierbarer Snapshot von `.venv`. Der Dateihash ist exakt der Hash des aktuellen `.venv`-`pip freeze`. Der Name `current` ist deshalb missverstaendlich: Bezogen auf die autoritative `.venv_flash`-Umgebung ist die Datei weder vollstaendig noch versionsgleich.

## 4. Requirements gegen die installierten Umgebungen

### 4.1 Zusammenfassung

| Vergleich | fehlende geforderte Pakete | Versionsabweichungen | zusaetzlich installiert | Ergebnis |
|---|---:|---:|---:|---|
| `requirements.txt` gegen `.venv_flash` | 0 | 7 | 58 | nicht reproduktionsgenau |
| `requirements_current.txt` gegen `.venv_flash` | 0 | 4 | 23, davon 20 funktional | nicht reproduktionsgenau |
| `requirements.txt` gegen `.venv` | 0 | 4 | 38 | nicht reproduktionsgenau |
| `requirements_current.txt` gegen `.venv` | 0 | 0 | nur `pip`, `setuptools`, `wheel` | exakter `pip freeze` |

### 4.2 `requirements.txt` gegen `.venv_flash`

Versionsabweichungen:

- `datasets==4.5.0` -> installiert `4.8.5`
- `hf-xet==1.2.0` -> `1.5.0`
- `huggingface-hub==1.3.7` -> `1.14.0`
- `packaging==26.0` -> `26.2`
- `setuptools==80.10.2` -> `81.0.0`
- `torch==2.10.0` -> `2.12.1` (Runtime-Fingerprint: `2.12.1+cu130`)
- `transformers==5.0.0` -> `5.9.0`

Die 58 Extras umfassen neben `pip`/`wheel` insbesondere `faiss-cpu`, `flash-attn`, `einops`, `trl`, `triton`, `openpyxl`, `torchaudio`, `torchvision` sowie CUDA-12- und CUDA-13-Runtimepakete. Damit kann die Datei `.venv_flash` nicht rekonstruieren.

### 4.3 `requirements_current.txt` gegen `.venv_flash`

Versionsabweichungen:

- `cuda-bindings==12.9.4` -> installiert `13.3.1`
- `packaging==26.0` -> `26.2`
- `torch==2.10.0` -> `2.12.1`
- `triton==3.6.0` -> `3.7.1`

Zusaetzliche funktionale Distributionen in `.venv_flash`:

`cuda-toolkit`, `einops`, `flash-attn`, `nvidia-cublas`, `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`, `nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, `nvidia-nvtx`, `torchaudio`, `torchvision`.

Hinzu kommen die nicht von `pip freeze` erfassten Bootstrap-Pakete `pip`, `setuptools` und `wheel`. Die Umgebung enthaelt ausserdem weiterhin die in `requirements_current.txt` gepinnten CUDA-12-Distributionen; der Freeze muss diesen gemischten installierten Bestand exakt dokumentieren und darf ihn nicht manuell auf eine vermeintlich reinere CUDA-Liste reduzieren.

### 4.4 `.venv`

`requirements_current.txt` stimmt fuer alle 93 Freeze-Eintraege exakt mit `.venv` ueberein. `pip list` zeigt zusaetzlich nur `pip==26.1.1`, `setuptools==80.10.2` und `wheel==0.47.0`.

Gegen `requirements.txt` weichen in `.venv` `datasets`, `hf-xet`, `huggingface-hub` und `transformers` ab; 38 Distributionen sind zusaetzlich installiert. `requirements.txt` ist somit auch fuer `.venv` kein exakter Freeze.

## 5. Statische Importabdeckung

Die Python-Dateien unter `src/` und `scripts/` wurden per AST statisch auf Top-Level-Imports untersucht. Als externe Importfamilien wurden gefunden:

`datasets`, `faiss`, `flash_attn`, `huggingface_hub`, `kernels`, `matplotlib`, `numpy`, `peft`, `sentence_transformers`, `torch`, `transformers`, `trl`.

`src` ist ein projektlokaler Import und wurde nicht als externe Distribution gezaehlt.

| Import | `requirements.txt` | `requirements_current.txt` | `.venv_flash` importierbar | `.venv` importierbar | Einordnung |
|---|---|---|---|---|---|
| `datasets` | ja | ja | ja | ja | abgedeckt |
| `faiss` (`faiss-cpu`) | nein | ja | ja | ja | alte Datei unvollstaendig |
| `flash_attn` (`flash-attn`) | nein | nein | ja | nein | fuer Flash-Trainingspfad nur `.venv_flash` |
| `huggingface_hub` | ja | ja | ja | ja | abgedeckt |
| `kernels` | nein | nein | nein | nein | optionales Umgebungsdiagnosemodul; in beiden Umgebungen nicht vorhanden |
| `matplotlib` | ja | ja | ja | ja | abgedeckt |
| `numpy` | ja | ja | ja | ja | abgedeckt |
| `peft` | ja | ja | ja | ja | abgedeckt |
| `sentence_transformers` | ja | ja | ja | ja | abgedeckt |
| `torch` | ja | ja | ja | ja | Version umgebungsabhaengig |
| `transformers` | ja | ja | ja | ja | alte Datei falscher Pin |
| `trl` | nein | ja | ja | ja | alte Datei unvollstaendig |

Die statische Analyse belegt Importvorkommen, nicht, dass jeder optionale Codepfad in einem autoritativen Lauf ausgefuehrt wurde. Insbesondere ist `kernels` in einem Diagnosepfad referenziert und kein nachgewiesener Pflichtbestandteil der finalen Runs.

## 6. Evidenz zur tatsaechlich verwendeten Umgebung

### 6.1 Bestehende autoritative Umgebungsentscheidung

Zentrale vorhandene Evidenz:

| Artefakt | SHA256 | Aussage |
|---|---|---|
| `audits/python_environment_authoritative_freeze_release_manifest_20260716.json` | `1993b6dd444aa04d4d8dc9d5cdda487213f2043a75baa2cddc4e941d879f7837` | legt `.venv_flash` als Freeze-Umgebung fest |
| `audits/addendum_python_environment_authoritative_freeze_release_20260716.md` | `2bb74ba7172a2da7d474dc5e47e1f1b88d6cbcc108d547073571927bb5c4bbe8` | fachliche Freigabe mit Provenienzwarnungen |
| `audits/derived/venv_flash_gpu_host_freeze_readiness_authoritative_20260716.json` | `cfd850a19ef419df63597efc6c69c9f5e466e7ce1a0a1465c2d37ffb5ddf3453` | GPU-Host-Fingerprint und `pip check` |
| `audits/python_environment_provenance_venv_vs_venv_flash_manifest_20260716.json` | `059bd18860502ba421c03cf7ce16dbf979c2074dd6560a56dcba4253de6075c9` | Vergleich der Umgebungen und Laufhinweise |
| `audits/audit_method_challenges_reproducibility_20260719.md` | `05316e8b5011d5d3c374b074a045a50e6a24a3dc88f138404ca9718bb72905c6` | konsolidierte historische Provenienzgrenzen |

Der autoritative GPU-Host-Fingerprint lautet: Python 3.11.15, PyTorch `2.12.1+cu130`, CUDA 13.0, FlashAttention `2.8.3.post1`, Transformers 5.9.0, PEFT 0.18.1, TRL 1.4.0, Accelerate 1.12.0, Datasets 4.8.5, Sentence Transformers 5.2.2 und FAISS 1.14.3. Der aktuelle lokale `.venv_flash`-Freeze stimmt hashgenau mit dem am 2026-07-16 festgehaltenen Freeze ueberein.

### 6.2 Trainingslaeufe

Die drei finalen LoRA-v2-Trainingsmetadaten weisen denselben Torch-/CUDA-/FlashAttention-/L40S-Fingerprint wie `.venv_flash` aus. Die bestehende Provenienzklassifikation ist `EENV-2/HIGH`. Beim Llama-Lauf ist der Interpreter direkter dokumentiert; bei beiden Qwen-Laeufen fehlen `sys.executable` und Pythonversion als direkte historische Felder. Separate Konsolenlogs aller drei Trainingslaeufe fehlen.

Zulaessige Aussage: Die finalen Trainings sind `.venv_flash` mit hoher Konfidenz zugeordnet.  
Nicht zulaessige Ueberdehnung: Fuer jeden Qwen-Trainingsprozess sei der absolute Interpreter unmittelbar im Laufartefakt gespeichert.

### 6.3 Evaluationslaeufe

- Alle 36 k3-Metadatendateien mit Interpreterfeldern speichern ausschliesslich `/home/ec2-user/nl2sql_testbench/.venv_flash/bin/python` (je Datei `sys_executable` und `absolute_interpreter_path`).
- Fuer die 48 historische Hauptlaeufe bestehen bei zwei Runs starke `.venv_flash`-Startsequenzbelege. Bei 46 Runs wurde der Prozessinterpreter nicht direkt persistiert.
- Es wurde keine positive Evidenz fuer `.venv` oder System-Python in einem autoritativen Evaluationsrun gefunden.

Damit ist `.venv_flash` fuer die k3-Erweiterung direkt und fuer die Hauptmatrix nur teilweise direkt belegt.

### 6.4 Analyselaeufe

Das autoritative Environment-Manifest ordnet die primaeren finalen Statistik- und Fehleranalysen `.venv_flash` zu. Drei reine Standardbibliothek-Hilfsschritte liefen dokumentiert mit System-Python; ihnen wird kein Einfluss auf wissenschaftliche Zahlen zugeschrieben. Fuer alle spaeter additiv erzeugten Hilfs- und Plotartefakte existiert nicht zwingend ein einheitlich persistiertes `sys.executable`.

### 6.5 Gesamtfeststellung

`.venv_flash` ist als **autoritative Freeze- und wissenschaftliche Hauptumgebung bestaetigt**. Die Aussage, saemtliche historischen Projektprozesse seien ausnahmslos und direkt nachweisbar in genau dieser Umgebung gelaufen, waere dagegen zu stark. Das Projekt besitzt dokumentierte Mischverwendung fuer rein technische Hilfsschritte und Provenienzloecher bei aelteren Runs.

## 7. Empfehlung fuer den finalen Freeze

### 7.1 Unveraendert erhalten

1. `requirements.txt` mit Hash und mtime als historischen, nicht autoritativen Abhaengigkeitsstand.
2. `requirements_current.txt` mit Hash und mtime als exakten `.venv`-Snapshot.
3. Beide virtuellen Umgebungsverzeichnisse bis zum erfolgreichen Freeze und Restore-Test; `.venv` als nicht autoritative Vergleichsumgebung kennzeichnen.
4. Die vorhandenen Environment-Manifeste, GPU-Host-Evidenz und Provenienzaudits.

Keine der bestehenden Requirements-Dateien sollte stillschweigend ueberschrieben oder inhaltlich zur `.venv_flash`-Datei umgedeutet werden.

### 7.2 Zusaetzlich erzeugen, aber nicht im vorliegenden Read-only-Audit

Fuer `.venv_flash` sollte ein neuer datierter Snapshot erzeugt werden, mindestens mit:

- unveraendertem Rohoutput von `python -m pip freeze`,
- `pip list --format=json`, `pip check` und nach Moeglichkeit `pip inspect --local`,
- SHA256 aller Snapshotdateien,
- `sys.executable`, `sys.version`, `sys.prefix`, `pyvenv.cfg` und `python -m pip --version`,
- Torch-Version einschliesslich Local-Version (`+cu130`), `torch.version.cuda`, CUDA-Verfuegbarkeit und GPU-Modell,
- Transformers-, PEFT-, TRL-, Accelerate-, Datasets-, Sentence-Transformers-, FAISS- und FlashAttention-Version,
- Betriebssystem-, Treiber- und CUDA-Toolkit-Fingerprint,
- ausdruecklicher Kennzeichnung, dass der Snapshot `.venv_flash` und nicht `.venv` beschreibt.

Zusaetzlich sollte ein separater datierter `.venv`-Freeze archiviert werden, weil `requirements_current.txt` diese Umgebung repraesentiert und historische Vergleichbarkeit ermoeglicht. Die beiden Freeze-Dateien duerfen nicht zusammengefuehrt werden.

Ein heute erzeugter Freeze belegt den heute vorhandenen Zustand. Er ersetzt nicht rueckwirkend die fehlenden Interpreterfelder aelterer Runs. Fuer moeglichst starke Reproduzierbarkeit sollten ferner die Paketquellen beziehungsweise Wheel-Identitaeten und ein tatsaechlicher Restore-Test archiviert werden; `pip freeze` allein garantiert keine plattformuebergreifend bitidentische Wiederherstellung.

## 8. Offene Unsicherheiten und methodische Grenzen

1. Bei 46/48 Hauptlaeufen fehlt ein direkt persistierter Runtime-Interpreter.
2. Bei den beiden Qwen-Trainings fehlt direkte `sys.executable`-Evidenz; die Zuordnung ist trianguliert.
3. Separate Konsolenlogs der drei finalen Trainings fehlen.
4. `requirements_current.txt` traegt keinen Umgebungsnamen und kann ohne diesen Audit faelschlich als allgemeiner Projektfreeze gelesen werden.
5. Keine bestehende Requirements-Datei repraesentiert `.venv_flash` vollstaendig.
6. `pip freeze` dokumentiert weder Wheel-Hashes noch in jedem Fall den verwendeten Paketindex oder die Hardwarekompatibilitaet.
7. Das optionale importierte Diagnosemodul `kernels` ist in keiner Umgebung installiert; dies ist fuer die bestaetigten autoritativen Runs nicht als Blocker belegt.
8. Der vollstaendige Projektfreeze mit rekursivem Gesamtmanifest, schreibgeschuetztem Archiv, externen Backups und Restore-Test ist laut bestehendem Reproduzierbarkeitsaudit weiterhin nicht abgeschlossen.

## 9. Read-only-Bestaetigung

Ausgefuehrt wurden ausschliesslich lesende Dateioperationen, statische AST-Importanalyse sowie `python --version`, `pip --version`, `pip list`, `pip freeze`, `pip check` und Metadatenabfragen in beiden vorhandenen Umgebungen. Es wurden keine Pakete installiert, entfernt oder aktualisiert; keine virtuelle Umgebung wurde veraendert; kein Training, keine Evaluation, keine Inferenz, kein Retrieval, keine SQL-Ausfuehrung und keine wissenschaftliche Analyse wurde gestartet.

Neu angelegt wurde ausschliesslich:

- `audit_python_environment_requirements_final_freeze_20260721.md`

Keine bestehende Projektdatei wurde veraendert.

## Abschlussstatus

`PYTHON-ENVIRONMENT-REQUIREMENTS-FREEZE-AUDIT: PASS MIT WARNUNGEN`

`AUTHORITATIVE-PYTHON-ENVIRONMENT: .venv_flash CONFIRMED WITH PROVENANCE LIMITATIONS`

`EXISTING-REQUIREMENTS-FOR-VENV-FLASH: INCOMPLETE`

`NEW-DATED-VENV-FLASH-FREEZE: RECOMMENDED_NOT_CREATED`
