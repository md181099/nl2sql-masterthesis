# Configuration Registry

Dieses Verzeichnis enthält historische, explorative und autoritative Trainings- und Evaluationskonfigurationen.

Die autoritativen Evaluationskonfigurationen werden nicht verschoben oder umbenannt. Ihre verbindliche Kennzeichnung erfolgt über:

- `authoritative_config_index.csv`
- `../reproducibility/authoritative_experiment_manifest.csv`
- `../audits/derived/generation_evaluation_authoritative_run_matrix_20260718.csv`

## Autoritative Auswertung

| Run-Familie | Anzahl |
|---|---:|
| `K3_EXTENSION` | 36 |
| `PRE_K3_AUTHORITATIVE_BASELINE` | 48 |

Die autoritative Ergebnismenge umfasst insgesamt 84 Läufe:

- 48 Läufe der ursprünglichen Hauptauswertung
- 36 Läufe der additiven `k=3`-Erweiterung

## Modellrollen

| Modelllinie | Rolle | Anzahl |
|---|---|---:|
| `Llama 3.2 3B Instruct` | `base` | 14 |
| `Llama 3.2 3B Instruct` | `lora_v2` | 14 |
| `Qwen 3.5 2B` | `base` | 14 |
| `Qwen 3.5 2B` | `lora_v2` | 14 |
| `Qwen 3.5 9B` | `base` | 14 |
| `Qwen 3.5 9B` | `lora_v2` | 14 |

Konfigurationen, die nicht im autoritativen Index erscheinen, gehören zur Entwicklung, zu Smoke-Tests, Sensitivitätsanalysen, Ablationen oder historischen Modellständen.

Die SHA-256-Werte im Index ermöglichen die Überprüfung, dass eine Konfiguration exakt dem eingefrorenen autoritativen Stand entspricht.
