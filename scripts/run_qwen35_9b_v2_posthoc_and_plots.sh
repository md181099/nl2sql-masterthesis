#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv_flash/bin/python3"
TRAIN_CONFIG="configs/train_lora_qwen35_9b_base_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5.json"
LOSS_CONFIG="configs/eval_posthoc_loss_qwen35_9b_v2_mixedval2500_schemaheaderfix_all_checkpoints.json"
PLOT_CONFIG="configs/plot_qwen35_9b_v2_training_diagnostics.json"
VALIDATION="data/sql_create_context/val_sft_qwen35_full_chat_v2_mixed_trainothers700_sqlcc1800_no_train_no_dev_overlap_seed42_schemaheaderfix.jsonl"
ADAPTER_ROOT="adapters/qwen35_9b_base/lora_v2_fullchat_old25k_r8_alpha16_mixedval2500_v2_schemaheaderfix_evalstop_maxlen2048_epochs5"

check_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA256 mismatch: %s\nexpected=%s\nactual=%s\n' "$path" "$expected" "$actual" >&2
    exit 1
  }
}

[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON" >&2; exit 1; }
[[ -d "$ADAPTER_ROOT/checkpoints" ]] || { echo "Training output/checkpoints missing: $ADAPTER_ROOT" >&2; exit 1; }
check_hash "0bfce20d1e97f0b42b61d3db67679e3feef46b94a58a09147e4a5fb82240815e" "$TRAIN_CONFIG"
check_hash "711b23a6dfca40234a33e9aca66506eb33df197f69b6f466fd875854bdb89c08" "$VALIDATION"
check_hash "df93d078575dbe4e3ae50f48f820685f6b63ac01484b22c02d3bc8e7cad31042" "$LOSS_CONFIG"
check_hash "f89b9898d05441ed910f0cdb2acd123efad0814bc5c5a5a9373809b79a6cafa0" "src/21_eval_qwen35_posthoc_loss_general.py"
check_hash "b2d2277c9e96a0bed6a1521a9f16f2908b1004afb61eeb05e0eaf98b2338db00" "src/23_build_qwen35_training_diagnostics_table.py"
check_hash "270301857c4a472c4e60629e095e7f9569ecfbc64dc8e90aff5ac335672fecc9" "src/24_plot_qwen35_training_diagnostics.py"
check_hash "c5d5ca88f02511743772e0fd45f04c5367ed55ae6dc852badd29969894207620" "$PLOT_CONFIG"

"$PYTHON" src/23_build_qwen35_training_diagnostics_table.py --config "$PLOT_CONFIG" --preflight
"$PYTHON" src/21_eval_qwen35_posthoc_loss_general.py --config "$LOSS_CONFIG" --run-all
"$PYTHON" src/23_build_qwen35_training_diagnostics_table.py --config "$PLOT_CONFIG" --require-posthoc
MPLCONFIGDIR=/tmp/qwen35_matplotlib_cache "$PYTHON" src/24_plot_qwen35_training_diagnostics.py --config "$PLOT_CONFIG"

echo "PASS: Qwen 3.5 9B v2 post-hoc diagnostics and plots completed."
