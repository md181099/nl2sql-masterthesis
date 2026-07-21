#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/nl2sql_testbench"
PY="$ROOT/.venv_flash/bin/python"
RUNNER="$ROOT/src/06_batch_run_dynamic_k3_sqltimeout_v3.py"
VALIDATOR="$ROOT/scripts/validate_dynamic_k3_qwen9b_sqltimeout900_v3_group_20260718.py"
RESULTS="$ROOT/results/k3_extension_20260717"
LOG_DIR="$ROOT/logs/k3_extension_20260718_sqltimeout900v3"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
mkdir -p "$LOG_DIR"

if pgrep -af "06_batch_run_dynamic_k3|06_batch_run.py|07_lora|21_eval|23_build|24_plot" >/dev/null; then
  echo "ERROR: A relevant writer is already active. No run was started." >&2
  pgrep -af "06_batch_run_dynamic_k3|06_batch_run.py|07_lora|21_eval|23_build|24_plot" >&2 || true
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: NVIDIA GPU/driver is not visible. No run was started." >&2
  exit 1
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.get_device_name(0) == "NVIDIA L40S", torch.cuda.get_device_name(0)'

problematic_base_top3_config="configs/eval_qwen35_9b_base_dynamic_bge_large_top3_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"

base_configs_before_top3=(
  "configs/eval_qwen35_9b_base_dynamic_bge_large_top3_gate070_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_base_dynamic_bge_large_top3_gate085_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_base_dynamic_bge_large_top10_structure_rerank_v2_top3_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_base_dynamic_bge_large_top10_structure_rerank_v2_top3_gate070_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_base_dynamic_bge_large_top10_structure_rerank_v2_top3_gate085_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
)

lora_configs=(
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top3_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top3_gate070_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top3_gate085_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top10_structure_rerank_v2_top3_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top10_structure_rerank_v2_top3_gate070_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
  "configs/eval_qwen35_9b_lora_v2_old25k_r8_alpha16_mixedval2500_v2_bestepoch1_dynamic_bge_large_top10_structure_rerank_v2_top3_gate085_k3_full_schema_maxin4352_full_aliasnames_sqltimeout900v3.json"
)

prefix_for_config() {
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_output_prefix"])' "$1"
}

assert_collision_free() {
  local config prefix log
  for config in "$@"; do
    test -f "$config"
    prefix="$(prefix_for_config "$config")"
    log="$LOG_DIR/$prefix.log"
    if compgen -G "$RESULTS/${prefix}_*.csv" >/dev/null \
      || compgen -G "$RESULTS/${prefix}_*_metadata.json" >/dev/null \
      || compgen -G "$RESULTS/retrieval_traces/${prefix}_*_retrieval_traces.jsonl" >/dev/null \
      || test -e "$log"; then
      echo "ERROR: Output collision for $prefix. Nothing was started." >&2
      exit 1
    fi
  done
}

run_configs() {
  local label="$1"
  shift
  local config prefix log
  for config in "$@"; do
    prefix="$(prefix_for_config "$config")"
    log="$LOG_DIR/$prefix.log"
    echo "=== START $label: $config ==="
    "$PY" "$RUNNER" --config "$config" 2>&1 | tee "$log"
    echo "=== COMPLETE $label: $config ==="
  done
}

assert_collision_free \
  "${base_configs_before_top3[@]}" \
  "${lora_configs[@]}" \
  "$problematic_base_top3_config"

run_configs base_before_top3 "${base_configs_before_top3[@]}"
run_configs lora_v2 "${lora_configs[@]}"
"$PY" "$VALIDATOR" --role lora_v2

echo "=== START FINAL DEFERRED RUN: Qwen-9B Base Top-3 ==="
run_configs base_top3_last "$problematic_base_top3_config"
"$PY" "$VALIDATOR" --role base

echo "All 12 remaining Qwen-9B k=3 runs completed; deferred Base Top-3 ran last and both groups passed validation."
