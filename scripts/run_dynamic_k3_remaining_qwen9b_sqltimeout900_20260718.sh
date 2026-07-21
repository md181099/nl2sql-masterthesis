#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/nl2sql_testbench"
V3_SCRIPT="$ROOT/scripts/run_dynamic_k3_remaining_qwen9b_sqltimeout900_v3_20260718.sh"

echo "NOTICE: Delegating to corrected sqltimeout900 v3 runner; deferred Base Top-3 runs last."
exec bash "$V3_SCRIPT"
