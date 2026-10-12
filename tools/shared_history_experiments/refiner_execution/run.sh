#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
if pgrep -f '[p]ython.*tools.shared_history_experiments.(loss_controls|refine)\.' >/dev/null; then
  echo '检测到 loss500/Refiner 实验仍运行；请等它结束，避免并行GPU负载污染测速。本脚本不停止进程。'
  exit 1
fi
export RESUME="${REFINER_CKPT:-$ROOT/outputs/p0_f9_shared_history/refiner_probe_2ep_20261012_184856/train/epoch_0002.pt}"
test -f "$RESUME" || { echo "Missing frozen epoch2: $RESUME"; exit 1; }
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_exec_speed_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS=2 REFINER_MODULE=tools.shared_history_experiments.refiner_execution.probe
echo 'READ ONLY epoch2 execution A/B; temporary gradients+Adam discarded; no new formal training or DEV evaluation.'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" --probe-steps "${PROBE_STEPS:-4}" --passes 2 "$@"
