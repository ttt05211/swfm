#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
if pgrep -f '[p]ython.*tools.shared_history_experiments.(loss_controls|loss_minimal|refine)\.' >/dev/null; then
  echo '训练仍在运行；请等结束再测速。本脚本不停止任何进程。'; exit 1
fi
export RESUME="${REFINER_CKPT:-$ROOT/outputs/p0_f9_shared_history/refiner_probe_2ep_20261012_184856/train/epoch_0002.pt}"
test -f "$RESUME" || { echo "Missing epoch2: $RESUME"; exit 1; }
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_pipeline_speed_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS=2 REFINER_MODULE=tools.shared_history_experiments.refiner_pipeline.probe
echo '只读冻结epoch2：先原版3次F/Q/prior/memory/logits replay，再相同完整训练A/B；NO FP64融合/降精度/新正式训练。'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" --probe-steps "${PROBE_STEPS:-4}" --passes 2 --profile "$@"
