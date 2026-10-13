#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
if pgrep -f '[p]ython.*tools\.(shared_history_experiments\.(loss_controls|loss_minimal|refine)\.|scene_ego_head\.experiment\.)' >/dev/null; then
  echo '其他训练仍在运行，请先等结束；本脚本不停止进程。'; exit 1
fi
export REFINER_MODULE=tools.scene_ego_head.experiment.internal_only.run EPOCHS=1
if [[ "${SCENE_EGO_RESUME:-0}" == 1 ]]; then
  : "${RUN_DIR:?恢复需指定原内部-only RUN_DIR}"
  export RESUME="$RUN_DIR/last.pt"
  test -f "$RESUME"
else
  export RESUME=""
fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_ego_internal500_$(date +%Y%m%d_%H%M%S)}"
echo '随机联合500步，H6 CE+Lovasz+ego；结束只评内部ego完整DEV512，无planner依赖或GT参考预测。'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" --mode pred "$@"
