#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
if pgrep -f '[p]ython.*tools\.(shared_history_experiments\.(loss_controls|loss_minimal|refine)\.|scene_ego_head\.experiment\.run)' >/dev/null; then
  echo '另一个训练仍在运行，请等结束再运行；本脚本不停止其他进程。'; exit 1
fi
export REFINER_MODULE=tools.scene_ego_head.experiment.run EPOCHS=1
if [[ "${SCENE_EGO_RESUME:-0}" == 1 ]]; then
  : "${RUN_DIR:?恢复需指定原 RUN_DIR}"
  export RESUME="$RUN_DIR/last.pt"
  test -f "$RESUME"
else
  export RESUME=""
fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_ego_joint500_$(date +%Y%m%d_%H%M%S)}"
echo '整个网络RANDOM联合500步；原版H6；CE+Lovasz+ego。结束自动同窗口内部/外部planner/GT参考，不训练外部对照第二模型。'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" --mode pred \
  --plan-cache "${PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}" "$@"
