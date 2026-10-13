#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
TASK="${1:-ego_checks}"
case "$TASK" in ego_checks) MODE=pred ;; gt_external) MODE=gt ;; *) echo '任务仅ego_checks或gt_external'; exit 2 ;; esac
shift || true
if pgrep -f '[p]ython.*tools\.(shared_history_experiments\.(loss_controls|loss_minimal|refine)\.|scene_ego_head\.(experiment|diagnostics)\.)' >/dev/null; then
  echo '另一个GPU任务仍运行，请等结束；本脚本不停止任何进程。'; exit 1
fi
EGO_CKPT="${EGO_CKPT:-$ROOT/outputs/p0_f9_shared_history/scene_ego_internal500_20261013_095714/last.pt}"
export RESUME="$EGO_CKPT"
if [[ "$TASK" == gt_external ]]; then export RESUME="${GT500_CKPT:-AUTO_GT500}"; fi
if [[ "${EGO_DIAG_RESUME:-0}" == 1 ]]; then : "${RUN_DIR:?恢复需原诊断RUN_DIR}"; fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/ego_diag_${TASK}_$(date +%Y%m%d_%H%M%S)}"
export REFINER_MODULE=tools.scene_ego_head.diagnostics.run EPOCHS=1
ARGS=(--mode "$MODE" --evaluate-only --task "$TASK" --ego-checkpoint "$EGO_CKPT")
if [[ "$TASK" == gt_external ]]; then ARGS+=(--plan-cache "${PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"); fi
if [[ "${EGO_DIAG_RESUME:-0}" == 1 ]]; then ARGS+=(--diagnostic-resume); fi
echo "只读任务=$TASK；0训练更新；不恢复旧Adam/不写原断点。Output: $RUN_DIR"
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" "${ARGS[@]}" "$@"
