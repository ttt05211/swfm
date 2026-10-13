#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TASK="${1:?Usage: run.sh gt|stc|long6|waymo2|waymo10 [extra arguments]}"
shift
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
: "${GT_CKPT:?Set GT_CKPT to completed full-GT train/epoch_0002.pt}"
: "${EVAL_DIR:?Set EVAL_DIR to a NEW evaluation directory; reuse it ONLY with FINAL_EVAL_RESUME=1}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
args=(--task "$TASK" --gt-checkpoint "$GT_CKPT" --out-dir "$EVAL_DIR"
  --config "$ROOT/configs/real_motion_occfm.yaml" --population "${POPULATION:-all}"
  --cpu-workers "${CPU_WORKERS:-4}" --frame-cache-mib 256)
case "$TASK" in
  gt|stc|long6)
    DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
    args+=(--dataroot "$DATAROOT" --dev-info "${DEV_INFO:-$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl}"
      --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}")
    if [[ "$TASK" == stc ]]; then
      args+=(--stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
      if [[ -n "${HEAD10_CKPT:-}" ]]; then args+=(--head10-checkpoint "$HEAD10_CKPT"); fi
    fi
    if [[ "$TASK" == long6 ]]; then
      args+=(--geniedrive-info "${GENIEDRIVE_INFO:-$ROOT/data/geniedrive/world-nuscenes_infos_val.pkl}")
    fi ;;
  waymo2|waymo10)
    args+=(--waymo-root "${WAYMO_ROOT:-$ROOT/data/waymo}" --raw-free-label "${WAYMO_FREE_LABEL:-23}") ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
if [[ "${FINAL_EVAL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "FROZEN FINAL task=$TASK; GT epoch2; Pred=original head10 only; no training/old Surface routes"
exec "$PY" -u -m tools.final_semantic_evaluation.run "${args[@]}" "$@"
