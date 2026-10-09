#!/usr/bin/env bash
# Separate frozen inference experiment; original long entries/weights untouched.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 3 ]] || { echo "用法: bash $0 [screen|all] [OUTPUT_DIR] [--resume]" >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
MODE="${1:-screen}"
case "$MODE" in screen|all) ;; *) echo '模式须为screen/all' >&2; exit 2 ;; esac
RUN="${SURFACE_COMPARE_RUN:-$ROOT/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787}"
OUT="${2:-$ROOT/outputs/p0_f9_joint_surface_ccr/geometry_carry_${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
EXTRA=()
if [[ $# == 3 ]]; then
  [[ "$3" == --resume && -f "$OUT/evaluation_state.json" ]] || { echo '续评须指定原输出目录和--resume' >&2; exit 2; }
  EXTRA+=(--resume)
else
  [[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT；继续评估须显式--resume" >&2; exit 2; }
fi
[[ -z "${SURFACE_MEAN_SOURCE:-}" ]] || EXTRA+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE")
case "${SURFACE_LONG_GRAPHS:-1}" in 1) ;; 0) EXTRA+=(--no-graphs) ;; *) echo 'SURFACE_LONG_GRAPHS须为0/1' >&2; exit 2 ;; esac
VAL_CACHE="${SURFACE_LONG_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
if [[ "$VAL_CACHE" != off ]]; then
  [[ -d "$VAL_CACHE" ]] || { echo "[MISSING] $VAL_CACHE；不用缓存可设SURFACE_LONG_VAL_CACHE=off" >&2; exit 2; }
  EXTRA+=(--val-history-cache "$VAL_CACHE")
fi
REQUIRED=("$ROOT/configs/real_motion_occfm.yaml" "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
  "$ROOT/data/p0_f9_v21_dev64_manifest.json" "$E14" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" "$RUN/training.json")
if [[ "$MODE" == screen ]]; then
  TRAIN_CACHE="${TRAIN_V18:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  TRAIN_INFO="${TRAIN_INFO:-$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl}"
  REQUIRED+=("$TRAIN_CACHE" "$TRAIN_INFO")
  EXTRA+=(--train-cache "$TRAIN_CACHE" --train-info "$TRAIN_INFO")
else
  if [[ -n "${SURFACE_CARRY_APPROVED_ROUTE:-}" ]]; then
    [[ "$SURFACE_CARRY_APPROVED_ROUTE" == static_carry ]] || { echo '用户指定只支持static_carry；不是原TRAIN gate通过' >&2; exit 2; }
    EXTRA+=(--approved-route "$SURFACE_CARRY_APPROVED_ROUTE")
  else
    [[ -n "${SURFACE_CARRY_SELECTION:-}" ]] || { echo 'all须设置SURFACE_CARRY_SELECTION；或显式SURFACE_CARRY_APPROVED_ROUTE=static_carry' >&2; exit 2; }
    REQUIRED+=("$SURFACE_CARRY_SELECTION")
    EXTRA+=(--selection-from "$SURFACE_CARRY_SELECTION")
  fi
  case "${SURFACE_CARRY_METRICS_ONLY:-1}" in
    1) EXTRA+=(--iou-miou-only) ;;
    0) ;;
    *) echo 'SURFACE_CARRY_METRICS_ONLY须为0/1' >&2; exit 2 ;;
  esac
  GENIE_INFO="${GENIEDRIVE_INFO:-$ROOT/data/geniedrive/world-nuscenes_infos_val.pkl}"
  EXTRA+=(--geniedrive-info "$GENIE_INFO"
    --population all --population-alignment geniedrive_code10s)
fi
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || { echo "[MISSING] $path" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ "$MODE" == all ]]; then
  "$PY" -u tools/real_motion/download_geniedrive_eval_info.py --out "$GENIE_INFO"
fi
mkdir -p "$(dirname -- "$OUT")"
echo '冻结平均5/6/8/12/14，阈值不变：baseline / static carry / predicted SE(2) carry / combined。'
echo 'screen一趟TRAIN64+dev64，共享首段；只用TRAIN规则选择候选。all只复测冻结候选+baseline。'
if [[ "$MODE" == all && -n "${SURFACE_CARRY_APPROVED_ROUTE:-}" ]]; then
  echo '显式用户选择static_carry，接受screen小幅Moving下降；不宣称原TRAIN门槛通过。'
fi
echo "模式=$MODE；原始4历史+自预测几何状态，未来仅GT ego poses；输出：$OUT"
"$PY" -u tools/real_motion/eval_p0_f9_surface_geometry_carry.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --run-dir "$RUN" \
  --runs-root "$ROOT/outputs/p0_f9_joint_surface_ccr" --out-dir "$OUT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$E14" --dataroot "$DATAROOT" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --experiment "$MODE" --cpu-workers "${SURFACE_LONG_CPU_WORKERS:-10}" --prefetch-workers 4 \
  --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --surface-query-workers 4 \
  --majority-workers "${SURFACE_CARRY_MAJORITY_WORKERS:-4}" \
  "${EXTRA[@]}" 2>&1 | tee -a "$OUT.log"
echo "完成：$OUT/summary.txt；详细指标/审计：$OUT/evaluation.json"
