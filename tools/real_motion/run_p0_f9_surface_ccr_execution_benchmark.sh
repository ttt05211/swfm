#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '通过 SURFACE_SPEED_OUT 等环境变量配置。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
SURFACE="${SURFACE_CKPT:-$ROOT/outputs/p0_f9_surface_ccr/full20430x3_20261008_131503_838/last.pt}"
FROZEN_B="${FROZEN_B_CKPT:-$ROOT/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt}"
E19="${E19:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
OUT="${SURFACE_SPEED_OUT:-$ROOT/outputs/p0_f9_surface_ccr/execution_speed_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
for FILE in "$SURFACE" "$FROZEN_B" "$E19" "$E14" "$ROOT/configs/real_motion_occfm.yaml" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '只读同20窗口×3测速：B、原surface eager/graph、投影复用eager/graph。'
echo '概率/六帧dense逐字节核对；fresh Strong计入；历史表面准备单独统计。不训练、不重跑full、不写原缓存。'
echo "输出 $OUT"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/benchmark_p0_f9_surface_ccr_execution.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$E19" --ccr-checkpoint "$SURFACE" \
  --frozen-b-checkpoint "$FROZEN_B" --base-checkpoint "$E14" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATAROOT" \
  --cpu-workers 10 --out-dir "$OUT" 2>&1 | tee "$OUT.log"
echo "结果 $OUT/summary.txt；分段 $OUT/speed.json。"
