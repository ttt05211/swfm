#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/nas/occ/swfm
cd "$ROOT"
if [[ "${CONDA_DEFAULT_ENV:-}" != "OccFM" ]]; then
  echo '请先 conda activate OccFM；不会调用 base Python。' >&2
  exit 1
fi
PY="$(command -v python)"
if pgrep -af '[p]ython.*(train_p0_f9|eval_p0_f9|benchmark_p0_f9|run_p0_f9_shared)' >/dev/null; then
  echo '检测到其他训练/评估/测速 Python；请先正常停止，避免同卡计时被干扰。不会自动终止进程。' >&2
  exit 1
fi
CONFIG=$ROOT/configs/real_motion_occfm.yaml
TEACHER=$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
BASE=$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt
CCR=${1:-$ROOT/outputs/p0_f9_point_ccr/screen20x3_20261006_131352_837/last.pt}
DEV=$ROOT/data/p0_f9_v18_se2_val_all_4369.pt
MANIFEST=$ROOT/data/p0_f9_v21_dev64_manifest.json
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
INFO=$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl
for file in "$CONFIG" "$TEACHER" "$BASE" "$CCR" "$DEV" "$MANIFEST" "$INFO"; do
  [[ -f "$file" ]] || { echo "[MISSING] $file" >&2; exit 1; }
done
[[ -d "$DATAROOT" ]] || { echo "[MISSING] $DATAROOT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
OUT=$ROOT/outputs/p0_f9_point_ccr/v18_fps_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p "$(dirname "$OUT")"
echo '仅测速：同20个窗口，18个scene-balanced + 2个高source压力窗口，重复3遍。'
echo '原始/无损V18各测6历史E14及4历史epoch19，再测4历史epoch19+点级CCR。'
echo '同时报fresh/cached Strong口径；六帧全部完成，逐字节检查；不训练、不改变阈值。'
echo "输出：$OUT"
"$PY" -u tools/real_motion/benchmark_p0_f9_point_ccr_v18_fps.py \
  --config "$CONFIG" --checkpoint "$TEACHER" --ccr-checkpoint "$CCR" \
  --base-checkpoint "$BASE" --dev-cache "$DEV" --population-manifest "$MANIFEST" \
  --dataroot "$DATAROOT" --dev-info "$INFO" --out-dir "$OUT" \
  --windows 20 --stress-windows 2 --repeats 3 --cpu-workers 8 \
  2>&1 | tee "$OUT.log"
echo "结果：$OUT/summary.txt；各窗口/分段数据：$OUT/speed.json"
