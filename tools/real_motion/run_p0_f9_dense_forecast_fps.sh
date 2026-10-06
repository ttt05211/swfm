#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/nas/occ/swfm
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "OccFM" ]]; then
  echo '请先 conda activate OccFM；终版 FPS 不会调用 base Python。' >&2
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"

OUT=$ROOT/outputs/p0_f9_point_ccr/dense_forecast_fps_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p "$(dirname "$OUT")"

echo '===== FINAL DENSE FORECAST FPS ====='
echo '唯一正式计时边界：CausalHistoryState -> 六帧完整 dense semantic occupancy。'
echo '计入：fresh KTA/Strong、V18 motion、SE(2) transport/layering、future projection/ownership、CCR、dense composition。'
echo '不计：history I/O、history-only source extraction/association/registration/canonical evidence、编译/预热、GT/metrics/hash/save。'
echo 'batch=1；20窗口=18 scene-balanced + 2 high-source stress；3遍；CUDA前后同步；FPS=总未来帧数/总wall time。'
echo '每个窗口正式计时前先做旧路径 vs 新数据流逐字节 parity；失败立即停止，不自动重训、不静默降级。'
echo "输出：$OUT"

"$PY" -u tools/real_motion/benchmark_p0_f9_dense_forecast_fps.py   --config "$CONFIG"   --checkpoint "$TEACHER"   --ccr-checkpoint "$CCR"   --base-checkpoint "$BASE"   --dev-cache "$DEV"   --population-manifest "$MANIFEST"   --dataroot "$DATAROOT"   --dev-info "$INFO"   --out-dir "$OUT"   --windows 20   --stress-windows 2   --repeats 3   --cpu-workers 8   --ccr-cpu-workers 4   2>&1 | tee "$OUT.log"

echo "结果：$OUT/summary.txt"
echo "完整样本：$OUT/dense_forecast_fps.json"
echo "固定人口：$OUT/fps_manifest.json"
