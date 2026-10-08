#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '使用 CCR_RESUME=/完整路径/last.pt 指定恢复，不接收位置参数。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
TEACHER="${E19:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
EPOCH2_CKPT="${EPOCH2_CKPT:-$ROOT/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt}"
CCR_HISTORY_CACHE="${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
CCR_VAL_CACHE="${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
OUT="${CCR_SURFACE_OUT:-$ROOT/outputs/p0_f9_surface_ccr/full20430x3_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT；恢复也使用新目录。" >&2; exit 2; }
for FILE in "$TEACHER" "$E14" "$EPOCH2_CKPT" "$ROOT/configs/real_motion_occfm.yaml" \
  "$ROOT/data/p0_f9_v18_se2_train_full.pt" "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
for DIR in "$CCR_HISTORY_CACHE" "$CCR_VAL_CACHE" "$DATAROOT"; do
  [[ -d "$DIR" ]] || { echo "[MISSING] $DIR；本脚本不会自动重建缓存。" >&2; exit 2; }
done
EXPECTED_B=fb6f7bdfa9e7ebca8f479d7005372ad328e09a2ed44e84514cfb180722942204
ACTUAL_B="$(sha256sum -- "$EPOCH2_CKPT")"
[[ "${ACTUAL_B%% *}" == "$EXPECTED_B" ]] || { echo 'Frozen B SHA256 与已确认 epoch2 不同；停止，避免拿错模型。' >&2; exit 2; }
if pgrep -f '[p]ython.*(train_p0_f9_joint|train_p0_f9_point_ccr|train_p0_f9_surface_ccr|train_p0_f9_height_shared_field)' >/dev/null; then
  echo '另有训练进程。请先安全保存并停止；本脚本不 kill 任何进程。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
EXTRA=()
if [[ -n "${CCR_RESUME:-}" ]]; then
  [[ -f "$CCR_RESUME" ]] || { echo "[MISSING] $CCR_RESUME" >&2; exit 2; }
  EXTRA+=(--resume "$CCR_RESUME")
fi
if [[ -n "${CCR_STOP_AFTER_EPOCH:-}" ]]; then
  EXTRA+=(--stop-after-epoch "$CCR_STOP_AFTER_EPOCH")
fi
echo '表面一致性 CCR：完整 TRAIN20430 x 3轮；复用 Frozen B 权重和 TRAIN 正权重，新 optimizer/全程余弦。'
echo '先验证：冻结 epoch19 motion 和原动态 CCR，只训练 CCR 内部静态编码投影/条件读出；无外挂 logit 修正器。'
echo '保留 fast pipeline / logical batch4 / source128；TRAIN/VAL 几何缓存 require，只读，不新增大缓存。'
echo '每轮 dev64；最后 dev512 + Frozen B 同窗口20x3正式 FPS。不会自动启动从头联合训练。'
echo "输出：$OUT；Ctrl+C/TERM 完整更新后保存；kill -9 只能恢复最后周期断点。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/train_p0_f9_surface_ccr.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$TEACHER" \
  --warm-start-head "$EPOCH2_CKPT" --base-checkpoint "$E14" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --out-dir "$OUT" \
  --ccr-history-cache "$CCR_HISTORY_CACHE" --ccr-history-cache-mode require \
  --ccr-val-history-cache "$CCR_VAL_CACHE" \
  --ccr-history-cache-ram-mib 1024 --ccr-val-history-cache-ram-mib 512 \
  --descriptor-disk-mib 0 --descriptor-ram-mib 256 --frame-cache-mib 4096 \
  --ccr-fast-train --ccr-cpu-execution native_parallel --ccr-cpu-workers 10 \
  --ccr-prefetch-workers 6 --ccr-motion-superbatch-updates 4 --ccr-motion-streams 4 \
  --samples-per-role 1024 --cpu-workers 10 --epochs 3 --train-fraction 1 --lr 0.0003 \
  --eval-windows 64 --fps-windows 20 --speed-repeats 3 "${EXTRA[@]}" 2>&1 | tee "$OUT.log"
echo "结果：$OUT/summary.txt；详细：$OUT/screen.json；新断点：$OUT/last.pt"
