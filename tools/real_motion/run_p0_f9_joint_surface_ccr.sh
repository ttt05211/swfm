#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '通过 JOINT_RESUME / JOINT_EPOCHS 等环境变量配置；不接收位置参数。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
TRAIN_HISTORY="${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
VAL_HISTORY="${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
OUT="${JOINT_SURFACE_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/full${JOINT_EPOCHS:-20}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖：$OUT；resume 也用新目录。" >&2; exit 2; }
for FILE in "$E14" "$ROOT/configs/real_motion_occfm.yaml" "$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
for DIR in "$TRAIN_HISTORY" "$VAL_HISTORY" "$DATAROOT"; do
  [[ -d "$DIR" ]] || { echo "[MISSING] $DIR；不自动重建缓存。" >&2; exit 2; }
done
if [[ "${JOINT_EVALUATE_ONLY:-0}" != 1 ]] && pgrep -f '[p]ython.*train_p0_f9_' >/dev/null; then
  echo '已有训练进程，请先安全停止；不会 kill 任何进程。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
EXTRA=()
if [[ -n "${JOINT_RESUME:-}" ]]; then
  [[ -f "$JOINT_RESUME" ]] || { echo "[MISSING] $JOINT_RESUME" >&2; exit 2; }
  EXTRA+=(--resume "$JOINT_RESUME")
fi
if [[ "${JOINT_EVALUATE_ONLY:-0}" == 1 ]]; then
  [[ -n "${JOINT_RESUME:-}" ]] || { echo '只读评估需要 JOINT_RESUME。' >&2; exit 2; }
  EXTRA+=(--evaluate-only --eval-population "${JOINT_EVAL_POPULATION:-dev512}")
fi
if [[ -n "${JOINT_STOP_AFTER_EPOCH:-}" ]]; then EXTRA+=(--stop-after-epoch "$JOINT_STOP_AFTER_EPOCH"); fi
if [[ -n "${JOINT_MAX_UPDATES:-}" ]]; then EXTRA+=(--max-updates "$JOINT_MAX_UPDATES"); fi
if [[ "${JOINT_SURFACE_REFERENCE:-0}" == 1 ]]; then EXTRA+=(--surface-reference-execution); fi
echo '干净一阶段：运动 + 静态/动态 Surface CCR 全部随机初始化、全部训练；无 teacher/KD/AE。'
echo "全部20430/轮，${JOINT_EPOCHS:-20}轮全周期余弦，无tail；batch4/source128；现有 TRAIN/VAL 缓存只读。"
echo '每轮固定dev64，最后dev512；不自动full4369、不选dev-best。不接受旧Local或冻结CCR断点。'
echo "输出 $OUT；Ctrl+C/TERM 在完整更新后保存；kill -9 仅恢复最近周期断点。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/train_p0_f9_joint_surface_ccr.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --base-checkpoint "$E14" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" --dataroot "$DATAROOT" \
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --out-dir "$OUT" --ccr-history-cache "$TRAIN_HISTORY" --ccr-history-cache-mode require \
  --ccr-val-history-cache "$VAL_HISTORY" --ccr-history-cache-ram-mib 1024 --ccr-val-history-cache-ram-mib 512 \
  --descriptor-disk-mib 0 --descriptor-ram-mib 256 --frame-cache-mib 4096 \
  --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --surface-query-workers 4 --ccr-prefetch-workers 4 \
  --epochs "${JOINT_EPOCHS:-20}" --seed 20261008 --window-batch-size 4 --source-budget 128 \
  --motion-lr 0.0005 --repair-lr 0.0003 --samples-per-role 1024 --cpu-workers 10 \
  --checkpoint-every 128 "${EXTRA[@]}" 2>&1 | tee "$OUT.log"
echo "结果 $OUT/summary.txt；每轮 epoch_XXXX.pt；恢复 $OUT/last.pt。"
