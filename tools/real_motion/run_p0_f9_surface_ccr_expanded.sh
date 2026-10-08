#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '通过 SURFACE_CKPT / SURFACE_EVAL_RESUME 设置路径，不接收位置参数。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
TEACHER="${E19:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
FROZEN_B="${EPOCH2_CKPT:-$ROOT/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt}"
SURFACE_CKPT="${SURFACE_CKPT:-$ROOT/outputs/p0_f9_surface_ccr/full20430x3_20261008_131503_838/last.pt}"
CCR_VAL_CACHE="${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
OUT="${SURFACE_EVAL_OUT:-$ROOT/outputs/p0_f9_surface_ccr/expanded_$(date +%Y%m%d_%H%M%S)_$$}"
EVAL_EXECUTION="${SURFACE_EVAL_EXECUTION:-eager}"
[[ "$EVAL_EXECUTION" == eager || "$EVAL_EXECUTION" == full_chunk_graph ]] || { echo 'SURFACE_EVAL_EXECUTION必须是eager或full_chunk_graph' >&2; exit 2; }
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT；恢复也使用新目录。" >&2; exit 2; }
for FILE in "$TEACHER" "$E14" "$FROZEN_B" "$SURFACE_CKPT" "$ROOT/configs/real_motion_occfm.yaml" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
[[ -d "$CCR_VAL_CACHE" ]] || { echo "[MISSING] $CCR_VAL_CACHE；不会自动重建缓存。" >&2; exit 2; }
EXPECTED_B=fb6f7bdfa9e7ebca8f479d7005372ad328e09a2ed44e84514cfb180722942204
ACTUAL_B="$(sha256sum -- "$FROZEN_B")"
[[ "${ACTUAL_B%% *}" == "$EXPECTED_B" ]] || { echo 'Frozen B 不是已确认的 epoch2；停止。' >&2; exit 2; }
if pgrep -f '[p]ython.*(train_p0_f9_|validate_p0_f9_surface_ccr_expanded.py)' >/dev/null; then
  echo '另有训练/同类扩大验证在运行，请先安全结束；本脚本不 kill 进程。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
EXTRA=()
if [[ -n "${SURFACE_EVAL_RESUME:-}" ]]; then
  [[ -f "$SURFACE_EVAL_RESUME" ]] || { echo "[MISSING] $SURFACE_EVAL_RESUME" >&2; exit 2; }
  EXTRA+=(--resume-eval "$SURFACE_EVAL_RESUME")
fi
if [[ -n "${SURFACE_EVAL_MAX_WINDOWS:-}" ]]; then EXTRA+=(--max-windows "$SURFACE_EVAL_MAX_WINDOWS"); fi
echo '一次完成：固定第三轮 Surface CCR + Frozen B 同20窗口x3正式FPS/逐字节校验，然后 full4369 一次扩大验证。'
echo '不训练、不调阈值、不改变候选/网络；CUDA Graph仅纯静态分块，未通过字节检查/未更快则沿用 eager。'
echo "全量质量评估执行=$EVAL_EXECUTION；默认eager，不按单窗口预热FPS自动启用变长尾块建图。"
echo '同时报告 DEV512、之外3857窗口及之外场景；已参与研究的数据，不是全新独立测试。'
echo "复用完整VAL缓存（只读）。输出：$OUT；Ctrl+C保存整数进度，kill -9恢复最后32窗口周期保存。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/validate_p0_f9_surface_ccr_expanded.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$TEACHER" \
  --ccr-checkpoint "$SURFACE_CKPT" --frozen-b-checkpoint "$FROZEN_B" --base-checkpoint "$E14" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dataroot "$DATAROOT" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --ccr-val-history-cache "$CCR_VAL_CACHE" --ccr-val-history-cache-ram-mib 512 \
  --cpu-workers 10 --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 \
  --eval-surface-execution "$EVAL_EXECUTION" \
  --out-dir "$OUT" "${EXTRA[@]}" 2>&1 | tee "$OUT.log"
echo "结果：$OUT/summary.txt；完整：$OUT/expanded_validation.json；恢复：$OUT/evaluation_progress.pt"
