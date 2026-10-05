#!/usr/bin/env bash
# Explicit immutable full-joint snapshot; never trains/recalibrates/promotes.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM" >&2; exit 2; fi
if [[ $# -lt 1 || $# -gt 3 ]]; then echo "用法: bash $0 RUN_DIR [dev64|dev512|full4369] [CHECKPOINT]" >&2; exit 2; fi
ROOT=/root/nas/occ/swfm
cd "$ROOT"
RUN="$(realpath -e -- "$1")"
if [[ "$RUN" == */model ]]; then MODEL="$RUN"; RUN="${RUN%/model}"; else MODEL="$RUN/model"; fi
case "$RUN" in "$ROOT"/outputs/p0_f9_joint_causal_columns/*) ;; *) echo "拒绝非full-joint实验目录" >&2; exit 2 ;; esac
POPULATION="${2:-dev64}"
case "$POPULATION" in dev64|dev512|full4369) ;; *) echo "必须显式指定dev64/dev512/full4369" >&2; exit 2 ;; esac
CHECKPOINT="${3:-$MODEL/last.pt}"
[[ -f "$CHECKPOINT" ]] || { echo "[MISSING] $CHECKPOINT" >&2; exit 2; }
OUT="${FULL_JOINT_EVAL_OUT:-$RUN/eval_${POPULATION}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
PY="$(command -v python)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
EXTRA=()
CPU_WORKERS="${FULL_JOINT_EVAL_CPU_WORKERS:-8}"
[[ "$CPU_WORKERS" =~ ^([1-9]|1[0-6])$ ]] || { echo 'FULL_JOINT_EVAL_CPU_WORKERS 必须为1到16。' >&2; exit 2; }
RAW_WORKERS="${FULL_JOINT_EVAL_RAW_WORKERS:-$((CPU_WORKERS < 4 ? CPU_WORKERS : 4))}"
RAW_DEPTH="${FULL_JOINT_EVAL_RAW_DEPTH:-$RAW_WORKERS}"
if [[ "${FULL_JOINT_EVAL_OPTIMIZED:-1}" == 1 ]]; then EXTRA+=(--optimized-inference); fi
if [[ "${FULL_JOINT_EVAL_IO_OPTIMIZED:-0}" == 1 ]]; then EXTRA+=(--buffered-chunk-io); fi
if [[ "${FULL_JOINT_EVAL_COLUMN_OPTIMIZED:-0}" == 1 ]]; then
  [[ "${FULL_JOINT_EVAL_OPTIMIZED:-1}" == 1 ]] || { echo 'column优化需要开启FULL_JOINT_EVAL_OPTIMIZED' >&2; exit 2; }
  EXTRA+=(--optimized-column-probability)
fi
if [[ "${FULL_JOINT_EVAL_COLUMN_MAP_PREFETCH:-1}" == 0 ]]; then EXTRA+=(--no-column-map-prefetch); fi
if [[ "${FULL_JOINT_EVAL_FIXED_MONITOR:-0}" == 1 ]]; then EXTRA+=(--fixed-monitor-thresholds); fi
EXECUTION_BACKEND="${FULL_JOINT_EVAL_EXECUTION_BACKEND:-eager}"
case "$EXECUTION_BACKEND" in eager|async_readback|graph_async|reuse_graph_async) ;; *) echo '无效的FULL_JOINT_EVAL_EXECUTION_BACKEND' >&2; exit 2 ;; esac
if [[ "$EXECUTION_BACKEND" != eager ]]; then
  [[ "${FULL_JOINT_EVAL_OPTIMIZED:-1}" == 1 && "${FULL_JOINT_EVAL_IO_OPTIMIZED:-0}" == 0 && "${FULL_JOINT_EVAL_FEATURE_BACKEND:-cpu}" == cpu ]] || {
    echo 'execution优化需要原分字段上传、CPU byte features和optimized inference；先通过独立测速。' >&2; exit 2;
  }
  EXTRA+=(--execution-backend "$EXECUTION_BACKEND")
fi
echo "只读快照评估 $POPULATION；last/epoch固定0.5/0.5/REMOVE-off，不用dev调阈值，不改变训练断点。"
echo "评估会使用CPU/GPU；大范围dev512建议先安全暂停训练。输出 $OUT"
echo 'CPU-only有界窗口预取；网络batch保持256；固定Strong all-6、概率及整数指标exactness检查不变。'
echo "执行后端 $EXECUTION_BACKEND（默认eager；不根据速度自动晋升）"
"$PY" -u tools/real_motion/eval_p0_f9_joint_causal_columns.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$CHECKPOINT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --dev-info /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --population "$POPULATION" --out-dir "$OUT" \
  --cpu-workers "$CPU_WORKERS" --batch-size 256 \
  --raw-prefetch-workers "$RAW_WORKERS" --raw-prefetch-depth "$RAW_DEPTH" \
  --column-feature-backend "${FULL_JOINT_EVAL_FEATURE_BACKEND:-cpu}" "${EXTRA[@]}"
echo "评估完成：$OUT/summary.txt；可继续恢复原训练，不会重置LR/RNG。"
