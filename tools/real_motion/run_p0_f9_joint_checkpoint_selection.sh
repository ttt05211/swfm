#!/usr/bin/env bash
# Read-only lineage audit + one window-major dev512 comparison. No full4369.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || { echo "用法: bash $0 已完成的20轮目录 [候选数8]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
RUNS_ROOT="$ROOT/outputs/p0_f9_joint_causal_columns"
cd "$ROOT"
RUN="$(realpath -e -- "$1")"
case "$RUN" in "$RUNS_ROOT"/*) ;; *) echo '拒绝不明确的实验路径' >&2; exit 2 ;; esac
PY="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
# Rejected TRAIN fastpaths stay OFF even when an old shell exported =1.
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
if pgrep -f '[p]ython.*(train_p0_f9_joint_causal_columns_full.py|benchmark_p0_f9_joint|eval_p0_f9_joint)' >/dev/null; then
  echo '请等训练/最终评估结束或安全暂停；不自动kill进程。' >&2; exit 2
fi
OUT="${JOINT_SELECTION_OUT:-$RUN/checkpoint_selection_$(date +%Y%m%d_%H%M%S)_$$}"
"$PY" -m pytest -q tests/test_joint_checkpoint_selection.py
echo '一次完成：真实续训链审计 -> 全部历史dev64记录筛选 -> 最多8个保留权重共享dev512评估。'
echo '全程CPU特征，原batch256；首窗概率逐项一致性检查；中断可显式--resume；不删除、不自动部署、不跑full4369。'
echo "输出: $OUT"
"$PY" -u tools/real_motion/eval_p0_f9_joint_checkpoints.py \
  --run-dir "$RUN" --runs-root "$RUNS_ROOT" --shortlist-size "${2:-8}" \
  --population dev512 --out-dir "$OUT" --cpu-workers 8 --checkpoint-every 8
echo "请发回 $OUT/selection.txt 和 $OUT/summary.txt；完整数据见 comparison.json。"
