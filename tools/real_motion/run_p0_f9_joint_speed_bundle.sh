#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# == 1 ]] || { echo "用法: bash $0 原训练目录" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
RUN="$(realpath -e -- "$1")"
case "$RUN" in "$ROOT"/outputs/p0_f9_joint_causal_columns/*) ;; *) echo '拒绝不明确的实验路径' >&2; exit 2 ;; esac
if pgrep -f '[p]ython.*(train_p0_f9_joint_causal_columns_full.py|benchmark_p0_f9_joint|eval_p0_f9_joint_causal_columns.py)' >/dev/null; then
  echo '请先安全停止旧训练/测速/评估；不会自动kill。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
OUT="${LOCAL_SPEED_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/speed_bundle_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || { echo '拒绝覆盖已有测速' >&2; exit 2; }
"$PY" -u tools/real_motion/check_column_native_cpu.py
"$PY" -m pytest -q tests/test_local_speed_bundle.py
echo '只跑一次TRAIN64速度对照和dev8完整候选一致性/耗时；不重建cache/prior，不保存科学训练更新，不自动续训。'
"$PY" -u tools/real_motion/benchmark_p0_f9_joint_speed_bundle.py --run-dir "$RUN" --out-dir "$OUT"
echo "结果: $OUT/summary.txt"
