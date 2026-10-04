#!/usr/bin/env bash
# Opt-in completed15 -> total20; old artifacts are read-only. No GPU IDs guessed.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -ge 1 && $# -le 3 ]] || { echo "用法: bash $0 原实验目录 [总轮数20] [GPU IDs: 0 或 0,1]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
RUN="$(realpath -e -- "$1")"
case "$RUN" in "$ROOT"/outputs/p0_f9_joint_causal_columns/*) ;; *) echo '拒绝不明确的实验路径' >&2; exit 2 ;; esac
EPOCHS="${2:-20}"; GPUS="${3:-0}"
case "$GPUS" in 0|0,1) ;; *) echo '此服务器包装脚本只接受已确认的0或0,1；其他编号用manager --gpus显式指定' >&2; exit 2 ;; esac
if pgrep -f '[p]ython.*(train_p0_f9_joint_causal_columns_full.py|benchmark_p0_f9_joint|eval_p0_f9_joint_causal_columns.py)' >/dev/null; then
  echo '请先安全停止旧训练/测速/联合评估；不会自动kill进程。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
export SWFM_LOCAL_FAST_SUPERVISION="${SWFM_LOCAL_FAST_SUPERVISION:-0}" SWFM_LOCAL_STATIC_ROI="${SWFM_LOCAL_STATIC_ROI:-0}"
"$PY" -c 'import sys,torch; expected=len(sys.argv[1].split(",")); assert torch.cuda.device_count()==expected and torch.cuda.is_bf16_supported(), "GPU visibility/BF16 mismatch"; print("VISIBLE_GPUS",[(i,torch.cuda.get_device_name(i)) for i in range(expected)])' "$GPUS"
"$PY" -u tools/real_motion/check_column_native_cpu.py
"$PY" -m pytest -q tests/test_joint_training_distributed.py
echo '仅从已完成的last.pt追加训练；保留optimizer和历史，后续从末端LR余弦下降，无LR重启。'
echo '全局batch4/source128不翻倍；复用已有暖缓存；每轮dev64，最终dev512。'
if [[ "$GPUS" == 0,1 ]]; then
  echo '双卡切换有独立rank RNG，不承诺单卡/双卡逐位一致；仅rank0写checkpoint；不会覆盖15轮结果。'
else
  echo '单卡追加；TRAIN监督索引/整数ROI默认关闭（配对loss gate未通过）；Column特征保持CPU路径。'
fi
EXTRA=()
if [[ -n "${FULL_JOINT_EXTEND_OUT:-}" ]]; then EXTRA+=(--out-dir "$FULL_JOINT_EXTEND_OUT"); fi
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py extend \
  --run-dir "$RUN" --to-epochs "$EPOCHS" --gpus "$GPUS" \
  --column-feature-backend cpu --sampling-workers 6 --profile-every 32 "${EXTRA[@]}"
