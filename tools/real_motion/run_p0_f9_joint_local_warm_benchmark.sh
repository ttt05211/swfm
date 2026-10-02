#!/usr/bin/env bash
# Speed diagnosis only: no full training, no checkpoint overwrite or auto-resume.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM" >&2; exit 2; fi
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
if pgrep -f '[p]ython.*(train_p0_f9_joint_causal_columns|benchmark_p0_f9_joint_local_warm)' >/dev/null; then
  echo "先安全停止旧联合训练，避免争抢CPU/GPU或改变last.pt；本脚本不会kill进程。" >&2; exit 2
fi
RUN_DIR="${LOCAL_WARM_CONTINUE:-${LOCAL_WARM_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/warm_speed_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}}"
if [[ -z "${LOCAL_WARM_CONTINUE:-}" && -e "$RUN_DIR" ]]; then echo "拒绝覆盖 $RUN_DIR" >&2; exit 2; fi
EXTRA=()
if [[ -n "${LOCAL_WARM_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$LOCAL_WARM_CHECKPOINT" ]]; then echo "[MISSING] $LOCAL_WARM_CHECKPOINT" >&2; exit 2; fi
  EXTRA+=(--checkpoint "$LOCAL_WARM_CHECKPOINT")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "仅计时：默认严格4历史→6未来；TRAIN128+8高source压力样本，几何RAM缓存关闭。"
if [[ -n "${LOCAL_WARM_COMPARE:-}" ]]; then
  echo "本次复用已冻样本/cache/prior；只做旧/新CPU流水线一次性比较，保留10%显存余量。"
else
  echo "独立子进程测batch4/8/16/32/64/128，OOM即停扩容；source上限同步增大，保留10%显存安全余量。"
  echo "初建缓存、TRAIN16诊断权重与单独cProfile不算稳态吞吐。"
fi
echo "不启动15轮。输出 $RUN_DIR"
if [[ -n "${LOCAL_WARM_COMPARE:-}" ]]; then
  if [[ -n "${LOCAL_WARM_CONTINUE:-}" ]]; then echo "比较模式不能同时接续旧报告" >&2; exit 2; fi
  echo "复用旧样本/cache/prior：只比较reference batch4和优化batch4/8/16/32，不重扫64/128。"
  # Tiny synthetic GPU integration gate, not another dataset experiment.
  "$PY" -m pytest -q tests/test_local_cpu_pipeline.py -k cuda
  "$PY" -u tools/real_motion/benchmark_p0_f9_joint_local_warm.py \
    --compare-run "$LOCAL_WARM_COMPARE" --out-dir "$RUN_DIR" --max-window-batch "${LOCAL_WARM_MAX_BATCH:-32}"
elif [[ -n "${LOCAL_WARM_CONTINUE:-}" ]]; then
  if [[ ! -f "$RUN_DIR/contract.json" ]]; then echo "[MISSING] $RUN_DIR/contract.json" >&2; exit 2; fi
  CONTINUE_EXTRA=()
  if [[ "${LOCAL_WARM_FINISH_EXISTING:-0}" == 1 ]]; then CONTINUE_EXTRA+=(--finish-existing); fi
  echo "接续诊断：复用已完成的缓存、prior和batch结果；不是恢复正式训练。"
  "$PY" -u tools/real_motion/benchmark_p0_f9_joint_local_warm.py \
    --continue-run "$RUN_DIR" --max-window-batch "${LOCAL_WARM_MAX_BATCH:-128}" "${CONTINUE_EXTRA[@]}"
else
"$PY" -u tools/real_motion/benchmark_p0_f9_joint_local_warm.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --geometry-cache "${FULL_JOINT_GEOMETRY_CACHE_DIR:-$ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1}" \
  --cache-gib "${FULL_JOINT_GEOMETRY_CACHE_GIB:-48}" \
  --history-frames "${LOCAL_WARM_HISTORY_FRAMES:-4}" \
  --sample-windows "${LOCAL_WARM_SAMPLE_WINDOWS:-128}" \
  --max-window-batch "${LOCAL_WARM_MAX_BATCH:-128}" --out-dir "$RUN_DIR" "${EXTRA[@]}"
fi
echo "完成。请发 $RUN_DIR/summary.txt 和 $RUN_DIR/cpu_profile.txt。"
echo "改变batch/source会改变优化步数：不能在旧full15断点上静默切换；暖缓存本身可复用。"
