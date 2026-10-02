#!/usr/bin/env bash
# Long full-data training, fresh models; whole-run cosine and exact resume.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM" >&2; exit 2; fi
EPOCHS="${1:-15}"
case "$EPOCHS" in 15|20) ;; *) echo "用法：bash $0 [15|20]" >&2; exit 2 ;; esac
FULL_ROOT=/root/nas/occ/swfm
cd "$FULL_ROOT"
PY="$(command -v python)"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
RUN_DIR="${FULL_JOINT_OUT:-$FULL_ROOT/outputs/p0_f9_joint_causal_columns/full${EPOCHS}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
WINDOWS="${FULL_JOINT_WINDOW_BATCH:-4}"
SOURCES="${FULL_JOINT_SOURCE_BUDGET:-128}"
WORKERS="${FULL_JOINT_CPU_WORKERS:-8}"
HISTORY="${FULL_JOINT_HISTORY_FRAMES:-4}"
if [[ -n "${FULL_JOINT_RESUME:-}" && -z "${FULL_JOINT_HISTORY_FRAMES:-}" ]]; then
  HISTORY="$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["model_configs"]["motion"]["history_frames"])' "$FULL_JOINT_RESUME")"
fi
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$FULL_ROOT/configs/real_motion_occfm.yaml" \
  "$FULL_ROOT/data/p0_f9_v18_se2_train_full.pt" "$FULL_ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  "$FULL_ROOT/data/p0_f9_v21_dev64_manifest.json" "$FULL_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
EXTRA=()
GEOMETRY_CACHE="${FULL_JOINT_GEOMETRY_CACHE_DIR:-$FULL_ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1}"
EXTRA+=(--causal-geometry-cache "$GEOMETRY_CACHE" --causal-cache-gib "${FULL_JOINT_GEOMETRY_CACHE_GIB:-48}"
  --causal-cache-ram-mib "${FULL_JOINT_GEOMETRY_CACHE_RAM_MIB:-4096}")
if [[ -n "${FULL_JOINT_RESUME:-}" ]]; then
  if [[ ! -f "$FULL_JOINT_RESUME" ]]; then echo "[MISSING] $FULL_JOINT_RESUME" >&2; exit 2; fi
  EXTRA+=(--resume "$FULL_JOINT_RESUME")
fi
if [[ "${FULL_JOINT_PAIRED_CONTROL:-0}" == 1 ]]; then EXTRA+=(--paired-control); fi
if [[ "${FULL_JOINT_PREWARM:-0}" == 1 ]]; then EXTRA+=(--prewarm-causal-cache); fi
if [[ "${FULL_JOINT_REFERENCE_CPU:-0}" == 1 ]]; then EXTRA+=(--reference-cpu-pipeline); fi
EXTRA+=(--profile-every "${FULL_JOINT_PROFILE_EVERY:-0}")
EXTRA+=(--sampling-workers "${FULL_JOINT_SAMPLING_WORKERS:-0}" --io-workers "${FULL_JOINT_IO_WORKERS:-2}")
if [[ "${FULL_JOINT_PERSISTENT_POOL:-1}" == 1 ]]; then EXTRA+=(--persistent-sampling-pool); fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FULL_ROOT:$FULL_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import torch,sys; print(sys.executable,torch.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 unavailable"'
"$PY" -m pytest -q tests/test_joint_causal_columns_full.py tests/test_column_runtime_pipeline.py tests/test_causal_geometry_cache.py
mkdir -p "$RUN_DIR"
if [[ -n "${FULL_JOINT_RESUME:-}" ]]; then
  echo "断点恢复一阶段：$FULL_JOINT_RESUME；原总轮数$EPOCHS，恢复optimizer/RNG/整段余弦，跳过prior。"
else
  echo "随机初始化一阶段：每轮全部20430窗口，共$EPOCHS轮；整个周期统一余弦下降，无tail阶段。"
fi
echo "window batch<=$WINDOWS / source budget=$SOURCES；每轮dev64，最后dev512；不跑full4369。"
echo "历史occupancy=$HISTORY帧（新训练默认严格4帧）；未来6帧。旧6帧断点只续跑旧协议，不能视为4帧实验。"
echo "默认不同时训练V18-only对照；E14只作评估参考；不会覆盖旧实验。输出：$RUN_DIR"
echo "固定因果几何lazy缓存：$GEOMETRY_CACHE；上限${FULL_JOINT_GEOMETRY_CACHE_GIB:-48}GiB；cold Strong仍走GPU，异步写盘，绝不缓存learned poses/labels/features。"
"$PY" -u tools/real_motion/train_p0_f9_joint_causal_columns_full.py \
  --config "$FULL_ROOT/configs/real_motion_occfm.yaml" \
  --train-cache "$FULL_ROOT/data/p0_f9_v18_se2_train_full.pt" \
  --dev-cache "$FULL_ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$FULL_ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$FULL_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --epochs "$EPOCHS" --history-frames "$HISTORY" --window-batch-size "$WINDOWS" --source-budget "$SOURCES" \
  --cpu-workers "$WORKERS" --eval-batch-size 256 --frame-cache-mib 256 \
  --out-dir "$RUN_DIR/model" "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/run.log"
echo "完成。发回 $RUN_DIR/model/summary.txt、epoch_history.json 和 progress.jsonl。"
echo "last.pt用于断点恢复；epoch_*.pt仅权重快照；candidate.pt是最终固定TRAIN阈值模型。"
