#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
OUT="${LOCAL_REPLAY_OUT:-$ROOT/outputs/local_replay_$(date +%Y%m%d_%H%M%S)_$$}"
PILOT="${LOCAL_REPLAY_PILOT:-$ROOT/outputs/p0_f9_joint_causal_columns/source_repair_20261005_224502_885}"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo '只读导出16 TRAIN+16 dev；含高source压力样本；完整网格/4历史/6未来。'
echo 'CPU-only，不占GPU，不改缓存/旧checkpoint，不重训；日志尾部不是模型断点。'
"$PY" -u tools/real_motion/export_p0_f9_local_replay.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --checkpoint "$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --pilot-dir "$PILOT" --out-dir "$OUT" --cpu-workers 4
echo "只需下载：$OUT/replay.zip"
