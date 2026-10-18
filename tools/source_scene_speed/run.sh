#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${SCENE_SPEED_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1, "ONE visible GPU, no torchrun"'
PILOT="${SCENE_PILOT_DIR:-}"
if [[ -z "$PILOT" ]]; then
  mapfile -t candidates < <(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 1 -type d -name 'scene_reader_stage1_1ep_*' | sort)
  complete=()
  for d in "${candidates[@]}"; do
    [[ -f "$d/baseline/results.json" && -f "$d/reader/results.json" && -f "$d/comparison.json" ]] && complete+=("$d")
  done
  if [[ "${#complete[@]}" != 1 ]]; then
    printf 'Set SCENE_PILOT_DIR to the completed paired run. Found %s candidates:\n' "${#complete[@]}"
    printf '%s\n' "${complete[@]}"
    exit 1
  fi
  PILOT="${complete[0]}"
fi
export SCENE_SPEED_DIR="${SCENE_SPEED_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_reader_speed_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo "READ ONLY parameters + original training logs + same64 H6 FPS. PILOT=$PILOT OUT=$SCENE_SPEED_DIR"
exec "$PY" -u -m tools.source_scene_speed.run --root "$PILOT" --out "$SCENE_SPEED_DIR" \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATA" \
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" \
  --windows "${SCENE_SPEED_WINDOWS:-64}" --repeats 2 --cpu-workers 4 --modality both "$@"
