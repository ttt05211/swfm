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
INTEGRATED="$("$PY" -c 'import sys; from tools.source_scene_integrated_speed.report import select_integrated; print(select_integrated(sys.argv[1],sys.argv[2] or None))' "$ROOT/outputs/p0_f9_shared_history" "${INTEGRATED_DIR:-}")"
CONTROL="$("$PY" -c 'import sys; from tools.source_scene_integrated.report import select_reference; print(select_reference(sys.argv[1],sys.argv[2] or None))' "$ROOT/outputs/p0_f9_shared_history" "${SCENE_KTA_CONTROL_DIR:-}")"
export SCENE_RESOURCE_DIR="${SCENE_RESOURCE_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_integrated_speed_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo "READ ONLY matched KTA-query/integrated; GT+STC H6; ONE GPU/model process; CONTROL=$CONTROL INTEGRATED=$INTEGRATED OUT=$SCENE_RESOURCE_DIR"
exec "$PY" -u -m tools.source_scene_integrated_speed.run \
  --control-dir "$CONTROL" --integrated-dir "$INTEGRATED" --out "$SCENE_RESOURCE_DIR" \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATA" \
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" \
  --windows "${SCENE_RESOURCE_WINDOWS:-32}" --repeats 2 --cpu-workers 4 --modality both "$@"
