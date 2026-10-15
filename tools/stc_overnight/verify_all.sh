#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${STC_VERIFY_GPU:-0}"
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
if [[ -z "${STC_REPLAY_BUNDLE:-}" ]]; then
  STC_REPLAY_BUNDLE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_overnight_bundle_*/stc_overnight_bundle.zip' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$STC_REPLAY_BUNDLE" ]] || { echo 'Set STC_REPLAY_BUNDLE to original exported ZIP'; exit 1; }
if [[ "${STC_VERIFY_RESUME:-0}" == 1 && -z "${STC_VERIFY_DIR:-}" ]]; then echo 'Set original STC_VERIFY_DIR'; exit 1; fi
STC_VERIFY_DIR="${STC_VERIFY_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_kta_recency_ALL_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--bundle "$STC_REPLAY_BUNDLE" --out "$STC_VERIFY_DIR" --model "${STC_VERIFY_MODEL:-clean1000}"
  --population all --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATAROOT"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${STC_VERIFY_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "FROZEN SAME-model baseline vs KTA+recency; ALL4369; no training. OUT=$STC_VERIFY_DIR"
exec "$PY" -u -m tools.stc_overnight.verify_all "${args[@]}" "$@"
