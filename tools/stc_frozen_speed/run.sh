#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${STC_FPS_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch,os; assert "OccFM" in sys.executable; assert "," not in os.environ["CUDA_VISIBLE_DEVICES"]; assert torch.cuda.device_count()==1'
if [[ -z "${STC_REPLAY_BUNDLE:-}" ]]; then
  STC_REPLAY_BUNDLE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_overnight_bundle_*/stc_overnight_bundle.zip' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "${STC_REPLAY_BUNDLE:-}" ]] || { echo 'Set STC_REPLAY_BUNDLE to existing exported ZIP'; exit 1; }
VALIDATED="${NORM_VALIDATED_DIR:-${NORM_WORKFLOW_DIR:+$NORM_WORKFLOW_DIR/all}}"
if [[ -z "$VALIDATED" ]]; then
  VALIDATED="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 3 -path '*/stc_norm192_final_*/all/results.json' -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$VALIDATED/results.json" ]] || { echo 'Set NORM_VALIDATED_DIR to completed norm192 ALL evaluation directory'; exit 1; }
OUT="${STC_FPS_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_norm192_frozen_fps128_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo "Freeze verified norm192, SINGLE GPU/process,128 fixed windows; no full VAL rerun. OUT=$OUT"
args=(--out "$OUT" --bundle "$STC_REPLAY_BUNDLE" --validated-run "$VALIDATED"
  --windows 128 --repeats "${STC_FPS_REPEATS:-2}" --horizon-group 6 --cpu-workers "${STC_FPS_CPU_WORKERS:-2}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" --dev-info "${DEV_INFO:-$DATA/nuscenes_infos_val_temporal_v3_scene.pkl}"
  --dataroot "$DATA" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
exec "$PY" -u -m tools.stc_frozen_speed.run "${args[@]}" "$@"
