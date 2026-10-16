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
SOURCE="${STC_VERIFY_SOURCE:-${STC_VERIFY_DIR:-}}"
if [[ -z "$SOURCE" ]]; then
  SOURCE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_verify_all_*/contract.json' -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$SOURCE/state.json" ]] || { echo 'Set STC_VERIFY_SOURCE to existing serial evaluation directory'; exit 1; }
if [[ -z "${STC_REPLAY_BUNDLE:-}" ]]; then
  STC_REPLAY_BUNDLE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_overnight_bundle_*/stc_overnight_bundle.zip' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$STC_REPLAY_BUNDLE" ]] || { echo 'Set STC_REPLAY_BUNDLE to original exported ZIP'; exit 1; }
if [[ "${STC_PARALLEL_RESUME:-0}" == 1 && -z "${STC_PARALLEL_DIR:-}" ]]; then echo 'Set original STC_PARALLEL_DIR'; exit 1; fi
OUT="${STC_PARALLEL_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_verify_parallel_$(date +%Y%m%d_%H%M%S)}"
args=(--source-dir "$SOURCE" --out-dir "$OUT" --bundle "$STC_REPLAY_BUNDLE"
  --horizon-group "${STC_VERIFY_HORIZON_GROUP:-6}" --cpu-workers "${STC_VERIFY_CPU_WORKERS:-2}"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${STC_PARALLEL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "ONE GPU / TWO processes; import existing complete windows; H6; SOURCE=$SOURCE; OUT=$OUT"
echo 'With --stop-source, SIGINT is sent ONLY to the original module with this exact output directory.'
exec "$PY" -u -m tools.stc_overnight_parallel.run "${args[@]}" "$@"
