#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${NORM_FINAL_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch,os; assert "OccFM" in sys.executable; assert "," not in os.environ["CUDA_VISIBLE_DEVICES"]; assert torch.cuda.device_count()==1'
if [[ -z "${STC_REPLAY_BUNDLE:-}" ]]; then
  STC_REPLAY_BUNDLE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_overnight_bundle_*/stc_overnight_bundle.zip' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "${STC_REPLAY_BUNDLE:-}" ]] || { echo 'Set STC_REPLAY_BUNDLE to original exported ZIP'; exit 1; }
if [[ "${NORM_FINAL_RESUME:-0}" == 1 && -z "${NORM_FINAL_DIR:-}" ]]; then echo 'Set original NORM_FINAL_DIR'; exit 1; fi
OUT="${NORM_FINAL_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_norm192_all_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
ARGS=(--out "$OUT" --bundle "$STC_REPLAY_BUNDLE" --horizon-group 6 --cpu-workers "${NORM_FINAL_CPU_WORKERS:-2}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dev-info "${DEV_INFO:-$DATA/nuscenes_infos_val_temporal_v3_scene.pkl}" --dataroot "$DATA"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "$MODE" == probe ]]; then
  echo "Frozen 2/4 process speed probe on ONE GPU; OUT=$OUT; no fitting or formal score."
  exec "$PY" -u -m tools.stc_norm_final.probe "${ARGS[@]}" --windows "${NORM_PROBE_WINDOWS:-32}" "$@"
elif [[ "$MODE" != all ]]; then echo 'Mode must be all or probe'; exit 1; fi
WORKERS="${NORM_FINAL_WORKERS:-2}"
if [[ "$WORKERS" == auto ]]; then
  [[ -f "${NORM_SPEED_REPORT:-}/speed.json" ]] || { echo 'Set NORM_SPEED_REPORT to completed probe directory'; exit 1; }
  WORKERS="$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete"; assert r["recommended_processes"] in (2,4); print(r["recommended_processes"])' "$NORM_SPEED_REPORT/speed.json")"
fi
if [[ "${NORM_FINAL_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
echo "FROZEN norm192 + previous SAME-window control; ALL4369; GPU=$CUDA_VISIBLE_DEVICES; workers=$WORKERS; OUT=$OUT"
exec "$PY" -u -m tools.stc_norm_final.run "${ARGS[@]}" --processes "$WORKERS" "$@"
