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
SOURCE="${GT_FPS_SOURCE:-${STC_FPS_DIR:-}}"
if [[ -z "$SOURCE" ]]; then
  SOURCE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/shared_gt_stc_fps128_*/results.json' -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$SOURCE/frozen/final_model.pt" ]] || { echo 'Set GT_FPS_SOURCE to completed shared_gt_stc_fps128 directory'; exit 1; }
if [[ -z "${STC_REPLAY_BUNDLE:-}" ]]; then
  STC_REPLAY_BUNDLE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -path '*/stc_overnight_bundle_*/stc_overnight_bundle.zip' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "${STC_REPLAY_BUNDLE:-}" ]] || { echo 'Set STC_REPLAY_BUNDLE to existing exported ZIP'; exit 1; }
OUT="${GT_REUSE_DIR:-$ROOT/outputs/p0_f9_shared_history/gt_inference_reuse_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo "Frozen GT ONLY, same source model/keys/H6; baseline CPU2, optional Strong CPU4 labeled separately; no training/CUDA edits. SOURCE=$SOURCE OUT=$OUT"
args=(--source-fps "$SOURCE" --bundle "$STC_REPLAY_BUNDLE" --out "$OUT"
  --windows "${GT_REUSE_WINDOWS:-32}" --repeats "${GT_REUSE_REPEATS:-2}" --cpu-workers "${STC_FPS_CPU_WORKERS:-2}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dev-info "${DEV_INFO:-$DATA/nuscenes_infos_val_temporal_v3_scene.pkl}" --dataroot "$DATA")
exec "$PY" -u -m tools.gt_inference_reuse.run "${args[@]}" "$@"
