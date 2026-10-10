#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUNS="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
OUT="${STC_YAW_OUT:-$RUNS/stc_yaw_dev64_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
      --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
      --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
      --population dev64 --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
      --config "$ROOT/configs/real_motion_occfm.yaml" --out-dir "$OUT"
      --cpu-workers "${STC_CPU_WORKERS:-4}" --parallel-majority)
if [[ "${STC_YAW_RESUME:-0}" == 1 ]]; then
  [[ -n "${STC_YAW_OUT:-}" ]] || { echo '续评指定本次原 STC_YAW_OUT，不是旧四设置/geometry目录' >&2; exit 2; }
  ARGS+=(--resume)
fi
if [[ -n "${STC_CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$STC_CHECKPOINT")
else
  ARGS+=(--runs-root "$RUNS" --run-dir "${SURFACE_RUN_DIR:-$RUNS/full20_nohup_resume_20261008_190410_787}")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ "${STC_NO_GRAPHS:-0}" == 1 ]]; then ARGS+=(--no-graphs); fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '固定dev64：Occ+Pred和STC+Pred各比较原版/yaw一致性；仅改可靠yaw，不改XY/z/tilt。'
echo '不训练、不aligned、不重跑GT设置或失败路面方案、不自动全集。'
echo "Output: $OUT"
CONTRACT=""
if [[ "${STC_YAW_RESUME:-0}" == 1 ]]; then CONTRACT="$OUT/contract.json"; fi
exec "$PY" - "$CONTRACT" "$ROOT" "${ARGS[@]}" "$@" <<'PY'
import json, os, re, sys
contract, root = sys.argv[1:3]
environment = os.environ.copy()
if contract:
    with open(contract, encoding='utf-8') as handle:
        recorded = json.load(handle)['runtime_environment']
    if not isinstance(recorded, dict) or any(
        not re.fullmatch(r'SWFM_[A-Za-z0-9_]+', k) or not isinstance(v, str) or '\0' in v
        for k, v in recorded.items()):
        raise RuntimeError('invalid frozen SWFM environment')
    environment = {k:v for k,v in environment.items() if not k.startswith('SWFM_')}
    environment.update(recorded)
os.execve(sys.executable, [sys.executable, '-u', root+'/tools/real_motion/eval_p0_f9_stc_trajectory_yaw.py',
                          *sys.argv[3:]], environment)
PY
