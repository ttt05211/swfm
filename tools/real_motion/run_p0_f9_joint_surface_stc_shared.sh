#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUNS="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
OUT="${STC_OUT:-$RUNS/stc_shared_${STC_POPULATION:-all}_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
      --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
      --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
      --out-dir "$OUT" --population "${STC_POPULATION:-all}"
      --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${STC_CPU_WORKERS:-4}"
      --frame-cache-mib "${STC_FRAME_CACHE_MIB:-512}" --geometry-cache-mib "${STC_GEOMETRY_CACHE_MIB:-512}"
      --speed-windows "${STC_SPEED_WINDOWS:-8}" --parallel-majority)
if [[ "${STC_POPULATION:-all}" != all ]]; then
  ARGS+=(--population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}")
fi
if [[ "${STC_RESUME:-0}" == 1 ]]; then
  [[ -n "${STC_OUT:-}" && -z "${STC_CONTINUE_FROM:-}" ]] || { echo '续评指定新版 STC_OUT，取消 STC_CONTINUE_FROM' >&2; exit 2; }
  ARGS+=(--resume)
elif [[ -n "${STC_CONTINUE_FROM:-}" ]]; then
  ARGS+=(--continue-from-dir "$STC_CONTINUE_FROM")
else
  ARGS+=(--runs-root "$RUNS" --run-dir "${SURFACE_RUN_DIR:-$RUNS/full20_nohup_resume_20261008_190410_787}")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ -n "${STC_CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$STC_CHECKPOINT"); fi
if [[ "${STC_NO_GRAPHS:-0}" == 1 ]]; then ARGS+=(--no-graphs); fi
if [[ "${STC_SPEED_ONLY:-0}" == 1 ]]; then ARGS+=(--speed-only); fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '共享同历史GT/Pred准备与表面证据；Strong/未来投影/输出分开；冻结权重、阈值、四设置同人口。'
echo '先逐字节校验+相同窗口AB/BA测速；确实更快才自动接续到新目录，原目录结果不变。'
echo "Output: $OUT"
CONTRACT=""
if [[ "${STC_RESUME:-0}" == 1 ]]; then CONTRACT="$OUT/contract.json"
elif [[ -n "${STC_CONTINUE_FROM:-}" ]]; then CONTRACT="$STC_CONTINUE_FROM/contract.json"; fi
# A new terminal need not reconstruct old SWFM execution flags by memory.
# No shell eval; only validated SWFM string variables from the frozen contract.
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
os.execve(sys.executable, [sys.executable, '-u', root+'/tools/real_motion/eval_p0_f9_joint_surface_stc_shared.py',
                          *sys.argv[3:]], environment)
PY
