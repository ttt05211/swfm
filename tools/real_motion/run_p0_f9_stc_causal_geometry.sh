#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUNS="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
OUT="${STC_CAUSAL_OUT:-$RUNS/stc_causal_dev64_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
      --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
      --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
      --population dev64 --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
      --config "$ROOT/configs/real_motion_occfm.yaml" --out-dir "$OUT"
      --cpu-workers "${STC_CPU_WORKERS:-4}" --parallel-majority)
if [[ "${STC_CAUSAL_RESUME:-0}" == 1 ]]; then
  [[ -n "${STC_CAUSAL_OUT:-}" ]] || { echo '续评指定原 STC_CAUSAL_OUT，不是旧四设置目录' >&2; exit 2; }
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
echo '固定dev64一趟：四基线+历史路面位姿补偿+STC保守地面列稳定+组合；不改网络/权重、不训练。'
echo '动态速度只诊断；未来GT仅评分/误差统计；无aligned、mask调整、调阈值或自动全集。'
echo "Output: $OUT"
CONTRACT=""
if [[ "${STC_CAUSAL_RESUME:-0}" == 1 ]]; then CONTRACT="$OUT/contract.json"; fi
# Same-screen resume in a new terminal restores its frozen execution flags.
# Do not reuse a different experiment's prefix and never evaluate shell text.
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
os.execve(sys.executable, [sys.executable, '-u', root+'/tools/real_motion/eval_p0_f9_stc_causal_geometry.py',
                          *sys.argv[3:]], environment)
PY
