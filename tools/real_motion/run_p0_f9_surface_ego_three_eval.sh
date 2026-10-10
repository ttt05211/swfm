#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# == 0 ]] || { echo '使用 EGO_THREE_* 环境变量' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
TRAINING="${EGO_THREE_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_20261010_231306_partial_screen_three_epochs}"
OUT="${EGO_THREE_EVAL_OUT:-${TRAINING}_eval_dev512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
ARGS=(tools/ego_experiments/eval_surface_ego_three_population.py
  --training-dir "$TRAINING" --out-dir "$OUT"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
  --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${EGO_THREE_CPU_WORKERS:-4}")
if [[ -f "$OUT/state.json" ]]; then ARGS+=(--resume); fi
echo "仅评估已有head_epoch3.pt，不训练、不读取训练bank。输出：$OUT"
"$PY" - "$TRAINING" "${ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1])/'training.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('SWFM_')}
env.update(c['runtime_environment'])
os.execvpe(sys.executable,[sys.executable,'-u',*sys.argv[2:]],env)
PY
echo "结果：$OUT/summary.txt"
