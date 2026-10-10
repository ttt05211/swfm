#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# == 0 ]] || { echo '使用 EGO_CONTROL_* 环境变量' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
SOURCE="${EGO_CONTROL_SOURCE:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_20261010_231306_partial_screen_three_epochs}"
OUT="${EGO_CONTROL_OUT:-${SOURCE}_kinematic}"
POP="${EGO_CONTROL_POPULATION:-dev64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
launch() {
  "$PY" - "$SOURCE" "$@" <<'PY'
import json, os, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1])/'training.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('SWFM_')}
env.update(c['runtime_environment'])
os.execvpe(sys.executable,[sys.executable,'-u',*sys.argv[2:]],env)
PY
}
TRAIN=(--source-dir "$SOURCE" --out-dir "$OUT")
if [[ -f "$OUT/last.pt" ]]; then TRAIN+=(--resume); fi
echo "复用原10129窗口，不补缓存。两种小控制头各3轮；固定epoch3；WM/CCR冻结。输出：$OUT"
launch tools/ego_experiments/train_ego_kinematic.py "${TRAIN[@]}"
EVAL_OUT="${OUT}_eval_${POP}"
EVAL=(--training-dir "$OUT" --out-dir "$EVAL_OUT" --population "$POP"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
  --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${EGO_CONTROL_CPU_WORKERS:-4}")
if [[ -f "$EVAL_OUT/state.json" ]]; then EVAL+=(--resume); fi
launch tools/ego_experiments/eval_ego_kinematic.py "${EVAL[@]}"
echo "结果：$EVAL_OUT/summary.txt"
