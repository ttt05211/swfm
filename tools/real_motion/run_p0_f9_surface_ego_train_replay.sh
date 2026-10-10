#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# == 0 ]] || { echo '使用 EGO_REPLAY_* 环境变量，不转发额外参数' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
PAIR="${EGO_REPLAY_PAIR:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_ablation_20261010_214909_837}"
OUT="${EGO_REPLAY_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_train_replay_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--pair-dir "$PAIR" --out-dir "$OUT"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --windows 64 --cpu-workers "${EGO_REPLAY_CPU_WORKERS:-4}")
if [[ "${EGO_REPLAY_RESUME:-0}" == 1 ]]; then
  [[ -n "${EGO_REPLAY_OUT:-}" ]] || { echo '恢复必须指定原 EGO_REPLAY_OUT' >&2; exit 2; }
  ARGS+=(--resume)
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '只读：既有TRAIN1024 bank+old320/A/B；完整cached TRAIN复核+固定64窗历史实时重提。'
echo '缓存/训练式/评估式特征、标签/cmd、batch64/单窗、train/eval模式；不训练，不读未来occupancy。'
echo "A/B来源：$PAIR；新输出：$OUT"
# Restore original feature execution flags BEFORE importing project modules.
exec "$PY" - "$PAIR" "${ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1]) / 'contract.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('SWFM_')}
env.update(c['source']['original_training']['runtime_environment'])
os.execvpe(sys.executable, [sys.executable, '-u', 'tools/ego_diagnostics/surface_train_replay.py', *sys.argv[2:]], env)
PY
