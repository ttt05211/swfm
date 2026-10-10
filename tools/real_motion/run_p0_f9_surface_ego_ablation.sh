#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
SOURCE="${EGO_AB_SOURCE:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_head_screen_20261010_194309_837}"
OUT="${EGO_AB_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/ego_ablation_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ $# != 0 ]]; then echo '使用 EGO_AB_* 环境变量；不转发可能改变训练/评估路径的额外参数。' >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
TRAIN=(--source-dir "$SOURCE" --out-dir "$OUT")
if [[ "${EGO_AB_RESUME:-0}" == 1 ]]; then
  [[ -n "${EGO_AB_OUT:-}" ]] || { echo '恢复必须指定原 EGO_AB_OUT，不生成新目录' >&2; exit 2; }
  TRAIN+=(--resume)
fi
echo '同一只读TRAIN1024 bank；不重提特征，不重训WM/CCR。A/B各2000步≈125轮，全周期余弦。'
echo '原损失A vs 固定10m参考点几何损失B；同原初始化/样本顺序；整轮均值+完整TRAIN误差。'
echo "原实验：$SOURCE；新输出：$OUT"
"$PY" -u tools/real_motion/train_p0_f9_surface_ego_ablation.py "${TRAIN[@]}"
if [[ "${EGO_AB_SKIP_EVAL:-0}" == 1 ]]; then exit 0; fi
EVAL_OUT="${OUT}_eval_dev64"
EVAL=(--pair-dir "$OUT" --out-dir "$EVAL_OUT"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
  --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${EGO_AB_CPU_WORKERS:-4}")
if [[ "${EGO_AB_RESUME:-0}" == 1 && -f "$EVAL_OUT/state.json" ]]; then EVAL+=(--resume); fi
echo '固定dev64一趟12路：OCC/STC × GT/external/prior/old320/A/B。历史共享；未来几何各自重算。'
# Restore original SWFM flags ONLY in the evaluation child, before importing the
# prediction modules. No shell eval, no changes to the caller's environment.
exec "$PY" - "$SOURCE" "${EVAL[@]}" <<'PY'
import json, os, sys
from pathlib import Path
training = json.loads((Path(sys.argv[1]) / 'training.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('SWFM_')}
env.update(training['runtime_environment'])
os.execvpe(sys.executable, [sys.executable, '-u', 'tools/real_motion/eval_p0_f9_surface_ego_ablation.py', *sys.argv[2:]], env)
PY
