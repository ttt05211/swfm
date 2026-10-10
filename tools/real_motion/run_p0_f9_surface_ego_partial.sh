#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# == 0 ]] || { echo '使用 EGO_PARTIAL_* 环境变量' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
PARENT="${EGO_PARTIAL_PARENT:-}"
if [[ -z "$PARENT" ]]; then
  PARENT="$("$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / 'outputs/p0_f9_joint_surface_ccr'
candidates = []
for p in root.glob('*/training.json'):
    try: c = json.loads(p.read_text())
    except (OSError, ValueError): continue
    if (c.get('protocol') == 'surface_frozen_ego_full_one_epoch_geometry_v1'
            and 'partial_screen' not in c and (p.parent/'bank').is_dir()): candidates.append(p.parent)
if len(candidates) != 1:
    sys.exit('请显式 export EGO_PARTIAL_PARENT=原全量输出目录；找到 '+str(len(candidates))+' 个候选：'+str(candidates))
print(candidates[0])
PY
)"
fi
OUT="${EGO_PARTIAL_OUT:-${PARENT}_partial_screen}"
ARGS=(--parent-dir "$PARENT" --out-dir "$OUT" --max-windows "${EGO_PARTIAL_WINDOWS:-10240}")
if [[ "${EGO_PARTIAL_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
launch() {
  "$PY" - "$PARENT" "$@" <<'PY'
import json, os, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1])/'training.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('SWFM_')}
env.update(c['runtime_environment'])
os.execvpe(sys.executable,[sys.executable,'-u',*sys.argv[2:]],env)
PY
}
echo "只使用现有缓存（最多10240），不提取剩余特征。原任务先Ctrl-C停稳：$PARENT"
echo "新半量试训目录：$OUT；固定一轮，原全量断点/缓存/代码不变。"
launch tools/ego_experiments/screen_surface_ego_partial.py "${ARGS[@]}"
EVAL_OUT="${OUT}_eval_dev64"
EVAL=(--training-dir "$OUT" --out-dir "$EVAL_OUT"
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
  --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${EGO_PARTIAL_CPU_WORKERS:-4}")
if [[ "${EGO_PARTIAL_RESUME:-0}" == 1 && -f "$EVAL_OUT/state.json" ]]; then EVAL+=(--resume); fi
launch tools/ego_experiments/eval_surface_ego_full.py "${EVAL[@]}"
echo "半量结果：$EVAL_OUT/summary.txt；不自动续全量。"
