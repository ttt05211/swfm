#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
CACHE="${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
JSON="${STC_PLANNER_JSON:-}"
if [[ -z "$JSON" ]]; then
  for candidate in "$ROOT/data/come_protocol/bevplanner_ego_in_bev_with_yaw.json" \
    "$ROOT/../_received/bevplanner_ego_in_bev_with_yaw.json" \
    /root/nas/occ/new_code/data/come_protocol/bevplanner_ego_in_bev_with_yaw.json \
    /root/nas/occ/data/come_protocol/bevplanner_ego_in_bev_with_yaw.json; do
    if [[ -f "$candidate" ]]; then JSON="$candidate"; break; fi
  done
fi
if [[ -z "$JSON" || ! -f "$JSON" ]]; then
  echo '缺原始JSON（2,651,124 bytes）。本地 _received/bevplanner_ego_in_bev_with_yaw.json 已核对官方SHA。' >&2
  echo "上传到 $ROOT/data/come_protocol/bevplanner_ego_in_bev_with_yaw.json，或设置 STC_PLANNER_JSON=实际路径。" >&2
  exit 2
fi
OUT="${STC_PLANNER_AUDIT_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/planner_origin_all_$(date +%Y%m%d_%H%M%S)_$$}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo '只读核验全部planner缓存窗口及完整场景row0起点；不跑模型，不需STC/GT占据，不调整预测。'
exec "$PY" -u tools/real_motion/audit_p0_f9_original_planner_t0.py \
  --dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}" \
  --plan-cache "$CACHE" --planner-json "$JSON" --out-dir "$OUT" "$@"
