#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '用 SURFACE_MEAN_FPS_WINDOWS / SURFACE_MEAN_FPS_REPEATS 配置。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUN="${SURFACE_COMPARE_RUN:-$ROOT/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787}"
OUT="${SURFACE_MEAN_FPS_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/mean_fps_$(date +%Y%m%d_%H%M%S)_$$}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
EXTRA=()
[[ -z "${SURFACE_MEAN_SOURCE:-}" ]] || EXTRA+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE")
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '仅测已选平均模型FPS：默认256窗口×3次；场景内随机、场景均衡，不按耗时挑窗口。'
echo '固定4历史→6未来，ADD raw0.5/REMOVEoff；Strong和六帧输出实时计算，不读取预测缓存。'
echo "输出 $OUT；普通/加速执行逐字节核对；不重训、不跑全集精度。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/benchmark_p0_f9_joint_surface_mean_fps.py \
  --run-dir "$RUN" --runs-root "$ROOT/outputs/p0_f9_joint_surface_ccr" --out-dir "$OUT" \
  --config "$ROOT/configs/real_motion_occfm.yaml" --base-checkpoint "$E14" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATAROOT" \
  --windows "${SURFACE_MEAN_FPS_WINDOWS:-256}" --repeats "${SURFACE_MEAN_FPS_REPEATS:-3}" \
  --cpu-workers 10 "${EXTRA[@]}" 2>&1 | tee "$OUT.log"
echo "结果 $OUT/summary.txt；每窗口/分段/显存见 speed.json。"
