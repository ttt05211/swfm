#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
for ARG in "$@"; do
  case "$ARG" in
    --help|-h) exec "$PY" tools/real_motion/train_p0_f9_surface_ego_head.py --help ;;
    --out-dir|--out-dir=*|--resume)
      echo '启动脚本使用 EGO_HEAD_OUT / EGO_HEAD_RESUME 指定目录和恢复，避免训练与评估路径不一致。' >&2
      exit 2 ;;
  esac
done
RUNS="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
OUT="${EGO_HEAD_OUT:-$RUNS/ego_head_screen_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
      --train-cache "${TRAIN_V18:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
      --train-info "${TRAIN_INFO:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl}"
      --config "$ROOT/configs/real_motion_occfm.yaml" --out-dir "$OUT"
      --train-windows "${EGO_TRAIN_WINDOWS:-1024}" --epochs "${EGO_EPOCHS:-20}"
      --batch-size "${EGO_BATCH_SIZE:-64}" --cpu-workers "${EGO_CPU_WORKERS:-4}")
if [[ -n "${STC_CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$STC_CHECKPOINT")
else
  ARGS+=(--runs-root "$RUNS" --run-dir "${SURFACE_RUN_DIR:-$RUNS/full20_nohup_resume_20261008_190410_787}")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ "${EGO_HEAD_RESUME:-0}" == 1 ]]; then
  [[ -n "${EGO_HEAD_OUT:-}" ]] || { echo '续训必须指定原 EGO_HEAD_OUT，不创建新目录' >&2; exit 2; }
  ARGS+=(--resume)
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '仅训练轻量 ego 头：TRAIN1024默认20轮，现有冻结mean5/6/8/12/14的WM+CCR不重训。'
echo '共享四历史物体/静态编码 + 历史ego；六未来GT派生导航指令，非纯历史导航自由设置。'
echo "输出：$OUT；Ctrl+C/SIGTERM在完成更新后保存head_last.pt；--resume保持LR/Adam/RNG。"
"$PY" -u tools/real_motion/train_p0_f9_surface_ego_head.py "${ARGS[@]}" "$@"
if [[ "${EGO_SKIP_EVAL:-0}" == 1 ]]; then exit 0; fi
EVAL_OUT="${EGO_EVAL_OUT:-${OUT}_eval_dev64}"
EVAL_ARGS=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --plan-cache "${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
  --head-checkpoint "$OUT/head_last.pt" --out-dir "$EVAL_OUT"
  --population dev64 --population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --cpu-workers "${EGO_CPU_WORKERS:-4}")
if [[ "${EGO_EVAL_RESUME:-0}" == 1 ]]; then
  [[ -n "${EGO_EVAL_OUT:-}" ]] || { echo '续评指定原 EGO_EVAL_OUT' >&2; exit 2; }
  EVAL_ARGS+=(--resume)
fi
echo '固定dev64一趟对比OCC/STC各自的GT / 外部planner / 内部ego；不aligned，不调阈值。'
exec "$PY" -u tools/real_motion/eval_p0_f9_surface_ego_head.py "${EVAL_ARGS[@]}"
