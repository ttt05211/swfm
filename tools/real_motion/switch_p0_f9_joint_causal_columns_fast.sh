#!/usr/bin/env bash
# Preserve progress before replacing either b700519 or a signal-safe cache run.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM" >&2; exit 2; fi
if [[ $# != 2 || ! "$2" =~ ^[0-9]+$ || "$2" -le 1 ]]; then echo "用法: bash $0 OLD_RUN_DIR PYTHON_PID 或 --graceful-from-pid PYTHON_PID" >&2; exit 2; fi
ROOT=/root/nas/occ/swfm
cd "$ROOT"
OLD_PID="$2"; GRACEFUL=0
read_args() {
  [[ -r "/proc/$OLD_PID/cmdline" ]] || return 1
  mapfile -d '' -t OLD_ARGS < "/proc/$OLD_PID/cmdline"
}
arg_value() {
  local key="$1" i
  for ((i=0; i<${#OLD_ARGS[@]}-1; i++)); do
    if [[ "${OLD_ARGS[i]}" == "$key" ]]; then printf '%s' "${OLD_ARGS[i+1]}"; return; fi
  done
  return 1
}
if [[ "$1" == --graceful-from-pid ]]; then
  read_args || { echo "PID不存在，未停止/启动任何进程" >&2; exit 2; }
  OLD_MODEL="$(arg_value --out-dir)"
  [[ "$OLD_MODEL" == */model ]] || { echo "PID输出不是model目录，拒绝停止" >&2; exit 2; }
  # The causal-cache launcher is newer than the installed graceful signal
  # handler. Never assume b700519 supports a final checkpoint after SIGTERM.
  arg_value --causal-geometry-cache >/dev/null || { echo "不是signal-safe cache运行，改用旧目录+PID模式" >&2; exit 2; }
  OLD_RUN="$(realpath -e -- "${OLD_MODEL%/model}")"; GRACEFUL=1
else
  OLD_RUN="$(realpath -e -- "$1")"
fi
MODEL="$OLD_RUN/model"
case "$OLD_RUN" in "$ROOT"/outputs/p0_f9_joint_causal_columns/*) ;; *) echo "拒绝非joint实验路径" >&2; exit 2 ;; esac
[[ -f "$MODEL/last.pt" ]] || { echo "[MISSING] $MODEL/last.pt" >&2; exit 2; }
verify_target() {
  read_args || return 1
  local script=0 arg
  for arg in "${OLD_ARGS[@]}"; do
    [[ "${arg##*/}" == train_p0_f9_joint_causal_columns_full.py ]] && script=1
  done
  [[ "$script" == 1 && "$(arg_value --out-dir)" == "$MODEL" ]]
}
verify_target || { echo "PID与指定full训练目录不符，拒绝停止任何进程" >&2; exit 2; }
EPOCHS="$(arg_value --epochs)"; WINDOWS="$(arg_value --window-batch-size)"; SOURCES="$(arg_value --source-budget)"
case "$EPOCHS" in 15|20) ;; *) echo "原epochs不是15/20，拒绝切换" >&2; exit 2 ;; esac
# This launcher reproduces the documented recipe; refuse custom seed/config/data.
for pair in \
  "--config|$ROOT/configs/real_motion_occfm.yaml" \
  "--train-cache|$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  "--dev-cache|$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  "--population-manifest|$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "--base-checkpoint|$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  "--dataroot|/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes" \
  "--train-info|/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl" \
  "--dev-info|/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ "$(arg_value "${pair%%|*}")" == "${pair#*|}" ]] || { echo "原population/config不是标准recipe，拒绝停止" >&2; exit 2; }
done
[[ "$(arg_value --seed || printf 20261002)" == 20261002 ]] || { echo "custom seed，拒绝停止" >&2; exit 2; }
PAIRED=0
for arg in "${OLD_ARGS[@]}"; do if [[ "$arg" == --paired-control ]]; then PAIRED=1; fi; done
CACHE_DIR="$(arg_value --causal-geometry-cache || printf '%s' "$ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1")"
LOG="$ROOT/outputs/p0_f9_joint_causal_columns/launch_fast_resume_$(date +%Y%m%d_%H%M%S)_$$.log"
NEW_RUN="$ROOT/outputs/p0_f9_joint_causal_columns/full${EPOCHS}_fast_resume_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)_$$"
[[ ! -e "$NEW_RUN" ]] || { echo "新输出已存在，未停止旧进程" >&2; exit 2; }
OLD_STAMP="$(stat -c '%i:%Y:%s' -- "$MODEL/last.pt")"
if [[ "$GRACEFUL" == 0 ]]; then
  echo "等待旧训练下一次atomic last.pt保存；通常最多约256批。不会删除旧目录。"
  deadline=$((SECONDS+1800))
  while [[ "$(stat -c '%i:%Y:%s' -- "$MODEL/last.pt")" == "$OLD_STAMP" ]]; do
    verify_target || { echo "旧进程已结束或身份改变；没有发送信号，请手动检查last.pt" >&2; exit 2; }
    if (( SECONDS >= deadline )); then echo "等待保存超时，未停止旧训练" >&2; exit 2; fi
    sleep 1
  done
else
  echo "已核验signal-safe运行：$MODEL；发送TERM请求当前batch结束后保存，无需等待256批。"
fi
verify_target || { echo "保存后进程身份不符，拒绝发送信号" >&2; exit 2; }
kill -TERM "$OLD_PID"
deadline=$((SECONDS+120))
while kill -0 "$OLD_PID" 2>/dev/null; do
  [[ "$(awk '/^State:/ {print $2}' "/proc/$OLD_PID/status" 2>/dev/null || true)" == Z ]] && break
  if (( SECONDS >= deadline )); then echo "旧进程未退出，未启动第二个训练进程" >&2; exit 2; fi
  sleep 1
done
if [[ "$GRACEFUL" == 1 && "$(stat -c '%i:%Y:%s' -- "$MODEL/last.pt")" == "$OLD_STAMP" ]]; then
  echo "未观察到停止时新atomic checkpoint；拒绝自动恢复，请检查旧日志。旧目录/缓存均保留。" >&2; exit 2
fi
FULL_JOINT_OUT="$NEW_RUN" FULL_JOINT_RESUME="$MODEL/last.pt" FULL_JOINT_WINDOW_BATCH="$WINDOWS" FULL_JOINT_SOURCE_BUDGET="$SOURCES" \
  FULL_JOINT_PAIRED_CONTROL="$PAIRED" FULL_JOINT_GEOMETRY_CACHE_DIR="$CACHE_DIR" \
  nohup bash tools/real_motion/run_p0_f9_joint_causal_columns_full.sh "$EPOCHS" > "$LOG" 2>&1 &
echo "新训练 PID=$!；恢复旧模型/optimizer/RNG/cosine，不重做prior。"
echo "日志: $LOG"
echo "查看: tail -f '$LOG'"
