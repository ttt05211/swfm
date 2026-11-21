#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
STEP="${1:-help}"; if [[ $# -gt 0 ]]; then shift; fi
WORK="${GAST_WORK:-$TASK_ROOT/outputs/gast_baseline_gt}"
GAST_DIR="${GAST_REPO:-$TASK_ROOT/../GAST}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
GAST_ENV_NAME="${GAST_ENV:-gast}"
UPSTREAM=c4687255132e580468e1007168c57b00ce36b76a
export PYTHONPATH="$TASK_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# All preparation and rendering steps are permitted on a CPU-only instance.
export CUDA_VISIBLE_DEVICES=""
gp() { "$CONDA_BIN" run --no-capture-output -n "$GAST_ENV_NAME" python "$@"; }
dp() { gp "$@"; }
need_conda() { [[ -n "$CONDA_BIN" ]] || { echo 'conda not found; initialize the existing miniconda installation first'; exit 1; }; }
data_args=(--work "$WORK" --dataroot "${GAST_DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}")
if [[ -n "${GAST_ASSETS:-}" ]]; then data_args+=(--assets "$GAST_ASSETS"); fi
if [[ -n "${GAST_INFO:-}" ]]; then data_args+=(--info "$GAST_INFO"); fi
echo "GAST_STEP=$STEP; WORK=$WORK; official=$GAST_DIR"
case "$STEP" in
  code)
    if [[ ! -d "$GAST_DIR/.git" ]]; then
      if [[ -e "$GAST_DIR" ]]; then echo 'Existing non-Git GAST directory; set GAST_REPO to a NEW path'; exit 1; fi
      git clone "${GAST_CLONE_URL:-https://ghfast.top/https://github.com/chenst27/GAST.git}" "$GAST_DIR"
    fi
    if [[ -n "$(git -C "$GAST_DIR" status --porcelain --untracked-files=no)" ]]; then echo 'Official checkout has edits; refusing to overwrite them'; exit 1; fi
    git -C "$GAST_DIR" checkout "$UPSTREAM"
    echo "GAST_CODE_READY $UPSTREAM; official sources unchanged"
    ;;
  env)
    need_conda
    if ! "$CONDA_BIN" run -n "$GAST_ENV_NAME" /bin/true >/dev/null 2>&1; then
      "$CONDA_BIN" create -n "$GAST_ENV_NAME" python=3.10 -y
    fi
    gp -c 'import pathlib,sys; assert sys.version_info[:2]==(3,10) and pathlib.Path(sys.prefix).name==sys.argv[1], "Existing environment differs; choose a NEW GAST_ENV name. No environment was removed."' "$GAST_ENV_NAME"
    gp -m pip install --no-cache-dir setuptools==69.5.1 wheel==0.43.0 numpy==1.23.5 opencv-python==4.8.1.78
    gp -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    gp -m pip install --no-cache-dir --only-binary=:all: torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    gp -m pip install --no-cache-dir --only-binary=mmcv-full mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
    gp -m pip install --no-cache-dir -r tools/gast_baseline/requirements.txt
    gp -m pip check
    gp -c 'import torch,mmcv,torch_scatter,pyvista; print("GAST_ENV_CPU_OK",torch.__version__,torch.version.cuda,mmcv.__version__,pyvista.__version__,"NO GPU required")'
    ;;
  weights|adopt-weights|adopt-info)
    need_conda; gp -u -m tools.gast_baseline.download "$STEP" --work "$WORK" "$@"
    ;;
  infos)
    need_conda
    set +e
    dp -u -m tools.gast_baseline.data probe "${data_args[@]}"
    result=$?
    set -e
    if [[ "$result" == 2 ]]; then
      echo 'Current selected metadata lacks required ego fields; downloading ONLY the official VAL info.'
      gp -u -m tools.gast_baseline.download infos --work "$WORK"
      dp -u -m tools.gast_baseline.data probe "${data_args[@]}"
    elif [[ "$result" != 0 ]]; then exit "$result"; fi
    ;;
  data)
    need_conda; dp -u -m tools.gast_baseline.data prepare "${data_args[@]}"
    ;;
  cpucheck)
    need_conda; gp -u -m tools.gast_baseline.cpu_check --work "$WORK" --gast-root "$GAST_DIR"
    ;;
  check|infer)
    need_conda
    export CUDA_VISIBLE_DEVICES="${GAST_GPU:-0}"
    gp -u -m tools.gast_baseline.export "$STEP" --work "$WORK" --gast-root "$GAST_DIR"
    ;;
  render-libs)
    if ! command -v apt-get >/dev/null; then echo 'Install system EGL/OSMesa software-rendering libraries with the host package manager'; exit 1; fi
    if [[ "$(id -u)" != 0 ]]; then echo 'Run this step as root: apt-get update && apt-get install -y libegl1 libgl1 libopengl0 libosmesa6 libgl1-mesa-dri'; exit 1; fi
    apt-get update
    apt-get install -y libegl1 libgl1 libopengl0 libosmesa6 libgl1-mesa-dri
    ;;
  render-check|render)
    need_conda
    export LIBGL_ALWAYS_SOFTWARE=1 PYVISTA_OFF_SCREEN=true
    export VTK_DEFAULT_OPENGL_WINDOW="${GAST_VTK_BACKEND:-vtkOSOpenGLRenderWindow}"
    if [[ "$STEP" == render-check ]]; then gp -u -m tools.gast_baseline.render --check --out "$WORK/rendered";
    else
      gp -u -m tools.gast_baseline.render --check --out "$WORK/rendered"
      gp -u -m tools.gast_baseline.render --assets "$WORK/comparison_assets.zip" --out "$WORK/rendered" "$@"
    fi
    ;;
  status)
    need_conda; dp -u -m tools.gast_baseline.status --work "$WORK"
    ;;
  *)
    echo 'CPU phase: code -> env -> weights -> infos -> data -> cpucheck; optional render-libs -> render-check'
    echo 'GPU phase (ONE GPU): check -> infer. Then CPU: render. Reconnect: status; resume: SAME infer command.'
    echo 'Optional env: GAST_WORK, GAST_REPO, GAST_ASSETS, GAST_INFO, GAST_DATAROOT, GAST_GPU.'
    ;;
esac
