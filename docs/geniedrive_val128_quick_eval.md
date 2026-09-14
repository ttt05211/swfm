# GenieDrive val128 quick evaluation

This adapter evaluates the official GenieDrive occupancy checkpoint with the
frozen SWFM `Moving-mIoU v2` contract. It does not change SWFM training code,
prepared data, checkpoints, or existing outputs.

## Isolation and protocol

- External code, weights, annotation and outputs live under
  `/root/nas/occ/external_baselines/geniedrive_val128`.
- A separate `geniedrive-occ` conda environment contains the legacy OpenMMLab
  dependency stack.
- Existing nuScenes and Occ3D directories are linked through a new overlay;
  the original dataset directory is not modified.
- GenieDrive is stateful. The exporter uses its official single-GPU sequential
  sampler and traverses the complete validation set, while saving only the 128
  frozen SWFM sample tokens.
- The official checkpoint consumes the ground-truth future ego plan. This is
  permitted by the current OccFM-fut/SWFM protocol and is recorded in every
  prediction index. Do not describe this result as action-free forecasting.

## 1. Paths

```bash
export SWFM_ROOT=/root/nas/occ/swfm
export EXTERNAL_ROOT=/root/nas/occ/external_baselines/geniedrive_val128
export NUSCENES_ROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
export PREPARED_VAL=$SWFM_ROOT/data/prepared_val
```

`NUSCENES_ROOT` must directly contain `gts/` and `v1.0-trainval/`. Point it at
the already installed Occ3D-nuScenes tree; no new raw dataset is required.

## 2. Download from China mirrors

Install the small downloader in the current/base environment:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
bash $SWFM_ROOT/scripts/external_baselines/download_geniedrive_cn.sh
bash $SWFM_ROOT/scripts/external_baselines/link_geniedrive_data.sh
```

The GitHub script tries four China proxy endpoints. To select a local working
proxy explicitly:

```bash
export GITHUB_MIRROR_URL=https://gh-proxy.com/https://github.com/Huster-YZY/GenieDrive.git
```

Only `genie_occ.pth` and `world-nuscenes_infos_val.pkl` are downloaded from
Hugging Face. Video-generation checkpoints are not needed. The adapter pins and
checks GenieDrive revision `da48a529ffbe14136688e9b7a56f5d1061c366c5`.

## 3. Separate environment

```bash
conda env create -f $SWFM_ROOT/configs/external_baselines/geniedrive_occ_environment.yml
conda activate geniedrive-occ

# Required on managed images that globally expose a Python 3.12 torch build.
source $SWFM_ROOT/scripts/external_baselines/sanitize_geniedrive_environment.sh

pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  mmdet==2.28.2 mmsegmentation==0.30.0 mmengine
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r $EXTERNAL_ROOT/GenieDrive/occ_gen/requirements.txt
python -m pip install -v -e $EXTERNAL_ROOT/GenieDrive/occ_gen

python - <<'PY'
import torch, mmcv, mmdet
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('GPU', torch.cuda.get_device_name(0))
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__)
assert torch.cuda.is_available()
PY
```

If `import torch` mentions
`/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_python.so`, the
Python 3.8 environment is being contaminated by host-level path variables.
Source `sanitize_geniedrive_environment.sh` in the active shell and rerun
`python -m pip install -v -e ...`; the already installed packages do not need
to be reinstalled.

If the server image cannot load the CUDA 11.6 PyTorch build on L40S, use a
CUDA-compatible container for this environment only. Do not upgrade MMCV or
MMDetection inside the main SWFM environment.

## 4. Run and score

```bash
export SWFM_ENV=base
export GENIEDRIVE_ENV=geniedrive-occ
export GPU_ID=0
bash $SWFM_ROOT/scripts/external_baselines/run_geniedrive_val128.sh \
  2>&1 | tee $EXTERNAL_ROOT/results/run.log
```

The command fails unless all 128 selected nuScenes tokens are found. Main files:

```text
/root/nas/occ/external_baselines/geniedrive_val128/results/
├── val128_manifest.json
├── selection_audit.json      # all 128 tokens and Occ3D files found before inference
├── inference_summary.json
├── native_metric_sanity.json # must reproduce official 50.47/41.47/35.83 mIoU
├── predictions/
│   ├── index.json
│   └── geniedrive_*.pt       # 128 files, each [6,200,200,16] uint8
└── geniedrive_val128_moving_miou_v2.json
```

The exporter stops before Moving-mIoU scoring if the full-val native semantic
mIoU differs from the checkpoint's published in-repository reference by more
than 0.5 percentage point at any horizon. This catches a wrong checkpoint,
annotation, traversal order, or incompatible environment.

Inspect the final comparison:

```bash
python - <<'PY'
import json, os
p=os.path.join(os.environ['EXTERNAL_ROOT'],'results','geniedrive_val128_moving_miou_v2.json')
r=json.load(open(p))
print('GenieDrive Overall:', r['SWFM']['overall'])
print('GenieDrive Dynamic:', r['SWFM']['dynamic'])
print('GenieDrive Moving-mIoU v2:', r['SWFM']['Moving-mIoU_v2'])
print('Same-val128 KTA:', r['KTA_composed_baseline']['Moving-mIoU_v2'])
PY
```

`SWFM` in the JSON is the evaluator's historical prediction-slot name; for
this run it contains GenieDrive results, as confirmed by
`predictions/index.json` (`source: GenieDrive`).
