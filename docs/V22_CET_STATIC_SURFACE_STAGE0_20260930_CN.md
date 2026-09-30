# V22 CET Stage-0：静态交界面上限与零训练基线

状态：只实现无需训练的 dev64 Stage-0。方法名称统一为 **Causal Emergence Tokens (CET)**；主分支是 Surface Emergence Token，未来 Entity Emergence Token 只保留为辅助方向，本实验不生成动态物体。

## 这次一次性回答的问题

对冻结 V18 的 1/2/3s 输出，只在“未来查询网格进入、历史网格尚未覆盖”的交界外侧构造窄走廊，且永远不覆盖 V18 已占用体素。同一遍 dev64 数据同时比较：

- 类别 `CORE_ROAD_SIDEWALK={11,13}`；
- 类别 `EXTENDED_GROUND={11,12,13,14}`；
- 宽度 0.8 / 1.6 / 3.2m；
- `SCOPE_GT`：交界走廊内的逐 horizon 精确 GT 静态表面，是职责范围上限；
- `CAUSAL_GT`：还要求走廊点附近存在真实历史地表锚点，隔离 causal support 损失；
- `NEAREST_COLUMN`：完全不用未来 GT，复制最近历史地表列的类别、高度与厚度；
- `TANGENT_PLANE`：完全不用未来 GT，在最近历史列基础上继续局部地表切平面。

GT 只进入两个明确标记的 upper bound 和指标计算。两个 deterministic baseline 只读取六帧历史 Occ3D、历史/未来 ego pose 与冻结 V18 输出。

## 运行

先同步新分支：

```bash
cd /root/nas/occ/swfm

git fetch \
  https://ghfast.top/https://github.com/ttt05211/swfm.git \
  feature/v22-causal-emergence-tokens

if git show-ref --verify --quiet refs/heads/feature/v22-causal-emergence-tokens; then
  git switch feature/v22-causal-emergence-tokens
  git merge --ff-only FETCH_HEAD
else
  git switch -c feature/v22-causal-emergence-tokens FETCH_HEAD
fi

git rev-parse HEAD
```

激活环境并设置已确认路径：

```bash
conda activate OccFM
export PY="$(command -v python)"
export ROOT=/root/nas/occ/swfm
export RUNTIME_CONFIG=$ROOT/configs/real_motion_occfm.yaml
export DEV_V18=$ROOT/data/p0_f9_v18_se2_val_all_4369.pt
export V20_STAGE1_DEV512=$ROOT/data/p0_f9_v20_stage1_v2_dev512_v19split_769edc52
export V21_DEV64_MANIFEST=$ROOT/data/p0_f9_v21_dev64_manifest.json
export BASE_CKPT=$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt
export BASE_CKPT_SHA256="$(sha256sum "$BASE_CKPT" | awk '{print $1}')"
export DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
export DEV_INFO=$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl
export CET_RUN_ROOT=$ROOT/outputs/p0_f9_v22_cet_surface_stage0
export CET_JSON=$CET_RUN_ROOT/dev64_all_scopes_widths.json
export CET_LOG=$CET_RUN_ROOT/dev64_all_scopes_widths.log
mkdir -p "$CET_RUN_ROOT"
```

若 dev64 manifest 已存在，不重建；缺失时才从冻结 dev512 identity/order 生成：

```bash
if [ ! -f "$V21_DEV64_MANIFEST" ]; then
  "$PY" -u tools/real_motion/build_p0_f9_v21_dev_manifest.py \
    --stage1-cache "$V20_STAGE1_DEV512" \
    --output "$V21_DEV64_MANIFEST" \
    --count 64
fi
```

一条命令跑完 2 个类别集合 × 3 个宽度 × 4 层表示，共享 raw load、V18 forward、静态历史对齐与 Moving support：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/eval_p0_f9_v22_cet_surface_stage0.py \
  --config "$RUNTIME_CONFIG" \
  --val-cache "$DEV_V18" \
  --population-manifest "$V21_DEV64_MANIFEST" \
  --checkpoint "$BASE_CKPT" \
  --expected-checkpoint-sha256 "$BASE_CKPT_SHA256" \
  --dataroot "$DATAROOT" \
  --info-pkl "$DEV_INFO" \
  --cpu-workers 8 \
  --moving-workers 8 \
  --output "$CET_JSON" \
  2>&1 | tee "$CET_LOG"
```

结果不需要临时 heredoc，直接使用仓库内汇总器：

```bash
"$PY" -u tools/real_motion/summarize_p0_f9_v22_cet_surface_stage0.py \
  --input "$CET_JSON"
```

## 判定

dev64 只允许选择一次类别集合、宽度和 deterministic baseline。机械门槛为：

1. 同配置 `SCOPE_GT ΔmIoU >= +0.50 pp`；
2. `CAUSAL_GT / SCOPE_GT >= 70%`；
3. 至少一个完全因果、无需训练的基线 `ΔmIoU > 0`；
4. 该基线 addition occupancy precision 至少 60%；
5. scene-level 正增益场景数不少于负增益场景数。

全部通过才冻结最佳 surface token support 并进入小规模 learned token；上限不足则停止；上限存在但 deterministic baseline 不足时，只做 dev64 learned surface-token probe，不直接扩大到 dev512/full。

报告同时保存每个 horizon/per-class 指标、scope/causal positive BEV 比例、target voxel retention、addition precision/recall、scene delta、完整 exactness 与分阶段耗时。`SCOPE_GT` 必须满足：所有新增体素均为 GT occupied、语义全部正确、所有可添加 scope target 全部恢复，否则 evaluator 直接报错。
