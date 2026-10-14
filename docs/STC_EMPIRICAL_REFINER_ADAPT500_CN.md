# STC 统计校准历史污染：冻结主体 / Refiner 500 步对照

## 实验决定

不降低 log-prior 强度，不新增置信度网络或阈值。两路均从正式 GT epoch2（完整 TRAIN20430×2、update10292）初始化，冻结 Shared Encoder / STWM，仅用全六时距 CE＋Lovász 微调现有 SemanticRefiner。

| 项目 | clean | empirical |
|---|---|---|
| 主体初始化、数据顺序、分组、优化器 | 同 GT epoch2；新 Adam/RNG | 相同 |
| TRAIN 历史有效性 | dense-valid | dense-valid |
| 历史污染 | 无 | 固定随机选择约 50% window；仅历史四帧 |
| 未来条件/目标 | GT ego / 干净 GT occupancy | 相同 |
| 学习参数 | Refiner（包括其特征 lift / evidence pooling） | 相同 |
| 预算 | 固定 500 更新，原 TRAIN20%=4086 的部分遍历 | 相同 |
| LR / schedule | 1e-4，500 步整周期 cosine，floor .01 | 相同 |
| 先验 | c=1、原 epsilon=1e-4 | 相同 |

这里两路 dense-valid 是隔离污染效果的决定：若沿用 GT mask_lidar，Encoder 会看不到大量 unknown 区域的模拟假占据。它不是给 STC 使用 GT mask。评估 OCC 仍走原 lidar-known，STC 走发布完整语义 / dense-valid，评分两者均完整网格，不套可见性 mask。旧正式 GT 模型不改。

没有 ego 头、command、planner、motion / existence loss、动作 ADD/REMOVE 头或 prior gate。冻结只约束参数，不代表可以复用旧输入的几何：改变历史后，CC、匹配、KTA、注册、source patch、Strong、prior、memory 全部重建。干净窗口复用兼容历史几何，只重建有效性相关表示；没有新持久几何 / 学习特征 bank。

## 模拟器如何使用实际统计

输入是已完成真实噪声拟合的 `noise_model.json` 及绑定 `evaluation.json`，确认源 GT checkpoint SHA 和 18/132 scene 拆分。只将经验分布用于 TRAIN 历史，不使用真实 VAL 网格作为训练输入或目标。

- 类别 × 水平距离的 GT→STC 混淆，区分漏占据、假占据、错类。零支持单元不臆造概率。
- 假占据集中在原历史占据边界附近，按拟合边界距离直方图分配，不能在空中均匀撒点。
- 采用组件 **count×voxel size** 加权抽样的联合 XYZ extent 来选相关场尺度，避免把组件数中位数误当体素质量中位数。
- 四帧错误用拟合条件 bit pattern；静态通过历史 ego 配准对应，已匹配动态 source 通过纯历史注册跟随物体，未匹配 / 出界位置独立抽样。
- 动态 annotation-crop 的“同类 STC 无支持”只借作逐体素时间模式，不据此按 39.7% 整车删除；没有另加一遍无法解耦的人工速度噪声。
- 不修改未来 ego 条件、未来 occupancy 目标，不向生成器传未来字段。

这仍是**经验相关随机场近似**，不是 STCOcc 前端的精确模拟：全窗尺度抽样 / Gaussian copula 不保证真实连通域数量、组件形状、动态抖动或联合噪声完全复现；薄表面、有限空闲候选、跨帧语义变化和概率裁剪会改变边际。故 `progress.jsonl` 记录实际各 family 体素数、距离分母 / 错误量、clipping、CPU simulation / geometry 时间、实际与原分组 source 数，不声称严格匹配分布。

## 评估与科研边界

不是旧 DEV64/DEV512：从 noise profile 的 remaining132 scenes 中，按固定 scene-balanced round-robin 选 64 个窗口，两路完全相同；缺 STC 帧直接失败，不补别的窗口。报告 OCC+GT、STC+GT 的 Transport / SemanticRefiner、Refiner 增益和 empirical-clean 差。

这是 **VAL-calibrated augmentation 开发实验**。拟合使用同帧 VAL GT，且之前全 VAL 分数已被查看；remaining scenes 只是没有用于这次噪声拟合，不是全新独立测试。不能把本次收益直接标为无目标域信息 zero-shot，不能隐瞒 VAL 校准。若之后进入正式主表，需要披露或改用 TRAIN 前端预测统计；不自动推广、扩大训练或扫参数。

## 服务器直接执行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export CALIB_RUN_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_$(date +%Y%m%d_%H%M%S)"
unset CALIB_RESUME
# 若 NOISE_FIT_DIR 仍是刚跑完的统计目录，显式绑定；否则自动找最近 stc_noise_fit_*/noise_model.json。
if [[ -n "${NOISE_FIT_DIR:-}" ]]; then export NOISE_MODEL="$NOISE_FIT_DIR/noise_model.json"; fi
bash tools/stc_noise_adaptation/run.sh && cat "$CALIB_RUN_DIR/summary.txt"
```

一次依次完成两路各 500 更新及各自剩余场景 64 窗 OCC/STC 评估。没有先跑另一趟首检；可选 `--max-updates 8` 仅用于自愿 smoke，保留原 500 步 cosine。共用 driver 的 RANDOM/epoch1 启动文字不代表随机权重或完整 epoch，最终以新 contract / summary / initialization audit 为准。

Ctrl-C 只在完整更新 / 评估窗边界保存；同目录续跑，已 complete 的 arm 不重训：

```bash
export CALIB_RUN_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际这次目录"
export CALIB_RESUME=1
bash tools/stc_noise_adaptation/run.sh && cat "$CALIB_RUN_DIR/summary.txt"
```

保持 NOISE_MODEL / GT_CKPT 与首次相同，不混用旧 ROBUST_RESUME / RUN_DIR。新输出保存 profile 快照、每路 GT 快照 / 初始化 SHA、Adam/RNG checkpoint 和冻结核验；原 checkpoint / optimizer / RNG / cache 不改。阶段时间是质量/训练耗时，不是 FPS。

## 本地验证

Windows 遵守 windows-python-env-guard，非 base yoloe + 既有测试依赖。真实小网格 CPU/CUDA 覆盖固定随机污染 / 未来字段不可访问、跨帧持续与动态注册对应、未知标签 / 零支持处理、坏 profile 绑定拒绝、污染后 source/Strong 重建、冻结参数严格不变而 Refiner 更新；实际共用 trainer 的新目录、500 步 schedule、Adam/RNG 同目录恢复及 hooks 还原。另测两路流程 / 固定 eval keys / snapshot / 已完成 arm 不重跑。

本地新实现及旧历史重建 / 有效性 / GT trainer 合计 35 项通过（18.18s）；旧 CPU-only 噪声统计另起独立进程 6 项通过（0.64s），因为它检查进程未导入 Torch，不能与 Torch 测试混跑。Bash 语法通过。未运行真实服务器训练。

本地小网格不证明真实 STC 质量或 L40S 吞吐。真实收益以本次服务器 matched 结果为准。
