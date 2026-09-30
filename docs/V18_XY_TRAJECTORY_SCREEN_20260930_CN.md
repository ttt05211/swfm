# V18：冻结证据上的 XY 轨迹小规模训练

## 目的与边界

用户实际 dev64 motion-gap 结果：GT XY + 预测 yaw 比 V18 提升 **5.233286 pp mIoU / 25.877208 pp MovingMicro**，18 个场景全部正向；仅替换 GT yaw 的提升是 0.170498 pp mIoU。XY 平均误差随预测距离增加：1/2/3s 约 0.496/1.043/1.645m。这是投入 XY 学习的证据，**不是本版能得到相同涨点的保证**。

本版直接训练可部署的 XY 预测头，不重复 oracle、不训练生成模块、不改变 source 集合。固定 Clean-E14 的 encoder、已有时空 Transformer、source 几何/pivot/Z、yaw、existence、Strong SE(2) renderer、A1 CLEAR/WRITE 和 frozen Moving 协议。旧 checkpoint 不修改，也不复制到新小头 checkpoint。

V18 原本已有未来 query 自注意力，因此本版不把“新增时序注意力”作为新颖性。它是一次明确、可否证的 **kinematic-conditioned joint trajectory residual** 小规模实验。也不重复已失败的长尾重采样、手工类别权重、anti-regret/safety/router。

## 一次运行中的三个受控训练组

| 组 | 输入与输出 | 用途 |
|---|---|---|
| `linear_xy_refit` | 原来的每时刻 query → 线性 XY 修正 | 判断是否只需继续拟合线性头 |
| `joint_xy_position` | 六个 query 联合 + 历史 context/kinematics/KTA/base XY → 六时刻位置修正 | 控制新增容量和历史输入的收益 |
| `integrated_xy_trajectory` | 与联合位置头相同输入、参数量、初始权重 → 六段速度修正 → 积分至位置 | 判断速度积分的轨迹参数化是否额外有效 |

三个头最后一层均为零初始化：`XY_new = actual_frozen_XY + delta_XY`。不是把原 BF16 线性层改用 FP32 重算，从而保证 update=0 数值一致。联合两组都采用 LayerNorm + 两层 128 维 GELU MLP。修正用 FP32；编码使用与部署相同的逐窗口 BF16 V18 forward。

积分组是 `delta_p[h] = 0.5 * sum(delta_v[0:h+1])`，t0 修正为零。该映射可逆，**不强制恒速、不限制转弯/加速，也不裁剪修正幅度**。输入中的缺失历史 offsets/velocities 用历史 validity 清零。监督 mask、未来 GT occupancy/annotation、目标类别、GT yaw 和 source 的监督资格均不是头的输入。

数据流：历史 occupancy/causal motion → 冻结 V18 encoder/时空 Transformer → 冻结 query/context + causal kinematics → 小 XY 头 → 原 observed-source SE(2) renderer/A1 → 六帧 occupancy。推理对完整 source population 生效，不用 GT-valid 或监督资格筛除难预测 source；不改变 yaw/existence。

## 监督和固定实验预算

沿用 V18 原始 source-centred residual target、0.8m source footprint。只有两个仍能影响 XY 的原损失组：`SmoothL1(beta=1)` + `0.25 * soft SE(2) overlap loss`。后者使用固定的预测 yaw 与训练标签 GT yaw。冻结 existence/yaw 的损失对新参数为常数，不重复相加；不增加 velocity/acceleration 新 loss，也不加类别权重。

`screen` 固定：512 个 TRAIN windows、32 个 TRAIN-only calibration windows、原冻结 dev64。TRAIN 校准集按场景 round-robin 选择，**整个校准场景从优化集移除**；所有 TRAIN 场景与冻结 dev512 parent scenes 不重叠。manifest、顺序、种子、bank 内容 fingerprint、基线 SHA256 和运行 config fingerprint 都写入合同。

每组 1024 个真实 optimizer updates；默认 batch=128 **sources，不是 windows**。三组每一步用同一批均匀抽样 source、AdamW(lr=3e-4, weight_decay=.01)、clip norm=5。不以类别、误差或历史长度重采样。保留有至少一个有效原监督标签的 source，标签缺失不引入 NaN geometry 梯度。

每 256 步在 TRAIN 校准集上计算六帧 ADE 与 3s FDE，按 `0.5*(ADE+FDE3s)` 选择；没有 3s 有效标签时使用 ADE。保留 update=0 作为真正的零修改参照。模型选择结束后，**仅一次 dev64 pass**，同时渲染三组 selected 和 last。last 只是诊断，不能根据 dev 结果选它来部署。

`smoke` 是 4/2/2 windows、每组 8 updates 的接口验证，永远不会被标为有效模块。常规一键命令直接运行 screen，先通过本地/服务器合成预检查，不要求另跑旧 oracle 或 smoke。

## 验收与停止

合成与单元测试验证零初始化、完整 source、yaw/existence 不变、mask/NaN、真实 loss/backward、TRAIN 场景隔离、真实 A1 renderer 和完整 CLI/摘要/小头 checkpoint。dev 第一窗口检查 frozen ccf7d77 forward、原 Strong/A1 exactness、零 XY 修正逐 voxel 一致、changed-cell counts 与全网格计数一致。

每个 selected candidate 的 screen 门槛在看本版 dev 之前固定：

- 整体和每个报告时刻 1/2/3s，IoU/mIoU/MovingMacro/MovingMicro 均不得下降（数值容差 1e-9）。
- 整体 ΔmIoU ≥ **0.30 pp** 且 ΔMovingMicro ≥ **1.00 pp**。
- selected update 必须 >0；仅在 `screen` 中才可过门槛。

摘要同时显示 TRAIN/calibration 拟合、dev XY ADE/FDE/p90、history-valid=1/6 分层、三时刻 occupancy 涨跌、场景增益和耗时。轨迹标签误差降低不等于 renderer 后指标提高，两者分别验收。

三组都失败就 `stop_xy_version_no_automatic_retry`，不自动再试参数或扩大数据。任何组通过也只标记 `screen_pass_review_before_expansion`，不是完整 dev/test 上已验证有效。三个算法共享 dev64 是探索性比较，不是独立测试；不能据此宣称论文结论或估计显著性。

部署入口 `load_xy_adapter` 默认拒绝失败、last、smoke、update=0 或错误 base/config 的 checkpoint。`forecast_with_xy` 读取 `include_gt=False` 的历史输入，以原 renderer 返回完整六时刻预测。不设置 learned confidence gate 或 all-reject 回退来伪装收益。

## 速度和存储

V18 在 512+32 个窗口上只运行一次，预提取紧凑 causal query/context。三组共享同一 bank；训练不读 nuScenes 原始文件，不重复 V18 encoder/时空 Transformer，不渲染 dense voxel。bank 上限每个 256MiB，训练 bank 驻留 GPU，仅索引 batch；CPU 只保留紧凑数组，无新增磁盘 cache。

dev 的 raw load、V18 forward、history/source 准备、Moving support 只做一趟，六个输出共享。保存六份小 XY 头 best/last，不保存六个完整 V18 网络或 dense prediction。每 16 个 bank 窗口、每 64 个训练 updates 和每个 dev 窗口打印进度；分阶段真实耗时写入 summary，不根据低 GPU 利用率虚构提速倍数。

## 服务器执行

已有路径完全沿用用户确认的 `/root/nas/occ/swfm`、OccFM、Clean-E14、TRAIN/DEV V18、dev64 manifest、nuScenes 和 `configs/real_motion_occfm.yaml`。无需 Stage-1、prototype bank 或新的独立 dev512 cache。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_v18_xy_trajectory_screen.sh screen
```

新输出默认 `/root/nas/occ/swfm/outputs/p0_f9_v18_xy_trajectory/screen_<时间>_<commit>/model`。发回该目录 `summary.txt` 即可；完整 strata/原始统计在 `summary.json`。可用 `V18_XY_OUT` 指定**尚不存在的**输出目录；`V18_XY_CPU_WORKERS` 默认 8，`V18_XY_BATCH_SIZE` 默认 128。路径/环境/GPU 缺失明确失败，不静默使用 CPU 或前 64 条。

## 实验材料记录

按 academic-research-suite 的固定实验边界实施：输入为用户的真实 motion-gap 摘要、冻结 V18 代码和缓存合同；产物为三个实际训练头、受控脚本、可审计 checkpoint、摘要与测试。真实 nuScenes/GPU 训练仍待服务器执行，不能把本地合成拟合或单元测试记为实际涨点。windows-python-env-guard 用于本地测试，禁止调用故障的 Anaconda base；不改变服务器 OccFM 环境。

本地验收（2026-09-30）：新增 18 个测试；全仓库非 integration 测试 **614 passed / 4 skipped / 1 deselected**，439 个 Python 文件 AST 解析、Bash 语法及 diff whitespace 检查通过。未执行远端 CI 或真实 GPU 训练；既有空指标/Transformer/GradScaler warning 不代表本版出现新的报错。
