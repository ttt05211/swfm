# Surface CCR 无损执行优化与干净联合训练（2026-10-08）

## 本轮改了什么

不再增加模块。保留现有 V18 Transport + Surface-consistent CCR：共享点编码，动态读出，CCR 内部表面描述与静态条件读出。候选/ownership/compositor/阈值不变。

执行优化有两处：

- 六帧 canonical projection 已经算过的 float64 静态相位、高度、目的体素，直接交给静态读出，避免再变换和离散一遍。只保留本次 forward 的临时数组，不写缓存，不缓存 learned poses/features。
- 历史表面 KD-tree 查询并行，复用缩放后的查询坐标。最多 4 个查询线程，小集合仍单线程；树的邻居顺序、表面描述算术和网络参数不变。

旧 namespace 的七个依赖文件未改，既有 TRAIN/VAL 缓存可复用；无新增大缓存，也不移动冻结 FPS 边界。六帧 live phase 仍计入预测延迟，历史表面表示准备时间仍单列，不宣称 raw-input E2E FPS。

本地 RTX3050 上已测实际 Surface head 概率和六帧 dense 字节一致，也测 CPU/CUDA 实际联合反传和 next-update 恢复。CPU 合成 60000 点、5 遍参考/优化交替的几何微基准：投影 109.01→86.56ms（约1.26×），历史表面描述 324.60→268.07ms（约1.21×）。这是几何微基准，**不是整网 FPS、服务器训练速度或质量验证**。最初保存过多中间数组的版本反而更慢，已舍弃。

## 先做一趟同窗口测速，不重跑 full

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_surface_ccr_execution_benchmark.sh
```

默认使用现有 epoch19 Transport / Frozen B epoch2 / Surface CCR epoch3 路径，缺文件就报准确路径，不重建。固定原 20 窗口×3，B、原 surface eager/预热 graph、投影复用 eager/预热 graph 同趟比较；全部概率和六帧 dense 逐字节核对，动态与 B 核对。概率不一致立即报错，不放宽容差。`summary.txt` 和 `speed.json` 在新的 `outputs/p0_f9_surface_ccr/execution_speed_*`。

边界保持 `CausalHistoryState → fresh Strong/KTA + live motion + learned CCR + live projection conditioning + SIX dense outputs`。历史表面准备独立给出参考/优化时间。测速不修改权重、阈值或旧缓存，不自动开启训练，不根据小样本自动升级部署后端。

## 干净联合训练配方

`train_p0_f9_joint_surface_ccr.py` 是新入口，不是冻结验证脚本改名：

- V18 运动网络和完整 Surface CCR **全部随机初始化、全部训练**。E14 只读取结构配置/执行预检，不加载它的初始化权重；没有 epoch19/B/旧 Local teacher，没有 KD/AE。
- 严格 4 历史→6 未来；每轮完整 TRAIN20430，默认 **20 轮全周期余弦**，floor=初始 LR 的 0.1，无固定 LR tail。batch4/source128，整窗口源过多时仍完整保留，不丢 source。
- 原 V18 motion loss + 现有等窗口、等角色 importance-corrected weighted ADD BCE；运动 LR5e-4，CCR LR3e-4；两组梯度范数各 clip5。训练时仍可用 future GT 做监督，但 GT 不进入模型输入或候选。
- TRAIN 前256窗口的完整未采样合法 support 在随机初始化时统计正权重，固定不重算、不用 DEV；raw sigmoid ADD≥0.5，REMOVE关闭。每轮 dev64，最后 dev512，固定最后轮，不 DEV-best/扫阈值/自动续训/full4369。
- renderer 的硬离散坐标用 detach 后的副本；CCR 的 live source context/future queries 保留计算图，loss 实际回传到运动网络。并行 CPU 作业不接触 RNG 或 autograd；采样主线程按记录顺序执行。
- 只读现有 TRAIN/VAL history geometry，在线 sampled feature/GT target/learned projection 重建；不保存它们。frame缓存每 split4GiB，history RAM TRAIN1GiB/VAL0.5GiB，表面描述 RAM256MiB，不按宿主机总内存无限扩张。

验证过的冻结静态结果不等于从头联合结果。先验来自 RANDOM 输出而非 B，动态/运动都会学习，因此新训练是新实验；不承诺精度不变或一轮仍十几分钟。日志有真实训练分段。

### 启动

```bash
bash tools/real_motion/run_p0_f9_joint_surface_ccr.sh
```

如果希望先看两轮，**仍按20轮完整余弦**，只在第二轮末安全停下：

```bash
JOINT_STOP_AFTER_EPOCH=2 bash tools/real_motion/run_p0_f9_joint_surface_ccr.sh
```

这不是把训练周期改成两轮。之后去掉 stop 参数恢复，LR 周期仍20轮。不要同时启动两份训练；wrapper 检查已存在训练进程，但不会 kill 它们。

### 中断与 resume

Ctrl+C 或 SIGTERM 在当前完整更新结束后写 `last.pt`；`kill -9`/断电只能恢复最近每128步的周期断点。每轮另留 `epoch_XXXX.pt`，不删前轮。原 checkpoint 不覆盖，resume 输出新目录。

```bash
JOINT_RESUME=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/实际训练目录/last.pt \
  bash tools/real_motion/run_p0_f9_joint_surface_ccr.sh
```

保存恢复：完整模型、两组 AdamW 状态、正权重、epoch/batch/update/窗口计数、Python/NumPy/sampling/Torch/CUDA RNG、已完成的每轮/最终评估。每轮评估前后都有阶段记录；未完成评估重做，已完成的不再重复。checkpoint 原子替换、保留 `last.previous.pt`；失败的半步绝不把污染后的权重配上旧 cursor 保存。

训练数据/顺序/总轮数/batch/source/LR/loss/关键实现改变将拒绝 resume，而非偷偷重置优化器。**旧 Local、Frozen B、冻结 Surface CCR checkpoint 均不能作为这个入口的 resume**。默认20轮，若初次明确 `JOINT_EPOCHS=15`，恢复也必须保持15，不能直接变20。

### 随时只读评估

先让训练安全保存，或显式选择一个固定 `epoch_XXXX.pt`；不要把正在替换的 `last.pt` 当成稳定实验标识。只读评估把载入的权重当快照，不改源文件/优化器/RNG，不写新的训练 `last.pt`：

```bash
JOINT_RESUME=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/实际训练目录/epoch_0002.pt \
JOINT_EVALUATE_ONLY=1 JOINT_EVAL_POPULATION=dev512 \
  bash tools/real_motion/run_p0_f9_joint_surface_ccr.sh
```

只有明确 `JOINT_EVAL_POPULATION=full4369` 才评估完整 VAL。各设置都输出 IoU/mIoU/MovingMacro/MovingMicro、分时距与编辑质量。当前评估直接复用 VAL 固定几何，不重新构建大缓存、不执行旧 Local 对照；和本次训练的 transport 比较。最终正式 FPS 尚需使用新训练权重实测，旧冻结候选的 FPS 不能冒充新模型结果。

## 验证范围

实际 CPU/CUDA 小网络联合更新、CCR→motion 梯度、next-update optimizer/RNG/LR 一致、并行采样同 RNG 同更新、empty support、TRAIN full counts/phase preflight、CLI 中断及最终评估恢复、失败半步不发布、只读 eval 原文件 SHA 不变均有测试。外部 nuScenes I/O/完整人口/FPS 的 CLI 测试使用 fixture，不冒充服务器真实数据验证；本地没有完整 nuScenes 缓存。
