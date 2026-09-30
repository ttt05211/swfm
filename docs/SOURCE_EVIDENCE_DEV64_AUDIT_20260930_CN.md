# Source evidence：先验证历史证据，再决定下一版方法

本次仅实现一次 dev64 诊断，不训练、不生成未见物体、不构建 AE/prototype、
不重新生成数据缓存。目的不是保证涨点，而是分清 V18 的运动、已有 source 的
历史形状和静态记忆分别还剩多少可利用空间。

## 冻结协议

- Clean-E14 `epoch_0014.pt`，检查 clean-tail 协议、arm、epoch 和 SHA256。
  SHA 在运行前从用户确认的 checkpoint 路径获取并写入结果；不是另一个科学基线认证。
- 使用已有 V21 dev64 manifest 的唯一 identity/order，从 V18 full4369 严格取样。
  缺失 manifest 时，才从已确认的 Stage-1 dev512 冻结同一 scene-balanced population。
  缺失/重复/指纹不符直接失败，不取 full4369 前 64 个。
- 指标仍为原始 1/2/3 秒 horizon-first IoU/mIoU/Moving 指标，不改 class/support。
- 无干预 renderer 与 V18 每个窗口逐 voxel 相等；首个窗口额外验证冻结 forward、
  Strong/A1 exactness 及所有 variant 的增量计数与全量计数一致。
- 输入配置是 tracked `configs/real_motion_occfm.yaml`，不是缺少 METRIC 的 V20 frozen YAML。

## 一次运行中的比较

| 组 | 干预 | 能回答什么 |
| --- | --- | --- |
| V18_BASE | 冻结 V18 | 唯一比较基线 |
| T0_GT_MOTION | 原 t0 shape + 有效缓存 GT source-centered SE(2) label；完整 A1 CLEAR/WRITE | 运动误差还剩多少；不是只补正确 voxel |
| HISTORY_GT_ALIGN_* | GT 历史身份和历史→t0 对齐，过去真实 observed shape；分别 GT/预测 motion | 若能正确对齐，过去形状是否有帮助 |
| HISTORY_CAUSAL_ALIGN_* | 无 GT 的历史关联和 XY ICP；分别 GT/预测 motion | 因果历史证据能否转化为收益 |
| HISTORY_COMMON_* | 同一组 GT-certified 正确 identity、同一组 ICP accepted 的历史 observation；只改变对齐 | 控制关联/population 后比较几何对齐误差；这组是 GT-conditioned diagnostic |
| HISTORY_GT_SELECT_* | 从因果候选中按整个历史 observation 选一个或弃权，每个 source/horizon 单独选择 | 候选选择是否比全部合并更有希望；保留候选全部错误 voxel |
| STATIC_MEMORY | 所有静态类别历史 memory 直接补入 V18 free | 不限路面窄边界，真实因果静态记忆的增益 |
| STATIC_PATCH_GT_SELECT | 按固定 4×4 BEV cells（1.6m×1.6m）整 patch 选/弃权，包含所有 Z/静态类别 | 静态候选选择空间 |
| *_VOXEL_GT_FILTER | 逐 voxel 用未来语义筛掉错误，只作为理想 precision 对照 | 容易被夸大的 hindsight 数字，不进入训练路线判断 |

历史 shape 只来自过去五帧输入的真实 Strong source occupancy；不读未来 shape。
与冻结 V18 一致，动态 source 从 supplied Occ3D 历史语义提取，不额外做 mask_lidar
筛除。这不等于每个 voxel 都被 lidar 直接观测：结果分别报告历史/t0 source 总
voxel 数与 lidar-observed 数。静态 memory 则沿用 V19 的 observed/clearing 协议。
GT shape alignment 用过去/t0 annotation 的 identity、center、yaw，不改变 world Z。
GT motion 使用缓存中的 source-center SE(2) 标签，不直接拿 box center 代替 source center。
无有效位移/yaw 标签时保留 V18 对应预测，并计数；无 GT survival/existence gate。

因果关联仅用同类 centroid、历史 CV、25m/s 物理位移及残差 gate；近邻距离差
小于 0.4m 时 fail closed，一对一匹配。ICP 为确定性 XY trimmed registration：
每份最多 512 点、5 次迭代、1.6m correspondence、至少 6 对/50% inliers、最多
45° yaw；pedestrian 不拟合 yaw。它是首轮确定性基线，不是可学习方法的最优值。

历史附加几何始终 add-only、保护 occupied，按冻结 current source 顺序处理 collision。
历史-only/dormant source 不渲染，仅报告历史 annotation track 数；本实验不宣称测试了
全部 source induction。已有 occupied voxel 的 shape 删除/纠错也不在历史增补干预内。
所以历史增补的负结果不能单独否定所有 source reconstruction 方法。

整候选/patch 的 GT-assisted utility 为新加 voxel 中 `正确语义数−错误语义数`，
并非 mIoU 最优化，也不是全局 oracle；不能称严格上限、不能保证学习后仍有收益。
COMMON 的对比避免把匹配 population 差异错误归因于 alignment；整体 GT/causal
两组仍同时包含身份关联、接受率和配准差异。

## 报告与资源控制

最终 JSON 包含每 horizon/class 指标、相对 V18 的增益、scene 正负数、FP/语义错误/
static 与 dynamic FN、整候选误补/损坏/删除统计、匹配/接受/身份验证/OOB 计数及性能。
`dynamic_fn_inside_t0_gt_motion_occupied_support` 只是 GT-motion occupied union 内的 FN，
不是 ancestry 归因，更不是新增 source 数。

记录 Git SHA、两份实现源码 SHA、配置 SHA、checkpoint SHA 和 population fingerprint。
每窗口 flush progress JSONL，并显示 raw-load/history 阶段，不长时间静默。
读取原始数据和 V18 推理只走一趟（首窗口 exactness 会额外 forward）；历史提取、
registration、静态投影及 Moving support 使用至多 8 workers；每个 BLAS 单线程。
复用同窗口/common 候选 raster 和基线 metric counts，只对 changed voxel 计算增量。
输出仅 JSON/JSONL/文本，不落 dense tensor、AE 或新的 checkpoint。GPU 大多空闲
可能是几何计算的正常现象，最终依阶段 timing 判断，不以显存占用判断提速。

`resource_triage` 的 0.5/1.0pp 是预先声明的资源分流启发式，不是论文接受门槛。
不会自动开训练，GT-assisted/GT-filter rows 也不能充当真实模型效果。
所有调参/选路目前是 dev64 exploratory；确定方案后应冻结再验证 dev512。

## 服务器一次运行

```bash
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
conda activate OccFM
bash tools/real_motion/run_p0_f9_source_evidence_dev64.sh
```

不要 source 此脚本；即使 preflight 失败也只退出子 bash，不关闭交互终端。
默认使用用户确认的服务器输入路径，输出新建时间戳目录，不覆盖旧实验。
缺少路径会明确打印 `[MISSING]` 后停止，不猜替代路径。
可用 `SOURCE_EVIDENCE_CPU_WORKERS=8` 控制 worker，`SOURCE_EVIDENCE_OUT` 指定新的输出目录。
运行完成后发回 `summary.txt`；详细审计保存在 `audit.json`，崩溃时还有 progress JSONL。

本地验证仅能证明代码与 synthetic contract；真实 nuScenes/GPU 数字须由服务器实际跑出。
本次本地：20 项 NumPy/stdlib 测试通过，6 项 Torch-dependent evaluator 测试因缺少
兼容的本地 Torch 环境跳过；`real_motion/tools/tests` 全量 Python 编译和 bash 语法
检查通过。服务器脚本会先执行同一组 26 项测试，再启动真实数据诊断。
