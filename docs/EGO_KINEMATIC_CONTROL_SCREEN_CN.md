# 内部 ego 头：运动学参数化集中试验（2026-10-11）

## 动机与边界

三轮旧头在 planner-covered DEV495 上仍逊于外部 BEV-Planner：OCC 平均
mIoU/IoU 18.040683/26.548403（外部 19.990703/28.578451），STC
11.293439/21.598709（外部 12.210279/22.622076）。OCC 3s 平均 XY
2.464773m，外部 1.383695m。保持正式 WM/CCR、GT 版本和外部 planner 不变。

旧代码的实际限制不是已证明的唯一失败原因：

- `ego_history[-1,5:7]` 是最后半秒的位移 secant，不是瞬时速度。恒加速度下
  `v_secant = v(t0) - a*dt/2`，直接外推具有系统滞后。
- 旧 decoder 虽有跨 query attention，但没有显式约束六张 XY 与 yaw 的运动学一致性。
- 原冻结物体/表面特征主要为物体搬运和补全学习，不能假定它们等同规划专用特征。
- 大样本三轮只比因果 prior 改善约4.8cm；不保证加轮数或增加容量能解决问题。

“预测少量控制量，再可微积分成轨迹”有既有研究依据，例如
[ITRA（2021）](https://arxiv.org/abs/2104.11212)。本实验只借鉴运动学归纳偏置；
不是复现该工作的 CVRNN/闭环生成，也不是把它宣称为新颖性。

## 新候选

仍只接受原七个 bank 字段和原六个离散导航命令，无 CAN bus、新地图/图像、
未来连续 GT 位姿或 occupancy。未来命令仍是 GT 派生导航条件，不能叫无导航预测。

1. 用实际历史时间，把三段位移 secant 按中点朝向旋回当前车体轴，修正转弯 chord，
   拟合当前速度与纵向加速度。相邻加速度方向/量级不一致则拒绝加速度外推，避免放大噪声。
2. 历史 yaw 拟合角速度；结合当前速度/加速度做常加速度、常角速度的历史运动学初值。
   这也是一个独立的 **无学习对照**，不将其收益归功于新网络。
3. 小 MLP 直接读取四历史 ego 状态＋六命令，预测三个连续控制 knot 的加速度/角速度修正。
   共享场景版再用单 query attention 读取冻结物体和表面 token。
4. 每0.5s内做8次中点积分，读出完整六个 XY/yaw。纵向速度刹停后不会自动倒车；
   倒车保留历史方向；静止起步使用最近历史方向，全静止默认当前 **ego 的 +X 车头轴**，
   不把 LiDAR 导航标签坐标轴混入运动积分。
   48步积分使用批量累计/反射运算而非逐步Python调用；测试与标量刹停再起步递推及梯度一致。

当前参数量：历史版8710；场景版34502；旧头427907。不采用更复杂的 Transformer 或新分支。
目标仍只有原固定 R10m 的 SE(2) 参考点 Smooth-L1，没有增加多项 loss。
首次零 readout 等于运动学初值，不等于旧 CV 初值；因此不是纯训练预算对照。

## 本地模拟：只验证机制，不证明真实涨点

独立 NumPy 10ms 运动积分产生数据，不调用候选积分器造标签。历史特征使用原
`ego_history_features`；scene token 是与目标独立的噪声，**没有把未来运动编码进特征**。
导航为三值 GT 派生条件，所有方法一样。各 seed：TRAIN2048，独立同分布512、
速度上限/位姿噪声变化512；两个固定 seed21/22，所有学习头3轮、batch64、各96步，
同 R10m/AdamW/余弦/样本顺序。该预算不冒充真实8998训练窗口的423步。

包含匀速、加速、转弯、刹车，以及 **历史无法观测的未来突变负对照**。所有类别/种子/方法
均报告，不以模拟测试挑最佳权重或将模拟分数当 nuScenes/planner 指标。

最终实现两 seed 平均3s位置误差（m）：

| 路线 | 同分布 | 速度/噪声变化 |
| --- | ---: | ---: |
| 原历史 CV | 3.7370 | 3.9146 |
| 原 Transformer 三轮 | 3.5100 | 3.6552 |
| 纯历史运动学 | 1.1033 | 1.3548 |
| 历史控制头三轮 | 1.1126 | 1.3502 |
| 场景控制头三轮 | 1.1046 | 1.3479 |

**主要收益来自可观测的运动学初值，不是已验证的学习收益。** 无法观测的未来突变
仍约5m级；匀速带噪声情况下，纯运动学反而约0.219–0.233m，CV约0.145–0.161m。
保护减少了原无约束拟合的噪声放大，但不能声称所有场景不下降。

本地先检查无保护拟合，发现匀速噪声退化，再加一致性保护，同 seed 重测，并完成
静止可起步修复后的最终复测。不是隐藏随机搜索或只报告胜出试次。
原始本地产物位于 `tmp/ego_kinematic_sim_20261011_a/`、
`tmp/ego_kinematic_sim_20261011_noise_guard/`、`tmp/ego_kinematic_sim_20261011_final/`，
以及积分向量化后的 `tmp/ego_kinematic_sim_20261011_vectorized/`（43.22s CPU，表中精度不变）。
完整 `simulation.json` 保留各 family/seed 的全部轨迹误差，不含真实数据精度。

## 一趟服务器试验

只读既有完成的三轮目录以及其相同10129窗口 bank；fit8998 / scene-held-out1131。
**不补缓存、不复制大 bank、不重新训练 WM/CCR。** 两个新头同初始化策略、同样本顺序，
各完整3轮423步、统一余弦3e-4→3e-6、最终epoch3；旧epoch3直接读已有权重，不重训。
所有候选/epoch报告保留，不自动按 holdout/dev 选优、扩训或替换正式方法。

默认固定DEV64一趟比较12路：OCC/STC × GT、外部、旧epoch3、纯运动学、历史控制、场景控制。
每窗原历史特征每模态仅提一次，每路线全部未来几何明确使用本路线 pose。
所有路线预测完成之后才读取未来 occupancy 评分；GT pose 只进入明确的GT对照/误差报告。
无 aligned、camera/lidar mask 或阈值搜索。报告IoU/mIoU与ADE/FDE/yaw，耗时不叫FPS。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export EGO_CONTROL_SOURCE=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_20261010_231306_partial_screen_three_epochs
export EGO_CONTROL_OUT="${EGO_CONTROL_SOURCE}_kinematic"
bash tools/ego_experiments/run_ego_kinematic.sh
cat "${EGO_CONTROL_OUT}_eval_dev64/summary.txt"
```

中断：一次Ctrl-C等待返回；相同代码、输出目录重跑同命令，自动恢复两头Adam/游标/RNG/
余弦及评估整窗整数前缀。kill -9只能恢复最近周期保存，不承诺恢复最后打印步。
与旧头不同参数化，不能把旧`last.pt`加载成新头的训练断点。

若需要请求DEV512，显式设 `EGO_CONTROL_POPULATION=dev512`，先固定原512名单再取
planner交集，预计同495，不补样本、不称独立测试。默认不自动跑该扩大评估。

实现全部放在 `tools/ego_experiments/`，未修改 legacy `real_motion` / `tools/real_motion`
Python 指纹；旧银行、头、训练/评估恢复继续兼容。新模型、训练、评估和wrapper自身源码
SHA单独绑定新契约。旧来源只读（互斥锁文件除外）。

本地针对性31项CPU测试通过，覆盖运动学独立解析解、停止/倒车/起步、梯度、
两头精确Adam/RNG恢复、源bank只读、旧三轮/partial/人口回归，以及真实小网格冻结WM/CCR
的12路线未来投影及权重不变。不是完整CI，不是实际L40S/CUDA或nuScenes质量验收。
