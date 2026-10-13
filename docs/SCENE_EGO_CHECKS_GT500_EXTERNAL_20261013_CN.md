# 冻结内部ego检查与GT500接外部planner推理

用户授权两项只读任务，不再训练、不继续500余弦、不改正式GT/Pred入口。
新代码仅 `tools/scene_ego_head/diagnostics/`，旧组件/训练fingerprint均不变。

## 任务一：ego_checks

冻结已完成内部Pred500：
`outputs/p0_f9_shared_history/scene_ego_internal500_20261013_095714/last.pt`。

- DEV512同人口比较ego头与其history kinematic prior的ADE、3s FDE、六时刻XY/yaw。
- 只执行历史Encoder与ego头，不执行STWM/Strong/Refiner/稠密预测。
- 原训练人口前4个batch（最多16窗，全部属于500步已见窗口）计算一次冻结forward，
  对CE+Lovasz与直接ego R10分别求VJP；报告pose、共享F、ego参数的范数比/余弦与逐时距。
- pose梯度将yaw除以10换成R10平移等价尺度；不把米/radian未缩放梯度混成结论。
- 固定eval模式/梯度开启；无loss.backward、clip、Adam.step或参数更新。
  小TRAIN子集不是完整TRAIN质量，负余弦不是自动证明有害联合训练。
- checkpoint、源Adam/RNG、模型权重都保持只读；独立保存完整诊断窗口续跑状态。

## 任务二：gt_external

自动寻找最近**已完成**且协议为CE+Lovasz+ego ONLY的GT `loss500_minimal_*/last.pt`；
必须500步/原TRAIN4086/DEV512/B4-source128，拒绝旧full-loss/其他轮次。
打印实际FROZEN_SOURCE，可显式GT500_CKPT指定而不猜timestamp。

原GT500主体保持所有权重，**不添加内部scene ego头**，原8710参数辅助头不执行。
只历史F/STWM/lift→外部六pose→fresh Strong/CLEAR/连续prior/memory/Refiner，
绝不复用GT未来背景、GT pose或修改planner；无aligned、阈值选择。

为了实际判断是否值得放弃当前内部ego，当前Pred500也在**同一planner覆盖交集**
重新做只读占据推理。两个模型各自计算学习特征，不跨模型复用F/运动/预测。
原DEV512∩v4 camera_pred cache，预计495实际打印；缺失所有模型同时排除不补GT。
未来GT/Moving annotations只在两者六预测结束后用于指标（CPU预取可载入监督包，
但这些字段不作为预测器输入）。输出1/2/3s mIoU/IoU/Moving及轨迹误差和外部减内部差值。
GT500+planner仍是GT训练后的轨迹替换测试，不叫“独立planner条件训练模型”。
两者主体训练条件/ego结构不同，非单变量头消融，不与原完整512结果直接减分。

## 服务器依次运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export EGO_CKPT="$PWD/outputs/p0_f9_shared_history/scene_ego_internal500_20261013_095714/last.pt"

export RUN_DIR="$PWD/outputs/p0_f9_shared_history/ego_checks_$(date +%Y%m%d_%H%M%S)"
bash tools/scene_ego_head/diagnostics/run.sh ego_checks && cat "$RUN_DIR/summary.txt"

# 不覆盖上一个诊断目录；自动识别已完成GT minimal500来源。
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/gt500_external_$(date +%Y%m%d_%H%M%S)"
export PLAN_CACHE=/root/nas/occ/new_code/cache/come_main_table/camera_pred
bash tools/scene_ego_head/diagnostics/run.sh gt_external && cat "$RUN_DIR/summary.txt"
```

GT自动识别错误/不存在时，显式 `export GT500_CKPT=/实际GT500目录/last.pt`。
若诊断中断，原RUN_DIR保持不变，设置 `EGO_DIAG_RESUME=1` 重跑对应任务。
冻结快照在独立frozen目录，原训练last.pt/summary/optimizer/cache不写入。
恢复使用已冻结来源，不随新last.pt或自动扫描结果变化；data/architecture/runtime/
种群/实施代码与fingerprint变化拒绝。零科学更新，不宣称正式FPS或自动推广。

本地相关38passed：真实CPU/CUDA小网格VJP/模型不变、ego-only未来GT/dense毒化、
GT旧辅助头不运行/GT未来背景毒化、同人口双模型原子恢复，以及真实只读CLI两任务
500来源/严格数据runtime/禁止训练和source Adam-RNG毒化/来源字节不变/恢复回归。
无nuScenes或L40S实际结果，等待服务器summary；不会假定外部planner一定更好。
