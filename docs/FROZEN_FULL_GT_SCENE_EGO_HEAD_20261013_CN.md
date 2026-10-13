# 两阶段：完整GT主体冻结，只训练场景ego头

## 做什么

后续用户授权的随机头10轮＋旧bank只读复用＋完整VAL入口见 `FROZEN_EGO_RANDOM10_REUSE_ALL_VAL_20261013_CN.md`。旧默认仍3轮/DEV512；新脚本run10_all.sh明确选10轮/all，新目录独立，不直接续接旧1轮周期。
用户新指定的随机头100轮＋50/80/100快照＋旧10轮头同DEV512四次评估，见 `FROZEN_EGO_RANDOM100_FOUR_DEV512_20261013_CN.md`，入口run100_compare.sh。

读取当前完整TRAIN20430×2的**最终epoch_0002.pt**，冻结Shared Encoder、STWM和SemanticRefiner。
复用现有SceneEgoHead（默认143094参数），读取四帧历史ego运动和同一Encoder的历史场景F，按已声明的6时刻right/left/straight导航分支输出XY/yaw。
不是重训主体，也不是从随机初始化进行Pred全网联合训练。
头以完整TRAIN20430训练，默认3轮/batch64/seed21/AdamW3e-4/整三轮cosine最低1%；只用固定R10几何监督。
不加CE/Lovász反传给头、不更新冻结主体，不恢复旧motion/existence损失，不训练外部planner。
GT主体原训练仍是CE+Lovász；两阶段都需要计入方法训练成本。

## 因果边界与缓存

一次提取当前冻结Encoder的**纯历史紧凑场景tokens**：4×8×8×64特征、t0几何/时间/有效性、历史ego。
默认保留FP32，不压缩为低精度，不缓存完整F或任何future-conditioned memory/背景。
TRAIN未来位姿只作R10标签；离散GT派生destination command仍是显式导航条件，不能称navigation-free。
历史帧reader从旧metadata截取最后四帧，只有这些历史占据/mask/pose进入Encoder。
新bank写入新EGO_RUN_DIR/history_tokens/，64窗一块、完整块SHA/身份/源权重/实现/运行时绑定；不能混用旧Surface头bank或另一轮GT特征。
64宽×256tokens的特征本身约1.25GiB（20430窗），加其他字段和存储开销约1.4GiB，实际manifest.tensor_mib为准。
整bank载入CPU，按batch搬GPU；不把整个bank常驻GPU，拼接时CPU有暂时额外副本，缓存预算不等于全部RSS。
首次提取会有成本；之后头的三轮不重复执行Encoder，更不执行训练renderer/Refiner。
旧纯历史几何缓存仍用于DEV推理只读，不覆盖任何旧目录。

## 入口检查、评估与恢复

正式头训练之前，一窗同GT位姿分别走旧GT缓存背景入口和现场重建入口：比较背景、F/prior/memory/logits。
背景必须一致；连续张量rtol2e-4/atol2e-5固定，且六帧argmax零差异。不以GT诊断分数替代Pred指标，不放宽失败。
检查结果在same_pose_entrance_audit_v2.json；失败则停，不开始头优化。

2026-10-13服务器一窗定位：原GT和重建入口自身重跑各有1体素变化，完全同prior/memory的BF16 Refiner重复则字节一致；跨入口BF16从input层出现差异、2体素改变，memory/coarse prior/point prior分别有4/12/60元素转BF16后改变。FP32跨输入数值通过且argmax0差异，但FP32与原BF16有313体素变化，不能作为无损替换。

v2仅在入口审计的两次完整forward内启用严格确定性算子，关闭cuDNN benchmark、启用cuDNN确定性；BF16及原数值容差不变。不支持确定性的算子直接报错，没有warn-only或自动放行。退出及异常均恢复原执行设置，bank/头训练/DEV推理不改变。用于隔离浮点聚合顺序的影响，**不是证明原普通执行字节等价，也不保证服务器gate通过**。
原same_pose_entrance_audit.json的FAIL保留。本次未有头断点；代码契约改变须开新EGO_RUN_DIR，勿覆盖/删除旧目录或强改stage_contract恢复。新检查通过才自动继续用户预定的1轮头训练及DEV512。

头训练后仅运行完整DEV512内部ego预测：六帧预测、1/2/3s及平均mIoU/IoU/MovingMicro、ego头与其prior的ADE/FDE/yaw。
预测ego用于全部Strong背景、Transport和memory投影；不沿用GT未来背景，不事后GT对齐。
无外接planner或GT参考重复预测、不按DEV选择epoch，最终预定第3轮。
固定Body digest在阶段前后相同、优化过程中trainable/grad参数数均0；body_freeze_audit.json公开。
头的head_last.pt保存Adam/LR位置/RNG/完整update游标，32步和轮末保存；Ctrl-C完整update保存。
bank按完整块恢复，DEV按完整窗整数累计恢复；同一EGO_RUN_DIR、同GT_CKPT、同预算配置，不静默迁移。
GT主体及运行中trainer的任何源文件/断点fingerprint没有变化。

## 等GT训练完成后再运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export EGO_RUN_DIR="$PWD/outputs/p0_f9_shared_history/frozen_gt_scene_ego_$(date +%Y%m%d_%H%M%S)"
bash tools/scene_ego_head/frozen_gt/run.sh && cat "$EGO_RUN_DIR/summary.txt"
```

若想预先固定只训1轮，首次运行前export HEAD_EPOCHS=1；此配置须随恢复保持不变。
第一轮主体或未完成epoch2会拒绝；不自动等待、启动原训练或抢占正在训练的GPU。
新终端恢复时设回**原**EGO_RUN_DIR（不是新date）及GT_CKPT，然后：

```bash
FROZEN_EGO_RESUME=1 bash tools/scene_ego_head/frozen_gt/run.sh
```

开始时显示FROZEN_BODY；缓存显示FROZEN_EGO_BANK，不等于已进行优化。
优化显示FROZEN_EGO_TRAIN/FROZEN_EGO_EPOCH；最终DEV显示FROZEN_INTERNAL_EVAL。
只在TRAIN使用轨迹标签训练不是验证泄漏，但必须如实报告GT条件预训练＋冻结头训练、导航条件和总预算。
不保证内部头超过external planner；当前阶段检验强历史表示的可用性，不默认后续解冻或扩训。

## 本地验证

新冻结头专项7项＋原完整GT训练/外部全集入口回归，共15passed in36.02s。
包含实际CPU/CUDA小网格历史特征与内部六帧前向、未来GT pose/标签/背景毒化、同位姿背景差异gate、仅头Adam更新、断点恢复、bank前缀/来源拒绝、完整模拟CLI的512窗口及源文件字节不变。
CLI帮助/Bash语法/diff检查通过，旧full_gt/refine/scene组件源码没有改动。
模拟人口用于接线验证，不是实际nuScenes精度或L40S速度；服务器训练未代启动。
增加缓存/头训练分开计时、将源GT训练性能移入source_GT_training_performance后，实际CLI专项再次1passed in24.80s；不把旧主体0.4秒/窗误当成当前头训练速度。

v2确定性审计及原冻结头/一窗诊断回归：14passed in26.58s，含非零Refiner残差CPU/CUDA实测、两次审计一致、权重/精度不变、严格不支持算子异常及设置恢复。仅本地小图验证，不代表真实服务器数据已经通过。
