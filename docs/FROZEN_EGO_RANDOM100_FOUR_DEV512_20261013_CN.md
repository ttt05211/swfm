# 随机ego头100轮：保存50/80/100，同DEV512评10/50/80/100

用户指定四个头对比，不重新训练10轮基线；新头从随机初始化完整训练100轮，不续旧10轮Adam/权重。Shared Encoder/STWM/Refiner始终为完整GT epoch2/update10292冻结主体。TRAIN20430、batch64、seed21、R10监督、AdamW3e-4、100轮完整cosine最低1%，每轮320updates，总32000。

固定保存head_epoch50.pt、head_epoch80.pt、head_epoch100.pt（head_last.pt仍保存Adam/RNG/游标用于中断恢复）。50/80是同一个100轮cosine中的快照，**不是各自50轮/80轮降到最低LR的独立实验**；现有10轮基线原来采用10轮cosine，这一差别如实报告。

历史tokens沿用此前兼容bank只读导入，不改旧manifest、不提取、不导旧头/Adam/RNG。旧10轮头复制到新目录frozen/head10.pt作为只读评估快照，校验主体/特征/头结构/完整10轮update计数和参数hash；它不进入新头训练初始化。

100轮完成后，按10/50/80/100顺序各跑一遍相同固定DEV512，不裁planner窗口、不用全集代替，不按分数选择/自动promote。固定主体、头结构、population/order、完整6帧和full-grid指标；报告Transport/Refiner的1/2/3s及平均IoU/mIoU/Moving、轨迹ADE/FDE/yaw。四次结果在head_comparison.json，summary.txt给四行表。

每个头独立dev512_head{N}_state.json整数计数恢复，绑定checkpoint SHA/主体digest/精确512keys。旧文件和失败记录不覆盖、不放宽入口gate、不切推理精度。恢复同一次100轮任务不重置学习率或重训10轮，完成的前缀预测可跳过。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export EGO_RUN_DIR="$PWD/outputs/p0_f9_shared_history/frozen_gt_scene_ego_100ep_dev512_$(date +%Y%m%d_%H%M%S)"
unset FROZEN_EGO_RESUME EGO_REUSE_DIR HEAD10_CKPT
bash tools/scene_ego_head/frozen_gt/run100_compare.sh && cat "$EGO_RUN_DIR/summary.txt"
```

自动定位按修改时间找最新兼容完整bank和已完成10轮头，不读取DEV分数；打印History source和baseline head供核对。若有多个10轮头、希望固定某一个，export HEAD10_CKPT为本次10轮成功目录的head_epoch10.pt。
中断恢复：设回原EGO_RUN_DIR、原GT_CKPT及首次打印的EGO_REUSE_DIR/HEAD10_CKPT，执行`FROZEN_EGO_RESUME=1 bash tools/scene_ego_head/frozen_gt/run100_compare.sh`。不要把旧10轮/1轮任务目录改成100轮恢复。
所有条件仍是GT派生导航，不是navigation-free；GT主体预训练成本需计入。这里只增加头训练预算，不重训主体、不保证超过planner/改善DEV。

本地19项回归通过（51.03s）：真实小bank头100轮/50与80快照/中断恢复参数一致，模拟CLI四头同512keys/独立eval state恢复/源GT与10头字节不变，旧缓存与CPU/CUDA前向/诊断回归；Bash/diff检查通过。Windows checkpoint metadata路径已规范化，不修改源文件。模拟人口仅验证接线，尚无真实nuScenes/L40S的100轮质量或速度结论。
