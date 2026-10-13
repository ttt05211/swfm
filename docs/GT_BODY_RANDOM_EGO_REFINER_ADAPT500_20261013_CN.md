# GT epoch2主体＋随机ego：局部联合微调500步

按用户最终选择：GT完整epoch2/update10292初始化Encoder/STWM/Refiner，SceneEgoHead独立seed21随机初始化（末层0，起点是历史运动学prior），不加载10/50/80/100头、GT Adam或GT RNG。
冻结Encoder/STWM并固定eval模式；只训练SemanticRefiner全部参数及ego头。Refiner1e-5、ego1e-4、全500步cosine最低1%，B4/source128、同既有TRAIN20%4086选择/seed20261011、第一轮前500组约2000窗，不称完整epoch。
完整6帧CE+Lovász+ego R10、权重1；motion/existence辅助不计算。Pred所有未来Strong/prior/memory用内部ego，无GT对齐。沿用可微forward_pred_window，不使用冻结头的NumPy预测入口训练；硬Strong背景不回传，连续prior/memory投影仍将Occ梯度传到ego。

旧历史几何缓存只读，现场跑冻结Encoder和STWM、可训练Refiner，不提取新bank或用旧紧凑tokens替代稠密训练。模型架构/推理后端不另加模块，不保证速度或优于10轮头。GT预训练成本和GT派生navigation必须声明；不是整网RANDOM/导航自由。

独立pred_adapt/入口，旧GT/frozen_gt/内部500源码不改。源GT快照在新RUN_DIR/frozen/GTepoch2.pt，正式本次断点在RUN_DIR/train/last.pt，初始化及冻结审核在根目录。generic驱动旧JOINT_OCC RANDOM启动标题不代表主体随机重置，以initialization_audit/contract.initialization=GT_epoch2_body_RANDOM_ego_seed21_fresh_Adam为准。

结束自动内部Pred完整DEV512 Transport/Refiner的1/2/3s及平均IoU/mIoU/Moving、ego六时刻XY/yaw；不跑外部planner、不选DEV-best。首次源模型恢复要求GT完整epoch2、原数据/架构metadata SHA一致。原GT/source文件只读。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2
export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/GT2_random_ego_refiner_adapt500_$(date +%Y%m%d_%H%M%S)"
unset PRED_ADAPT_RESUME RESUME
bash tools/scene_ego_head/pred_adapt/run.sh && cat "$RUN_DIR/summary.txt"
```

中断后设回原RUN_DIR和GT_CKPT，执行`PRED_ADAPT_RESUME=1 bash tools/scene_ego_head/pred_adapt/run.sh`，仅恢复本次train/last.pt的Adam/RNG/500步游标，不从其它实验续接。

针对性本地3passed in8.59s：真实CPU/CUDA GT权重严格初始化、ego末层0、全6前向/反向Occ到ego非零，Refiner/ego更新、transport无梯度/无Adam状态/参数hash保持；实际CLI两步→三步保存恢复、GT源字节不变、旧函数恢复。Bash/diff检查通过。小图证明接线，不是实际nuScenes质量或L40S速度；未代启动服务器。
