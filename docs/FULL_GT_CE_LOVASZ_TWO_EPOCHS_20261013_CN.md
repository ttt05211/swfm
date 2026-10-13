# 完整 TRAIN 的 GT 主体，两轮 CE＋Lovász

## 固定配置

- 新目录、独立随机初始化，完整 TRAIN20430 ×2；不导入旧20%模型或其Adam。
- 四历史、六未来；GT未来ego位姿作为条件，不预测ego、不读取导航command。
- 原 Shared Encoder＋STWM＋连续Transport prior＋Semantic Refiner；原 sequential H6执行。
- 仅完整六时距全网格 CE＋Lovász，各权重1。物体XY/yaw由占据梯度学习；不计算direct motion/existence/ego loss。
- 移除ego网络参数。STWM已有existence输出结构保留但不用于合成/监督，其参数无梯度、不发生AdamW衰减。
- 原 seed20261011、batch4/source128、运动5e-4/Refiner3e-4、全程两轮余弦floor0.01、BF16网络、4线程有界预取。
- 只读复用既有完整TRAIN/VAL历史几何缓存；不构建新的learned feature bank，不采纳此前FAIL提速后端。
- 第1轮自动固定DEV512评Transport/Refiner的1/2/3s mIoU、IoU、MovingMicro，随后恢复同一两轮计划训练；第2轮也评同人口。
- 两轮都报告，预定最终epoch2，不按dev挑最佳、不自动扩至20轮。之后外部planner推理是单独任务，本入口不执行。

新协议 `shared_semantic_refiner_full_gt_ce_lovasz_2epoch_v1`。旧GT/Pred及缓存/历史文件不改。
通用驱动保留空ego optimizer参数组和历史CLI的aux权重元数据占位，均不参与目标；实际目标以 `contract.model.objective`（CE/Lovász=1，其余=0）及新engine为准。
E14文件仅提供模型尺寸/缓存来源契约，绝不复制其权重。新初始化去掉ego头，与旧含ego辅助头的初始化/裁剪也不构成严格单变量消融。

## 服务器启动

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2
unset RESUME FULL_GT_RESUME
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/full_gt/run.sh
```

不要与其它GPU训练同时运行。脚本打印并固定RUN_DIR，旧目录不删除。首轮是全部20430窗，不再是4086。
流程：`train/last.pt` 到真实epoch1边界 → `train/epoch_0001.pt` 只读评到 `train/eval_epoch1/` → 接 `train/last.pt` 完成epoch2及评估。
根 `summary.txt` / `comparison.json` 汇总两轮；第二轮checkpoint为 `train/epoch_0002.pt`，不是根目录的last.pt。

## 中断恢复

按一次Ctrl-C，等保存完整update或完整评估窗口并返回提示符。硬kill只能恢复最近周期checkpoint。
恢复时明确设置原来的流程根目录，不指向 `train` 或另建日期目录：

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_实际日期时间
unset RESUME
FULL_GT_RESUME=1 bash tools/shared_history_experiments/full_gt/run.sh
```

保留原Adam、RNG、epoch/batch游标和整周期cosine。评估中断复用同目录整数计数及窗口身份检查；已完成评估不重跑。
代码/输入/预算不一致明确拒绝，不静默转入旧500/20%协议。修改总轮数不是此入口的resume。

## 本地验证及边界

18项针对性与原Refiner回归通过（CPU/CUDA小网格及模拟数据集）：完整人口接线、ego/command/辅助标签不读、CE/Lovász到XY和memory梯度、原epoch边界Adam恢复、只读中期评估、整数评估续算/身份拒绝、两轮流程和完成后不重复。
没有本地nuScenes数据；不代表L40S吞吐、完整数据质量或模型已经收敛。没有代启动服务器训练，不继承旧模型FPS。
