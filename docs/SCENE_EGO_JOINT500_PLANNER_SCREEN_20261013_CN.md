# 场景 ego：随机全模型联合 500 步与外接 planner 对照

用户选择**全模型随机初始化联合训练**，不是冻结主体只训练头。
旧 GT Refiner、loss500_minimal、速度探针及历史实验均不修改、不恢复。

## 审查与实际接线

复用四历史 SharedEncoder F，coverage 加权 8×8 tokens/历史 t0 坐标/时间；
143094 参数 scene ego 用 cross-attention 读 F，输出六步×三导航模式的 SE(2) 增量残差。
模式先选择再累计；运动学初值；t0 z/tilt 保留，不预测完整四元数姿态。
输入不含未来占据/标签/GT 连续位姿；GT 派生 destination navigation 是声明的条件。

整个 Encoder/STWM/Refiner/ego RANDOM 联合优化，原 batch4/source128、seed20261011。
使用原 TRAIN4086 顺序的前 500 个优化分组、500 步全程余弦，不是完整一轮。
原版 H6 完整 CE+Lovasz+ego R10；motion/existence 直接损失不计算。
模型中旧 existence 输出结构保留但不受直接监督、不作为几何开关。
几何背景每次按本次内部 pose 重建，硬 Strong/CLEAR 背景显式 detach；
continuous prior/memory 把 occupancy 梯度传到 XY/yaw、ego、F。
未来 GT 只在 forward 返回后进入 CE/Lovasz/直接 ego 监督。
数据读取器可提前在 CPU 载入监督包，不把这些数组传给内部预测器。

第一次 ego/Refiner 零末层可能阻断部分上游路径，是初始化性质，不是假断梯度。
新增实际第二步梯度检查并把结果记进训练日志；不改变 loss 或初始化来“通过检查”。

## 评估定义

固定最终 step500，自动在 manifest 原 DEV512 的 planner-covered 交集评估：

1. internal：本模型 scene ego。
2. external：现存 v4 BEV-Planner+yaw 六未来世界位姿。
3. gt_reference：未来 GT ego 条件参考，非独立训练的 GT 模型。

三路**同一主体权重、同一窗口、同一历史 F/运动输出**，只切未来 pose；
每路背景/prior/memory/Refiner 都按该路 pose 重建，不把内部/GT 背景用于 external。
planner 缺失窗口从三路同时排除，不补样，不以 GT 替代 planner。
打印 requested/covered/missing/身份名单；可能是495而不是512，实际以打印为准。
仅从旧 camera_pred NPZ 读取 identity/tokens/future_e2g，绝不读取 BEVStereo 历史或 latent。
原 v4 tag/SHA 声明、六 rigid poses、t0 z/tilt/时间 identity 校验；archive stat 与实际
读取的 pose/identity 指纹锁入新契约，原 JSON 身份审计由之前独立实验完成。

三路完整六帧 argmax 后，同全网格口径报告1/2/3s mIoU/IoU/MovingMicro、轨迹误差；
GT 支持/标签仅用于评分，不做 GT pose alignment、阈值/最佳轮次/路线选择。
这不是内部头与外部 planner 的同容量/同训练数据算法消融，只回答同一模型换轨迹的效果。
不包含 STC、完整数据训练、正式 FPS 或自动替换主结果。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export PLAN_CACHE=/root/nas/occ/new_code/cache/come_main_table/camera_pred
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_joint500_$(date +%Y%m%d_%H%M%S)"
bash tools/scene_ego_head/experiment/run.sh && cat "$RUN_DIR/summary.txt"
```

原模块仅供架构与缓存契约，不导入 E14 或其他已训模型权重。
无新 frozen feature bank；原 history cache 要求已完整且只读。
先等其他 GPU 训练结束；脚本不会停止任何进程。源码检查失败不自动 merge/stash/retry。

Ctrl-C/SIGTERM 在完成 update 后保存新 last.pt；若崩溃只能回周期断点。
训练与三路整窗评估都支持同目录恢复：

```bash
# RUN_DIR 必须是刚才实际打印的原目录，不能重新生成时间戳。
SCENE_EGO_RESUME=1 bash tools/scene_ego_head/experiment/run.sh
cat "$RUN_DIR/summary.txt"
```

500步完成后 resume 不再追加步数；评估恢复不重新计算已提交窗口指标。
新协议锁模型/Adam/RNG/余弦/实现/人口/外部pose来源；拒绝旧GT/head断点静默移入。
训练runner仅作用于该进程且所有注入在 finally 恢复，旧trainer文件字节不修改。

## 本地验证边界

相关43 passed / 1 skipped：实际CPU/CUDA小网格完整训练反传、ego/F梯度、
mode先选后积、未来GT/背景毒化、真实CPU Strong、共享一次历史编码、
选择性planner加载/错identity拒绝/同人口三路整窗恢复、CPU Adam/RNG恢复、
实际CLI固定500预算/暂停续训/旧协议拒绝及旧Refiner/执行探针回归。
skip 为Windows不支持的Linux/Triton编译候选，不是此次训练路径的跳过。
没有nuScenes数据、L40S吞吐或真实500步质量结果，不能承诺超过planner。
