# 内部 ego 的独立 Pred 三轮试训

保持当前 GT 三轮试训的规模：TRAIN20430 的固定 scene-balanced 20%（4086窗），三轮整周期余弦、seed20261011、batch4/source128、最终同 DEV512。不是全量训练，也不是在现有 GT 权重上微调。

原模型结构、随机初始化方式和损失不变：Transport + Surface CCR + 8710参数历史运动学 ego 头，完整六未来全网格 CE+Lovász、source 几何/existence、R10 ego 几何辅助。训练两未来帧一组，逐窗口网络/反传顺序不变。不同浮点执行不称字节无损训练。

Pred 的所有未来 Strong/背景、搬运、owner/fallback、CCR 投影/phase、软合成使用内部预测 ego。未来 GT 位姿只作为 ego 标签；未来 GT occupancy 只作为训练监督/评估标签。导航命令仍是既定 GT-derived destination 条件，不称 navigation-free。占据到 ego 的梯度沿用既有0.1→1 ramp；前向始终完全 Pred，不混 GT pose。t0 的 z/tilt 约定不变。

只读复用旧 TRAIN/VAL 纯历史几何与紧凑支持，丢弃未来 GT ego 条件化背景/投影，实时按 Pred 重算。无新 feature bank、无大缓存重建、无旧权重/Adam/RNG 写入。新代码隔离在 `tools/joint_occ_experiments/pred/`，不改变既有 GT 的恢复指纹。

最终固定 DEV512 一趟报告六读出：hard/soft × Transport、ADD-only、ADD+REMOVE；预先指定软 ADD-only 为主候选，不按 dev 分数挑阈值/轮次。软输出使用已测的三未来帧快速 renderer，首个实际评估窗所有六帧/三软路线对旧 renderer 逐字节核验，失败停止。旧54.945FPS/GT软44.248FPS不直接赋给新 Pred。

## 开始与恢复

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export RUN_DIR="$PWD/outputs/p0_f9_joint_occ_ego/pred3_soft_$(date +%Y%m%d_%H%M%S)"
bash tools/joint_occ_experiments/pred/run.sh
cat "$RUN_DIR/summary.txt"
```

无需先8步测速；直接跑相同20%×3轮，再固定DEV512。Ctrl-C/SIGTERM 在完成 optimizer update 或完整评估窗边界保存，周期每64更新保存 `last.pt`；kill -9只能恢复已写入的周期断点。

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/这里换成本次打印的pred3_soft目录
RESUME="$RUN_DIR/last.pt" bash tools/joint_occ_experiments/pred/run.sh
cat "$RUN_DIR/summary.txt"
```

必须同一 Pred 目录恢复；不能导入 GT checkpoint，不能改变数据、源码实现、模型、损失、LR/schedule或批次。评估支持已完成整数前缀恢复，不重复计分。原 GT 正式模型、三轮 GT 结果、已有软测速与其他实验均保持不变。

本地49项相关测试通过（Pred梯度/未来GT毒化/六帧软读出/整数评估续跑/Adam和RNG恢复，以及原GT、两帧训练、软快速推理回归）；CLI help、Bash语法检查通过。修复了评估续跑对调用者已保存errors列表的原位修改。当地无nuScenes/L40S，未代跑真实训练或承诺Pred精度/FPS。
