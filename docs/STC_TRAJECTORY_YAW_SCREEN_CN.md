# 冻结 planner 轨迹切线 yaw：最后一次小规模对照

只改变可靠的未来 yaw。不是 COME 的 aligned 评分，不是无损优化，也不是官方原 planner 复现。效果尚未在真实服务器数据上验证。

## 固定范围与规则

- 原平均权重 5/6/8/12/14、四历史→六未来、ADD raw0.5、REMOVEoff；网络和 checkpoint 不变，不训练、不校准。
- 同一冻结 dev64 population：Occ+Pred、STC+Pred，各跑原版与候选。共四路，不重跑 GT-conditioned 设置，不重复失败的 z/tilt 与 STC 地面过滤。
- 只读四帧历史的实测 ego pose/timestamp；四点二次拟合求 t0 速度，历史 body heading 与速度方向的差用于 sideslip 校准。
- 预测 XY 加 t0 形成七点路径；未来仍按名义0.5s slot。每个 waypoint 用相邻三点二次函数在该点求导，末点用最后三点单边求导，避免简单前后位移引入半步方向偏移。
- 历史速度1–40m/s、拟合RMSE≤0.2m、sideslip≤10°、加速度≤8m/s²；未来速度1–40m/s、局部加速度≤8m/s²、方向变化≤45°/s、相对原yaw修正≤15°。否则保留原始pose。规则一开始固定，不扫描。
- 新yaw = 轨迹切线方向 + 历史sideslip；用世界Rz左乘原旋转，保持 planner XYZ、齐次行逐字节相同及 yaw-free roll/pitch 不变。
- 四路完整预测后才读取未来语义与真实未来pose用于评分/误差审计。未来真实pose不进入候选函数，也不做事后对齐；Camera历史不借用 GT visibility。
- 候选六pose完全不变时复用该设置的原六帧输出；首次真正改变yaw时仍做 canonical 六帧/概率 exactness 检查。记录实际 forward 数与复用数，不把四路对照计时当单网FPS。

## 运行

无需下载新数据、重建缓存或平均权重。采用已确认的服务器目录默认值；外部mean路径与原STC四设置相同，可用 `STC_CHECKPOINT` 显式指定现有平均文件。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
(
  set -euo pipefail
  git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
  git merge --ff-only FETCH_HEAD
  export STC_YAW_OUT="$PWD/outputs/p0_f9_joint_surface_ccr/stc_yaw_dev64_$(date +%Y%m%d_%H%M%S)"
  unset STC_YAW_RESUME
  bash tools/real_motion/run_p0_f9_stc_trajectory_yaw.sh
  cat "$STC_YAW_OUT/summary.txt"
)
```

Ctrl-C/SIGTERM仅在完整窗口边界保存；kill -9只能恢复最后周期状态。单独续评：

```bash
# STC_YAW_OUT 指定本次打印的实际输出目录，不能使用旧四设置/geometry目录。
export STC_YAW_OUT=/实际的本次输出目录
export STC_YAW_RESUME=1
bash tools/real_motion/run_p0_f9_stc_trajectory_yaw.sh
```

实现/权重/人口/数据/运行开关契约改变拒绝同目录续评；新终端恢复记录的 SWFM 开关，不执行环境中的shell文本。仅新输出可写，原数据、权重、旧训练和四设置结果保持只读。

## 怎么判断

同时看两个设置的1/2/3s和均值IoU/mIoU；审计全人口及真正改动样本的原/新yaw误差median/p90、修正比例与回退原因。yaw误差降低不保证占据指标提升，反之也不据此挑阈值或权重。

这只是一次 dev64 诊断，任何正结果都不自动代替正式四设置表或证明全集提升。若没有清晰增益，到此停止冻结启发式调整，不再自动试另一组规则。没有无训练方案能保证弥补 STC 输入中的系统性缺失/错占据或真实规划XY误差。

之前九路集中对照已失败：历史路面pose补偿使Occ+Pred均值mIoU−1.212540pp、STC+Pred−0.403252pp；STC地面时序稳定−0.306691pp；组合−0.644499pp。不得把这些分支偷偷并入新候选。

本地测试覆盖可靠yaw/异常回退、曲线路径与wrap、大世界坐标、planner XYZ与body tilt保持、未来GT改变不影响预测、实际CPU小模型原预测字节一致、原权重不变、完整窗口断点恢复及CLI输出保护。CPU测试不是L40S/真实dev64精度验收。
本次专项与旧STC回归57 passed、1 CUDA skipped；CLI help、Bash语法通过。依 windows-python-env-guard 使用项目解释器，没有调用Anaconda base；这不是全仓CI。
