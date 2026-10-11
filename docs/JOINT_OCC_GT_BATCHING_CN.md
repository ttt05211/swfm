# GT-only 网络执行合批（实施历史，服务器数值未通过）

2026-10-11服务器结果：packed_h2虽1.48035倍提速，但模型/Adam数值门槛失败；packed_h1更慢且失败，均不采用。以下保留实施历史，不是当前推荐执行。新窄版仅两未来帧并行、保留逐窗口网络与反传，见 `JOINT_OCC_GT_HORIZONS_CN.md`；旧失败报告不能授权新训练。

本轮针对实际训练日志 .762005 s/窗：soft render+CE/Lovasz .364405，occupancy backward .262491；输入等待仅 .0000044。
旧全局 batch4 实际是四窗口逐个网络前向、各六未来逐帧反传。不是数据预取或显存容量本身不足。

新增 `tools/joint_occ_experiments/batching/` 独立 opt-in：

- 四窗口物体沿独立 batch 轴合并 WM；CCR canonical 行合并编码/读出，物体索引按窗口偏移，静态负编号不偏移。
- CPU learned-pose ownership 投影最多四工作线程，先一次 GPU→CPU XY/yaw 读回；worker 不访问 CUDA、模型或未来标签。
- 未来帧默认两帧并行 grid_sample/splat/有序概率合成、全网格 CE/Lovasz/反传；另测仅网络合批、每次一帧。
- 六帧 render VJP 仍先在边界累积，再对四窗口共享 WM/CCR 图做一次联合反传。
- 每帧独立 CE 有效体素均值、Lovasz present-class 均值，然后按原窗口/六帧平均；不池化类别、不抽稀体素。不复用上轮失败的 prepared/sparse loss 后端。
- source 几何、existence、R10 ego、clip、Adam、batch4/source128、训练窗口顺序、更新次数及全程余弦均保留。GT-only 先做，Pred 不自动扩展。
- ADD/REMOVE owner/fallback/前后遮挡顺序不变；新代码只用于训练，原快速硬推理和评估入口未改。不宣称继承/重测了旧 FPS。

CUDA 合批改变 GEMM 形状/累积次序，不能称字节无损。CPU 小网格通过不证明 L40S/BF16 数值通过或一定加速。
GPU 先做同一已保存断点/RNG/八批真实 Adam 的两遍交替 A/B/C：current_fused、packed_h1、packed_h2；首用预热和实际概率/梯度审计不计吞吐。
只读测速没有发布科学更新，原源码/权重/Adam/RNG/history 缓存不写。计时含目标准备，共享输入读取不计。

原模型/Adam 容差仍 rtol=2e-4/atol=2e-5；首窗 grouped renderer 概率2e-5/2e-6、loss2e-6/3e-7、梯度3e-5/2e-7（rtol/atol），不因失败放宽。
基线自身 repeat 不一致仍 FAIL。候选需损失/梯度、warm/八步模型+Adam 数值门槛全部通过、两遍均快、平均≥1.1倍、reserved 显存≤90%。
当前先前基线 CUDA repeat 的漂移尚未定位，不把它算作任何新方案通过的理由。
OOM/非有限/结构错误停止；不重试已执行一半的优化步，保留之前健康 checkpoint，不悄悄降低 batch 或改损失。

## 服务器命令

用户已经停止训练；不再 kill，不自动启动竞争进程。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
export GT_RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt3_fast_20261011_123726
export BATCHING_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt_batching_probe_$(date +%Y%m%d_%H%M%S)
unset RESUME BATCHING_PROOF
bash tools/joint_occ_experiments/batching/run.sh probe
cat "$BATCHING_OUT/summary.txt"
```

测速源来自上述原始 `last.pt` 实际 cursor，不硬编码第8步；八批×三执行×两遍只做临时训练，无 dev 评估或新缓存。先发 summary，门槛未通过不要切换。

若 packed_h2 全部通过，显式选择后接续（若仅 h1 通过，将 HORIZON_GROUP=1）：

```bash
export BATCHING_PROOF="$BATCHING_OUT/speed.json"
export BATCHING_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt_batching_resume_$(date +%Y%m%d_%H%M%S)
export HORIZON_GROUP=2
bash tools/joint_occ_experiments/batching/run.sh resume
```

只接受同一源 checkpoint SHA 和当前合批代码指纹的成功报告。新目录导入原 weights/Adam/RNG/cursor，完整科学 contract 逐字段对照；仅 implementation 明确变更，记录 batching_import.json。
新 checkpoint 绑定新 execution 指纹，因此原 run_gt.sh 不能误续新合批 checkpoint；恢复合批须使用同一入口和 group。
Ctrl-C/SIGTERM 等当前完整 Adam 步完成保存；不要连续 Ctrl-C 或 kill -9。

以后暂停再恢复合批的同目录：

```bash
export RESUME="$BATCHING_OUT/last.pt"
bash tools/joint_occ_experiments/batching/run.sh resume
```

初次导入后已有同 execution 的 checkpoint 不需重新测速。改变源码/group/data/loss/全局batch仍拒绝无声续训。
若新测速失败，旧训练仍可原样恢复：

```bash
export RUN_DIR="$GT_RUN_DIR"
export RESUME="$RUN_DIR/last.pt"
bash tools/joint_occ_experiments/run_gt.sh
```

本地验证：17项新合批/恢复/只读测速及43项原fused/模型/probe回归，共60项CPU通过；Bash语法通过。范围为真实小网格六帧概率/梯度（含旋转/平移、ragged legal、ignore/free）、四窗口 actor 隔离、空窗口、两次Adam、只读测速模型/RNG重置、显式旧导入/新断点恢复及拒绝失败证据。不是完整CI、nuScenes效果实验或CUDA测速。
