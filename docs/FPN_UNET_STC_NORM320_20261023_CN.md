# 当前完整 FPN＋U-Net 的 STC 归一化适配

独立实验，不修改正式模型、旧缓存、GT 推理或刚完成的 age 候选。没有本地最新科学权重，代码测试不等于已验证涨分。

## 固定方案

- 来源：完整 TRAIN20430×2 的最终 `epoch_0002.pt`，严格拒绝旧 norm192 或 20% pilot 权重。
- 原 DEV64 必须是 64 个不同场景。按原固定顺序前32场景仅用 STC 历史拟合，后32场景评估；不是随机划窗，不让同场景跨组，不挑选分数较好的子集。
- 仅更新 U-Net 内 10 个 GroupNorm 的 γ/β，共320个标量。FPN、WM、卷积、lift、importance、empty 全部冻结。
- 历史四帧 STC 去除动态语义投票，利用已知历史位姿与任务给定的未来 ego 位姿，投影到0.5秒网格。静态占据至少三票同类、free必须四票；每类最多512个均匀抽样点。不读取未来占据、GT mask、GT实例或 Moving 标注来拟合。
- 拟合损失为上述静态共识点 CE；整张 U-Net 前向、仅损失选点，不裁剪改变 GroupNorm 统计。AdamW lr0.001、wd0、clip1，固定200步余弦衰减到0.01倍，轮转32窗，不按验证指标选 checkpoint。
- 仅在内存暂存拟合32窗0.5秒的原 prior/memory，约1.8GB CPU RAM，不创建新 learned bank，不存六帧上游张量或训练图。继续运行时可重新构建这段小缓存。
- 后32窗共用一次原 FPN/WM/几何，比较原 Transport、原 U-Net、适配 U-Net。所有六帧预测结束后才读取未来 GT 计分，报告完整网格1/2/3秒 mIoU、IoU、Moving，以及正负差值。**后32窗与此前完整64的数字不能直接相减**。
- 不叠加 age、后验 bias、静态补洞或 KTA 改动，不扫学习率/步数。不自动推广到全集或部署。

这是**无标签目标域适配**，不是原始零适配 Camera 推理；未来 ego 仍为任务声明的 GT 条件。后32窗与适配场景隔离，但原64已用于多轮研发，不宣称新的独立测试集。静态共识也可能继承 STC 的系统性错误，且会影响动态类别，必须同时检查 Moving，不能承诺收益。

## 服务器运行

同步本次代码后，在原仓库 OccFM 环境运行（入口不停止其他进程）：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export STC_NORM_GPU=0
export STC_NORM_RESUME=0
export STC_NORM_DIR="$PWD/outputs/p0_f9_shared_history/fpn_stc_norm320_$(date +%Y%m%d_%H%M%S)"
bash tools/fpn_unet_stc_norm/run.sh
cat "$STC_NORM_DIR/summary.txt"
```

自动查找唯一已完成的 `fpn_unet_full_2ep_*` 目录。若找到多个，明确设置原两轮训练目录，不按修改时间或分数自动选择：

```bash
export FPN_UNET_FULL_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际完整两轮训练目录
```

固定200步，并非重训主体两轮。输出目录有独立来源快照、`norm_last.pt`（仅 GN 参数及本实验 optimizer）、`contract.json`、`evaluation.json`、`counts.json`、`summary.txt`。原 checkpoint 不写入。

按一次 Ctrl-C 会在完整优化步或完整评估窗退出；拟合每20步及退出保存 patch，评估逐窗保存整数计数。若在拟合输入准备时停止，仅保留协议，恢复会重新准备输入：

```bash
export STC_NORM_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/本次适配输出目录
export STC_NORM_RESUME=1
bash tools/fpn_unet_stc_norm/run.sh
```

适配额外成本不计为原冻结推理 FPS；这里只做质量验证，不运行新 FPS 测试。
