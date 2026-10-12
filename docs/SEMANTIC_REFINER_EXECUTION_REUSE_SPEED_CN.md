# Semantic Refiner 执行复用：只读 A/B

本轮只优化执行，不改网络、目标、数据预算，不接正式 trainer，也不干扰正在跑的 loss500。新文件在 sibling `refiner_execution/`，不修改 `refine/`、`loss_controls/` 或其恢复指纹。既有 TRAIN/VAL 几何只读复用。

## 四种执行

| 路径 | 内容 |
|---|---|
| reference | 当前完整 H6 Refiner |
| query_reuse | 固定 query 坐标/phase 复用、整幅 FP32 特征只拷贝一次 |
| projection_reuse | 上项＋每 forward 复用历史/未来逆矩阵、coverage 上传、stencil 常量 |
| memory_h6 | 上项＋六时距×四历史 static memory warp 合批 |

坐标 cache 只有两种 shape/device/chunk 条目，不含权重、历史内容或 learned features。动态 memory splat 与 Transport source-overlay 顺序保持；Q 的逐时距线性读出先不改，以免同时改变 GEMM 形状。future inverse 仅当前可微 forward 复用，不能把预测 ego/学习投影存入历史缓存。没有减少 future/voxel、降频 Lovász 或改变 source budget/global batch。

### BF16 反传陷阱与修复

最初直接把 `x.float()` 搬到 query 循环外，前向通过但严格梯度 FAIL：原版每块先把 FP32 梯度转回 BF16，再累加；朴素复用会先汇总 FP32 梯度再转 BF16。这不是无损改变，初版不能推广。

目前使用共享前向 storage、每块独立 cast-back VJP 的 `_SharedQueryCast`：前向只做一次 FP32 拷贝，反向仍按原 dtype 转换每块梯度，不再以“大约相近”为由放宽检查。即便数学/小模型通过，CUDA scatter/grid-sample 仍可能存在运行漂移，全尺寸必须验证。

## 测速口径

冻结**原两轮 Refiner epoch2**，不是正在跑的 loss500。每路同一权重、Adam、RNG、原 TRAIN4086 首4个 batch 分组（最多16窗口），一轮 warmup 后两遍正序/逆序测速。执行真实 full6 CE+Lovász+source geometry/existence+ego、反传、clip 和临时 Adam；临时更新全部丢弃，模型/优化器/RNG 恢复，原 checkpoint/SHA/summary 不写。该探针不是扩大训练或 DEV 上学习。

target preparation 包含，共享输入 I/O/几何准备不含；warmup、audit、Profiler 不含。CPU 数值快照在计时外，不占额外GPU常驻显存。严格前向(loss/完整六帧logits)、所有参数梯度、warmup/多步 model+Adam、baseline 自身 repeat、两遍速度和≥10%显存余量分别报告。完整前向 audit 覆盖 warmup 首窗口全部六帧，而不是证明全部 population 逐字节一致。

任何数值 FAIL 留在报告中，不过滤测速样本、不放宽预算、不自动选最快接训练。梯度预算 rtol=3e-5/atol=2e-7；loss预算2e-6/3e-7；model+Adam 延续2e-4/2e-5。`pass_gate` 还要求 warmup 首窗六帧 argmax 无变化。所有完成路径速度均可报告，但不能把失败路径称无损。可选 Profiler 单独测 reference 和最快完成的 candidate，不因选它做profile就选成正式后端。

中断只丢弃临时 probe，不保存模型步数；重新试需新输出目录，不覆盖原 diagnostics。结果以 `speed.json`、`summary.txt` 为准；通用只读 runner 的 `results.json` 不是新 DEV 质量评估。

## 本地初筛（不是服务器收益）

RTX3050、实际8×8×4小网格 Encoder/STWM/Refiner/几何/CE/Lovasz/反传/Adam，query chunk刻意设17，不代表真实200×200×16 workload。保留逐块cast-back后：reference0.13221、query0.13010、projection0.12122、memory-H60.12239秒/窗；投影复用约1.091×。梯度和前向三候选均通过，baseline重复状态一致。memory-H6没有两遍都更快，gate仍FAIL，说明“合批越大越快”不成立。**不能外推L40S提升9%或正式FPS。** 朴素cast复用的早期14%–20%速度伴随梯度FAIL，不作为有效结果。

## 服务器命令

等 loss500 结束再跑；脚本若检测到 loss500/Refiner 训练会退出，不停止任何进程。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git switch feature/shared-history-execution-speed-v2
git pull --ff-only origin feature/shared-history-execution-speed-v2
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/refiner_exec_speed_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/refiner_execution/run.sh --profile
cat "$RUN_DIR/summary.txt"
```

默认源：`outputs/p0_f9_shared_history/refiner_probe_2ep_20261012_184856/train/epoch_0002.pt`；不同实际路径用 `REFINER_CKPT`。`PROBE_STEPS=8` 可显式改诊断样本数，默认4即可，未自动执行更大探针。输出 Profiler trace/top30 CUDA/CPU operator 可定位 grid_sampler backward、scatter/sort/copy 等是否仍为主瓶颈。训练速度不是六帧推理FPS，本探针未重算正式FPS或DEV512质量。
