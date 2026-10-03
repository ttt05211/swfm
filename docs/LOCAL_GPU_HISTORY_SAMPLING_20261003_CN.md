# Local 可选 GPU 历史列采样后端

## 范围与潜在负面效果

这是运行时加速，不是新网络。只把已选列的历史坐标映射、uint8 label/visibility/source-membership 读取放到 GPU；在线候选、GT 监督、抽样 RNG、hard renderer 及共享 live source-query 梯度保持原实现。默认仍是 CPU。不改变四历史→六未来、batch4/source128、15 轮整段余弦或冻结 TRAIN prior；不覆盖原实验或重建几何缓存。

不能保证 GPU 一定更快：L40S 的 FP64 运算、H2D 传输和额外 kernel 启动也有成本。每个窗口一次上传原历史字节数组，最多 32 个 query/chunk，工作空间预算 256 MiB（包含保守 scratch 估计），不增加服务器 CPU/RAM 缓存预算。显存不足或预算超限回退 CPU。其余程序异常不会被吞掉。

数值风险是 float64 求和顺序在 floor 边界上的差异。因此：

- inverse/matrix multiplication **仍按原 NumPy 次序**计算，不用 float32/TF32 插值或近似网格。
- 每个点检查保守 float64 rounding bound；边界不确定时重跑**整个原 horizon** CPU sampler，不能仅重算一个点子集改变 BLAS/dense-map 算法。
- 前三个窗口、之后每 128 个窗口逐字节核对 history/flags。不一致直接拒绝当前 optimizer update，不以“指标接近”代替一致性。
- 动态 source membership 用独立 actor/frame key，允许多个 source 拥有同一个历史体素；缺失 registration 保持 UNKNOWN；保留原 observed flag。
- 历史索引和 GPU 数据仅存活于当前 window/batch。禁止跨 predicted pose 复用；不写入 geometry cache。
- 纯 CPU worker 只做 packing；CUDA、source latent gathering 和 autograd 始终属于主线程。

本地 Windows 只有 CPU Torch：可以测试同一 Torch 采样代码的 CPU 执行、边界回退、真实联合 AdamW/梯度/RNG 一致性；**不能据此声称已验证 L40S 的 CUDA 正确性或速度**。仓库另有实际 CUDA 的字节/AdamW 检查，服务器测速入口先运行这些检查。

本地全量回归：969 passed / 14 skipped / 1 deselected；另补 actual native compact 候选→Torch 采样→native CPU reference 的检查通过（CUDA 对应项因无 GPU 跳过）。编译检查和两个 Bash 入口语法检查通过。空窗口不会上传无用历史卷，也不会消耗首次核对配额。

## 一次合并检查与测速

在服务器 OccFM 环境，先按实际 `--run-dir` 安全停旧训练；不要 kill -9。然后通过 ghfast 拉取本分支并 ff-only 更新。以下沿用用户已提供的路径；如果运行目录已换，必须显式改 RUN，不能信号发送给猜测的 PID。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
PY="$(command -v python)"
RUN=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_resume_20261003_183450_837
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py stop --run-dir "$RUN"
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
export LOCAL_WARM_COMPARE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99
export LOCAL_WARM_GPU_FEATURES=1
export LOCAL_WARM_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/gpu_features_$(date +%Y%m%d_%H%M%S)
bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
cat "$LOCAL_WARM_OUT/summary.txt"
```

`LOCAL_WARM_COMPARE` 必须是已有成功测速目录（有 contract.json/records.pt/warm_cache.json/diagnostic_weights.json）；无需提供新数据路径。其数据/配置/namespace 指纹不符合则直接报错，**不自动冷重建**。可替换为用户最后成功的测速目录。

该模式只跑两个独立子进程：当前 native CPU features 与 GPU features，均 batch4/source128/workers6，完全相同初始化、样本、prior。包含实际 forward/backward，但不保存科学更新；不做 dev scoring、batch 扩容、额外 cProfile、自动重训或自动 resume。GPU 启动前的实际 CUDA 小测试和前三窗口验证不计入稳态吞吐。

报告 `GPU_FEATURE_COMPARISON`：速度比、CPU byte 检查次数、boundary/OOM/budget 回退、10% 显存余量。只有全部检查通过、GPU 真正执行了部分采样、无 OOM/budget 回退且总吞吐提升至少 5% 才建议显式启用；否则保留 CPU，不能宣称提速。OS cache/短样本波动仍存在，ETA 不含全量评估和 checkpoint I/O。

## 保持科学状态的续训

仍从所停运行目录**最新** last.pt 恢复，不固定回到 20694 或 22522。只允许性能后端切换，不允许 batch/source/epoch/LR/RNG 变化；resume 自动创建新目录、复用已存在暖缓存、恢复 AdamW/RNG/cosine/cursors、跳过 prior。

```bash
# 仅在 summary.json 中 gpu_feature_comparison.pass_gate == true 时启用。
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py resume \
  --run-dir "$RUN" --column-feature-backend gpu --profile-every 32

# 无净收益或想退回 CPU：在当前新运行目录安全停后，显式 cpu 恢复。
# "$PY" -u tools/real_motion/manage_p0_f9_joint_training.py resume \
#   --run-dir <当前新运行目录> --column-feature-backend cpu
```

新增的 `column_feature_backend` 属于 execution 参数，不进入 scientific identity，不改旧 checkpoint protocol；旧断点默认 CPU，只有显式选择才迁移。progress.jsonl 记录 `column_feature_backend`、GPU fallback/verification 和分段计时。此版本**不加速候选生成**，不应把历史采样的加速比例宣称为整个训练的比例。
