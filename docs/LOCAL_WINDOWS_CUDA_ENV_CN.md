# 本地 Windows CUDA 环境：复用现有安装

## 已确认的环境

- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB，compute capability 8.6。
- 原有独立环境：`F:\anaconda3\envs\yoloe`，Python 3.10.16、PyTorch 2.6.0+cu118。
- 项目解释器：`.venv-cuda-check\Scripts\python.exe`，Python 3.10.16。
- 项目虚拟环境使用 `--system-site-packages` 只读复用 yoloe 的 Torch/CUDA、NumPy、SciPy、PyYAML；没有重新安装 Torch/CUDA。
- pytest 和缺少的 einops/einops-exts/easydict 小型依赖安装在项目虚拟环境，不修改原 yoloe。
- `.venv-selector-check` 仍是原来的 Python 3.12 CPU 环境，没有修改。

2026-10-05 实际验收：CUDA 矩阵反传、真实稀疏修复头的 FP32/BF16 六帧前向、反传和 AdamW 更新全部通过。
上述五组回归在真实 3050 上 **103 passed in 56.24s**，包含 CUDA 字节采样、Graph/异步回读和显存生命周期检查，不再是 CPU-only 跳过 CUDA 的结果。
`pip check` 也通过。此处记录环境功能与回归结果，不代表新方法取得了真实数据精度或完整速度增益。

**这不是完全独立的依赖副本：项目 CUDA 环境依赖现有 yoloe。不要删除或升级 yoloe 后仍假定版本不变。**

## 0xc0000022 的诊断

同一个 yoloe 解释器，使用相同的干净 DLL/PATH，Codex 执行沙盒内在 Python 启动前以 `0xC0000022` 退出；沙盒外正常启动并成功执行 CUDA 前向/反向。
因此本次失败是执行隔离导致的拒绝访问，不是缺少 CUDA 安装，也不需要重新下载 CUDA/PyTorch、换驱动或修改 Anaconda base。

Codex 运行本地 CUDA 检查时，应申请 `require_escalated` 执行该项目解释器/启动脚本。普通终端不需要 Codex 的沙盒权限参数。
若再次出现 `0xC0000022`，停止在同一隔离条件下重试，不执行 `F:\anaconda3\python.exe`，不启动故障 base。

## 使用

在仓库根目录，用 PowerShell 执行：

```powershell
& .\tools\real_motion\run_local_cuda.ps1
```

脚本使用明确的项目解释器，临时设置独立环境的 DLL 路径，排除混入的 Anaconda base/系统 CUDA Toolkit 路径。
命令结束后恢复 PATH 等环境变量；不会激活或修改 base。Windows loader 错误框被抑制，错误改为明确的退出码。

默认运行 `check_local_cuda_env.py`：实际分配 CUDA 张量、矩阵反传、真实 `SparseRepairHead` 的 FP32/BF16 六帧前向/反传及 AdamW 更新，检查 source latent 梯度。
**这是环境功能验证，不是模型质量验收或 FPS 测量。**

运行测试：

```powershell
& .\tools\real_motion\run_local_cuda.ps1 -PythonArgs @(
  '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
  'tests/test_sparse_evidence_repair.py',
  'tests/test_source_repair_pilot.py',
  'tests/test_column_gpu_sampling.py',
  'tests/test_column_execution.py',
  'tests/test_shared_column_evidence.py'
)
```

不要使用裸 `python/pip/pytest`：它们可能解析到被禁止的 Anaconda base。

## 复现项目环境

只有项目虚拟环境不存在时，才使用原独立解释器创建：

```powershell
& 'F:\anaconda3\envs\yoloe\python.exe' -m venv --system-site-packages .venv-cuda-check
& .\tools\real_motion\run_local_cuda.ps1 -PythonArgs @('-m', 'pip', 'install', '-r', 'tools/real_motion/requirements_local_cuda_checks.txt')
```

Codex 下仍需在执行沙盒之外运行；不要覆盖已有环境。这里不会下载 Torch/CUDA。

4 GiB 显存不能照搬 L40S 的 batch。后续真实窗口回放必须明确实际 batch、峰值显存、CPU/GPU 完整边界；本机通过 CUDA 功能检查不代表服务器训练或完整推理已经提速。
