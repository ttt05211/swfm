# Local 编译 CPU 内核（可选执行后端）

本轮只改执行实现，不改网络、loss、候选集合、标签、重要性权重、采样 RNG 顺序、学习率或断点合同。默认仍为 NumPy；显式启用 `native` 后，失败直接报错，不把 NumPy 回退伪装成编译内核测速。

## 编译的部分

- generation 当前 free 列筛选、静态历史 mismatch 列筛选。
- 完整列的 ownership/fallback/KEEP/ADD/REMOVE 合法性。
- 动态 source 的整数 support 膨胀（原来的 4-neighbor cross，而非 8-neighbor），顺序和 Z 边界。
- 未来 GT 编辑监督、正负桶识别（GT 仅用于监督和原 TRAIN 采样）。
- 历史语义/可见性/成员索引读取、重复 anchor 展开。

所有 SE(2)/ego 变换、float64 减法/除法/floor、矩阵乘法顺序仍为原 NumPy。C++ 无浮点运算、RNG、全局 scratch、额外线程或 CUDA。ctypes 在调用 C ABI 时释放 GIL，现有四个 CPU worker 可独立执行；GPU forward、source feature/梯度链接、RNG 抽样仍在原线程。完整候选不会为提速而截断。

## 依赖与安全

需要服务器已经存在 `c++`、`g++` 或 `clang++`，也可用 `CXX` 指定编译器可执行文件；不接受混入 shell 命令的编译器字符串。不自动安装软件。无 NumPy/Python 开发头文件、PyTorch extension、Ninja 或 CUDA toolkit 依赖。首次编译/ABI 检查在启动阶段，不在 worker 或稳态计时里。

源码位于 `real_motion/native/column_cpu.cpp`。编译缓存按源码 SHA256、ABI、编译器版本、flags、系统/架构分目录，加载前检查 binary SHA256。默认缓存目录为 `outputs/p0_f9_joint_causal_columns/native_cpu_cache`。损坏或不完整的 artifact fail-closed，不覆盖原科学数据/几何缓存。Windows 用 MSVC 无 CRT/SDK 的整数 DLL；Linux 用普通 C++ shared library。

输入 dtype/shape 在 Python 检查，native 再检查坐标和 flat bounds。历史输入只读，每次调用独立输出/每个 horizon 独立 scratch；数组引用保留到 C 调用返回。不得在 worker 运行时切换后端环境变量。

## 服务器一次性对照

先停止训练，激活 OccFM，更新当前分支。复用已存在的冻结诊断目录：

```bash
LOCAL_WARM_NATIVE=1 \
LOCAL_WARM_COMPARE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99 \
LOCAL_WARM_MAX_BATCH=4 \
bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

脚本先编译、运行整数/完整候选/patch/多步 AdamW 一致性测试及可用的真实 CUDA 梯度检查，再分别测 `optimized_b4`（最新 NumPy）和 `native_b4`。两者都用 batch4/source128/workers4、同样的记录/prior/初始化和 warm hits。只跑两种后端，不重建 prototype、几何缓存或 prior，不做 batch 扩容，不启动 15 轮，不保存科学训练更新。

`summary.txt` 的 `NATIVE_CPU_COMPARISON` 给出真实 speedup、编译 artifact、实际调用计数；各 trial 保留原 host/CUDA-stream 分段计时。cProfile 仍单独执行，不算吞吐。服务器端 GPU 一致性测试未完成前，不宣称 CUDA 路径通过。

只有对照通过且服务器吞吐值得切换时才考虑 `FULL_JOINT_CPU_BACKEND=native`。NumPy/compiled 的 batch、source、history 和 schedule 保持相同，允许原合同的断点恢复；**不能**借此把旧六历史断点改成四历史，或静默改变 batch/source。后端开关默认 `numpy`，本脚本不自动恢复/启动训练。

编译后的微内核加速不能等同整轮加速：未编译的 float64 transform、渲染、候选 Python orchestration 和 GPU 仍有耗时；以服务器同样本配对吞吐为准。

## 完整 CPU 优化包（ABI 2）

两项主要优化同时交付，不改变完整候选/标签、抽样顺序、重要性权重或训练配方：

- **TRAIN 两遍扫描**：第一遍用 XY、kind、actor、class、uint64 竖向支持掩码扫描全体候选，得到原顺序的有效候选和正负桶；原主线程消费原 RNG 抽样后，第二遍只为选中行生成 flat/base/fallback/legal/target/context。所有候选仍参与分桶，不是减少难负样本或先筛选高置信候选。Z>64 的非当前合同走原完整实现，不截断竖向格子。
- **批量 native 调用**：每个 horizon 一次处理全部动态 source 的整数 support；stamp 工作区只清一次 BEV，但保持原 cross 膨胀、边界和 XY 顺序。每个 patch/map chunk 一次读取全部历史帧，通过指针保留独立的原整数索引，不再复制一个 F×N×3 大数组；坐标变换仍逐帧使用原 float64 运算顺序。

同批完成相关冗余消除：复用一个窗口内不可变的对齐历史证据（不复用预测 XY/yaw）；TRAIN prior 完整 voxel/action 计数直接在扫描中累加，不生成全体列数组；日志用一次 detached scalar transfer，减少逐项读回的同步，loss/梯度和非有限值检查不变。`diagnostics_readback` 会单独计时，不能把原 loss 内的 GPU 等待移动后误报成计算量下降。

临时内存有界：support 工作区由完整 source 体素数给出安全上界，历史批量索引预算 32MiB、dense map 每 chunk 最多 65536 点，超过索引预算保留原路径，不删候选。`compact_descriptor_bytes` 仅是被扫描有效候选的紧凑描述大小，**不是**整进程 RAM 或峰值临时内存；不增加持久 GT/预测/feature 缓存，不重建几何缓存。

用下列模式一次比较上一版 native 执行与整包 native 优化：

```bash
LOCAL_WARM_NATIVE_BUNDLE=1 \
LOCAL_WARM_COMPARE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99 \
LOCAL_WARM_MAX_BATCH=4 \
bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

只启动 `previous_native_b4`（bundle=0）与 `optimized_native_b4`（bundle=1）两个配对子进程，batch4/source128/workers4 不变；同样的已冻记录/prior/初始化、严格 warm hits、RAM 几何缓存为零。旧 native 函数保持不变作为对照，只有新优化分支使用 ABI 2 新函数。编译、合成一致性检查和独立 cProfile 不算稳态吞吐；不会重建 prior/cache、扫描大 batch、启动 15 轮、覆盖旧报告或保存科学训练更新。

`NATIVE_BUNDLE_COMPARISON` 同时报告整步 speedup、完整/紧凑/物化候选数及真实调用计数。CPU 微测试的候选加速比不等于服务器 CUDA 端到端加速比；以此配对结果决定是否启用。默认后端仍 NumPy；显式 native 下默认启用 bundle，可用 `FULL_JOINT_CPU_BUNDLE=0` 对照回退到旧 native，编译/ABI 故障不静默回退。
