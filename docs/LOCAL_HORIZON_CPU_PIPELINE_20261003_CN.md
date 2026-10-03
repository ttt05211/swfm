# Local 暖缓存训练：按未来帧拆分的 CPU 流水线

> 本文记录 719202c 的初版。服务器回测未提速；后续共享容量/整数分桶修正见
> `LOCAL_SHARED_STRATA_CPU_20261003_CN.md`，固定 3+3 与 native ABI 不变的说明已被替代。

## 本次边界

仅优化在线 CPU 调度；不更换 Local 网络、loss、梯度连接、数据 population 或训练配方。
严格保持 4 历史 → 6 未来、window batch ≤ 4、source budget 128、15 轮整段余弦下降。
科学训练合同与现有 checkpoint/cache 协议不升级；这不是新一轮实验。

服务器已确认额度为 10 CPU / 80 GiB 内存。观测到约 40 GiB anon 与约 39 GiB
file cache，其中约 33 GiB inactive_file；不能把宿主机 `top` 的总内存当容器预算，
也不能把全部 file cache 当作无代价可分配内存。本次不扩大 RAM/cache/prefetch，
不执行 drop_caches、不复制多进程数据集、不重建已有约 30 GiB 的几何缓存。

最近的 128 个训练窗口：约 0.12677 s/window；输入等待约 0.00073，准备约
0.01687，候选等待约 0.02517，抽样/物化约 0.01202，特征等待约 0.02464
s/window。优先优化候选与特征串行等待，而非继续扩大加载预取。

## 实现

1. 原来的一个任务处理一个窗口的六个 horizon，改为每个 horizon 一个任务。
   每窗口的动态历史证据只计算一次，六个任务共享只读 setup。
2. 原 caller-owned RNG 仍按窗口 → horizon 的原始顺序抽样。只生成原始 ids
   与 importance weights；compact voxel 物化搬到 CPU worker，不重抽、不截断。
3. 候选和特征分开排队，合计默认 6 个工作线程（3 候选 + 3 特征），最多 8 个。
   不是每个池 6 个。两名 I/O worker、原来的预取协调线程及主训练线程保持不变。
   OMP/MKL/OpenBLAS 均保持单线程，特征 worker 不再嵌套线程池。
4. 每窗口只构建一份只读 `ColumnHistoryIndex`；其父任务排在所有依赖任务之前。
   1/2/6/8 线程均验证 FIFO 依赖，不会出现 worker 提交子任务再等待自己导致的死锁。
5. worker 不访问 CUDA、Torch latent 或 RNG。主线程按原始顺序 assemble，保留
   columns → same-source future query → motion encoder/decoder 的 live gradient。
6. 持久缓存仍只保存因果固定几何；绝不缓存 learned poses、future labels、抽样 ids
   或在线特征。缓存 namespace 和 native ABI/source 均不变。

`SWFM_COLUMN_CPU_HORIZONS=1` 仅在 native + compact bundle + optimized pipeline
启用时生效；`0` 可退回原 window-level 调度。其他 reference/NumPy 路径保持可用。
`--sampling-workers` 在 horizon 路径表示两个队列的合计预算。

## 断点恢复

本次已安全停止的断点是：

```text
/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_20261003_114256/model/last.pt
completed_update = 20694
next_update = 20695
```

使用已有 manager；只额外指定 performance-only 参数与断点编号校验：

```bash
SWFM_COLUMN_CPU_HORIZONS=1 CUDA_VISIBLE_DEVICES=0 python -u \
  tools/real_motion/manage_p0_f9_joint_training.py resume \
  --run-dir /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_20261003_114256 \
  --expected-update 20694 --sampling-workers 6 --profile-every 32
```

上面的 `python` 是 Linux 上已激活 OccFM 的解释器，不适用于 Windows base。
manager 从旧 execution contract 读取真实路径和训练参数，恢复模型、AdamW moments、
所有保存的 RNG、epoch/batch cursor 与原 77172 步整段余弦；跳过已完成 TRAIN1024 prior
及显式 prewarm，创建新的输出目录，不覆盖旧 checkpoint。若断点不是 20694、旧进程
未停止或科学合同不一致则直接拒绝。仍可用 manager `stop` 或 Ctrl-C 安全结束。

## 验证和计时口径

- NumPy/reference/native 对照：候选集合、全部字段、标签、ids、importance weights、
  特征、caller RNG、loss、参数/梯度、AdamW moments 连续多步逐项一致。
- 生产 native 两队列 + 4 历史在 1/6/8 workers 验证连续 20695–20697 更新。
- 保留既有完整 trainer 的中断/恢复、原子 checkpoint、首次 live forward/renderer
  exactness 测试；新增 expected-update、performance-only recipe、缓存 namespace 测试。
- 空候选不影响 motion 学习；worker 抛错在 optimizer.step 前传播，不保存部分更新。

本地仓库测试结果：`923 passed, 10 skipped, 1 deselected`（264.80 s）。
Python 编译、launcher Bash 语法和 `git diff --check` 通过；其中 native C++ 测试
使用真实编译库，不是用 mock 替代。CUDA/真实数据集集成仍未在本地执行。

本地无 CUDA/真实 nuScenes，不声称已在 L40S 实测提速。恢复后的常规训练日志就是
本次性能验证，不另开测速/训练实验。新增任务数量、两个池 worker 数、history-index
与物化 worker 耗时，便于定位。每 32 步记录分段计时，其他步不插入计时同步。
worker 累加耗时会重叠，不能相加作为 wall time；物化移出 selection 后其字段数值
自然减少，也不能单凭这一项宣称提速。以足够多暖缓存更新的 `seconds/window` 为准，
每轮 dev64 和最终 dev512 保持原协议。
