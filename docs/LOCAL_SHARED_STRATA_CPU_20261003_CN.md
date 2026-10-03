# Local 在线 CPU：分桶与共享优先队列修正

本文件补充并替代 `LOCAL_HORIZON_CPU_PIPELINE_20261003_CN.md` 的固定 3+3 调度部分。
不修改任何训练科学合同；仅更换 transient CPU 计算和调度。

## 已收到的服务器证据

两段都取最近 128 个 update，cache hit 均为 100%；不是 cold cache 导致：

| 每窗口耗时（ms） | 原 4 worker | 719202c 固定 3+3 | 差值 |
|---|---:|---:|---:|
| 总耗时 | 118.328 | 121.110 | +2.782 |
| 输入等待 | 1.061 | 3.509 | +2.447 |
| 准备 | 19.286 | 9.913 | -9.373 |
| 候选等待 | 24.585 | 29.861 | +5.276 |
| 抽样 | 9.766 | 17.494 | +7.728 |
| 特征等待 | 21.223 | 21.783 | +0.560 |

719202c 没有取得实测吞吐收益，这两段反而慢约 2.35%。它们属于不同更新和窗口，
不能仅凭差值断言每一项的因果归属。代码中能直接确认的问题是：caller 每 horizon
重新扫描完整候选建立六个抽样桶；固定分池在纯候选阶段闲置另一半线程。

## 本次修正

1. 增加 `swfm_sampling_strata` 纯整数 C ABI 内核，计数/填充两次线性扫描，
   一个 packed int64 buffer 给出原始六个桶的索引视图。候选 worker 构建一次，
   不再在主线程构造三套种类掩码、正负布尔索引和对应复制。
2. caller 仍按相同窗口/horizon/population/positive-negative 顺序执行原 `rng.choice`。
   population、draw budget、importance weights、抽样 ids 和 RNG 消耗不变。
   TRAIN prior 只统计全量 counts，不构建不会使用的抽样桶。
3. 候选的完整重复检查加入有序 fast path：严格递增时无须排序；其他情况仍对
   **全部候选**运行原来的 unique 检查。不是只检查被抽中的 query，不允许漏掉重复。
4. 同样合计 6 个 worker，改成共享容量的优先队列。候选阶段可用全部 6 个，
   ready feature/index jobs 优先于尚未开始的候选；不能抢占正在运行的任务。
   index 父任务与依赖 feature 同优先级且先入队，1/2/6/8 worker 不发生嵌套等待死锁。
5. 异常记录到对应 Future，worker 继续工作；退出可取消尚未运行的任务，运行中
   CPU 任务正常结束。仍由主线程做 Torch/CUDA/latent gather、backward 和 checkpoint。

日志中 candidate capacity=6、feature capacity=6、`online_shared_worker_pool=true`
表示**同一个 6 线程池**，不是 12 个。没有扩大 window batch、source budget、I/O
prefetch、frame cache 或 4 GiB geometry RAM 配额。临时多出的抽样索引只存活一个 batch，
不会写入 persistent cache。80 GiB 限额下不清空 OS file cache，不复制 dataset 多进程。

native ABI 由 2 增到 3，因为新增 C 函数。恢复时只需在已有编译缓存中编译一个新小库；
旧库不覆盖，48 GiB 配额的因果几何 namespace/已有约 30 GiB 文件完全不变。
Torch 模型/checkpoint/train population/cosine/AdamW/RNG 协议均不升级。

## 本地局部计时，不是服务器训练提速

真实编译内核；固定原始候选与 RNG，每个模式 7 次交错重复，各 150 draws，取中位数。
比较旧 `sample_queries` 完整扫描和新六桶路径。新总成本包括每次重新准备 native 桶，
不是只报从 caller 搬走后的计时；相同 ids/weights/RNG 已逐项验证。

| 候选 rows | 旧 draw (us) | 新 caller draw (us) | 新分桶+draw (us) | 新总成本加速 |
|---|---:|---:|---:|---:|
| 5,000 | 113.88 | 67.40 | 110.81 | 1.03x |
| 20,000 | 188.01 | 71.31 | 172.56 | 1.09x |
| 100,000 | 566.99 | 68.59 | 461.11 | 1.23x |

这说明 caller 串行部分明显缩短，但全部抽样计算的缩减有限；不能把 caller 1.69–8.27x
直接当成整段训练加速。也不能据此承诺 L40S 会快多少。服务器常规续训的
`seconds/window` 才是验收依据，无须另开训练/验证实验。

另做了完整 CPU candidate+feature 合成对照（200×200×16、每窗口 32 sources、
4 窗口、同 RNG/ids/weights、先 warm 一次，再交错重复 6 次取中位数）：
固定 3+3 且无 prepared buckets 为 157.585 ms/window，共享 6 worker+新分桶为
156.379 ms/window，只有约 0.77% 的收益。两者都使用本次完整候选重复检查实现，
且均不含 GPU/模型/IO，因此不把它冒充旧提交对新提交的服务器训练对照。
这个证据不支持大幅 CPU 提速，也不值得要求用户为本修正额外中断训练。
下一次自然暂停时可更新；进一步明显加速需要更大粒度的批量采样/编译实现。

仓库全量回归：`945 passed, 10 skipped, 1 deselected`（288.74 s）；
Python 编译与 Bash launcher 语法检查通过，native 使用实际编译库。

## 下次自然暂停时的安全切换（不要求现在中断）

已知当前运行目录：

```text
/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_resume_20261003_183450_837
```

在 OccFM 环境先使用 manager `stop --run-dir <上述目录>`，等待安全完成当前更新并发布
最新 last.pt；再通过 ghfast fetch + ff-only 更新代码，最后 manager
`resume --run-dir <上述目录> --sampling-workers 6 --profile-every 32`。
不要把第 22522 步的日志编号当成停止后的断点，不要退回第 20694 步，不手改原始配方。
resume 创建新输出，读取原 execution contract 并恢复最新 cursor/optimizer/RNG，
跳过已完成 prior 和显式 prewarm。Ctrl-C/manager stop 的 checkpoint 保护仍保持。
