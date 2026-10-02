# Local 暖磁盘缓存 / batch 计时与四帧历史修正

## 先跑一个命令，不启动正式训练

在服务器激活 `OccFM`，仓库快进后执行：

```bash
bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

读取 full20430 的唯一有序身份，固定随机抽取128个 TRAIN 窗口，加8个不重复的高 source 压力窗口。先完整写入这些窗口的因果几何缓存，再启动彼此独立的 GPU 子进程测旧池/常驻4线程/常驻6线程，随后 batch=8/16/32/64/128（source budget=32×batch），OOM即停止扩容。每个子进程从同样初始化或只读快照开始；OOM 仅终止本次试跑，其他错误不吞掉。预热、压力样本和 cProfile 单独计时，不混入普通吞吐。若想更短的第一趟，可设置 `LOCAL_WARM_SAMPLE_WINDOWS=64 LOCAL_WARM_MAX_BATCH=64`。

输出 `summary.txt`、`summary.json`、`cpu_profile.txt` 与 `profile_stages.json`。包含实际 batch、source 数、暖缓存命中、显存 allocated/reserved、CPU主线程/候选等待/特征等待/输入等待、CUDA stream 时间，及吞吐最优与最大安全 batch（留10%显存）。CUDA stream 时间不等于 kernel-active 利用率，不能与重叠CPU时间相加。CPU worker 累积时间不是 wall time。15轮估计仅是样本代表性成立时的纯训练估计，排除完整首次建缓存、评估、存盘；OS文件缓存与256MiB帧缓存仍可能影响小样本。

磁盘几何 RAM cache=0，防止把小样本全在RAM的速度冒充完整20430训练速度。原geometry根目录跨协议共用48GiB硬限额，满额/缺少安全剩余空间会明确报错，不删旧条目、不静默当作warm。记录子集文件限1GiB；试跑更新不保存科学checkpoint，不修改旧 `last.pt`。诊断 TRAIN16 prior仅用于速度，不作为正式训练prior；full仍使用固定TRAIN1024。最大安全batch是**最大已测**安全值，不是完整20430永不OOM的保证。

## 中断接续 / 大 batch SDPA 限制

时序 block 的 attention batch 是 `source_count × stem_height × stem_width`，不是窗口数。20×20 tube 经 stride-2 stem 后为10×10；当 source 多于655时可能越过 CUDA SDPA 的65535维度限制并报 `invalid configuration argument`，不是 OOM。新版在超过该值时仅按独立 batch 维分块；不切时间、不混 source、不detach，不新增参数，已有 checkpoint state_dict 和四/六帧合同均不变。小 batch 原路径不变，CPU测试比较输出与输入/参数梯度；CUDA回归测试须有GPU才运行。依据：[PyTorch #142228](https://github.com/pytorch/pytorch/issues/142228)、[PyTorch #146704](https://github.com/pytorch/pytorch/issues/146704)。

测速每完成一档都会更新 `summary.txt/json`。非OOM错误仍失败退出，但保存 `failed_partial` 摘要；child另存 `failure_<phase>_<trial>.json`，不会把kernel错误冒充OOM或在污染的CUDA上下文里重试。人工接续仅复用已完成trial，重新启动缺失trial的独立子进程；不恢复正式optimizer。校验records内容/有序identity、base/snapshot、TRAIN info、runtime config/几何namespace、prior与原始计时行。旧v1结果注明复用，新结果记录Torch/CUDA/GPU/model代码hash，不把不同代码版本宣称为严格同软件对比。

```bash
# 使用本次实际失败目录；不重建缓存/prior，不重测4/8/16。
LOCAL_WARM_CONTINUE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99 \
  bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

若只想立即补齐现有成功结果与最小CPU热点报告、不再测32/64/128，加 `LOCAL_WARM_FINISH_EXISTING=1`。报告会明确标记 `completed_existing_trials_only`，不宣称完成全容量扫描。接续摘要的elapsed仅包含本次接续时间，旧trial的时间保存在各自measurement中。原completed trial JSON、records、contract与checkpoint不覆盖；仅更新生成报告。

兼容原版 `warm_cache.json`：原 `CausalGeometryCache.stats()` 只有统计计数，没有 `directory`。新报告补充实际目录和namespace；旧报告保持原样，接续从原合同计算namespace并要求对应磁盘目录非空，child仍逐窗口按原始因果输入hash校验缓存内容和身份，所有窗口必须命中后才允许该batch的训练更新。缺失目录不新建、不重新预热，也不把统计计数当作内容完整性的证明。回归测试通过真实cache/prefill生成报告，并同时覆盖原计数-only格式和新格式，而非手造不存在的schema字段。

## 帧数验收：此前六帧确有协议差异

上游 `upstream_occfm/forecast/datasets/nuscenes_dataset.py` 在 cache path 根据 `HIST_LAST=4` 把六槽位的前两个 latent 与轨迹置零。此前本仓库只有 trajectory 前缀置零；V18 encoder、flat features、column patches、历史ICP、静态记忆和frontier实际仍读六帧 occupancy。因此旧 E14 与所有旧六帧结果不应表述为标准四帧协议。

新版 full launcher 新训练默认严格四帧 `[-1.5,-1,-0.5,0]s`，预测六帧 `[0.5,1,1.5,2,2.5,3]s`。保留原六槽位cache的最后四帧，encoder实际只编码四帧（不是六帧zero-padding）。flat ABI保留，但最早两帧offset/valid/velocity与跨界segment置零，并重建第一保留帧velocity，不复用含被排除帧的平均速度。KTA与t0 source path只依赖最后两帧，GT labels与窗口identity不变。raw只加载四帧 occupancy/观察掩码/pose；ICP、记忆、footprint、candidate age、sampler与column attention均同步四帧。历史pose的12槽位trajectory仅保留上游兼容ABI，不被joint模型作为额外观测消费。

四帧checkpoint有独立 `full_train_history4_v1` 合同和缓存namespace；六帧与四帧不能混读/续训。旧六帧读取与resume保留，launcher续训自动采用checkpoint帧数，不擅自丢掉旧进度。冻结E14仍六帧，只能作**非同输入预算**历史参考；论文四帧比较需训练四帧 V18-only 对照（full可用 `FULL_JOINT_PAIRED_CONTROL=1`，会增加成本）。本次没有伪造四帧基线或GPU测速数字。

## 完整暖缓存入口（不是小计时默认行为）

```bash
FULL_JOINT_PREWARM=1 FULL_JOINT_HISTORY_FRAMES=4 \
  FULL_JOINT_PROFILE_EVERY=32 FULL_JOINT_SAMPLING_WORKERS=6 \
  bash tools/real_motion/run_p0_f9_joint_causal_columns_full.sh 15
```

仅在计时后决定开正式训练；根据报告另设 `FULL_JOINT_WINDOW_BATCH` 与 `FULL_JOINT_SOURCE_BUDGET`。完整prefill有真实首次成本，保留模型参数/optimizer/RNG，未来GT不进入缓存。暖缓存仅固定历史几何，不缓存模型pose、online候选、标签、特征或梯度。批量/上限改变会改变每轮优化步数和完整余弦长度，所以旧断点严格拒绝不一致recipe；不能为了快默默改变同一实验。
