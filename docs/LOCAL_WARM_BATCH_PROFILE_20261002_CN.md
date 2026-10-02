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

## 一次性 CPU 热点优化 / 旧新路径比较

服务器已测 batch4/8/16/32 吞吐约4.8–4.9 windows/s，batch64虽未OOM但43408MiB reserved超过10%安全余量，batch128 OOM。不能把增加显存占用当作吞吐改善。batch4 的256个窗口约53s：candidate等待18.2s、patch等待10.7s、prepare/render10.7s，是优先优化对象；cProfile是另一个串行化worker的诊断pass，不拿它的累计时间当正常并行占比。

本次集中落地以下保持训练行为的优化，默认用于完整Local训练：

- 一个窗口的六个horizon共用因果历史索引：历史pose/registration逆矩阵、source membership；只索引本次被采样actor，紧区间bool表总量≤8MiB/窗口，超限走有序searchsorted。不存GT、learned pose/候选/特征；每次prepare重建，不改原几何磁盘cache namespace，不增加20k窗口特征缓存。
- 每窗口合并六个采样任务，共用索引，保留主线程逐窗口/逐horizon唯一RNG的原抽样和importance权重。稀疏XYZ生成减少N×P×P×Z×3整数/浮点临时数组；逐坐标float64乘加顺序与reference一致。边界、membership、重复anchor不同class的flags均逐元素验收。
- 候选构建把相对ego pose逆矩阵移出每actor追加循环；历史raster仅要BEV支持与Z范围时不重复做3D unique。支持dilation/argwhere、候选顺序、上下界、标签完全相同。刚生成并验证的plan直接进入内部标签函数，避免立即重复validate；公共action_targets仍完整验证。
- 整个window batch只读回一份XY/yaw CPU数组；保留原始source query计算图。完成首次原设备renderer exactness、且cache含完整background后，窗口的CPU准备/渲染可在同一个有界worker pool并行；cold/首检仍在原CUDA-owning主线程。workers不跑forward/CUDA/latent gather/RNG；assembly和source latent gather仍在主线程。
- source feature gather的actor合法性/非空判断使用已经存在的CPU plan，减少每horizon小CUDA reduction的同步；index_copy仍可微且不detach source queries。未改变模型参数、目标、LR、window/source budget或resume合同。`FULL_JOINT_REFERENCE_CPU=1`可选择诊断reference路径。

CPU回归覆盖4/6历史、1/4窗口batch、多步AdamW，比较逐字段候选、逐元素采样特征、labels、抽样RNG、loss、source梯度、optimizer state和参数更新；真实完整预热几何验证并行准备和冷路径首检。CPU通过不代表L40S吞吐或GPU舍入已经实测，服务器速度以以下一次性比较为准。

```bash
LOCAL_WARM_COMPARE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99 \
  bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

比较创建新目录，复用旧128+8身份/顺序、records、TRAIN16权重、checkpoint快照（如有）和geometry cache，不重建prior/cache，不重扫64/128。依次独立进程测 reference batch4、optimized batch4/8/16/32，最后只profile建议档。参考路径关闭共享索引、并行warm准备、批量readback和CPU actor判断；renderer的anchors转换移出source循环等公共小优化两边共有，因而是保守的旧/新recipe比较，不声称精确复现旧提交的每个CPU调用。

`summary.json/txt`同时打印固定batch4吞吐比、推荐档吞吐比、实际显存/命中/阶段时间和条件性15轮纯训练ETA；推荐只来自优化档，within3%仍优先小batch，保留10%显存余量。正常吞吐无cProfile；OS文件缓存/窗口代表性仍会影响数字，不把测得的全局收益预先保证为某个倍率。若异常，保存部分摘要、停止，不自动启动正式训练或反复重试。

## 第二轮 CPU 优化：候选/采样，不再扫 batch

5722b80 的实测 batch4 吞吐6.034 windows/s（原reference4.921），256窗口约42.4s：候选等待18.49s、特征等待8.04s、prepare/render4.86s。不能把cProfile中的线程等待或worker累计时间当正常运行占比。

- TRAIN仍构建完整候选与真实edit标签、保持原抽样与importance权重，只延迟12维context计算到已抽中的普通ColumnPlan；保留全候选support centre/age，绝不从小子集重算中心。wrapper禁止直接读取未materialize的context，支持空/倒序/负索引/重复索引（正常训练不抽重复）。异常尺度回退完整context验证，pose非有限直接报错。
- history/ego-only frontier先给潜在XY，CURRENT baseline只在该支持检查free；静态支持狭窄时局部索引，广覆盖时继续完整连续扫描，避免把索引拷贝当优化。新geometry可存可选XY，旧cache缺字段直接派生；无重建、namespace变更或GT/learned信息落盘。
- source dilation使用与scipy默认完全相同的4-neighbor cross stencil；不变成8-neighbor，不减小支持。候选unique验证可用有界int64键，Python-int先验证乘法范围，超范围/特殊dtype回退原字节键，重复仍报错。
- float64变换后的**新**坐标工作区原地执行减origin、除step、floor，避免多份大临时数组；三轴分别比较取代N×3临时bool+short-axis reduction。不改变矩阵乘法/浮点精度，不修改共享历史points/occupancy。TRAIN正负bucket用一次changed归约，population顺序、抽样RNG、权重保持一致。

完整训练默认启用；网络/目标/LR/预算/历史4或6合同/optimizer与RNG恢复均不变。CPU比较新增`previous_b4`（上一轮优化，关闭本轮kernels）与`same_batch4_speedup_vs_previous`。公共合法性检查的小优化两边共享，比较保守，不声称逐调用复刻旧提交。

```bash
LOCAL_WARM_COMPARE=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/warm_speed_20261002_082412_3c09a99 \
  LOCAL_WARM_MAX_BATCH=4 bash tools/real_motion/run_p0_f9_joint_local_warm_benchmark.sh
```

只测reference/previous/new三条batch4，每条同一冻结样本、prior、初始化与磁盘cache；不测8/16/32/64/128、不重建prior/cache、不启动full15。最后一小份profile另算，不混入吞吐。新增full-grid32-source、宽/窄支持、旧/新cache字段、极限voxel边界、跨int64范围unique和多步loss/gradient/AdamW/RNG一致性回归。Windows CPU合成计时只用来排除明显退化，不能替代真实L40S吞吐；没有保证新的15轮耗时。

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
