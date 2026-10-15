# dense-valid＋共享 source：完整 TRAIN 两轮双卡实验

## 固定设定

- 随机初始化完整 Shared Encoder／STWM／SemanticRefiner。E14 文件只提供结构元数据，**不导入任何预训练权重**。
- 原 TRAIN20430 两轮，原 seed20261011、顺序、全局 B4/source128 分组及两轮完整余弦不变。原始 source 全部保留，超预算的单窗不截断。
- 仅 full6 CE＋Lovász；无 ego 头、ego loss、motion／existence 辅助 loss，无导航条件。
- TRAIN 的历史 known 为 dense_valid；OCC 评估保留原 lidar-known；STC 使用完整公开语义＋dense-valid，不输入 GT visibility。
- 默认 source_policy=shared_pinned：固定原 t−1/t0 匹配及两帧 KTA，只向前补齐身份，不做多帧速度拟合。
- 两张 GPU 切分**同一个全局 batch**，不是每卡 B4。梯度按全局窗口数归一化再 SUM，之后全局 clip5 和 AdamW；不缩学习率，也不因双卡扩大 batch。
- 只用相同的 remaining64，不是最早 DEV64 或 DEV512。每2000个**全局 optimizer update**保存快照并评 OCC+GT／STC+GT，结束追加评估。两卡各32窗，先汇总整数计数再计算指标，不平均两卡 mIoU。不会自动选 best。
- 因此这是 dense_valid＋共享 source 的组合实验，不应写成只改 known 的严格单变量消融；SOURCE_POLICY=baseline 可显式保留旧关联，默认不额外运行第二组。

## 正式接口，不搬实验的临时绑定

`real_motion/shared_source_track.py` 是从 ab8339b 提取的纯历史数据构建接口，输出 component 索引、原始质心／时间、track_valid、原版 pair_velocity、registration 和 registration_valid。

`tools/shared_source_track/adapter.py` 提供两个明确入口：

- TrackProvider：缓存训练／OCC 评估路径。从原只读缓存取当前 component、原几何和原 GT Strong 背景；**不执行旧关联构建器，也不消费旧 registrations**。新 track 构造一次，STWM 与 feature memory 都从它的 registrations 读取。
- SharedTrackPredictor：实时 GT／外部 Pred／内部 head 推理接口。四帧 component 直接产生新 track；Strong/CLEAR 读取同一个 pair_velocity，不再匹配。每次依据传入的未来 ego 独立构建背景、prior 和 memory；不能把 GT 背景拿给 Pred 用。此轮训练及定期评估只运行 GT ego。

STWM 的 source_valid 仍依赖 registration 是否通过；ICP 拒绝不能被 track_valid 覆盖。known/observed 不与这两个标记混用。

旧 record 的 KTA、anchor、中心先进行2e-4m兼容性核验，再**保留原数组及转换顺序**，不会用 center32＋KTA32 覆盖旧 anchor。该容差继承参考实验，不能证明所有样本 bitwise 相同；实时小网格回归另外检查了 anchor/KTA 原值和六帧 Strong 背景一致。

## 缓存与时间

旧 CCR TRAIN/VAL 缓存是 require/read-only，不重建、不覆盖。关联派生量使用新的 `cache/shared_source_pinned_v1/rank0|rank1/<namespace>`；命名空间绑定新策略、几何配置、实现 SHA。每卡最多8GiB磁盘／128MiB内存，用独立 writer 根避免多进程覆盖；满后只停止接纳，不删除旧数据。

每个 rank 的新缓存根持有进程 lease；不能同时有第二个训练任务写同一根。如需同时跑另一组，设置不同的 TRACK_CACHE，而不是共用 writer 根。旧缓存只读不受此限制。

新缓存只含历史 track、registration、原质心和时间，不含特征、目标、未来 pose 或预测。通用缓存哈希的 future_poses ABI 槽使用固定 marker，不使用未来姿态。原始 observed 仍参与原缓存身份，但训练学习特征的有效性在读取后独立重建为 dense_valid。

首次遇到窗口时按需生成新 track，下一轮／恢复可复用。不是重新缓存整个占据数据集。首轮冷构建可能抵消一部分双卡收益。

两卡不保证两倍速度：逐窗 renderer、CPU 准备、样本 source 数量差异、同步及通信均有开销；单窗全局 batch 时另一张卡必须参与梯度汇总但没有本地 forward，不复制样本填满 GPU。保留原 BF16 与 H6，不采用之前未通过的执行后端。

## 服务器启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export CUDA_VISIBLE_DEVICES=0,1
export SOURCE_POLICY=shared_pinned
export DENSE_DDP_DIR="$PWD/outputs/p0_f9_shared_history/dense_valid_shared_full2_ddp_$(date +%Y%m%d_%H%M%S)"
unset DENSE_DDP_RESUME
bash tools/dense_valid_ddp/run.sh
```

不做额外首检训练；脚本只检查解释器和两张 GPU 是否可见。要求两张 GPU 位于同一台机器、同一个进程可见，分别独立容器的两张卡不能直接用这个命令。

旧 paired workflow 仅提供64窗名单：默认 `outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954/clean/results.json`。不导入它的模型、噪声参数或优化器。

恢复同一个新实验：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export SOURCE_POLICY=shared_pinned
export DENSE_DDP_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/这里填刚才的实际目录
export DENSE_DDP_RESUME=1
bash tools/dense_valid_ddp/run.sh
```

`last.pt` 每64个全局更新及安全停止时保存，含两卡 RNG、Adam、原余弦及游标；另存 epoch_0001/0002 和 update_002000 等快照。评估可从各 rank 的完整窗整数计数恢复，不重评已完成前缀。输出包含 summary.txt、results.json、progress.jsonl 和 eval_update_*.json。原 checkpoint 与各实验目录不写。

## 本地验证边界

已用真实两进程 CPU/Gloo 验证全局均值、3/4/1窗分片、空 rank、全局 unused 参数和 Adam/RNG恢复；用真实小网格网络验证原 H6 反传与新 runner 的保存／续训／最终64窗汇总。共享 source 另有 CPU/CUDA 小网格回归，包括拒绝配准的有效性、旧关联禁用、原 KTA/anchor、独立未来背景、缓存命中和整数续评。

本地只有一张 CUDA 卡，**没有实际双 CUDA/NCCL 性能结果，也不保证真实数据涨点**。服务器吞吐与质量以新日志为准。
