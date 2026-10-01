# 一阶段 V18 Transport + Causal Columns 小规模 screen

本版本只回答「已有生成/refine 是否能与 V18 一阶段共同优化」。不扩展新物体类别，不重写 V18 renderer，不运行 full4369，不把未收敛的随机初始化实验包装成正式方法。

## 结构与梯度

```text
六帧历史 occupancy / source mask / motion features
  → 原 V18 spatial-temporal encoder → 六个 future source query
       ├→ 原 XY / yaw / existence heads → 原硬 SE(2) renderer
       │                                  ↓（几何 stop-gradient）
       │                         当前预测的候选 / ownership / fallback
       └→ 小型 source projection ─────────────┐
                                             ↓
历史 full-Z column patches → 共享 column encoder + cross-attention
                       ├→ frontier generation ADD / KEEP
                       └→ static / source refine ADD / KEEP / REMOVE
                                    ↓
                          原 ownership-constrained compositor
```

动态 refine 使用 **同一 source、同一 horizon** 的 live future query，不 detach。静态/refine 和 frontier 没有动态 source 时输入零向量；projection 无 bias，不伪造 source。它们仍共享 column 网络，但不声称每个静态任务都通过 V18 的动态 encoder。历史语义输入、source association、ICP 均为因果证据。

硬 renderer、整型体素索引、候选位置与所有权截断梯度；并非整个 V18 冻结。column loss 可更新 source-query decoder / encoder，原 motion loss 直接更新 XY/yaw/existence。没有对不可微坐标假称端到端。

这不是 GAST 的 Gaussian 场景网络或 GSCA 复刻；这里只采用一阶段多任务共同优化的训练组织方式。当前 shape 表示、SE(2) renderer、column 表示及 edit ownership 均保持本仓库设计。

## 训练合同

- V18 和 column **随机初始化**；Clean-E14 只提供架构配置与冻结评估参考，绝不加载为初始权重。
- 原 V18 目标：source-centred XY SmoothL1 + existence BCE + `19 × periodic yaw` + `0.25 × soft SE(2) overlap`。V18 AdamW `lr=5e-4, weight_decay=1e-4`，motion gradient clip=5。
- 保留原 column 的生成 BCE / refine action CE、合法 action mask、重要性采样权重及 TRAIN-only 类别先验校正。column AdamW `lr=3e-4, weight_decay=0.01`，column clip=1。
- 一份联合 optimizer、一次 joint loss backward；附加 V18-only 对照拥有独立 optimizer，不属于部署方法。
- V18-only 对照与联合模型具有完全相同的初始 V18 参数、窗口顺序、运动目标、学习率与裁剪。避免只在一个欠训练 transport 上看到补偿就宣布胜出。
- 每次用当前运动输出在线重建 candidates、ownership、fallback、GT action labels。**不复用旧冻结 V18 的训练 bank**。
- 每个窗口六个 horizon 总计最多 256 列（理想情况下 GEN128、REF128），每类正负样本及 static/dynamic 分层采样，重要性权重还原各 strata；缺失 strata 不重复补足。所有 source 保留。
- 梯度 probe 每 16 步记录 column loss 对 future query 的梯度；零初始化 action heads 导致第 1 步 link 梯度为零是正常现象，不能用第 1 步判链路失败。
- 真正无 source 且无合法 query 的窗口显式跳过，不计作 successful update。`executed_windows` 与 `successful_updates` 分开报告，恢复时用前者恢复顺序。

## 固定规模和结果判读

screen：从 full20430 选择 scene-balanced TRAIN1024，另留 scene-disjoint TRAIN64 作最终阈值校准。训练 1024 次窗口尝试（正常为1024次更新），batch=一个窗口，单窗口最多256列。

**这是约一遍窗口训练，不是旧固定 bank 的 replay epochs，也不是已经证明随机初始化网络收敛。** prior audit 只计数，无训练；不计入 epoch。1024步不是1024轮，不得据此断言已达到上限。

固定 update256 /512 看 dev64，阈值 `.5/.5/REMOVE-off`；不据此选 best、停止或调参。最后仅在 TRAIN64 固定格点做一次校准，随后固定 checkpoint / thresholds，一趟 dev512 同时统计其固定 dev64 子集。report horizons=1/2/3秒，保留六帧预测。

一趟同时报告：当前 learned transport、GEN、REF、JOINT，以及 frozen E14 / 同预算随机初始化 V18-only。两个参考复用 raw、Strong、source geometry，不重复 source-history/ICP 准备。Moving 指标保持原冻结协议。

通过条件严格区分：column 相对自身 transport 的有效增益、梯度 link 存在、joint 相对 paired control / E14 的整体及各 horizon 非退化。全部关闭、零编辑不能叫生成成功；失败 artifact 默认不允许部署。现有冻结 V18+columns 成功版本保持不变。

小规模随机初始化可能没训过 E14，这是单次 screen 的限制。若分支增益存在但总性能仍欠训练，报告这一事实，不自动重跑或扩大训练。本入口没有 full4369 模式。

## 速度及存储

1. 继承上一版重叠历史 patch 的精确 inverse-map cache。
2. 由每 actor 反复新建线程池改为每 horizon 共用一个池；小/离散 actor 直接稀疏映射，复用 inverse matrices，不回到旧 sampler 重算。
3. TRAIN/dev 各自默认256MiB RAM-only LRU，历史 semantics+mask immutable，future GT 只读 semantics，不强迫加载不存在的 future mask。相同 frame 并发读取 single-flight，不同 frame 不被全局 I/O 锁串行化。
4. 只预取下一窗口 raw 数据，加载/解压在 CPU，CUDA 与模型始终在主线程。显式 `include_gt=False` 部署不加载未来 occupancy。
5. 六 horizon 采样合成一个 column NN batch；final GEN/REF/JOINT/诊断及 dev64/dev512 共享 forward / sparse counts。TRAIN阈值格点也共享推理。
6. 输出仅 `last.pt`（含优化器/RNG/paired control）、`candidate.pt` 和 JSON/日志，不生成新的 dense cache 或每128步保留一份模型。
7. `progress.jsonl` 分开记录 `input_wait_seconds` / `compute_seconds`，summary 给出 prior audit、训练（含paired control）、dev64 monitor、TRAIN校准、final dev512 wall time，避免省略 I/O 等待造成虚假提速。

特征与输出一致性已由 unit/synthetic 测试覆盖；真实 nuScenes/CUDA 提速及增益需服务器结果，不能从本地 CPU 测试虚构。

本次本地合成基准（200×200×16、一个 horizon、20100 query、batch256、4 mapping workers）：原参考 sampler 10.190s，当前实现含地图构建0.400s，采样25.45×，所有 feature keys 逐元素相同，cache2.331MiB。此数字包含此前的 inverse-map 优化，**不是本次相对上一版的新增25倍提速，更不是整段 eval 提速承诺**。

## 一次运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_joint_causal_columns.sh screen
```

服务器封装脚本先跑专门测试，使用已确认路径和新的带时间戳输出目录，不覆盖任何旧实验。发回该次 `model/summary.txt` 与 `model/progress.jsonl` 即可，所有对照在同一次运行内完成。

如需恢复，仅传 `train_p0_f9_joint_causal_columns.py --resume OLD/model/last.pt` 和原始全部参数，使用 **新** `--out-dir`。严格校验 population、配置、模型、参考、训练合同，恢复两个 optimizer、sampling RNG 和 Torch/CUDA RNG；candidate 不可用于训练恢复。
