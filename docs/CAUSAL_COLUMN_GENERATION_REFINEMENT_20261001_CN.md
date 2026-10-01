# 共享稀疏柱状生成与 source 归属精修

协议：`p0_f9_causal_column_generation_refinement_v1`。

本版实现两个目标：未来 query 新进入区域的路面/人行道连续生成，以及原 V18 搬运后的历史证据精修。它是一个需要真实数据检验的候选，不承诺涨点；全拒绝不是成功。保留原 Clean-E14，既不继续失败的 XY 训练，也不改变 V18 的 XY/yaw/existence。

## 结构和数据流

```text
历史六帧占据 + lidar visibility + 原 V18 的预测运动
        ↓ 静态世界坐标对齐 / 动态 same-source 因果关联与有界 ICP
局部 7×7×完整 Z 历史柱状 patch + visible-owner/background 信息
        ↓ 共享：语义 embedding → 有序 Z 投影 → 小型 2D CNN
历史时空 tokens ← 2 层 cross-attention ← 候选柱 query
        ├─ generation head：每个 Z bin 的占据 logit，类别继承
        └─ refinement head：每个 Z bin 的 KEEP / ADD / REMOVE
        ↓ TRAIN 分数校正及冻结门控 → source 分层 compositor → 输出
```

默认宽度 64、4 个 attention heads、2 层 cross-attention、embedding 维度 8。保留每个 Z bin 的位置，不用高度均值/顶部高度代替三维结构。历史 token 有时间、patch 位置、visibility 和 source membership。未匹配动态帧/越界 patch 使用 UNKNOWN=18，与 FREE=17 分开；全 UNKNOWN 时使用零 dummy token，避免 all-masked attention 的 NaN。

Generation query 读取**因果 frontier anchor 周围的历史 patch**，而不是只读取待生成的空白位置。后者在 query 距历史 grid 超过 patch 半径时会变成六帧全 UNKNOWN，悄悄失去路面高度/语义 evidence。Query→anchor 的偏移在 context 中显式保存，网络向新 query 输出全 Z。Refine 则读取编辑位置附近、经同一 source motion 逆对齐的历史 patch。

两个新分支共享新增的轻量网络，不改原 V18 网络，也不与其优化器共享参数。模型只输出 occupancy/actions，继承 source/静态 anchor 类别，不预测任意新类别或任意三维坐标。

## 候选和写入权限

| 分支 | 因果候选 | ADD | REMOVE |
| --- | --- | --- | --- |
| generation | 六帧历史 grid footprint union 之外，距历史 road11/sidewalk13 frontier ≤3.2m | 原 V18 free 且 refine 后仍 free | 不允许 |
| 静态 refine | 已访问区域里存在历史 road/sidewalk 残差的柱；历史 Z 范围膨胀一 bin | 原 V18 free | 只删除继承的静态类别，恢复 free；不改其他类别 |
| 动态 refine | 原 t0 source，至少一帧历史成功关联/配准；预测搬运的历史 shape union，XY 一格/Z 一 bin padding | 原 V18 free；保持原 source 写入顺序 | 只删除该 source 在最终 V18 输出中的可见归属 voxel，恢复下一层 source 或 A1 清理后的背景 |

Generation 的“新区域”是**真实 grid entry**，不是把历史 grid 内的 mask_lidar=false 当成新区域。动态历史先注册到同一 t0 source，再用该 source 的原 V18 SE(2) 运动搬运；不把移动汽车的不同历史位置简单 ego-align 后叠成拖影。关联使用已有的同类一对一/25m/s/歧义拒绝规则；ICP 至少六对应点、至少 50% inliers，不使用 GT box，保持 world Z 不变。

最终合成：先执行合法可见 source REMOVE，恢复 lower layer；再按冻结 source 顺序写入 refine ADD；最后 generation 只填仍 free 的位置。source 重叠采用原 V18 的 last-source-wins 顺序。下层 source 与被删除 source 同类时，没有可观测删除收益，因此不允许该 REMOVE。

**重要限制：**这不是任意 occupied 区域的语义重写，也不是任意动态 birth/dormant 生成器。本版支持有历史证据、且 t0 仍有可靠 source 的动态遮挡恢复；完全没有 t0 source 的动态物体暂不生成。不把狭窄的静态连续补全包装成通用“新物体生成”。

## 两个训练目标，而非损失堆叠

1. Generation：class-conditioned occupancy 的加权 BCE；只有继承类别与 GT 一致才是 ADD positive，不能仅凭 GT occupied 就标为正确添加。
2. Refine：合法 KEEP/ADD/REMOVE 的加权 CE。ADD 同样要求继承类别匹配 GT。REMOVE 要求**删除后实际恢复的 lower-layer label 正好等于 GT**；GT≠当前 source 类别不足以证明删除有益。两种动作都不能修正语义时，KEEP-on-tie。

权重来自完整、未采样的 TRAIN 候选 voxel/action 数；不从正样本增强后的 bank 或 dev 估计。Generation pos_weight 截断为 [1,20]；refine action 权重按出现 action 的中位数/count 截断为 [0.2,20]。

TRAIN 为 generation/static-refine/dynamic-refine 分别抽取 edit-positive 与 all-KEEP 柱，保留逆包含概率，避免稀有动态 refine 消失或 positive oversampling 改写自然先验。两个任务损失各自归一化后平均；**权重并不等于目标指标的精确 surrogate**。

推理时 generation logit 减去 `log(pos_weight)`，refine action logits 减去 `log(class_weight)`，再做合法 action 的 sigmoid/softmax。这是对 weighted proper loss 最优分布的先验校正，不是有限样本的概率校准保证，不能直接把加权后的原始概率当高置信度。

## 一次有界 screen

- TRAIN1024：从完整 TRAIN20430 按场景和时间分层取样，不取前 1024；与 dev 场景不重叠。
- TRAIN calibration64：32 个独立 TRAIN 场景，每场景两个时间位置，与 optimization 场景隔离。
- 1024 次更新，batch256 **柱**（不是窗口）；AdamW 3e-4、weight decay0.01、clip1，CUDA BF16。
- 最终 checkpoint 只在 held-out TRAIN 上校准三个独立门控：generation ADD、refine ADD、refine REMOVE。预声明 levels 为 0.50/0.75/0.95/disabled，64 个联合设置；额外单分支设置共用同一 raw/V18/model pass。保留普通 argmax-aligned 0.5 起点，不假定只有极高分数才可能有效。
- 校准比较真实、分层合成后的冻结 IoU/mIoU/Moving 各 horizon 原始计数，不用 correct-minus-wrong 代理 mIoU；要求三个 ablation 总体和各 horizon 的四种指标非负。
- 校准后先保存冻结模型及阈值，再读取 dev labels。一次 dev512 pass 同时给出冻结 dev64 子集与全部 dev512 的 V18/生成/refine/联合结果。
- 同趟输出预声明阈值 `(0.50,0.50,0.50)` 的 `diagnostic_*`，及合法 query-voxel 的 score mean/max/过阈值数量，帮助辨别 all-reject、分数塌缩与预测有害；它们不用于 dev 上二次选模型。
- 只检验最终 1024-update 候选，不在 dev 上挑 epoch/best checkpoint，不自动追加训练或扩大规模。

dev64 是 dev512 的子集，两者不是独立统计检验；本 screen 仍是探索性验收。通过也不能保证未见场景不下降。后续 full4369 必须显式启动，且不重新校准。

通过条件：仅生成有真实 semantic-correct 添加；仅 refine 有真实纠正；生成/refine/联合的整体与 1/2/3s 四种指标都不下降；联合 mIoU 严格正增益；dev64 和 dev512 同时满足。全拒绝/关闭任何一支导致没有真实正确编辑，不能成为“有效”方法。精修导致真实 occupancy 被误删、scene 负收益、add 精度和 source-layer REMOVE 决策均单独报告。

## 效率与存储

不训练 AE，不需要 prototype bank，不保存新的 dense occupancy cache。TRAIN 只在 RAM 中存 uint8 局部 patch、legal/target 和 float32 几何/importance；默认含拼接峰值预算 4096MiB，超预算立即报错，不偷偷落盘。原 V18 cache/raw scene 本身的 RAM 不计入这个 bank 预算，应另外留出内存。

每窗口原始数据/V18/历史 evidence 只准备一次；原 V18 forward 冻结，训练不反复跑它。历史 source extraction 与静态 memory 使用有界 CPU threads；仅局部 query patch 进新网络，不展开整场景密集 learned 3D feature volume。预测 query 分块；四组结果与所有门控使用共享概率和稀疏精确 metric delta。CPU 预处理阶段低 GPU 利用率是预期行为，不通过盲目占用显存解决。

每窗口打印开始/完成和耗时；progress.jsonl 给出 raw+V18、renderer、source_history、static_memory 阶段时间。不能在没有真实服务器 profile 的情况下承诺用时。只保存 `last.pt`（每128步覆盖，同协议恢复）和最终 `candidate.pt`，以及小型日志/报告。

### Patch 采样提速（兼容已有 candidate.pt，无需重训）

服务器第220窗口 profile：14.648s 总耗时中，119 次 NumPy patch sampling 累计13.125s；网络 forward 累计0.366s。主要瓶颈不是 GPU，也不是 0.556s 的 V18/历史准备。

`causal_column_sampling.py` 为单窗口/单 horizon 的密集重叠 query 建立局部 inverse-map cache。生成与静态 refine 共享几何逆映射，source 各自使用原配准；相同历史 voxel 不再因49个邻域点重叠被重复变换几十次。缓存只存 uint8 原始 labels/visibility/source bits，最多64MiB，不写磁盘；空间分散、候选很少或超预算的 group 回到原始 sparse sampler，绝不截断候选。原始 sampler 保留为参考。地图构建最多六个 CPU threads，坐标变换维持原 float64 运算顺序/world-Z/越界 UNKNOWN；GPU batch256、模型与门控不改。

本地200×200×16网格、20100 queries/一个 horizon 合成测试：首版原采样10.318s→优化0.699s（14.8×）；最终使用只读 sliding-window view 批量 gather，并包含 ego roll/pitch 后，原采样11.706s→优化含地图构建0.561s（20.9×），history digest完全一致，缓存2.33MiB。**这只是本地采样基准，不是服务器整段 eval 的实测提速承诺。** 单元测试进一步逐元素核对所有 feature keys、不同线程数、旋转/Z、source归属、unknown padding、预算回退，以及相同网络batch下预测概率逐元素一致。progress.jsonl 新增按horizon的地图构建/patch gather/网络与传输耗时。

已有模型可先做只读同窗口检查和计时，不重训：

```bash
python tools/real_motion/profile_p0_f9_causal_columns.py --model-dir <原model目录> --window 220
```

该工具对 actor 分层采样，检查真实数据新旧历史特征完全一致后，重复同窗口预热/计时。它不保存结果、不改变模型/阈值、不将失败候选提升为成功。不应与当前GPU评估并行执行。

诊断读取同一个内存 checkpoint 快照并对该快照计算 SHA256，避免文件哈希与加载间的并发替换。原 screen 在评估完成后会重新保存 `screen_pass`，因此文件 SHA 改变不一定表示模型改变。诊断结束仍检查原文件：只有全部权重（含 dtype/shape）、阈值、合同和 population 完全一致，且 `screen_pass` 不变或由 false 完成到 true，才报告外部序列化/验收更新；其他变化继续报错。该例外仅适用于计时工具，正式 expanded evaluation 的文件 SHA 严格检查保持不变。

## 已覆盖的逻辑验收

测试覆盖原 renderer 六 horizon 的逐 voxel 一致性、空 t0 source、随机重叠物体的独立完整 layer recomposition 对照、下层类别恢复、occupied 保护、未来 GT 修改不影响候选/特征、真实因果 ICP 排序和 world Z、动态 patch motion alignment、importance/natural class counts、weighted score correction、非法动作梯度屏蔽、两个 head/共享主干学习、全 UNKNOWN、全拒绝/单 horizon/Moving 降级拒绝、checkpoint role/权重/阈值完整性、端到端 smoke、下一更新断点精确恢复。

合成网络可学习不等于真实 nuScenes 有收益；本地无真实服务器数据和 CUDA，真实效果与 GPU 性能仍待该 screen。

## 运行

服务器固定使用用户确认的 ROOT、E14、V18 train/dev、nuScenes info 和 dev64 manifest，不猜测新数据路径。不依赖当前 shell 中遗失的 TRAIN_V18 等变量。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
bash tools/real_motion/run_p0_f9_causal_columns.sh screen
```

如服务器不在该分支，不应把整条 feature 分支误合并进其他分支；先确认 `git branch --show-current`。脚本默认通过 OccFM 的 `command -v python` 解析解释器、检查 CUDA/BF16、跑本模块测试后执行一趟 screen；缺路径或环境立即明确报错。

输出 `outputs/p0_f9_causal_columns/screen_<timestamp>_<commit>/model/summary.txt`，只需发回这个文件。摘要标明是否是诊断/失败，不会把关闭分支的零修改称为增益。

支持显式继续同协议 last checkpoint 到一个**新目录**：`train_p0_f9_causal_columns.py --resume <last.pt>`，并需原参数。缓存/bank/population/config/seed/budget/权重一致，否则拒绝。不能用 candidate.pt 当训练断点。

扩展评估入口为 `eval_p0_f9_causal_columns.py`，population 可为 dev64/dev512/full4369；默认拒绝 failed/smoke/identity 模型。显式 `--allow-diagnostic` 可以读取失败候选做分析，但不提升其部署资格，且不改 checkpoint/阈值。

## 方法定位与相关工作

历史证据聚合与多帧 attention 是既有思路，应如实引用 [GAST / GSCA](https://arxiv.org/pdf/2608.15279)，不以“也用了 attention”作为创新宣称。这里可检验的区别是：冻结 source transport 后，共享的局部全-Z证据网络以两个不同角色输出；生成的真正 grid-entry/free-only 支持与 refine 的 source ownership/可恢复 lower layer 形成明确写入边界；动态 evidence 按 same-source 对齐，而非通用整场景融合。

这是小规模候选实现，不是已经证实的新 SOTA；论文只能根据真实 ablation/完整验证报告描述其作用与限制。
