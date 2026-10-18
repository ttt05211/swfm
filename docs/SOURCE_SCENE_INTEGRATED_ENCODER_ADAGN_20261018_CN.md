# Scene–Temporal Encoder + AdaGN：联合结构候选

## 这次决定

用户明确要求同时实现 STWM 编码端整合和 Refiner AdaGN，不拆成两轮实验。基线是最新的 **Stage2 + KTA Query、无 ICP**，不是 no-query Stage2、旧 Surface/CCR 或 clean1000/norm192 模型。

本次是联合架构对照，不能将结果归因于某一个模块。旧网络、训练入口、权重、结果和缓存保持原状；新代码全部在 `tools/source_scene_integrated`。失败可直接继续使用 `tools/source_scene_kta_query`，不需要回滚或覆盖旧文件。

## 数据流

```text
四帧历史占据 / dense-valid + 历史 ego
  ├─ 固定 shared-pinned source 身份 / 原两帧 KTA / track_valid
  └─ 共享 height-aware Encoder → 四帧 F
       ├─ 原三尺度特征场：100²、25²、8²
       ├─ source 中心采样 + 原 XY/time 编码 + coarse GAP 初始条件
       │    ↓
       │  Scene–Temporal Block ×2
       │    ↓ H[source,4,128]
       │  Future queries + 原 KTA/20 投影 → 原 Future Decoder ×2
       │    ├─ 六时距 XY/yaw 修正 → 几何 Transport prior
       │    └─ Q + 学习的历史相对 yaw → 原 direct-history feature memory
       └─ 原 lift ────────────────────────────────┘

Transport prior + aligned memory → AdaGN Refiner → 18 类残差 logits
    → log(smoothed prior) + residual → 全网格 CE+Lovász / 推理 argmax
```

图中尺寸是正式配置；单元测试使用缩小尺寸。未来 GT ego 是这次 GT 配置的显式投影条件，未来占据只作为监督/评估目标，不进入历史编码。没有内部 ego 头或导航输入。

## STWM：每个 Block 只读一次场景

一个 Block 的执行顺序：

1. PreNorm → 多尺度 deformable scene cross-attention → 残差 → history-valid mask。
2. PreNorm → 每个 source 内四帧 temporal self-attention → 残差 → mask。
3. PreNorm → FFN → 残差 → mask。

只执行两层这样的 Block。删除原来独立的 global MHA，原两层 Reader 的 FFN 加两层 Temporal 的 FFN，合并为总共两个 FFN。不再先全部 Scene Reader、再全部 Temporal。

保留原三尺度采样、可学习 offsets/weights、物理坐标步长、历史坐标 Jacobian、越界处理。新场景采样算子与旧 Reader 的局部分支在同权重 CPU 测试中逐元素一致；但完整网络不是等价重排。

coarse GAP 经线性投影只加入初始历史 token，不再每层注入，也不增加额外 attention。GAP 是每帧共有的场景摘要，不能等价替代带位置的全局查询，这是本次明确接受并待验证的取舍。粗尺度仍参与可学习空间采样，但不保证学到与旧 global attention 相同的信息。

保持完整四帧 H 给未来 Decoder，不把 H 先均值池化。兼容字段 `history_source_context` 的 masked mean 仍存在，但不是 Decoder 输入。所有时序注意力都以 source 为独立 batch，不跨 source 混合；缺失历史逐子层清零，t0 必须真实有效。

原 Fourier XY/time 编码、XY/40、真实历史时间归一化均不变。原 KTA → Query 投影、KTA residual skip、Future Decoder、未来运动头保留。历史 yaw 仍由 `H_t-H_t0` 读出，t0 gauge 严格为零。继续不计算/消费 ICP，使用 track_valid，历史点从各自历史中心直接搬到未来中心。

## Refiner：只调制残差支路

旧版是在整个残差块之后做 FiLM，连 skip 一起缩放/平移。新版为：

`x_next = x + Convs((1 + gamma_h) * GN(x) + beta_h)`

- GN：1 group、affine=False。
- 时间 MLP：1 → width → width，输入保持 `.5*(h+1)/3`。
- 每块单独 width → 2*width 投影，weight/bias 零初始化。
- 原三块 dilation=1/2/3，原深度可分 3D 卷积、宽度和 readout 均不变。
- 原 memory、importance pooling、coverage、coarse prior、全分辨率分块查询、phase、epsilon=1e-4 均不变。

条件投影零初始化只保证初始调制为单位，**不保证整个残差块是恒等映射，也不保证与旧 Refiner 中间特征相同**。初始最终输出保持 Transport argmax，依赖的是原 readout 最后一层的零初始化。初始条件投影为零时，时间 MLP 的梯度可能暂时为零；条件投影可先得到梯度，之后再将梯度传回时间 MLP，不是永久断链。

不增加门限、Surface 描述、owner repair、ADD/REMOVE、额外监督或新自定义 CUDA/Triton 算子。

## 公平初始化与隔离

先构建与现有 KTA-query 对照相同的随机初始化模型，再在独立 RNG 上构建新结构。仅将语义对应且形状相同的 **新鲜随机参数** 映射过去：共享 Encoder、局部采样、Temporal MHA/FFN、未来 Decoder、KTA 投影、yaw/XY 头、memory/readout、对应空间卷积等。新 GAP、AdaGN/时间变化部分保留自己的初始化。

没有读取已训练的控制模型权重。控制目录只读 `results.json`，用于预算/人口/实现校验和最终比较；旧 E14 文件只沿用原入口的结构/只读几何缓存协议，不导入其训练权重。CPU 初始化后 RNG 与控制模型严格相同；训练各 rank RNG 沿用原 seed 规则。

新 checkpoint 协议、variant、loader、output lock、独立 identity cache 命名空间；拒绝从旧 checkpoint resume。旧纯历史缓存只读 require。原 Query/Stage2/full/source/motion audit 源码不修改，已有 fingerprint 不受影响。

## 固定实验

- RANDOM TRAIN4086 ×1，原 seed、keys、order、global B4/source128 grouping。
- 默认双卡训练；与控制的 world_size 必须一致。最终 rank0 单卡评估，其他 rank 等待。
- 完整六时距 CE+Lovász，仅此两项；无 ego/motion/existence loss。
- 原 AdamW、学习率、整轮 cosine、梯度归一化/裁剪、BF16/FP64 几何、H6。
- TRAIN/OCC/STC 均 dense-valid，GT future ego，原 remaining64。
- 复用刚完成的 KTA-query 一轮结果，不重训控制、不用全数据两轮结果充当同预算对照。
- 最终比较 OCC/STC Transport/Refiner、Moving，以及本模型 learned−KTA 和条件 GT 框运动诊断。
- 不按开发分数自动挑选、续训、部署；反复检查过的 DEV64 不是独立测试集。没有新的 FPS 或质量收益声明。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export INTEGRATED_RESUME=0
export INTEGRATED_DIR="$PWD/outputs/p0_f9_shared_history/scene_integrated_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_integrated/run.sh > "${INTEGRATED_DIR}.log" 2>&1 &
tail -F "${INTEGRATED_DIR}.log"
```

脚本只自动接受唯一完成的 `scene_stage2_kta_query_1ep_*`；有多个则列出并要求显式 `SCENE_KTA_CONTROL_DIR` 指定，不偷偷选最高分或最新修改时间。

最终：`cat "$INTEGRATED_DIR/comparison.txt"`。中断续训：设回原 `INTEGRATED_DIR`，再 `export INTEGRATED_RESUME=1`，运行同一入口；模型、Adam、各 rank RNG、完整 update 游标和整数评估状态恢复。

## 本地验证范围

专项测试覆盖共同新鲜初始化/RNG、原局部采样严格一致、AdaGN 不调制 skip、初始 Transport 输出、每层缺失 mask、四帧 Decoder context、无 ICP/未来标签输入、CPU/CUDA 全六时距梯度、空 source、独立 loader、实际 CUDA 两步停训/恢复/只读控制结果、GT/STC 整数续评。

最终 44 项新旧回归测试通过（27.30 秒），含 CPU/CUDA 实际前向/梯度/两步停续；CLI help、Bash 语法检查通过，旧 Query/Stage2/Stage1/full/共享几何代码零 diff。额外验证零条件投影更新后时间 embedding 能得到梯度、encoder 不跨 source 混合。

这些测试不代替 L40S 正式尺寸显存、速度及同预算质量验证；当前未在服务器代启动训练。
