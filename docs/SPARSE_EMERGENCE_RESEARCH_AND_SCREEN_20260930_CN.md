# 稀疏新增几何：文献判断与一次性小样本实现

日期：2026-09-30。范围：冻结 V18 observed-source transport，补充不依赖 t0 source shape 的新增几何解码。本文不是系统综述，不声称已经验证真实数据收益。

## 结论与证据边界

选择 **独立 query → occupied point set**，不是 dense completion、prototype 检索或新的 tokenizer 大模型。OPUS 将 occupancy 表述为 occupied 坐标与语义集合预测，使用 Chamfer/最近邻语义监督；SparseWorld-TC 把独立未来 queries 与历史图像及轨迹条件结合。这支持“新坐标与类别可以直接解码”的表示选择，不证明冻结 V18 上的稀疏残差预测能成功。[OPUS §3.2–3.3](https://arxiv.org/html/2409.09350)、[SparseWorld-TC §3](https://arxiv.org/html/2511.22039)。

必须纠正两点：**采样不是生成新几何的必要条件，随机 query 也不是生成准确的充分条件**；novel-view reconstruction 不能直接证明未来 birth forecasting。这里实现的是确定性条件几何预测，不是概率采样生成模型。原论文以图像为输入，我们沿用 V18 的历史语义 occupancy 输入协议，不能直接比较它们的表格分数。

证据/写作意图清单：

| 判断 | 来源定位 | 可以支持 | 不支持 |
|---|---|---|---|
| occupied-only 点集表示避开 dense free-class 竞争 | OPUS §3.2–3.3 | 本实现选择几何/语义集合输出 | 本实现肯定不会塌缩 |
| 独立未来 queries 可解码新增位置与语义 | SparseWorld-TC §3.1–3.3 | 非 source-shape 搬运式表示 | 隐藏动态实例一定可预测 |
| 多尺度/AR 是可行生成范式但迁移成本较大 | OccWorld / OccTENS 方法 | 不优先替换当前场景编码体系 | 这些路线效果较差 |
| Gaussian 并不意味着整个流程非扩散 | SCube / InfiniCube 摘要与作者页 | 更正生成链路分类 | 所有 Gaussian 模型都使用 diffusion |
| 当前小分支是否非降且生成正确几何 | 待服务器 screen | 只能由冻结评估决定 | 合成测试或空输出证明成功 |

## 路线比较

| 路线 | 生成机制与迁移判断 | 本轮决定 |
|---|---|---|
| OPUS / SparseWorld / SparseWorld-TC | 点位置、类别由 query 解码；不必复制旧形状。SparseWorld 与 TC 的初始化/递归条件并不完全相同 | 借鉴表示与训练原则，做两个小 decoder；不是复刻论文 |
| OccWorld / OccLLaMA / RenderWorld / I²-World | occupancy tokenizer 与 AR 场景预测，通常新增 tokenizer/预训练及预测链路 | 不为补充能力替换整个 V18 |
| OccVAR / OccTENS | 多尺度量化表示、粗到细 next-scale；OccTENS 还建模 temporal AR | 不增加多阶段场景 tokenizer |
| NOVA | 非量化连续潜空间视频 AR；不是现成 occupancy 小插件 | 不假设免预训练/免适配 |
| DrivingForward / GaussianFormer-2 | 前馈 Gaussian 重建/occupancy 表示可借鉴，但新增 Gaussian 参数、superposition/rendering 接口 | 暂不更换冻结体素合成路径 |
| SCube / InfiniCube | 生成链路含 latent diffusion，后续 feed-forward appearance 或 rendering 不等于整体非扩散 | 不作为本轮“非扩散轻插件”的依据 |

主来源及核对范围：

- [SparseWorld](https://arxiv.org/html/2510.17482)：Range-Adaptive Perception、State-Conditioned Continuous Forecasting；不能简化成与 TC 完全相同的未来初始化。
- [OccWorld](https://arxiv.org/abs/2311.16038)、[官方实现](https://github.com/wzzheng/OccWorld)：tokenizer/世界模型分工。
- [OccLLaMA](https://arxiv.org/abs/2409.03272)、[作者项目](https://vilonge.github.io/OccLLaMA_Page/)：多模态离散建模。
- [RenderWorld](https://arxiv.org/abs/2409.11356)：AM-VAE / Img2Occ / 世界模型，不是一个免 tokenizer 的点头。
- [I²-World，ICCV 官方论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_I2-World_Intra-Inter_Tokenization_for_Efficient_Dynamic_4D_Scene_Forecasting_ICCV_2025_paper.pdf)：intra/inter tokenization；本轮只用于路线分类。
- [OccTENS](https://arxiv.org/html/2509.03887)：§III-A multi-scale quantizer、§III-B temporal/spatial next-scale。
- [OccVAR OpenReview 稿件](https://openreview.net/pdf/6ff44767af80014d9a1f7c12c16197edc14827ee.pdf)：仅核对检索到的主来源题名/摘要；全文访问受限，**不确认接收状态、不作为关键实现证据**。
- [NOVA](https://arxiv.org/html/2412.14169)：摘要及方法框架；视频领域连续潜空间，不声称已验证 occupancy 迁移。
- [DrivingForward，AAAI 官方论文](https://ojs.aaai.org/index.php/AAAI/article/download/32793/34948)：surround-view Gaussian reconstruction，区别于未来未知动态生成。
- [GaussianFormer-2](https://arxiv.org/html/2412.04384)：probabilistic Gaussian superposition，另一种 occupied 表示；未移植其 renderer。
- [SCube](https://arxiv.org/abs/2410.20030)、[作者项目](https://research.nvidia.com/labs/toronto-ai/scube/)：hierarchical voxel latent diffusion + VoxSplats。
- [InfiniCube](https://arxiv.org/abs/2412.03934)、[作者项目](https://research.nvidia.com/labs/toronto-ai/infinicube/)：sparse voxel geometry/latent diffusion 与 world-guided video generation。

检索采用题名、arXiv ID、作者项目/官方论文交叉核对；检索词包括 OPUS occupancy sparse set、SparseWorld-TC independent future query、OccTENS next-scale、NOVA non-quantized autoregressive、SCube/InfiniCube diffusion、GaussianFormer-2 occupancy。没有进行完整引用网络覆盖、DOI 全量核对或 meta-analysis；这些论文的任务、输入、计算量和评估不同，不以其涨点为本实现的预期收益。

## 数据流与两个实现

```text
六帧 <=t0 语义 occupancy + 历史 ego pose + 未来 ego query
  ├─ 冻结 V18 原 forward / Strong SE(2) renderer → V18 prediction
  └─ 历史对齐到 query frame（仅输入重采样）
       → 因果 patch + 历史局部 columns + query pose/time/位置/V18 histogram
       → 小型空间编码 + SpatialTemporalBlock
       → 独立 learned queries + FutureQueryBlock
       → 新 XYZ + 17 类语义 + patch existence
       → 离散点集，仅写入 V18-free voxel
```

没有修改 V18 网络、参数、训练 cache、Strong renderer、metric/support；复用的是已有 Transformer **类/设计**，新 encoder 权重独立训练，不伪称直接复用 V18 的预训练特征。辅助 checkpoint 不包含 V18 参数。

- `direct`：两层 query decoder 后一次解码。
- `refined`：同样两层、同样参数量；每层输出位置并反馈 query，平均两层同类型监督。不是多套模型集成，也不声称它是 next-scale VAR。
- 每个 1.6m×1.6m patch：4 queries × 16 points，最多 64 patches/horizon；默认总点预算 4096/horizon。上下文为 3.2m×3.2m 的六帧 columns。每 patch 不限制语义为 static，动态也能解码，但必须单独验真。
- 因果候选：历史占据证据 3.2m 邻域，或 t0 对齐后的 query-entry unknown，且 V18 至少有 free voxel。超预算按固定空间顺序均匀选 patch，**不从未来 GT 挑正例、不查询未来 annotation**。
- 局部输出坐标有界；floor 栅格化、同 voxel 最高置信度稳定合并；越界丢弃，不溢出邻 patch。该 floor renderer 不反传，训练通过连续几何损失提供梯度。
- 点置信度为 patch presence × 最大语义概率，**不是已证明校准的逐点 occupancy 概率**。

三类 loss，权重均为 1，没有六个新损失项：

1. `geometry`：只在 positive patch 上做双向 L1 Chamfer，单位为体素；对预测点与残差 GT 点做集合监督。
2. `semantic`：只在 positive patch 上监督最近 GT 点的 17 类 CE；train-only 类频率平方根逆权重，裁剪到 0.5–4。没有 free 类参与形状/语义竞争。
3. `presence`：patch 是否存在任何 addable GT 点的 BCE；dynamic-positive/static-positive/empty 分层取样，按逆采样概率恢复自然 patch prior。

这不是保证不存在塌缩：存在性仍可能全部低分、几何可能错误。训练日志保存三类 loss、XYZ/semantic/presence head 梯度、抽样正例类别频率；最后额外报告抽样 train patch 的逐格几何/语义命中和 presence 分数，区分训练本身没拟合与泛化/校准失败。raw 输出与最终阈值输出都要报告。

## 一次性小样本合同

默认 `screen`：128 个 train windows、16 个 **TRAIN split 内、与 train 场景完全互斥**的 calibration windows、冻结 scene-balanced dev64。train/cal/dev ordered keys、指纹、配置、Clean-E14 SHA、Git SHA 保存为 `execution_contract.json`。不依赖单独的 V18_DEV512 文件。

1. 一次准备 train 输入/target，复用给两个变体。训练目标是同一因果 patch 内 V18-free 的未来 occupied GT；包含 V18 现有运动误差的残差，不把这些误差全部叫 birth。
2. 每个变体 512 updates，batch=64 **patches**；相同初始化、样本序列和 AdamW 日程。V18 永远 eval/frozen。无每128步昂贵 dev monitor。
3. 最后只在 TRAIN-calibration 上测试预声明阈值 `0.05/0.10/0.20/0.40/0.60/0.80/0.95`。选择有正确新增且 overall/1s/2s/3s 的 IoU/mIoU/MovingMacro/MovingMicro 都不降的阈值；通过者按正确新增 TP、mIoU、较严格阈值排序。
4. 先把阈值写入 `calibration.json`，再准备并一次评估 dev64；不按 dev 调阈值，不在 dev 上 ensemble 两个变体。阈值全不通过时正式输出 abstain，并明确失败。
5. 同一次 dev 准备报告候选 scope 的 GT 上限与 64点/patch 的预算 GT 上限，不新增 oracle 实验链；不用于挑选候选、阈值或后续自动重试。这些上限包含非新增实例残差，不能与 V21 birth-only ceiling 混用。

Gate：整体和每个 report horizon 的四个冻结指标均非降（1e-9 数值容差），**且非零正确新增几何**。没有正例的 Moving 指标仅允许 baseline/selected 同时 undefined，有限指标变为 NaN 直接失败。通过也只是 screen 证据，不是 dev512/test 的保证，不自动扩大。

新增质量分开计数：

- `unseen_static_semantic_tp`：GT 为静态类，输入重采样历史的同一 query voxel 没有任何静态占据，且预测新增语义正确。这是输入表示上的新几何，不等于证明一个物理静态物体从未被观测过。
- `birth_semantic_tp` / `dormant_semantic_tp`：仅在 GT-only V21 responsibility 和 fail-closed future component attribution 确认后统计。未来 annotation 只用于这个验收标签，不进入模型、support 或推理。
- 有 t0 source 的物体移动产生的新 voxel 可参与残差训练，但**不计入 birth/dormant 生成 TP**。动态 TP 为 0 时不能声称动态生成已得到实证。

`threshold=0` 是最后权重的 raw 诊断，不能被选为正式阈值；用来辨别不会解码/解码了但不准确/校准全拒绝。零新增非降不是成功。add-only 不覆盖已有 occupied voxel，但 false positive 仍会损害 IoU/mIoU，不能宣称数学保证非降。

## 运行、开销与部署边界

服务器已确认的 ROOT=/root/nas/occ/swfm、OccFM、V18 train/full4369、Clean-E14、nuScenes train/dev info、V21 dev64 manifest 和 `configs/real_motion_occfm.yaml` 均由脚本显式检查。不存在的输入只报 `[MISSING]`，不猜路径。无需重建 Stage-1 / prototype bank。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens &&
git merge --ff-only FETCH_HEAD &&
bash tools/real_motion/run_p0_f9_sparse_emergence_screen.sh
```

一次运行包括预检查、两个变体及最终摘要。`smoke` 参数只跑 4/2/2 windows、8 updates，不用于方法结论。输出新建于 `outputs/p0_f9_sparse_emergence/screen_<time>_<sha>/model`，不会覆盖旧实验。失败 gate 正常产出 summary，不自动重训。

优化：history query-frame alignment 跨三个 report horizon 并行、I/O/Moving/attribution 使用8 workers；train bank 一次构建驻 RAM、两个变体共享；每个 calibration row 只解码一次复用阈值；指标使用经过逐格全量对照测试的 changed-cell raw counts。train bank peak bounded 512MiB（256MiB chunks + concat），eval population 2GiB 硬上限；不落地 dense 预测/cache，只保存 JSON、日志和两个小辅助 checkpoint。没有承诺具体服务器运行分钟数或 GPU 利用率。

`forecast_with_emergence` 支持六个 future horizons，`include_gt=False`，不读 future occupancy/annotation。`load_emergence` 强制检查协议、基础 SHA、runtime config，默认拒绝 failed/smoke checkpoint；允许显式 diagnostic load 不等于验收通过。当前只有 screen，**没有自动 main 训练入口**。

## 本地验收与实际效果

`tests/test_sparse_emergence.py` 覆盖 history 对齐、unknown/free 区分、GT 输入隔离、采样校正、稳定冲突/越界、V18 occupied 保护、零提案 identity、positive/empty 梯度、两变体训练、合成新类别新位置过拟合、非有限 gate、train/cal 场景隔离、校准全拒绝、changed-cell/full-grid 指标一致及失败 checkpoint 拒绝。

本地 CPU 合成测试是实现检查，不是 nuScenes small-screen。服务器真实 screen 尚未运行，不能由论文/单测保证非降或新增质量。若两个变体失败，本次按合同停止，不继续“自动找下一个阈值/再训一轮”。

本次本地验收：596 passed / 4 skipped / 1 deselected（其中新增16项）；435个 Python 文件 AST 检查及 bash 语法检查通过。每个默认 decoder 为186,841参数，纯 FP32 权重约0.713MiB；这不是 GPU 训练峰值显存估计。未声称已通过远端 CI 或真实 nuScenes screen。
