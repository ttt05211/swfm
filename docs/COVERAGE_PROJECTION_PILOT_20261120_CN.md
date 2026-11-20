# 覆盖信息 + 有效 Prior + 投影监督：独立小预算实验

用户授权三项一起改，固定 TRAIN20%×1 轮，检查指标和视觉效果。本实验不替换任何正式默认、历史权重、结果或缓存。

## 起点和预算

- 起点是最新 causal-forward + FPN + Scene-Temporal2 + AdaLN-Zero Decoder2 + KTA query/skip + 无 ICP + history-only Memory + plain 3D U-Net。
- 随机初始化，原先语义相同的参数与最新前向关联小实验逐值相同；不导入两轮/旧版/适配权重。
- 复用原 TRAIN4086 个窗口、顺序、B4/source128 分组，共 1031 更新、一轮完整余弦，AdamW 和两组学习率 5e-4/3e-4 不变。
- 按用户后续要求，默认两张卡训练；全局B4不变，通常每卡2窗，按source数平衡整窗分工。原控制来自单卡，明确记录此执行差异，不声称逐位等价。评估始终仅一张卡。
- 控制默认目录：`outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train`。核验已完成、前向关联、AdaLN、4086×1、同64、2816122参数等契约。不能误用旧 shared-pinned 或完整两轮结果作同预算控制。

## 三项联合变更

1. Prior 覆盖由当前帧真实有效标签的逆 warp 和相同 source splat 的支持并集构成；先记录支持，再做 Strong 的空白填充。不把填充 free 认证为已观察 free，不把四帧 Memory union 当作 t0 Prior 支持。搬到静态足迹外的 source 仍有支持。
2. Memory 继续原历史特征、importance 和直接运动搬运；从融合分母是否大于零获得 coverage。两个覆盖通道进入原 U-Net 的 fine 层，19→21，新增 432 参数，总参数 2816554。其他通道/层数不变。未知处 Prior 输入为零向量；输出为 `coverage * log(Prior) + U-Net logits`。coverage 只门控 Prior，不限制任何位置的最终预测。覆盖量 detach，避免模型通过关闭支持绕过先验。
3. 原六帧 CE+Lovasz 保留，增加训练期可微语义/深度体积射线投影。俯视和固定斜俯视，64×64，采样间隔不超过0.4m，覆盖整个未来网格。每窗按 `(4*零起始update+窗槽位)%6` 轮换一个 horizon，原六帧语义损失仍全部计算。用现有18类概率决定透明度，无额外 opacity head；class0仍参与监督。语义为每条射线18类分布的 L1；深度按固定视角深度跨度归一化，用GT命中掩码，不用 pred∩GT。GT空射线仍通过背景语义约束误报。投影项总权重预先固定1，不做DEV搜索。

投影监督借鉴 GaussRender 的训练原则，但实现是纯 PyTorch 体积射线积分，**不是官方 Gaussian CUDA rasterizer 的复现**。不需要未来照片、额外相机标注或新增推理模块。投影损失的显存和耗时需以服务器实际日志为准。

## 数据、输出和恢复

- 直接从原始 TRAIN GT 读取四历史/六未来，在线重建前向关联、KTA、历史读取和 Strong；不依赖已清理的 TRAIN/VAL 几何缓存，也不新写 dense tensor/geometry 磁盘缓存。
- `--train-cache` / `--dev-cache` 是保留的窗口元数据索引；只读用于原人口/分组核验，不使用其中的旧关联或未来运动监督。
- 输入标签缓存128MiB，最多一个后续批次的有界预取。仅 last / last.previous / epoch_0001 权重和小日志，不每步存模型。
- 每64更新及 TERM/INT 后保存完整更新边界的 Adam、每个rank各自CPU/CUDA/Python/NumPy RNG和全局游标。全局均值梯度先SUM，再统一裁剪及Adam；较小批次允许空rank，无重复/填充样本。投影轮换使用原全局窗槽位，不因双卡拆分改变horizon。恢复固定同目录/双卡数；新v2协议不把旧单卡v1任务静默切为双卡。已完成训练时保留 frozen epoch 文件，评估按已存整窗计数继续。
- 自动对同一 remaining64 分别新跑原版和候选的 OCC/STC；dense_valid、GT future ego、full-grid、1/2/3s平均。只报告 mIoU、IoU。关闭 FREE、age、STC去抖动/距离校准，不叠加以前的后处理。
- 预测六帧完成后才为评估读取未来 GT。整窗整数混淆矩阵持久化；中断不重复计数。旧 recorded 控制指标另存，当前同执行路径的 fresh 控制用于主比较。
- 固定同64前8窗，各导出 GT/原版/候选、t0/1/2/3s BEV PNG、HTML图册及uint8原始占据，OccWorld/COME色表。不是按提升大小挑图。原始体素包可在本机沿用已有 COME 相机/VTK 画三维图；服务器 BEV 预览不冒充三维渲染。
- `visuals_and_metrics.zip` 只打包结果、图和16组小体素对照，不包括代码、训练集、权重或大缓存。代码仍走 Git。

## 启动

在服务器 OccFM 环境和仓库根目录：

```bash
export PILOT_GPUS=0,1
export COVERAGE_DIR="$PWD/outputs/p0_f9_shared_history/coverage_projection_dual_20pct_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/coverage_projection/workflow.sh > "${COVERAGE_DIR}.log" 2>&1 &
echo "$COVERAGE_DIR"
tail -f "${COVERAGE_DIR}.log"
```

`PILOT_GPUS=0,1` 默认；训练两个进程退出后自动切为首张物理卡评估，也可用 `PILOT_EVAL_GPU` 指定单张卡。`COVERAGE_CONTROL_DIR` 可显式指定唯一同预算控制。断开终端不停止。恢复：先设置原双卡 `COVERAGE_DIR`，然后 `COVERAGE_RESUME=1 nohup bash tools/coverage_projection/workflow.sh > "${COVERAGE_DIR}.resume.log" 2>&1 &`。不要同时启动两个相同目录的任务。

结果：`summary.txt`、`evaluation.json`、`visuals/index.html`、`visuals_and_metrics.zip`。训练进度：`progress.json` / `progress.jsonl`。完整实验没有在本机假装运行；本地仅有32个TRAIN窗口，不能替代4086窗。

## 已完成的实现验证

- 新增测试和原 FPN/AdaLN/前向关联/恢复回归共44项通过，包含同64评估在第3窗中断后恢复、256次成对前向无漏窗或重计、先预测后读取未来标签及整数混淆矩阵核验。新增真实双进程CPU/Gloo测试，比较B4/B3/B1（含空rank）的CE/Lovasz/投影梯度均值和Adam结果，核对全局投影时距、双rank RNG保存、恢复后权重一致。
- CPU/CUDA 小网格真前向+全六帧 CE/Lovasz/投影反向，检查新增覆盖通道、FPN、Memory importance和source位移头梯度；实际Adam更新和CPU RNG断点续跑逐值一致。
- 运行一个真实TRAIN窗 scene-0429 的完整200×200×16、六帧随机模型前向，并对原生网格投影损失反传；输出尺寸6×18×640000，投影梯度有限且非零，本机峰值分配1571.2MiB。仅配置来自最新模型，不导入训练权重。这是运行检查，不是质量结果或FPS。
- 新Prior概率与旧Transport前向逐值一致；Memory CPU逐值一致。CUDA scatter_add 本身存在重复浮点差异，本轮新旧最大差4.47e-8、原版重复也4.47e-8，严格byte不一致如实保留，不声称byte parity。
- Bash语法检查通过。本机只有一张GPU，双进程同步在CPU/Gloo验证，未声称实测双CUDA吞吐。真实双卡完整一轮、DEV64分数及训练后的视觉效果待服务器执行，不能据此承诺改善或自动展开到两轮。
