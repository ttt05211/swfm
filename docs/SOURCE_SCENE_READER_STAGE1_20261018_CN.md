# 第一阶段：Source Scene Reader，同预算随机初始化对照

## 最新预算：TRAIN20% × 1轮（2026-10-18）

按用户最新要求，两组都默认只训一轮，完整一轮余弦，结束各评同remaining64。协议升级为v2，拒绝将旧两轮计划checkpoint当一轮断点；使用新目录。可显式SCENE_EPOCHS=2创建新的两轮计划，但不自动扩训。网络、数据人口、损失、匹配审计与旧结果不变。

## 边界与回退

独立新增 `tools/source_scene_stage1/`，不修改旧网络、trainer、评估、source匹配、缓存协议或默认入口。旧模型与 clean1000/norm192/41.78FPS 结果继续保留，但不作为本次初始化，也不继承其分数或速度。

本次明确以**最新 dense-valid / shared-pinned source**为主线。最近两帧身份及原两帧KTA不变；没有 half-KTA/多帧速度混合、历史年龄衰减、静态额外支持、后验校准、norm192、污染或STC监督适配。

失败时停止新任务，继续使用原模型和原入口即可；不需要 `git reset`、覆盖 checkpoint 或把新权重载入旧模型。新模型不是旧checkpoint的兼容替代品，也不会自动推广。

## 实际网络

四帧共享高度编码 Encoder 保留。新增 source motion 分支：

1. 对原生四帧 `F` 构建三尺度特征：原尺寸、XY各缩小约4倍、最多8×8；共享同一次特征投影。
2. 两层 Scene Reader：reference-guided 多尺度稀疏双线性采样（4 heads、每尺度每head 4点，随配置头数一致）＋对粗尺度全部场景token的显式全局cross-attention＋FFN。
3. Source参考坐标和真实历史时间使用标准 Fourier position/time encoding。粗场景token位置也转为t0坐标。采样偏移用t0方向的XY，再映射回各历史原生特征坐标；不物化额外对齐特征图、不二次搬运同一特征体。不宣称二维高度压缩特征是完整SE(3)等变表示。
4. 两层 masked Temporal Transformer，保留四个历史token，不先全局平均成单向量。
5. 原六时距 Future Query Decoder，以及 KTA→Query 条件保留；输出XY残差、yaw变化和未来Q。
6. 原 KTA residual skip、Geometry Transport、registration-conditioned feature memory、Semantic Refiner 不变。Q仍通过原 source_query 参与动态memory。

Reader不消费独立的class embedding、source mask通道、人工速度/位移状态MLP或valid数值通道。类别仍用于原source提取/匹配/几何语义搬运，不能声称整个方法不使用语义规则。几何source成员关系也保留。

**历史有效性仍是原registration通过规则，不切成track_valid。** mask用于时序/未来cross-attention；无效参考不产生虚假位置token。只有t0的source也可运行；零source有明确空输出。不存在新ICP替代头，也没有“仅按中心平移就等价刚体配准”的假设。

本阶段不是跨source未来联合动力学：每个source读取场景上下文，时序和未来解码仍按source独立。

## 对照与预算

只有两组，顺序运行，不混训练两模型的梯度：

- `baseline`：当前 SharedTransport＋最新 shared-pinned＋dense-valid。
- `reader`：新 SceneTransport，其余相同。

固定 TRAIN20430 中原 round-robin 选出的4086窗（20%），seed20261011，一轮完整余弦；原 global B4/source128分组，不截source、不drop窗口。每个窗口仍监督全部六未来帧。只使用 CE＋Lovász；motion/existence/ego损失均不计算，没有ego头或导航。GT future ego明确作为几何条件。

两模型从随机初始化开始。构造同一seed的原模型后，将**新鲜随机**且名称/形状一致的 Encoder、Refiner、KTA投影、未来decoder/head参数赋给Reader；不导入任何旧训练权重。共同模块初始化逐元素相同，架构/参数量不同，因此不是等参数或单输入通道消融。两组使用相同AdamW、LR（motion5e-4/refiner3e-4）、weight decay、global clip5、样本与一轮日程。

一轮完成后保存 `epoch_0001.pt`，在**相同 remaining64**评OCC+GT和STC+GT，分别报告Transport和Refiner、1/2/3s及平均mIoU/IoU/Moving。**训练、OCC评估、STC评估全为dense-valid**，不重演先前OCC评估known错配。每次评估仅rank0的一张GPU运行，另一训练rank等待；没有双卡评估。

固定第1轮，不按DEV选best。DEV64反复看过，不是独立测试。小预算结果只筛研究方向，不保证全集或收敛结论；不自动加轮数或换主线。最终比较器核对两组人口/known/日程/源策略/数据和实现指纹，拒绝混旧结果。

## 缓存、恢复与旧结果保护

- 旧CCR原始历史几何缓存仅require/read-only，命名空间和manifest检查保留；不创建新的learned feature bank。
- 新共享track派生缓存仅写入新实验arm目录 `track_history/rank*`，不覆盖历史cache根。
- Reader history不再计算未使用的固定patch/mask；memory读取与原版同算式、同顺序，测试逐数组字节一致。
- 同目录 `last.pt` 原子保存权重/Adam/完成游标/每rank RNG/已有评估。信号到达后完成当前global update或eval窗口再停；失败的半步不发布。
- 改arm、人口、world size、known、日程、实现或数据后拒绝resume。评估整数计数按完整窗口续跑。
- workflow只在上一arm真正complete后进入下一arm；正常提前停止不会误启动另一个训练。
- 新checkpoint用 `model.checkpoint_model()` 显式加载，旧工厂不会静默解释新结构。

## 服务器命令

尚未代启动服务器。同步当前分支后：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

# 一卡可设0，双卡训练设0,1。评估始终只在rank0一张卡进行。
export SCENE_TRAIN_GPUS=0,1
export SCENE_EPOCHS=1
export SCENE_PILOT_DIR="$PWD/outputs/p0_f9_shared_history/scene_reader_stage1_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_stage1/workflow.sh > "${SCENE_PILOT_DIR}.log" 2>&1 &
tail -F "${SCENE_PILOT_DIR}.log"
```

`REFERENCE_DIR`默认原 `stc_empirical_refiner500_20261014_194954`，仅取其中 `clean/results.json` 的remaining64 keys，不导入其权重或分数。数据/E14/旧cache路径沿用此前脚本，可用环境变量指定。

结束后：`cat "$SCENE_PILOT_DIR/summary.txt"`。分组细节在两arm的 `results.json`，完整评估在 `eval_epoch_0001.json`。续跑重新设置相同 `SCENE_PILOT_DIR`、`SCENE_EPOCHS` 和 `SCENE_TRAIN_GPUS`，重跑相同workflow命令，自动加载各arm自己的last。

不自动加入服务器首检或全量训练。若只想手动做数步管线检查，明确设置新 `SCENE_ARM_DIR` 后可单独 `bash tools/source_scene_stage1/run.sh reader --max-updates 8`，总一轮余弦不变；它不是8步收敛试验。

## 独立只读：最近两帧匹配正确率审计

`association.py / audit.py / audit.sh` 不依赖训练结果，不改原匹配。

- 同128个按场景round-robin选择的VAL窗口，GT/STC都只读取最近两帧历史语义、历史poses和时间。
- 原同类/世界XYZ质心互为最近邻/25m/s×名义0.5s规则先执行，GT框不作为匹配输入。
- 各帧独立按同类别标注框的体素中心包含关系归属instance_token，不用相同最近邻冒充真值。框归属为近似GT：固定主支持比例>=0.5，第二框支持>=0.2或并列标记为merged/overlap，低purity/无框单列；不调阈值追正确率。
- 可多个fragment对应同一GT身份，不强制一框一域；recall按两帧均有可归属component的**不同instance**计数，不把fragment重复当多次召回。
- 报已归属配对precision、可判定覆盖率、可识别实例recall、碎裂/合并/歧义、正确配对的KTA速度误差（GT框中心差用实际时间），以及类别/距离/GT速度分组配对计数。
- 额外执行原registration仅作诊断：分别统计正确/错误/未知身份配对被accept/reject多少，不反过来改变原匹配或训练。
- 这不是完美逐体素实例GT，条件precision必须与覆盖率一起看；多个相邻窗口也不是独立统计样本。128审计人口不冒充remaining64质量人口。

```bash
export PAIR_AUDIT_DIR="$PWD/outputs/p0_f9_shared_history/source_pair_audit128_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_stage1/audit.sh > "${PAIR_AUDIT_DIR}.log" 2>&1 &
tail -F "${PAIR_AUDIT_DIR}.log"
```

仅CPU，不需要原始图像或点云；需要现有 `v1.0-trainval` 的sample/annotation/instance/category/pose元数据。逐8个完整窗口保存，续跑同PAIR_AUDIT_DIR并设PAIR_AUDIT_RESUME=1。不会自动下载数据，也不自动启动这个审计。

## 本地验证记录

已用安全非base yoloe环境完成专项CPU/CUDA检查：初始KTA/Transport argmax、六帧CE/Lovász梯度、位置与时间可辨识性、无效历史mask、空source、同随机共享模块初始化、memory原数组一致、checkpoint加载/恢复、同预算比较拒绝known/人口错配、GT/STC实际小网格评估和整数续评；旧shared-track/GT/H6/DDP/known/全集接口回归亦运行。

最终专项与上述相关旧回归共 **59 passed / 53.18s**；新训练/审计CLI帮助和三个Bash入口语法检查通过。随后仅补充固定DEV64/名义dt校验和CPU执行预算指纹；不改变数值前向。没有代跑服务器、真实匹配正确率或新模型质量结果。

真实已下载子集的两个窗口（17/24 sources），两模型均完整H6/200×200×16前向通过，memory为6×16×100×100×16，初始残差零及Transport argmax通过。没有在VAL上优化、没有新质量分数或FPS。

另有可显式运行的 `replay_check --windows 1 --train-updates 2`，只用ZIP内原TRAIN窗口检查两步完整反向，所有临时随机权重丢弃，不保存checkpoint、不将其作为正式训练或效果证据。**本地RTX3050 4GiB实测在baseline的完整尺寸H6 Lovász稳定排序阶段OOM，未完成真实尺寸两步训练。** 该失败保留，不缩小监督、不降时距/换loss；服务器完整尺寸训练显存/速度尚待验证。小网格CPU/CUDA完整反向和精确Adam恢复已通过，不能混称完整真实数据训练已通过。最终测试数字以交付摘要为准。
