# 用户确定的消融预算：全量一轮＋20%一轮，只报mIoU/IoU

2026-11-07执行更新：五组训练代码已在 `tools/paper_ablation_train` 独立实现，运行/恢复说明见 `PAPER_ABLATION_TRAIN_ONLY_20261107_CN.md`。本次只实现并测试训练入口，未启动数据集实验；所有评估保持关闭。下面的表格仍是后续完整实验计划，不表示已实现或运行评估。

用户随后要求一个总脚本、五组全部双卡：`tools/paper_ablation_train/workflow.sh`顺序运行，globalB4/数据/步数/日程不变。小E历史为单卡，新M0/M1/S0双卡，合同明确执行变化及负载平衡策略，不声称单卡字节复现；不为此增加控制训练。

本文件覆盖PAPER_ABLATION_PLAN_CURRENT_FORWARD_20261029_CN.md中此前“C/D完整两轮”的建议。用户明确要求：消融只报告mIoU和IoU，主组件C/D也只训练完整TRAIN一轮，其余短组仍为TRAIN20%一轮。本轮逐行记录配置与复用安排，不运行训练/推理或改动已冻结实现。

用户随后指出FPN＋Query加法早已训练，核验后调整复用策略：该版本既有TRAIN20%×1，也有FULL×2（旧full目录fpn_unet_full_2ep_20261021_112202）。第三表的ego条件比较直接复用旧pinned关联下已经匹配的Additive/AdaLN两组，不把旧Additive与最新forward的E直接相减。这样新增训练由6组降为5组：2个全量一轮＋3个20%一轮。第三表多一行旧关联AdaLN参照，不新增训练。以下为更新后的预算。

## 共同协议

- 输入四帧历史，预测六帧，完整六帧CE＋Lovasz；同global B4/source128、ordered keys/seed/初始化和原AdamW/LR设置。
- 当前forward关联/FPN/2+2 AdaLN-Zero/pure-history Memory/U-Net/no ICP为完整E；除指定消融项外保持当前版本。
- 训练用原GT TRAIN；STC用于评估，不进行STC监督训练或新的标签校准。
- 结构表统一关闭历史去抖动、FREE/距离/后验校准、age和norm适配。全部使用dense-valid。
- 默认评估同一固定remaining64，OCC＋GT历史和STC＋GT历史各报告mIoU、IoU，主数字为1/2/3s平均。S3单独用同一条Pred future ego贯穿Decoder和全部未来几何。
- 消融不报告或计算MovingMicro/MovingMacro，不读Moving支持标注；不附加参数/FPS实验。
- 所有新训练组从随机初始化开始，公共新鲜模块/RNG匹配；不从原完整E训练权重微调成删模块模型。
- 新完整训练只跑一轮，不再默认加到两轮；只保存恢复last和固定末轮权重/必要合同及整数confusion，不保存稠密中间张量缓存。

## 表一：主组件，五行，完整TRAIN一轮协议

| ID | 配置 | KTA | STWM | Memory＋Refiner | 训练数据和轮数 | 本次操作 |
|---|---|---|---|---|---|---|
| A | 零物体运动baseline | 关 | 关 | 关 | 无训练 | 直接几何推理 |
| B | KTA搬运 | 开 | 关 | 关 | 无训练 | 直接几何推理 |
| C | KTA＋STWM | 开 | 开 | 关 | TRAIN100%=20430窗×1轮，约5145更新 | 新训练1组 |
| D | KTA＋Refiner | 开 | 关 | 开 | TRAIN100%=20430窗×1轮，约5145更新 | 新训练1组 |
| E | 完整模型 | 开 | 开 | 开 | 同完整TRAIN一轮 | 复用既有epoch_0001.pt，不新增训练 |

A/B仍使用正确future ego。A只把物体自身速度/位移和yaw修正设为零，同步重建Strong/CLEAR/几何；不能取消自车投影。B为规则KTA，无学习运动修正或Refiner。

C没有Refiner/Memory支路，直接对学习运动形成的Transport施加六帧CE＋Lovasz。D保留共享FPN和四帧历史Memory/Refiner，KTA驱动几何及Memory搬运，移除STWM Reader/Temporal/Decoder及学习历史yaw，不引用E的运动输出。完整E checkpoint直接取Transport或推理时关闭运动可作只读诊断，不能替代C/D独立训练。

E的复用来源为原 `adaln_forward_full_2ep_20261026_215351/epoch_0001.pt`，约5145步。它位于原两轮余弦的中途，不能声称它是“重新设计的一轮完整余弦”的终点。因此C/D保持原两轮余弦的学习率函数（scheduler horizon约10292步），但实际运行只到第1轮/约5145步停止，配合同一原epoch1训练顺序、globalB4、world size和初始化；不继续第二轮。这样无需额外重训E，同时表内各学习模型的暴露次数和学习率过程一致。

未来实现需独立记录run_epochs=1、schedule_reference_horizon=10292和stop_update≈5145，保存实际游标；不能把旧两轮入口强行改头后伪造来源或直接用只接受epoch2的旧评估器。若源epoch1缺失或合同不符，必须先报告，不能静默用epoch2代替。若改成一轮内完整降到底的新余弦，则E也需新增一组匹配训练；这不是本次省资源的默认方案。

## 表二：Memory，三行，TRAIN20%一轮协议

| ID | Memory配置 | 其余网络 | 训练数据和轮数 | 本次操作 |
|---|---|---|---|---|
| M0 | 无Memory：同形状零输入送入Refiner | 完整E其他部分 | TRAIN20%=4086窗×1轮，约1031更新 | 新训练1组 |
| M1 | 四帧历史Memory，仅自车对齐 | 完整E其他部分 | TRAIN20%=4086窗×1轮，约1031更新 | 新训练1组 |
| M2 | 四帧历史Memory，自车＋source运动对齐 | 完整E | 同TRAIN20%×1轮 | 复用已有小预算E，不新增训练 |

M1只改变动态Memory的source位移/旋转对齐；静态Memory、importance/coverage、Transport Prior、Strong和STWM学习预测均按完整配置。训练和推理都使用同一种Memory策略。M0/M1与M2使用同一个4086窗口列表、全局组顺序和一轮余弦；按最新用户要求，新组双卡，旧M2为单卡，明确记录该执行差异。

小预算E来源为 `adaln_forward_single_free20_20261025_161644/C_train/epoch_0001.pt` 的raw读出，关联必须是当前forward，FREE/age关闭。最终用原contract核验，而不是仅依靠目录名。不能混入旧pinned关联的AdaLN或FPN/U-Net对照。无法核验时，增加一组匹配的小预算E（而非重新训练完整两轮）。

## 表三：多尺度与ego条件，TRAIN20%一轮，ego条件旧匹配对照复用

| ID | 多尺度 | Decoder ego条件 | future ego | 历史关联/训练预算 | 本次操作 |
|---|---|---|---|---|---|
| S0 | 池化金字塔 | AdaLN-Zero | GT | 当前forward；TRAIN20%×1 | 新训练1组 |
| S1a | FPN | Query相加 | GT | 旧shared-pinned；已完成TRAIN20%×1 | 复用旧匹配对照，不新训 |
| S1b | FPN | AdaLN-Zero | GT | 旧shared-pinned；已完成TRAIN20%×1 | 复用旧匹配对照，不新训 |
| S2 | FPN | AdaLN-Zero | GT | 当前forward；已完成TRAIN20%×1 | 复用同一个小预算E，不新训 |
| S3 | FPN | AdaLN-Zero | Pred | 当前forward；使用S2同一权重 | 只替换推理轨迹，不新训 |

本表按比较项分组：S0对S2回答多尺度构建；S1a对S1b回答条件方式；S2对S3回答轨迹条件鲁棒性。关联列必须保留，不对S1a/S2作“只改变ego条件”的差值。S0保留最新U-Net/forward关联/条件方式，仅比较多尺度构建方案，并保持实际输出尺度100/50/25及坐标/通道约定。旧的同时改变FPN＋U-Net的结果不能充当池化单项控制。

旧匹配一轮对照已报告的最终Refiner数字（仅mIoU/IoU）：

| 旧shared-pinned下的条件方式 | OCC mIoU | OCC IoU | STC mIoU | STC IoU |
|---|---:|---:|---:|---:|
| Query相加（S1a） | 46.810713 | 60.794105 | 17.202257 | 28.548596 |
| AdaLN-Zero（S1b） | 47.776497 | 60.943147 | 17.470307 | 28.559785 |

上述匹配控制见FPN_ADALN_ZERO_TRAIN20_20261024_CN.md及原控制fpn_unet_20pct_1ep_20261021_100854/results.json，原AdaLN结果用户已回传；训练顺序/预算/数据和world size在当时入口校验。复用时继续核验原报告，保留旧关联的范围说明。它支持该关联条件下的ego注入方案比较，不声称已经完成最新forward下的单变量对照。若未来坚持第三表所有行都必须当前forward，则才恢复新增S1小训练；当前省资源默认不做此重复。

全量两轮Query加法checkpoint也已存在，但不能将其原始分数直接填入当前TRAIN20%一轮表；不借推理时换前向关联掩盖其训练关联差异。完整权重可留作后续明确同预算的历史配置比较，不需要本轮再训练。

S3用外部planner＋yaw现有预测，Decoder/Strong/CLEAR/Prior/Memory均接受同一条Pred轨迹。它是同一模型的推理条件鲁棒性检查，不是另训一个Pred模型。第三表使用小预算S2的权重比较；不能把完整两轮E的Pred分数混进来。若planner无法覆盖原64窗口，另取同覆盖交集做GT/Pred成对参考并标实际窗数，不增加训练。

## 精确训练账单

假设两个既有E checkpoint及合同可核验复用：

| 预算 | 新增配置 | 新训练数 | 约global updates |
|---|---|---:|---:|
| 全TRAIN×1 | C、D | 2 | 2×5145=10290 |
| TRAIN20%×1 | M0、M1、S0 | 3 | 3×1031=3093 |
| 既有E（全量epoch1与小预算epoch1） | 表一E、表二M2、表三S2/S3 | 0 | 0 |
| 旧pinned匹配条件对照 | 表三S1a/S1b | 0 | 0 |
| 几何参考/轨迹替换 | A、B、S3 | 0 | 0 |
| 总新增 | — | 5 | 约13383 |

所有新增训练都只运行一轮；完整E在每个表的训练预算不同，分别引用对应checkpoint。总体新增更新量约为原来单次完整TRAIN两轮（10292步）的1.30倍，不是5次原完整训练，也不是实测GPU小时。不同分支每步成本不相同。

如果原小预算E不能复用，新增数变为2组全量一轮＋4组小训练，其中一个小E供两张表共同使用；如果原全量epoch1不能复用，再按匹配日程处理完整控制。若旧pinned条件匹配报告不能核验则先标缺，不自动混成forward单变量对照。当前没有根据假设擅自重训或切换到epoch2。

## 论文与运行记录边界

- 表一标“TRAIN100%×1”，表二和表三标“TRAIN20%×1”；不将两轮E的48.711/62.810或18.722/28.446直接填写到一轮消融表。
- 正式主结果仍保留原两轮模型，消融一轮是单独的资源受限协议。训练后变化大小以实际结果为准，不因旧E后期变化小就假设新变体也已收敛。
- 三表只显示OCC和STC各自的mIoU/IoU，不报告Moving。需要1/2/3s明细时放JSON，主表固定平均。
- STC适配四行补充表仍可使用原最终epoch2做冻结推理，单列有监督校准来源，且只报mIoU/IoU；它不属于这六组训练，不改变三张结构消融的raw协议。
- 本次用户要求逐项整理预算和已有训练复用，不代表已启动服务器训练；后续实现时使用独立入口，原权重、实验目录和论文原结果保留。
