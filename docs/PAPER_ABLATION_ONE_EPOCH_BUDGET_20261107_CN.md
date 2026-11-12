# 用户确定的消融预算：全量一轮＋20%一轮，只报mIoU/IoU

2026-11-12评估安排。三张消融表只评OCC。E、M2和表三的FPN＋AdaLN-Zero＋GT共用正式两轮模型的原版VAL4369结果，mIoU/IoU为48.711339/62.809965。表三FPN＋Query加法＋GT复用已有OCC全集结果。待评共八行，表一A/B/C/D，表二M0/M1，表三池化金字塔和完整模型Pred轨迹。

2026-11-07执行更新：五组训练代码已在 `tools/paper_ablation_train` 独立实现，运行/恢复说明见 `PAPER_ABLATION_TRAIN_ONLY_20261107_CN.md`。本次只实现并测试训练入口，未启动数据集实验；所有评估保持关闭。下面的表格仍是后续完整实验计划，不表示已实现或运行评估。

用户随后要求一个总脚本、五组全部双卡：`tools/paper_ablation_train/workflow.sh`顺序运行，globalB4/数据/步数/日程不变。小E历史为单卡，新M0/M1/S0双卡，合同明确执行变化及负载平衡策略，不声称单卡字节复现；不为此增加控制训练。

本文件覆盖PAPER_ABLATION_PLAN_CURRENT_FORWARD_20261029_CN.md中此前“C/D完整两轮”的建议。用户明确要求：消融只报告mIoU和IoU，主组件C/D也只训练完整TRAIN一轮，其余短组仍为TRAIN20%一轮。本轮逐行记录配置与复用安排，不运行训练/推理或改动已冻结实现。

## 共同协议

- 输入四帧历史，预测六帧，完整六帧CE＋Lovasz；同global B4/source128、ordered keys/seed/初始化和原AdamW/LR设置。
- 当前forward关联/FPN/2+2 AdaLN-Zero/pure-history Memory/U-Net/no ICP为完整E；除指定消融项外保持当前版本。
- 训练用原GT TRAIN。评估使用OCC历史和dense-valid原版读出。
- 本轮GT轨迹配置统一评VAL4369，报告1/2/3s平均mIoU、IoU。Pred使用同一条预测轨迹贯穿Decoder和全部未来几何，评估预测覆盖窗口。
- 消融不报告或计算MovingMicro/MovingMacro，不读Moving支持标注；不附加参数/FPS实验。
- 所有新训练组从随机初始化开始，公共新鲜模块/RNG匹配；不从原完整E训练权重微调成删模块模型。
- 新完整训练只跑一轮，不再默认加到两轮；只保存恢复last和固定末轮权重/必要合同及整数confusion，不保存稠密中间张量缓存。

## 表一：主组件，五行

| ID | 配置 | KTA | STWM | Memory＋Refiner | 训练数据和轮数 | 本次操作 |
|---|---|---|---|---|---|---|
| A | 零物体运动baseline | 关 | 关 | 关 | 无训练 | 直接几何推理 |
| B | KTA搬运 | 开 | 关 | 关 | 无训练 | 直接几何推理 |
| C | KTA＋STWM | 开 | 开 | 关 | TRAIN100%=20430窗×1轮，约5145更新 | 已训完，待评全集 |
| D | KTA＋Refiner | 开 | 关 | 开 | TRAIN100%=20430窗×1轮，约5145更新 | 已训完，待评全集 |
| E | 完整模型 | 开 | 开 | 开 | TRAIN100%×2轮 | 复用epoch_0002的原版全集结果 |

A/B仍使用正确future ego。A只把物体自身速度/位移和yaw修正设为零，同步重建Strong/CLEAR/几何；不能取消自车投影。B为规则KTA，无学习运动修正或Refiner。

C没有Refiner/Memory支路，直接对学习运动形成的Transport施加六帧CE＋Lovasz。D保留共享FPN和四帧历史Memory/Refiner，KTA驱动几何及Memory搬运，移除STWM Reader/Temporal/Decoder及学习历史yaw，不引用E的运动输出。完整E checkpoint直接取Transport或推理时关闭运动可作只读诊断，不能替代C/D独立训练。

C/D沿用了 `adaln_forward_full_2ep_20261026_215351` 原两轮余弦的前半程（分母10292，5145步停止）。这是已完成训练的日程。表一E按最新决定使用该目录 `epoch_0002.pt` 的原版VAL4369结果，E与C/D的差值同时包含结构及训练轮数差异。

C/D评估以其各自完成的一轮断点和实际训练合同为准。

## 表二：Memory，三行

| ID | Memory配置 | 其余网络 | 训练数据和轮数 | 本次操作 |
|---|---|---|---|---|
| M0 | 无Memory：同形状零输入送入Refiner | 完整E其他部分 | TRAIN20%=4086窗×1轮，约1031更新 | 已训完，待评全集 |
| M1 | 四帧历史Memory，仅自车对齐 | 完整E其他部分 | TRAIN20%=4086窗×1轮，约1031更新 | 已训完，待评全集 |
| M2 | 四帧历史Memory，自车＋source运动对齐 | 完整E | TRAIN100%×2轮 | 共用表一E的原版全集结果 |

M1只改变动态Memory的source位移/旋转对齐；静态Memory、importance/coverage、Transport Prior、Strong和STWM学习预测均按完整配置。训练和推理都使用同一种Memory策略。M0与M1使用同一个4086窗口列表、全局组顺序和一轮余弦。M2采用两轮全量E，涉及M2的差值也包含训练数据量和轮数差异。

## 表三：多尺度与ego条件，四行

| 多尺度 | Decoder ego条件 | future ego | 训练预算 | 本次操作 |
|---|---|---|---|---|
| 池化金字塔 | AdaLN-Zero | GT | TRAIN20%×1 | S0已训完，待评全集 |
| FPN | Query相加 | GT | TRAIN100%×2 | 复用已完成的VAL4369结果 |
| FPN | AdaLN-Zero | GT | TRAIN100%×2 | 共用E、M2的原版全集结果 |
| FPN | AdaLN-Zero | Pred | 同E权重 | 用已有预测轨迹推理，评估完整VAL的预测覆盖窗口 |

FPN＋Query来源为 `fpn_unet_full_2ep_20261021_112202/epoch_0002.pt`，沿用原评估报告，OCC mIoU/IoU为48.513241/62.593428。评估设置见 `FPN_UNET_ALL4369_20261023_CN.md`。第三表逐行保留训练预算和报告来源。

第四行使用E的 `epoch_0002.pt`，Decoder和全部未来几何使用同一条外部planner预测轨迹。以VAL4369为输入清单，按实际预测覆盖报告分数。既有planner缓存曾覆盖4219窗，比较GT与Pred时取同一批窗口。

## 精确训练账单

已完成的新训练和复用来源如下。

| 预算 | 新增配置 | 新训练数 | 约global updates |
|---|---|---:|---:|
| 全TRAIN×1 | C、D | 2 | 2×5145=10290 |
| TRAIN20%×1 | M0、M1、S0 | 3 | 3×1031=3093 |
| 既有完整模型（全量epoch2） | E、M2、表三第3和第4行 | 0 | 0 |
| 既有FPN＋Query全集结果 | 表三第2行 | 0 | 0 |
| 几何参考 | A、B | 0 | 0 |
| 总新增 | — | 5 | 约13383 |

五组新增训练均已完成一轮。三张表共用同一份E的两轮原版全集结果。

## 论文与运行记录边界

- 每行标训练数据量和轮数。C/D为全量一轮，M0/M1和池化为TRAIN20%一轮，完整E为全量两轮。涉及不同训练预算的分差同时反映配置与训练预算差异。
- 三表只显示OCC的mIoU/IoU。1/2/3s明细在JSON，主表固定平均。

## 两卡并行评估

`tools/paper_ablation_eval/run.sh` 先准备一次VAL窗口和位姿清单，然后并发启动八个独立进程。第一张卡负责A、C、M1和池化，第二张卡负责B、D、M0和Pred。每个进程评自己的OCC输入，保存各自的整数计数。此前已启动的任务可以在协调进程退出后用原目录恢复已有OCC计数。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
export ABLATION_EVAL_GPUS=0,1
export ABLATION_EVAL_DIR="$PWD/outputs/p0_f9_shared_history/ablation_all_parallel_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/paper_ablation_eval/run.sh > "${ABLATION_EVAL_DIR}.log" 2>&1 &
tail -f "${ABLATION_EVAL_DIR}.log"
```

看到 `ABLATION_WAVE_STARTED 8 processes / 4 per GPU` 后，进度见总目录内的各组 `.log`，分数汇总在 `summary.txt`。恢复时保留原 `ABLATION_EVAL_DIR` 和同一设置，重跑入口会从各组计数恢复。默认训练来源为 `ablation_five_dual_20261107_203511`，E和planner路径分别可用 `FORWARD_FULL_DIR`、`PLAN_CACHE` 指定。

如果八组OCC已经累计到最后一窗，可直接从计数汇总三张表。

```bash
python -m tools.paper_ablation_eval.report "$ABLATION_EVAL_DIR" | tee "$ABLATION_EVAL_DIR/summary_occ.txt"
```
