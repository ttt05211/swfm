# 五组新增消融：统一双卡，仅训练入口

最新预算见 `PAPER_ABLATION_ONE_EPOCH_BUDGET_20261107_CN.md`。本次只实现训练，不运行数据集训练或质量评估，不实现新的评估器。

## 配置与预算

| CLI arm | 表中ID | 实际变化 | 数据×轮数 | 默认训练卡数 |
|---|---|---|---|---|
| `c_transport` | C | 删除Memory和Refiner；只训练KTA＋STWM的Transport输出 | 全量20430×1 | 2 |
| `d_kta_refiner` | D | 删除Reader/SceneTemporal/FutureDecoder/学习历史yaw；固定KTA搬运Prior和动态Memory，训练FPN＋Memory＋Refiner | 全量20430×1 | 2 |
| `m0_no_memory` | M0 | Memory恒零，删除其lift/importance/empty参数；其余Prior/U-Net/运动保持 | 20%4086×1 | 2 |
| `m1_ego_memory` | M1 | Memory动态点保留各自历史世界位置，只按未来ego投影；不做source平移/旋转，静态warp和importance/覆盖归一化不变 | 20%4086×1 | 2 |
| `s0_pool` | S0 | 原生P2 stem/投影后空间平均池化，保持三尺度坐标/通道；移除学习深层和FPN自顶向下融合 | 20%4086×1 | 2 |

M1的Prior、Strong和source运动预测仍是完整学习版本，只有Memory不使用source运动。M0/M1/C不再保留无效的学习历史yaw头。D保留空间FPN，因为Refiner仍需历史特征；“无STWM”不是删除共享空间特征编码器。S0仍有多尺度局部采样，只比较学习FPN与池化金字塔。

所有组从随机初始化开始，仅复制参考E构造器产生的**新鲜随机公共参数**；不读取E训练权重。TRAIN为原始GT，四历史六未来、dense-valid、GT future ego、原KTA/前向关联、无ICP、full6 CE＋Lovasz、global B4/source128。没有污染、真实STC训练、辅助loss、历史衰减、FREE偏置、后验/距离/norm适配。

## 如何复用E，避免多训控制组

全量组读取 `adaln_forward_full_2ep_20261026_215351/contract.json`，训练到第一轮结束（约5145步），学习率分母仍为原两轮总更新10292。其逐步LR严格匹配原E的epoch1，**不是一轮完整余弦**。最终比较E应取该旧任务的 `epoch_0001.pt`，不能拿epoch2混入本表。

短组读取 `adaln_forward_single_free20_20261025_161644/C_train/contract.json`，同一4086窗口及顺序，使用原单轮完整余弦（约1031步）。按用户新要求，三短组也改为双卡；旧E单卡、新组双卡及分配规则写入合同，不声称与单卡逐字节一致。公共随机初值、全局窗口组、目标/loss、优化器、LR和总更新预算仍保持。两份参考只读取合同JSON；保留其DEV keys作为将来评估元数据，不打开VAL、STC或Moving样本。软件版本、关键代码SHA、TRAIN相关文件SHA、种子、LR和顺序不符仍报错。

若参考目录不同，总脚本分别支持 `ABLATION_FULL_REFERENCE` / `ABLATION_SMALL_REFERENCE`；单组脚本支持 `ABLATION_REFERENCE`。必须是同一种E合同，不能任意换旧版合同绕过核验。所有新训练都要求恰好两张卡，`ABLATION_GPUS`默认`0,1`。

## 服务器命令

同步本次代码后，只运行一个总脚本，按C、D、M0、M1、S0顺序训练，**每组都占用同一对GPU**，不并发抢卡。每组仍是独立随机初始化，不会把前一组权重接到下一组。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export ABLATION_GPUS=0,1
export ABLATION_SUITE_DIR="$PWD/outputs/p0_f9_shared_history/ablation_five_dual_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/paper_ablation_train/workflow.sh > "${ABLATION_SUITE_DIR}.log" 2>&1 &
tail -n 20 -f "${ABLATION_SUITE_DIR}.log"
```

总目录含 `suite_contract.json`、`suite_state.json`、五个arm子目录及各自日志。`suite_state.json`显示当前组和已完成组。中途失败即停队列，不自动重试或跳过失败组。重新指定**同一个总目录**运行，会核验并跳过完成项，从未完成项的last继续：

```bash
export ABLATION_SUITE_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/原总目录
nohup bash tools/paper_ablation_train/workflow.sh > "${ABLATION_SUITE_DIR}.resume.log" 2>&1 &
```

也保留单组入口，例如：

```bash
bash tools/paper_ablation_train/run.sh c_transport
```

同任务续训要求arm、TRAIN文件、代码、日程、两卡设置和目录一致。双卡v2不会把旧单卡/v1的Adam/RNG断点静默转为新任务。单组的`--max-updates N`仅暂停，不改余弦日程。完成后不重新优化。

## 双卡利用与可观察性

保留原global B4/source128，不增加batch或重复样本。B4固定2+2分配；在满足窗口数平衡的划分中，选source数差最小的一种，按global组编号轮换rank，避免重窗口总落同一卡。B3只能2+1；B1只能一张卡计算，另一张参与归并，不补假窗口来伪造利用率。

`progress.jsonl`逐步记录`rank_windows`/`rank_sources`，每16步及首末步记录两rank的`input_wait_seconds`和`forward_backward_seconds`。控制台也显示窗口/source分布。CPU关联、Strong构建、两卡同步及稀疏组仍会影响利用率；这些高层优化减少可避免的倾斜，不保证100%占用或两倍加速。CPU预取仍每rank两线程、深度2，避免改科学预算或创建大磁盘缓存。

## 输出及磁盘约束

- `contract.json`：配置、匹配E来源SHA、顺序/预算/初始化/优化器/代码校验；`cache_policy.json`记录缓存边界。
- `progress.jsonl`和`summary.txt`/`results.json`：训练进度、CE/Lovasz/梯度/计时；没有mIoU、IoU或Moving分数。
- 开始即保存update0恢复点，之后默认每64步原子更新 `last.pt`，最多保留一份 `last.previous.pt`；末轮保存 `epoch_0001.pt`。不保留每步/每2000步的权重集合。可用 `--checkpoint-every`改变恢复频率，不改变科学日程。
- 各checkpoint含Adam、每rank RNG及完整更新游标。两卡按全局窗口数归一化并SUM梯度，再clip/Adam；空rank不补样本，无source的C窗口不虚造梯度或触发AdamW衰减。
- 只读既有TRAIN纯几何cache，缺项直接报错，不回退到旧builder；新增前向track仅保存在有界RAM。学习特征/Memory/Prior/logits不落盘，无新的多GB张量缓存。
- 旧训练/推理入口、权重、缓存协议及源文件均未修改。新checkpoint使用独立协议，暂未接入旧严格epoch2评估loader；后续按用户要求另接评估。

## 验证范围

此前47项新旧检查已通过。此次38项训练相关检查通过：保留CPU/CUDA六帧反向/Adam-RNG恢复回归，新增队列顺序、双卡要求、已完成跳过/中断续训、源数平衡/不丢不补窗口检查及实际子进程日志/停止处理。双CPU/Gloo使用新分配规则跑通真实训练循环、空rank、时间日志与两rank Adam/RNG暂停恢复，完整与续训参数/矩保持一致。Bash语法及diff检查通过。两张物理CUDA卡的全数据利用率仍需服务器实际日志确认；本地没有代跑科学实验或质量评估。
