# 五组新增消融：仅训练入口

最新预算见 `PAPER_ABLATION_ONE_EPOCH_BUDGET_20261107_CN.md`。本次只实现训练，不运行数据集训练或质量评估，不实现新的评估器。

## 配置与预算

| CLI arm | 表中ID | 实际变化 | 数据×轮数 | 默认训练卡数 |
|---|---|---|---|---|
| `c_transport` | C | 删除Memory和Refiner；只训练KTA＋STWM的Transport输出 | 全量20430×1 | 2 |
| `d_kta_refiner` | D | 删除Reader/SceneTemporal/FutureDecoder/学习历史yaw；固定KTA搬运Prior和动态Memory，训练FPN＋Memory＋Refiner | 全量20430×1 | 2 |
| `m0_no_memory` | M0 | Memory恒零，删除其lift/importance/empty参数；其余Prior/U-Net/运动保持 | 20%4086×1 | 1 |
| `m1_ego_memory` | M1 | Memory动态点保留各自历史世界位置，只按未来ego投影；不做source平移/旋转，静态warp和importance/覆盖归一化不变 | 20%4086×1 | 1 |
| `s0_pool` | S0 | 原生P2 stem/投影后空间平均池化，保持三尺度坐标/通道；移除学习深层和FPN自顶向下融合 | 20%4086×1 | 1 |

M1的Prior、Strong和source运动预测仍是完整学习版本，只有Memory不使用source运动。M0/M1/C不再保留无效的学习历史yaw头。D保留空间FPN，因为Refiner仍需历史特征；“无STWM”不是删除共享空间特征编码器。S0仍有多尺度局部采样，只比较学习FPN与池化金字塔。

所有组从随机初始化开始，仅复制参考E构造器产生的**新鲜随机公共参数**；不读取E训练权重。TRAIN为原始GT，四历史六未来、dense-valid、GT future ego、原KTA/前向关联、无ICP、full6 CE＋Lovasz、global B4/source128。没有污染、真实STC训练、辅助loss、历史衰减、FREE偏置、后验/距离/norm适配。

## 如何复用E，避免多训控制组

全量组读取 `adaln_forward_full_2ep_20261026_215351/contract.json`，训练到第一轮结束（约5145步），学习率分母仍为原两轮总更新10292。其逐步LR严格匹配原E的epoch1，**不是一轮完整余弦**。最终比较E应取该旧任务的 `epoch_0001.pt`，不能拿epoch2混入本表。

短组读取 `adaln_forward_single_free20_20261025_161644/C_train/contract.json`，同一4086窗口及顺序，使用原单轮完整余弦（约1031步）。两份参考只读取合同JSON；保留其DEV keys作为将来评估元数据，不打开VAL、STC或Moving样本。训练卡数、软件版本、关键代码SHA、TRAIN相关文件SHA、种子、LR和顺序不符会报错，不静默更换控制。

若参考目录不同，设置 `ABLATION_REFERENCE=/实际目录/contract.json`；必须是同一种已匹配E合同，不能任意换旧版合同绕过核验。若实际原全量E是单卡，则设置 `ABLATION_GPUS=0`匹配它；默认针对已约定的双卡全量控制。

## 服务器命令

同步本次代码后，在仓库根目录、OccFM环境中执行。以下命令只训练，不会轮末评估；五组是独立随机任务。

```bash
cd /root/nas/occ/swfm
conda activate OccFM

# 两个全量一轮任务，默认双卡，顺序执行。
bash tools/paper_ablation_train/run.sh c_transport
bash tools/paper_ablation_train/run.sh d_kta_refiner

# 三个20%一轮任务，默认单卡，匹配已有小E控制。
bash tools/paper_ablation_train/run.sh m0_no_memory
bash tools/paper_ablation_train/run.sh m1_ego_memory
bash tools/paper_ablation_train/run.sh s0_pool
```

这些是五条独立命令，不建议在同一对卡上同时启动全量组。短组可分别用 `ABLATION_GPUS=0` / `ABLATION_GPUS=1`，两卡各跑一组，仍然保持每组单卡。换卡编号不换world-size。

关闭终端仍需继续时，用已激活环境下的nohup，例如：

```bash
export ABLATION_DIR="$PWD/outputs/p0_f9_shared_history/ablation_c_train1_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/paper_ablation_train/run.sh c_transport > "${ABLATION_DIR}.log" 2>&1 &
tail -n 10 -f "${ABLATION_DIR}.log"
```

中断后续训同一任务：

```bash
export ABLATION_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/原任务目录
ABLATION_RESUME=1 bash tools/paper_ablation_train/run.sh c_transport
```

`arm`、原卡数、TRAIN文件、代码、日程及目录必须一致。`--max-updates N`仅在全局第N步暂停，不缩短余弦日程；恢复时去掉该参数即可继续剩余窗口。不要把不同arm的last移入新任务。完成后再次续训只确认完成，不重新优化。

## 输出及磁盘约束

- `contract.json`：配置、匹配E来源SHA、顺序/预算/初始化/优化器/代码校验；`cache_policy.json`记录缓存边界。
- `progress.jsonl`和`summary.txt`/`results.json`：训练进度、CE/Lovasz/梯度/计时；没有mIoU、IoU或Moving分数。
- 默认每64步原子更新 `last.pt`，最多保留一份 `last.previous.pt`；末轮保存 `epoch_0001.pt`。不保留每步/每2000步的权重集合。可用 `--checkpoint-every`改变恢复频率，不改变科学日程。
- 各checkpoint含Adam、每rank RNG及完整更新游标。两卡按全局窗口数归一化并SUM梯度，再clip/Adam；空rank不补样本，无source的C窗口不虚造梯度或触发AdamW衰减。
- 只读既有TRAIN纯几何cache，缺项直接报错，不回退到旧builder；新增前向track仅保存在有界RAM。学习特征/Memory/Prior/logits不落盘，无新的多GB张量缓存。
- 旧训练/推理入口、权重、缓存协议及源文件均未修改。新checkpoint使用独立协议，暂未接入旧严格epoch2评估loader；后续按用户要求另接评估。

## 验证范围

共47项新旧检查通过，其中32项新增训练测试。覆盖五组真实CPU/CUDA六未来前向/反向、输入不读取未来占据、公共随机参数/RNG匹配、删除分支不存在、M1保持世界点/原importance、S0三尺度坐标、TRAIN合同/顺序/LR预算、只读缓存、真实Adam/RNG暂停续训及完成后不多训。另用双CPU/Gloo进程核对全局损失归一化和空rank同步；启动器在小模型上完整走通两步训练/轮末保存，只构造一个TRAIN源、没有质量评估。原full/pilot关键测试一并回归。两张物理CUDA卡的全数据运行仍需服务器执行；本地没有代跑科学实验或质量评估。
