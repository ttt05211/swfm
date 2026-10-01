# 原 V18 XY 专项：严格冻结预测 yaw / GT-yaw 训练课程

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan / implementation
- Origin Date: 2026-10-01
- Verification Status: UNVERIFIED（真实服务器训练收益尚未验证）
- Version Label: xy_specialist_paired20_v1

## 目的与证据边界

原 dev64 motion-gap audit：仅替换 GT XY、保留预测 yaw 的整体 ΔmIoU 为 +5.233286 pp；
仅 GT yaw 为 +0.170498 pp；两者共同替换为 +6.125720 pp。这证明该样本有位置预测缺口，
不证明这些收益可被学到，也不证明原多任务梯度就是失败原因。

之前 full-core 20% continuation 和 source interaction 均未通过筛选。本次不新增模块、
不重新跑 oracle，不重复 frozen-latent 小 MLP；仅检验原网络 XY 专项训练和训练姿态课程。

## 两组一次验证

| 组 | train shape loss 的预测 yaw | calibration / dev / deployment yaw |
|---|---|---|
| frozen_yaw_xy | 原 Clean-E14 yaw | 原 Clean-E14 yaw |
| scheduled_gt_yaw_xy | 按概率选 GT yaw，或原 E14 yaw | 原 Clean-E14 yaw |

两组从同一 Clean-E14 出发，训练原 encoder、时空块、future-query decoder、XY head。
自己的 yaw/existence head 参数 requires_grad=False，输出不参与最终预测。
**仅冻 head 参数不能保证输出不变**：共享特征变化后它仍会变。因此最终输出必须组合：

`learned_core(history).XY + frozen_E14(history).yaw/existence -> original SE(2)/A1 renderer`

这是两个神经前向，不假称单模型无开销。评估中原 E14 前向在所有候选之间共享。
训练只在 RAM 存储每个窗口原 BF16 per-window 前向的 yaw/existence 小数组（上限64MiB），
不建磁盘缓存，不保存 context/latent，不把 GT 当模型输入，不根据未来 GT 剪掉推理 source。

## GT yaw 混合的具体定义

它是 **shape-loss pose curriculum**，不是自回归网络输入上的经典 teacher forcing。
按每个 yaw-enabled、监督有效且有 yaw 标签的 source/horizon，Bernoulli 选择 GT yaw。
概率从0.5线性下降，在前2/3 successful updates 内降到0；最后至少1/3完全用原预测 yaw。
课程按实际有 XY 监督的 batch 总数固定，空监督窗口单独记录，不会使零GT阶段消失。
随机掩码由固定seed+update决定；概率与选择数量写入 progress.jsonl。
不直接插值角度，不存在 ±π 的插值绕远问题。

GT residual/displacement 的 source-centred SE(2) 定义保持原样，不改成 box-centre 平移。
两组唯一优化目标：

`L = SmoothL1_beta1(predicted_XY_residual, GT_source_residual) + 0.25 * original_soft_SE2_overlap`

shape 的 target 始终来自原 GT source pose；混合只改变 predicted shape 的 yaw。
不优化 existence BCE / periodic yaw，不加新正则、hard-example 权重或阈值扫参。
GT yaw、validity、监督标记只参与 loss，正式 forward 是固定五项历史输入白名单。

## 数据、训练与选择

- full TRAIN20430 按上一版的同一规则、同一seed20260930选4086窗口；不做错误导向重抽。
- 排除32个 TRAIN calibration 场景；其64个窗口不参与本次梯度更新。
- 原 E14 曾见这些 TRAIN 场景，因此不是新的独立测试；dev64也已复用，结论属于探索性。
- 每组固定3个子集epoch，whole-window/source packing默认256 sources / 8 windows，无截断。
- 完整恢复 E14 AdamW moments 与保存的尾部 LR，并固定该 LR；不重启5e-4高学习率。
- 每epoch使用 TRAIN calibration 的实际 renderer mIoU 选 best，包含update0零修改对照。
- 最后一趟共用 raw/Strong/E14/Moving support，评估两个 selected 与两个 last；不在dev选epoch。
- 沿用原门槛：整体 ΔmIoU ≥0.30pp、ΔMovingMicro ≥1pp，1/2/3s 与整体四项指标非负。
- update0、smoke、last诊断不能作为通过的可部署模型。失败不自动重训/扩数据/开新实验。

原 V18 的历史执行是10轮 cosine + restored AdamW 固定 tail LR续训到15轮，选E14。
本次是从已训练好的 E14 做受控专项续训，不能把3个20% epoch说成重现15轮全集从头训练。

## 服务器执行

使用已确认的 OccFM 环境与文件路径，脚本自行设置所有变量，无需旧shell环境。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
bash tools/real_motion/run_p0_f9_v18_xy_specialist_20pct.sh screen
```

一条 screen 命令含 CPU synthetic tests 和 CUDA预检查，不要求再单独跑一趟服务器 smoke。
可选 `smoke` 只验证管线，不证明有效性。默认两组一次跑完，不新增生成能力实验。

输出位于 `outputs/p0_f9_v18_xy_specialist/screen20_<time>_<sha>/model/`：
`summary.txt`、`summary.json`、`train_history.json`、`progress.jsonl`、`execution_contract.json`，
以及每组 best/last 两个 checkpoint；不覆盖原 E14 或之前实验。
新 checkpoint 有独立protocol、base SHA、config fingerprint、deployment组合合同与通过状态；
不能直接用原 V18 loader 当一个完整单网络部署，必须加载对应冻结 E14 并组合预测。

## 本地实现验收（非真实训练收益）

- 专项新增14项测试，与原 source-interaction 回归合计29项通过。
- dependency-light 全仓库：643 passed / 4 skipped / 1 deselected，28条原有 warnings。
- 446个 Python 文件 AST 检查通过，Bash入口语法检查通过。
- 完整 synthetic CLI 跑通缓存、两组更新、实际 SE(2)/A1、calibration、selected/last 和存盘。
- 自动验证：XY/core确实有梯度、head参数不变、最终 yaw/existence 与冻结输出逐元素相同；
  改动/删除未来标签不改变 forward；六帧部署只请求 include_gt=False；GT课程末段为零。
- 本地仅用项目安全虚拟环境，不调用故障的 Anaconda base。没有服务器数据/GPU，尚未跑真实screen。
