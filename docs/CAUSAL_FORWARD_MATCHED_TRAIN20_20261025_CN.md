# AdaLN-Zero：向前关联 TRAIN20%×1 匹配实验

## 用户确认的范围

暂不运行刚准备的AdaLN完整两轮。先验证同网络、同训练预算下新关联是否值得纳入正式方法。只新增一个随机初始化训练组C，不加载A的训练权重，不增加训练轮数、网络层数、loss或其他校准。

| 组别 | 训练关联 | 评估关联 | 操作 |
|---|---|---|---|
| A recorded | 原shared-pinned | 原关联 | 复用已完成AdaLN TRAIN4086×1报告 |
| A fresh | 原shared-pinned | 原关联 | 同B的一次冻结A权重重跑，显式报告执行漂移 |
| B | 原shared-pinned | 新向前关联 | A的同一checkpoint，只读推理 |
| C | 新向前关联 | 新向前关联 | 相同fresh初始化，TRAIN4086×1，最后同64评估 |

最终报告分开列 `B−A fresh`（推理输入策略）、`C−B`（同新推理策略下的训练适配）、`C−A recorded`（总效果）和 `A fresh−A recorded`。不能将它们混成三个独立随机模型，也不隐藏重复推理的浮点差异。不自动判优/扩大两轮或全集，不据此承诺STC涨分。

## 固定科学设置

沿用当前AdaLN-Zero 2,816,122参数：FPN、Scene–Temporal2、cross-first Decoder2、三分支AdaLN-Zero、KTA Query+skip、历史yaw、纯历史Memory、普通3D U-Net不变。原seed20261011、顺序4086窗、globalB4/source128分组、1031左右actual updates、H6 CE+Lovasz、AdamW/LR/单轮cosine/BF16/FP64几何与完成的A契约逐项核对。参数和fresh state_dict逐元素相同，post-init CPU RNG一致；不继承A的Adam或trained weights。

TRAIN只用原GT TRAIN；STC只是评估。TRAIN/OCC/STC都dense-valid，显式GT future ego同时用于Decoder和所有renderer。无ICP、ego/motion/existence loss、导航、头、age0.5、FREE bias、norm适配、噪声或真实STC训练。

变量是此前冻结实验的**整套关联策略**，不是一个求解器单变量：从早到晚、速度预测后的XY距离、有速度greedy/未知速度MNN、连续优先、窗口内缺帧重连、现有速度门限和旧0.4m歧义差。直接复用`tools/causal_forward_track/track.py`，不重设规则/调阈值。最终t0组件/顺序/数量不变；KTA仍只最近一对差分/名义0.5s，没有t−1就不补造速度。旧回溯关联也未使用预测未来，不能宣传为“非因果变因果”。

## 缓存和执行

新增隔离`tools/causal_forward_train`，旧网络、AdaLN pilot/full、原关联与训练源码不改。复用原一轮FPN训练循环的私有命名空间，只有一个计数检查的source-policy元数据字面量在私有副本中改为新策略，避免合同误记shared-pinned；实际训练循环/Adam/归约/日程/恢复不变。私有evaluator使用新ForwardPredictor，不修改旧evaluator全局。

训练缓存只复用旧只读组件/pure t0几何，**不消费旧关联、registration或Strong背景**。独立有版本`C_train/forward_identity_history/rankN`缓存保存history-only新身份；namespace绑定新旧实现SHA，未来姿态不进入身份key（固定占位摘要）。CPU预取后为每窗替换KTA/anchor，保留原t0顺序及features/source预算。原缓存记录/字段不原地改。TRAIN、OCC和STC都用新身份重建读取与Memory；每次forward用本路速度和显式future poses重建Strong/CLEAR。训练可能比旧缓存背景略慢，不作FPS承诺。

A/B、CPU配对审计、C训练和汇总顺序执行。B和所有评估仅单卡；C默认双卡globalB4。中断后同目录恢复B完整窗口、审计完整窗口和C的model/Adam/rank RNG/cursor；不会重启单轮cosine或自动重跑A训练。

## GT框配对审计

同A/B/C的固定64，两个模态各自共享同四帧CC、poses和时间。先完成两种关联，再读取nuScenes GT框和instance_token，不将框/GT未来占据反馈给匹配器。沿用既有多数体素归属规则（purity0.5、secondary0.2），无阈值搜索或ICP。统计目标是最近`t−1↔t0`配对；较早历史链接变化不被冒充为最近配对改善。

分别报告全部source、最近配对发生变化的source：条件正确率、可归属coverage、可识别实例recall、正确配对的KTA速度误差及原GT速度分层。额外列changed old/new状态迁移、双方都可归属时wrong→correct和correct→wrong。未匹配/合并/低纯度不擅自算错或算对；缺帧不补假观测。GT框归属是条件诊断，不是完美体素实例真值，重复窗口不是独立样本。

## 服务器运行

同步代码后：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export SCENE_TRAIN_GPUS=0,1
export FORWARD_EVAL_GPU=0
export FORWARD_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_matched20_$(date +%Y%m%d_%H%M%S)"
export FORWARD_RESUME=0
# 有多个完成的AdaLN一轮实验时必须明确指定，不按日期或分数自动选：
# export FORWARD_CONTROL_DIR="/实际完成的fpn_adaln_zero_20pct_1ep_目录"
nohup bash tools/causal_forward_train/workflow.sh > "${FORWARD_DIR}.log" 2>&1 &
tail -f "${FORWARD_DIR}.log"
```

默认仅接受原一轮协议且complete的唯一`fpn_adaln_zero_20pct_1ep_*`目录；空失败目录/未完成/full模型不会被选。有多个则停止，要求FORWARD_CONTROL_DIR。不要把FPN加法目录填进去。C训练前严格核对A的报告、ordered TRAIN/DEV keys、实际更新数、runtime/data/oldimplementation/STC清单；原SHA不兼容则停止，不放宽。

最终总表：`cat "$FORWARD_DIR/summary.txt"`；配对诊断：`cat "$FORWARD_DIR/pair_audit/summary.txt"`；C训练进度：`tail "$FORWARD_DIR/C_train/progress.jsonl"`。中断后确认无旧进程，设回原FORWARD_DIR、相同控制及卡数并`FORWARD_RESUME=1`重新运行workflow。完成阶段只核对恢复，不重复计数；已存在但没有有效断点的失败训练目录不会被删除/静默重训。

没有服务器执行连接，本地只能完成功能/CPU/CUDA小网格测试，不声称已远程训练或产生新同64科学结果。原完整两轮命令仍保留，是否随后运行由这次结果和用户决定。

## 验证记录

安全yoloe非base环境，新训练/正确率/冻结关联/AdaLN pilot与full/FPN pilot与full/原框归属共56项测试通过（33.12s）。覆盖相同2,816,122参数及fresh state_dict/RNG、真实CPU/CUDA六帧新关联/Memory/反传、旧缓存背景剥离和记录不原地改、实际planner经JSON合同检查、私有evaluator完整窗整数恢复与标签后评分、实际CUDA原循环保存恢复Adam/RNG/新策略协议及最终不重复评估、GT交叉身份与缺帧unresolved/changed-only分母、A/B/C来源检查与负增量保留。B/C的CLI与Bash语法检查通过后交付，不运行新服务器科学实验。旧FPN/AdaLN/Stage2/原pair源码零diff。
