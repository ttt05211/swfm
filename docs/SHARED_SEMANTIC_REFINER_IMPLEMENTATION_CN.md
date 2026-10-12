# Shared-History：全网格语义残差Refiner首版（2026-10-12）

## 实现范围

独立目录 `tools/shared_history_experiments/refine/`。保留共享Encoder/Source-STWM/KTA+SE(2)/当前ego辅助头，整体替换修复头。旧Shared ADD-only、actions REMOVE、validation流程及其指纹文件均未修改；不替运行中的实验换模型。用户已授权推送及先小范围测速、正常后按原20%预算训1/2轮。无服务器执行连接，实际测速/训练需用户运行下述命令，尚无真实mIoU/FPS。

原型默认GT未来ego条件、4历史→6未来、TRAIN20%4086窗×2整周期余弦、原seed20261011、batch4/source128，固定最终DEV512。可在启动前固定EPOCHS=1；不能改变原周期静默续训。旧E14只供architecture/cache metadata，不导入权重；RANDOM新模型/新协议，拒绝旧CCR权重/Adam当resume。

## 模块与实际参数

新增SemanticRefiner默认46,483参数，不含原Encoder/STWM/ego：

- 共享F=[4,64,100,100]，1×1学习投影展开为[4,16,100,100,16]，保留Z。
- 同一预测source SE(2)运输对应历史source特征；静态/未归属历史证据按ego配准。源于四历史的注册和known，不用未来GT占据/实例/visibility。
- 动态source的未来Q通过128→16投影加入其对应空间证据，非全场景广播。静态/动态使用共享importance权重及归一化汇聚，多个来源不按owner挑winner；无证据使用可学习empty token，未知不当observed FREE。
- 当前M_h=[6,16,100,100,16]；与粗Transport18类概率的XY平均池化拼接，1×1投影到32维。
- 三个局部残差块：GroupNorm、XY depthwise conv（dilation1/2/3）、Z depthwise conv、32→64→32通道混合；时距FiLM条件化，六帧共用权重。
- 原网格全域分块查询：读取局部32维feature+原始18类粗概率+3维采样相位，MLP 53→64→32→18，chunk65536。不是source/road/halo可编辑域，不构造全分辨率高通道3D U-Net，也不加入剪枝或第二修正器。
- 18类残差末层权重和bias为零。有限prior logits=log((P_T+1e-4)/(sum_cP_T+18e-4))，final=softmax(prior_logits+delta)，argmax输出语义。初始argmax等于Transport；raw概率不同，训练后不承诺不损坏。

空间与高度维顺序统一[C,X,Y,Z]；source motion/world→future ego变换保持float64，插值和网络为FP32/BF16。静态memory inverse warp、动态memory可微forward splat在同粗格汇聚，point cell/关联离散索引不微分，插值权重及XY/yaw/pose可微。

## 不再使用的CCR机制

不构建SurfaceAtlas/KD-tree描述、canonical support materialization、旧27维ABI、ADD/REMOVE合法域、base/fallback头条件、手工静态冲突禁止编辑或动作阈值。旧缓存artifact可能包含这些历史字段，但新history consumer不读取其support/conflict/descriptor。

TransportPrior仍使用原t0静态语义连续ego warp、缓存GT-ego条件Strong背景fill及source-order overlay；测试与旧连续edits=False Transport一致。**没有声称整个系统移除了几何先验/动态分解/source顺序。**被移除的是修复阶段的owner/fallback权限链；所有类别都可由最终decoder重新判断。

## 监督与训练接口

完整六帧stable log-softmax CE + present semantic classes0..16的全网格Lovasz；FREE17参与CE，不参与semantic Lovasz类平均；invalid/unknown忽略。不采样体素/减少horizon/降低Lovasz频率，不新增动作BCE/一致性loss/蒸馏。

保持原source几何/existence与ego R10辅助及原AdamW分组/clip5/whole-cycle cosine floor0.01。训练为逐window完整H6前向及一次总loss backward，globalB4梯度累积；不是将旧RenderGradientBridge直接套进新增feature图（那样可能漏掉memory路径）。训练与评估都用同一个forward_window/残差读出。

零初始化使第一步上游Refiner梯度为零，这是普通零末层残差初始化性质，非永久断链。第一步末层更新后，memory/lift/共享F/运动链应获得梯度；已在真实小网格第二步检查。GT配置ego只作辅助，无新场景ego头；Pred配置尚未接入，不用GT pose进行Pred评估。

## 缓存与恢复

仅复用既有TRAIN/VAL纯历史几何require-mode；不覆盖旧artifact/namespace，不持久缓存learned feature或未来预测。新有界history RAM存source crops、静态known覆盖和动态历史点采样坐标；不做Surface拟合。

新协议 `p0_f9_shared_semantic_residual_gt_20pct_v1` 与实现指纹锁定模型、数据、预算、损失和新文件依赖。Ctrl-C/SIGTERM完整update/window后保存，last.pt包含model/Adam/RNG/cursor；轮末额外epoch_0001/0002.pt。evaluate-only写独立子目录，原训练checkpoint/summary不写。同输出严格resume，旧ADD-only/REMOVE断点不兼容，评估按完整窗口整数计数恢复。

## 服务器入口：先丢弃式测速，再训练两轮及逐轮DEV512

新增独立 `refine/experiment/` 控制器。2个预热update（第2个单独测occupancy→memory/shared F/XY梯度）+8个计时update（最多32窗），完整六帧训练前后向及Adam，所有临时权重更新丢弃，**不写任何.pt**。测速完成/finite loss和grad/非零梯度链接/实际CUDA/至少10%显存余量同时通过，才启动正式新RANDOM实验；没有“必须比旧模型更快”的自动承诺，也不按dev筛模型。

正式阶段完整两轮余弦不变：精确第1轮边界停并保存epoch_0001.pt→只读DEV512→同train/last.pt恢复→第2轮完成与DEV512。根summary/comparison汇总两轮Transport与新Refiner；不重复训练旧ADD-only，不继承旧数值/FPS。Ctrl-C转发到child，在完整update或评估窗口保存；同目录恢复复用完成阶段。测速中断后如显式恢复，重新做小测速（旧临时权重未保存），不删除旧诊断目录。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/refiner_probe_2ep_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/refine/experiment/run.sh
```

只测速、尚不训练可追加 `--probe-only`。恢复完整流程：

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/原refiner_probe_2ep目录
WORKFLOW_RESUME=1 bash tools/shared_history_experiments/refine/experiment/run.sh
```

中途只读DEV512：

```bash
TRAIN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际refiner_probe_2ep目录/train
RESUME="$TRAIN_DIR/last.pt" RUN_DIR="$TRAIN_DIR/eval_latest_$(date +%Y%m%d_%H%M%S)" \
  bash tools/shared_history_experiments/refine/run.sh --evaluate-only
```

最终报告Transport及semantic_refine、各1/2/3s IoU/mIoU/Moving和性能分段。普通refine/run.sh仍可独立训练（启动前EPOCHS=1或2），experiment/run.sh固定2轮及两次自动DEV512；两者不要混用输出目录。整个流程不按dev挑best。

## 本地验收与未验证项

新Refiner10项+流程3项+旧Shared13项+双动作8项共34passed，含RTX3050真实CUDA/BF16小网格：新prior与旧continuous Transport、零残差argmax、一次encoder、未来GT/旧support字段毒化、全域非候选类别重标、chunk数值一致、memory顺序汇聚、motion/pose梯度、unknown/empty source、第二步上游学习、精确CPU两步Adam/RNG恢复、整数评估恢复、真实CLI TRAIN4086契约/轮快照/旧协议拒绝及只读评估；另测真实临时CPU更新不写checkpoint、gate失败不启动正式训练、只测速后恢复、逐轮评估次序与完成阶段不重跑。

CLI测试使用模拟dataset身份和小网格，实际网络/renderer/CE/Lovasz/backward运行，但不是nuScenes质量。local chunk数值parity不等于L40S全网格六帧byte parity或旧H6严格gate通过。本地无完整服务器数据，未验证200×200×16真负载显存/吞吐/正式FPS，不能承诺旧0.354秒窗或54.945FPS。

首版全域修复扩大表达能力，但新学习的height展开/证据汇聚可能丢细节或改变可靠性；source历史配准失败、非刚体、unknown新生、STC域迁移和Pred ego误差依然存在。zero-init只是初始化保护，不是质量保证。所有旧结果保持基线，新架构需要独立同预算验证。
