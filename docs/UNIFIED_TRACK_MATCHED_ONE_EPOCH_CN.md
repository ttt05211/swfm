# Unified track / multi-frame KTA：同预算随机重训一轮

用户2026-10-15授权：冻结screen中KTA均值下降但OCC Moving退化，做小规模同预算重训，不直接扩大完整训练。本任务继续独立分支 `experiment/unified-source-track-kta-dev64`；只新增 `tools/unified_source_training/`、专项测试和本文档，不改旧冻结screen、正式模型、训练器或缓存源码，不影响其他分支。

## 固定预算

- 两路独立随机同初始化：baseline、unified_fit，不从GTepoch2/DEV最优checkpoint微调。
- 同TRAIN20%=4086窗口，各一轮；seed20261011、同窗口顺序及batch4/source128分组、相同完整一轮余弦。
- 原SharedEncoder / STWM / SemanticRefiner；只完整六帧CE+Lovasz，不加运动、existence或ego辅助监督，无ego头。
- GT未来ego显式条件。源组件提取不变，source数与预算不变；新track和KTA同时在TRAIN、OCC评估、STC评估使用，不再拿旧残差直接套新KTA。
- 两路均沿用原两帧Strong/CLEAR背景，隔离track/KTA；不是整个Strong的正式新统一版。
- 训练前检查固定DEV64的STC四历史文件；两路最终同64窗OCC+GT/STC+GT，Transport和Refiner均报告1/2/3s IoU/mIoU/Moving。STC保持发布完整语义/all-known，全网格无metric mask，不做GT对齐、门限/最优epoch选择。
- KTA沿用已冻结screen方案：既有关联保护/ICP、原始世界质心按实际历史时间拟合斜率，当前source中心不变。无新匹配门限、shape代价或阈值。

## 缓存和执行

baseline沿用旧历史几何。unified_fit可复用旧t0 source提取和点云、原始语义/known/poses；previous/更老帧组件、统一关联、配准、source patch/memory读取和KTA现场CPU重建，绑定有界process-local RAM cache。旧KTA/anchors在新row中同步替换，旧磁盘记录与监督文件不写；旧source residual标签不重写，且CE+Lovasz-only不消费它。Strong背景不变且同GT ego条件，继续复用原缓存；STC重新用真实STC历史建立，不混GT历史几何。

沿用现有4worker/depth2有界CPU预取，几何build在预取worker执行；不存学习特征bank、不重建全量磁盘cache、不调整batch或future数。旧record/geometry只读，新增派生track不写旧namespace。

两个子进程依次训练/最终评估，没有权重共享或互相学习。last.pt保存Adam/RNG/余弦/完整更新游标，固定epoch_0001另存；根workflow/comparison记录相同初始化hash、同数据契约（仅implementation含arm差异），Ctrl-C完整update或eval窗恢复，完成路不重训。`--max-updates 8`可让两路各保留8步，随后同根resume剩余同一轮，不重置余弦。默认不跑这个smoke，直接各一轮。

评估明确：原数据预取器可能已把未来目标数组读入CPU，预测器只接收四历史字段；未来GT语义/annotations/Moving不参与预测，指标阶段才消费。评估用确定性BF16受控执行，质量wall time不等于正式FPS。不自动完整TRAIN扩训或推广。

## 本地验证

45项新训练/旧冻结screen/原GT与正式评估回归通过，21.99s。包含真实CPU/CUDA完整CE+Lovasz反传、新anchor/source读取一致、旧输入和背景不变、未来字段毒化、实际CLI部分更新到完整一轮的Adam/RNG/游标恢复、两路初始化/预算核验及已完成路跳过、真实小图64窗OCC/STC评估和整数续评。CLI help、Bash语法通过。小图不是nuScenes质量或L40S速度结论，没有代启动服务器训练。

## 服务器入口

在原独立worktree更新本分支后运行；代理只用ghfast.top，不改原开发目录分支。分支更新发布后使用：

```bash
conda activate OccFM
cd /root/nas/occ/swfm-unified-source-tracks
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git experiment/unified-source-track-kta-dev64
git merge --ff-only FETCH_HEAD
unset RESUME TRACK_TRAIN_RESUME
export TRACK_TRAIN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/track_matched_20pct_1ep_$(date +%Y%m%d_%H%M%S)
bash tools/unified_source_training/run.sh
cat "$TRACK_TRAIN_DIR/summary.txt"
```

恢复：进入同一独立代码worktree，设置原实际TRACK_TRAIN_DIR，再 `export TRACK_TRAIN_RESUME=1; bash tools/unified_source_training/run.sh`。不是从原full_gt/冻结DEV64目录resume；旧训练完全保留。脚本默认数据/架构参考/旧cache路径都指向 `/root/nas/occ/swfm`，不复制数据。

E14文件仅提供原架构与只读cache契约，绝不把E14权重导入任一路；本次随机重训不读取先前GTepoch2或平均网络权重。
