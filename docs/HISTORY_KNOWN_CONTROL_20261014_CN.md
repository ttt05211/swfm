# 历史 known 输入约定：独立单变量对照

本轮只做 known，不实施学习先验权重、不叠加已失败的噪声增强、不训练 ego 头。
不是将旧模型的推理 mask 改成全 True 后报新成绩。

## 实际变化与保持不变

两路均相同 GTBody（SharedEncoder/STWM/18 类 SemanticRefiner）、相同 RANDOM 初始化、
seed20261011、TRAIN4086×1、B4/source128、H6 全六帧 CE+Lovasz、单轮完整余弦。
初始化 tensor digest 实测一致才发布成对结果。基础 E14 checkpoint 仅供结构和历史缓存身份，
不导入它的权重；不存在 ego/motion/existence 辅助 loss、旧 Surface 或 ADD/REMOVE。

- `lidar_known`：TRAIN/OCC 推理继续原历史 `mask_lidar`。
- `dense_valid`：TRAIN/OCC 的 Encoder 和历史 memory 都使用 `0 <= semantic < 18` 的输入有效性。
  完整发布网格为全 True；255 等真正无效语义仍 False，投影出界仍由原采样器处理。
- STC 在两路中都只读完整发布 semantics，用 dense 输入有效性，绝不读入 GT sensor mask。
- 所有最终评分保持完整网格，无 camera/lidar metric mask、无 GT pose 对齐。

这里 known 是“是否作为历史输入证据”，不是占据、正确性或感知置信度。
共享 Encoder 的 known 和 memory coverage 必须一起更新，不能只改一个通道。
source 连通域/关联/registration/KTA、历史语义、历史/未来 ego、Strong 底稿和 Transport 算式不变。
两路 learned motion 训练后会不同，因此不能承诺 Transport 最终分数不变。

复用原纯几何缓存 READ ONLY；新的 PolicyHistoryInputs 在预取 CPU 线程内按固定策略重建
known 相关 memory coverage，使用独立有界 RAM 缓存。没有学习特征 bank，也不混用旧 ego bank。
本入口只支持既定 nuScenes 完整四历史，不把缺帧/补帧推断为真实有效观测。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/known_20pct_1ep_$(date +%Y%m%d_%H%M%S)"
unset RESUME KNOWN_RESUME
bash tools/history_validity/run.sh && cat "$RUN_DIR/summary.txt"
```

用户最新要求：两路各1轮，直接完整运行，不做首检。不是只修改日志里的轮数：
总更新数、学习率周期、完成判定、最终checkpoint和对照表都改为单轮。
两轮版本的旧输出/8步断点保留，但不能接入单轮周期；必须使用上述新目录。
如果新的单轮实验中断，设置原单轮 RUN_DIR 后恢复：

```bash
export KNOWN_RESUME=1
bash tools/history_validity/run.sh
cat "$RUN_DIR/summary.txt"
```

中断后仍设原 RUN_DIR 和 KNOWN_RESUME=1；完整完成的路不会重训或重评。
失败不自动重试，旧断点、缓存、已完成 formal/long6/Waymo 实验不改。

最终只评同固定 DEV64 OCC+GT / STC+GT，各报告 Transport/Refiner 和 1/2/3s 指标，
根目录 summary/comparison 给 dense-valid 相对原策略差值，不自动选优、扩512/full或推广。
这轮刻意重训同预算控制：不把旧正式全量 epoch2 或旧500步结果冒充成对基线。
两路训练预算共8172窗口，时间以真实训练日志为准，质量评估耗时不是 FPS。

## 解释边界

既有冻结 DEV64 只改 known：Refiner mIoU−2.9506/IoU−5.6960；GT 头 FDE1.7234→2.3693，
STC 头2.4061。是输入分布影响的证据，不证明统一后一定恢复或解释所有 STC 掉分。
噪声增强短对照没有收益，不继续叠加。新实验可能提升 STC 同时降低 GT，两者都必须报告。
尚未验证实际 nuScenes 质量或 L40S 吞吐，不继承旧47.99 mIoU/61.52 IoU/任何正式FPS。

本地验证：新8项（含实际 CPU/CUDA 小网格前反传、known/coverage 同步更新、
初始 prior 对照、无效语义/未来标签隔离、整窗续评、两路8步工作流模拟/恢复、
真实 runner 的 Adam/RNG/轮快照/跨策略恢复拒绝）及既有噪声/GT/正式评估40项共48通过。
测试不是服务器真实数据训练，也不证明 STC 涨点。首次发现 Windows 路径分隔符导致策略
fingerprint 未绑定；已在新 runner 规范路径识别，跨策略恢复确实拒绝，旧代码没有更改。
最新单轮改动的8项专项回归10.67s通过：直接完整工作流、单轮步数/LR周期/快照、
两路同预算、旧两轮协议拒绝以及CPU/CUDA前反传；Bash语法检查通过。未代跑服务器。
