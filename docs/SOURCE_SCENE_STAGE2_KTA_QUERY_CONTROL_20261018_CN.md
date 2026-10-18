# 第二阶段只恢复 KTA→Query：同预算控制

## 为什么做

同remaining64冻结诊断已显示：旧 SharedTransport 的未来XY/yaw相对KTA贡献OCC Transport +2.7914pp、STC +.9905pp；Stage2只有+.0270/+.0110pp，未来修正约1.6/3.0cm。旧版主要改善条件归属的低速/静止交通参与者位移误差，但运动组误差/MovingMicro不改善，不能把它包装成全面更好的运动预测。

先验证去掉 KTA Query 条件是否让“相对KTA的残差”难以学习；不是直接恢复ICP、增加监督、加训练预算或宣布新版失败。

## 唯一架构改动

沿用Stage2 Source Scene Reader、masked temporal、learned history yaw（t0 gauge=0）、track_valid、无ICP直接历史搬运、原两帧KTA skip、同一SharedEncoder/Refiner。只将未来Query从

`q = future_query`

改为原Stage1形式

`q = future_query + Linear(2→128) / SiLU / LayerNorm (KTA_xy / 20)`。

KTA来自原匹配的历史两帧，不是GT未来位移。仅增加640参数（一般为5×d_model），不是等参数消融；不恢复类别/速度/patch数值描述子，不新增匹配、配准、阈值或loss。Q也会进入memory，因此需区分Query直接帮助Refiner和未来运动本身改善：最终同时报告learned与固定Q的KTA-only干预。

新增独立包 `tools/source_scene_kta_query`，模型forward在本包显式实现而不修改旧源或临时替换模块。投影归零时用测试验证整个CPU前向等于原Stage2；历史context/yaw与KTA条件无直接依赖，变化发生在future decoder之前。新checkpoint有独立协议/loader，旧Stage2/full不能静默载入新模型。

## 初始化与同预算

- RANDOM TRAIN4086×1，同seed20261011、选窗/排列、global B4-source128、H6完整CE+Lovász、AdamW分组/LR、一次整轮cosine和dense-valid三输入协议。
- 复用完成的**Stage2 TRAIN20%×1轮**作为no-query对照；不是Stage1 Reader，也不是完整TRAIN×2轮。旧组不重训、不载入其训练权重/Adam。
- 先生成与原Stage2相同的新鲜随机公共权重，再在fork_rng内构造带投影模型，只拷贝新鲜公共state；多出来的投影随机初始化。公共state逐张量一致，额外构造不改变原初始化后的CPU RNG；实际训练仍使用原rank reseed/恢复。
- 仍双卡global分组训练；最终rank0单卡评估同remaining64，rank1等待。world_size也须与旧对照一致。
- 旧cache仅require只读，新identity_history/rank*在新输出根；新学习特征不持久缓存。
- 对照的训练键/评估键/数据hash/顺序/预算/LR/损失/known/precision/world_size/公共架构及旧源码hash必须匹配；不能用异预算结果冒充单变量。

## 最终评估

每模态每窗只编码一次，先完成六帧learned和KTA_future输出（只关闭未来XY/yaw，保留新Q/历史yaw、同步改prior与memory），再统计GT未来占据/moving/框运动。标准最终比较只使用learned的Transport/Refiner，对照表另列本模型learned−KTA。

`source_motion`记录XY/yaw修正量、框条件归属的moving与slow_or_stationary位移误差/改善损害；归属未覆盖或歧义保留，不将它当完美点运动GT。low-speed组是source交通参与者，不是道路/建筑。继续监控MovingMicro，不能只追mIoU而忽略旧版“改善低速却损害运动”的取舍。

固定epoch1，不按DEV挑路/挑epoch、不自动扩训或推广；既有DEV反复查看，非独立测试。没有新FPS结论。

## 运行

```bash
export SCENE_TRAIN_GPUS=0,1
export KTA_QUERY_RESUME=0
export KTA_QUERY_DIR="$PWD/outputs/p0_f9_shared_history/scene_stage2_kta_query_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_kta_query/run.sh > "${KTA_QUERY_DIR}.log" 2>&1 &
tail -F "${KTA_QUERY_DIR}.log"
```

自动定位唯一完成的 `scene_stage2_1ep_*`；多份请显式设置 `SCENE_NO_QUERY_DIR`，不按分数或mtime选。老实验采用自定义目录名时也设置该变量。

`last.pt`保存model/Adam/rank RNG/完整update cursor和评估；完成保存`epoch_0001.pt`，自动同64评估。Ctrl-C/TERM在完整update后保存，同目录设置`KTA_QUERY_RESUME=1`续训/续评。不接受旧Stage2断点续入。

完成后 `cat "$KTA_QUERY_DIR/comparison.txt"`。`results.json`保留完整指标与运动统计；`progress.jsonl`为本次实际global训练秒/窗。原no-query结果、训练入口、源码指纹和已有checkpoint均不改，失败可以直接停止新入口。
