# 冻结 WM 的 ego 头：训练预算与几何监督一次对照

## 为什么做

用户服务器的原 TRAIN1024 / batch64 / 20轮头只完成320次更新。固定dev64上，原外部planner vs 内部头平均mIoU：
OCC 18.350472 vs 16.760100；STC 11.960050 vs 10.928355。内部头未采用，原 GT / Camera / Pred 主表不改。
OCC内部头1/2/3s平均XY误差0.565/1.668/2.875m，外部为0.339/0.611/1.341m。
单个末batch的XY损失约0.43不足以证明收敛；yaw原周期损失只有0.003–0.008，存在监督尺度风险。

此次仅回答：预算/余弦时长是否不足、几何监督是否更合适，以及小头有无超越历史外推的可学收益。
不承诺加步数或改损失必然改进，不重训WM/CCR、不扩大TRAIN人口。

## 固定设计

| 路线 | 训练 | 作用 |
|---|---|---|
| prior | 不训练；原零残差头的历史匀速/匀角速度外推 | 因果运动初值诊断，不是新planner |
| old320 | 已有320步权重，只读 | 原屏幕结果复测 |
| A_original | 原结构、原XY Smooth-L1 + yaw周期损失，2000步 | 预算＋余弦时长对照 |
| B_geometry | 完全同结构，单个SE(2)参考点几何损失，2000步 | 与A隔离监督目标 |

A/B用原seed、原冻结网络构造顺序复原初始头，并核验原断点的CPU RNG见证；复原失败就拒绝“预算唯一变量”的说法并停止。
两头从头初始化，不拿失败old320做最低LR续训；每一步同一batch、同一样本顺序、AdamW、clip5、lr3e-4→3e-6全周期余弦。
同原TRAIN1024 bank、batch64，2000步是125轮，不是2000个新窗口；不按dev追加轮数。

B固定五个参考点 `(0,0),(±10,0),(0,±10)` 米，分别经预测和GT SE(2)变换，所有六时距全部点的XY坐标做一次Smooth-L1。
只有一项几何损失，不额外叠加原XY/yaw损失；位置误差和转角引起的空间偏移都按米计量。
不用GT形状、未来occupancy或mask，不穿过硬体素renderer反传。它不增加未来信息，不能保证解决XY误差。

整轮损失按所有实际样本加权平均，含最后不足64的batch（本人口无不足batch）；每200步及最终步做整个TRAIN1024轨迹误差统计。
报告prior/old320/A/B的六时距XY/yaw mean/median/P90、ADE/FDE及每时距right/left/straight计数。
TRAIN诊断是同训练人口IN-SAMPLE，不称held-out验证；不能拿它代替dev结果。

## 旧特征与断点保护

原实验：`outputs/p0_f9_joint_surface_ccr/ego_head_screen_20261010_194309_837`。
不重新提取特征、不复制bank，不修改原头、WM/CCR、training.json或特征文件。
逐shard核验原contract/key/内容指纹及SHA，并核验原权重与冻结mean SHA。
原头的代码指纹囊括所有Python文件；复原旧manifest时仅排除本次新增的三份ablation Python文件，所有原特征/几何代码必须严格匹配。
新增实验记录当前完整实现指纹；绝不是直接忽略fingerprint不一致。其他旧代码更新会fail-closed。

A/B每32个完整成对更新原子保存到新目录 `pair_last.pt`，保存两优化器、order/cursor、整轮累积量、monitor、RNG和契约。
SIGINT/SIGTERM在完成A和B同一步后停止；SIGKILL只恢复最后周期断点。完成训练的resume不改写pair权重SHA，允许继续同一dev评估。
改变预算/损失/源码/人口/模型配置禁止静默resume。原目录只读SHA前后核验，输出互斥lease。

## 一次固定dev64评估

OCC/STC各自评 `GT / external / prior / old320 / A / B`，一趟共12路。
共享每种输入的历史特征与冻结WM history-only准备；各候选的Strong、未来投影/phase/合成live重算。
仅当同一历史下完整六未来pose逐字节相同时复用同窗输出，不跨历史/模型缓存预测。
导航仍是显式GT派生destination-row条件，不称无导航预测；无aligned、未来GT可见性、阈值选优，GT只作为GT路线条件或事后评分。
未来语义必须等所有路线预测结束才读取；整窗所有路线整数/轨迹计数一起保存，失败半窗不提交。
标准无mask IoU/mIoU；轨迹误差与TRAIN差距用于分析，默认不自动选择或推广A/B，也不自动扩到全集。

## 服务器完整命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_surface_ego_ablation.sh
```

默认使用上面的已完成原实验，依次训练A/B并评固定dev64。`EGO_AB_SOURCE`可指向同协议的原目录；不以最近mtime猜来源。
新训练输出打印为`ego_ablation_*`，评估目录为同名`_eval_dev64`；后者`summary.txt`包含TRAIN与全部dev结果，只需发这一个摘要。
评估子进程在导入预测模块前恢复原bank的SWFM执行环境（不修改父shell）；无shell eval。

```bash
# Ctrl+C后恢复本次对照，不恢复旧WM或old320训练：
export EGO_AB_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_ablation_实际打印目录
EGO_AB_RESUME=1 bash tools/real_motion/run_p0_f9_surface_ego_ablation.sh
```

同一命令适用于训练或评估中断；已有完整pair不重训，dev整数前缀从同一目录续评。仅训练可设`EGO_AB_SKIP_EVAL=1`。
不自动启动服务器任务；本地测试不等于实际L40S精度/FPS。12路质量评估的wall time不称单模型FPS。

## 本地验收

专项及相关回归156 passed / 9 skipped，两个CLI帮助及Bash语法检查通过。
覆盖成对Adam/order/RNG/部分整轮均值的续训一致性、完成后resume不改pair SHA、源bank与旧实现指纹、
12路历史共享/未来几何live重算、未来标签读取顺序、整窗原子续评，以及实际小网格冻结WM/CCR的CPU路径。
CUDA相关测试因本地环境跳过；尚未运行服务器TRAIN1024/dev64，不宣称新头改善或实际GPU提速。
