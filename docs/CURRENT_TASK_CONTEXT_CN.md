# 当前任务摘要（2026-10-09）

## 当前目标

Clean Joint Surface CCR 已完整随机联合训练20轮；固定5/6/8/12/14整网等权平均已完成DEV512选择、full4369质量和256×3正式FPS验证。
当前主候选mIoU44.153241 / MovingMicro32.214410；同256×3无损并行Strong多数投票正式FPS已到54.945，详见下文，旧路径保留默认。
最新任务：用户选择与I²-World对齐的Occ3D-Waymo 2Hz zero-shot，实现冻结同一均值模型的评估入口。只改数据适配/评估，不训练、改权重/结构/阈值或重启失败方案，不冒称已跑真实Waymo。
另一个已授权任务：冻结6秒推理的几何接续四路对照已实现，见下方新增章节；不改并行Waymo任务或原默认入口。

## 核心决策与约束

- 4 历史→6 未来。固定 weighted ADD raw sigmoid@0.5，REMOVE-off；没有 KD/AE。
- Surface 描述进入 CCR 编码，实时投影高度/相位进入静态条件读出；不是外挂后处理修正器。
- 正式模型应运动/静态/动态一起干净随机初始化训练；当前 20 轮正是这一版，不能混称早先冻结验证。
- 原 checkpoint、optimizer/RNG、缓存、实验目录保留。平均文件只用于评估，不能拿它当 resume。
- 同训练契约、同模型配置、同 TRAIN 正权重；只平均可学习参数，固定 buffers 不平均。
- 评估共享确定性历史/几何和 Moving 区域，不能共享候选之间 learned poses/features/输出。
- 固定 Dense Forecast FPS 边界：CausalHistoryState→fresh Strong/KTA+live motion+CCR+实时投影+六帧 dense。历史表示准备另报，不称 raw-input E2E。
- 新平均正式FPS已测40.250（256窗口/150场景×3）；不能沿用旧冻结Surface/Frozen B的FPS。
- DEV64/DEV512 和 VAL 扩大人口均已参与研究，不称独立测试；本次已授权固定平均的 full4369，不自动重训/部署。

## 当前进度与服务器锚点

完整 TRAIN20430×20，102893 updates，whole-cycle cosine floor0.1，无 tail。所有 `epoch_XXXX.pt` 保留在原及续训目录。
当前完成锚点（文件在 run 根目录，不是 model 子目录）：

```
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787
  last.pt
  training.json
  epoch_XXXX.pt
  summary.txt
```

其他轮次按相同 contract 在 `/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr` 下查找。
VAL 只读缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1`。
TRAIN 缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_v1`。
单卡 L40S，10 核、80 GiB；不再自动尝试多卡/改变 batch4/source128。

DEV64 关键 Joint 结果：

| epoch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| 6 | 53.476777 | 41.666093 | 33.921350 | 28.234393 |
| 8 | 53.531044 | 41.574074 | 33.866847 | 28.646939 |
| 12 | 53.574938 | 41.553765 | 34.735137 | 30.010121 |
| 20 | 53.499071 | 41.172214 | 33.323547 | 27.574565 |

第 6 轮最高 mIoU；第 12 轮最高 IoU/Macro/Micro。不应只按最后一轮判断效果。
按 mIoU top5 的平均来源固定为 5/6/8/12/14，跨较大训练间隔，不保证平均更好。

原第 20 轮 DEV512：

| branch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| static_repair | 52.541410 | 39.619555 | 23.362541 | 28.837789 |
| dynamic_repair | 50.628250 | 39.324149 | 24.779588 | 30.681102 |
| joint | 52.543060 | 40.043394 | 24.779588 | 30.681102 |

## 最新已完成 DEV512 对照与冻结选择

| candidate | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| epoch6 | 52.451547 | 40.248853 | 24.912767 | 30.599135 |
| epoch8 | 52.488365 | 40.300115 | 25.243657 | 30.535807 |
| epoch12 | 52.506929 | 40.199564 | 25.281199 | 30.875808 |
| mean5/6/8/12/14 | 52.372722 | 40.432276 | 25.718056 | 31.486519 |

平均的三个时距 mIoU/Macro/Micro均最高，但IoU略低，不称四项全面提升。
收益主要体现于Transport；平均Transport mIoU39.362646 / Micro30.014088，CCR再增+1.069630 / +1.472431 pp。
四候选512窗口合计705.52s，这是质量eval耗时，不是正式FPS。
新平均 full4369 和正式FPS均已完成。

2026-10-09 服务器全集报告：固定平均（5/6/8/12/14）在 full4369、150 场景上
IoU55.211054 / mIoU44.153241 / MovingMacro28.256901 / MovingMicro32.214410。
同模型 Transport：53.631093 / 43.218661 / 26.889216 / 30.729273；CCR 增益分别
+1.579960 / +0.934580 / +1.367684 / +1.485137 pp。
相对旧 Local full4369：IoU-0.347217 / mIoU+0.124314 / Macro+0.968314 / Micro+0.974187 pp。
耗时1649.02s是带GT/整数指标的质量评估，不是FPS；仍不称独立测试。
结果：`outputs/p0_f9_joint_surface_ccr/mean_full4369_20261009_104036_848/full_validation.json`（服务器）。

正式FPS（同均值，256窗口/150场景×3，actualCUDA）：fused_eager152.852ms/39.254FPS；fused_graph149.067ms/40.250FPS，P90=186.348ms，256/256六帧和概率byte parity。
仅2个图捕获，计时外；无拒绝。Strong82.109ms是主瓶颈；读出24.515ms已包含实时phase14.053ms，不能重复相加。峰值allocated253.885MiB。
历史准备149.995ms、surface descriptor104.183ms单独排除，不称raw-input端到端FPS。
服务器结果：`outputs/p0_f9_joint_surface_ccr/mean_fps_20261009_113139_838/speed.json`。

## 本轮实现与验收

独立分支 `feature/v22-surface-aware-ccr`，工作副本 `swfm-surface-ccr`。原 `swfm-v20utc` 的用户未提交改动不动。
用户已授权该新分支提交/推送；不能推断新服务器训练/部署授权。

新增选择/平均模块、共享历史批量评估模块、CLI 与 Bash wrapper。
跨续训目录严格校验 lineage、完成 epoch、报告、源 SHA、原 TRAIN 权重，缺文件/重复冲突停止。
新平均权重标记 evaluation_only，不含原 optimizer/RNG。
四候选共用一次窗口准备，分别实时运行模型；每 8 个完整四模型窗口保存整数状态，严格 --resume。
既有 trainer/model/cache 依赖文件未改，避免破坏原训练 resume 或大缓存 namespace。
120 项相关本地回归通过，含真实 CPU/CUDA 单路径 vs 批量计数一致、中断恢复、平均/源文件只读与空类别指标，以及原联合恢复/表面读出回归。
增加只读缓存 namespace 预检和明确四窗口 CPU 预取后，10 项专项复查也通过；新文件 AST 和 Bash 语法检查通过。未声称全仓 CI 或真实 nuScenes 质量通过，本地没有完整服务器数据。

操作与风险见 `docs/JOINT_SURFACE_CHECKPOINT_AVERAGING_CN.md`。
服务器运行 `bash tools/real_motion/run_p0_f9_joint_surface_checkpoint_comparison.sh`；结果在新 comparison 目录，不覆盖训练。
中断后指定原 comparison 输出目录和 `SURFACE_COMPARE_RESUME=1`，不是恢复训练。
本次增加 `eval_p0_f9_joint_surface_mean_full.py` 与 wrapper，复用同一评估引擎，仅加载已完成对照中的平均文件。
只读已选来源，完整VAL4369不做DEV512筛选、不导入epoch20子集报告；新结果写full_validation.json，严格同目录整数resume。
操作见 `docs/JOINT_SURFACE_MEAN_FULL_VALIDATION_CN.md`。121 项相关本地回归通过，包含完整人口、来源冻结/只读、真实共享CPU/CUDA路径和CLI续评等价；AST/Bash语法通过。模型/trainer/缓存依赖未改。服务器未运行，不冒充全集结果或完整CI。

新增平均权重正式FPS入口；复用原 `paired_speed`（旧20×3默认兼容），不改训练/model/cache依赖。
19项相关本地回归通过，含真实CPU/CUDA读出及六帧合成一致、均值FPS算法、权重只读、缺帧拒绝、计时内建图拒绝、旧入口回归；CLI help/Bash语法通过。
这不是服务器实测FPS或全仓CI。服务器命令：`bash tools/real_motion/run_p0_f9_joint_surface_mean_fps.sh`；输出新 `mean_fps_*` 目录。
可用 `SURFACE_MEAN_FPS_WINDOWS=512` 扩大人口；中断保留partial日志，不改断点，重测使用新输出，不混拼前次计时。

本轮简单无损候选：`real_motion/strong_warp_execution.py`。保留原FP32逐时距GEMM、5e-3边界判定及FP64 NumPy纠错；合并边界索引与标签/known回传，去掉每时距any/变长gather同步。只保留本次最多6网格的临时buffer，不缓存预测。
默认仍reference；ContextVar显式scope可选buffered，异常自动恢复、不会改其他线程。7个历史几何namespace文件、10个训练指纹文件及native ABI都未改；旧权重/缓存继续兼容。
RTX3050上只读真实replay.zip：6窗口×5，六帧逆变换32.848→24.410ms（1.346×），与旧CUDA及CPU FP64参考逐字节一致。这不是整网FPS或L40S预测。
33项相关本地CPU/CUDA回归通过，含越界/边界、跨6网格批次、Strong anchor/components/CLEAR、旧FPS和graph读出；Bash语法通过。
同一正式入口设`SURFACE_MEAN_FPS_COMPARE_STRONG=1`，成对测原fused_graph与buffered Strong/fused_graph，默认仍256×3同seed/同人口，不自动采用候选。

## 新增：冻结平均 Surface CCR 的6秒递推（2026-10-09）

用户已授权实现新版6s评估；旧Local epoch19入口原样保留。不重训，不改变权重/阈值。
新入口 `run_p0_f9_joint_surface_long6s.sh`（默认dev64）及 `run_p0_f9_joint_surface_geniedrive_long6s.sh`（官方公开代码2569人口）。
只读同一已完成DEV512对照的5/6/8/12/14平均文件；严格4→6→最后4预测→6，报告1–6秒。
固定接续reconciled为主结果、redetect同人口对照，共享第一段；无GT身份接续、不按分数挑路线。
每段实时Surface Atlas/phase/source读出，ADD raw0.5/REMOVEoff；第二段禁止真实未来缓存/adapter读取，保留handed velocity。
第一段已有VAL cache只读核验；官方早期300窗口实时四历史重建，不丢样。省掉旧Local memory/frontier计算；native融合投影、4窗口预取、8192完整块静态CUDA graph有界复用。
未来标签/Moving support仅在所有路线预测完成后读取，Moving保留原始t0；第一段四历史/六张Transport/六张Surface逐字节gate，第二段完整概率/六帧gate。
同目录整数断点恢复：每8个完整窗口周期保存，SIGINT/SIGTERM窗口边界保存；模型/人口/实现/执行开关变更拒绝resume。
159项相关本地回归通过、28项跳过（CUDA相关未在本次CPU环境复现），包含实际Surface网络、native CPU/NumPy两段一致、早期起点、预测历史cache隔离、完整CLI snapshot/中断恢复。Bash语法和CLI help通过；不冒充全仓CI或服务器6s精度。
操作见 `docs/JOINT_SURFACE_MEAN_LONG6S_CN.md`。服务器下一步跑新dev64入口，发回summary；不用重建prototype/cache、重新平均或恢复训练。

## 6秒服务器验收与下载修复（2026-10-09）

Surface平均dev64真实6秒已完成，first/second exactness通过。接续主路径4–6s平均mIoU23.305861/Micro8.004605，
redetect对照23.148164/7.179412，接续增益+0.157697/+0.825193 pp；64窗口103.21s，不称FPS。
601/630当前source成功接续；长期预测仍衰减，不与旧dev512不同人口直接比较。
结果服务器：`outputs/p0_f9_joint_surface_ccr/long6s_mean_dev64_20261009_170347_837`。

GenieDrive全集入口下载官方metadata时报网络Errno101，尚未进入评估、未创建评估状态。
修复下载器：默认官网→网络失败尝试HF-Mirror；可显式`GENIEDRIVE_DOWNLOAD_ENDPOINT`或`HF_ENDPOINT`，
仍固定revision/size/SHA，校验错误不重试、不发布/反序列化、不覆盖原文件。标准HTTPS_PROXY兼容，
所有地址不可达时明确离线上传指引，无长traceback；只移除本次自己创建的.part。
76项下载/人口/Surface两段/接续相关回归通过，4项CUDA跳过；服务器镜像连通性未验证。
下一步更新后显式镜像重跑全集，`SURFACE_LONG_REDETECT=0`可省对照；下载阶段失败不使用resume。
完全离线时上传同一官方固定metadata到`data/geniedrive/world-nuscenes_infos_val.pkl`，再离线校验。
无模型/权重/阈值/训练/cache namespace变更；约2569窗口60–80分钟仅是dev64外推估算。

## 未解决问题与下一步

1. buffered Strong已提交并实测：L40S151.236→149.391ms，整网仅1.0124×；逆变换12.061→10.407ms，多数投票66.047→65.949ms。该路线收益小，不继续单独抠warp。
2. 用户授权试多数投票：新增显式4线程分块+紧凑临界坐标+仅在回退邻域取类别，仍原native整数投票及SciPy float32决胜。原路径保留默认，训练/缓存指纹及native ABI未改。
3. 本地真实6窗口×7六帧多数投票59.873→19.949ms（3.001×），完整SciPy/native/新路径byte parity；仅内部CPU阶段，不声称整网FPS、L40S或训练提速。详情`docs/STRONG_MAJORITY_EXECUTION_CN.md`。
   93项相关CPU/CUDA回归通过（含旧FPS及最新6秒递推），CLI/Bash/diff检查通过；不冒充全仓CI或服务器结果。
4. 服务器同256×3成对验收已完成：原155.390ms/38.613FPS，新109.200ms/54.945FPS、P90=145.082ms，整网1.423×；概率及六帧dense byte parity256/256。多数投票67.410→21.910ms，Strong84.759→39.042ms。显存allocated253.885MiB不变，不改权重/阈值，不自动修改旧默认或重训。
5. 全集已追回mIoU/Moving，IoU仍略低；不增加重模块，不改结构/阈值。

## 新增：I²-World Waymo 2Hz zero-shot（2026-10-09）

用户选择按I²-World对齐并授权实现。新入口 `run_p0_f9_joint_surface_waymo.sh`，文档 `docs/WAYMO_I2WORLD_ZERO_SHOT_CN.md`。
官方源码固定 `II-World@661d830f9b34ee03ce368db164a72753ab8764a3`；全局timestamp排序后stride5、202validation场景，未来index1/3/5报告名义1/2/3s，场景边界重复有效历史/未来，不静默丢样。实际时间/补帧数量另报。
固定同一5/6/8/12/14均值，四总历史（含t0），ADD raw0.5/REMOVEoff；用原Surface预测路径及无损native/majority执行，逐进程重做输入/Transport/概率/六帧exactness。
只读历史voxel_label及ego poses；未来pose是显式条件，未来occupancy在六帧预测完成后才读取。无未来annotations/mask，历史dense visibility全真与官方输入一致，不加额外mask信息。
复现官方Waymo→nuScenes18类映射；默认raw free23，已确认文档发布free15时显式切换并记报告，未知/混合编码拒绝。不能再映射模型预测。
同时报告官方零IoU排除的mIoU和union有效类保留零分的标准mIoU；binary occupied IoU，无camera/lidar mask，per-horizon累计后求均值，官方逐时距round2均值另给。不硬搬nuScenes Moving指标，不把eval时间称FPS。
默认256MiB不可变原始帧LRU；不使用nuScenes几何cache、不存learned features/预测。每8完整窗口整数checkpoint，Ctrl-C/SIGTERM边界保存；严格同目录、模型/metadata/人口/NPZstat/实现/执行契约resume。
数据仅需validation0.4m NPZ、可信官方 `waymo_infos_val.pkl`、`cam_infos_vali.pkl`；下载链接、目录和audit/full/resume命令在上述文档，不自动下载大数据。
本地新协议与旧Surface/四历史递推相关回归运行，真实小网格CPU模型预测覆盖早期/中间/末端；未跑真实Waymo或本次CUDA服务器，不宣称zero-shot精度/速度或全仓CI。
下一步服务器准备这三项数据，先WAYMO_AUDIT_ONLY=1检查编码/人口/文件，再运行完整zero-shot；发回summary.txt。只实现2Hz，10Hz源码名义时距歧义未静默复制。

## 新增：冻结6秒几何接续四路对照（2026-10-09）

用户已授权尝试提高6秒精度；保留并行 Waymo 工作和旧6秒入口，单独新增 `run_p0_f9_surface_geometry_carry.sh`。
真实 GenieDrive 公开代码人口2569/150已跑完：4/5/6s mIoU30.572013/27.095679/24.488694、平均27.385462；IoU42.032229/38.331931/35.324381、平均38.562847；2330.31s，不是FPS。
既有接续匹配31783/33451来源，不能再把身份匹配率低当作主要原因；论文Table2实际人口仍未独立确认。

新实验一趟 TRAIN64+dev64 跑 baseline/static_carry/se2_carry/combined，冻结同均值、阈值和原第二段 motion forward。
static_carry 原始t0静态直接投影到第二段，保留首段CCR静态新增；动态foreground/重叠动态fallback不变。
se2_carry 只将过去预测形状按首段预测中心/yaw对齐到3s，可靠唯一身份才替换ICP；split/merge/错语义/4m质心异常回退。
这是改变第二段输出的推理算法候选，不是无损后端；相对baseline可能删除/重标静态背景，单独报告corrected/damaged/removed。
四张初始真实历史之外不新增观测；保留causal几何状态必须公开说明stateful rollout；未来GT ego poses仍显式条件。

仅TRAIN64固定规则选择：avg4–6 mIoU+0.05pp、avgIoU/Micro非负、4/5/6mIoU分别非负；DEV64全报告不选路线。
all需要同权重/代码、内容指纹完整的screen recipe，仅冻结候选+baseline官方2569人口，默认不自动启动。
第一段共享且不变；未来GT occupancy/Moving仅全部候选预测之后读取。每8完整窗口整数状态，Ctrl-C/SIGTERM边界保存，严格同目录resume。
native fused projection/graph+4线程无损多数投票；现有VAL几何只读，不写大缓存，原权重/训练/cache namespace不改。
138项相关本地回归通过、20项CUDA跳过，包含真实CPU Surface四路参考字节一致、动态保护/歧义回退、CLI snapshot只读/中断恢复、全集入口和旧long/projection/graph/FPS回归。
CLI help、Bash语法通过；真实服务器候选精度及本次CUDA路径尚未跑，不宣称提分/正式FPS/全仓CI。
操作与比较边界见 `docs/SURFACE_LONG_GEOMETRY_CARRY_CN.md`。下一步服务器跑screen并发summary，无需重建prototype/重训。

服务器首次screen在TRAIN首窗口 four-history input4(KTA)差1.9967556e-6退出，未完成窗口。
本次补旧缓存升级算术兼容：仅KTA严格等于 `float64(FP32 anchor)-float64(FP32 normalized xy)*40` 后转FP32时接受。
诊断副本使用实时KTA，实际prep/record输入、权重不改；其他输入原严检、六帧Transport及Surface byte gate继续。
本地真实Surface大坐标fixture复现微米级缓存舍入；误改KTA/类别/mask拒绝。扩大相关回归142项通过、20项CUDA跳过。
代码指纹改变，原失败目录保留，新输出重跑，不使用旧目录resume，不重建缓存或训练。

## 仅按需检索的历史

冻结 Surface CCR 完整 TRAIN×3 DEV512 mIoU40.418031/Micro31.052754；扩大 VAL4369 mIoU43.991199/Micro32.043717。
Frozen B 与旧 Local 的静态差距集中 road11/sidewalk13；Surface 内部几何条件读出已追回大部分冻结静态差距。
这些是早先冻结实验，不是新联合精度。旧阈值、halo、共享场、点 CCR 支线不重启。
详细历史仍保留在 `docs/CLEAN_JOINT_SURFACE_CCR_CN.md`、`docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`，需要时检索，不复制旧日志。
