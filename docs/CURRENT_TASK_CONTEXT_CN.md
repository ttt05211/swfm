# 当前任务摘要（2026-10-10）

## 当前目标

Clean Joint Surface CCR 已完整随机联合训练20轮；固定5/6/8/12/14整网等权平均已完成DEV512选择、full4369质量和256×3正式FPS验证。
当前主候选mIoU44.153241 / MovingMicro32.214410；同256×3无损并行Strong多数投票正式FPS已到54.945，详见下文，旧路径保留默认。
最新已完成：用户服务器跑完static_carry全集，明确确认作为最终论文6秒长时预测路线；avg4–6 IoU41.218153 / mIoU29.117864。
记录在 `docs/SURFACE_STATIC_CARRY_LONG6S_FINAL_RESULTS_20261009_CN.md` 与 `docs/results/` 原始摘要/结构化转录。
Waymo两套zero-shot已全部完成并归档：2Hz 7998/7998，joint平均IoU61.209875 / mIoU52.152906；10Hz 39987/39987，joint平均IoU75.100701 / mIoU65.997882。2Hz相对同模型Transport为+0.957959 / +0.418007 pp；10Hz为+0.337958 / −0.113556 pp，不能称两协议全面提升。10Hz按原native+2/+4/+6，即0.2/0.4/0.6s，不是物理1/2/3s；不与2Hz绝对分数直接比较。档案`docs/WAYMO_ZERO_SHOT_FINAL_RESULTS_20261010_CN.md`及`docs/results/surface_ccr_waymo_2hz_10hz_20261010.json`；无需再resume或重跑Waymo。

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
数据已下载并解压；服务器只读metadata统计39987原始帧→7998锚点，7796同场景链接全部stride5，15个timestamp跳变（最大1.199943s）。原逐对0.35–0.65s硬检查误拒绝官方数据；现保留官方人口/实际pose/timestamp，将跳变记入`timestamp_gap_audit`，仍拒绝非正时间/倒退帧及总体错频率/错单位。不重采样、不丢窗、不改模型时间步。
时间检查修复的Waymo/Surface/FPS CPU回归42通过、12项GPU/native相关跳过；2Hz服务器完整结果已收到，见下节。

### Waymo 2Hz完整服务器结果（2026-10-10）

固定nuScenes均值、无Waymo训练/校准/阈值搜索；全部7998锚点保留，无重采样/丢窗。

| 名义时距 | Transport IoU | Joint IoU | Transport mIoU | Joint mIoU | ΔIoU pp | ΔmIoU pp |
| --- | --- | --- | --- | --- | --- | --- |
| 平均 | 60.251916 | 61.209875 | 51.734899 | 52.152906 | +0.957959 | +0.418007 |
| 1s | 67.552118 | 68.209418 | 59.487389 | 59.602591 | +0.657300 | +0.115202 |
| 2s | 59.659763 | 60.669601 | 50.866556 | 51.381103 | +1.009838 | +0.514547 |
| 3s | 53.543867 | 54.750606 | 44.850752 | 45.475025 | +1.206739 | +0.624273 |

本次I²-World零IoU排除口径与标准mIoU数值恰好相同，不代表两种定义总是等价。
7796同场景stride5链接中15个timestamp异常（0.1924%，最大1.199943s），实际跨度另报；上表仍是index-based名义时距。
raw free23、无camera/lidar metric mask，未来ego pose显式条件；未来occupancy仅预测后读取，Moving未测。
可确认CCR在该zero-shot人口上三个时距有正增益；没有逐场景统计/置信区间，不能声称所有场景提升、动态指标提升或优于I²-World论文。
服务器摘要：`/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo_i2world_2hz_20261009_225100_83458/summary.txt`，详细`waymo_validation.json`。
这是用户粘贴服务器摘要，不是本地原JSON；未提供累计wall time或snapshot SHA，不猜测。评估wall time不等于正式Dense Forecast FPS。

### Waymo两协议最终归档（2026-10-10）

用户提供10Hz完整摘要：status=complete、39987/39987；四进程同目录恢复后完成，最后invocation复用保存前缀35959。原BrokenProcessPool原因未确认，不记录为OOM。
2Hz/10Hz完整Transport+Joint、全部三个时距/两种mIoU口径、timestamp gap/目标跨度和源目录已记录到独立结果文档、两份摘要及结构化转录JSON；只是用户服务器结果归档，未本地复跑或下载原counts/contract。
2Hz三时距IoU/mIoU全正收益。10Hz Joint平均75.100701/65.997882，Transport74.762743/66.111438；最短0.2s mIoU−0.557557、0.4s−0.006147、0.6s+0.223036pp。不据此再调阈值/重训/挑权重，不称SOTA。
最终10Hz目录`outputs/p0_f9_joint_surface_ccr/waymo10_parallel4_20261010_055012`。两协议均无Moving或正式FPS；没有跨全部invocation总wall time或服务器实际HEAD/snapshot SHA，未知保持null。
详细索引`docs/WAYMO_ZERO_SHOT_FINAL_RESULTS_20261010_CN.md`。以下Waymo入口/测速/接续内容为已完成阶段的实施历史，不是新的待跑任务；后续仅按需收集既有原JSON/契约作复现记录。

用户明确要求10Hz按I²-World代码做：同固定提交load_interval=1/eval_metric=miou，模型forward_test取六未来零基下标eval_time=1/3/5，即native第2/4/6帧，实际名义0.2/0.4/0.6s，不照抄配置注释的错误物理1/2/3s标签。
独立`waymo_i2world_10hz.py`及`run_p0_f9_joint_surface_waymo_10hz.sh`，原2Hz实现/指纹不改，运行/续评保持兼容。全native人口按实际metadata计算（当前39987），共用一次六帧模型预测评分三下标，不重复三次。保留训练slot clock、同冻结均值/四历史/阈值；未训练或插值适配10Hz，不能称物理长时距或同历史预算比较。
同一份metadata/NPZ，无新下载/大缓存；只读源数据/权重，六预测后才读未来GT，完整窗口整数resume拒绝2Hz/10Hz混拼。默认10Hz进程2worker，用户可同卡并发但资源竞争不保证总耗时更短；不干预现有2Hz进程。
本地合成元数据/实际小网格CPU及native模型、旧2Hz/Surface/FPS/递推回归78通过、16项CUDA跳过；Bash语法和CLI help通过，未跑真实10Hz或正式测速。服务器下一步更新后独立后台启动10Hz，发回各自summary。协议解释和audit/full/nohup/resume命令见Waymo文档。

最新服务器分段：2Hz约0.800s/窗，其中历史/搬运准备0.290s、证据构建与投影0.437s、head0.020s；10Hz约1.004s/窗，对应0.479/0.448/0.020s。前两段占总耗时91%/92%，I/O约0.010s，不应继续把原始帧读盘或显存当首要瓶颈。`evidence_projection`包含canonical support、Surface Atlas描述及实时投影，尚无内部细分，不能宣称全部耗时来自投影。
历史component在state构建和causal_source_history内重复提取，相邻窗口四历史亦重叠。用户授权后已实现独立10Hz fast入口，旧2Hz/10Hz实现文件及指纹不改；不干预正在跑的2Hz。

## 新增：Waymo 10Hz无损执行与接续（2026-10-10）

`run_p0_f9_joint_surface_waymo_10hz_fast.sh`：1024MiB/32帧只读内容键component/世界点LRU、取消同窗口重复提取、相同K16/半径2.5/FP64算式的4096行Surface描述并行、最多一个next-history纯CPU预取。
配准/Strong/support/Atlas/learned features/未来投影均live，不写大缓存；预取只在当前六预测完成后调度，无未来GT进入预测。缓存single-flight短锁，日志不能隐式阻塞worker；线程累计耗时不与wall time相加。
默认先做同连续16窗、两遍交替旧/new实际六帧+概率byte gate及测速；每计时pass清几何LRU只暖首窗，不拿全小人口暖命中冒充真实吞吐，再开始全native39987固定评估。
`WAYMO10_CONTINUE_FROM`显式读原停止目录，原contract/state SHA+JSON快照保存新输出；原实现/数据/权重/语义全同才迁移整数前缀，允许仅CPU worker与新增执行信息变化。新目录常规resume仍严格契约；kill -9只恢复最后周期保存窗口，不假装恢复progress尾条。
本地相关CPU回归135通过/18跳过，RTX3050 CUDA/native/真实静态Graph重放及Surface/Strong/递推回归105通过；实际模型运动输入/配准/evidence/plan/概率/六帧/整数指标一致、CLI原结果只读和迁移/续评验收。Bash语法/CLI help通过，不声称全仓CI或真实Waymo/L40S验收。
本地12万合成表面点相同4worker：describe625.588→207.769ms、3.011×、全部bytes相同；只是内部CPU微基准，不是完整eval/FPS。
服务器新版`outputs/p0_f9_joint_surface_ccr/waymo10_fast_20261010_040604`已接续：同窗口旧0.522039339→新0.396463246s，1.316741×、耗时减少24.05%。最近1255–1510共256窗平均0.4739s，不与成对16窗不同负载直接算倍率，也不与原并发旧10Hz约1.004s/窗硬比。
父级prepare0.2189s、evidence_projection0.1858s；子级state/tracks/tubes/Strong0.1695s（总窗35.8%）、surface邻域拟合0.0850s（17.9%），head仅0.0153s。历史就绪0.0013s，缓存没有明显等待；打印每32窗hits+96符合三历史帧复用。不要继续优先加缓存容量或改网络。
上述是暂停前的server日志；用户现在要求继续无损加速并利用闲置资源，新增实现见下节。不把父子计时相加、不自动切换运行中代码。按旧均值从1510算剩余约5.1h仅外推，不是新并行ETA。
操作见`docs/WAYMO_10HZ_FAST_EXECUTION_CN.md`。原10Hz锚点`outputs/p0_f9_joint_surface_ccr/waymo10_20261009_233328`保持只读；继续等待新10Hz完整质量，不重跑/调参2Hz，不改旧权重、nuScenes缓存或长6s结果。

### Waymo暂停后的无损内核与有界并行（2026-10-10）

新增文件，不修改旧2Hz/10Hz/fast-v1实现指纹；native独立ABI1/编译缓存，旧column ABI保持原样。
纯历史占据点齐次坐标/配准排序只读LRU；同FP64整批变换+稳定最小距离C++冲突选点；四帧tube线程与fresh Strong重叠；原trimmed ICP同算式重用当前source树；相同K16/2.5半径/归约顺序的C++表面fit，不缓存learned state/注册矩阵/未来相位。
serial-v2本地整体仅1.035–1.047×，用户指出收益小，遂增加独立spawn多窗口并行，不继续堆局部微优化。
默认2进程×2线程、同一GPU，各保留独立模型/Graph/历史LRU；连续8窗/任务、最多2任务在途。仅传小整数指标，主进程有序连续提交，每8窗周期保存；Ctrl-C/SIGTERM收完在途块，无指标洞/重计数。整块错误不部分提交，旧源目录contract/state及权重只读。
本地真实RTX3050、200×200×16/16请求source模拟、随机小模型、相同16窗×2交替：旧fast-v1单进程4线程0.723398→新双进程各2线程0.372681s/窗，1.941067×，含历史/六帧/GT/指标/IPC；初始化/首窗warm单列，只暖各worker首窗。不能称真实Waymo精度、L40S倍率或单窗口正式FPS。
CPU相关回归97通过/3CUDA跳过；真实CUDA/native/双进程/Graph/Surface/Strong/旧递推回归147通过；Bash语法/CLI help/diff检查通过，不称全仓CI。含实际双进程概率/六帧/整数gate、source只读迁移、speed-only不推进计数、保存权重路径resume及拒绝更改参数。
入口`run_p0_f9_joint_surface_waymo_10hz_parallel.sh`；文档`docs/WAYMO_10HZ_PARALLEL_EXECUTION_CN.md`。下一步显式`WAYMO10_PARALLEL_CONTINUE_FROM=.../waymo10_fast_20261010_040604` + `WAYMO10_PARALLEL_SPEED_ONLY=1`，先新目录短程比单/双进程。原前缀取真实保存cursor；用户发回`WAYMO10_PARALLEL_PAIRED_SPEED`后才判断是否值得完整resume；未代跑服务器或重启评估。
并行progress打印吞吐seconds/window，worker_seconds及stage重叠不相加；平均128/256新窗口才判断稳定吞吐。独立单模型Dense Forecast FPS口径/论文方法不变。

### 服务器双进程已验收与同v2布局探针（2026-10-10）

服务器`waymo10_parallel_20261010_050832`短测单进程0.3506→双进程0.1701s/窗、2.062×，SIX/概率/整数gate通过；随后恢复原计数，最新5088/39987、累计吞吐0.235s/窗。剩余约2小时17分只是按当前负载外推，worker0.486–0.621s不是总吞吐。不拿16窗和全集累计不同人口算退化。
用户授权比较4进程×1线程；新增独立`run_p0_f9_waymo10_worker_layout_probe.sh`，原并行实现/指纹一个文件都未改。新入口只读停止目录完整契约/实际cursor，在整个探针期间持有原kernel lease阻止并发resume。相同v2、相同64连续窗、AB/BA两遍，首窗warm与全六帧/概率/运动/整数gate计时外，每遍重置历史LRU，只写新receipt/speed/summary，不写科学state、不自动全评。两布局共6个worker常驻但一次只算一组，idle模型显存需公开，不把评估吞吐称FPS。
4×1建议门槛为平均至少快10%且每遍快；不满足保留2×2。若换布局，新目录显式迁移原保存整数前缀；普通旧目录resume仍严格2×2。文档`docs/WAYMO_10HZ_WORKER_LAYOUT_PROBE_CN.md`。
本地CPU相关47通过/2CUDA跳过；真实RTX3050 CUDA相关30通过，含实际2/4进程Graph/概率/六帧/计数一致、旧接续、源目录并发锁、只测不推进计数；CLI help/Bash语法通过。不是服务器布局提速证据，不称全仓CI。下一步服务器安全暂停后跑探针，发回最终summary，不动权重/阈值/训练/正式FPS/其他Camera任务。

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

## 用户选择静态接续并授权全集复测（2026-10-09）

真实四路screen：static_carry TRAIN64 avg4–6 mIoU+1.469280pp/IoU+2.499484pp/Micro-0.025451pp；
dev64 +1.553857/+2.611150/-0.022158pp。combined略差，SE2无益；原TRAIN Moving非负规则未通过。
用户明确接受小幅Moving下降，固定static_carry，要求跑同官方人口全集并只报告IoU/mIoU。
新增显式 `SURFACE_CARRY_APPROVED_ROUTE=static_carry` all入口：记录用户选择，不改旧screen/门槛，selected_train_route仍null。
同冻结均值/阈值/官方metadata，仅baseline+static，第一段共享不改；不训练、不用未来标签预测。
all默认metrics-only，不读/计算Moving支持，结果标记未评估/null，摘要报告1–6s及1–3/4–6平均的IoU/mIoU。
不计算未用SE2分支；每时距dense整数指标只扫描一次复用dataset/scene。旧默认screen原样，严格同目录resume。
本地专项和旧Surface/两段递推/官方人口/Strong/FPS回归130项通过、20项CUDA跳过；CLI help/Bash语法/diff检查通过。
含metrics-only省略Moving标签访问、整数中断恢复、源screen/权重只读、固定静态路线逐字节一致；不冒充全仓CI/真实GPU或全集收益。
本地不具备完整服务器数据，未代跑或下载完整评估产物；服务器结果现已收到，见下节。

## 最终6秒论文指标已由用户确认（2026-10-09）

用户粘贴完整all摘要：2569窗口/150场景，static_carry 4/5/6s IoU44.768673/41.083833/37.801952，
mIoU32.350774/28.876602/26.126216；avg4–6 IoU41.218153 / mIoU29.117864。
相对同人口baseline avg4–6增益+2.655306 IoU / +1.732402 mIoU pp；1–3s逐项不变。
用户明确要求记录并将static_carry作为最终论文长时预测结果，baseline保留消融，不再加入SE2/combined。
原full4369主表IoU55.211054/mIoU44.153241不变，不拿2569长人口1–3s均值替换。
未来GT ego poses显式条件，保留历史静态几何的stateful causal rollout；不输入未来occupancy/mask/annotation。
这是推理方法改动，不是无损执行；没有6s重训/阈值校准。全人口Moving/FPS未测，不借用screen或6帧FPS值。
按GenieDrive公开代码metadata人口对齐；论文Table2实际起点集合未独立确认，不直接声称严格SOTA。
档案为用户服务器摘要转录，非本地原evaluation.json；实际运行目录/HEAD/snapshot SHA未收到，不猜路径或指纹。
此次仅文档/指标归档，不改变模型、训练、旧默认入口或其他Waymo改动。

## Camera / Pred ego 当前决定（2026-10-10）

用户只做 STCOcc Camera，不做 BEVStereo。按 I²-World-STC 官方历史完整 semantics + 无 camera/lidar metric mask，不额外借用 GT 历史 visibility，不进行 mask 选优。需要的 known 是预测网格有效性，不是真实 sensor visibility。
服务器旧 `new_code/cache/come_main_table/camera_pred` 与 `camera_gt` 各4219窗/150场景，四历史及六 `future_e2g` 保留；原 BEVStereo/planner 路径已失效。仅考虑复用严格对齐的 camera_pred 预测世界 ego pose，绝不复用 BEVStereo 历史/latent/搬运结果；六未来 XY 世界坐标、yaw 相对 t0 非累积。四设置使用共同人口并同人口重算 Occ+GT，不混旧4369分数。
STC包扫描尚未确认存在，官方 I²-World 下载链接已核实；现已实现官方ZIP下载检查、逐值无损uint8紧凑解压、冻结Surface四设置 evaluator。四设置同planner-covered人口（all预计4219），一趟完整六帧后才读未来GT语义，报告1/2/3s与均值标准IoU/mIoU并附官方排除零类诊断，不混旧4369数字。
严格身份/order/tag、当前z/tilt与独立yaw、文件/pose指纹及固定mean检查；Pred全部未来几何用预测pose，无GT回退/事后对齐；Camera无GT visibility/cache复用。每8完整窗口整数保存，Ctrl-C/SIGTERM安全停、同输出目录严格resume，旧训练与缓存只读。
本地STC+Waymo+geometry-carry相关回归63 passed/4 CUDA skipped；实际小模型NumPy/native四设置概率/六帧exactness、无future GT读入、错序/缺失fail-closed、compact逐值相等和中断不重计通过。CLI help/Bash语法通过，不称全仓CI/真实STC精度。
未下载完整STC数据、未训练/代跑服务器。入口 `run_p0_f9_joint_surface_stc.sh`；顺序命令见 `docs/SURFACE_STC_CAMERA_PRED_EGO_CN.md`，调研见 `docs/CAMERA_PRED_EGO_RESEARCH_20261010_CN.md`。

## STC四设置执行提速（2026-10-10）

用户服务器421–484窗四设置1.7103s/窗；准备0.8247s、证据/投影0.7111s、head0.0406s。
新增独立 `run_p0_f9_joint_surface_stc_shared.sh`：复用原生CPU warp/Surface fit；
同一实际历史GT/Pred共享history-only tube/运动与canonical Surface证据，单bundle、内容/pose/visibility key。
Strong/未来变换/owner/fallback/phase/prob/dense各自重算；不训练/改阈值，不跨模型/历史复用learned输出。
初次8相同窗口四路六帧逐字节检查+AB/BA实测，>1%提速才接着剩余窗口；不能预告真实L40S倍数。
显式只读旧STC状态桥接到新输出，源/新租约全程持有、契约与integer counts校验，原入口/源码/旧结果不改。
本地CPU小模型byte与整数恢复、不同历史失效、科学变更拒绝、CLI源只读/严格恢复、租约及篡改检查通过。
相关回归69 passed/7 CUDA skipped；含新终端launcher原SWFM环境安全恢复，无shell eval。
真实CUDA/STC数据速度与质量仍待服务器内建检查；详见 `docs/SURFACE_STC_SHARED_EXECUTION_CN.md`。

## 冻结 Pred ego / STC 因果适配集中对照（2026-10-10）

用户明确不做 aligned 修分；优先查 Pred 几何误差与 STC 时序噪声，不改网络、不重训。
最新约千窗前缀四设置平均 mIoU：OccGT42.723579 / OccPred21.645784 / STCGT20.107599 / STCPred14.205545；未完成全集，不能混早期64窗数字。
新增独立 `run_p0_f9_stc_causal_geometry.sh`，只跑固定dev64：四基线、各自历史路面z/tilt补偿、STC地面列稳定及组合，共九路。
路面补偿保留planner世界XY/yaw；历史区域不足、查询在内点凸包外、>0.8m高度或>3度倾斜修正均回退。不修复真实XY/yaw规划误差，不使用未来GT位姿对齐。
地面稳定只改历史一致支持的class11/12/13/14既有1–3格地面列，最大1格；不把缺失当free、不新增列、动态/其他占据逐字节保护，history masks不变。
动态组件速度暂不覆盖，只报告同类因果匹配/初速/质心抖动，避免与网络residual冲突；不宣称已解决动态噪声或能恢复二十点落差。
同mean5/6/8/12/14、ADD0.5/REMOVEoff、4→6、无mask/阈值/epoch搜索；GT-conditioned基线仍仅按原协议读GT ego。
全部九路预测后才读未来语义与审计GT；每8完整窗口整数保存、同目录严格resume，新终端恢复本次SWFM flags。旧四设置入口/数据/cache/权重/训练/前缀不改。
摘要一趟输出九路1/2/3s及均值IoU/mIoU、位姿误差分解、STC t0质量、修正支持/拒绝和动态抖动；详细契约/原始计数保留。
本地相关回归109 passed/2 CUDA skipped，含真实CPU九路推理、原四基线六帧字节一致、权重不变、GT读取边界、地面向量化参考一致、鲁棒平面拒绝、CLI只读/中断恢复；Bash语法/help通过。
操作与限制见 `docs/STC_CAUSAL_GEOMETRY_SCREEN_CN.md`。本地CPU回归不代表真实服务器候选增益；尚未跑L40S/dev64，不自动扩大到4219全集。

## 最新：失败适配后的单次 yaw 一致性 screen（2026-10-10）

服务器九路dev64已完成：历史路面z/tilt OccPred mIoU−1.212540pp、STCPred−0.403252pp；STC时序地面Pred−0.306691pp、组合−0.644499pp，均失败。156.23s/64包含九路，不是单网络FPS变慢。基线OccPred18.350472/28.307956、STCPred11.960050/21.202278（mIoU/IoU），不能与此前约千窗混拼。
用户授权再试一次无训练方案：新增独立 `run_p0_f9_stc_trajectory_yaw.sh`，固定dev64 OccPred/STCPred原版与候选共四路；原网络、mean、XY/z/body tilt/阈值不变，可靠历史heading校准预测XY切线yaw，低速/倒车/不稳定/大修正原pose回退。
仅四历史pose/timestamp与六预测pose进入适配；全部预测完成后才读未来GT用于审计。没有aligned、GT mask、未来真值校正，不重复失败ground分支。不改旧STC实现指纹/结果，单独整数状态/严格resume；候选全pose不变复用原预测，首次真实改yaw仍验完整六帧/概率。
新规则是待验证因果推理候选，不是COME已有方法或官方原planner reproduction；不保证收益、不自动全集或调参。无清晰增益就停止此路线。详见 `docs/STC_TRAJECTORY_YAW_SCREEN_CN.md`；下一步只需服务器运行新入口、发回summary。
本地新yaw+旧STC协议/共享执行/失败geometry回归57 passed、1 CUDA skipped；实际CPU小模型原六帧/概率byte parity及权重不变、未来GT隔离、首次延后真实修正仍exactness、严格整数续评/CLI输出保护通过，Bash语法/help/diff通过。不声称全仓CI、CUDA或真实dev64收益；windows-python-env-guard仅使用已验证项目解释器。

## 仅按需检索的历史

冻结 Surface CCR 完整 TRAIN×3 DEV512 mIoU40.418031/Micro31.052754；扩大 VAL4369 mIoU43.991199/Micro32.043717。
Frozen B 与旧 Local 的静态差距集中 road11/sidewalk13；Surface 内部几何条件读出已追回大部分冻结静态差距。
这些是早先冻结实验，不是新联合精度。旧阈值、halo、共享场、点 CCR 支线不重启。
详细历史仍保留在 `docs/CLEAN_JOINT_SURFACE_CCR_CN.md`、`docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`，需要时检索，不复制旧日志。
