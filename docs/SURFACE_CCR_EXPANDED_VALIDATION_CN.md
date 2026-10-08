# 固定 Surface CCR 第三轮：扩大验证与输出一致的执行提速

## 已有证据与本次范围

用户服务器完整TRAIN20430×3冻结验证：DEV512 joint mIoU40.418031 / MovingMicro31.052754，vsB +0.380824 / 0 pp；vs Old Local REMOVE-off +0.041886 / +0.941836 pp。同窗口正式FPS B49.696705、新CCR40.102844，六帧120.732→149.615ms。精度追回，但速度余量很薄。

本次不重训、不加新模块、不改候选/阈值/数值精度。固定输入为：

```
Surface /root/nas/occ/swfm/outputs/p0_f9_surface_ccr/full20430x3_20261008_131503_838/last.pt
B       /root/nas/occ/swfm/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt
E19     /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
VAL     /root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1
```

加载器验证第三轮完整边界、teacher/config/manifest、模型/决策合同、B warm-start SHA，并核对原共享/动态权重逐元素未变。三份权重先快照，原权重、optimizer、RNG和大缓存不写。

## 执行优化，不是新方法

`real_motion/surface_ccr_execution.py`只对纯静态分块捕获CUDA Graph，复用执行图而不是复用历史learned features。每次上传新输入、更新实时source row0、执行原算术、回读全部概率。保持8192分块、BF16、原候选顺序和六个读出；动态/混合分块仍eager。

每个新形状首次捕获需与eager概率逐字节一致。图数量≤4，超过就淘汰；权重变化拒绝使用旧session；捕获不支持/字节不一致明确记录并回退原完整eager，不缩候选或batch。

正式20窗口（全部dev64中18 scene-balanced +2 source-only压力窗口）×3轮，轮转B/eager/graph顺序。所有概率、六帧dense以及动态概率在计时外检查。graph只有实际执行、无拒绝并且paired mean至少快2%才被标记为该预热FPS实验的更快后端；不按mIoU挑方法。**此结论不再自动决定全量单遍质量评估的后端。**

### 2026-10-08 变长单遍性能回归修复

用户运行 `expanded_20261008_150128_838`：125窗口的surface读出2629ms/窗口，B8.35ms，输入等待1.41ms。缓存等待不是主要瓶颈；尚未拿到原运行的实时图子统计，不能直接把全部2629ms归因于建图。

代码存在确定的图生命周期问题：graph以尾块实际长度为key，只有4个LRU槽；不同窗口的不同尾块会反复capture/校验/淘汰，而单窗口预热FPS没有包含这种单遍开销。本地RTX3050合成变长8192+尾块、6窗口、source数逐次变化的head-only实测（含全部capture）：旧策略平均64.0ms，固定大块图15.9ms，原始eager7.27ms，全部概率字节一致。这不是服务器全模型加速或正式FPS结果。

修复后：全量质量评估默认 `eager`。可显式 `SURFACE_EVAL_EXECUTION=full_chunk_graph`，它只捕获8192的完整纯静态分块，变长尾块一律eager，不padding、不改矩阵batch形状/候选/阈值。该模式最多两个形状key（有/无source），不会因变长尾块反复建图。保留正式FPS的原实验路径/计时边界，结果中的 `speed.selected_execution` 只描述预热FPS；`quality_eval_execution` 描述本次质量评估。

每窗口 `progress.jsonl` 新增cache_hit、input_wait、完整候选数、执行子计时/计数；每32窗口结果JSON也更新VAL缓存统计和图执行统计。Ctrl+C/失败的finally同样保存执行统计。每32窗口控制台打印读出和capture毫秒。计时子项是包含于surface_probability的主机计时，不能再与大项相加或称GPU利用率。

FPS固定：CausalHistoryState→fresh Strong/KTA+live motion+learned CCR+**live投影相位**+六帧dense。不把future phase移到计时外，不缓存Strong、poses、labels或readout。历史表面表示准备另报，不称raw-input E2E FPS。统计六帧均值/P90、峰值/增量显存以及各host stage；host不是GPU活跃利用率，live phase是readout的嵌套子项，不能重复相加。

本地RTX3050，48000合成候选、5遍head-only：surface eager20.49ms、graph17.73ms，约13.5%降低；不包括历史几何、live phase、Strong、motion、composition。仅是当前本机运行，不能与历史不同运行数值拼接，也不意味着L40S全模型FPS已改善。

## 一次full4369扩大验证

`validate_p0_f9_surface_ccr_expanded.py`在同一pass计算Transport、B、Surface CCR和Old Local REMOVE-off，复用完整VAL固定历史几何，原learned预测均实时计算。Old Local仍同transport/同窗口且固定(.5,.5,REMOVE-off)。不加载完整TRAIN cache，降低内存占用；只有VAL RAM LRU512MiB、frame LRU1GiB及有界预取。

每个dense混淆计数只计算一次，以整数累计到full、DEV512、DEV512之外窗口、DEV512之外场景和各scene；不重复读取模型、不用overall均值相减来构造子集分数。报告IoU/mIoU/MovingMacro/Micro及1/2/3s，17类IoU，road11/sidewalk13的TP/FP/FN，静态/动态ADD precision/recall、编辑质量和逐scene变化。每个验证窗口检查B/new动态概率字节一致。

这些VAL人口已用于研究，只称扩大验证。3857个非DEV512窗口不等于全新独立测试；场景之外集合为空时明确报告 unavailable。

## 运行与安全恢复

```
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_surface_ccr_expanded.sh
```

一次先跑paired FPS，再跑full4369；不启动训练，也不自动部署。B SHA固定；candidate取用户已报告的具体第三轮路径，不自动扫描best。

Ctrl+C/SIGTERM在当前窗口完成后保存到新输出 `evaluation_progress.pt`。kill-9只能恢复最近32窗口周期保存。恢复严格校验checkpoint、人口/顺序、缓存namespace、执行参数与实现指纹；从已完成的整数cursor接续。已完成的FPS直接复用，不重测，不碰训练游标。此次仅允许已知 `36714f1` 实现指纹在CCR head文件SHA完全未变时迁移到执行修复版；其余合同字段仍须完全相同，未知旧实现拒绝。新执行后端可改变但必须在结果中明示；累计performance包含旧版prefix，不能据此宣称同窗口paired提速。

```
SURFACE_EVAL_RESUME=/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/本次目录/evaluation_progress.pt \
  bash tools/real_motion/run_p0_f9_surface_ccr_expanded.sh
```

恢复也写新目录，不覆盖旧输出。需要计划暂停可设置 `SURFACE_EVAL_MAX_WINDOWS=32`；它只限制本次执行，不改变full人口或筛选checkpoint。

结果为 `summary.txt`、`expanded_validation.json`（含180条paired FPS trial和完整分段/场景/类统计）及 `progress.jsonl`。失败保存错误，不能把未完成prefix当full结果。

当前已停止的具体运行可续：

```bash
SURFACE_EVAL_EXECUTION=eager \
SURFACE_EVAL_RESUME=/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/expanded_20261008_150128_838/evaluation_progress.pt \
  bash tools/real_motion/run_p0_f9_surface_ccr_expanded.sh
```

未保存该文件时拒绝恢复，不自动从零跑；kill-9仅恢复最后周期统计。

## 本地验收和下一步

相关本地回归81 passed/1 skipped；606个Python AST和Bash语法检查通过。真实CPU/CUDA graph非零geometry权重/输入变化、8192+尾块/不同source数/road-sidewalk字节一致、图淘汰/拒绝回退、动态不变、静态冻结训练回归、实际前向/投影/合成/整数统计CLI stop/resume、VAL cache/正式dataflow合同。CLI外部nuScenes与FPS使用mock，不代替服务器真实数据。未声称全仓CI通过。

先看扩大验证收益与正式速度。当前命令不启动随机初始化联合训练、不改阈值、不扩写缓存、不覆盖候选。扩大验证完成后才实施同一结构的干净联合训练。
