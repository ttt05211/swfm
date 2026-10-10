# 扩大 TRAIN 的 ego 头单轮试训

## 固定定义

用户先授权扩大训练规模，随后明确改为只训一轮。本次只实现这一次单轮试训，不自动扩训。

- 身份人口为原 TRAIN20430 的全部窗口，严格四历史、六未来。
- 先按 `SHA256(seed:scene)` 确定性留出 10% 的 TRAIN 场景；剩余场景用于 ego 头训练。窗口比例未必恰好 90/10，以实际打印的 fit/holdout 为准。每个 fit 窗口只优化一次。
- 留出场景不进入 ego 优化器，但冻结 WM 已在完整 TRAIN 上训练过；不能称整个系统没有见过留出场景。
- 仍为原约 0.43M 参数的头、同原随机初始化，不续训失败的 old320/A/B。WM/Surface CCR 完全冻结，原 GT 和外部 planner 路径不变。
- 只用 B 的单项固定 R10m 五参考点 SE(2) Smooth-L1，监督六个未来时刻的相对 t0 绝对 XY/yaw。
- batch64，AdamW，学习率 3e-4 → 3e-6，在这一轮内余弦下降。没有 20 轮或尾部续训。
- 固定导出 `head_epoch1.pt`，即使留出集没有改善也仍评估第 1 轮。留出集、dev 均不选择权重。
- 最后只评固定 dev64 的 OCC/STC × GT/external/internal 六路。导航仍是 GT 派生 destination command，显式 navigation-conditioned；不是无导航预测。
- 不用未来 occupancy/mask 生成特征，不做事后 GT pose alignment，不改阈值。未来 occupancy 标签只在全部六路预测后评分。

## 旧 TRAIN/cache 接口诊断如何解释

服务器已核查原 TRAIN64：七个特征字段在 bank / TRAIN 实时 / eval 实时之间全部逐字节一致，标签、指令和 train/eval 模式一致，完整 TRAIN1024 报告复现。

batch64 与单窗仍未通过原严格预算，最大 XY 差约 2.09mm、yaw 0.010°。原门槛失败保留，不重标成 PASS。这个量级不能解释约 0.29m → 3.17m 的 TRAIN/dev FDE 差距；扩训用于检验样本覆盖不足及过拟合，而非宣布已排除所有分布/数值问题。

## 效率与文件保护

已验证的旧 1024 bank 按 key 只读复用。其余特征一次性提取，使用四线程有界历史预取、512MiB 原帧 LRU 和 512MiB 纯因果几何缓存；不运行未来 Strong/render。完整特征 bank 在头训练前一次性放入设备，之后不重复读取原始 occupancy。

首次主要成本是为剩余 TRAIN 构建历史特征，而不是小头的一轮反传。实际 `bank_mib`、构建/训练/源核验分段耗时会写入 summary；本地没有真实服务器数据，不承诺真实耗时或增益。新 bank 的特征张量有 4GiB 显式加载上限，不另建几十 GiB 稠密 geometry 缓存。

原 bank 的历史实现指纹包含所有顶层 `tools/real_motion/*.py`。为避免破坏已验证的旧 bank，本次 Python 位于独立 `tools/ego_experiments/`；没有修改旧 head、提取公式、协议或 evaluator。新入口自身源码另行绑定。

所有新文件写到独立输出目录，旧权重/cache/优化器不写入。`last.pt` 是续训断点；`head_epoch1.pt` 是评估专用导出，不能拿它恢复优化器。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm

export EGO_FULL_OUT="/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_$(date +%Y%m%d_%H%M%S)"
bash tools/real_motion/run_p0_f9_surface_ego_full.sh
cat "${EGO_FULL_OUT}_eval_dev64/summary.txt"
```

默认复用原来源：

```text
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_head_screen_20261010_194309_837
```

若实际路径不同，用 `EGO_FULL_SOURCE` 显式指定。不要指向 A/B 的 paired 目录。

入口自动恢复原来源记录的 `SWFM_*` 特征环境，再执行新的 Python，因此不会因新终端遗漏环境变量静默换公式。还会核验原 Torch/CUDA、权重、TRAIN cache/info、metadata、源历史文件与代码契约。

## 中断、恢复与结果

Ctrl-C / SIGTERM 会在完整 bank shard 或优化步边界保存；kill -9 只能恢复最后一次已落盘的更新。训练每 32 步周期保存，整轮完成也保存。建 bank 支持逐 shard 恢复，评估支持整窗恢复。

新终端恢复时必须使用相同输出路径，不重新生成时间戳：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export EGO_FULL_OUT="/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_原时间戳"
export EGO_FULL_RESUME=1
bash tools/real_motion/run_p0_f9_surface_ego_full.sh
```

已完成训练的 resume 不再执行优化步、不重写头导出 SHA；可以只继续未完成的 dev64 评估。人口、代码、batch、轮数、环境或源文件改变会拒绝续训，不静默降级。

`EGO_FULL_SKIP_EVAL=1` 可仅完成建缓存及单轮训练；之后同目录恢复去掉该变量即可做一次固定评估。

只需发回 `${EGO_FULL_OUT}_eval_dev64/summary.txt` 内容：它合并 TRAIN 留出集、最终轮、先验、OCC/STC 三种轨迹条件的 IoU/mIoU 和轨迹误差。若效果不好，不自动重试、扩训或替换正式指标。

本地专项与相关回归：180 passed / 5 CUDA skipped；覆盖真实 CPU 小网格冻结 WM 的历史提取、旧 bank 只读复用、部分 bank/训练/评估恢复、短 batch 加权、源码/游标/导出破坏拒绝，以及最终轮而非 best 选择。CUDA 测试跳过，不等于已验证真实 L40S 指标。

## 已建到一半时先试训（追加，不改变原全量契约）

`EGO_FULL_BANK window=7168/20430` 表示完成了 7,168 个历史特征窗口，不是完成了这些窗口的头训练。旧任务要等 bank 完整才开始训练。为了省掉没有必要的剩余提取，增加独立的 `run_p0_f9_surface_ego_partial.sh`。

先在原终端按一次 Ctrl-C 并等待安全保存及提示符。新入口会持有原目录的内核 lease；如果原进程仍运行，它会拒绝启动，不自动杀进程。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_surface_ego_partial.sh
```

它只读已落盘的连续 bank 前缀，最多 10,240 窗；如果现在只有 7,168 窗，就用这 7,168 窗，不补齐、不重建、不复制大 bank。沿用原全量场景的 fit/holdout 身份，固定随机初始化、batch64/R10m/单轮余弦，固定第1轮头，然后同一个 dev64 六设置评估。此快筛是按场景排序的已有前缀，不是均衡抽样、不能冒充完整 TRAIN 的代表性指标。

默认新输出为 `原全量目录_partial_screen`，结果为 `原全量目录_partial_screen_eval_dev64/summary.txt`。如果只找到一个全量任务，入口自动识别；多于一个时拒绝猜测，请显式指定：

```bash
export EGO_PARTIAL_PARENT="原全量任务的完整路径"
bash tools/real_motion/run_p0_f9_surface_ego_partial.sh
cat "${EGO_PARTIAL_PARENT}_partial_screen_eval_dev64/summary.txt"
```

半量试训中断后，用同一父目录和 `EGO_PARTIAL_RESUME=1` 恢复。人口/receipt固定，不会因父 bank 以后增长而偷偷扩大 pilot 数据。

原完整训练的三份绑定文件完全未改，既有缓存和 `training.json` 都不写入；新试训自己的代码单独绑定，不破坏原 `EGO_FULL_RESUME=1`。如果需要完整实验，原入口继续补完 bank 并按原完整人口训一轮；只能复用 bank，**不能将半量头/Adam 当全量续训断点**，两次余弦总步数及人口不同。

没有按 dev 自动挑权重、调阈值或启动全量，也不据半量结果断言完整训练一定有效/无效。发回新的合并 summary 后再决定。

## 复用相同缓存训3轮，再评 dev512

用户已要求此对照。新的 `run_p0_f9_surface_ego_three.sh` 默认只读已完成的
`ego_one_epoch_full_20261010_231306_partial_screen` 契约，复用其10129窗口（fit8998/holdout1131）和完全相同的缓存 receipt。
不补剩余窗口、不重提特征、不更换WM/CCR；原全量及一轮入口/文件完全不改，旧断点保持兼容。

从原随机初始化重新训练3轮，batch64、同R10m目标、3e-4→3e-6整段余弦；预计423次更新。
不是在一轮最低学习率的断点续两轮。固定第3轮 `head_epoch3.pt`，不挑 dev-best。
然后固定dev512，OCC/STC × GT/external/internal 六路；直接复用原评估引擎，无aligned、mask或阈值调整。
默认沿用dev64 manifest中的完整512个 `parent_keys`，检查实际选中恰好512，不只评64也不静默丢窗。

```bash
bash tools/real_motion/run_p0_f9_surface_ego_three.sh
cat outputs/p0_f9_joint_surface_ccr/ego_one_epoch_full_20261010_231306_partial_screen_three_epochs_eval_dev512/summary.txt
```

输出默认为原pilot目录旁的 `_three_epochs`；源路径可用 `EGO_THREE_SOURCE`，输出可用 `EGO_THREE_OUT` 指定。
中断后同输出设置 `EGO_THREE_RESUME=1` 重跑入口，恢复Adam/随机数/游标；训练完成只接续评估，不重复训练。
本地仅3项必要测试及CLI/Bash检查；未运行真实L40S或承诺增益。评估时间不是FPS。

### 3轮已完成后的512人口检查修复（仅评估）

服务器已经成功完成3轮423步，留出FDE2.219904m。旧 evaluator 将manifest父集合、请求512和planner可用集合硬要求同为512而拒绝评估。
日志未给出具体父集合/覆盖数量，不能断言缺多少行。使用新增 `run_p0_f9_surface_ego_three_eval.sh`，直接读取既有 `head_epoch3.pt`，不重新训练或读取TRAIN bank。
为保持checkpoint绑定的代码SHA，原3份three文件一个都不改；新人口修复 evaluator 单独绑定评估契约，旧导出仍通过原验证。

若manifest已有selected512则原样用；否则从不少于512的冻结parent按原scene-balanced规则固定512，再求planner交集。
不补行、不pad、不在缺行后重抽凑512。全部六设置同一交集；少于512时显式称 `dev512_planner_covered_intersection`，报告父集合/请求/实际/缺失数量和完整身份，不冒充完整512。
若parent本身少于512仍拒绝，并明确提示换完整parent manifest。

```bash
bash tools/real_motion/run_p0_f9_surface_ego_three_eval.sh
```

默认原three输出旁的 `_eval_dev512`，读取源training.json恢复原SWFM环境。已有state时自动接续同一评估，不写训练文件。
仅5項人口专项测试以及CLI/Bash检查；真实服务器父集合/覆盖数量等待实际输出，未重新训练或宣称dev效果。
