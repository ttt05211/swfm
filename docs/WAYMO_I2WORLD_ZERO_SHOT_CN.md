# 冻结 Surface CCR：对齐 I²-World 的 Occ3D-Waymo zero-shot

## 完整结果已归档（2026-10-10）

2Hz完整7998锚点：Joint IoU61.209875 / mIoU52.152906；10Hz完整39987锚点：Joint IoU75.100701 / mIoU65.997882。
两个协议时距/人口不同，不直接比较绝对分数；2Hz修复双指标提升，10Hz平均IoU提升但mIoU略降。
全部Transport/Joint、三个horizon、指标与时间审计见[最终结果](WAYMO_ZERO_SHOT_FINAL_RESULTS_20261010_CN.md)。
下文为协议与复现操作，不表示需要重新启动已完成的评估。

## 当前定义

只评估现有 **epoch5/6/8/12/14 单一均值模型**，不训练、蒸馏或使用 Waymo 调阈值。
V18/CCR 权重、weighted ADD raw sigmoid@0.5、REMOVE-off 均不变。
不将 nuScenes TRAIN/VAL 几何缓存用于 Waymo，不持久化预测或 learned features。

对齐的官方源码固定为 [II-World@661d830](https://github.com/lzzzzzm/II-World/tree/661d830f9b34ee03ce368db164a72753ab8764a3)：

- `configs/world_model/ii_generate_world_waymo.py`：2Hz `load_interval=5`，预测未来6帧。
- `mmdet3d/datasets/waymo_world_dataset.py`：完整 metadata 按 timestamp **全局排序后 `[::5]`**；不是逐场景抽样。
- `mmdet3d/datasets/pipelines/loading.py`：`voxel_label` 和 Waymo→nuScenes 18类映射。
- `mmdet3d/datasets/occ_metrics.py`：无 lidar/camera mask，三个报告时距分别累计，再取均值。

边界行为不偷偷改：场景开始重复最早有效历史帧；结束重复最后有效未来帧、pose 和标签，保留全部抽样 t0。
在 2Hz 下未来下标1/3/5为名义1/2/3秒。记录实际时间跨度和补帧数量；补帧目标的实际时间可以短于名义时距。
官方配置注释显示39962原始帧/7994抽样锚点/202场景；本入口从实际文件计算人口，默认检查202场景，不凭注释制造样本。

**时间间隔审计：** stride5 是名义2Hz的按下标抽样，不是严格每0.5秒重采样。用户下载的官方 metadata 实测39987原始帧、7998锚点；7796个同场景相邻链接全部frame step=5，其中15个真实时间间隔超出0.35–0.65秒，最大1.199943秒。入口保留全部锚点、实际pose/timestamp和原边界行为，不插帧、删窗或把真实间隔改写成0.5秒。`timestamp_gap_audit` 报告数量、比例、场景、例子和frame step直方图；仍拒绝非正时间/倒退帧号，以及中位数偏离名义2Hz的错频率/错时间单位。名义1/2/3秒标签沿用官方index定义，实际跨度另报，可能长于名义时距。

**明确的不可等同之处：** 我们固定四帧总历史（包括t0），沿用已训练模型。I²-World 的 temporal tokenizer/cache 配置含previous/current语义，不能宣称输入预算或架构完全相同。
我们使用未来 ego poses 作为条件，不声称同时预测 ego trajectory。未来 occupancy 不参与 source 提取、motion 或 CCR；六帧预测完成后才读取 GT。
历史 `history_observed` 为全真，表示与官方 dense occupancy 输入一致；不读取额外 Waymo observation mask。评价也不使用这些 mask。

## 下载什么

只需要三个部分，不需要相机图像、原始 LiDAR、Waymo TFRecord、训练集、I²-World 权重或 VAE：

1. **Occ3D-Waymo validation 0.4m NPZ**：官方发布 [voxel04/validation](https://drive.google.com/drive/folders/1_RK67cSoNkvnWjhIAOQ3dy7UBqqKjIXG)，场景000–201的 tar。
2. **waymo_infos_val.pkl**：I²-World 提供的 [metadata 文件](https://drive.google.com/file/d/1VNQCMsThtGpr8vlk6NbQcv-auqUWRVjR/view)。
3. **cam_infos_vali.pkl**：[ego pose metadata 文件](https://drive.google.com/file/d/1zQ_7ZuZ2sPOhmIMH0BRj3s2V1qlaCFf7/view)。名字含cam，但我们只读取各帧第一条 `ego2global`，不需要图像。

数据说明入口：[I²-World dataset preparation](https://github.com/lzzzzzm/II-World#-prepare-dataset)。
pickle 可以执行代码，只使用可信官方来源；入口记录两个 PKL 的 SHA256，不自动下载或反序列化未知来源文件。
下载 tar 后先查看其内部目录，再解压到满足下面结构的位置；不自动覆盖已有数据。

```text
/root/nas/occ/swfm/data/waymo/
  waymo_infos_val.pkl
  cam_infos_vali.pkl
  validation/
    000/000_04.npz
    000/001_04.npz
    ...
    201/..._04.npz
```

官方 I²-World loader 使用 **raw free=23**；Occ3D 文档/另一种发布版本可能是 **raw free=15**。
默认严格23，遇到15/255/未映射标签立即停止，绝不把free误当manmade或从未来GT猜编码。
只有确认所下载的文件使用free15时，显式设置 `WAYMO_RAW_FREE_LABEL=15`，报告会注明 encoding normalization。
两种编码都保持0–14的同一类别映射，禁止对模型已经输出的nuScenes类别再次进行Waymo映射。

## 服务器命令

当前分支 `feature/v22-surface-aware-ccr`。使用已有 OccFM 环境与已安装的 native CPU 编译器。
默认从原联合训练run的已完成DEV512 comparison查找**同一个冻结均值文件**；不重新平均或挑权重。

```bash
conda activate OccFM
cd /root/nas/occ/swfm

# 数据审计：不加载模型、不读取未来GT。与正式评估使用不同输出目录。
WAYMO_ROOT=/root/nas/occ/swfm/data/waymo \
WAYMO_AUDIT_ONLY=1 \
bash tools/real_motion/run_p0_f9_joint_surface_waymo.sh

# 完整2Hz zero-shot，另建输出；不需要 nuScenes 几何缓存/E14断点。
WAYMO_ROOT=/root/nas/occ/swfm/data/waymo \
bash tools/real_motion/run_p0_f9_joint_surface_waymo.sh
```

如果冻结均值comparison不在默认目录：

```bash
WAYMO_CHECKPOINT=/absolute/path/mean_top5_dev64_mIoU.pt \
WAYMO_ROOT=/root/nas/occ/swfm/data/waymo \
bash tools/real_motion/run_p0_f9_joint_surface_waymo.sh
```

或者设置 `SURFACE_MEAN_SOURCE` 为原已完成 comparison 目录。默认原训练锚点是
`outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787`；可用 `SURFACE_RUN_DIR` 改为正确锚点。

每8个完整窗口保存整数计数。Ctrl-C/SIGTERM在完成当前窗口后安全停止，**不要kill -9**。
续评指定原输出，不是训练resume；数据、权重、人口、实现和执行设置变化会拒绝混拼。

```bash
WAYMO_ROOT=/root/nas/occ/swfm/data/waymo \
WAYMO_OUT=/absolute/path/to/waymo_i2world_2hz_... \
WAYMO_RESUME=1 \
bash tools/real_motion/run_p0_f9_joint_surface_waymo.sh
```

`WAYMO_MAX_WINDOWS=32` 是显式前缀诊断，不是场景均衡正式测试或完整分数；不根据诊断自动改方法。
默认 native 4worker、已验收的并行Strong majority、bounded full-chunk CUDA graphs；每次进程重新验证独立四历史输入、Transport、CCR概率和六帧dense一致。
默认frame LRU为256MiB，只保存不可变输入/评分标签，不是模型特征/预测缓存。
无需改选中模型；首次检查/图捕获的耗时计入本次eval日志，不伪称稳态FPS。

## 看哪些结果

`summary.txt`：Transport 和 Joint 的 IoU、I²-World-reproduction mIoU、标准 mIoU；整体与1/2/3秒。
`waymo_validation.json`：全部18类IoU、整数状态出处、实际时间/补帧审计、输入编码、权重/metadata/实现fingerprint、分段耗时。
`state.json`：可恢复完整窗口的整数混淆矩阵；`progress.jsonl`：运行进度。

官方 `count_miou` 把**恰为0的类别IoU改为NaN**后求mean；因此额外发布标准版本，保留union>0类别的0分，避免误读/抬分。
JSON同时给出官方逐horizon四舍五入后的均值，方便核对其打印表；未四舍五入结果用于分析。
binary IoU是所有非free类的occupied IoU，不是18类semantic mIoU；均不使用visibility mask。
不把nuScenes的MovingMacro/Micro定义硬搬到Waymo，当前协议没有这两项。

2Hz入口不提供静默切换。另有独立10Hz入口，按下述官方index规则执行，不能把0.2/0.4/0.6s标成物理1/2/3s。
本入口不测正式FPS，eval wall time含读盘/历史表示/GT/metrics。正式FPS继续用既有冻结边界。

## 独立10Hz：严格按官方代码的帧下标协议

用户要求10Hz直接照I²-World代码评估。已核对同一固定提交的配置和
[模型 `forward_test`](https://github.com/lzzzzzm/II-World/blob/661d830f9b34ee03ce368db164a72753ab8764a3/mmdet3d/models/ii_world/world_model/ii_world.py)：
`load_interval=1`，`eval_metric='miou'`，分别设置 `eval_time=1/3/5`，预测与GT都取六未来中的**零基下标** `[eval_time]`。
因此是native未来第2/4/6帧，名义0.2/0.4/0.6秒；配置注释中的“1s/2s/3s”不能当物理时距。
输出用 `eval_time_1/3/5` 标识并同时报native step、名义秒数及实际timestamp跨度，不制造物理1/2/3秒对齐。

数据使用同一份metadata/NPZ，不需再下载10Hz数据；检查全部native帧文件，缺失停止、不跳过。
官方全局排序后stride1、场景边界重复有效历史/未来、标签映射、无mask和零IoU排除逻辑保持。
用户当前39987原始帧对应39987个10Hz锚点（最终以实际文件审计为准），约为2Hz人口的5倍。

复用同一冻结均值、四总历史→六预测、ADD raw0.5/REMOVEoff；不训练/调参/改权重。
与官方冻结逐帧模型一样，保留训练时的输出slot语义，不插值、缩放或偷偷改模型时钟；我们的内部0.5秒slot clock/嵌入保持，数据实际是0.1秒帧步。
这是**原始帧下标zero-shot协议复现**，不是声称模型训练过10Hz或物理1/2/3秒准确率公平比较；四总历史预算仍与官方previous/current预算不同。

新adapter/CLI/wrapper独立新增，原2Hz实现文件及fingerprint不变；并发2Hz完成或原目录续评不受本次代码添加影响。
一次六帧预测同时累计三个独立horizon整数计数，等价于同一个确定性预测器的三个eval_time单独评分，但不重复推理三次。
仍只在六帧预测完成后读取未来GT；不持久化learned feature/预测或使用nuScenes缓存。每8完整窗口保存，SIGINT/SIGTERM窗口边界保存，禁止2Hz/10Hz状态混拼。

```bash
conda activate OccFM
cd /root/nas/occ/swfm

# 审计使用独立新输出；不加载模型、不读未来GT。
WAYMO10_AUDIT_ONLY=1 bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz.sh

# 可与已有2Hz并行，同一张卡资源会竞争，不保证总耗时更短。
WAYMO10_OUT="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_$(date +%Y%m%d_%H%M%S)"
nohup env WAYMO10_OUT="$WAYMO10_OUT" WAYMO10_CPU_WORKERS=2 \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz.sh \
  > "$WAYMO10_OUT.log" 2>&1 < /dev/null &
echo "10Hz PID=$!；输出=$WAYMO10_OUT；日志=$WAYMO10_OUT.log"
```

默认2个CPU worker及最多2线程Strong majority，保守留出并行任务资源；显式`WAYMO10_CPU_WORKERS=4`可恢复4线程预算，但不称并发加速。
若有空闲第二张卡，单独设置 `CUDA_VISIBLE_DEVICES=1`（这是进程选择，不是改全局batch或双卡训练）。
中断后使用 `WAYMO10_OUT=/原10Hz输出 WAYMO10_RESUME=1 bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz.sh`；不要传2Hz目录。
完整eval wall time不是FPS；本地测试不是本次L40S/真实Waymo精度验收。

## 本地验收范围

合成metadata/NPZ覆盖全局抽样、边界、映射、无future标签泄露、空/零类别、官方整数计数与四舍五入、中断恢复。
时间跳变回归同时检查锚点/窗口完全不变、实际跨度保留、audit不加载模型或未来GT，以及错频率/错单位/非正时间/倒退帧号仍被拒绝。
小网格真实V18+Surface网络运行六帧并与reference逐字节比较；原Surface/递推相关回归亦运行。
本地没有真实Waymo完整数据或本次CUDA环境，不能将这些测试当作Waymo分数、L40S速度或全仓CI。
