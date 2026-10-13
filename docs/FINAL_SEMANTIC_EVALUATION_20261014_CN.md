# 最终新版模型：冻结评估入口

## 固定模型和范围

仅使用完整 TRAIN20430 ×2 的 Shared-History / STWM / continuous Transport / 18-class Semantic Refiner：

`outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt`，epoch2 / update10292。

STC Pred 使用同一冻结主体绑定的**原始完整10轮 ego 头**（head_epoch10.pt，10轮完整余弦，3200 updates）。拒绝100轮周期的第10轮快照、其他主体、未完成头。未给 HEAD10_CKPT 时按文件契约寻找兼容原始10轮头，不读取验证分数。

新增 `tools/final_semantic_evaluation/`，不修改任何 trainer 或旧实验入口。只读评估；不训练、不改权重、不调整阈值、不自动选择方法。源 checkpoint 在独立输出目录快照；不写历史几何/学习特征缓存。启动可复用/编译既有 native 整数执行库，此构建缓存不是几何缓存。

## 五个任务

| task | 输入与未来 ego | 人口与输出 |
|---|---|---|
| gt | GT 历史4帧，GT未来位姿 | VAL4369；Transport / Refiner，1/2/3s、平均IoU/mIoU/Moving |
| stc | 完整发布STCOcc-Res语义历史；GT / 内部Pred10 | 两行同一实际STC历史覆盖集合；不按外部planner裁剪；完整网格无可见性评分mask |
| long6 | GT真实历史4帧；GT ego through6s；第二段只用第一段最后4张Refiner自预测 | 默认固定GenieDrive公开代码4+20元数据人口2569/150，仅预测12帧，1–6s及4–6s平均；标准指标为主，公开代码零分类别排除/round2单独输出 |
| waymo2 | 映射后的真实Waymo占据历史；GT ego | 原固定全局排序[::5]协议，预计7998 anchors/202 scenes；1/2/3s；原场景边界repeat padding |
| waymo10 | 同上，原固定全局排序[::1] | 预计39987 anchors/202 scenes；eval_time=1/3/5对应0.2/0.4/0.6s，不能标成物理1/2/3s |

原始缓存val4369只用记录身份和入口审计，不用未来标签特征作为模型输入。GT历史观测使用历史mask_lidar；STC历史全网格有效，不读历史GT camera/lidar mask。所有未来标签/Moving支持只在该窗口全部预测完成后读取。

Pred10沿用声明的GT派生目的地导航command，不是navigation-free；预测ego用于Strong背景、Transport和memory的**全部未来几何**，z/tilt沿t0，无事后GT位姿对齐。STC输入分布改变可能降分，不能继承GT或旧STC指标。

## 6s和Waymo不是换路径调用旧神经网络

旧Surface CCR不能加载新Refiner权重。新入口保留纯几何准备、CC/source关联、Strong背景与坐标算子，但神经前向全部使用新模型。

6s严格 `4 -> 6 -> last4 -> 6`。第二段可见性只来自初始真实历史观测union的投影，不读未来sensor mask；source从自预测历史重新提取。**不使用旧Surface static-carry、owner handoff、ADD/REMOVE等试错组件**，所以不能宣称沿用旧static-carry的分数或源身份接续。这是明确的新模型递推路线，结果需重测。默认人口对齐公开代码，论文Table2确切人口尚未独立核验；可用 `--long-alignment legacy --population dev512` 单独作旧完整6s交集诊断，不混入正式表。

Waymo保留原类别映射、原始free标签选择、stride和边界协议。模型仍用训练时0.5s槽时钟，不重新标定或插值10Hz。同时报告标准mIoU和I2公开代码排除零分类别mIoU；不生成没有Waymo标注支持的Moving指标。

## 运行

服务器先 `conda activate OccFM`，在 `/root/nas/occ/swfm`：

```bash
export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
# 如自动发现头失败，明确设置已完成原始10轮文件（不能填100轮的head_epoch10.pt）：
# export HEAD10_CKPT=/实际原始10轮目录/head_epoch10.pt
export STC_ROOT="$PWD/data/stc_camera/compact"
export WAYMO_ROOT="$PWD/data/waymo"
export POPULATION=all
unset FINAL_EVAL_RESUME
STAMP=$(date +%Y%m%d_%H%M%S)
for TASK in gt stc long6 waymo2 waymo10; do
  export EVAL_DIR="$PWD/outputs/p0_f9_shared_history/final_${TASK}_${STAMP}"
  bash tools/final_semantic_evaluation/run.sh "$TASK" || break
done
```

Waymo默认原始free=23，与之前I2-World下载协议一致；仅数据实际是Occ3D free15格式时明确 `export WAYMO_FREE_LABEL=15`，不能据分数选映射。无数据自动下载/解压。

单独任务可直接指定 `EVAL_DIR` 和 task。想短诊断时用 `--max-windows 64`，结果明确标为prefix diagnostic，不作为全集。程序会先做nuScenes一窗同位姿入口审计，固定原容差/零argmax差异/Strong背景一致；失败停止，不自动放宽。确定性作用域仅用于该入口检查，正式推理仍保留原BF16。

Ctrl-C停在已完成窗口边界，state.json保存整数计数；中断窗口不计入结果。恢复同任务、同参数、同源码、同输入：

```bash
export EVAL_DIR=/实际中断的输出目录
export FINAL_EVAL_RESUME=1
bash tools/final_semantic_evaluation/run.sh gt   # 替换为当时task
```

结果：`summary.txt`、`evaluation.json`、`population.json`、`contract.json`、`state.json`；保留checkpoint/data/code指纹及人口缺失列表。目录必须新建，恢复不能改变人口/代码/源权重。质量评估wall time不是FPS，未改计时边界，不继承旧54.9FPS。

## 本地验证边界

27项测试通过（31.10s），包含新入口13项及既有完整GT/冻结头回归：同位姿真实小网格CPU/CUDA前向、内部头前向、未来输入拒绝、指标与旧正式指标整数计数等价、STC无mask读取、两段递推因果顺序、只接受原始完整10轮头、完整窗口恢复及坏计数拒绝、Waymo真实时距标签。CLI help、Bash语法检查通过。没有本地nuScenes/Waymo全集数据，服务器正式分数与用时尚待运行。
