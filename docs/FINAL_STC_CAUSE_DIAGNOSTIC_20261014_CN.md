# 新版GT / STC / 内部Pred：一次集中只读诊断

## 固定范围

仅完整GT epoch2/update10292 SharedEncoder/STWM/SemanticRefiner + 原始完整10轮ego头。固定原DEV512内场景均衡64窗，STC缺窗直接拒绝，不换人口。新目录 `tools/final_semantic_diagnostics/`，**不改final_semantic_evaluation、旧trainer或网络文件**，因此不破坏当前long6/Waymo评估源码契约。

没有optimizer、重训、epsilon/阈值/轴方向扫描选优，也没有外部planner、未来GT位姿对齐或GT mask插入Camera预测。

## 六条读出

每条完整预测六帧；共同报告1/2/3s Transport和Refiner的标准全网格IoU/mIoU/Moving。

| 名称 | 历史语义 / known | 未来ego | 用途 |
|---|---|---|---|
| occ_gt | GT / 原历史lidar known | GT | 原GT基线 |
| occ_all_known_DIAGNOSTIC | 同一GT语义 / 全True | GT | 单独隔离known分布，不当STC改法 |
| occ_internal | GT / 原known | 原10轮内部头 | 同窗OCC Pred基线 |
| stc_gt | STC / 全True | GT | 原STC GT基线 |
| stc_internal | 同一STC / 全True | 原10轮内部头 | 原STC Pred基线 |
| stc_GTscene_ego_DIAGNOSTIC | 同一STC / 全True | 头读取GT历史所得预测pose | oracle历史诊断：隔离head场景输入影响，不能作为Camera方法指标 |

额外只跑轻量头：同GT语义、全True known下的ego误差，区分语义质量和known策略对头的影响，不增加第七条dense场景预测。GT/全True/STC三条件的历史运动学prior必须完全一致，否则停止。

未来GT语义与Moving annotations仅在上述六条预测全部完成后读取。GT未来位姿是GT行显式条件/Pred误差监督，GT派生导航命令仍公开声明。诊断不会把future GT语义/annotations/mask传给预测器。

## 四组检查与解释

1. **输入身份与感知质量**：历史+未来token同场景/连续/时距，compact逐帧SHA与提取receipt核对；有原ZIP时直接核CRC和语义字节。t0全网格混淆矩阵、每类IoU、错/漏占据、precision/recall，以及历史known内/外分开统计。固定八种XY方向和原GT前/当前/后帧只在诊断指标中比较，绝不修改预测输入。前后帧比较没有pose对齐，可能反映运动，不能单独证明错帧；非identity方向得分高也不能自动证明轴bug或选作正式改法。
2. **known分布**：只把GT历史known改全True，语义、权重、ego不变，观察预测与场景特征/头误差变化。GT训练encoder使用历史mask_lidar，STC使用全True；这是需要明确审计的输入差异。Transport静态prior读t0语义，不按known过滤；encoder/memory会使用known。没有将GT可见性送入正式STC。
3. **Refiner与先验**：每时距新增/删除/重标、改对/损坏/wrong-to-other-wrong；地面、建筑植被、动态语义及Moving支持分组。按wrong-free/false-occupancy/wrong-class分别累计先验胜者相对真实类的logit barrier、残差对真实类的相对优势及最终margin，报告精确均值/整数直方图。`ΔZ_GT-ΔZ_old > L0_old-L0_GT` 仅是超过旧胜者的必要条件，第三类仍可能赢。epsilon保持1e-4；约9.21是一热先验处的算术差，不是所有体素实际差，也不是校准后的感知置信度。净改对数量不是class-balanced mIoU增益。
4. **ego场景输入敏感性**：同历史ego/同导航/同头的GT known、GT all-known、STC与pure kinematic prior轨迹误差；头相对prior修正大小、世界XY/z/tilt/完整旋转误差；同STC场景改用GT-history-head预测pose的单独反事实读出。另记录source数量/≥3帧历史注册、速度、相对线性运动的偏差，后者包含真实加速转弯，不把它等同纯感知噪声。特征channel mean/std/RMS只是描述统计，不能单独证明域偏移。

输出evidence_summary精确列出Refiner增益、mask单变量差、头FDE差和轨迹反事实差；**不自动给“根因已证实”结论**。拿结果后再区分坐标/数据错误、known依赖、感知底稿错误、残差纠错不足和ego域偏移，不靠重复失败的随机500步实验或改参数猜分。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export STC_ROOT="$PWD/data/stc_camera/compact"
export DIAG_DIR="$PWD/outputs/p0_f9_shared_history/stc_cause64_$(date +%Y%m%d_%H%M%S)"
export CPU_WORKERS=3
unset STC_DIAG_RESUME

bash tools/final_semantic_diagnostics/run.sh
```

默认按契约自动找原始10轮头；需要显式时 `HEAD10_CKPT=/实际原始10轮目录/head_epoch10.pt`。自动使用既有 `data/stc_camera/stc-results.zip` 做独立原ZIP检查；原ZIP另存可 `STC_ZIP=/实际路径`，指定路径不存在就拒绝，未指定且默认不存在则明确只核receipt，不下载。ZIP与receipt SHA不一致停止，不重解压或覆盖。

结果：`summary.txt`、`evaluation.json`、`state.json`（逐窗原始计数/分布/身份）、`population.json`、`contract.json`、checkpoint快照。CPU保留本窗口被评分时距的logits供诊断，约1.6GiB临时host额外内存，不持久保存大预测体；时间不是FPS。共六路且有CPU统计，比单模型评估慢；不承诺几分钟或总RSS等于cache预算。建议已有任务资源有余量再启动。

Ctrl-C只保存完整诊断窗。同目录、同代码/输入/权重：

```bash
export DIAG_DIR=/原诊断目录
export STC_DIAG_RESUME=1
bash tools/final_semantic_diagnostics/run.sh
```

源码/人口/数据改变拒绝恢复。源checkpoint、旧实验目录与缓存不改；没有自动训练、选择或推广。运行后只需发 `summary.txt`，必要时再提取evaluation/state中特定字段，不必贴全部大JSON。

## 当前证据与本地验证边界

用户已报告完整4369 GT Refiner47.991901/61.519695，STC GT18.320217/27.243256，STC Pred12.496515/21.109608（mIoU/IoU）。Refiner相对Transport增益分别+4.110228/+6.805043、+.321334/+.406372、+.268605/+.228408。这证明收益随条件变化，不单独证明接口错或epsilon是根因。当前正式原模型/分数保持；诊断结果没有生成前不能保证解决。

本地测试使用真实小网格CPU/CUDA整条六路前向和真实Moving转换；验证不改权重/无grad、future标签读取次序、原ZIP/compact语义比对、编辑/margin精确统计、完整窗续评和坏计数/身份拒绝。CLI用模拟元数据测试固定64窗、源快照只读和续评，非真实nuScenes质量结论。

新诊断7项＋正式评估接口23项，共30 passed（最终23.76s）；CLI help与Bash语法检查通过。没有本地nuScenes原始数据，不宣称已经确认服务器掉分根因或任何改善方案有效。
