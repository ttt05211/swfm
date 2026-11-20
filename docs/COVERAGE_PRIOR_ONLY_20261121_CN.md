# 覆盖信息与有效 Prior：关闭投影监督的同预算对照

本实验回答：上一轮联合方案下降，是否与新增投影监督有关。只新增一个训练组，不改变现有主配置。

## 三组比较

| 显示名称 | 覆盖输入及有效 Prior 融合 | 投影监督 | 本次操作 |
|---|---|---|---|
| Original | 关闭 | 关闭 | 复用上一轮已完成的同64窗原版结果 |
| Projection ON | 开启 | 开启 | 复用上一轮已完成的同64窗联合版结果 |
| Projection OFF | 开启 | 关闭 | 新随机训练4086窗一轮，再评同64窗 |

Projection OFF 与 ON 的架构、初始化、样本顺序、双卡分工、优化器、学习率与1031步预算一致。OFF 不构建投影器、不执行投影渲染/损失；不是把投影算完后乘零。两者比较用于检查这项训练监督的影响，不能预先认定它是退化原因。Original 与 OFF 的比较还包含覆盖输入/融合方式，以及原控制单卡、新组双卡的执行差异。

## 固定设置

- 最新 causal-forward、FPN、Scene-Temporal2、AdaLN-Zero Decoder2、KTA query/skip、无ICP、直接历史yaw Memory、原宽度U-Net。
- 覆盖结构与上一轮完全相同：Prior/Memory两个覆盖通道；未知Prior输入中性；输出为 `coverage * log(Prior) + U-Net logits`，完整网格仍可预测。2816554参数。
- RANDOM TRAIN4086×1；1031个GLOBAL更新；GLOBAL B4/source128；seed20261011。直接调用上一轮相同的fresh模型工厂，不载入原版或联合版的已训练参数。
- 原六未来CE+Lovasz；AdamW两组LR5e-4/3e-4；一轮余弦floor0.01；全局均值梯度SUM后统一clip5与Adam。较小批次可有空rank，不重复填充窗口。
- 双卡训练、单卡评估；dense-valid；GT future ego同时供Decoder和renderer。FREE/age/校准关闭，STC仅评估。
- 单独协议 `forward_coverage_prior_ONLY_RANDOM_TRAIN4086x1_dual_v1`；新目录，不能从联合版权重续训。

参考联合实验默认目录：

```
/root/nas/occ/swfm/outputs/p0_f9_shared_history/coverage_projection_dual_fix_20261120_190540
```

原控制从该目录的contract追溯。CPU预检联合组的完整checkpoint、原控制训练完成游标、原人口/预算、代码指纹和已完成OCC/STC64整数混淆矩阵；旧C_train的历史报告可仍为running，不把其旧评估状态误当训练完成状态。旧结果、权重、模型源文件均保持原样。

## 启动

服务器在仓库根目录、OccFM环境：

```bash
export PILOT_GPUS=0,1
export PRIOR_ONLY_DIR="$PWD/outputs/p0_f9_shared_history/coverage_prior_only_20pct_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/coverage_prior_only/workflow.sh > "${PRIOR_ONLY_DIR}.log" 2>&1 &
echo "$PRIOR_ONLY_DIR"
tail -n 20 -f "${PRIOR_ONLY_DIR}.log"
```

如联合实验目录不同，先设置 `JOINT_COVERAGE_DIR`。脚本不自动重试、加轮或选择DEV最佳权重。两张训练卡退出后，只用首卡评新组同64窗OCC/STC；可用 `PILOT_EVAL_GPU` 改评估卡。

每64步保存完整边界的Adam、每rank RNG及全局游标；中断后指定原来的 `PRIOR_ONLY_DIR`，再运行：

```bash
PRIOR_ONLY_RESUME=1 nohup bash tools/coverage_prior_only/workflow.sh > "${PRIOR_ONLY_DIR}.resume.log" 2>&1 &
```

若训练已完成，仅续评/导图：

```bash
nohup bash tools/coverage_prior_only/eval_only.sh > "${PRIOR_ONLY_DIR}.eval.log" 2>&1 &
```

完成训练后重进workflow保留原epoch文件，不重写其字节指纹。评估按完整窗口恢复整数计数，STC使用自己的source数量，不受GT TRAIN source128分组字段约束。

## 输出与验证范围

`summary.txt` / `evaluation.json` 比较三组的Transport和Refiner，仅报告mIoU、IoU；评分仍为全网格1/2/3s及其平均。原版与联合版不重跑推理，新组完整六未来预测后才读取评估标签。

`visuals/index.html` 固定同64前8窗、OCC/STC各8张，四行GT/Original/Projection ON/Projection OFF，四列t0/1/2/3s。复用原包GT/原版/联合版数组并核验同窗同位姿，OccWorld/COME色表。服务器生成BEV预览，原始uint8体素也保留，可在本机做COME三维渲染。下载 `visuals_and_metrics.zip` 即可，不含代码、权重、训练数据或大缓存。

历史标签缓存128MiB、一个批次预取；不产生dense tensor磁盘缓存。训练只保留last/previous/epoch快照和小日志。

测试覆盖：相同fresh权重及初始化后RNG；投影被禁止调用时完整六帧CE+Lovasz反向与Adam/RNG恢复；双进程CPU/Gloo的B4/B3/B1空rank和恢复；真实评估准备GT/STC source数不一致、逐窗断点恢复、三组图册及原结果只读；旧覆盖/评估/诊断回归。完整4086窗实验由服务器运行，本地执行检查不构成新质量分数。

本地43项新旧回归通过，两个shell入口语法检查通过。真实TRAIN scene-0429一窗、原生 `6×200×200×16` 六未来CE+Lovasz反向与Adam更新通过；在强制禁止构建/调用投影器的情况下，Refiner head权重发生有限更新，峰值分配约5035MiB。该检查使用随机网络，无质量评分、无已训练参数导入。98个既有实现文件与远端比较一致，其中两份Windows检出文件仅换行符不同；不发布或改写它们。
