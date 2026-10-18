# 最终结构候选：Integrated 2+2 / 完整 TRAIN ×2

## 用户明确决定

4+4同预算一轮没有显示容量收益：参数1194146→2191042，OCC最终mIoU46.546783→46.358199、STC17.075513→17.053976，Moving略退。用户同意收口2+2并完整训练两轮。该短训结果不证明更深在任何预算下都无用，也不按DEV自动挑最好epoch。

本次固定最新 Integrated：共享历史Encoder、Scene–Temporal×2、Future Query Decoder×2、KTA Query+skip、无ICP/track_valid、学习历史相对yaw、direct-history feature memory、三块AdaGN Refiner。正式配置1194146参数。不是早期Stage2 no-query或旧独立global-attention Reader，更不是clean1000/norm192/Surface。

## 固定训练协议

- **全 TRAIN20430 窗 ×2轮，从随机初始化开始**。各轮每窗一次；原seed20261011、round-robin人口和原分组顺序。
- 全球B4/source128 grouping，源不截断，H6完整六时距。
- 仅全网格CE+Lovász；无ego/motion/existence/alignment loss。
- 只用原TRAIN的GT历史/未来目标，无真实STC训练、模拟污染、先验系数修改、posterior校准、norm192或旧头。
- TRAIN、OCC/STC评估均dense-valid；GT future ego显式条件；无内部ego或导航输入。
- 默认双卡0/1训练；仍是global B4，不翻倍batch。每次评估只rank0单GPU，其他rank等待。
- 原AdamW/LR（motion5e-4、repair3e-4）、原梯度SUM/裁剪、BF16与FP64几何、**完整两轮一次余弦**，不在轮边界重启。

新的 `tools/source_scene_integrated_full` 与旧入口隔离。直接调用原Integrated新鲜初始化工厂，逐参数及初始化后CPU RNG与原2+2相同；从未加载其pilot权重。旧E14参数文件仅提供原架构/缓存契约，未导入训练参数。旧历史几何cache require只读，新identity_history/rank*属于新输出。

## 保存和自动评估

每2000个GLOBAL update及epoch结束保存、评同一remaining64：OCC/STC、Transport/Refiner，full-grid标准1/2/3秒及Moving。延续原query评估，还保留同模型KTA_future和条件GT框运动诊断；不是训练额外损失，也不用于选择/自动扩训。

原分组通常两轮10292步，但以实际全量cache分组为准，checkpoint记录epochs_updates。每2000步的名字为 `step_0002000.pt` 等；轮末 `epoch_0001.pt`、`epoch_0002.pt`；`last.pt` 保存完整update边界/Adam/各rank RNG/epoch/batch和完成评估。独立整数评估状态支持完整窗口恢复，评估期间保持训练RNG。

每次评估结束立即在日志打印四行标准分数，JSON保留全部诊断；`summary.txt` 汇总所有已完成评估，不必等训练全部结束。没有best保存/自动promotion/第三轮。DEV64反复检查过，只是开发人口，正式全集评估以后另跑。

## 启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export INTEGRATED_FULL_RESUME=0
export INTEGRATED_FULL_DIR="$PWD/outputs/p0_f9_shared_history/scene_integrated22_full_2ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_integrated_full/run.sh > "${INTEGRATED_FULL_DIR}.log" 2>&1 &
echo "$INTEGRATED_FULL_DIR"
tail -F "${INTEGRATED_FULL_DIR}.log"
```

这是后台启动，记下输出目录。不要同时另开4+4训练/测速抢同一GPU。

默认评估参考路径沿用先前clean/results.json，**仅读取contract.eval_keys这一份固定64窗ID**，不读其分数或权重，不使用历史模型。E14也仅架构/缓存契约，训练参数全随机。数据、keys、源码、输入STC库存、运行世界大小与日程全部绑定到新contract。

## 续训与查看

新终端需要重新设置 `INTEGRATED_FULL_DIR` 为原目录，不能重新生成时间戳：

```bash
export SCENE_TRAIN_GPUS=0,1
export INTEGRATED_FULL_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/这里填写原目录"
export INTEGRATED_FULL_RESUME=1
nohup bash tools/source_scene_integrated_full/run.sh >> "${INTEGRATED_FULL_DIR}.log" 2>&1 &
tail -F "${INTEGRATED_FULL_DIR}.log"
```

仅在原进程已退出后执行续训，不能同时双写。同目录原world/data/code/预算恢复，不将旧pilot/旧Stage2/4+4断点静默转换。已完成的update/评估不重复，完成两轮后resume不再训练。

随时查看：`cat "$INTEGRATED_FULL_DIR/summary.txt"`。单点评估文件 `eval_step_0002000.json` 等，整数状态在 `eval_step_0002000_state.json`；日志有`INTEGRATED_FULL`进度和`INTEGRATED_FULL EVAL`结果。

## 验证范围

专项覆盖20430×2各轮完整唯一人口、固定same64、全周期余弦及跨轮cursor、2000步/epoch保存条件、实际CUDA小网格跨轮停续/单次评估/完成后零重复、新loader协议/结构保护、与原2+2新鲜初始化/RNG严格一致。旧模型/训练/测速/缓存源码保持不变。

最终35项新旧回归通过（26.62秒），包括actual CUDA小网格两轮停续、LR跨轮连续、结束后零重复、即时结果打印与最新model语义保护；CLI help、Bash语法检查通过，旧相关代码零diff。

本地验证不代替L40S全尺寸速度/显存和全量质量，不继承旧分数/FPS。当前没有服务器执行连接，未代启动训练；以上命令由用户服务器执行。
