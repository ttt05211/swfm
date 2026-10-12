# 单路 REMOVE：20% TRAIN 两轮、逐轮 DEV512（2026-10-12）

用户明确不重训已完成 ADD-only。新入口只训练一次独立 RANDOM GT ADD+REMOVE：固定 TRAIN4086×2、seed20261011、batch4/source128、完整六未来 CE+Lovasz/motion/existence/ego、整个两轮余弦、sequential H6。不开启此前暂停的其他结构/提速实验。

## 自动流程

1. 自动查找唯一已完成 `outputs/p0_f9_shared_history/shared_20pct_2ep_*` 的 Shared ADD-only 两轮结果及 epoch_0001.pt。多个匹配不按分数挑选，需显式 ADD_ONLY_DIR。
2. 开始新 ADD+REMOVE，达到精确第1轮更新边界安全保存。
3. 对第1轮新快照 DEV512：Transport、同模型 REMOVE-off、ADD+REMOVE，权重/Adam/RNG只读。
4. 旧 ADD-only 第1轮快照只读评一次同 DEV512，结果写新工作目录，绝不恢复旧训练。旧第2轮结果直接复用。
5. 同一新 last.pt 恢复原两轮周期，训完第2轮并自动 DEV512，同样三路。
6. comparison.json/summary.txt 分别给出 epoch1/2 相对旧独立 ADD-only 的差值、同一个新模型启用 REMOVE 的增量。

新/旧 TRAIN/EVAL keys、数据SHA、预算、学习率、损失、namespace/执行精度核验；旧初始化/网络输入/输出头/H2与新H6不同，结果不是严格单变量消融，不把所有差异归因删除。旧两轮基线mIoU40.553310、IoU52.881533、Moving28.035021仅当精确来源契约匹配时使用。不会复用update1041等中期结果冒充第1轮末结果。

## 启动

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git switch feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/remove_validation_$(date +%Y%m%d_%H%M%S)
bash tools/shared_history_experiments/actions/validation/run.sh
cat "$RUN_DIR/summary.txt"
```

若旧结果存在多个候选，指定启动提示的正确旧实验目录：

```bash
export ADD_ONLY_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际旧shared_20pct_2ep目录
bash tools/shared_history_experiments/actions/validation/run.sh
```

仅输入/路径预检失败的空workflow目录可能已创建；使用新的RUN_DIR，勿删除历史目录。不要在正在训练的旧代码目录上切换运行源码。

## 恢复

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际remove_validation目录
VALIDATION_RESUME=1 bash tools/shared_history_experiments/actions/validation/run.sh
cat "$RUN_DIR/summary.txt"
```

Ctrl-C/SIGTERM向当前child请求安全边界保存。第1轮训练/评估中断恢复后先补完该阶段，再训第2轮；不跳过评估、不重新随机初始化、不把续训当新一轮余弦。kill -9只恢复最近发布的训练/整数评估断点。若刚完成训练且评估未结束，恢复只评剩余，不再执行训练。workflow/来源/实现变更拒绝恢复。

目录：`add_remove/last.pt` 与 epoch快照；`add_remove/eval_epoch1`；`add_remove/results.json`（最终第2轮）；`baseline_epoch1`（旧快照只读评估）；根 comparison.json/summary.txt。旧源目录不写入，新评估不覆盖新训练last.pt。

## 验收范围

控制器测试覆盖只允许一个新训练arm、两轮中间评估顺序、同周期恢复、旧第2轮直接复用、旧目录/epoch快照不变、评估中断恢复、完成后不重跑、错误人口/预算预检。原双动作真实小网格删除梯度、六帧fast/reference一致、Adam/RNG恢复与CLI测试保持通过。它们不是服务器nuScenes精度结论，REMOVE有效性由本次实际结果判断，不自动扩大训练/挑best/调阈值。
