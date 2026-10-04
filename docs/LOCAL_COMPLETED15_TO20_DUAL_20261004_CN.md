# Local 已完成15轮 → 追加至20轮（显式新实验）

不能把旧 `--epochs 15` 断点直接改成20：前15轮已经按15轮余弦学习率优化，
并不是从头按20轮计划训练。新增 `manage_p0_f9_joint_training.py extend` 是明确的
新续训实验，另存输出，原15轮文件只读；普通 `resume` 仍禁止更改科学配方。

## 训练和学习率

- 必须使用完整15轮 `last.pt`（含AdamW moments、RNG、数据游标）；不能用 `candidate.pt` 或 epoch 权重。
- 每轮仍完整20430窗口，严格4历史→6未来，source/label/loss/renderer不变。
- 新5轮从原断点最后使用的两组LR开始，分别余弦下降至其10%，不恢复初始LR，不固定LR尾训。
  当前原周期floor为初始LR的10%，所以续段起始通常约 motion=5e-5、columns=3e-5，
  末端约5e-6、3e-6。准确起始值取checkpoint，不硬编码。
- 使用独立 extension protocol/continuation provenance；日志明确不是whole-20-from-start。
- 原epoch history保留，继续监控16–20，每轮固定0.5/0.5/REMOVE-off dev64。
- 追加轮次16–20的权重快照全部保留，仍只有last.pt含完整断点状态。
- 最终按相同TRAIN-only流程校准并评估dev512；最终阈值可能变化，不能把所有联合增益变化都归因于权重训练。
- 不根据dev选best、不自动续至25、不自动full4369、不自动部署通过。

## 双卡实现与边界

显式 `--gpus 0,1` 使用torchrun两进程和NCCL同步数据并行。
因为一次迭代包括motion→硬CPU几何→linked columns，代码采用显式梯度同步，
不是把一个普通forward直接包在DDP中。

- GLOBAL window batch=4、source budget=128不变，不是每卡batch4。
- 先按原全局source预算组batch，再分window给rank0/rank1；无DistributedSampler补齐、重复、drop-last。
  超大source单窗口保留完整，一个rank会得到空分片，但仍参与同一global更新。
- 四项motion loss按全局有效label分母归一化；两类column loss按全局importance/legal分母归一化，
  再对全局存在的task平均。梯度SUM后按原5/1范数裁剪，再AdamW。不是平均两个局部mean。
- 全局未使用的参数保持grad=None，避免错误weight decay；无标签/无query的全局step不触发AdamW。
- CPU候选和GPU历史采样分卡并行，每rank默认6个sampling worker、2个IO worker；适用于用户确认的20核/160GB/2×L40S。
- 暖几何缓存继续复用；两个rank只读取/有界补写因果几何，不缓存learned poses/features/GT。
- 首次1→2迁移保留rank0原采样RNG，rank1用确定性jumped流；后续保存两rank的采样/Torch/CUDA/Python/NumPy RNG。
  浮点归约顺序、每rank随机流改变，不能承诺单卡与双卡逐位相同。
- 只rank0写checkpoint和日志，epoch/final验证也只rank0运行。安全停止会集体完成当前global步后保存原子last.pt。
  `kill -9`只能恢复最后已发布的周期断点，不保证恢复尚未保存的更新。
- 后续resume必须仍两卡；不允许静默迁回单卡。不支持两卡paired-control或fresh random-init。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_joint_extend.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_resume_20261004_092659_838 \
  20 0,1
```

包装脚本先检查native CPU后端并跑包括两张实际GPU的NCCL数值/RNG测试，失败则不启动正式续训。
本地Windows CPU测试不能代替服务器双L40S NCCL/真实数据吞吐验证；不承诺2×速度。

停止/恢复（将实际新输出目录替换到 `RUN`，不重新用extend）：

```bash
PY="$(command -v python)"
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py stop --run-dir "$RUN"
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py resume --run-dir "$RUN" --gpus 0,1
```

恢复仍另存新输出，完整保留optimizer、rank RNG、游标及续段余弦位置。
