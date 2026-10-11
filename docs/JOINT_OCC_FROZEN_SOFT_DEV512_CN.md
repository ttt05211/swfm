# 冻结 GT 三轮 checkpoint：DEV512 软输出与同窗推理测速

只评已有随机初始化 GT-only 三轮模型，不训练，不修改正式 Surface CCR 平均模型或其 FPS。
默认来源：`outputs/p0_f9_joint_occ_ego/gt3_horizon_float_resume_20261011_151144/last.pt`。
新评估目录保存独立 `source_snapshot.pt`；原权重、Adam、RNG、完整历史缓存只读。

## 质量

精确采用源 checkpoint 的 512 个评估窗口，不做 planner 交集、不补窗口。
固定四历史、六未来，报告 1/2/3s 的标准全网格 mIoU、IoU、MovingMicro。
未来 GT ego 是 GT 配置的条件；未来占据/运动标注仅评分使用，不决定输出。

同窗报告六种读出：hard/soft × Transport、ADD-only、ADD+REMOVE。
hard 保持 ADD0.5 / REMOVE0.95；soft 使用既有训练概率合成并 argmax 得到语义网格，
无重新校准、阈值搜索或自动挑选路线。soft ADD-only 仅将 REMOVE 概率置零。
此读出与此前 DEV64 loss/action audit 完全同口径，不把小样本优势直接当全集结论。

## 测速

完成 DEV512 后，在其中固定场景均衡 32 窗上，测 hard ADD+REMOVE、soft ADD-only、
soft ADD+REMOVE；每窗各路线独立重新前向、生成全部六帧，两个交替顺序计时。
每窗每路线预热不计时；CUDA 同步计真实墙钟耗时，包括完整六帧 CPU 语义输出。

包含：实时重算 Strong、WM、CCR 投影与读出、六帧合成；不计：磁盘读取、历史
实例关联/表示/Surface 描述准备、标签/指标、预热/graph capture、checkpoint 写入。
不把历史缓存中的 Strong 背景省略后冒充实时 Strong 速度。
所有测速输出逐体素验证等于原缓存质量路线六帧，失败保留 FAIL，不放宽。

这是同窗暖推理诊断，不是原始相机端到端 FPS，也不替代旧 256 窗×3 正式速度。
六帧 latency(ms) 与 `6/latency` 输出帧 FPS 同时报；显存 reserved 含公共 allocator/graphs。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export SOFT_OUT="$PWD/outputs/p0_f9_joint_occ_ego/soft_dev512_$(date +%Y%m%d_%H%M%S)"
bash tools/joint_occ_experiments/soft_inference/run.sh
cat "$SOFT_OUT/summary.txt"
```

发回 `summary.txt` 即可，详细整数计数/逐窗测速保存在 `state.json`、`evaluation.json`。
Ctrl-C 在完整质量或测速窗口后保存，不重跑已保存窗口；恢复原目录：

```bash
export SOFT_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/soft_dev512_实际目录
SOFT_RESUME=1 bash tools/joint_occ_experiments/soft_inference/run.sh
cat "$SOFT_OUT/summary.txt"
```

只复制和评估已完成的 GT 三轮模型；若原数据、架构或源码契约改变，会拒绝续评。
不在运行中的训练目录写评估结果，无新特征 bank，无自动重训或方法推广。
