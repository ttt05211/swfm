# Local 单卡训练/评估加速包

## 不变的科学与恢复合同

四历史 → 六未来、完整 TRAIN20430、batch4/source128、全部候选与顺序、采样 RNG、模型参数形状、loss 权重/归一化、importance 权重、TRAIN 阈值网格、冻结 Moving 指标不变。
不删候选、不跳窗口、不增加跨更新 learned feature/pose cache、不把未来 GT 放入输入。CPU 监督索引仅用于 loss。
单卡普通 resume 仍恢复 Adam、RNG、batch cursor 和 LR；从已完成 15 轮追加到 20 轮走现有显式 extend，原 15 轮结果只读，不重置 optimizer/LR。追加五轮使用已公布末端 LR 的余弦下降，不伪装成从头的整体 20 轮余弦训练。

## 本次合并的优化

- TRAIN：在已有 CPU 标签上产生排序监督索引，避免 GPU boolean indexing 的动态形状/host 同步；保留同一 Smooth-L1、BCE、periodic yaw 与 SE(2) overlap 求和顺序。column loss 的合法性与采样权重在 upload 前检查，保留 GPU 有限性检查。只缓存有界的常量 footprint 坐标画布。
- 完整候选：稀疏静态区域通过原 fixed causal `static_xy` 列表直接扫描，编译整数内核访问 ROI；密集区域、早期 cache 缺少该字段时保持原 full scan。没有截断候选或改变排序。C++ 源码指纹更新会产生新二进制，不覆盖旧库，也不要求重建几何缓存。
- 推理：仅当前窗口复用 history/observed byte 输入，仅当前 horizon 复用原 CPU FP64 actor 变换与 membership；128-query byte tiles 减少 kernel launches，**网络仍为原 batch256**。字节采样内消除重复读取，但不去重网络 query。
- 数值边界：仍用原 floor 的 FP64 guard。边界/预算/OOM 回退到完整原 horizon 的 CPU map，不对较小子集重算 BLAS。first3/every128 窗口校验 CPU/GPU byte 一致；失败直接报错。
- Eval/calibration：三个 report horizon 的 candidate 构造并行、共享一个 window history index；四路 ablation 共用一次模型概率。独立评估预取下一窗口的 immutable CPU 几何，首窗口原 renderer all-six exactness 保护继续生效，不启用 persistent disk writer。
- E14：同一 provider/dataset/window 只复用冻结 E14 的小型指标计数，最多 1024 个窗口；不缓存改变中的 joint/control 预测。paired control 每次仍重算。仅作参考的 renderer 不再分配 owner/fallback 密集层。
- 独立 evaluator 新增显式 `full4369`：必须是 4369 个唯一 key，包含冻结 dev512，且与 TRAIN scene-disjoint；绝不取 cache 前 512 条替代 selection population。

## 一次合并测速

服务器上先安全停止训练。不要 kill -9；不需要重建 prototype、prior 或几何缓存。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD

RUN15=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_history4_resume_20261004_092659_838
bash tools/real_motion/run_p0_f9_joint_speed_bundle.sh "$RUN15"
```

这一趟完成 TRAIN64 的旧/新实际 forward/backward/Adam 耗时，以及 dev8 完整候选四路评估、真实 CUDA 概率逐元素一致性与耗时。每组从同一 immutable snapshot、同一 Adam 状态与随机种子开始，任何实际采样窗 warm miss 都报错，不写共享几何 cache。诊断更新只在内存中，不保存科学训练 checkpoint。

输出新的 `speed_bundle_*/summary.txt` 和 `audit.json`。训练与评估都会列 seconds/window 与 speedup，并记录分段/byte fallback 信息。
kernel 编译、数据加载、首次 renderer/allocator 预热与直接概率对照不计入稳态吞吐。顺序/OS frame-cache 影响仍存在，不是严格 GPU 活跃时间或论文 FLOPs 比较。
TRAIN64/dev8 是小样本速度诊断；推算一轮/full4369 时间须带 `if_representative`，**没有真实服务器测量之前不宣称提速多少倍**。

## 延长训练与随时评估

一致性检查通过、测量结果可接受后，用同一个已完成的 15 轮目录追加，脚本会新建输出，不覆盖原结果：

```bash
bash tools/real_motion/run_p0_f9_joint_extend.sh "$RUN15" 20 0
```

需要中断时使用 Ctrl-C 或 manager `stop --run-dir 当前追加目录`，等待 `STOPPED safely`。普通恢复用新追加目录，不再对 RUN15 重复 extend：

```bash
PY="$(command -v python)"
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py resume \
  --run-dir "$CURRENT_RUN" --gpus 0 --column-feature-backend gpu \
  --sampling-workers 6 --profile-every 32
```

`CURRENT_RUN` 必须是本次脚本实际打印的追加训练目录，不推测时间戳。
权重/结构/阈值冻结后才跑完整评估；中途通常用 dev64/dev512：

```bash
bash tools/real_motion/run_p0_f9_joint_interim_eval.sh \
  "$CURRENT_RUN" full4369 "$CURRENT_RUN/model/candidate.pt"
```

`candidate.pt` 沿用已存 TRAIN 校准阈值；last/epoch checkpoint 用固定 0.5/0.5/REMOVE-off。评估先复制 immutable snapshot，即使原 last.pt 后续原子轮换也不会误报“checkpoint changed”。不修改 optimizer/RNG/源文件。

## 回退与验证边界

`SWFM_LOCAL_FAST_SUPERVISION=0`、`SWFM_LOCAL_STATIC_ROI=0` 可分别关闭两项 CPU fastpath；独立 evaluator 的 `--column-feature-backend cpu` 或包装脚本 `FULL_JOINT_EVAL_FEATURE_BACKEND=cpu` 保留 CPU 特征后端。默认不切换 batch/source，不启动双卡，不自动重训/重试/提升部署版本。

本地测试覆盖四/六帧、监督空集、同 loss/gradient/三次 Adam/RNG、完整候选字段/顺序、byte sampling、boundary/budget、四路 metrics、E14 count reuse 与 live paired-control、因果预取、完整 population，以及已有 stop/resume/extend 回归。实际 CUDA 测试仅在有 CUDA 的服务器上运行；CPU 模拟不能代替该验证。
