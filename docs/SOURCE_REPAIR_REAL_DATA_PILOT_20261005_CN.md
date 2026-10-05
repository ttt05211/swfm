# 稀疏历史证据修复：真实数据单次验证

这不是正式替换 Local，也不是保证达到 40 FPS。CPU 模拟筛选支持尝试
`local_consensus + cached_future`，但真实 nuScenes 的质量、覆盖率、L40S
延迟必须以本入口实际输出为准。

## 不变的部分

- 使用已选择的 epoch19、严格四历史→六未来；不修改原模型和 optimizer。
- V18 的 XY/yaw/Strong/KTA/A1 renderer、Moving 指标、源顺序保持不变。
- 生成阈值 0.5、旧 refine ADD 0.5、REMOVE 0.95，新 sparse ADD 0.5；不扫阈值。
- 历史注册/关联沿用现有因果协议；已有因果缓存只读，新增几何磁盘配额为 0。
- 输出独立目录，不会覆盖 full/checkpoint_selection/shared_pilot。

## 这次改变什么

四帧注册动态 source 点、历史可见 road/sidewalk 点形成稀疏证据集合，保留
精确注册后的世界坐标。source-local 整数 key 只用于邻域查找，不用量化
坐标替代真实点。添加一圈六面邻居作为明确标记的、**未观测 halo**。

轻量共享点编码器输入位置/类别/四帧出现与可见性/邻域一致性，以及原 V18
history context 和六个 future queries。点编码一次，以小 MLP 产生六个 ADD
概率；继承类别，按原预测 SE(2) 和完整 future ego SE(3) 投影到六帧。

只向原 transport 的 free voxel 写入；保护原 occupied voxel。不同静态
语义的 canonical/raster 冲突 fail-closed；静态先写、动态 source 按原顺序写。
动态已有 t0 点由 V18 负责，不让 repair 重复写入。**不产生从未观测的动态
source，不实现 REMOVE**，不声称旧 KEEP/ADD/REMOVE 的等价替换。

生成分支仍使用旧网络，写在修复之后，仅向仍 free 的地方写入。FPS 单独
比较 GEN-only 分批和旧完整分批在全部六帧的动作是否完全一致；不一致就
拒绝该速度结果，不能把改变预测算成提速。

## 一趟完成的工作

1. dev64 一次旧 probability pass，拆出静态 ADD、动态 ADD、REMOVE、generation、joint。
2. 同时报告新候选域 GT oracle、原正确 ADD 的静态/动态覆盖量；oracle 只用于诊断。
3. 少量真实联合反传测速：原路径与 ADD-only 新路径使用相同窗口组，旧模型不落盘更新。
   新路径暂不训练 GEN/REMOVE，所以这是**结构成本诊断，不是完整等价训练加速比**。
4. 固定 TRAIN20%（约 4086 windows）一遍迁移，冻结 epoch19 motion，仅训练轻量头。
   GT BCE + 0.25×选中 query 的旧 ADD 概率蒸馏；静态/动态分层采样通过逆概率权重
   恢复自然候选分布。未来 GT 不进入证据、候选、features 或推理。
5. final dev64 同时评估旧 teacher、新 static/dynamic/refine/joint。默认不跑 dev512/full4369。
6. 训练后测旧/新 refine-only 和旧/新 joint 的六帧延迟，均包含 fresh prior、live motion、
   证据构建/映射/六个 dense 输出；排除 I/O、源提取/注册、GT、metrics、graph warmup。

新方法没有减少历史帧、只输出三帧、缓存 learned encoding、只计 MLP 时间等捷径。
prepared-input FPS 仍不是原始传感器端到端 FPS。迁移耗时包含 teacher KD，不能拿
它直接推算未来不带 teacher 的完整联合训练速度。

## 服务器运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_source_repair_pilot.sh
```

脚本用已确认的 epoch19/cache/manifest/nuScenes 路径，不依赖前次 shell export。
默认 TRAIN20% 一遍、dev64、6 个 FPS 窗口×两遍；最后直接打印 `summary.txt`。
可用 `SOURCE_REPAIR_SMOKE=1` 单次小额自检（2 更新/2 dev 窗口）；无需先手动跑 smoke。
`SOURCE_REPAIR_DEV512=1` 是显式追加 dev512 的选项，不是独立测试。

Ctrl-C/SIGTERM：完成当前更新后保存新协议 `migration_last.pt`。续训必须保持全部
数据、实现 fingerprint、训练预算/顺序、优化器和 RNG 合同相同，且用新输出目录：

```bash
SOURCE_REPAIR_RESUME=/实际本次输出目录/migration_last.pt \
  bash tools/real_motion/run_p0_f9_source_repair_pilot.sh
```

上面 `/实际本次输出目录` 是需要填入日志中的目录，不是预设服务器路径。
`kill -9`/OOM 只恢复最后一个已完成的周期断点（每 32 更新），不保存半步 optimizer。
完成后有 `bundle.json`、`progress.jsonl`、`summary.txt`、teacher snapshot 和新的头断点。
head checkpoint 明确 `deployable=false`，不能作为旧 Local checkpoint 加载。

门槛要求 joint mIoU 和整体/1/2/3s MovingMicro 不低于旧 teacher，完整六帧延迟
不超过 150ms。未通过不自动再训、改阈值或扩到 full；以结果决定修复域还是编码器
瓶颈，不能用小样本 oracle 冒充真实预测。
