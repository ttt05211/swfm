# 共享历史证据：一次运行的有限迁移实验

## 范围与保护

此入口实现已确认方案的**原型/迁移验证**，不是已经达到40 FPS的新正式方法。
旧 epoch19、原训练入口、checkpoint、阈值、数据缓存均保持不变。新 protocol 为
`p0_f9_native_full_z_shared_evidence_pilot_v1`，旧 `load_joint()` 不将其识别为可部署模型。
严格四历史、六未来，固定0.5/0.5/0.95；不跑full4369、不调阈值、不自动选模型/重试/扩大训练。

## 一次运行包含

1. 相同dev64人口：原196-token teacher、原权重36-token探针、未训练共享结构。
2. CPU/device的全部六帧baseline、ownership/fallback、完整候选顺序/合法域检查；首两窗口完整全Z采样检查；报告帧teacher最终逐voxel检查。
3. 同批次真实联合forward/backward/optimizer：current、device_geometry、shared auto/dense/tile，正反序两遍。此测速主干不冻结、不保存科学更新，固定因果准备之外的完整在线几何/编码都计时。
4. 真正全部六帧生成FPS：每次重建Strong/KTA prior、实时motion、候选、历史编码、生成/refine和dense合成，最终D2H也计时。新网络训练后在同次运行再次测速。
5. 确定性scene-balanced20%训练人口（4086窗口），一遍短程迁移；冻结epoch19运动主干，加载可兼容权重，GT监督加0.25倍TRAIN抽样query teacher KL；每32步保存，SIGINT/SIGTERM完成当前步后安全停止。
6. 固定dev64最终三分支对照；可用`SHARED_PILOT_DEV512=1`在**同次**运行再评估dev512（仍是selection population）。门槛未过仍生成报告，不自动重训。

零训练36-token失败**不**淘汰可训练结构；初始共享结构得分不代表训练后的效果。

## 结构和数据契约

- source中心V18时空主干和原运动损失不改。迁移阶段冻结，联合测速实际反传。
- 原生四帧语义/visibility ->低通道两层可分离空间编码；独立保留全部Z，无高度mean/top-only。
- 在完整SE(3)逆变换后的每个Z位置查特征，而不是把不同高度都当同一个XY。
- 每帧9个位置（原7-cell patch的-3/0/+3坐标），保留时间/位置、source query、基线/fallback；query相关membership在读出时注入，不进入共享encoder。
- embedding、column projection、decoder、heads、source projection、TRAIN校正权重可迁移；旧64通道patch CNN**不声称**等价迁移到新低通道native CNN。
- dense和tile为同一模型。halo=2，地图外的**中间层**也归零，防止边界漏入；无空间统计归一化。分块/整图选择看实际区域并集与halo成本。
- session仅一个window/一次forward图；跨horizon复用并保留梯度，optimizer更新后禁止复用。不跨优化步缓存learned features。
- 当前模型运动输出在设备上构建ownership/fallback、动态支持域、候选和采样；GT只进入targets/正负分层/损失；REMOVE恢复fallback、refine覆盖顺序及generation最后free-only写入保持。
- 新GPU sampler的generator状态单独保存，**不是**原NumPy RNG的无缝替换，不允许把旧full断点静默恢复到新结构。
- source搬运/末位owner与次末位fallback按六帧批量计算；同一source的重复点先去重，不会把fallback错误地设为当前source。训练六帧已抽样query合并读出，保留逐query的horizon、类别、membership和source latent。
- 旧网络沿用现有Graph后端；新结构仅对纯reader捕获Graph，历史编码每次重新计算。图输出须复制再汇总，不能把同一可覆盖缓冲区留在多个chunk里；首窗口检查六帧全部reader概率字节。编码prefill和后续读出使用相同BF16策略。
- 联合训练速度的旧对照也启用已有GPU全patch采样，不与故意关闭优化的旧路径比较。两种sampler的人口/预算/importance相同，但NumPy与device RNG的实际抽样ID不同，结果不是固定ID的内核微基准。

## 尚未完成/不能宣称的部分

固定历史source提取/关联/配准/frontier仍为原CPU前端，既有Strong prior重建仍含NumPy接口；六帧FPS已把prior计入，不能宣称整个方法完全GPU原生。
现阶段未实现普遍有效的旧模型共享K/V：相邻patch有重叠不等于memory相同，不凭空记账收益。
FPS采用resident source输入+已注册历史的边界，**不是raw sensor端到端**；原始加载与固定CPU准备单独记录。没有服务器GPU/assets时只能报告CPU单元测试，真实速度和精度必须看服务器此次bundle。

## 运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_shared_evidence_pilot.sh
```

一次运行即完成上述全部项目，默认不启动完整15/20轮。
可选`SHARED_PILOT_SMOKE=1`将所有项目预算改成2窗口/2更新（只能作运行检查）。
`SHARED_PILOT_NO_TRAIN=1`仅探针+测速；默认执行一遍20%人口迁移。

安全停止时`migration_last.pt`在新输出目录；使用相同预算/teacher/data/config恢复，依旧写新目录：

```bash
# 指向此次实际输出的断点，不能用旧full的last.pt。
export SHARED_PILOT_RESUME=/实际/shared_pilot目录/migration_last.pt
bash tools/real_motion/run_p0_f9_shared_evidence_pilot.sh
```

恢复会先检查新protocol、实现内容指纹、优化步数、数据人口和诊断预算，拒绝旧full断点或静默改配方。
若原目录已有完整探针/测速结果且合同完全一致，则只读复用，避免重复前置实验；仍做当前进程renderer预检。
SIGINT/SIGTERM完成当前优化步并保存；硬杀/崩溃只保证上一次32步周期断点。速度报告标明哪些阶段来自此前已完成结果。

Moving support 每个 horizon 使用原 adapter 的 `(boolean_mask, moving_records, excluded)`
三元组中的第一项，并在进入统计前验证六帧数量、布尔类型和完整网格尺寸；不把实例/排除元数据当作 mask，
也不允许广播或回退到全网格，从而保持 frozen Moving 指标不变。
若失败在首个 probe、尚无 `migration_last.pt`，没有可恢复的迁移更新；更新修复代码后重新运行脚本，
保留失败目录作诊断，不设置 `SHARED_PILOT_RESUME` 或复用旧输出路径。

只读复用原因果几何缓存；RAM配额256MiB，新增磁盘缓存配额0，保留原完整性检查。
`summary.txt`可直接发回；`bundle.json`含详细时钟和guards，`progress.jsonl`含迁移更新与实际采样。

门槛：同人口同阈值mIoU及aggregate/1/2/3s Moving不低于teacher，真实联合训练更快、训练后六帧延迟≤250ms。未通过仅说明此次有限预算原型未通过，不自动重试，也不夸大已达到目标。
