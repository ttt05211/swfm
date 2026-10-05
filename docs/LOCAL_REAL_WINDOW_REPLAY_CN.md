# 小规模真实数据导出与本地回放

用途：在本地 RTX3050 上定位真实原始数据 → 几何/证据 → 网络的瓶颈，不再仅用随机 token 的核函数耗时推断服务器性能。不是新实验训练或部署流程。

## 服务器只执行一次

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/export_p0_f9_local_replay.sh
```

脚本输出一个新建目录及 `replay.zip`。只需下载这个 zip，不需要下载完整 nuScenes、2.7GB 的 TRAIN cache 或整个实验目录。不使用 GPU，可以与现有 GPU 训练同时执行，但 CPU、读盘会短暂竞争。导出期间不 kill、续训、改写原 checkpoint 或缓存。

默认包含：

- 选定 epoch19、CleanE14 原文件的不可变快照；当前 sparse migration 的最后完整断点（仅回放 head 权重，绝不恢复原优化器）。
- 16 TRAIN + 16 dev 原始窗口。每个 split 12 个 identity/scene-based 样本及4个 source-count 压力样本，选择不读取未来标签或模型误差。TRAIN来自当前 migration 的冻结20%人口；dev来自冻结 dev64。
- 完整 200×200×16 网格、最后四帧 history occupancy/visibility/poses、全部六帧 future ego poses、官方轨迹条件、原始 V18 输入字段。
- **独立 labels 文件**：六帧 GT occupancy、运动监督字段及冻结 Moving support masks。`target_source_mask_tube` 虽然名字含 target，但本来就是因果 t0 source footprint，属于模型输入，不是未来形状标签。
- resolved runtime 配置、checkpoint/cache/info/manifest 指纹、代码版本、迁移完成步数和有界日志尾部。日志抓取时间不保证对应相同的 checkpoint 步，不能混称同一个完成边界。

**不包含**预计算 learned features、模型预测、固定几何缓存。这样本地能真实测几何重建，不能省掉最慢步骤再报全流程提速。压力样本和代表样本分开标记；这32条不能作为质量验证、收敛依据或总体吞吐的精确估计。

默认非压缩体积上限4GiB，开始前要求至少4.5GiB空闲。实际 zip 大小会打印，不预先承诺大小。输出目录必须不存在，失败保留 `.partial` 和状态文件，不发布半包。原 checkpoint 快照按一次打开的 inode 复制，允许训练原子替换 `migration_last.pt`；若原地改写则拒绝。

若迁移输出目录不是默认的 `source_repair_20261005_224502_885`，只需显式指定已经存在的实际目录：

```bash
LOCAL_REPLAY_PILOT=/实际/迁移输出目录 bash tools/real_motion/export_p0_f9_local_replay.sh
```

## 本地验证与计时

把下载的包路径传给回放脚本，不必手动解压。使用 `run_local_cuda.ps1` 中已验证的项目 CUDA 解释器，禁止 Anaconda base；环境说明见 `LOCAL_WINDOWS_CUDA_ENV_CN.md`。

```powershell
# 将下面路径替换为下载后实际位置。
& ./tools/real_motion/run_local_cuda.ps1 -PythonArgs @(
  'tools/real_motion/replay_p0_f9_local.py',
  '--bundle', 'D:/实际位置/replay.zip', '--validate-only'
)
```

完整真实性检查：逐 member SHA256、manifest identity/order、4→6、网格尺寸、SE(3)、KTA source identity、标签隔离。zip 不解压，拒绝路径穿越、未知 member、重复 entry 和超限内容；窗口 tensor 用 `weights_only=True` 读取。模型 checkpoint 是本仓库服务器导出的可信 pickle，实际运行模型前须明确同意：

```powershell
& ./tools/real_motion/run_local_cuda.ps1 -PythonArgs @(
  'tools/real_motion/replay_p0_f9_local.py',
  '--bundle', 'D:/实际位置/replay.zip',
  '--out-dir', 'outputs/local_replay_check_new',
  '--trust-repository-checkpoints', '--device', 'cuda',
  '--windows', '4', '--cpu-workers', '2', '--head-chunk', '512', '--train-probe'
)
```

不会裁 grid、减少 history、删 source 或默默回退 CPU。初始4条按 TRAIN/dev、普通/压力交错；显存不足明确失败。head 分块512只是降低峰值激活，不更改样本或 source 身份。

`replay.json` 拆开冷几何重建、live V18、renderer、真实 evidence union、六帧 mapping、**全量邻域特征**、head、六帧 dense ADD。可选训练探针是每条独立、冻结 motion、GT-only/no KD、全新 AdamW 的一次 head 更新；不保存新模型，也不改变旧实验。几何先重建再复用一次，cold/warm数值分别标识。

这个入口不含旧 generation/CNN、GT metrics、I/O 的全部部署链，因此**不能称完整联合 FPS**。首条包含 renderer exactness 与 CUDA 冷启动，不能作为稳态速度。3050 与 L40S 的绝对耗时/显存不可直接换算；用于寻找数量级错误和验证语义，最终提速仍需要服务器同人口、同权重、同边界测量。
