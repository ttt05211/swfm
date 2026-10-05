# 冻结 epoch19 的 GenieDrive 公开代码对齐评估

本入口不训练、不调阈值、不写原 checkpoint。它对齐公开源码的筛选和指标，不宣称复现论文 Table 2 的实际运行人口或模型。

## 冻结证据

- 官方仓库：<https://github.com/Huster-YZY/GenieDrive/tree/da48a529ffbe14136688e9b7a56f5d1061c366c5>。
- `occ_gen/configs/world_model/vae_e2e_long.py`：3 previous + t0，20 future keyframes，`use_gt_traj=True`。
- `occ_gen/mmdet3d/datasets/nuscenes_world_dataset.py`：全局 timestamp 排序，历史/未来不足时重复最后合法 index；evaluate 排除有重复 index 的整个窗口。因此正式人口需要24个 distinct、contiguous 的 sample。
- `occ_gen/mmdet3d/datasets/occ_metrics.py::count_miou`：忽略 free class，且将 **恰好零** 的 class IoU 变为 NaN 后平均；每个 horizon 四舍五入至两位。
- 官方 metadata：<https://huggingface.co/ANIYA673/GenieDrive/blob/17e37acfff5b10517393a669ecf471f75f34d43f/world-nuscenes_infos_val.pkl>。
  文件 103781328 bytes；SHA256 `0426072260a908260625c6dd91b9f06919726f5265c10849b87156d282547ded`。先认证再反序列化，不下载 checkpoint/train metadata。

本地对该真实官方 metadata 的只读解析得到 **6019 samples、2569个完整4+20起点、150 scenes**。其中 **300个** 起点只有4或5帧历史，不能由旧六历史缓存交集恢复。源码排序和重复排除算法另外有 literal-reference 单元对照。

冻结 ordered `(scene_name,t0_token)` 指纹 `3bdfe0d56e8bccd6c499239550de7dc7827e969772d57bb570eb52bc34075604`；服务器筛选必须逐序一致，否则在预测前报错。

当前官方 long config 仍命名 `II_World`，实现导出的是 `EE_World`；公开仓库未提供论文 Table 2 起点 manifest。这里冻结并明确报告 **public-code alignment only / paper_table_population_verified=false**。不能因为代码人口已经对齐，就直接认定与论文所有对比方法完全可比。

## 模型输入与新旧人口隔离

1. 与官方 metadata / 本地 nuScenes 的 token、timestamp、scene、prev/next、验证集 sample 集合逐一核对；不能靠“150 scenes”推定一致。
2. 完整4+20筛选只读取 metadata identity；模型仍仅预测12帧未来。后8帧不会加载 occupancy/mask/annotation，也不读取其 ego poses。
3. 常规起点复用旧 cache 的因果 motion tensors，但 raw observation 严格 last4；早期300个起点由四张历史 occupancy 构造 Strong/KTA/feature/tube。兼容六槽 ABI 的前两槽为空，不能用补帧或额外两张历史。
4. 第一段所有6帧部署 exactness check 保持；第二段严格来自第一段预测、初始真实可见性和允许的未来 ego pose。GT semantic / Moving support 仅在全部预测完成后用于指标。
5. 不将官方 metadata 中 future boxes / trajectories 等字段复制进 record，不调用六历史 official-trajectory ABI。
6. 配置强制完整 Occ3D grid，0.4m、200×200×16、origin=(-40,-40,-1)，不使用未来 LiDAR/camera mask。
7. 两种人口的 contract 不兼容。旧评估入口默认行为及已有 resume contract 保持；新评估必须新建目录，之后用相同目录和参数 `--resume`。每个完整 window 原子记录计数，不覆盖训练 ckpt/optimizer/RNG。
8. 新人口不允许 `--compare-e14`：这些早期起点没有六帧历史，不能伪装成同预算 E14 对照。

## 输出定义

`evaluation.json / summary.txt` 同时包含：

- 标准 metrics：保留 IoU=0 的类别，Moving 冻结原协议。
- `geniedrive_code_compatibility`：同一批预测、同一原始累计计数，但按作者源码 drop-zero 和 round2；不额外前向。
- exact-zero 被排除的类别列表、absent 类别列表；空指标为 null，不编造成0。
- 完整24-token sequence / selected-key population审计、official file/code 指纹、早期缺缓存起点列表。
- 固定 `redetect` 和 `reconciled` 两条推理路径，共享第一段；没有 GT 分配或自动挑高分路径。不能拿新人口分数减旧人口分数解释为方法改进。

用户服务器实际数据不可在本地代跑。这里只验证官方 metadata 和 synthetic/network/CLI tests，未报告 nuScenes 实测分数。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD

# 官方索引首次自动下载到新建 data/geniedrive 子目录；约99 MiB。
# 已存在则只校验，不覆盖错误文件；不需要安装 HF SDK。
OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/geniedrive_code_epoch19_$(date +%Y%m%d_%H%M%S)
bash tools/real_motion/run_p0_f9_joint_geniedrive_long6s.sh "$OUT"
cat "$OUT/summary.txt"
```

若中断，用**本次实际 OUT**（不是重新生成时间戳）运行：

```bash
bash tools/real_motion/run_p0_f9_joint_geniedrive_long6s.sh "$OUT" --resume
```

额外选项：`GENIEDRIVE_INFO` 可指定下载/校验位置；`LONG_EVAL_CPU_WORKERS` 默认8；`LONG_EVAL_BATCH_SIZE` 默认256；`LONG_HANDOFF_MODES=redetect` 可以只做原始路径，不能称为 reconciled 结果。下载既有数据集/模型、安装 GenieDrive 全框架均不必要。

耗时：旧 dev64 的两个固定路径约4.2秒/窗口，2569窗口粗估3小时；CPU/I/O/场景组成会变，本次 progress 与 stage_seconds 才是实测依据。指标兼容统计本身不增加模型前向。
