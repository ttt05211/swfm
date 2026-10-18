# Stage2 全量 epoch2：未来运动只读诊断

## 目的与边界

解释新模型 Transport 从 step2000 到10292几乎不变，而旧结构 Transport 曾随训练改善的现象。检查“修正量太小”“修正没帮助”“只帮助运动物体却伤害其他区域”等可能性，不预先认定原因。

独立入口 `tools/source_motion_audit/run.sh`，只读取完整 Stage2 epoch2、同一 remaining64、真实四帧 OCC/STC；单卡单进程，不创建 optimizer，不训练、不改变原 trainer/source/KTA/历史有效性规则或旧结果。GT future ego 是此配置明确条件；不是 Pred 实验。

## 四个预先固定的干预

| 路线 | 未来 XY residual | 未来 yaw residual |
|---|---|---|
| learned | 当前学习值 | 当前学习值 |
| KTA_future | 0 | 0 |
| XY_only | 当前学习值 | 0 |
| yaw_only | 0 | 当前学习值 |

每窗编码一次得到共同的 F、Q、lifted history 和历史相对 yaw；干预时这些都不变，原两帧 KTA、source 身份、背景也不变。改变未来运动时，同时重算几何 Transport 和对应 direct-history memory，再用同一 Refiner 读出，不能只改占据却保留另一条运动对应的 memory。

KTA_future 不是删除整个神经运动模型：它保留 Q、历史 yaw 以及原有语义门控；这是隔离**未来运动修正**的干预，不代表一个重新训练的 KTA 模型。learned 路线直接调用原 `forward_window(fresh=True)`，另一次原路径重复推理记录 byte 漂移，不过滤失败窗口、不放宽容差。干预不是严格字节等价或 FPS 测试。

## 指标与 GT 的用途

每个模态、每条干预均先完成完整六帧预测，之后才查询未来 GT 占据、moving support 与框的运动信息。GT 标签/框不进入网络或 source 对应关系。nuScenes 元数据本身会在数据源初始化时加载，因果限制指这些标签不参与预测计算，而不是声称数据库此时尚未加载。

- 四路线 × Transport/Refiner 的标准全网格1/2/3s mIoU、IoU、MovingMicro。
- 同样整数混淆矩阵：全网格、GT-moving support、support 外区域，逐类别/时距、FP/FN。support 外包含背景，**不能称作精确静止实例集合**。
- learned 相对 KTA 的 corrected/damaged、破坏正确 free、错类变另一错类计数。计数不等于 mIoU 因果分解。
- 每 source/时距：未来 XY 修正米数、有效 yaw 度数；历史有效帧 yaw 绝对值分布（不含 t0 的固定零 gauge），头参数范数。
- t0 component 用固定框包含率归属到 nuScenes instance，歧义/低纯度/未覆盖仍标 unresolved，不强行匹配；未来按同一 instance token 查询。
- 比较框中心**位移**与预测 source 位移，不直接比较组件质心绝对位置和框中心，避免形状偏心产生假运动误差。使用实际 sample 时间差，按固定0.5m/s分组 moving / slow_or_stationary。
- GT 框归属是有条件近似，不是完美点运动 GT；多个碎片可对应同一实例，时距/重叠窗口统计也不独立。无未来框的实例不算轨迹误差，覆盖量一并报告。
- yaw 误差只统计现有允许 yaw 的类别，不顺手修改原类别规则。

## 运行

```bash
export SCENE_FULL_DIR="$PWD/outputs/p0_f9_shared_history/scene_stage2_full_2ep_20261018_113440"
export MOTION_AUDIT_GPU=0
export MOTION_AUDIT_DIR="$PWD/outputs/p0_f9_shared_history/stage2_motion_audit_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_motion_audit/run.sh > "${MOTION_AUDIT_DIR}.log" 2>&1 &
tail -F "${MOTION_AUDIT_DIR}.log"
```

输出 `summary.txt`、`results.json`、`state.json`、逐窗 `windows.jsonl`。每8个完整窗口发布整数累计状态；Ctrl-C/SIGTERM在完整窗口后保存。断电可能重算最后未发布的不足8窗，恢复以 state 为准，JSONL可能保留未发布尾部的重复记录，不以JSONL行数当科学进度。

同一输出目录续诊断：设置 `MOTION_AUDIT_RESUME=1` 再运行。校验原 checkpoint、原实现/配置/人口、输入文件统计、元数据和新入口源码、运行环境契约；禁止把其他模型或人口结果拼接进来。源 checkpoint 从头到尾校验哈希，原缓存没有 writer。

## 如何解释

- learned≈KTA 且修正近零：支持“未来修正近乎没学到”，但需结合原路径重复漂移大小。
- 修正非零、框位移误差和 Transport 都不改善：支持“修正缺乏有效性”，不代表梯度被完全截断。
- moving 改善、slow/static 或区域外损害：解释指标取舍，而不是笼统说运动完全没学。
- XY/yaw分别干预仅定位敏感环节，不是 DEV 选路部署。
- 该诊断不能单独证明 KTA Query 或 ICP 哪个必须恢复；二者与其他结构变化混在一起，仍需同预算单变量实验才能作因果结论。

本地测试是小网格实际 CPU/CUDA 前向与指标/恢复检查；没有最新全量权重和本地完整框标注，真实 epoch2 结论须等服务器结果。旧版本、正式默认和所有历史分数不变。
