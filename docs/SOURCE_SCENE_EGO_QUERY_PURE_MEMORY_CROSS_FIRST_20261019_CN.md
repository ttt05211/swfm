# 最新2+2：Ego Query＋纯历史Memory＋Cross-first联合候选

独立目录 `tools/source_scene_ego_conditioned/`；不修改旧Integrated2+2、训练入口、已有checkpoint、旧缓存或mean3结果。

## 三项联合改动

1. 保留六个独立 learned horizon queries 和原 KTA Query/skip。新增 `Linear(4,128)→SiLU→Linear(128,128)` ego embedding，最后线性层权重/bias零初始化。输入为 `inv(T0) @ Th` 提取的相对t0 XY、sin/cos yaw；XY仅做固定20m数值归一化，不截断。不是ego预测头，不增加ego loss。零初始化仅保证新增ego贡献为零，不代表三个改动后的整个模型等价于旧版。
2. 新模型删除 `columns.source_query` 模块、参数及所有Memory调用。动态内容只用lifted historical features，importance只读历史特征和原coverage，并在六时距间复用。保留学习的importance、静态融合、empty表示、归一化、学习的历史相对yaw和可微运动对齐。Q仍经XY/yaw→几何/Memory位置影响最终占据，不直接改写内容或权重。
3. 两层Decoder均使用 PreNorm `History Cross-Attention→Future Self-Attention→FFN`。Self-Attention重新归一化cross更新后的query；history有效性mask只传cross历史K/V，六个future queries不加padding/因果mask。

统一forward在任何Decoder计算之前确定一份六帧future poses，同时供ego features、Strong/CLEAR、Transport prior和Memory使用。显式传入轨迹时强制新建背景，不能误用GT缓存背景。source anchor＋XY residual→world和现有world-Z yaw约定完全不改；不把source输出变成相对未来ego位移。没有future GT occupancy/可见性mask/框进入网络。

当前pilot使用GT future ego。它现在既是行为条件也是输出投影条件，比旧模型仅用于未来几何的依赖关系更强，需明确披露。接口能接显式Pred轨迹，但此实验不是Pred效果证据；未新增内部ego头/导航输入，也不自动迁移到旧Pred路径。

## 固定验证预算

- RANDOM、原TRAIN20430中相同round-robin TRAIN4086×1轮，原seed20261011/组顺序/global B4/source128（不截断sources）。
- H6、full-grid CE＋Lovász ONLY，无motion/existence/ego/alignment额外loss；dense-valid用于TRAIN/OCC/STC；GT future ego。
- 复用已完成最新Integrated2+2的TRAIN20%×1轮报告作为控制，不重训控制，不导入训练权重。共同模块初始化和初始化后CPU RNG与fresh控制一致；本次是三个联合变量，不是单变量或等参消融。
- 原motion LR5e-4、Refiner LR3e-4、整轮cosine、AdamW/clip不变；新ego projection归motion optimizer，不另加ego optimizer。
- 默认双卡仅用于训练，global B4不变；最终remaining64 OCC/STC和learned/KTA运动诊断仅rank0单GPU评估，无best selection或自动扩大训练。
- last.pt每64更新保存；epoch_0001.pt和last.pt独立保存；评估完整窗口整数计数支持恢复。

## 服务器命令

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
export SCENE_TRAIN_GPUS=0,1
unset EGO_QUERY_RESUME
export EGO_QUERY_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_query_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_ego_conditioned/run.sh > "${EGO_QUERY_DIR}.log" 2>&1 &
tail -f "${EGO_QUERY_DIR}.log"
```

入口只接受唯一已完成的 `scene_integrated_1ep_*` 2+2小实验作为控制；多于一个则停止，显式 `export INTEGRATED_PILOT_CONTROL_DIR=/实际原小实验目录`，不按分数或mtime选，也不接受完整TRAIN epoch2/deep44/旧Stage2作为等预算控制。

完成后：`cat "$EGO_QUERY_DIR/comparison.txt"`；日志 `SCENE_EGO_QUERY n/1031`，最终评估 `EGO_QUERY_EVAL n/64`。

恢复：设回相同 `EGO_QUERY_DIR` 和控制目录，`export EGO_QUERY_RESUME=1`，运行同一入口。必须同代码、输入、world size、contract；不续旧版本checkpoint。完成实验再次resume不重复训练/评估。

## 本地验证与边界

新10项＋原Integrated/Full/Mean3回归，共32 passed（27.64s）。包含真实CPU/CUDA full6前后向、相对ego世界坐标规约不变性、cross-first精确算式、无Q对象Memory仍工作、运动仍可回传、六时距位置不同、单轨迹/无双算/未来label poison、空source、零初始化梯度逐步打开、实际CUDA训练断点恢复及源报告只读。

CUDA scatter正常重复存在约1e-7级归约漂移；Memory精确结构身份测试使用已有deterministic_entrance审计上下文（原BF16精度），退出后再做普通训练backward。没有放宽数值阈值、没有修改生产确定性开关，不声称普通推理bitwise一致。新方案的真实质量/FPS仍未测，本地微型fixture不当科学训练结果；无服务器正式训练自动启动。
