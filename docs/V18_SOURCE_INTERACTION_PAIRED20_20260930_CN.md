# V18 source-interaction：20% 数据配对联合续训

## 目的与唯一实验

此前冻结 V18 latent、只训练外部 XY 头的 512-window screen 没有通过。
它不能证明原 V18 的端到端表征无法改善，也不能证明增大数据一定奏效。
本实验同时给出原架构续训对照与加入交互的候选，区分“数据/联合续训收益”与“结构收益”。

| 条件 | 原 V18 联合续训 | Source-interaction V18 |
|---|---|---|
| 初始化 | 同一个 Clean-E14，校验 SHA256 | 原参数完全相同，新 attention 输出投影初始化为零 |
| 原参数 AdamW moments | 从 Clean-E14 恢复 | 同样恢复，新层才初始化 moments |
| 训练 population / 顺序 | 同一冻结 4086 windows | 完全相同 |
| 可训练参数 | 全部原 encoder、时空 Transformer、decoder、XY/existence/yaw heads | 全部原参数 + 一层 source attention |
| 默认预算 | 3 个子集 epoch | 同样 3 个子集 epoch |
| loss / LR / clipping | 原四组 loss、原 checkpoint 固定 LR、clip=5 | 相同，新层使用相同 LR |
| source / KTA / pivot / renderer / 指标 | 原合同不变 | 同样不变 |

这不是从头重训，也不是完整训练集的三轮；每组共消费 4086 × 3 = 12258 个训练窗口。
step 数由整窗口 packing 决定，不是 screen1024 的旧 1024-update 合同。
两个原有源中的 yaw 输出也联合更新；冻结 yaw head 并不等于共享 decoder 变化后 yaw 预测不变，因此不采用那种“伪冻结”。

## 结构和因果边界

```text
原 V18 的历史 local semantic/source-mask tubes + 历史运动 + KTA
    → 原共享 encoder / spatial-temporal blocks
    → 原 six-horizon future-query decoder
    → [候选新增] 稀疏 causal source attention
    → 原 XY residual / existence / relative-yaw heads
    → 原 source-centred SE(2) + A1 CLEAR/WRITE renderer
```

每个 source 的六个 future queries 读取同一窗口内、t0 距离不超过 30m 的最多 16 个其他 source 的历史 context。
边信息是 t0 相对 XY、历史最后帧相对速度与 observed class 是否相同。
历史局部 tube 本来就含邻居的间接证据；新增的是显式跨 source 表征交互，不宣称原 V18 完全没有邻居信息。
窗口之间绝不互相 attention；无邻居/空 source 时 correction 严格为零。
新层使用零初始化输出投影而非追加零值 token，避免改变原 attention 的归一化分母。
在实际 CUDA/BF16 启动时另检查两组初始 XY/yaw/existence 输出与冻结 V18 逐元素一致，不通过即停止。

GT future displacement/yaw/validity/existence/supervised flags 只能进入 loss 或评估，不进入 forward 或邻居构建。
不按未来 GT validity 删除 context sources，不截断大窗口内的 source。
部署 helper `forecast()` 只请求 `include_gt=False`，仍输出六帧。
生成模块、dense completion、prototype banks、new-source tokens 均不参与本实验。

## 数据、选择与成功口径

正式入口强制 full20430 TRAIN cache，因此 floor(20430 × 0.2) = **4086**。
先按 seed+scene-name 的稳定 SHA256 选出 32 个 TRAIN 场景作 calibration，不参加本次梯度更新。
其余场景 round-robin 分配 4086 个窗口，每场景按完整冻结 cache 时序取等间隔的中点秩，不取前缀。
calibration 每场景取 1/4 与 3/4 时序秩，共 64 个窗口；该 population 与优化场景、official dev 场景分离。
原预训练 V18 见过 TRAIN calibration，因此它不是“基础模型从未见过的测试集”；最终 official dev64 才做外部比较。
所有 key/order、seed、population fingerprint、配置 fingerprint、git SHA 和 checkpoint SHA 写入 execution_contract.json。

每轮结束后先原子保存两组 last checkpoint，再在 TRAIN calibration 上用正式真实 renderer 和冻结 mIoU 选 best。
只按该 mIoU 严格改善选 checkpoint，保留 update=0 无修改对照；不再用 ADE 替代目标指标选模型。
三个默认 epoch 是固定预算，不因校准或 dev 失败自动延长/改 loss/重复训练。
最后一次 dev64 共同评估两组 best 与 last，共用 raw/Strong/V18 baseline/Moving support 准备。
dev64 严格沿用现有冻结 manifest，不选 checkpoint，不扫阈值、不重抽样。
每个 variant 报实际 occupancy、XY/yaw/velocity/acceleration、per-class/history strata、scene deltas 与 damaged/corrected voxels。

实用 screen 门槛沿用上一版：ΔmIoU ≥ 0.30pp、ΔMovingMicro ≥ 1.00pp，且 overall / 1/2/3s 的 IoU、mIoU、MovingMacro、MovingMicro 均不下降。
不达门槛不宣称有效；best=0 不代表训练没运行，也不是模型改善。
交互结构是否有额外贡献必须看 interaction−continuation 的差值，不只看相对 frozen V18。
dev64 已被多次用于探索，本次即使通过也只是候选筛选证据，不能替代之后冻结条件的 dev512/full4369 验证或统计显著性证明。

## 运行与资源

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_v18_source_interaction_20pct.sh screen
```

runner 使用用户确认的路径，runtime 配置是 configs/real_motion_occfm.yaml，不是 V20 history-frozen YAML。
无需新 Stage1、V18_DEV512 或任何 prototype 文件。缺文件/错误环境/非 BF16 GPU 会明确退出，不降级 CPU 或猜路径。
默认一组 step 预算是 ≤256 sources、≤8 windows；一个超预算的大窗口单独处理，绝不删 source。
可以用 `V18_SI_SOURCE_BUDGET` / `V18_SI_WINDOW_BUDGET` 改 packing；两组仍相同，实际参数写入合同。
这些量不是 completion 的 millions-of-voxels batch，不能用 V20 的旧 step 时长外推。
训练只读已有 V18 cache tensor，不每 step 读 nuScenes，不保存巨大 latent 或 voxel bank。
CUDA encoder/attention 为 BF16；原 loss 的坐标/yaw/soft-SE2 几何在两组都以 FP32 计算。
只保留各组 best/last 四个模型文件（last 含 Adam 状态），日志每 32 steps 报关键训练量。
运行时长待 L40S 实测，日志包含每轮训练/校准和最终共享 dev 的真实分项时间。

可选 `smoke` 是 8 TRAIN / 2 calibration / 2 dev / 1 子集 epoch，仅验证线路，不是效果实验；正常可直接 screen，runner 已执行小型单测和 BF16 初始等价检查。
runner 不覆盖已有输出目录、Clean-E14 或任何缓存；出错直接退出，已完成 epoch 的 last checkpoint 不丢失，但当前入口不提供断点续训。

运行输出在 outputs/p0_f9_v18_source_interaction/screen20_<时间>_<gitSHA>/model/。
返回 `summary.txt` 即可；`summary.json`、`train_history.json`、`progress.jsonl` 保存完整细分结果。
summary 同时列 best 与 last，以免 update=0 掩盖候选实际发生了什么。
新 checkpoint 格式与原 frozen V18 分离。`load_candidate()` 默认拒绝 smoke/失败/update0/last 作为通过的部署模型。

## 验证边界

本地通过新增单元/合成完整流程测试：20% 场景与时间覆盖、窗口隔离、无 GT forward、初始化等价、Adam moments 恢复、不别名共享、原四组 objective 一致、空 source/NaN missing label、共享 encoder/XY/yaw/new-attention 梯度、checkpoint 选择/保护、真实 A1 renderer 和冻结 raw-count metrics。
本地 Windows 使用项目专用 Python，禁止 F:\\anaconda3\\python.exe。
本地没有 nuScenes 服务器数据/CUDA，真实 20% 联合续训和实测收益尚未验证；不能用单测通过冒充实验成功。

提交前实际检查：新增测试 15 passed；全仓非 integration 测试 629 passed / 4 skipped / 1 deselected，28 条现有 warning；443 个 Python 文件 AST 检查、训练入口 --help、runner bash -n 与 git diff --check 通过。
