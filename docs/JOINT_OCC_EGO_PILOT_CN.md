# Transport + Surface CCR + Ego：最终占据监督的小规模联合训练

## 本次范围

独立入口 `tools/joint_occ_experiments/run.py`，不改旧模型、旧训练器、旧执行后端或缓存签名。
支持 `--mode gt / pred / paired`。用户当前决定先单独训练 GT-ego：`run_gt.sh` 只实例化、优化和评估 GT 模型，不在后台训练 Pred。
paired 才一次运行两个完整模型，从相同随机初始化开始，使用相同 TRAIN 人口、窗口顺序、dropout RNG、Adam 配置和预算；这是条件配置的公平对照，不共享权重、不互相提供学习收益，也不是旧 BCE 与新 CE 的消融。

默认固定 TRAIN20430 的 scene-balanced 20%（4086 窗）×3 轮，全周期余弦，最后评估冻结 DEV512。旧版服务器仅完成 paired8 步：GT 1.410、Pred 1.470、两路含输入 2.900 s/窗；约88%的两路开销在 dense 渲染/loss/backward，输入仅0.019 s/窗。不是精度验证或稳态训练 ETA；新版尚无服务器吞吐/精度结果。

## 数据流与监督

```text
四帧历史 occupancy/可见性/ego pose
  ├─ 原 V18 编码器/Transformer → source XY residual、yaw、existence、关联特征
  ├─ 历史 ego 状态 + 六个导航命令 → 8710 参数控制头 → 六帧 ego SE(2)
  └─ 因果物体关联、canonical 支持点、surface 描述
        ↓
GT 配置用 GT ego；Pred 配置只用内部 ego 预测
        ↓
即时 Strong/background、source ownership、Surface CCR 的 ADD/REMOVE 概率
        ↓
训练：连续 warp/splat + 概率合成 → 六帧全网格 CE + semantic Lovász
推理：原硬 raster/compositor → Transport / ADD-only / ADD+REMOVE
```

总体损失只有三个分组，默认权重均为 1：

- 最终六帧占据：CE + Lovász，忽略未知标签，CE 保留 free17；Lovász 对有 GT 支持的语义0–16计算，不做 voxel 抽样。
- Source 辅助：可靠观测 source 的32个形状点 SE(2)几何 SmoothL1 + existence BCE。提供远离目标时的运动梯度；不再叠加原 translation/yaw/shape 多套损失。
- Ego 辅助：固定 R10m 五参考点的 SE(2)几何 SmoothL1。GT 配置也训练同一个内部头，但投影仍使用 GT ego。

Pred 的 forward 始终使用预测轨迹，不掺未来 GT 连续姿态。占据到 ego 的梯度在训练前20%步由0.1升到1，改变 backward 强度、不改变 forward。GT-derived destination navigation commands 是显式条件，不能称 navigation-free。

历史匹配、候选集合、取整地址、owner/合法性和多数投票仍是离散选择；连续 source XY/yaw、ego 变换、Surface context/phase、splat 权重和两种动作可回传。不要将其称为所有离散操作都可微。

训练的 soft static inverse warp 并不等价于硬 Strong 多数投票；出当前网格的填充来自当前 ego 条件下即时背景。新增首8个评估窗的 soft-argmax/hard 指标差异报告，不能通过选路线或阈值抹平差距。

ADD 仍只写入原 baseline free voxel；REMOVE 仍受 source ownership/候选合法性限制，动态移除恢复硬渲染器的真实下层语义，静态移除为 free。不是任意类别重标/全网格删除。

硬评估固定 ADD0.5，REMOVE0.95，同时报告 REMOVE-off，不作 DEV 阈值搜索或 best-epoch 选择。输出为 `last.pt` 中所选配置的 `models.gt` / `models.pred`；包含所有活动配置的 Adam、RNG、余弦预算和完成游标。

## 旧缓存复用及性能边界

| 旧资产 | 用法 |
| --- | --- |
| TRAIN20430 / VAL4369 V18 record cache | 复用历史输入与训练监督；原缓存只读 |
| TRAIN/VAL CCR history geometry | 复用 current source、点云、registrations、compact canonical support |
| GT未来 ego 条件的 backgrounds / conflicts / footprint / memory | GT 可复用背景；Pred 全部丢弃并按当前预测 ego 重建，绝不沿用 GT future geometry |
| 冻结 WM 提取的 ego feature bank | 不复用；新 WM 联合更新会使这些特征过期 |
| 旧 checkpoint | E14只提供网络结构与固定缓存协议，不复制权重、不能在新协议中 resume |

没有新增全量特征 bank、没有旧缓存重建或写入。已有 namespace/manifest 缺失或不匹配时直接报出真实路径，禁止自动重建/猜选不匹配目录。

已实现的训练节省：

- 四个有界 CPU 线程预取 raw 数据、缓存和历史 surface 描述；带 single-flight/LRU，两个模型只做一次历史 fit，不在训练主线程逐窗串行重算。
- 复用既有 native-parallel 投影和 Strong 多数投票；Pred 的 learned 几何每次即时计算，不缓存。
- 每窗一次 CCR 历史编码、六时刻共同读出；静态坐标网格只构造一次。
- 六帧完整 dense loss 逐时刻渲染反传；在 poses/targets/yaws/mapped/actions 边界积累导数，再和辅助监督一起通过一个 multi-root backward 回传共享 WM/CCR。每窗共享网络只回传一次，而非每个 horizon 遍历一次。临时 leaf 的所有梯度显式传回原图，不是 stop-gradient；保留六帧全网格监督。
- 融合稀疏有序概率合成，反传用一个 dense 梯度缓冲，避免 CopySlices 链按每个 source/edit 复制整网格梯度。保留叠加顺序、重叠、owner/free 掩码及 REMOVE 下层恢复规则。
- Lovász 保留完整 stable sort、同样的 present-class 定义和公式，使用解析一阶 backward，减少 autograd 中间图；没有 voxel 抽样、近似排序或取消 Lovász。两个融合算子仅支持普通一阶训练，不支持二阶优化。
- 评估复用旧有界执行调度，新增两动作 readout；CUDA graph 首次须通过本模型 eager/graph bytes parity，失败回到 eager 并记录。

全网格 warp/sort/backward 比原 sampled action BCE 有额外开销，不能承诺恢复旧 BCE 的训练速度。`performance.paired_pipeline` 保留旧字段名，但 `active_modes` 明确实际活动配置，包含它们的训练及输入等待，按最近64步统计；CPU worker sums 会重叠，不能直接相加。旧54.945 FPS不赋给新方法，旧路径本身未修改。新旧在数学上一阶等价，但共享反传汇总顺序变化，尤其 CUDA BF16 不承诺优化器更新逐字节一致。

默认 `--training-backend fused`；`reference` 保留原实现供数值/性能诊断。配置、后端、实现指纹都进入 resume 契约，不能中途静默改 mode/backend。旧 paired8 不直接恢复到新版 GT-only，保留旧目录；本次使用独立 GT-only 目录重新开始8步，仅损失旧8步训练预算。

默认额外 RAM 上限：帧缓存每 split 2GiB、纯几何每 split 512MiB、历史描述512MiB，加上两组有界预取；不申请几十 GB 新 disk cache。实际 RAM/峰值显存需服务器实测。

## 服务器命令

代码更新后，在现有 OccFM 环境运行。无需下载数据或重建任何 bank。

先以相同3轮契约只执行8步，生成时间/显存/梯度链和缓存检查；这8步保留，随后直接接续，不浪费预热结果：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
unset RESUME
export EVAL_POPULATION=dev512
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt3_fast_$(date +%Y%m%d_%H%M%S)
bash tools/joint_occ_experiments/run_gt.sh --max-updates 8
cat "$RUN_DIR/summary.txt"
```

确认没有 OOM/异常后完成同一批4086窗×3轮，并在 DEV512 评估 GT-only 硬模型：

```bash
RESUME="$RUN_DIR/last.pt" bash tools/joint_occ_experiments/run_gt.sh
```

不想先停8步可直接运行 wrapper。一次 Ctrl-C/SIGTERM 等完整 optimizer update 保存后退出；用相同 `RUN_DIR` 和 `RESUME` 接续。不要 `kill -9`；崩溃只能回到最近的周期 checkpoint。

新终端恢复时，把 `RUN_DIR` 设为真实旧路径，不能重新生成时间戳。
没有外部 planner/STC 依赖，DEV512不再因 planner 缺17行而变成495窗。本次尚未训练 Camera 噪声鲁棒性，不用 validation 的 STC 预测训练模型。

若希望中途只读评估当前断点：

```bash
RESUME="$RUN_DIR/last.pt" bash tools/joint_occ_experiments/run_gt.sh --evaluate-only
```

请先安全停止训练，避免同目录并发写入。评估完整窗口保存原始整数计数，重跑相同命令续评，不改变源 checkpoint。活动配置最终报告和分段性能在同目录 `summary.txt` / `results.json`；缓存 provenance 在 `cache_audit.json`，逐步日志在 `progress.jsonl`。后续若需 Pred，再用独立目录 `MODE=pred bash tools/joint_occ_experiments/run_paired.sh`，不是复用 GT optimizer/RNG。

## 本地验收限制

新增21项CPU测试：真实六时刻 backward、occupancy到ego/XY/yaw/动作梯度、Pred未来GT毒化隔离、owner/free写入、REMOVE下层恢复、稀疏/批量loss数值梯度、并行历史描述字节一致、空source/evidence窗口、硬评估整数续评，以及独立CLI同目录停止/恢复Adam-RNG等价。
本轮融合训练/单路优化：两文件合计38项CPU测试通过；包含有序重叠合成全部输入梯度/gradcheck、Lovász ties/free/unknown/零子梯度、真实六帧渲染、GT与Pred共享一次反传的全部参数及Adam moments数值一致，以及 gt/pred/paired 各模式同目录恢复与模式/后端变更拒绝。CUDA/BF16 实际吞吐仍待服务器8步测量，不把CPU数值一致说成CUDA逐字节一致。
最终连同原joint/surface/recovery相关回归：81 passed / 7 CUDA skipped（不是全仓CI）。CPU合成+CE/Lovász synthetic microbenchmark，18类×640000体素×24有序编辑，2次计时均值：reference 2.884182s，fused 2.000638s，1.4416×。不包含网络/数据/几何，不是L40S速度、整轮ETA或FPS；实际可用`benchmark_fast.py`复核，默认不在训练前另跑这个实验。
加入旧 surface/ego/recovery 的相关回归后：103 passed / 17 skipped（CUDA）/ 16 deselected（native参数组）。21项新增测试复查通过，CLI help、Bash语法和diff检查通过，借用的 sibling upstream 与仓库固定gitlink commit一致。
最初包含native组的尝试报现有Windows DLL读取权限错误，不是这些组通过；不改旧 DLL/权限。CUDA不可用，native/CUDA/L40S实际吞吐及真实 nuScenes效果留待服务器验证，不宣称全仓CI通过。
