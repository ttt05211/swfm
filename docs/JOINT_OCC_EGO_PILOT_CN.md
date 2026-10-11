# Transport + Surface CCR + Ego：最终占据监督的小规模联合训练

## 本次范围

独立入口 `tools/joint_occ_experiments/run.py`，不改旧模型、旧训练器、旧执行后端或缓存签名。
一次运行两个完整模型：GT-ego 与内部 Pred-ego。两者从相同随机初始化开始，使用相同 TRAIN 人口、窗口顺序、dropout RNG、Adam 配置和训练预算；不是旧 BCE 配置与新 CE 配置的消融。

默认固定 TRAIN20430 的 scene-balanced 20%（4086 窗）×3 轮，全周期余弦，最后评估冻结 DEV512。未启动服务器训练，不能视为已收敛方法、涨点证据或新 FPS。

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

硬评估固定 ADD0.5，REMOVE0.95，同时报告 REMOVE-off，不作 DEV 阈值搜索或 best-epoch 选择。最终输出模型为 `last.pt` 的 `models.gt` / `models.pred`；它同时包含两路 Adam、RNG、余弦预算和完成游标。

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
- 六帧完整 dense loss 逐时刻 backward、窗口内梯度累积，避免一次保留整个 batch 的六帧 dense 渲染图。不是减少监督/支持集。
- Source/edit 合成仅对触及体素做 sparse scatter，避免每个 source 复制完整18类体积；Lovász 按类别批量 sort，消除17次逐类别同步。
- 评估复用旧有界执行调度，新增两动作 readout；CUDA graph 首次须通过本模型 eager/graph bytes parity，失败回到 eager 并记录。

全网格 warp/sort/backward 比原 sampled action BCE 有额外开销，不能承诺更快。`performance.paired_pipeline` 包含输入等待和两个模型训练，按最近64个 paired updates 统计；CPU worker sums 会重叠，不能直接相加当总耗时。旧54.945 FPS测量不赋给新方法，旧路径本身未修改。

默认额外 RAM 上限：帧缓存每 split 2GiB、纯几何每 split 512MiB、历史描述512MiB，加上两组有界预取；不申请几十 GB 新 disk cache。实际 RAM/峰值显存需服务器实测。

## 服务器命令

代码更新后，在现有 OccFM 环境运行。无需下载数据或重建任何 bank。

先以相同3轮契约只执行8步，生成时间/显存/梯度链和缓存检查；这8步保留，随后直接接续，不浪费预热结果：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/paired3_$(date +%Y%m%d_%H%M%S)
bash tools/joint_occ_experiments/run_paired.sh --max-updates 8
cat "$RUN_DIR/summary.txt"
```

确认没有 OOM/异常后完成同一批4086窗×3轮，并在 DEV512 评估两路硬模型：

```bash
RESUME="$RUN_DIR/last.pt" bash tools/joint_occ_experiments/run_paired.sh
```

不想先停8步可直接运行 wrapper。一次 Ctrl-C/SIGTERM 等完整 paired update 保存后退出；用相同 `RUN_DIR` 和 `RESUME` 接续。不要 `kill -9`；崩溃只能回到最近的周期 checkpoint。

新终端恢复时，把 `RUN_DIR` 设为真实旧路径，不能重新生成时间戳。
没有外部 planner/STC 依赖，DEV512不再因 planner 缺17行而变成495窗。本次尚未训练 Camera 噪声鲁棒性，不用 validation 的 STC 预测训练模型。

若希望中途只读评估当前断点：

```bash
RESUME="$RUN_DIR/last.pt" bash tools/joint_occ_experiments/run_paired.sh --evaluate-only
```

请先安全停止训练，避免同目录并发写入。评估完整窗口保存原始整数计数，重跑相同命令续评，不改变源 checkpoint。两路最终报告和分段性能均在同目录 `summary.txt` / `results.json`；缓存详细 provenance 在 `cache_audit.json`，逐步日志在 `progress.jsonl`。

## 本地验收限制

新增21项CPU测试：真实六时刻 backward、occupancy到ego/XY/yaw/动作梯度、Pred未来GT毒化隔离、owner/free写入、REMOVE下层恢复、稀疏/批量loss数值梯度、并行历史描述字节一致、空source/evidence窗口、硬评估整数续评，以及独立CLI同目录停止/恢复Adam-RNG等价。
加入旧 surface/ego/recovery 的相关回归后：103 passed / 17 skipped（CUDA）/ 16 deselected（native参数组）。21项新增测试复查通过，CLI help、Bash语法和diff检查通过，借用的 sibling upstream 与仓库固定gitlink commit一致。
最初包含native组的尝试报现有Windows DLL读取权限错误，不是这些组通过；不改旧 DLL/权限。CUDA不可用，native/CUDA/L40S实际吞吐及真实 nuScenes效果留待服务器验证，不宣称全仓CI通过。
