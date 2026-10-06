# Height-aware Shared Causal Field：有限服务器训练

这是 `real_motion/height_causal_field.py` 的 **shared_field**，不是此前慢速的
`SharedEvidenceColumns` / source-repair pilot，也不使用 AE、prototype 或蒸馏。

## 方法和实验边界

四帧原生历史 occupancy/visibility → 一次完整有序高度编码和共享空间场 →
按当前运动预测逆映射读取历史场/逐高度语义/可见性/归属 → 结合 source future query
解码 generation ADD 与 refine KEEP/ADD/REMOVE → 原有合法 proposal compositor。
不减少 proposal、不截断动态 source、不降低网格/Z/未来帧数；生成和恢复共享场。

本地 TRAIN12 × 32 遍仅 384 更新，新头随机初始化；DEV12 比旧联合头低约
0.72 mIoU，不能据此推断充分训练的上限。本地 4.5–6.9 倍是四个真实 replay
窗口上的**六帧推理**加速，不是联合训练加速或 L40S 实测。

这次预算固定为 scene-balanced TRAIN4086（full20430 的20%）× **3遍**；
固定 epoch19 的运动预测，**只训练新共享场头**，隔离表示/数据规模问题。
不是干净从头联合训练；不续用旧 Local optimizer，也不替换旧实验。
采样、重要性校正、GEN BCE 与 refine action CE 沿用原定义，不另加损失；
本地同样的等窗口任务损失均值。每窗重新编码场并立即 backward，4窗梯度累积
后 AdamW 更新；不会累积4份全网格反传图。

学习率2e-3，AdamW WD=.01，梯度裁剪5；整个短训统一余弦到初始LR的0.1倍，
不设tail。每轮每窗恰好一次，batch≤4/source预算128；source超预算的单窗
独立处理，不丢 source。更新数按每轮确定性实际分组计算，不用简单除4估计。

TRAIN256的全部未采样合法 query 统计权重；训练时重要性恢复自然类别分布。
权重校准只用 TRAIN，dev不参与；严格固定阈值 `(0.5,0.5,0.95)`，不在本实验调参。
每遍dev64监控，最终dev512对比相同四历史的旧 epoch19 joint/transport。
最终轮交付，不根据dev选best，不跑full4369、不自动重试/扩大/部署。
输出动态/静态refine采样数、两项loss、真实完整训练步/输入等待耗时。

若有足够数据后依旧丢增益/Moving，停止本版；不把零改动、只改静态或低loss当成功。
诊断近似质量门槛：mIoU相对旧joint不低超过0.05pp，整体及1/2/3s MovingMicro
不低超过0.10pp，同时实测完整六帧推理至少3倍。通过也不是部署授权/独立测试。

## 运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_height_shared_field.sh
```

脚本只用已确认服务器路径，检查缺文件就明确报 `[MISSING]`，不会猜路径。
读取 `$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt`；
可通过唯一位置参数指定该已选checkpoint的同内容副本。每次新建
`outputs/p0_f9_height_shared_field/screen20_<时间>_<PID>`；终端和 `.log` 都打印路径。
旧 epoch19/base、原 optimizer、原 geometry 文件只读。
既有 `causal_geometry_cache_v1` 按原合同自动定位namespace，零新增几何磁盘写入；
固定几何和原始帧的RAM缓存各256MiB，prefetch有界。不会缓存 learned field/label/pose。

第一次 prior/初始评估不属于训练稳态；脚本每32步打印进展。
L40S上的新训练耗时不预先承诺。摘要中的训练时间含新场编码/反传/AdamW；
输入等待单列，不能把本地20秒 neural mini-fit当服务器完整训练时间。

## 安全中断和恢复

Ctrl+C 或 `kill -TERM <本次PID>` 在当前完整 optimizer update 后保存，不能用
kill -9。`last.pt` 原子发布并保留 `last.previous.pt`；保存新头/AdamW/NumPy和
Torch CPU/CUDA RNG/逐轮逐batch位置/校准权重/完成报告/合同。故障只能回到最后
**已完成**的周期断点，不把半次失败更新保存成合法断点。最多丢32步；Ctrl+C正常
则不丢完成步骤。开始prior中断尚无训练更新，重做未完成prior即可。

```bash
# 用终端打印的真实运行目录，不猜目录或随意选旧 Local last.pt
HEIGHT_FIELD_RESUME=/root/nas/occ/swfm/outputs/p0_f9_height_shared_field/<本次目录>/last.pt \
  bash tools/real_motion/run_p0_f9_height_shared_field.sh
```

恢复也创建新目录；所有科学合同/人口/初始化/epochs/LR一致才允许恢复。
评估窗口中断从该阶段开头重跑，但不会重做完成训练轮/重置optimizer/RNG。
监控用 `preserve_training_rng`，不推进训练随机序列。
不要用此断点恢复原Local 15/20轮；也不静默改变batch/epochs/训练人口。
`HEIGHT_FIELD_MAX_UPDATES=32` 可有意先停32步，再取消该变量恢复同一完整预算，
不会重定义余弦。独立 `HEIGHT_FIELD_SMOKE=1` 缩小诊断人口，仅代码冒烟，不能
当正式3遍screen断点续用；正常运行不需要另跑smoke。

## 交付结果和测速范围

`summary.txt`：逐轮效果、最终dev512/旧joint对照、训练耗时、实测FPS。
`screen.json` / `progress.jsonl`：详细增删质量/场景/各horizon/CPU分段。
`last.pt`：新协议可恢复实验checkpoint，显式 `deployable=False`。

同卡配对FPS从驻留source输入及已配准四历史开始，**计入重新Strong/KTA prior、
live motion、所有候选/共享场/所有六帧稠密输出**；不含磁盘I/O、初始提取/配准、
GT/指标、warm-up/graph capture和校验。`FPS=6/平均六帧延迟`，不是单次未来
更新频率，不是raw-sensor端到端；重复输出逐voxel验证在计时外。
不会用随机初始化头的timing掩盖真实训练后结果，也不宣称冻结motion训练速度
等于完整joint训练速度。
