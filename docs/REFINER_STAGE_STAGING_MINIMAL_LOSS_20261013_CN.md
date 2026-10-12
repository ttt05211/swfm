# Refiner：分段replay、staging与最小损失（2026-10-13）

用户要求：定位原版重复漂移；优化固定张量上传/同步/逐source合成；**暂不融合FP64**；考虑同时删除motion/existence辅助。此次仅实现隔离实验入口，不接正式默认、不自动启动完整数据训练，也不重跑已完成三路loss500。

## 1. 原版三次冻结replay

首个固定TRAIN窗口，恢复相同权重/RNG，用原train-mode forward（no_grad只用于只读审计）跑三遍，逐阶段记录：

`F → Q → Transport_prior → memory → logits`

每段原dtype/shape、exact、数值通过、最大差值和失败元素数；输出首次exact差异/首次numerical失败，以及六帧argmax改变体素数。也对每个候选与首个原版逐段对照。审计在计时之外，不改变原版，能区分“原版自身已在memory发生漂移”和“候选新增漂移”，不能仅因baseline也FAIL就豁免候选。

数值门槛仍2e-4/2e-5，单步gradient3e-5/2e-7、loss2e-6/3e-7等原门槛不放宽。全部阶段有限/结构一致；shape/NaN错误硬失败。只审计首窗不是完整population逐字节证明。

## 2. 新执行候选

| 路径 | 内容 |
|---|---|
| reference | 原正式eager执行 |
| staged_inputs | 每forward纯历史/明确GT-ego条件/导航输入按dtype合并pin/H2D；source点云concat后按原actor切片；静态one-hot直接FP32scatter；网格常量bounded2entry |
| staged_sync | 上项＋inv_ex后集中检查error info，每个window在loss/Adam之前检查；没有取消奇异矩阵安全检查 |
| staged_compose | 上项＋Linux/Triton编译FP32逐元素overlay计算，source覆盖顺序和原逆序VJP公式保留 |

**FP64精度、source逐物体刚体move/einsum/bmm不改，不做FP64 fused affine，也不把它换FP32。** 没有删full6/体素、降低Lovasz频率、改变source/global batch、稀疏监督或添加修复规则。逐source unique/splat地址汇聚仍保留（完整ragged renderer融合尚未实现），不可称已经去掉所有CPU/GPU同步或全部source循环。FP32编译覆盖也可能改变舍入，因此必须通过相同A/B而非宣称无损。

上传全部位于被测forward内；不把成本移到计时外。纯shape/origin/step常量可复用，不缓存F/Q/memory、预测运动或学习特征。不会读取future occupancy/masks/annotations作为输入；future GT ego是原GT协议明确条件，原缓存背景是确定性历史+GT-ego条件Strong，不误称纯历史。

CPU数学fallback只用于验证source覆盖/VJP，**不是编译速度证据**。Windows或缺Triton明确跳过compiled候选；Linux编译器不支持明确报告，不把fallback包装成通过，非编译类型的shape/NaN错误不吞。编译/warmup/profile均不计稳态速度；新增源码在sibling目录，原refine、loss_controls、refiner_execution无改动，旧恢复指纹保持。

继续冻结原两轮epoch2、相同TRAIN4086首4组≤16窗、同原Adam/RNG/全6损失，两遍顺逆序临时前后向/Adam，更新丢弃，原文件/SHA不写；**测速仍用原full损失，不能混入删loss收益**。源是完成的原Refiner，不是loss500。可选编译路径失败不推广，完成的速度全保留。结果：speed.json/stage_replay.json/summary.txt；不重评DEV512，不接正式后端。

本地RTX3050小8×8×4真实训练初筛：reference.12157/staged_inputs.11162/staged_sync.11554秒窗；前向阶段、严格参数梯度、本身replay通过。不是200网格/L40S提升承诺。compiled本地Windows未实测。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git switch feature/shared-history-execution-speed-v2
git pull --ff-only origin feature/shared-history-execution-speed-v2
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/refiner_pipeline_speed_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/refiner_pipeline/run.sh
cat "$RUN_DIR/summary.txt"
```

脚本检测Refiner/loss实验运行时退出，不停止别的进程。中断丢临时probe，需新目录重试；不伪恢复训练步数。默认原epoch2路径可用REFINER_CKPT覆盖，profile自动包含reference与最快完成候选，但不自动选后端。

## 3. 双辅助同时删除：单独500步入口

`loss_minimal/` 新RANDOM、相同seed/初始化/原TRAIN4086首500分组/500步余弦/batch4-source128，**仅CE+Lovasz+ego R10**。direct source geometry与existence不只乘零，完全不计算它们的标签/损失。仍保留模型existence头以保住相同初始化顺序/结构，它无直接梯度，不会因zero-grad AdamW偷偷衰减。XY/yaw仍通过占据可微投影更新。初始零Refiner末层后续上游梯度由第二步真实测试检查。

500是partial pass，epoch cursor=1只代表500周期，不是完整TRAIN轮。完成一次DEV512，旧三路不重跑；新protocol/metadata隔离旧optimizer，Ctrl-C完整update后保存，同目录RESUME可恢复，不导入已学motion的旧epoch2。此入口**只准备好、未代用户启动**。

```bash
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/loss500_minimal_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/loss_minimal/run.sh
cat "$RUN_DIR/summary.txt"
```

恢复：`RUN_DIR=原minimal目录 RESUME=原minimal目录/last.pt bash tools/shared_history_experiments/loss_minimal/run.sh`。不同时运行训练与测速。

旧500对照删除existence：mIoU+.484/IoU+.598、Moving−1.521。两项同时删除没有测过，不能承诺后期追回Moving；新单路应对照已完成full的40.804106/55.824713/28.710151（同DEV512）。没有自动按dev选优/替换正式loss。

## 4. 完整数据训练轮数的判断

可以先固定1–3轮作为未来预算，不默认20轮，但本次没有启动或设置完整训练。一个full epoch约5140updates/20430windows，是当前20%×2（2060updates/8172windows）的约2.5倍窗口。原full-loss500独立余弦40.804，与两轮实验epoch1的41.493、epoch2的43.170不是同一scheduler曲线；原两轮本身1→2提升1.677pp，不能说已经饱和。按当前约.40s/window，一整轮仅compute约2.27小时，另加输入、评估、保存；少loss不保证有同样耗时/质量。

以后如决定full1–3轮，应事先定统一余弦周期、固定每轮DEV512，并看整体/动态指标及同模型Transport与Refiner贡献。不要把这次500已走完余弦的断点直接续成完整训练，也不要用反复查看DEV挑最优轮数包装为独立测试。
