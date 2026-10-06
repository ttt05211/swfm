# Point CCR 完整支持域的无损执行优化

## 本次选择与方法边界

继续在 Point CCR 的 canonical/source-space 修复路线优化，不回退旧逐列 Local。
方法仍为：四帧因果证据构建 canonical 点集，每个点编码一次；利用 live source
latent 和六帧不同的运动/ownership/context 输出六份 ADD/REMOVE 决策，再合成六帧。
没有把六帧修复强行设成一样，也没有删候选、缩小 halo、降低空间分辨率、改变阈值。
旧 epoch19 Local、Point CCR checkpoint 和 optimizer 均不修改。

GAST 的官方实现将历史 occupancy 的 batch/frame 维合并后共享编码，并将未来帧维
合并进行采样等张量操作。借鉴的是规则布局、共享计算和批量执行的经验，不复制其
BEV encoder、GSCA 或 temporal/spatial 网络。
参考：https://github.com/chenst27/GAST/blob/master/models/gast.py
这不表示 GAST 的所有未来计算都并行，也不把不同计时边界的 FPS 当作同口径比较。

## 同时完成的修改

1. ABI5 编译 CPU 内核：完整 lattice/history gathering、27维固定描述子组装、
   ADD/REMOVE ownership 合法性及 road/sidewalk 冲突检查。多次 NumPy 排序、
   unique/isin 和临时矩阵变为融合遍历/整数 lookup。
2. 固定线程池按 source/entity 构建证据；大完整域按 horizon 独立投影检查。
   按原顺序收集结果，输出数组 horizon 写入互不重叠。小训练子集不强行并行调度。
3. 因果固定描述子的有界缓存支持新内核；每次查询仍检查历史真实内容哈希。
   缓存不含未来监督、learned feature、learned pose 或训练计算图。
4. 可选窗口批量 Point MLP：保留逐窗口 loss 权重，actor 按窗口 source population
   偏移；live source/future latent 梯度仍可传回 motion。训练的 GEMM 合批可能改变
   浮点舍入，不能宣称与旧 optimizer 轨迹逐位一致。默认关闭，先测量。
5. 一次服务器运行同时比较完整六帧 FPS 和少量真实 motion+repair 联合反传。
   联合训练测速仅使用可丢弃克隆，不保存科学更新、不启动长训练。

世界坐标变换、SE(2)、floor 仍沿用原 NumPy float64 操作顺序。C++ 主要融合整数
查表/合法性与有限 FP32 描述子算术，不使用 fast-math 或浮点几何重关联。
Windows 深路径 MSVC 编译也修复；Linux 编译和 L40S 收益需服务器入口实际确认。

## 本地真实回放结果

设备 RTX 3050 Laptop，用户提供的 replay.zip，而非随机模拟数据。
32 个真实窗口（TRAIN16、dev12、4个高 source stress）检查完整描述子、投影计划、
概率及六帧 dense 输出逐字节一致；dev12 的 IoU/mIoU/MovingMicro 均一致。
这说明后端不改变当前 Point CCR 预测，不代表 Point CCR 已恢复旧 Local 精度。

| 项目 | 原执行 | 融合+并行执行 | 吞吐提升 |
|---|---:|---:|---:|
| 六帧延迟 | 336.6 ms | 270.0 ms | 1.247x |
| FPS = 6 / 平均六帧延迟 | 17.83 | 22.22 | 1.247x |
| 冷固定输入联合训练 | 163.9 ms/窗口 | 125.4 ms/窗口 | 1.307x |
| 暖固定输入联合训练 | 63.0 ms/窗口 | 60.2 ms/窗口 | 1.047x |

六帧比较使用相同6窗口、3遍交错顺序，含 fresh ALL-six Strong/KTA、live motion、
完整 canonical 输入、全部六帧修复及六份 dense 合成。排除磁盘/初始 source extraction
与 registration、GT/metrics、预热和正确性哈希；不等于 raw-sensor E2E FPS。
canonical 输入约 78.2→44.3 ms，六帧投影/合法性约 50.4→18.8 ms。
fresh Strong/KTA 本地约152 ms，仍是剩余主要耗时。

训练比较使用8个 TRAIN 真窗口，batch4，各臂16个计时更新（64窗口曝光），
包含真正 motion+repair backward、gradient clipping、AdamW；先做一个预热更新。
冷输入每步重建；暖缓存128MiB，内容哈希仍计费，预填时间独立报告。
GEMM合批、CUDA非确定性、温度和小样本波动可能影响小收益；暖缓存约5%的结果
不应被解释为数量级提速，也不外推15/20轮训练总耗时。

本地证据：

- `outputs/ccr_exact_execution_20261006_b/speed.json`：32窗口精确检查、FPS、dev12指标。
- `outputs/ccr_exact_execution_training_20261006_c/speed.json`：较长真实联合训练测速。

当前 Point CCR 在服务器原20%×3轮实验仍比旧 Local 低约0.827 mIoU pp /
0.582 MovingMicro pp。这批无损执行优化没有弥补质量差距，不能批准部署。
L40S 的 fresh 六帧40+ FPS尚未实测，不能由3050比例换算得到。

本次相关回归114项通过（含真实CUDA回放、全六帧输出、整数/边界规则、梯度、
batch与optimizer/RNG下一更新恢复），shell语法检查通过。三个已知warning分别是
测试fixture的list-to-tensor和空Moving类别均值；不表示已跑全仓库CI。

## 一次服务器检查

在 OccFM、无其他同卡任务时：

```bash
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_point_ccr_v18_fps.sh
```

使用已确认路径的 epoch19、Clean-E14、完成3轮 Point CCR checkpoint。一次比较：
同20个窗口、18个scene-balanced+2个高source stress、3遍、fresh/cached Strong，
原执行与 CCR fused/parallel；每个窗口检查六帧/latent/probability字节一致。
另从完整TRAIN选择6个scene round-robin+2个source stress窗口，对比冷/暖输入、
原执行/融合并行/批量头的少量真实联合反传。全局batch仍4、完整source不删除。
结果统一在新唯一目录 `summary.txt` / `speed.json`，不覆盖旧实验。
`CCR_JOINT_SPEED=0` 可关闭额外反传测速。编译失败或正确性失败直接报错，
不会静默切到 NumPy 后声称 native 收益；不会自动重训或自动推广新后端。

## 训练开关与恢复

Point CCR Python训练入口提供：

```text
--ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --ccr-batched-head
```

原默认 NumPy/逐窗口头保留。新开关纳入训练合同，改变合批/执行配置不会在旧断点
静默恢复；应明确使用新配置启动新的受控训练。相同合同下 optimizer/RNG/游标恢复
与下一更新一致性已有测试。不要为了测速将原19轮 Local 权重换成未过质量门槛的CCR。
