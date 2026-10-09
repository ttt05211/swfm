# Waymo 10Hz：有界并行评估与原计数接续

## 当前结论

用户暂停了昂贵的10Hz评估。旧fast-v1目录为：

```
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo10_fast_20261010_040604
```

最近256窗平均0.4739s；父级prepare0.2189s/evidence0.1858s，其中state/tracks/tubes/Strong0.1695s、surface邻域拟合0.0850s。读历史约0.0013s，head约0.0153s，故不能只加读盘预取或继续改网络。

本次新增两层**执行优化，不改变方法**：

1. serial-v2：同一FP64整批坐标计算；投影冲突由相同稳定最小距离规则的C++循环处理，不全局排序；缓存仅原始历史点/只读排序；同配准算式重用当前source树；四帧tube纯CPU准备与fresh Strong重叠；同K16/半径2.5表面拟合的C++严格归约，保留NumPy sqrt/clip/FP32转换。
2. parallel：默认**两个spawn进程，各两个CPU工作线程，同一张GPU**。各持有独立冻结模型、CUDA Graph及有界历史LRU，处理连续8窗口的小段。最多两段在途，跨进程只传每窗口两个分支的整数混淆矩阵、耗时和编辑数，不传大体素或CUDA张量。主进程按人口顺序提交计数。

一边做CPU几何时，另一个进程能继续准备或使用GPU。不是在线融合/ensemble、不是新训练方式，也不是单窗口延迟的两倍提速。**评估吞吐不能替代论文固定Dense Forecast FPS。**

## 已有本地证据与范围

- serial-v2在RTX3050大网格模拟中整体仅1.035–1.047×，不把局部内核收益当作整网收益。
- 新双进程：真实RTX3050 CUDA，200×200×16合成网格、16个请求source、相同16个连续窗口、两遍交替。旧fast-v1单进程4线程0.723398s/窗，新serial-v2双进程各2线程0.372681s/窗，**1.941067×**。包含原始历史/几何/六帧预测/未来GT读取/整数指标/IPC；startup+首窗预热单列，不含在稳态数值中。
- 两遍单进程分别0.753978/0.692817，新并行0.360343/0.385018s/窗。每遍清几何LRU，只暖各worker首窗，不把全人口暖缓存冒充连续吞吐。
- 所有模拟窗口的运动输入、概率、六帧dense哈希及两个分支的每窗口整数计数一致。每个新进程另做独立旧路径first-block核验。
- 这是**随机小模型、模拟数据、笔记本GPU**，不代表真实Waymo精度、L40S倍率、完整39987耗时或论文FPS。服务器默认先跑短程，没加速不自动启动全评。

可复现实验：`benchmark_waymo_parallel_local.py --out-dir <新的诊断目录> --device cuda --windows 16 --repeats 2`。仅生成自己拥有的模拟文件/随机诊断权重，不能作为论文训练权重或分数。

## 不变的科学契约

4总历史含t0→6未来；原5/6/8/12/14冻结均值；weighted ADD raw0.5/REMOVEoff；完整native anchors；I²-World eval_time=1/3/5，即native+2/+4/+6、名义0.2/0.4/0.6s。模型训练slot clock原样，不重采样/重定时。

未来pose显式条件。各worker完成六个预测并检查形状/独立exactness后，才访问未来occupancy。指标无额外visibility mask。无新学到的特征缓存、未来预测缓存或持久大几何缓存；仅单帧历史LRU。旧2Hz/10Hz/fast-v1实现文件未修改，nuScenes缓存和权重只读。

## 服务器先只测速

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git pull --ff-only origin feature/v22-surface-aware-ccr

PAR_OUT="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_parallel_$(date +%Y%m%d_%H%M%S)"
WAYMO10_PARALLEL_CONTINUE_FROM="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_fast_20261010_040604" \
WAYMO10_PARALLEL_OUT="$PAR_OUT" \
WAYMO10_PARALLEL_SPEED_ONLY=1 \
bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_parallel.sh

echo "接续目录：$PAR_OUT"
```

只读原contract/state，校验原实现/权重/人口/数据/配置/环境相同后，在新目录保存原SHA、JSON快照及重新签名的整数前缀。**实际接续位置取state.json，不取progress尾条1510**；kill -9只恢复最后周期保存状态。

短程为相同下16窗、两遍交替旧fast-v1单进程与新并行，先六帧/概率/运动/计数gate；包含GT/指标以测真实eval，但不将测速窗加进评估计数、不训练/调参。`speed.json`有每遍、worker PID、初始化耗时和边界。

请发`WAYMO10_PARALLEL_PAIRED_SPEED`。如果速度不值得，不必接着全评；旧输出完全保留，可原入口resume。若源文件哈希不匹配，拒绝迁移，不能通过删除契约字段强行继续。

## 确认收益后，继续同一并行目录

```bash
nohup env WAYMO10_PARALLEL_OUT="$PAR_OUT" WAYMO10_PARALLEL_RESUME=1 \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_parallel.sh \
  > "$PAR_OUT.log" 2>&1 < /dev/null &
echo "PID=$! 日志=$PAR_OUT.log"
```

新终端需要把`PAR_OUT`设为刚才实际打印的新目录。不要带CONTINUE_FROM或SPEED_ONLY到resume。普通resume严格同实现/线程/进程/chunk契约，并直接使用保存的checkpoint路径，不重新寻找均值。

每8完整窗口周期保存。Ctrl-C/SIGTERM让主进程停止提交，收完最多2×8窗口的在途结果并按顺序保存；不是立刻退出，可等少量窗口结束，勿kill -9。worker失败/计数不合法不提交错误窗口，只能恢复已保存连续前缀。没有39987个future堆积，也没有乱序结果洞或双计数。

`throughput_seconds/window`与progress的`seconds`为主进程有序小段wall/窗口，可能有一段等待、后一段近零的批次效应；请平均至少128/256新窗口。`worker_seconds`/model_stages是单worker耗时，会重叠，**不能相加称总wall time或CPU/GPU利用率**。原前缀串行耗时与新并行worker累计耗时不直接合并计算倍率。

允许单独诊断4进程×1线程，但不是默认；进程/线程总工作预算最多4，防止10核下多层池超卖。改变进程数需要新输出与显式计数迁移，不在旧目录静默resume。

## 本地验收

相关CPU回归97通过、3项CUDA跳过；真实CUDA/native/双进程/Graph及旧Surface/Strong/6帧递推回归147通过。覆盖byte精度、稳定冲突、ICP排序/树重用、早期/末端窗口、错误整段原子拒绝、暂停/续评、旧fast前缀只读迁移、仅测速不推进计数、从保存路径找权重及拒绝更改resume参数。不冒充全仓CI或真实Waymo服务器验收。
