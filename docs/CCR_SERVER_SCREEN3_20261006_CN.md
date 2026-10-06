# Pointwise CCR：20% TRAIN × 3轮

用户确认先训3轮。使用 pointwise 版本，不扩大 spatial 版本，不蒸馏、不用AE。

## 固定实验合同

- TRAIN20430 里按已有 train-only 场景/时间分层规则取4086个唯一窗口，3轮每轮遍历一次，共12258窗口曝光。
- 4历史→6未来；epoch19 Local 的运动网络完全冻结，新 point CCR 随机初始化。
- 整个3轮预算余弦下降，无固定LR tail；最后一轮候选，不按dev选择best。
- 每轮dev64监控，最后一次dev512，与完整旧epoch19 Local在相同population比较。
- CCR ADD=.5、REMOVE=.95；旧Local仍为(.5,.5,.95)。不搜索阈值。
- GT直接监督实际预测运动投影下的ADD/REMOVE结果，训练候选抽样不能读取未来GT。
- 训练先在完整因果域按source/class/history状态分层抽样，再做live投影、监督和特征编码。
- N/k恢复总体和的期望；有限样本归一化仍有抽样方差/偏差，不声称与旧目标数值等价。
- 六个未来时间的修复可以不同；REMOVE只恢复可见owner下层，不删除其他source。

CCR支持域不完整覆盖旧frontier GEN，不宣称完整功能等价。最终报告包含静态修复、动态修复、
CCR joint、完整旧Local，以及IoU/mIoU/MovingMacro/MovingMicro和逐horizon/scene指标。
建议0.2pp精度容忍仅作诊断，不是自动部署许可。一次3轮试验失败不会自动扩训。

## 速度和内存

旧固定因果几何缓存只读、零新增写盘。新的固定历史描述子缓存：默认磁盘16GiB、RAM1GiB，
后台有界队列写盘，磁盘保留2GiB。只缓存四帧观测/注册派生的固定输入与抽样索引，
不缓存未来GT、learned features、预测轨迹或owner/修复标签。

达到配额时不删缓存、不丢训练窗口、不降候选域；继续现场计算。因此不能保证每条都warm hit。
可通过 `CCR_DISK_MIB=0` 禁用新增描述子磁盘写入（保留1GiB有界RAM）。
原几何缓存和新描述子缓存不是同一种格式，不应人为混用/覆盖。

每步日志包含live motion/render、固定输入hash/cache、抽样投影/encoder/loss、反传和optimizer。
冷构建和input wait计入实际训练记录，不能把小样本warm benchmark倍率直接当服务器ETA。
最终同卡配对六帧FPS包含fresh prior、live motion、完整CCR候选/encoder和六个dense输出；
不复用描述子缓存偷缩FPS边界；不含磁盘I/O、初次source抽取/注册、GT/metrics和warmup。
重复六帧输出SHA校验在计时之外，不以单帧峰值或平均逐窗口FPS替代6/平均六帧延迟。

## 启动和恢复

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_point_ccr.sh
```

读取用户已确认的路径；dev64 manifest必须与epoch19中的冻结dev512 identity/order匹配，
full4369 V18 cache严格按key对齐，绝不取前512。缺文件直接报[MISSING]，不猜新路径。

Ctrl+C/TERM：完整更新后原子保存 `last.pt`，每32更新另有周期保存；崩溃/kill -9只能恢复
最近已保存更新。恢复不会重新校准prior或重置optimizer/RNG/cosine。

```bash
CCR_RESUME=/你实际输出目录/last.pt bash tools/real_motion/run_p0_f9_point_ccr.sh
```

恢复使用新输出目录，旧实验只读。更改epochs、population、网络/目标/采样预算/实现版本时
拒绝续训；不能把旧Local/height-field checkpoint当CCR断点。

结果在本次输出目录 `summary.txt`、`screen.json`、`progress.jsonl`；原epoch19权重不变。

本地验证包括99项相关测试（真实RTX3050/replay开启）、旧height-field入口回归，
三轮CLI精确恢复、失败更新不覆盖有效断点、CPU/CUDA真实point头反传、
输入缓存异步持久化与GT隔离。服务器真实3轮和L40S速度必须由本次运行给出，未伪造。
