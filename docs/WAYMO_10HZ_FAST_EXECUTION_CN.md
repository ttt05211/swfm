# Waymo 10Hz 无损执行加速（2026-10-10）

## 保持不变

原2Hz及原10Hz的实现文件/指纹未修改，正在运行的2Hz不需停止。
固定均值、四总历史、六张完整dense输出、全native锚点、标签映射、slot clock、ADD raw0.5/REMOVE-off保持原样。
10Hz仍按I²-World的零基未来下标1/3/5评分，即native +2/+4/+6，名义0.2/0.4/0.6s，不能称物理1/2/3s。
不训练、调阈值、减少候选、改K/半径、改FP64几何或改变网络矩阵分块8192。

## 新执行

1. 有界单帧几何LRU：复用component提取、配准用世界点及canonical静态点。默认1024MiB、最多32帧；token+实际occupancy/visibility/pose内容键。缓存值只读，不写磁盘或nuScenes缓存。
2. 构建motion state和CCR registration使用同一历史component，不再重复提取。四历史滑动时只新增一帧几何；匹配/ICP、Strong、canonical lattice、surface Atlas、learned features、实时未来投影均重新计算。
3. Surface描述在相同全类FP64坐标变换后，分4096行独立查询/拟合，复用现有4-worker CPU池；每个cKDTree子查询自身1线程。仍原K16、半径2.5、同算式/行内归约顺序，避免大邻域数组竞争内存带宽。
4. 当前六帧预测完成后才调度下一窗口纯历史几何，最多一个后台任务；不在worker运行模型/CUDA/未来GT指标。只在主线程读数据。缓存构建不持全程锁，日志统计不会隐式等待预取完成。

`history_io`在新入口包含历史几何就绪等待，不是纯读盘。新progress还拆开state/tubes/Strong、live registration、motion/layers、canonical support、Atlas build、邻域/拟合、六帧projection与phase。worker累计时间会重叠，不能当串行wall time相加。

## 检查与测速

服务器首次新运行默认先测相同连续16窗口：旧/new实际Strong+motion+CCR+六张dense，交替两次计时；运动输入、Transport、完整概率和六张dense逐字节检查。不读未来GT、不保存科学更新。
每次计时清空几何LRU，只预热首个窗口的四历史，后续仍一新帧/窗口，不能把16窗口全暖缓存当真实稳态。图捕获/逐字节检查在计时外。
输出`speed.json`及`WAYMO10_PAIRED_SPEED`，随后自动继续同一固定10Hz质量评估；任何一致性失败停止。
这不是正式Dense Forecast FPS、不是完整服务器平均耗时承诺；2Hz未结束时并发会有CPU/GPU竞争。
`WAYMO10_SPEED_WINDOWS=0`仅跳过短程计时，首个质量窗口仍完整独立reference exactness gate。

本地120000合成表面点、相同4 CPU worker，原describe均值625.588ms，新207.769ms，3.011×，全部descriptor bytes一致。
仅内部CPU阶段微基准，不是实测Waymo质量、L40S吞吐或模型FPS。原始本地输出位于`outputs/waymo_surface_descriptor_microbench.json`。

相关CPU回归135通过、18项CUDA相关跳过；现有RTX3050 CUDA环境另跑105项全部通过，含实际静态Graph重放、概率/六帧字节一致、CPU/native原路径和旧Surface/Strong/递推回归。
实际模型整数指标、重复/空/叠层邻域、缓存只读/预算/单次并发构建、日志不阻塞预取、场景边界、中断迁移和普通续评均覆盖。Bash语法及CLI help通过；不是全仓CI或完整真实Waymo验收。

## 从已停止的原10Hz接续（等2Hz跑完后）

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git pull --ff-only origin feature/v22-surface-aware-ccr

FAST_OUT="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_fast_$(date +%Y%m%d_%H%M%S)"
nohup env \
  WAYMO10_CONTINUE_FROM="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_20261009_233328" \
  WAYMO10_FAST_OUT="$FAST_OUT" \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_fast.sh \
  > "$FAST_OUT.log" 2>&1 </dev/null &
echo "PID=$!  目录=$FAST_OUT  日志=$FAST_OUT.log"
tail -f "$FAST_OUT.log"
```

原目录只读并在kernel lease下拍下contract/state SHA及JSON快照；新输出校验原实现哈希、人口、数据inventory、模型/配置/environment/阈值等完全一致。
只允许CPU worker数与新增无损执行信息变化；原共享文件变化或2Hz→10Hz混接拒绝。不要删除lock文件或绕过验证。
整数confusion与已完成窗口迁移到新contract，首个新增窗口重新验全部输入/概率/六帧；不会重算已保存的完整前缀，也不覆盖原state。
kill -9只能恢复最后一次保存，每8完整窗口保存，可能重算未保存尾部；不能把最后一条progress当作保存游标。
旧prefix和新运行的累计时间可能混合；对比新速度看`speed.json`或新`progress.jsonl`，不是累计平均。

## 再次恢复加速目录

```bash
FAST_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo10_fast_实际时间
nohup env WAYMO10_FAST_OUT="$FAST_OUT" WAYMO10_FAST_RESUME=1 \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_fast.sh \
  >> "$FAST_OUT.log" 2>&1 </dev/null &
```

此时不要同时设置`WAYMO10_CONTINUE_FROM`。普通resume仍严格同contract，worker/cache/执行参数变化拒绝。
优先Ctrl-C或SIGTERM，完整窗口边界保存；不修改任何checkpoint/optimizer/RNG。
