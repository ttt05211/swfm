# Waymo10：相同v2路径的2×2/4×1吞吐对照

当前L40S旧16窗探针单/双进程0.3506→0.1701s，2.062×，已通过六帧/概率/整数计数gate。用户随后恢复了并行全集：截至5088/39987，当前invocation累计吞吐0.235s/窗。单worker0.486–0.621s不用于算总ETA；剩余34899×0.235约2小时17分，负载变化会浮动。二者不同窗口，不用0.1701和0.235硬算性能退化。

用户授权进一步短测更多窗口并行。新入口只比较同一个v2实现的**2进程×2工作线程**与**4进程×1工作线程**，两者名义CPU工作预算均为4。进程内还有框架/预取线程，不能把这个乘积当作实际线程总数或CPU占用。

## 计时和安全

- 从原保存cursor起取相同64个连续窗口，默认AB/BA两遍；最多128窗。相同权重/support/阈值/图/缓存容量/chunk，唯一区别是进程/线程布局。
- 每个新worker独立做原路径first-block检查；全部探针窗完整六帧、概率、运动输入和两个分支整数指标/编辑数一致才计时。计时pass的整数指标继续核对，哈希在计时外。
- 包含实际历史准备、六帧预测、GT读取、整数指标和IPC；不是正式Dense Forecast FPS。启动/建图/gate/首窗warm单列。每pass清原始帧/历史LRU，只暖每worker首窗，保留滑动窗口的实际新帧miss。
- 两布局轮流执行，不同时运算。为避免重复CUDA启动，两组共6个worker常驻，但只有一组工作；比正式运行多持有idle模型/metadata/cache，OOM时关闭探针进程，保留旧计数，不能据此把正式4worker说成OOM。
- 源目录kernel lease在整个探针期间持有；源评估没安全退出则拒绝。不能同时恢复原评估做公平测速，也不能与其他GPU计算任务并行测速后声称独占吞吐。
- 原contract/state/权重只读并再次验SHA/字节。探针新目录只写receipt/speed/summary，不写评估state、不推进科学计数、不自动续评。普通旧resume指纹依赖一个文件都未改。
- 建议4×1的条件：平均至少快10%，且每个repeat都更快；否则保留2×2。短窗证据不能保证39987全场景倍率。

## 服务器操作

先找到主评估进程并对**它**发SIGTERM，不用kill-9，不杀worker：

```bash
pgrep -af '[p]ython.*eval_p0_f9_joint_surface_waymo_10hz_parallel.py'
# 核对含原 --out-dir 的主进程PID后：
kill -TERM <主进程PID>
```

等待原日志出现停止摘要、主进程退出；最多当前在途块还会完成。原计数取state.json，不取progress尾条。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git pull --ff-only origin feature/v22-surface-aware-ccr

WAYMO10_LAYOUT_SOURCE="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_parallel_20261010_050832" \
bash tools/real_motion/run_p0_f9_waymo10_worker_layout_probe.sh
```

发回`summary.txt`或`LAYOUT_SPEED`及最终推荐。无自动全评。源contract记录的配置、metadata路径、均值和执行开关均直接复用，原实现被别的任务改过时拒绝继续，不忽略哈希。

若4×1值得：新输出用原 stopped parallel目录显式迁移计数，跳过旧“v1单进程 vs4worker”重复探针。不是在旧目录更改参数resume：

```bash
RUNS="$PWD/outputs/p0_f9_joint_surface_ccr"
NEXT_OUT="$RUNS/waymo10_parallel4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNS"
nohup env WAYMO10_PARALLEL_OUT="$NEXT_OUT" \
  WAYMO10_PARALLEL_CONTINUE_FROM="$RUNS/waymo10_parallel_20261010_050832" \
  WAYMO10_PARALLEL_RESUME=0 WAYMO10_PARALLEL_SPEED_ONLY=0 WAYMO10_SPEED_WINDOWS=0 \
  WAYMO10_PROCESSES=4 WAYMO10_THREADS_PER_PROCESS=1 \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_parallel.sh \
  > "$NEXT_OUT.nohup.log" 2>&1 < /dev/null &
echo "PID=$! 输出=$NEXT_OUT 日志=$NEXT_OUT.nohup.log"
```

若4×1无收益：仍用原目录2×2的严格resume：

```bash
OLD="$PWD/outputs/p0_f9_joint_surface_ccr/waymo10_parallel_20261010_050832"
nohup env WAYMO10_PARALLEL_OUT="$OLD" WAYMO10_PARALLEL_RESUME=1 \
  WAYMO10_PARALLEL_CONTINUE_FROM="" WAYMO10_PARALLEL_SPEED_ONLY=0 \
  WAYMO10_PROCESSES=2 WAYMO10_THREADS_PER_PROCESS=2 \
  bash tools/real_motion/run_p0_f9_joint_surface_waymo_10hz_parallel.sh \
  >> "$OLD/nohup.log" 2>&1 < /dev/null &
echo "PID=$! 日志=$OLD/nohup.log"
```

本入口只做执行诊断，不改变10Hz的native+2/+4/+6下标评分，不改变训练slot clock，不改2Hz/nuScenes/6秒结果。

## 本地验收范围

CPU相关回归47通过、2CUDA跳过；真实RTX3050 CUDA环境相关30通过，含真正2/4个spawn模型、CUDA Graph及完整六帧/概率/运动/整数计数一致、旧接续、源目录kernel锁、失败关闭与只测不推进计数。CLI help、Bash语法通过。真实进程测试使用合成小网格/随机小权重，不是服务器布局倍率、Waymo质量或全仓CI；布局结论等待上述服务器64窗成对结果。
