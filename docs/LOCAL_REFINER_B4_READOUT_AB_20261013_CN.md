# 固定 B4 Refiner 合批／读出：本地 A/B

## 结论

只完成隔离候选及本地模拟测速，**没有修改正式 full_gt/refine trainer、缓存或服务器 checkpoint**，没有自动推送/部署候选。
最值得保留的是 `slice_queries`：连续切片替代逐块高级索引。合批在小网格有速度收益，但现有数值门槛失败，且网格增大在本地卡上反而变慢。
一次全体积上采样没有稳定明显的优势，不能默认它是最大的提速点。

## 实现范围

独立目录 `tools/shared_history_experiments/refiner_batch_readout/`：

- `reference`：原 sequential B4/H6。
- `slice_queries`：只将 `prior[:,:,ids]` 换成连续切片，保留逐块 FP32 cast 和 point MLP。
- `xy_upsample`：Z折入batch，一次XY双线性上采样，再连续切片读出。
- `volume_upsample`：直接体积上采样，XY×2、Z尺寸不变，减少重排拷贝。
- `packed_reference`：合并4窗×6时距，仍原grid_sample读出。
- `packed_upsample` / `packed_volume`：合批结合两种上采样。

保持同权重、全局batch4、H6、完整CE＋Lovász，逐窗逐时距归一化不改；packed时间条件重复[0..5]四次，不生成[0..23]。
Packed把各窗不同的几何图保留到一次multi-root backward；source/geometry仍独立逐窗。局部替换仅实例级forward上下文，异常也恢复，不改训练入口源码。
上采样与常规采样数学上对应，但浮点坐标重建、BF16逐块cast及参数梯度归约顺序不保证一致，不能标为无损。

## 环境和计时

Windows非base yoloe；Torch2.6.0+cu118，RTX3050 Laptop 4GiB，网络BF16。每候选从相同随机非零末层权重及Adam初始化开始；两遍正反顺序、预热排除，CUDA同步端点。
所有局部计时包括Refiner前后向、全部H6 CE/Lovász、clip/Adam、标签准备，不包括Encoder/STWM/几何准备/磁盘I/O。
随机memory/prior/标签，不是训练checkpoint或真实nuScenes窗口，没有质量分数/FPS。

| 模拟全网格 | 原版秒/窗 | 连续切片倍速 | 合批原读出倍速 | 合批＋体积上采样倍速 |
| --- | ---: | ---: | ---: | ---: |
| 32×32×16，chunk2048 | .039133 | 1.062× | 1.402× | 1.354× |
| 40×40×16，chunk4096 | .053368 | 1.129× | 1.281× | 1.312× |
| 64×64×16，chunk8192 | .231699 | 1.080× | .357×（变慢） | 未测 |

64网格合批原读出peak allocated3357MiB，原版1314MiB；合批XY上采样2974MiB且倍速.185×。这不是OOM证明，也不能唯一归因显存压力；算法选择/工作集/本地显存资源均可能影响，未做算子定因profile。不能外推L40S也会变慢/必然更快。
扩展版只在32/40测体积上采样；初版64只有5候选，保留负结果，未用新实现改写旧报告。

另做8×8×4、16维encoder/8维Refiner的完整网络测试fixture（不是生产规模）：包含实际SharedEncoder/STWM/Transport prior/memory/Refiner/loss/backward/Adam，预生成历史输入。
两次原版约.114–.115s/窗；连续切片1.055–1.069×，合批原读出1.348–1.381×。这是小图调度收益，不能作为整网服务器提速结论。

## 数值结果：失败仍为 FAIL

- 32/64：连续切片单步loss/gradient/logits及多步model+Adam通过，原版重复通过。
- 40：连续切片单步通过，但多步model+Adam FAIL；原版自身重复也FAIL（max5.2273e−5，3元素）。不能把所有差异归因候选，也不能豁免candidate FAIL。
- 上采样：32/64单步logits通过，但梯度/多步状态FAIL。40单步logits FAIL，max2.4414e−4；合成强prior下argmax未变不等于真实checkpoint质量不受影响。
- Packed：所有模拟尺度的单步参数梯度及多步状态均未通过。完整小fixture同样有多步FAIL（最大约.0028）；没有采纳。
- 预算固定：loss rtol2e−6/atol3e−7，梯度3e−5/2e−7，logits及model/Adam2e−4/2e−5；没有放宽。

9项CPU实际张量测试通过（正则格点/Z保留、非零读出、B4时距/窗口隔离、梯度、临时hook恢复、完整B4训练路径）。连同原full_gt/Refiner回归共24passed in13.28s，原训练源码无diff。
通过这些小测试不代表200×200×16 L40S冻结checkpoint门槛通过。

## 原始报告（本地临时目录，不上传大缓存）

- `tmp/refiner_b4_readout_cuda64_20261013_a.json`：初版5候选，warmup1/repeats3，含完整tiny replay。
- `tmp/refiner_b4_readout_cuda32_20261013_b.json`：7候选，warmup2/repeats5，含完整tiny replay。
- `tmp/refiner_b4_readout_cuda40_20261013_c.json`：非2幂网格，7候选，warmup2/repeats5，仅局部测速。

后两份含源码SHA、总显存及计时前free显存。报告始终含全部负结果。正式训练继续原H6。
若继续，优先准备同冻结GT checkpoint的L40S实际B4 `reference vs slice_queries` 只读测速；合批/上采样需要明确数值变化的质量验证，不能直接作为无损resume。
