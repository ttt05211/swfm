# CCR：共享空间上下文与训练侧先采样

## 本轮结论

训练的主要重复工作已能减少，但目前仍未达到“小幅精度损失”的目标。
本轮仅做本地真实 replay；不改原 epoch19、不运行服务器训练、不自动部署。
RTX3050 的倍率不能作为 L40S 的硬件换算系数。

## 方法与实现

保持四历史→六未来，在 canonical source/背景坐标中共享编码，六帧独立预测 ADD/REMOVE。
增加一次共享的邻域层：同 entity 内六面一阶/二阶邻域，保留完整 Z，不把不同 source 混在一起。
它没有六次 patch encoder；所有未来时刻读同一空间编码，但拥有各自的时序、query 和组合条件。

第一种随机初始化 + LayerNorm 的空间头没有追回指标，明确保留失败记录。
第二种从此前本地 CCR 初始化：新增空间支路为零初始化残差、没有额外 LayerNorm，
因此训练前可复现旧 CCR；随后另训练64遍。原 Local 的网络/权重不改变，无 KD/AE。

训练流程：

```text
固定历史 occupancy/visibility、因果 source 关联
     → 完整 canonical 支持域/手工描述/整数邻域（可复用）
     → 仅按 source/class 与 t0/past-only/halo 做因果分层抽样
     → 当前网络 motion → 样本的实时六帧投影/owner/fallback/GT edit labels
     → 抽样点及其唯一邻域依赖只编码一次
     → 六帧 readout → importance-weighted ADD/REMOVE BCE + 原 motion loss
```

完整候选域不截断，推理仍处理全部支持域。每个非空 stratum 都有非零抽中概率，
使用 N/k 权重恢复总体的和估计；归一化 loss 仍是具有有限样本方差/偏差的 ratio estimator，
不是逐像素等价于旧训练。未来 GT、预测分数和 future validity 不进入采样选择。
默认每个 static/dynamic role 1024点，若 strata 更多则至少每个 stratum 一个。

先采样之后仍使用完整静态人口的 road/sidewalk 冲突保护：另一类别没有被抽中，
不等于允许覆盖。动态投影、owner/fallback 和 GT action 标签每步重算。
静态世界点的已知 ego-query 冲突集不依赖预测运动，单独有界缓存，未来 ego 变更即失效。

固定分层索引也复用；与重新计算索引的样本、权重和 RNG 状态一致，不改变学习行为。
工程缓存只服务实现效率，不把缓存本身包装为论文的核心模块。

## 缓存合同及规模限制

`FixedCanonicalCache` 的内容键包含实际四帧 occupancy/visibility/pose、当前 source 顺序、
类别、中心、注册变换和网格，而不只看 scene/token。未来 GT、learned activations、
预测运动、owner、动态未来投影和训练标签不得存入缓存。
缓存中的 `labels` 仅指四帧历史 occupancy，`features` 仅指确定性历史手工描述。

RAM LRU 有配额；已知 ego 静态冲突集另有8MiB配额。可选磁盘缓存复用原仓库
`CausalGeometryCache` 的压缩、SHA、namespace、原子写入、配额及磁盘预留机制。
这是本机生成的可信 pickle artifact，不能导入不可信文件。损坏时直接报错，不静默覆盖。
磁盘实验关闭 descriptor RAM LRU，因此测量包含真实文件读取、解压和输入内容哈希。

8个常规窗口的完整描述/邻域/分层索引约67.42MiB，压缩约17.86MiB。按这8个窗口粗略外推，
20,430窗口可能新增约45GiB磁盘，实际分布未知。不能让全量描述驻80GiB内存，也不能
在用户100GiB总存储约束下未经容量核对就创建第二份全量缓存。
当前本地磁盘缓存总配额128MiB；没有自动建立服务器全量缓存。

## 质量：全部为开发样本，不是独立测试

原 Local 是全量训到 epoch19 的 head；本地新头只用 TRAIN16。两者训练预算不相等。
DEV12常规与4个压力窗口分别评估。阈值固定 ADD=.5 / REMOVE=.95，没有 DEV 扫阈值。

| DEV12 方法 | mIoU | MovingMicro | 相对原 Local |
| --- | ---: | ---: | --- |
| 原 Local epoch19 | 40.577554 | 31.846816 | — |
| 上一版点级 CCR，TRAIN16×64 | 39.738593 | 30.706335 | -0.838961 / -1.140480 pp |
| 随机空间头，TRAIN16×64 | 39.751266 | 29.879010 | -0.826289 / -1.967806 pp |
| 残差空间头，从本地 CCR 再训64遍 | 39.789673 | 30.703327 | -0.787881 / -1.143489 pp |

最后一项累计有2048次本地 head 更新，不假装与1024次更新是等预算架构对照。
相比其 frozen transport：DEV12 mIoU +0.139379、MovingMicro +0.015463；
TRAIN16 in-sample mIoU +0.283691、MovingMicro +0.152865。
这不足以归因为单一的数据量问题，也不是已经保住旧头的修复能力。
0.20pp 只是本轮“小幅下降”的诊断参考，不是用户已经批准的正式部署容差；本候选仍失败。

## 六帧推理

最后残差版本同卡交错测4个分层窗口，各路径重复2次。每次重建完整 canonical 证据、
邻域和候选，推理不使用训练 descriptor 缓存。

| 人口 | 原 Local 六帧 | 新 CCR 六帧 | 倍率 | 新 CCR FPS |
| --- | ---: | ---: | ---: | ---: |
| 常规2窗口 | 2.566698 s | 0.453794 s | 5.656× | 13.2219 |
| 高 source 压力2窗口 | 3.722642 s | 0.633198 s | 5.879× | 9.4757 |

FPS=6/平均六帧延迟；包括 fresh Strong/KTA prior、live motion、全部候选和六帧 dense 输出。
不含磁盘、最初 source extraction/registration、GT/metrics、预热；不是 raw-input E2E 或 L40S 数字。
当前 CCR 没有完整旧 frontier generation 职责，不能把倍率解释为等质量、等功能替换。

## 真实联合反传测速

同8个不同 TRAIN 常规窗口、batch4；完整 motion+head backward、clip、AdamW，
只更新可丢弃克隆，不保存科学更新。各模式一个预热batch、两个实测batch。
新的角色 BCE/因果 MC 与旧 column CE/GT分层采样不同，不宣称数值等价训练。
固定注册与 cold descriptor prefill 单列，不藏进“网络训练很快”的宣传数字。

首次预采样实测：旧 Local 0.159798，新残差 CCR 0.083036 秒/窗口，约1.924×。
后续分别实测 point/RAM、spatial/RAM、spatial/disk，并验证磁盘数组逐元素往返。
最后固定索引优化的实测数字：

| 路径 | 同轮旧 Local 秒/窗口 | CCR 秒/窗口 | 倍率 |
| --- | ---: | ---: | ---: |
| 点级头、固定输入 RAM | 0.161566 | 0.060694 | 2.662× |
| 空间头、固定输入 RAM | 0.154596 | 0.081829 | 1.889× |
| 空间头、固定输入磁盘，descriptor RAM=0 | 0.166308 | 0.106530 | 1.561× |

磁盘模式另保留约0.207MiB的静态 query 冲突元数据（上限8MiB），不把这部分说成完全零RAM。
数据见 `outputs/ccr_training_cache_final_20261006_v3/summary.txt`。
前一轮 point/RAM约2.307×、spatial/RAM约1.832×、spatial/disk约1.530×；两轮均有改善，
但不把短程两轮的差异全部归因为固定索引优化。
测速人口小、文件页缓存会影响磁盘模式，不是 NAS 冷读吞吐或全量训练 ETA。

## 入口和交付

- `real_motion/canonical_repair_context.py`：空间 head、因果采样与有界固定输入缓存。
- `tools/real_motion/pilot_p0_f9_ccr_spatial_warm.py`：质量、六帧速度和联合反传合并实验。
- `tools/real_motion/benchmark_p0_f9_ccr_training.py`：不重训的RAM/磁盘实际反传对照。
- `tests/test_canonical_repair_context.py`、`tests/test_ccr_real_replay.py`：包括真实CUDA。

完整相关测试包括真实CUDA、磁盘精确往返、未来GT隔离、全量/抽样投影一致性、
完整静态冲突保护、source/query反传及原Local回归。
最终104项测试通过（35.20秒）。
原 checkpoint/config 的 SHA 复核不变。所有新产物均在新的本地输出目录。

结论：训练侧“先采样、后投影标注”是有效的成本削减方向；目前精度未达可接受的小幅下降。
保留速度实现与完整失败记录，不自动启动20%/全量服务器训练，不再扩大本地架构搜索。
