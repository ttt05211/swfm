# 冻结 STC norm192 版本＋单进程128窗推理测速

## 冻结内容

版本名 **stc_gt_norm192_r1_20261016**：clean1000主体，half-pair/LS KTA、历史年龄权重、静态软支持、原后验校准，Refiner192参数固定为target臂final200。没有静态配准、动态补形状、额外训练、内部ego头或新阈值。

冻结的全集证据：STC+GT / ALL4369，mIoU **19.190065**，IoU **29.016647**。仍声明无标签目标历史适配、反复开发，不称完全zero-adaptation或独立test。只验证此STC+GT版本，不推断OCC+GT或STC+Pred。

`tools/stc_frozen_speed/release.json` 固定源主体SHA、192参数与校准JSON的语义SHA及验证分数。服务器入口要求读取已完成的旧 `all/results.json`，核4369/150、H6、同源/known/GTego/fullgrid、实际评估资产与分数。原JSON、checkpoint和缓存只读。

测速前，输出独立 `frozen/final_model.pt`：将192个数**一次合入完整模型权重**，保存模型结构、完整state_dict、固定posterior bias；没有Adam/optimizer，`training_allowed=False`。另有receipt记录合并后的tensor fingerprint、文件SHA和原全集报告SHA。约8.5MB，不是新特征bank。

测速模型从这个新冻结文件**重新加载**，不只是使用内存里临时改动的模型。`load_frozen(folder, device)` 可校验并加载；推理使用本模块 `engine.single` 或prepared/FPS路径，保留完整方法。原训练权重不覆盖，原12/100轮头等试错权重不进入。

## 修复的执行问题

新冻结单路推理：

- 不再每窗切换原版／norm参数。
- 不再为每窗重复计算旧组合的Refiner。
- 不在每窗调用 `gc.collect()`；保留Python正常自动回收，不永久关闭GC。
- 显式释放本窗大型临时引用，保持有界输入缓存。

旧 `tools/stc_norm_final` 双路审计脚本和四个旧恢复指纹文件不改，避免破坏已完成结果及旧恢复契约。新冻结单路是独立入口，不把“删了对照计算”包装成网络算法提速。

## 一次128窗跑两种不同计时

### A. 单模型 FPS，主结果

单GPU、**一个模型进程**，默认2个CPU工作线程；不会创建2/4个模型进程。使用完整VAL metadata中的固定scene-balanced128窗，不按难度/分数或byte parity筛窗，也不评全集。

计时边界沿用之前 Refiner retime 的 prepared-history 口径：

- 计时外：磁盘I/O、历史CC/source关联、注册与读取坐标、初始历史输入H2D。
- **计时内**：live KTA/年龄权重、共享Encoder/STWM、每次新算Strong/CLEAR、Transport/static support、投影与aligned memory、一次Refiner、校准、六帧argmax与完整CPU语义输出。
- 不缓存Strong背景、learned F、future prior或memory来省略它们的计算；历史输入准备只是纯几何与输入上传。
- GT future ego 是输入条件；无预测ego头、未来占据标签、指标计算进入计时。
- 默认H6、每窗2次计时重复。第一窗3次warmup，其他窗1次shape warmup，不计入统计。CUDA计时前后同步。
- 前8个固定窗口另外做一次带同步的分段profile；这组串行profile**不加入FPS**。

同时报：

```text
mean_six_frame_ms = 总计时时长 / 总调用次数 ×1000
window_fps        = 总调用次数 / 总计时时长
output_frame_fps  = 6×总调用次数 / 总计时时长
```

这是每窗预测六帧的FPS，不是raw-camera端到端FPS。不能将多进程总吞吐或各窗FPS的算术均值当单模型FPS。

BF16可能存在少量边界体素翻转；每次六帧与旧正式前向的norm路线对照，strict byte FAIL仍保留FAIL，不放宽容差、不切FP32、不丢窗口或timing。所有128窗都纳入实际执行时延，严格一致性单独报告；FAIL时不标成“逐体素一致参考FPS”。

### B. 含输入准备的推理速度 A/B，不叫FPS

同128窗交替先后顺序，比较：

1. 旧双路对照＋逐窗强制GC；
2. 冻结单路＋普通自动GC。

两者均包含历史I/O/source准备/完整推理，不含未来标签、质量指标、结果写盘。强制GC自身单独计时，不能把所有差距都归因于GC；总差距同时包含移除旧组合重复读出。这也不是严格复现旧四进程质量评估的整机吞吐。

## 服务器命令

先结束其他GPU训练／评估，避免竞争污染延迟。固定单卡，不运行并行评估工作流。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export STC_FPS_GPU=0
export NORM_VALIDATED_DIR="$PWD/outputs/p0_f9_shared_history/stc_norm192_final_20261016_125218/all"
export STC_FPS_DIR="$PWD/outputs/p0_f9_shared_history/stc_norm192_frozen_fps128_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/stc_frozen_speed/run.sh > "${STC_FPS_DIR}.log" 2>&1 &
tail -F "${STC_FPS_DIR}.log"
```

复用已导出的ZIP，默认自动定位；必要时设置 `STC_REPLAY_BUNDLE`。DATAROOT/DEV_CACHE/DEV_INFO/STC_ROOT沿用之前路径。输出目录新建，不覆盖旧结果。

结果：

```bash
cat "$STC_FPS_DIR/summary.txt"
ls -lh "$STC_FPS_DIR/frozen/"
```

每8窗打印 `FROZEN_STC_FPS n /128`，不是4369。原始每窗timings/parity/profile保存window_*.json；results/summary增量写入。该轻量128窗入口不自动拼接失败测次/恢复不同计时状态；若中断，已有结果和冻结文件保留，用新目录重新测，不覆盖。后台nohup可关终端。

## 本地实测与测试边界

- RTX3050 Laptop4GiB真实8窗H1联调complete：完整输入旧1.4075秒/窗→单路1.0353秒/窗，约1.36×；GC本身约0.0575秒/窗。不能将这组速度或FPS冒充L40S/128结果。
- 同8窗部分BF16比较有差异，最大5体素/六帧，原样记录，全部timings保留。CPU小几何严格一致性测试通过，不据此抹掉GPU byte FAIL。
- 真实单窗H6从冻结文件重新加载、全部六帧测速complete；该窗strict byte通过。不是128窗代表性结论。
- 新6项＋原norm192/parallel/replay/verifier/boundary共36项回归通过；覆盖freeze参数范围、拒绝覆盖、原来源只读、完整评估资产绑定、fresh Strong每次重建、无未来标签读取、只legacy调用强制GC、六帧输出、正确FPS汇总和不筛byte失败窗。
- 没有代启动服务器、改变官方计时边界、声称128实测已完成或发布新的正式FPS。
