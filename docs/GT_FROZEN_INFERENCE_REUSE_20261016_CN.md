# 冻结最终模型：GT推理高层执行复用

## 范围

只针对GT历史推理，源模型直接加载已完成GT/STC128测速的 `frozen/final_model.pt`，核receipt、state SHA、源results和metadata。保留norm192、half KTA、年龄权重、静态软支持、固定后验bias；不改模型参数、训练、loss、前端、阈值或几何规则。不写CUDA/Triton、不开compile，不改FP64几何和BF16网络精度。原 `stc_frozen_speed`、正式trainer、评估和冻结资产不改。

目标是尝试接近40 output-frame FPS，不保证成功。新入口是独立只读A/B，不直接推广到正式推理。

## 高层执行候选，对照只运行同一个reference

| 执行 | 改动 |
|---|---|
| reference | 原128窗测速的完整 `engine.fresh`，H6 |
| readout_reuse | 读出块共用一次 `x.float()`和小常量tensor；prior连续slice替代每块高级索引。原grid_sample、MLP、chunk大小及算式不变 |
| geometry_reuse | 复用已上传历史语义/known；同次前向复用历史位姿、逆矩阵、full-grid坐标；source点合并上传后按原顺序切回；coverage/8邻域常量不重复上传；Strong直接用原CPU GT pose输入，避免GPU→CPU绕行 |
| combined | 同时启用上面两类 |
| combined_cpu4 | combined加已有Strong native majority调度线程2→4；单独标记CPU预算变化，不修改其native实现 |

静态软支持仍每次新算四帧投票与相同mixture，移除的只是未使用的 `int(use.sum())` 统计及其隐式GPU同步，并利用初始one-hot的原语义替代重复argmax。source逐层叠加顺序、原splat与source_query操作不变，不用新的多source归约或FP64融合。

没有跨次缓存Strong、inverse pose、static prior、learned F、memory或logits；这些都在每次完整六帧计时内重建。准备边界仅沿用原历史关联/注册/读取坐标/初始H2D；没有把原计时内的大头移出去。无未来occupancy读取、质量分数或重新拟合。前四路CPU Strong工作线程默认仍2，只有明确命名的cpu4一路为4，并逐窗单独核六帧Strong bytes。

复核发现：原Predictor构造函数已经绑定 `ParallelNativeMajority(min(workers,4))`，原基准默认workers=2。**不存在“原路径没有挂native majority”的接线遗漏**；不要将CPU4调度候选说成修复遗漏或新增底层优化。

## 测试和计时

固定源128窗顺序的前32窗（可显式128），同窗先预热所有候选，两个pass按backend顺序正反交替，默认各重复2次。单GPU单模型进程，不混强制GC/旧模型双路诊断。全6帧argmax及CPU输出计入；单独前4窗serialized profile不计入FPS。报告总时间换算FPS，不平均逐窗倒数。

reference重复自身与候选都对照同窗首次reference，所有byte差异保留、timing不筛选、无容差变更。只要原版repeat或候选strict byte失败，严格gate仍FAIL；提速和输出差异分别报告，不把少量差异自动当无损。速度改善只有本地证据，不能外推L40S；不按此筛质量/自动切默认。

## 本地结果

RTX3050 Laptop / 4GiB，真实原始ZIP中固定8个GT窗口，完整H6，每pass两次重复。最终三路同轮A/B：reference **680.10ms**，combined **643.47ms**（**1.0569×**），combined_cpu4 **641.27ms**（**1.0606×**）；两pass均更快。CPU4相比combined只再省约2.2ms，不夸大线程扩展收益。前4窗profile显示Refiner约198.6→179.0ms、memory约138.1→128.8ms。

8窗CPU2/CPU4六帧Strong byte完全一致；原版自身repeat和两候选最终语义最多均差5体素，strict byte gate仍FAIL，不称无损。独立小网格测试完整prior/static-support/memory和非零残差readout tensor逐元素一致。local v2/v3的四路分解也表明两类复用均有小收益；最终以上述三路同次结果为准，不把历次最快耗时拼接成一组。

新增7项＋原冻结/复用/majority回归共 **65 passed / 13.20s**；Git Bash脚本语法通过。无新依赖，无原模型/训练缓存/正式推理路径改动。

这不是L40S或128窗FPS，没有达到“服务器回到40FPS”的实测结论；需要服务器同窗口A/B。原模型和正式推理默认路径未替换。

## 服务器

先结束其他GPU任务。复用刚完成的GT/STC128目录，不需再次冻结、评全集或训练：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export GT_FPS_SOURCE="${STC_FPS_DIR:-}"
# 若重开终端变量为空，脚本会查找最近shared_gt_stc_fps128_*；也可显式填原目录。
export STC_FPS_GPU=0
export GT_REUSE_WINDOWS=32
export GT_REUSE_DIR="$PWD/outputs/p0_f9_shared_history/gt_inference_reuse_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/gt_inference_reuse/run.sh --backends reference combined combined_cpu4 > "${GT_REUSE_DIR}.log" 2>&1 &
tail -F "${GT_REUSE_DIR}.log"
```

结果 `cat "$GT_REUSE_DIR/summary.txt"`，逐窗完整timings/profile/byte差异在window_*.json。完成32窗后若值得再确认，设置128和新的GT_REUSE_DIR重新运行；不覆盖旧测次、不拼不同执行状态的timings。仅reference+combined时可额外传 `--backends reference combined`。
