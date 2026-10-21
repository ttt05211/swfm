# 冻结 FPN＋3D U-Net：扩展512窗与同口径测速

不重训、不改架构/权重/阈值；只接受已完成 TRAIN4086×1 的本组合 `epoch_0001.pt`，不加载旧模型或中途checkpoint。新入口独立放在 `tools/fpn_unet_eval`，不修改训练源码及其原指纹。

## 范围和指标

从完整VAL4369中，保留checkpoint记录的原64窗为前缀，再排除它们、对排序后的剩余scene/token做固定scene-balanced round-robin取448窗。严格512个不同窗口，OCC/STC同人口；缺文件直接报错，不做planner或STC覆盖筛减、不按分数选窗口。

分别输出 original64、additional448、all512 的 OCC/STC Transport/Refiner 的1/2/3s IoU、mIoU、MovingMicro；保存逐时刻/逐类报告。all512先合并两个分组的整数confusion再计算，不平均64/448的分数。额外显示重新评估原64与训练报告的差值，差异如实保留。

这是**原64＋新增448的扩展开发集**，不宣称与以前历史DEV512/495完全同人口；新增的是窗口，场景可能与原64重叠，不是独立scene-heldout或正式test。OCC/STC历史都用dense-valid；读取器只访问semantics，不读取camera/lidar visibility。GT future ego显式同时用于Decoder和全部投影，不做GT pose事后对齐。future GT仅在六帧预测完成后用于指标；无ego预测头。

## FPS边界

默认取上述512中的固定前32窗、每窗两次暖机测量，单GPU单模型、CPU4线程配置；OCC/STC都测。输出 `six_frame_ms`、`output_frames/s=6/秒每窗`、P90、峰值allocated/reserved显存。

- 计时内：实时KTA计算、**fresh Strong/CLEAR**、FPN、Source WM、history lift、Transport Prior、Memory Transport、U-Net、全部六帧argmax与CPU结果。
- 计时外：磁盘I/O、历史实例提取/关联/表示构建、初始输入上传、GT标签/指标、启动/暖机与checkpoint I/O。
- 不复用未来背景、learned features或前次模型输出。计时路径与质量路径做同窗口六帧逐体素比较，同时记录质量路径自身repeat漂移。FAIL原样报告，不靠放宽容差变成PASS；时间样本保留，但有FAIL不能称已验证相同质量路径的FPS。

这与之前声明的**prepared-history → 六帧CPU输出**边界一致，不是raw camera端到端FPS，也不表示旧FPS数值可直接套用。显存reserved包含常驻allocator；不是模型裸权重显存。

## 服务器命令（同步本次新增文件后）

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export FPN_UNET_EVAL_GPU=0
export FPN_UNET_EVAL_RESUME=0
export FPN_UNET_EVAL_DIR="$PWD/outputs/p0_f9_shared_history/fpn_unet_frozen512_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$FPN_UNET_EVAL_DIR")"
nohup bash tools/fpn_unet_eval/run.sh >"${FPN_UNET_EVAL_DIR}.log" 2>&1 &
tail -f "${FPN_UNET_EVAL_DIR}.log"
```

若原终端仍有`FPN_UNET_DIR`，会使用其已完成的一轮模型；若没有，会查找唯一已完成的`fpn_unet_20pct_1ep_*`。多个候选时列出并停止，明确指定：

```bash
export FPN_UNET_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_20pct_1ep_实际时间
```

全部完成后：

```bash
cat "$FPN_UNET_EVAL_DIR/summary.txt"
```

中断后保留原输出目录，设 `FPN_UNET_EVAL_RESUME=1`，重新执行同一个bash入口；完整窗口整数计数与完整测速调用可恢复，不改变源训练。完整checkpoint单独snapshot，源SHA和模型state前后只读核验。

代码只能在服务器数据就绪后实际运行；本地小网格功能测试不构成512窗质量或L40S FPS结果。不自动完整训练、方法推广或再试结构。

本地安全yoloe新增8项＋原组合8项共16 passed（13.14s），含真实CPU/CUDA新forward/fresh速度路径、fresh Strong调用、只读冻结加载、实际整数指标及停续流程；Bash-n通过。首次检查发现人口选择tuple/list拼接与小测试输入非正式网格尺寸问题，分别修正新代码和测试夹具后全过，未放宽指标/输出容差。原tools/fpn_unet源码零diff，旧checkpoint指纹不受新目录影响。
