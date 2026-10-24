# 当前 FPN＋U-Net：固定 FREE −1 的新场景448窗复核

独立入口 `tools/fpn_stc_bias448/run.sh`。不改原模型、训练、age 或 ALL4369 默认入口。

## 固定的一个变化

同一完整两轮 checkpoint、同一 STC age0.5、同一前向的 FP32 logits：

`Z_calibrated[:, 17] = Z_original[:, 17] - 1`，其余语义类不动。

网络只算一次；然后原版 argmax 与校准版 argmax 各读出全部六未来。没有修改卷积 bias、残差倍率、运动、先验或 Memory。不是再训练。

这个参数来自本地原64中前32场景的**有标签目标域输出校准**；不是无标签适配或 zero-adaptation Camera。已在原64做过多轮研发，不能将原64当新测试。固定 −1，不再搜索，不按本次448挑参数/模型。绑定源 SHA `cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f`，拒绝套用到另一模型。

## 448窗怎样选

从完整 VAL4369/150场景元数据中，**排除原64窗所在的全部64个场景**；剩余场景内，按排序后的 `(scene, t0)` 做确定性场景平衡 round-robin，取448窗。不只是排除64个窗口，不沿用旧512/remaining4305/外部 planner 人口。不按标签或成绩筛选，不遇到缺 STC 就悄悄裁剪。

这448只是未参与本次标量选择的新场景复核，仍属 VAL，不是从未开发过的封存测试集。不能与 ALL4369 直接相减；新旧两行必须同448窗比较。

## 输出

保留 GT future ego 条件、四历史六未来、dense-valid、完整网格1/2/3s。先完成新旧两路全部预测，再读取未来 GT occupancy 和真实 GT Moving support，仅用于评分；不读 visibility mask 作输入或评分筛选。

`summary.txt` / `evaluation.json`：baseline 与 FREE−1 的平均和逐时距 mIoU/IoU/MovingMicro、差值，负结果也保留。由完整窗口整数混淆矩阵累加后算指标；不是逐窗分数平均。质量评估耗时不是 FPS。

源 checkpoint、模型参数、旧缓存只读，独立源快照与输出。每8窗和正常退出保存完整窗口状态；SIGINT/SIGTERM 完成本窗再退出，`STC_BIAS_RESUME=1` 从同一 `STC_BIAS_DIR` 继续。无自动扩大、部署或修改正式模型。

## 服务器

```bash
conda activate OccFM
export FPN_UNET_FULL_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_full_2ep_20261021_112202
export STC_BIAS_GPU=0 STC_BIAS_RESUME=0
export STC_BIAS_DIR="$PWD/outputs/p0_f9_shared_history/stc_free_minus1_new448_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/fpn_stc_bias448/run.sh >"${STC_BIAS_DIR}.log" 2>&1 &
echo "PID=$! OUTPUT=$STC_BIAS_DIR LOG=${STC_BIAS_DIR}.log"
```

完成后 `cat "$STC_BIAS_DIR/summary.txt"`。如果需续跑，使用原目录并设 `STC_BIAS_RESUME=1`，不要创建新目录。没有19+/29+达标保证；确认迁移效果与 Moving 代价后再决定是否做全集校准诊断。
