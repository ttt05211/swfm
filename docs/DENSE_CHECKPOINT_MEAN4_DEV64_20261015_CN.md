# 固定四断点等权平均与同 remaining64 对照

入口：`tools/dense_checkpoint_average/run.sh`。仅单卡推理，不训练，不改变原四个 checkpoint、Adam、RNG、报告或缓存。

固定来源：原 dense 双卡两轮目录的 `update_010000.pt`、`epoch_0002.pt`（10292），及对应第三轮目录的 `update_012000.pt`、`update_014000.pt`。默认按最近具有12000/14000的第三轮目录定位，前两轮目录从其 contract 读取，不按分数选择。可用 `DENSE_EPOCH3_DIR` / `DENSE_DDP_DIR` 显式指定。严格核验原协议、同训练来源、epoch2 SHA、模型配置、相同有序64窗口；缺任何指定断点直接报错，不以 last.pt 替代。

沿用旧平均实验的纯算术 helper：所有可学习参数在 CPU float64 中等权求均值，各1/4，再回原 dtype；固定 buffers 必须完全一致，只复制。没有 BatchNorm 统计平均，也不执行旧 Surface 网络。导出 `mean4.pt` 标记 evaluation_only/resume_allowed=False，不含 optimizer 或训练 RNG，不能当原训练断点恢复；它是一套模型，不是四次预测 ensemble。

结果读取与公平比较：

- 优先读取每步 `eval_update_*.json`，其次原 `results.json`，最后 checkpoint 内已有 evaluations；仅接受完整64窗、同有序 keys 和相同 contract 的记录。
- epoch2 优先采用第三轮目录已完成的10292 dense baseline。
- 历史表保留分数及其 known 标记。原两轮10000的 OCC 仍可能是 lidar-known，不会与后续 dense 分数直接混比。
- matched 表统一 OCC/STC dense-valid；完整且兼容的旧路由直接复用，仅缺失或 known 不一致的模态补评。通常补10000 OCC，再评 mean4 OCC/STC；未完成的14000评估会自动补评，不替换权重。
- 所有新推理采用现有 SharedTrackPredictor、GT future ego、四历史至六未来、full-grid、1/2/3s IoU/mIoU/Moving；STC不读取GT历史mask，无Pred、alignment、阈值或系数搜索。fresh 与旧 cached 入口并非逐字节一致性声明。
- 已反复查看这64窗，属于开发比较，不是独立测试，也不能保证平均提分。无自动选best/推广/扩大。

服务器同步后运行：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export AVERAGE_GPU=0
export DENSE_MEAN_DIR="$PWD/outputs/p0_f9_shared_history/dense_mean4_$(date +%Y%m%d_%H%M%S)"
unset DENSE_MEAN_RESUME
bash tools/dense_checkpoint_average/run.sh && cat "$DENSE_MEAN_DIR/summary.txt"
```

输出 `mean4.pt`、`summary.txt`、`comparison.md`、`comparison.csv`、详细 `results.json` 与每个补评模态的整数 state。Ctrl-C 保存完整窗口；在相同 `DENSE_MEAN_DIR` 下设 `DENSE_MEAN_RESUME=1` 再执行同命令，已完成部分不重预测。无旧 track cache writer lease，因此不需要开放原训练缓存写权限。

本地三个小数据测试覆盖等权算术/原文件只读、错误来源和人口拒绝、旧known标记、复用dense baseline、真实小网格网络及旧full-grid引擎中断/恢复与完成后零重预测；不是实际 nuScenes 质量结论。
