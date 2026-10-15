# 干净 GT / dense-valid / 全解冻500步

补齐旧实验缺失的控制组：从原始正式 GT epoch2 初始化（不是 clean500、污染或真实 STC 适配权重），
SharedEncoder、STWM、SemanticRefiner 全部解冻。主体 LR1e-5、Refiner LR1e-4，新 Adam/RNG，
同旧 clean Refiner500 的 seed、TRAIN20%4086 人口、前500 groups、B4/source128 名义分组、H6、500步余弦。
只有六个未来时距完整 CE+Lovasz；无 ego/motion/existence 辅助、无模拟污染、无真实 STC 训练。
500步只是部分 TRAIN 遍历，不是一轮。c=1、epsilon、几何和模型结构不变。

TRAIN 历史采用 dense_valid；OCC 评估保留原 lidar-known，STC 评估全 valid、不读 GT 可见性。
GT future ego 是显式条件，GT 未来占据仅作监督。兼容历史几何缓存只读；学习特征在线生成，不建新 bank。
不读取拟合噪声参数；旧 paired workflow 只提供原权重、配置、数据身份和精确 remaining64 名单。

唯一新训练500步后自动评同旧 remaining64 的 OCC+GT / STC+GT Transport/Refiner，
报告 IoU/mIoU/MovingMicro，并只读列旧 clean 与 empirical 的结果、同窗差值，不重训对照。
与 clean 的比较检验可训练范围；与 empirical 同时改变范围和污染。
本实验不能单独证明 known 是原因，也不是原正式训练原样续训；remaining64 已被反复检查，不是 pristine test。
原模型与旧结果不覆盖，不自动扩大训练、挑 best 或推广正式指标。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export REFERENCE_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
export CLEAN_FULL_DIR="$PWD/outputs/p0_f9_shared_history/stc_clean_full500_$(date +%Y%m%d_%H%M%S)"
unset CLEAN_FULL_RESUME
bash tools/stc_clean_full/run.sh && cat "$CLEAN_FULL_DIR/summary.txt"
```

Ctrl-C 在完整 update 边界保存。恢复须设置同一个 CLEAN_FULL_DIR，再 `export CLEAN_FULL_RESUME=1`，
运行同一脚本；总预算仍是500。若训练已完成，只续未完成评估，不增加训练。
使用旧 paired 的 clean/frozen/GTepoch2.pt 并验证 SHA；不改旧 trainer、模型文件或缓存 fingerprint。
