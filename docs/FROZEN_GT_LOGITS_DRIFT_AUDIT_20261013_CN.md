# 同位姿logits失败：一窗只读定位，不放宽门槛

服务器原冻结头第一窗：背景0差异、F字节一致，prior max1.1920929e-7、memory max7.1525574e-7均数值通过；logits max0.0390625、argmax差2体素，严gate失败。
ego头训练断点不存在，尚未提取新bank或优化头，原GT主体不变。
Refiner的3D卷积及point读出确实采用BF16；微扰跨量化边界后放大是待检假设，不直接把此报表当成根因证明或无损。

新增独立`tools/scene_ego_head/frozen_gt_diagnostics/`，不改frozen_gt/main或正式trainer的任何指纹、旧失败报告与容差。
冻结完整GT最终epoch2，只重放同一个DEV首窗，无头优化、无完整DEV打分、不训练、无精度自动切换。

检查：

1. GT、重建入口各自重复；比较F、运动XY/yaw/Q、pose、source targets/yaw、prior、memory、logits及argmax变化。
2. 完全同一prior/memory，Refiner两次调用，区分相同输入执行漂移。
3. 各自prior/memory的BF16 Refiner与首层input/time/3blocks/MLP读出trace；MLP trace只首chunk，全logits仍完整比较。
4. 统计memory、coarse prior和point prior转BF16后实际变化的元素数。
5. 仅Refiner的FP32参考，局部禁autocast及TF32，结束/异常均恢复；与两份输入及原BF16输出比较，不报告新的正式质量指标。
6. 参数摘要/source快照SHA、hook及临时精度恢复，无修改checkpoint/Adam/旧缓存。图层trace在CPU，有额外内存，不是FPS。

`audit.json`保留全部阶段的固定rtol2e-4/atol2e-5、dtype/stride/mismatch数量，以及变化体素所在horizon和top1-top2 margin。
`summary.txt`仅给关键重复/共用输入/FP32指标；仍FAIL的都保持FAIL，不自动判定或采纳新后端。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2
export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export AUDIT_DIR="$PWD/outputs/p0_f9_shared_history/frozen_gt_logits_audit_$(date +%Y%m%d_%H%M%S)"
bash tools/scene_ego_head/frozen_gt_diagnostics/run.sh && cat "$AUDIT_DIR/summary.txt"
```

保留失败目录`frozen_gt_scene_ego_20261013_180911`，本次不恢复它、不改HEAD_EPOCHS、不删审计或训练契约。
元数据/源模型初始化仍有启动耗时，但dense forward只一个窗口若干次，无512/全集指标；不把此时间当实际单窗推理FPS。

本地验证：5项测试通过（9.13s），覆盖CPU/CUDA小图、相同输入/原版重复漂移记录、异常时hook及autocast/TF32/精度设置恢复，以及模拟CLI的源文件只读。Bash语法及diff检查通过；没有本地nuScenes数据，尚无服务器根因结论。
