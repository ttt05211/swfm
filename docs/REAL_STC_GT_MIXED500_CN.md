# 扩大真实STC场景＋干净GT混合，固定500步

用户确认扩大STC训练场景、保留原remaining64评估；随后明确2000步太多，本次改为500，不自动扩训。原STC-only518窗×500结果保留，不导其权重/Adam。

## 数据与目标

- 原remaining64起点固定，并将这些起点所属**整个场景**排除训练。其余有VAL metadata的STC场景都成为候选（预计86场景，实际打印），不再只用旧18场景。只读旧paired实验的eval_keys及原GTepoch2 SHA，不读旧噪声统计/概率或模拟生成器。
- 干净GT来自原TRAIN20430候选池，与全部VAL场景隔离，固定seed随机化后按场景均衡无放回选择1000窗。不是拿留出的VAL GT回放。
- 每个update固定B4：STC、GT、STC、GT，窗口loss平均，严格1:1；500步各1000窗次。STC候选每轮无放回随机遍历、用尽再重复；如果候选多于1000，只实际采1000个不同STC窗口，不宣称全部候选已训练。输出候选/实际消费的不同窗口及场景数。所有实际source保留，B4不做source128截断。
- STC四历史=官方真实预测完整语义/all-valid/不读GT mask；GT四历史=原GT语义+历史lidar-known。两者目标都是各自随后六帧GT占据，GT未来ego明确条件。逐sample场景、next/prev链、关键帧间隔核验，无未来GT占据/annotation输入。

## 训练及评估

原正式GTepoch2/update10292初始化、新Adam/RNG、Encoder/STWM/Refiner全解冻，主体1e-5/Refiner1e-4，全6帧CE+Lovasz/H6/c1，不ego/motion/existence辅助，不额外模拟噪声/增强，不接另一个unified-source分支。总500步独立完整余弦，不续旧STC-only。

两套有界256MiB纯历史几何RAM+各256MiB帧缓存、CPU2线程/一组ahead，不存学习F/运动输出，不写旧cache或新bank；现场构建各模态真实组件/注册/KTA/Strong。名义RAM预算不是总进程RSS/峰值。

500后自动在同remaining64重评原GTepoch2及混合模型，OCC+GT/STC+GT各Transport/Refiner，标准full-grid IoU/mIoU/Moving。场景扩张与混合同时变化，此次是更充分的适配探索，不是1:1混合单变量消融，也不是理论上限。已使用VAL部分场景GT监督、留出也曾观察过，**非zero-shot/独立test/标准fullVAL新成绩**，原正式模型不覆盖。

## 运行与恢复

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export REFERENCE_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
export MIXED_STC_DIR="$PWD/outputs/p0_f9_shared_history/stc_gt_mixed500_$(date +%Y%m%d_%H%M%S)"
unset MIXED_STC_RESUME
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2 \
  && git merge --ff-only FETCH_HEAD \
  && bash tools/stc_mixed_adaptation/run.sh \
  && cat "$MIXED_STC_DIR/summary.txt"
```

Ctrl-C完整update/完整评估窗保存。原目录MIXED_STC_RESUME=1同入口恢复Adam/RNG/cosine/游标，不重置500周期。旧STC-only目录/协议拒绝作为新混合断点。无需先额外跑首检；`--max-updates`仅必要接口检查时暂停原预算。
