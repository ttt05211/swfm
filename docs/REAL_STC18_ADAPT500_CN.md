# 18场景真实STC有监督适配500步

这一条不使用仿照生成的污染，也不加载noise_model.json。旧profile/evaluation.json只提供已经固定的18/132场景名单；旧paired结果只提供原GTepoch2的SHA和相同remaining64评估起点名单。

- 输入：官方STCOcc发布的真实四帧语义，dense-valid，不读取GT历史camera/lidar mask。
- 目标：同场景连续后六帧GT语义。逐窗核四历史+t0+六future的sample next/prev链、scene identity和0.25–0.75s关键帧间隔，名义2Hz时距，实际future时间另记。
- 训练：所有18场景中metadata完整窗口（缺文件拒绝、不补序列），从原正式GTepoch2/update10292初始化，Encoder/STWM/Refiner全解冻；独立新Adam/RNG，主体1e-5/Refiner1e-4，原H6全6帧CE+Lovasz ONLY，c1，无ego头/运动或existence辅助/额外增强。
- 固定500更新。数据不足500批时固定seed逐遍洗牌循环；报告实际完整数据遍数。B4/source128沿用GT metadata名义分组，实际STC组件全部保留、sources记录，不截断凑预算，不宣称严格实际source上限128。
- STC组件/关联/注册/KTA均从真实STC重建，不拿GT历史geometry当STC几何。复用已有历史准备/Strong helper，不调用augmentation。CPU2线程、一个B4前瞻；512MiB纯历史RAM几何缓存与有界帧缓存，不写新缓存bank、不缓存学习F/运动/prior/memory。Strong未来投影live主线程，GT未来ego是明确条件。
- 500后自动评原GTepoch2与适配模型，在同remaining64上分别OCC+GT/STC+GT Transport和Refiner，标准full-grid IoU/mIoU/MovingMicro，未来GT语义/Moving只评分。旧clean适配500不是原模型基线，不能代替原epoch2评估。

这是使用18个VAL场景真实STC历史及GT未来标签的**有监督目标域适配**，不是zero-shot，不是完整标准VAL或pristine独立test。其余132场景与适配场景隔离，但此前分数已经看过。新实验不能覆盖原正式模型/主表；没有自动扩预算/推广/选择阈值。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
# 如果原Git代理403，临时直连（不改永久配置）；成功后再运行下一段。
git -c 'url.https://github.com/ttt05211/swfm.git.insteadOf=https://github.com/ttt05211/swfm.git' \
  fetch https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2 \
  && git merge --ff-only FETCH_HEAD

export REFERENCE_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
export REAL_STC_DIR="$PWD/outputs/p0_f9_shared_history/stc_real18_adapt500_$(date +%Y%m%d_%H%M%S)"
unset REAL_STC_RESUME
bash tools/stc_real_adaptation/run.sh && cat "$REAL_STC_DIR/summary.txt"
```

无需用户先跑额外首检。Ctrl-C完整更新保存last.pt（Adam/RNG/固定500余弦游标），评估按完整窗整数保存。续训/续评：设置原REAL_STC_DIR、REAL_STC_RESUME=1后同入口；不导入模拟污染模型或更改已保存预算。`--max-updates 8`只供必要接口检查，属于原500周期，恢复不重置日程。原训练器、checkpoint、历史cache目录均不修改。
