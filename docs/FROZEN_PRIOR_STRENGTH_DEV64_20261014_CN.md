# 冻结先验强度诊断：原版 1 vs 固定 0.5

本轮只验证先验压制是否值得继续研究，不训练、不学习gate、不模拟噪声、不自动替换正式模型。锁当前完整GT epoch2/update10292，原DEV512内既定scene-balanced DEV64，同窗OCC+GT和STC+GT。没有内部ego头、导航或外部planner；GT future ego是显式条件。

每个窗口/输入模态只做一次完整六帧前向。原版logits直接复用，不减再加；诊断版 `z_half = z_original - 0.5 * log_prior`，即保留原残差、只改最终先验强度。Refiner的coarse/point prior、memory、Encoder、STWM、Strong和Transport投影都不变。不是同时重新训练弱先验模型，也不改变epsilon=1e-4，不扫系数、不根据DEV挑局部体素系数。

两模态预测完成后才读future occupancy和Moving标签；STC完整发布语义/全有效输入，不读取GT可见性。评分全网格，无camera/lidar metric mask或GT pose alignment。三个路由Transport/original_c1/diagnostic_c05报告1/2/3s及平均IoU/mIoU/MovingMicro；修正/损坏同时以Transport和原Refiner为基准，按wrong-free、false-occupancy、wrong-class分类。总语义改对体素不等于类均衡mIoU收益。

这是开发集机制诊断，不是独立测试，也不是新正式Camera结果；无收益不能完全否定空间自适应gate，有收益不保证学习gate能迁移或保GT。旧GT正式分数/模型不替换。此轮不拟合STC噪声。后续若从VAL提取噪声参数或选增广配方，该VAL已用于方法开发，不能再称独立测试；优先作者TRAIN/冻结前端生成TRAIN，或只用TRAIN固定污染配方，另留场景隔离验证。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export STC_ROOT="$PWD/data/stc_camera/compact"
export PRIOR_DIAG_DIR="$PWD/outputs/p0_f9_shared_history/prior_strength_dev64_$(date +%Y%m%d_%H%M%S)"
unset PRIOR_DIAG_RESUME
bash tools/prior_strength/run.sh && cat "$PRIOR_DIAG_DIR/summary.txt"
```

只需回传summary。详细1/2/3s与每窗整数统计在evaluation.json/state.json。若中断，保留原PRIOR_DIAG_DIR，设置 `export PRIOR_DIAG_RESUME=1` 再运行同脚本；新窗口完整后存盘，不重复旧窗口，不覆盖源权重/缓存。缺失STC历史拒绝，不替换人口。可选STC_ZIP显式核原ZIP，否则仍核compact extraction receipt/所选历史语义bytes。

入口保留原GT同位姿检查与数值容差，不切FP32或免除FAIL。原权重文件/Adam/RNG历史缓存只读，代码不改旧模型/trainer/正式评估或指纹。计时仅质量诊断wall time，非FPS。本地测试只能验证真实小网格前向、对照公式/因果输入与断点行为，nuScenes实际涨点需服务器结果。
