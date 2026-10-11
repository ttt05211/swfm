# 冻结 GT 三轮 checkpoint：DEV512 软输出与同窗推理测速

2026-10-11：已完成 DEV512。hard Transport/ADD-only/ADD+REMOVE 的 mIoU 为
38.472382/37.889726/37.908709，soft 为39.544267/40.321095/40.211083。
软 ADD-only 的 CCR 净收益 +0.776828pp，但其六帧334.836ms，相比硬106.159ms仍慢。
用户随后取消单纯接口重测，又授权实际软推理提速；最新短测入口见下节。
边界v2将历史实例提取/匹配/H2D移到计时外，保留live Strong/KTA；整数质量只读复用，旧结果不改写。尚无新版L40S测速，不宣称速度改善。

## 最新：软推理专用优化与短测

新增隔离的 `soft_inference/fast.py`，不修改训练 renderer、权重、缓存内容或恢复指纹。

- 六未来按三帧合批；六张 uint8 语义网格统一回传 CPU，减少逐帧同步。
- 同一次 forward 复用 actor/class/合法动作行和几何常量；不跨窗口缓存 learned pose、plan 或概率。
- ADD-only 不执行恒零 REMOVE；保持原有 owner、fallback、源覆盖和 ADD 顺序。
- 推理合成原位更新新生成的 scratch 概率体，不复制完整体、不保存反传用中间量。
- 仍然18类、完整网格、原八角插值及 float64 几何，不删点、不降低精度、不调阈值。

实时 KTA 识别直接因果构造及旧 `upgrade_v1_record_targets` 的精确舍入公式；先严格
核验输入，再按已知公式实时重建，避免测速时悄悄替换冻结模型的输入。未知差异拒绝，
不放宽容差；全部六帧仍需输出 byte parity。

本地41项专项/相关回归通过。CPU合成单源640000体素 render+argmax 微测：ADD-only
3.292→3.460s，ADD+REMOVE3.410→3.246s。不是整网、没有GPU，不能外推L40S倍率；
CPU未显示稳定通用加速。新GPU路径是否值得用由下面同窗测量决定。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
bash tools/joint_occ_experiments/soft_inference/run_optimized.sh
```

自动找最新完整 `soft_dev512_*` 的冻结质量记录，验证 checkpoint、精确512窗口、整数
计数和原设置。只在新 `soft_speed_optimized_*` 目录保存短测速，不重评分512窗口。
旧质量目录若用了自定义名称，显式指定
`SOFT_RETIME_FROM=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/实际完整质量目录`。

32个同窗口×2个交替顺序，五路线：硬 ADD+REMOVE、旧软 ADD、旧软 ADD+REMOVE、
新软 ADD、新软 ADD+REMOVE。每条路线独立实时重算Strong、WM、CCR、六帧合成及CPU
输出。初始化/预热和首窗串行分段定位不计入速度；分段定位不是吞吐或GPU利用率。
已完成DEV512指标属于旧质量读出，不能把短测32窗一致扩写为新版512窗均已验证。

`summary.txt` 同时列六帧 ms、输出帧/s、两遍一致性与 `SOFT_OPTIMIZATION`。
候选只有六帧逐字节一致、两遍均更快、平均至少1.1倍且保留10%显存余量才过门；
任何失败保留FAIL，无自动替换部署或自动新实验。保留旧软、新软和硬路径。

Ctrl-C 后等完整测速窗口保存，按打印的新输出目录恢复：

```bash
SOFT_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/soft_speed_optimized_实际目录 \
SOFT_RESUME=1 bash tools/joint_occ_experiments/soft_inference/run_optimized.sh
```

下面原 `run.sh` 是第一次完整DEV512质量评估入口，已完成质量时无需再跑。

只评已有随机初始化 GT-only 三轮模型，不训练，不修改正式 Surface CCR 平均模型或其 FPS。
默认来源：`outputs/p0_f9_joint_occ_ego/gt3_horizon_float_resume_20261011_151144/last.pt`。
新评估目录保存独立 `source_snapshot.pt`；原权重、Adam、RNG、完整历史缓存只读。

## 质量

精确采用源 checkpoint 的 512 个评估窗口，不做 planner 交集、不补窗口。
固定四历史、六未来，报告 1/2/3s 的标准全网格 mIoU、IoU、MovingMicro。
未来 GT ego 是 GT 配置的条件；未来占据/运动标注仅评分使用，不决定输出。

同窗报告六种读出：hard/soft × Transport、ADD-only、ADD+REMOVE。
hard 保持 ADD0.5 / REMOVE0.95；soft 使用既有训练概率合成并 argmax 得到语义网格，
无重新校准、阈值搜索或自动挑选路线。soft ADD-only 仅将 REMOVE 概率置零。
此读出与此前 DEV64 loss/action audit 完全同口径，不把小样本优势直接当全集结论。

## 测速

完成 DEV512 后，在其中固定场景均衡 32 窗上，测 hard ADD+REMOVE、soft ADD-only、
soft ADD+REMOVE；每窗各路线独立重新前向、生成全部六帧，两个交替顺序计时。
每窗每路线预热不计时；CUDA 同步计真实墙钟耗时，包括完整六帧 CPU 语义输出。

包含：实时重算 Strong、WM、CCR 投影与读出、六帧合成；不计：磁盘读取、历史
实例关联/表示/Surface 描述准备、标签/指标、预热/graph capture、checkpoint 写入。
不把历史缓存中的 Strong 背景省略后冒充实时 Strong 速度。
所有测速输出逐体素验证等于原缓存质量路线六帧，失败保留 FAIL，不放宽。

这是同窗暖推理诊断，不是原始相机端到端 FPS，也不替代旧 256 窗×3 正式速度。
六帧 latency(ms) 与 `6/latency` 输出帧 FPS 同时报；显存 reserved 含公共 allocator/graphs。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export SOFT_OUT="$PWD/outputs/p0_f9_joint_occ_ego/soft_dev512_$(date +%Y%m%d_%H%M%S)"
bash tools/joint_occ_experiments/soft_inference/run.sh
cat "$SOFT_OUT/summary.txt"
```

发回 `summary.txt` 即可，详细整数计数/逐窗测速保存在 `state.json`、`evaluation.json`。
Ctrl-C 在完整质量或测速窗口后保存，不重跑已保存窗口；恢复原目录：

```bash
export SOFT_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/soft_dev512_实际目录
SOFT_RESUME=1 bash tools/joint_occ_experiments/soft_inference/run.sh
cat "$SOFT_OUT/summary.txt"
```

只复制和评估已完成的 GT 三轮模型；若原数据、架构或源码契约改变，会拒绝续评。
不在运行中的训练目录写评估结果，无新特征 bank，无自动重训或方法推广。
