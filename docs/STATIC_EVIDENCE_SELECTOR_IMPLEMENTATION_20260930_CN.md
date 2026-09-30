# 历史静态证据选择与搬运：一次有预算的 learned screen

## 已冻结的研究问题

保留 Clean-E14 V18 的全部预测，新增模块只回答：历史中实际观测过、
现在被 V18 预测为 free 的静态证据，哪些整 patch 值得重新放入未来视图？
这叫历史证据补全，不宣称生成从未观测过的新物体。

已有同一 dev64 审计给出的整 4×4 BEV patch **GT 辅助选择**上限是
ΔmIoU +2.287614 pp、添加 semantic precision 74.43%、18 个场景全部正收益。
全部历史静态证据直接放入仅 +0.052746 pp。因此本次只学习选择，不再改变形状、
搜索 anchors、训练 AE、加入动态 dormant/birth 或重复 oracle 梯度阶梯。
上限使用 GT，并不是 learned predictor 的成绩，不能承诺得到相同涨点。

## 单一结构和数据流

```text
六帧历史 occupancy + mask_lidar + ego poses
  → 各帧投影到未来 query，最新可见 free 清除旧静态证据
  → 固定 4×4 BEV patch，保留全 Z；只候选 V18-free 静态证据
  → 中心与上下左右 cross5 × 六帧 = 30 个历史 token
      每 token：18 类比例、可见率、高度均值/标准差、occupied 比例
冻结 V18 的未来预测 + 历史 proposal + 位置/horizon → 1 个 query token
  → 2 层时空 Transformer，width64 / heads4 / FF128 / dropout0
  → 一个 keep logit，sigmoid >= 0.5 保留整个候选 patch，否则弃权
  → 原始历史 voxel / 语义搬运，add-only，绝不覆盖 V18 occupied
```

同一正式函数构建训练与推理的特征，特征 API 无 future GT 参数。
投影使用 V18 已有协议允许的 future ego poses；并非预测车辆未来轨迹。
历史动态语义可作为拒绝证据输入，但 proposal 只能包含静态类别。
不另接 dense decoder、生成头或多个分支。V18 所有参数冻结。
六个未来 horizon 均可部署；冻结正式指标按 1/2/3s 汇总。

keep head 初始化为 p=0.05（零权重），即所有候选拒绝，输出逐 voxel 等于 V18。
这只是安全初始化，不保证训练后的 overall mIoU 不降低。
由于任何动态类别 prediction 都不被改变，冻结的 dynamic-class Moving
交并集保持逐项相同；即使错误静态添加落在动态 GT voxel 上也保持不变。

## 只有一个损失

每个 patch 的候选 voxel 由 GT 计算 correct-semantic 数 C 和错误数 W。
一个 importance-corrected BCE：

`L = sum(w * [C * softplus(-logit) + W * softplus(logit)]) / sum(w * (C+W))`。

它在固定 p>=0.5 的整 patch 决策下是 correct-minus-wrong voxel utility 的代理，
不是 mIoU 的精确可微优化。无需生成一个全 free 的 dense 分类分布。
每窗口每 horizon 最多抽 16 个 patch，按 C>W / C<=W 均衡抽样；
逆抽样权重 `stratum_population / sampled_count` 恢复原 population，
不把人为 50/50 比例当真实先验。不用 focal、pos_weight 或补充 loss。
GT 仅用于训练监督和评估计数；`--predict-only` 不读取 future occupancy GT。

## 一次 screen 与停止规则

- screen：train 1024 个 scene-balanced windows，固定 1024 optimizer updates。
- batch=512 **patches**，不是 512 windows；不要拿 update 数换算完整数据集 epoch。
- AdamW lr=3e-4、weight_decay=.01、clip=1；CUDA bf16 / CPU fp32。
- 同一冻结 dev64，全 population 每 128 updates 评估，无 threshold sweep。
- 仅保存 `best.pt`（部署）和 `last.pt`（optimizer/RNG 完整续训）。
- best selection 先要求固定 gate，再按 ΔmIoU 选择；若全部训练候选退化，保留
  update0 的 V18 身份基线，但明确报告最后 learned candidate，不能算成功。
- 硬门槛：dev64 overall ΔmIoU>=+0.5 pp，1/2/3s 各自非负，Moving 完全不变。
- 未通过：停止本版本，不自动加新实验、改阈值、重训或放大。
- 通过：人工显式启动 main，train full 20430 / 8192 updates / dev512，
  每 1024 updates 评估；结构/损失/阈值不变，需要 passing screen summary。
  dev512 是开发/selection 集，不冒充独立最终 test。

smoke 仅运行 4 train / 2 dev / 2 updates，验证管线，不用于科学准入。
正常使用直接 screen；它会在一次运行中做代码预检查、准备、训练、评估与摘要。

## 速度、空间和恢复边界

训练窗口各读一次，V18 forecast 和历史投影各准备一次。
之后训练只读取 RAM 中 float16 紧凑特征，监控复用 dev 的稀疏候选/raw counts；
不会每个 update 重读 nuScenes、重跑 V18、重算 GT Moving 或保存 dense cache。
六个 future 投影并行，上限 6 线程；I/O/Moving workers 默认 8，
OMP/MKL/OPENBLAS=1 防止线程嵌套。CPU 准备阶段 GPU 利用率低是预期，
进度按每窗口打印；不声称准备耗时为零或保证 GPU 长期满载。

每 sampled patch 特征及监督约 1.4KiB，screen 最高约 134MiB bank；
full train 约 2.7GiB，拼接临时占用约两倍。仅 bank 的峰值预算
screen 4096MiB / main 8192MiB，**不包括 V18 cache / nuScenes metadata / dev RAM**。
不足即 fail-closed，可显式设置 CLI RAM 预算，不靠静默换成巨大磁盘缓存。
实际速度、整进程 RAM 与 GPU 显存需服务器运行得到，本地不虚报。

续训会重新准备特征（没有新建大型永久 cache），逐项校验其 SHA、
train/dev identity/order、配置、checkpoint、batch/seed/eval/update budget，
恢复 optimizer 与 NumPy/Torch/CUDA RNG。只允许 `last.pt`，输出必须新目录。
`best.pt` 无 optimizer，故不能误当续训点。失败保留原目录与 last checkpoint。

## 服务器执行

输入全部使用用户已确认路径，运行配置为 tracked `configs/real_motion_occfm.yaml`，
不是缺少 METRIC contract 的 V20 Stage-1 frozen 配置。
dev64 manifest 缺失时，只从已确认的冻结 Stage-1 dev512 生成；不取 full4369 前64条。
输出新建在 `outputs/p0_f9_static_evidence_selector/screen_<time>_<sha>`。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_static_evidence_selector.sh screen
```

如果未初始化 upstream_occfm，使用代理初始化仓库声明的固定 submodule：

```bash
git -c url.https://ghfast.top/https://github.com/.insteadOf=https://github.com/ submodule update --init upstream_occfm
```

仅 screen gate 通过后，显式指定它的真实运行目录，才启动 main：

```bash
export STATIC_SELECTOR_SCREEN_DIR=/实际已完成的screen运行目录
bash tools/real_motion/run_p0_f9_static_evidence_selector.sh main
```

这里占位路径不是已确认输入，需替换成脚本实际打印的目录。
续训：`STATIC_SELECTOR_RESUME=/实际运行目录/model/last.pt bash ...sh screen`。
摘要 `summary.txt` 打印 overall / horizons / Moving / addition precision / scene
positive-negative / best 和 last candidate / 固定 gate；完整记录在 model/summary.json。

正式推理/独立评估入口：`eval_p0_f9_static_evidence_selector.py`；
传 `--predict-only` 输出六帧压缩 prediction，不读取未来 occupancy GT；
不传时只输出正式评估摘要，无大型预测落盘。

## 本地验证的含义

CPU 测试包含真实梯度更新、可分 synthetic utility 学习、实际存盘再加载、
断点恢复与不中断训练逐参数等价、GT 输入隔离、动态/occupied 保护、
新旧历史 memory 逐 voxel 对齐、稀疏/密集正式 metric 一致。
这些是实现正确性验证，不是 nuScenes learned gain。
本地缺少服务器数据和 GPU；真实 gate 必须以该 screen 输出为准。

本次本地验证：570 passed / 4 skipped / 1 deselected（非 integration 集），
426 个 Python 文件语法编译通过，Bash runner 语法检查通过。
新增 selector 测试 18 项含上述完整 synthetic 入口流程。
未将本地测试冒称为远端 CI 或 nuScenes GPU 实验。
