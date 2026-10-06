# 同窗口 V18 / 点级 CCR 六帧 FPS

仅测速，不训练、不调阈值、不评估 GT、不升级部署候选。点级 CCR 三轮 screen 的精度尚未通过门槛，本次速度结果不能替代精度证据。

## 六个对照

| 权重 / 历史预算 | 原执行 | 无损执行 |
| --- | --- | --- |
| Clean-E14 V18，六历史 | dense CUDA Strong + 原 V18 | native Strong + 重复投影/LayerNorm 复用 |
| epoch19 V18，四历史 | 同上 | 同上 |
| epoch19 V18 + 已训练三轮点级 CCR，四历史 | 同上 + 完整 CCR | 同上 + 同一完整 CCR |

不新增网络参数、不改 checkpoint/config、不改变 support、shape、类别、阈值、六帧 horizon 或源数量。仅本次 benchmark 的无损 arm 临时启用优化，默认训练/部署路径不变。没有启用 motion CUDA Graph，也没有对不存在的性能改善作保证。

native Strong 仍采用正式 inverse-warp/raster：只把 5×5×1 majority 的全未知体素投票改为已有 C++17 编译器生成的单线程整数核；浮点阈值/平票边缘重放冻结 SciPy 规则，不使用 fast-math。投影复用维持 attention 原路径。每个固定窗口检查所有六帧 Strong anchors、组件/class/source count、CLEAR、运动输出/latents、CCR 概率和最终 dense 输出逐字节一致，不一致直接终止，不 silent fallback。

## 固定人口与时钟

从冻结 dev64 identity/order 中选 18 个 scene-balanced round-robin 窗口，再加入 2 个剩余人口中 source 数最多的压力窗口；不读取 GT/error/latency 做选择。identity、stratum、source count 写入 `fps_manifest.json`。每个 arm 预热后测三遍，顺序交替，所有 arm 使用相同窗口。单独报告 scene-balanced / stress 子人口；不是全集平均速度。

两种口径一起测：

- `fresh_prior`：resident source 输入和已注册历史 → 当次完整六帧 Strong/KTA 重建 → live V18 →（可选）完整点级 CCR → 六帧 dense composition。
- `cached_prior`：除固定 Strong/KTA 已准备好外，其余相同；CCR 候选、特征、编码、时序 readout 和六帧 composition 仍每次现算，不缓存 learned features/predictions。

两者都在计时开始/结束同步 CUDA。source INPUT 上卡、初始 source 提取/历史配准、磁盘、GT/metrics、编译、预热、正确性 hash 均在计时外。raw/fixed preparation 时间独立报告；不是 raw-sensor E2E FPS。按 `FPS=6/mean_six_frame_latency` 计算，不平均各窗口 FPS。host 阶段耗时不是 CUDA active utilization。

用户回忆的 135 ms 不能预先认定属于哪种边界，应首先与同权重六历史 Clean-E14 的两个口径对照。Clean-E14 六历史与 epoch19 四历史数字只能描述执行时间，不能当作 matched-history 的科学精度对比。

## 服务器运行

已确认的 OccFM 环境和路径在 `tools/real_motion/run_p0_f9_point_ccr_v18_fps.sh`，缺文件会提前列出，不造路径、不创建新的 train/dev cache。读取 immutable snapshots，不修改旧实验或优化器/RNG；新建的输出只包含小型报告、checkpoint 快照和 native 编译物。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_point_ccr_v18_fps.sh
```

脚本检测其他训练/评估/测速进程时拒绝计时，不自动 kill。默认已确认点级 checkpoint：`outputs/p0_f9_point_ccr/screen20x3_20261006_131352_837/last.pt`；可通过脚本第一个参数显式传入另一条实际路径，但仍要求匹配 epoch19/config/population 的完成三轮 point-CCR 合同。

最后自动打印 summary。`speed.json` 保存全部 paired samples、分段、压力/代表人口和文件 hash。Ctrl-C/TERM 在当前 paired window 完成后停止，报告 `stopped`，没有训练断点可恢复也不需要恢复训练。

## 本地验收

健康 CUDA 项目环境、RTX 3050 实际 replay：普通 TRAIN 窗口和高 source DEV 压力窗口；六个 arm × 两种边界，同权重原始/优化路径的运动/概率/六帧结果相等。67 项相关测试通过，包括模型身份、native 阈值/平票/边缘、历史输入与 V18 回归。此验收证明这些已测路径的无损一致性，**不声称已经得到 L40S FPS**；服务器每个窗口仍强制相同检查。

初版测速入口错误使用了 `provider.model` 作为 E14，服务器在六历史检查处提前停止，未产生有效 FPS。`JointColumnProvider` 的构造函数实际把 E14 存到 `provider.reference`，再把 `provider.model` 切换到 epoch19 transport。修正版通过同一个 `resolve_arm_models` 绑定入口与真实 replay 测试：E14 两个 arm 必须引用独立的六历史 reference，epoch19/CCR 四个 arm 必须引用四历史 transport；不改变 provider 的 live 模型。日志和 `speed.json` 打印每个 arm 的实际 checkpoint SHA、历史预算。新增测试拒绝 alias、错历史预算和破坏 live provider 的情况，真实 replay 还确认 E14/epoch19 的运动输出确实不同，避免不同标签共用同一个模型而让无损检查虚假通过。
