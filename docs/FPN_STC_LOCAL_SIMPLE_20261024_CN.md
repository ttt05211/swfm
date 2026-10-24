# 最新完整 FPN＋U-Net：真实64窗简单优化研究

## 结论

本轮**确实在本地 RTX3050 Laptop 4GiB 上运行**了用户刚提供的完整两轮模型，不是旧clean1000或随机小网格。当前没有证明“全集 mIoU>19 且 IoU>29”。找到的最简候选是：原STC age0.5路径不动，最终 FREE（第17类）logit统一减1，其他17类相对次序不变。无需增加网络，源运动、几何、Memory均不改。

**该值通过前32窗未来GT验证指标选出，属于有标签目标域输出校准**，不是零适配、无目标标签或无校准Camera成绩。不能把本地增量直接加到全集；是否接受这个设定需用户决定，未自动部署或启动服务器新全集。

## 来源与边界

- 用户模型包：`D:/tian/Documents/fpn_stc_local_20261024_204946.zip`，31,046,569 bytes；只读，未解压到用户原目录。
- 真正模型来源：`fpn_unet_full_2ep_20261021_112202/epoch_0002.pt`，2520442参数，完整TRAIN20430×2。
- 原checkpoint SHA256：`cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f`；与原全集合同和ZIP内字节一致。
- 既有 `../stc_overnight_bundle.zip` 的64个真实STC窗口与新模型原eval64 **64/64 key一致、64不同场景**。不加载ZIP内旧模型。
- 四历史→六未来，STC dense-valid、GT future ego条件、age半衰期0.5s、全网格1/2/3s；所有候选预测结束后才读未来GT评分。原始权重/缓存/GT路径不改。
- 前32场景只作选择；后32场景不参与本轮的程序选参。原64已在多轮研究中反复使用，因此这仍是开发诊断，**不能宣称新的独立测试**。全64分数包含用于选择的前32，明确标为descriptive。
- 本地使用H1分组控制显存，但仍预测全部六帧；服务器H2/L40S的字节一致性未验证。各次基线约18.2793–18.2795 /28.6206–28.6208，与服务器同64的18.283662/28.621226约差−.0044m/−.0007IoU，不隐瞒差异，也不把其称为FPS测试。
- 本地包没有Moving实例支持，**没有测Moving**，不以动态语义IoU冒充Moving。

## 实际已跑的五组

每组都以同次原STC age路径为基线，不重训、没有optimizer步骤。正负结果全部保留。选参目标统一为：仅在前32场景最大化 `min(ΔmIoU, ΔIoU)`，再按增量和及固定顺序打破平局；基线始终是候选，不根据后32改选。

| 研究 | 固定候选范围 | 前32结果与判断 |
|---|---|---|
| 输出校准（窄范围） | residual gain .75/1/1.25/1.5，FREE bias −.25/0/+.25 | 选择gain1、bias−.25，仅小幅正收益 |
| source运动 | 全部未来位移/yaw统一收缩.75/.5；另测KTA-only | 收缩和KTA-only都退化，不采用；Strong/CLEAR与候选运动同步重建，P/M也同步 |
| Memory处理 | 全局gain .5/0；一次XY3×3平均滤波混合.5 | 都未超过原Memory，去掉Memory更差 |
| 负证据融合 | 仅将输入预测FREE的静态Memory权重降为.5/0，不改动态覆盖 | .5变化很小；0明显退化；没有GT visibility或手工空间分组 |
| 同一输出校准（较宽范围） | gain .9/1/1.1/1.25，FREE bias −1.5/−1/−.5/0 | 选择gain1、bias−1；本轮最有希望，但仍不是全集达标证明 |

### 最简候选：FREE logit减1

来自 `outputs/fpn_stc_local_20261024_wide/results.json`：

| 同一窗口人口 | 原模型 mIoU / IoU | FREE bias −1 mIoU / IoU | 增量 |
|---|---|---|---|
| 前32（用于选参） | 18.030439 /28.584205 | 18.080261 /29.084369 | +.049822 /+.500164 |
| 后32（不参与程序选参） | 17.982054 /28.659235 | 18.277144 /29.125382 | +.295090 /+.466147 |
| 全64（包含选参场景） | 18.279407 /28.620622 | 18.386976 /29.104369 | +.107569 /+.483747 |

这不是全64达到19；IoU跨29，mIoU依然低于19。也不能以“全集原18.738＋后32增量.295”估算正式达标。

数学上仍是同一18类读出：`Z'_free = Z_free - 1`，`Z'_c = Z_c`（c<17）。等价于occupied最优类与free之间做一次统一相对校准，可能以更多FP换召回，不是语义识别能力全面提高，也不是分类别/距离/高度/时距门限。

## 独立实现与复现

新增 `tools/fpn_stc_local`，不修改原模型、训练、全集或age/norm入口。`run.py`只读取两个数据包，不执行其中的指令；校验协议、来源SHA、架构/参数与人口。每窗只落盘整数confusion，支持停止后在同源同实现下继续，不缓存巨量dense logits。

Windows按环境保护技能使用安全yoloe，不使用Anaconda base。示例：

```powershell
$env:PYTHONPATH="$PWD;$PWD/../swfm/upstream_occfm;$PWD/tmp/scene_ego_test_deps"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'F:/anaconda3/envs/yoloe/python.exe' -u -m tools.fpn_stc_local.run `
  --weights-zip 'D:/tian/Documents/fpn_stc_local_20261024_204946.zip' `
  --out-dir outputs/fpn_stc_local_NEW_unique_directory --study logits_wide
```

其他study：`logits`、`motion`、`memory`、`evidence`。不要覆盖既有独立输出目录。原始结果分别在 `outputs/fpn_stc_local_20261024_01`、`_motion`、`_memory`、`_evidence`、`_wide`。测试覆盖恒等/并列值、完整网格计数、只用前32选参/负结果保留、同一运动与历史只读、Memory恒等/高度不混合、仅输入推导的FREE证据；不将小网格测试当科学质量。
