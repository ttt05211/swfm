# 第一版共享历史联合试训

独立入口 `tools/shared_history_experiments/`，不修改旧训练、推理、geometry cache namespace 或原断点。新结构随机初始化，旧 clean E14 只提供配置和历史缓存契约，不导入其权重。

## 固定配置

- GT future ego 配置，四历史→六未来，原 scene-balanced TRAIN20%：4086/20430 窗；原 seed20261011、window batch4/source128。
- 默认2轮；显式 `EPOCHS=1` 可以做1轮。选择后整个周期余弦下降，没有固定学习率 tail；1轮任务不能静默延长成2轮恢复。
- 最终固定 DEV512 报告 soft Transport 与 soft ADD-only 的 IoU/mIoU/Moving、1/2/3s分项；`EVAL_POPULATION=dev64` 可预先选小人口，但恢复不能改人口合同。
- 保留最终全网格 CE+Lovász、source SE(2)几何/existence、现有 ego R10 辅助。GT 几何不由 ego 头决定，不升级场景 ego 头。
- 两未来一组运行三组，六未来全部监督；连续 ADD-only 在训练和推理同时成立，REMOVE恒零，无硬动作阈值或 sampled weighted ADD 主损失。

## 实际改变

四帧逐高度语义embedding4维+有效性进入通道，stride2轻量共享stem与两层局部残差，默认 F=[4,64,100,100]。偶数卷积核匹配下采样格心，避免采样格偏移。精确原始200×200×16几何另存。

source/source关联/registration继续来自旧连通域准备。按各帧真实关联组件质心、t0朝向的16m物理区域，从F读10×10 tokens；真实组件体素栅格化mask保护小物体。历史相对位置、按实际历史时间计算的速度和有效性只从同一几何视图生成，不使用旧flat运动feature网络输入。STWM保留原配置维度、ST blocks/未来decoder与KTA+SE(2)输出，删除独立语义tube stem。

canonical metric点与支持人口/排序严格保留；逐帧逆registration在F上读点局部上下文，再用小型time/valid-aware聚合形成E。旧邻居数量/密度/年龄27维不再计算或输入；为兼容现有连续phase/几何接口保留零占位，不是隐藏输入。Surface12保持历史几何先验，live phase6仍使用公共未来source/ego变换。CCR每点历史只编码一次，六时距有界读出。

既有公共 `prepare_projection`、continuous context/soft splat与快速soft inference仍复用；不声明整数owner/合法域可微。shared encoder同时获得运动/CCR链路梯度，原 existence 仍依靠 BCE。首真实评估窗口检查 fast vs reference 完整六帧字节一致，失败不发布推广。

纯历史 compact geometry 只读复用；仅在本次有界RAM里构造确定性读取坐标/Surface描述，不生成新全TRAIN bank、不缓存F/H/Q/E。原始四帧需实时载入，旧tube或冻结feature bank不能代替。

训练时间包含新encoder、STWM、CCR和完整六未来CE/Lovász/backward。质量评估复用旧GT几何缓存，日志耗时不是fresh Strong正式FPS；不继承旧54.945FPS或旧主表质量。

## 运行命令

代码同步到服务器后，在 OccFM 环境运行；本地没有完整nuScenes数据，不能代跑真实训练。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export EPOCHS=2
export RUN_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/shared_20pct_2ep_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/run.sh
cat "$RUN_DIR/summary.txt"
```

默认只训练 shared，不自动训练第二个模型。建议首8步只确认资源与接口，可替换上述运行行为：

```bash
bash tools/shared_history_experiments/run.sh --max-updates 8
RESUME="$RUN_DIR/last.pt" bash tools/shared_history_experiments/run.sh
```

`--max-updates` 不缩短2轮余弦；保存整步游标后继续剩余预算。Ctrl-C/SIGTERM同样在整步/整评估窗边界保存；SIGKILL只能恢复最近周期checkpoint。

新终端恢复时明确使用刚才实际打印的目录，不能猜目录，也不能用旧 joint_occ checkpoint：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export EPOCHS=2
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际本次目录
RESUME="$RUN_DIR/last.pt" bash tools/shared_history_experiments/run.sh
```

`last.pt` 保存权重、三个Adam组、RNG、whole-cycle游标/数据/源码指纹；每个完成轮次另保留 `epoch_0001.pt` / `epoch_0002.pt`。同目录有内核租约，禁止同时resume同输出。

要只读查看第1轮，使用独立评估子目录，保持原EPOCHS/人口/variant；不写来源last/epoch或覆盖训练摘要：

```bash
RESUME="$RUN_DIR/epoch_0001.pt" RUN_DIR="$RUN_DIR/eval_epoch1" \
  bash tools/shared_history_experiments/run.sh --evaluate-only
```

## 可选公平旧结构控制

新结构2轮不能与旧3/20轮数字直接做架构因果对照。可选独立旧结构同4086窗口、同2轮、同loss/soft ADD-only控制；**不会自动执行**：

```bash
VARIANT=legacy_control EPOCHS=2 \
  RUN_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/control_20pct_2ep_$(date +%Y%m%d_%H%M%S)" \
  bash tools/shared_history_experiments/run.sh
```

两结构同seed与人口/order，不声称不同参数结构能逐参数同初始化。对照仍使用原27/Surface编码；shared才替换历史特征。

本地小网格/合成元数据检查只是实现验收，不是nuScenes精度或L40S速度。旧正式模型、暂停的内部Pred和其他线程未提交文档保留；不自动扩训、挑dev-best、调阈值或推送。

本地验收：13项新专项+7项原两未来训练回归，共20 passed（含可用CUDA/BF16实际小网格），CLI help与Bash语法检查通过。覆盖共享Encoder一次/梯度、过去位置和轴顺序、精确metric support与Surface、未来占据毒化、REMOVE恒零、六帧软参考/fast一致，以及真实CLI人口/快照/恢复/来源只读。不是完整CI，也未测完整服务器数据精度或正式FPS。
