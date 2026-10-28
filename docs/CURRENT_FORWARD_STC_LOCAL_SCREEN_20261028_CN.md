# 最新 forward AdaLN epoch2：本地 STC 筛查与冻结候选

用户于2026-10-28回传服务器固定候选全集摘要，输出 `Fixed target STC mIoU>=19 and IoU>=29: met=True`；同一入口仅在complete且4369窗时输出此标记。本地保存的是用户贴出的数值，没有直接下载服务器evaluation.json或逐窗confusion。以下覆盖原先“全集待验证”，不改变冻结系数或原默认。

| 路线 | 时距/s | IoU | mIoU | MovingMicro |
|---|---:|---:|---:|---:|
| occ_gt/transport | 1 | 59.870161 | 51.705078 | 43.422698 |
| occ_gt/transport | 2 | 53.975771 | 42.047486 | 25.134746 |
| occ_gt/transport | 3 | 49.991130 | 36.400623 | 15.415586 |
| occ_gt/transport | Avg | 54.612354 | 43.384396 | 27.991010 |
| occ_gt/semantic_refine | 1 | 65.475386 | 55.023828 | 44.582186 |
| occ_gt/semantic_refine | 2 | 60.253220 | 45.996796 | 27.820764 |
| occ_gt/semantic_refine | 3 | 56.382252 | 40.360227 | 18.051345 |
| occ_gt/semantic_refine | Avg | 60.703620 | 47.126950 | 30.151432 |
| stc_gt/transport | 1 | 29.118172 | 20.808048 | 26.499608 |
| stc_gt/transport | 2 | 26.676371 | 17.642094 | 17.210538 |
| stc_gt/transport | 3 | 24.675135 | 15.431648 | 11.207660 |
| stc_gt/transport | Avg | 26.823226 | 17.960597 | 18.305935 |
| stc_gt/semantic_refine | 1 | 31.059236 | 21.728093 | 27.683212 |
| stc_gt/semantic_refine | 2 | 29.115673 | 18.847181 | 19.264376 |
| stc_gt/semantic_refine | 3 | 27.264147 | 16.644298 | 13.324443 |
| stc_gt/semantic_refine | Avg | 29.146352 | 19.073191 | 20.090677 |

与同一最新epoch2原ALL4369最终Refiner相比，STC mIoU18.721545→19.073191（+0.351646）、IoU28.446166→29.146352（+0.700186）、Moving18.725074→20.090677（+1.365603）；OCC mIoU48.711339→47.126950（−1.584389）、IoU62.809965→60.703620（−2.106345）、Moving30.593620→30.151432（−0.442188）。候选达成用户19/29目标，代价为OCC质量下降。三个STC时距的最终Refiner mIoU/IoU/Moving均高于原版。此结果不能分解两个改动的独立贡献。

该配置继续固定为独立的有监督目标域校准候选；32个校准VAL场景包含在4369全集中，且既往完整VAL已查看，不能当作未接触目标域标签的zero-shot或独立测试结果。仍使用原2816122参数网络权重和同一GT future ego条件；未重新训练、未按新全集分数调参、未替换默认入口。新候选FPS尚未测量，不沿用其他模型FPS。源SHA、两个系数和历史估计规则见下文。


本轮全部使用用户刚提供的 `forward_adaln_epoch2_local.zip`，2816122参数的 forward关联 / FPN / AdaLN-Zero Decoder2 / U-Net epoch2。原始服务器checkpoint为 `outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/epoch_0002.pt`，原文件SHA256为 `7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0`。ZIP为11,132,055字节。只复用旧本地包的真实数据；没有加载旧clean1000、norm192或旧FPN权重。

数据是与本次训练记录逐key一致的原remaining64，每窗来自不同VAL场景，另有干净原TRAIN32。当前拟合固定前32，检查后32，并将后32拆为screen16/confirmation16查看。64窗过去已反复查看，均属于开发数据，不是独立测试。本地无Moving标注，所有表格仅标准全网格1/2/3s平均mIoU/IoU。

下面是同一完整64窗的结果。最后一行由封装后的服务器同款 `CandidatePredictor` 重新完成128次GT/STC真实推理获得，不是把单项增益相加。

| 方法 | STC mIoU | STC IoU | GT mIoU | GT IoU |
|---|---:|---:|---:|---:|
| 最新epoch2原版 | 18.228803 | 28.513938 | 51.258825 | 64.056196 |
| 原版＋全局FREE−1.5 | 18.300650 | 29.185949 | 49.436855 | 61.703691 |
| 四帧速度折中＋FREE−1.5 | 18.480909 | 29.224018 | 49.290389 | 61.643739 |
| 历史伪目标小解码块192＋速度折中＋FREE−1.5 | 18.453814 | 29.419061 | 48.761943 | 60.885282 |
| **冻结候选：中心/速度去抖动＋距离校准** | **18.526726** | **29.152295** | **49.482421** | **62.010494** |

冻结候选相对原版：STC约+0.298mIoU/+0.638IoU；GT约−1.776mIoU/−2.046IoU。后32窗原版18.063933/28.623019，候选18.621489/29.189383。原版新鲜复算与服务器同64的18.229030/28.514311、51.260686/64.055743接近；不同GPU/Torch以及原执行的细小数值漂移仍存在，不声称逐字节一致或改变原数值门槛。

首次完成本地筛查时尚无新的ALL4369结果；现已收到上述服务器实测摘要。原ALL4369为STC18.721545/28.446166、GT48.711339/62.809965。开发64与全集的类频率和窗口人口不同，达标判断使用服务器实测，不将本地增益加到原全集分数。

候选只保留两个改动：

1. 对最近一对原本已匹配、且至少有3个真实历史观测的source，以实际历史时刻对四帧质心作直线拟合。原最近两帧速度与拟合速度各取一半；当前观测中心与拟合的t0中心各取一半。原关联身份/有效性/类别门限/速度门限保持，最近一对未匹配的source不会凭较早历史获得新速度。历史Reader与Memory仍从实际观测位置读取；source几何、未来anchors、Strong/CLEAR和直接Memory搬运使用一致的新目标，位移只计一次。没有ICP或未来GT占据/框对齐。
2. 只给FREE类增加 `-2 + rho`，其中 `rho=min(最后观测自车坐标系中的水平距离/40m,1)`。输出网格点使用同一条条件future ego trajectory变换回t0计算距离。两模态、全部六个时距都使用同一公式。只调整占据/空的判决，18类整网格继续评分，没有visibility mask、类别白名单或按窗口规则。

**两项校准系数使用前32窗的GT未来占据标签拟合，属于有监督目标域输出校准。** 系数选择只读前32的STC/GT预测，目标为STC mIoU+0.5IoU，同时在拟合32上限制GT损失≤2mIoU/3IoU。初始25组和边界扩展25组都只用拟合32；两次选择相同的−2、+1。其余32不进入参数拟合。原网络权重不变；不能称无目标域标注/无目标域信息的zero-shot方法。source状态估计本身只用历史输入。训练过的320/482/7106参数适配分支均未混入最终候选。

完整试验及负结果保存在本地 `outputs/forward_local_20261028`：

| 类别 | 实际检查 | 主要结果 |
|---|---|---|
| 输出校准 | 单一FREE偏置、CE18偏置、soft-IoU18偏置、36参数缩放/偏置、真实指标18偏置、残差强度＋FREE | 单一偏置有效；复杂类别校准没有更好的综合收益；残差强度仍选1 |
| 无目标未来标签校准 | TRAIN32预测与STC32预测的18类/二分类/决策后验匹配 | 后32 mIoU收益小或为负，IoU约+0.1 |
| 原架构轻量适配 | 320 GN、482 GN＋head；历史伪目标或真实目标GT32；均有原TRAIN32 replay | 320/482无明显mIoU收益 |
| 小空间解码块 | 7106参数：历史伪目标、GT未来监督、GT未来目标占75%；固定192更新；另一个历史分支576更新 | 历史伪目标192略优，主要涨IoU；576不优于192；没有选入候选 |
| 输入/读出 | 嵌入3×3 binomial平滑50%/100%、Refiner左右翻转logit/概率平均 | STC收益接近零；TTA额外读出不保留 |
| 历史几何 | 静态概率四帧融合25%/50%、四帧OLS速度全替换/半折中、当前中心半折中 | 全替换不稳，半折中较好；中心去抖动有小增益 |
| 组合复核 | 原版/历史192/历史576 × 原pair/半OLS × 固定/GT拟合/历史伪目标拟合FREE | 真实重算，不加算单项增益；保留较简单的原权重方案 |

其中所有适配实验从此次epoch2出发，冻结上游，只在新文件保存适配参数。模拟噪声、旧头、旧Surface、旧norm192及旧优化后端未导入。loss为CE+Lovasz；历史伪目标只由STC四帧静态一致性构建，GT未来ego仍是明确条件。source原权重、数据ZIP、旧结果与原训练代码保持只读。无跨调用预测缓存的部署方案；研究缓存仅用于冻结上游的小规模对照，不能当作FPS。

冻结服务器候选位于 `tools/causal_forward_stc_candidate`，与研究入口分开，原默认入口不改。单卡双进程仅推理；严格固定原checkpoint SHA、同一recipe、ALL4369/150完整人口，合并整数confusion后计算指标。Moving仍按原服务器协议评分，不进入模型。断点按完整窗恢复，不在全集调参或自动晋升为默认。

服务器运行：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_FULL_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351"
export STC_CANDIDATE_GPU=0
export STC_CANDIDATE_RESUME=0
export STC_CANDIDATE_DIR="$PWD/outputs/p0_f9_shared_history/stc_state_range_candidate_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_stc_candidate/run.sh >"${STC_CANDIDATE_DIR}.log" 2>&1 &
tail -n 25 -f "${STC_CANDIDATE_DIR}.log"
```

结果为 `$STC_CANDIDATE_DIR/summary.txt`，每30秒日志给OCC/STC进度。恢复需设置原目录及 `STC_CANDIDATE_RESUME=1`。不需要上传适配权重，服务器不重新训练或重新拟合。首次服务器全集已经完成并回传met=True；保留命令用于复现，不自动重复运行。

检查：29项新旧回归通过，包括真实原模型加载/源SHA、无损缓存/拟合场景分离、source和直接Memory位移只计一次、GPU正式入口GT/STC各64窗新鲜复算、六时距读出/未来占据poison、单卡双worker与整数合并、不同recipe拒绝、完整窗中断续评/源只读。两个CLI、Bash语法及原模型/训练文件零diff通过。服务器新全集分数以上述用户回传为据，新FPS未测试。
