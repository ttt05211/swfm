# 当前任务摘要（2026-10-11）

## 独立最后对照：Pinned original pair / 原KTA不变（2026-10-15）

用户matched20%各1轮结果：unified_fit OCC Refiner mIoU+.717503/IoU+.026781/Moving−1.167401（Transport+.713404）；STC−.119398/−.205448/+.171161。不能只归旧checkpoint不适应；多帧平均稀释瞬时运动仍是假说，不是已单独定位。用户允许最后试统一接口、原两帧KTA不变，不再跑fit或新tracking设计分支。

仅新增tools/pinned_source_track四文件+test/doc，仍本独立experiment分支，不改旧c58训练、609冻结screen或其他分支。原t-1/t0同类别XYZ mutual-nearest/25m/s/.5匹配一次同时产出索引+原速度；固定末段，只延伸t-2/t-3，已有older歧义/缺帧/ICP保护不变。无multi-fit/最近两有效点回退；未配对KTA零。TRAIN直接保旧record/KTA/anchors bytes，仅换历史registrations/patch/memory；EVAL新PinnedPredictor四帧提取一次、不先跑旧four-frame prepare，Strong复用同一pair速度避免再匹配，仍原Strong背景算法。

只新训unified_pair RANDOM TRAIN4086×1/原H6 CE+Lovasz，无ego/aux；复用之前matched root/baseline完成权重metadata及指标，不导weights/Adam/RNG，不重训baseline。源workflow/data/code/last SHA核验，相同初始化在optimizer前验证，TRAIN keys/order/update数训练前验证，最终科学合同只protocol/implementation可不同。首次可按时间找最近完成同协议baseline（非按分数），打印冻结reference；resume固定source，PAIR_DIR新root/unified_pair，PAIR_RESUME=1只续新路，完成不多训。最终同DEV64 OCCGT/STCGT/no masks/alignment，与旧baseline并列，不自动采用/扩训。

51passed21.62s，真实CPU/CUDA小图、pair原速度/Strong六帧bytes一致、KTA对象不改、最后一对不重配/未匹配不拿older速度、真实trainer和外层入口新路部分/完整resume、baseline只读/初始化拒绝/模块绑定恢复；CLI/Bash通过。旧c58/609源码fingerprint不变，无服务器训练代跑。入口bash tools/pinned_source_track/run.sh，PINNED_REFERENCE_DIR可原TRACK_TRAIN_DIR；只同步本独立分支，ghfast.top代理，等用户真实summary后判断。

## 独立分支：统一track多帧KTA，同预算随机重训一轮（2026-10-15）

冻结64结果已complete：多帧KTA 3s均值OCC2.212→2.080m/STC3.740→2.961m，但旧GTepoch2直接换初值Refiner OCC mIoU−.188808/Moving−1.831655，STC+.197338/+.039153；统一两点OCC+.041881/+.148153、STC+.026406/+.000513。不推广Frozen fit，不断言尾部/旧残差分布哪个解释全部。用户明确授权做同规模预算重训。

仅在本独立worktree/experiment分支新增tools/unified_source_training六文件+专项test+说明；不改原冻结screen（其resume指纹不变）、正式model/train/cache或另一分支STC适配。baseline/unified_fit两路相同RANDOM init，TRAIN20%4086各一轮、seed20261011/同序B4-source128/H6完整6 CE+Lovasz/完整一轮余弦，无ego或运动/existence损失，无旧权重导入。新track同时TRAIN与OCC/STC最终评估使用；原Strong/CLEAR保持两帧控制。旧cached t0 source/原始历史可复用，新previous/older components、关联/注册/reads/KTA在原有CPU预取和有界RAM构建，KTA与anchors同步更新；不写旧cache、不建新bank。

最终同DEV64 OCCGT/STCGT Transport+Refiner，fullgrid/STC all-known、无aligned；源码/数据/STCinventory/相同初始化hash与科学契约核验。两个child顺序执行，完整update/window resume/Adam-RNG-cosine/epoch_0001与完成路跳过；可max-updates8两路首检后同一轮恢复，默认直接各一轮。输出TRACK_TRAIN_DIR新目录（默认原swfm outputs），TRACK_TRAIN_RESUME=1恢复。原E14仅架构+cache契约，不导其权重；原GTepoch2不读。45passed21.99s含CPU/CUDA反传、实际trainer恢复及64窗评估，CLI/Bash通过。尚未代启动服务器；用户代理只ghfast.top。只推此独立分支，不混其他开发代码。

## 独立工作副本：Unified source track / Multi-frame KTA DEV64（2026-10-15）

用户授权独立实现，不混其他开发分支。本工作副本swfm-unified-source-tracks/experiment/unified-source-track-kta-dev64基于88d8dac；仅新增tools/unified_source_tracks五文件、专项test与操作文档，旧model/trainer/final evaluation/cache源码不改。原swfm-surface-ccr及其当前STC适配开发、混合文档/tmp不动。新实验完整GTepoch2冻结、同DEV64 OCC+GT/STC+GT，baseline/unified_two/unified_fit三路，固定baseline Strong/CLEAR以隔离track/KTA；不是整个Strong的新正式统一版本。

保原associate_backwards的25m/s/歧义保护/缺帧接续与ICP，不加shape代价或门限；一张原始world-centroid身份表供KTA/注册及source patch/memory读取。ICP失败仍不供空间证据；速度原始质心actual-history-time slope，p0不改、future名义2Hz，单点零/两点fit=差分。旧磁盘cache含旧source配准/KTA，不能当新版，首版现场RAM重建不写旧/新geometry bank。后续采用需新派生cache版本，原语义/known/poses及规则不变组件可复用。

38passed in11.56s，含真实CPU/CUDA小网格、运动抖动正例和加速负例、源权重不变、GT未来标签/annotation全部候选预测后只评分、displacement-preserving KTA误差与有效点分组、整窗整数resume/实际CLI快照契约保护。初次测试临时父目录缺失、测试输出嵌入source训练目录被正确保护，修测试未放宽保护。入口TRACK_DIR新目录+bash tools/unified_source_tracks/run.sh，TRACK_RESUME=1同目录恢复；实际nuScenes指标未运行，未训练/自动扩训/推广。只在独立分支实现，远端推送和服务器评估尚未执行。

## 本线程最新实现：相同规模的内部 Pred 三轮

用户要求不扩大训练，试当前软配置的内部ego版本。新增隔离 `tools/joint_occ_experiments/pred/`：独立RANDOM、相同scene-balanced TRAIN20%4086窗/seed20261011/三轮余弦/batch4/source128；WM+CCR+ego共同更新，不导入GT权重。沿用两未来帧renderer、六未来完整CE+Lovasz及source geometry/existence/R10 ego监督。
前向所有未来几何使用内部Pred（不混GT pose），占据→ego梯度沿用0.1→1 ramp；仅复用只读纯历史缓存，GT未来背景/投影弃用重建。GT-derived导航条件仍公开，不称navigation-free。最终固定DEV512六读出，主候选预指定fast soft ADD-only；首实际窗三软路线完整六帧byte核验，不额外测速/扩训/调阈值。
Ctrl-C完整update/评估窗后保存，同一Pred目录恢复Adam/RNG/余弦及整数评估前缀。修复saved errors列表别名导致续跑修改调用者状态；原GT训练指纹不变。本地49项Pred/GT/两帧训练/软快速读出测试通过，CLI/Bash检查通过；无本地nuScenes/L40S，真实训练未代跑。入口 `bash tools/joint_occ_experiments/pred/run.sh`；详见 `docs/JOINT_OCC_INTERNAL_PRED_THREE_CN.md`。

## 最新任务：最终占据监督 + 内部 ego 的独立联合试训

2026-10-11最新用户授权实现软推理提速，不新训练/不重跑DEV512。新增隔离 `soft_inference/fast.py`：三未来合批、六张byte网格单次CPU回传、单forward合法actor/action索引与常量复用、ADD-only跳过恒零REMOVE、推理scratch原位合成不存反传中间量；18类/全网格/float64几何/原插值/owner/fallback/覆盖顺序不变。
新增 `run_optimized.sh`：自动找到已完成soft_dev512质量目录并校验source快照/整数计数/精确人口；只读复用512指标，新目录测同32窗×2交替五路（hard、旧soft ADD/ADD+REMOVE、新soft ADD/ADD+REMOVE）。新路径六帧byte gate必须通过，两遍更快/≥1.1×/10%VRAM余量才pass，无自动部署/新训练。首窗另作串行分段定位，不计吞吐。
计时边界v2保持历史extract/match/H2D在外，live Strong/KTA/网络/投影/六帧CPU输出在内；精确识别直接KTA和旧upgrade缓存舍入公式，再实时重建保持原冻结输入，不放宽输入/输出gate。未知编码拒绝。训练源码/恢复指纹/历史缓存内容不变；Ctrl-C完整测速窗后保存，同SOFT_OUT+SOFT_RESUME恢复，旧质量/权重AdamRNG只读。
本地41项专项及loss/action/fast回归通过，CLI/help/Bash/diff检查通过。CPU合成单源640000体素render+argmax微测ADD3.292→3.460s、ADD+REMOVE3.410→3.246s，没稳定通用加速；不能外推L40S/整网FPS，当前尚无新版服务器性能。旧正式54.945FPS及GT3质量原数据不改；详见 `docs/JOINT_OCC_FROZEN_SOFT_DEV512_CN.md`，仅执行新的短测速入口，不执行旧retime-only或完整run.sh。

最新用户问软推理能否提速及两路线优先级，纠正涨点分解：最终soft ADD-only−hard ADD-only=2.431369pp，其中Transport读出差1.071885，CCR净增益差0.776828−(−0.582656)=1.359484；后者不能混称2.43或当严格因果份额。
只读代码确认当前软推理逐horizon render_group([h])并逐帧CPU回传；每帧重新选合法actor/action行，per-actor/action unique排序+scatter，18类dense static grid_sample/合成，且走保存反传中间量的通用训练compositor。推理专用、两/三未来合批+统一回传、单forward索引复用/跳过ADD-only零REMOVE、再必要时GPU融合是具体候选，尚无分段CUDA证据或新版实测，不能承诺倍率/无损。
建议先有限预算优化冻结soft推理，因为40.321095已真实验证，硬前向重训收益未知且算法重采样差不一定可追回；若优化后仍显著慢，再转硬前向连续代理训练。旧计时接口用户取消重测；本轮只分析/更新记忆，没有新的执行后端、训练或服务器测评。

最新用户已返回GT三轮DEV512六读出：hard Transport/ADD/ADD+REMOVE mIoU38.472382/37.889726/37.908709，soft39.544267/40.321095/40.211083；soft ADD收益+0.776828pp，hard ADD收益−0.582656pp。soft REMOVE增量−0.110012pp/Moving−1.141919pp。实测硬106.159ms、软334.836/334.674ms六帧；不能将慢软输出直接替换快速推理，亦不能把旧正式54.945FPS赋给此GT三轮模型。
最新用户明确取消接口重测，当前只讨论如何朝硬合成优化训练；没有授权这轮启动新训练。计时边界已在本地隔离soft_inference代码修正：历史extract/match/H2D在外，fresh Strong/KTA/live网络/投影/六帧CPU输出在内；已有512整数质量可新目录只读复用，旧timing丢弃。4项本地专项通过，未重测服务器、未推送、未改变原训练/缓存指纹；复测入口目前不执行。
训练侧候选：前向对齐部署Strong背景、source硬落格/覆盖、owner/fallback/合法动作与固定ADD0.5/REMOVE0.95，反向连续代理/直通近似；保留CE+Lovász和source geometry/existence，Pred再经连续pose代理回ego。不是将uint8/one-hot直接CE即可；必须处理概率平滑/错误体素梯度及离散竞争代理偏差，不能承诺稳定性/涨点。推理仍旧硬后端，无额外soft pass；尚未实现硬前向训练器，需微调/重训才能评估。

2026-10-11最新用户要求冻结当前GT三轮，先DEV512软输出并同窗软/硬推理测速，不新训练。
新增独立 `soft_inference/` 入口，默认source `gt3_horizon_float_resume_20261011_151144/last.pt`（epoch3/update3088）。
源快照只读、精确源eval_keys512；hard/soft各Transport、ADD-only、ADD+REMOVE，完整六帧机制不改，报告1/2/3s全网格IoU/mIoU/Moving。soft仍既有连续context/概率合成argmax；hard固定ADD0.5/REMOVE0.95，无dev选路/阈值搜索。
随后固定DEV512内scene-balanced32窗×2交替暖测速：hard ADD+REMOVE、soft ADD-only、soft ADD+REMOVE，每路线独立重算Strong/WM/CCR/六帧CPU输出。排除I/O/历史表示/描述/labels/metrics/预热；不是旧54.945FPS或原始输入端到端。六帧必须与缓存质量路线byte parity，失败保留FAIL。
无优化器/新bank/旧缓存写入；原训练/执行指纹覆盖源码完全不改。新out/state整数计数与完整测速窗口支持Ctrl-C后同SOFT_OUT+SOFT_RESUME=1恢复，来源契约/快照变动拒绝。
本地4项新测试+原loss/action/fast相关27 passed；涵盖真实小网格与先前audit一致、未来labels毒化、fresh Strong而非缓存、单次硬合成、AB/BA完整六帧、失败保留、512质量及测速中断恢复/原权重AdamRNG字节不改。CLI/help/Bash检查通过，无本地nuScenes/L40S，尚未代跑DEV512或声称真实速度/涨点。
入口 `bash tools/joint_occ_experiments/soft_inference/run.sh`，新summary即可；详见 `docs/JOINT_OCC_FROZEN_SOFT_DEV512_CN.md`。原正式SurfaceCCR模型及数据表/FPS均不变。

2026-10-11用户已返回只读loss/action诊断：source3088步/3轮，TRAIN16四批+DEV64，不是DEV512。
WM占据/source梯度平均cosine+0.1821，仅1/4负；共享history近0且2/4负，yaw占据norm约source三倍；不能据局部统计认定普遍梯度冲突或删运动loss。existence占据梯度为0，当前仍依赖直接BCE。
同DEV64硬Transport/ADD/ADD+REMOVE mIoU39.892867/39.842671/39.867367；软40.910045/41.755960/41.598596。无CCR已有1.017178pp软硬gap，ADD增益软+0.845915/硬−0.050197；头score平均绝对差约3.07e-7，证据更指向合成/插值/阈值语义差，不是大规模头读出不一致；未进一步量化各机制因果份额。
硬ADD587262体素，改对195396/损坏377191/错到错14675；REMOVE8907，改对7272/损坏999，触发占合法unique约0.494%，静态仅2。REMOVE非失效；391866个ADD错误按owner规则不可同pass撤销。soft REMOVE增量−0.157364，不把降低0.95阈值当已证实改法。
建议下一步优先训练/硬推理合成对齐（硬Strong背景/owner与动作阈值，连续几何梯度仍保留），保留CE+Lovász/source/existence；不是已实现或已验证收益，不自动新训练/阈值搜索/改旧FPS。用户仅返回摘要，本轮只解读和更新任务记忆，无训练源码变更。

最新用户已完成 GT-only TRAIN20%×3：3088步，DEV512 Transport mIoU38.472382/IoU50.435229；ADD-only37.889726/50.679728；ADD+REMOVE37.908709/50.697061。
新CCR净mIoU−0.563673，REMOVE相对ADD-only仅+0.018983，Moving−0.127278；FIRST8软硬mIoU差1.697028pp不是全人口或CCR专属差距。原正式模型/指标保持不动。
用户授权一次集中诊断，非新训练：新增独立 `diagnostics/` TRAIN16原batch/source全六帧梯度 + 固定DEV64软硬三路。
查实际WM占据/source梯度范数与方向（包括WM→CCR共享feature路径），REMOVE合法/触发/改对损坏及分数分布，分解Transport与CCR软硬gap。
代码确认REMOVE通道正常；只改原基线owner并恢复固定fallback，不能撤销同次ADD错误。0.95硬阈值与连续训练不一致是待量化因素，不宣称已定位全部原因或证实梯度冲突。
没有优化器/clip/step/自动重训、阈值选择、新bank或旧缓存写入；旧top-level/horizons源码及训练恢复指纹未改。入口自动定位已完成GT三轮，也允许DIAG_CHECKPOINT显式指定；快照/整数批次窗口状态支持同DIAG_OUT续诊断。
详见 `docs/JOINT_OCC_LOSS_ACTION_AUDIT_CN.md`；只需服务器一趟 `bash tools/joint_occ_experiments/diagnostics/run.sh` 的summary，不启动15/20轮或新损失对照训练。
本地新诊断与原fused/两未来分组相关回归29项通过，补充非有限梯度拒绝后新专项8项通过；含分离梯度之和还原原实际WM训练梯度、原硬六帧不变、真实eval session、完整CLI中断/64窗续评不重计及source/Adam/RNG字节不改。CLI help/Bash/diff通过；无完整nuScenes/L40S，尚未得到真实诊断结果，不宣称已找到掉分主因或loss收益。

用户批准实现 GT/内部Pred 两种配置，随机同初始化，TRAIN20%×3轮；旧正式模型不修改。
新增 `tools/joint_occ_experiments/`：原Transport+SurfaceCCR+8710参数历史控制ego头，训练soft投影/合成与全六帧CE+Lovász，source几何/existence和R10ego辅助；硬推理保持owner规则，ADD0.5/REMOVE0.95及REMOVE-off同时报告。
Pred未来GT pose/background/label毒化测试通过，导航仍明确GT-derived条件；soft/hard并非同输出，首8评估窗只做gap诊断。
只读复用原TRAIN/VAL history geometry；Pred丢弃未来GT ego条件字段即时重建，旧冻结feature bank不复用、无新bank/缓存重建。
历史描述拟合+raw/cache读取CPU有界并行预取，GT/Pred共享一次历史fit；native实时投影/Strong，稀疏概率合成、批量Lovász、逐horizon backward，训练/评估同目录可恢复。
新增21项CPU真实梯度/几何与CLI恢复测试通过；相关回归103 passed/17 CUDA skipped/16 native deselected，Bash/help/diff检查通过。没有服务器nuScenes/GPU性能结果，不沿用旧54.945FPS或宣称已涨点。
本地相关回归的native DLL读取受Windows权限限制，旧DLL未修改；CUDA不可用，服务器需测实际backend。
详见 `docs/JOINT_OCC_EGO_PILOT_CN.md`；服务器入口 `bash tools/joint_occ_experiments/run_paired.sh`，可先`--max-updates 8`保留进度再同目录resume。
服务器首启在NuScenes初始化失败，未训练/写断点：我把info路径误传为第二位置参数version。TRAIN/DEV均已改显式info_pkl关键字；严格真实adapter签名的CLI fresh/resume测试先复现再修复，21项CPU回归通过。此次不改模型/缓存，原未创建的输出路径可直接重跑8步，不使用RESUME。
随后服务器旧版paired8成功：GT1.410/Pred1.470/两路含输入2.900 s/窗，dense渲染/loss/backward约88%。用户最新指定先GT-only，并授权优化共享重复反传和概率合成。新增`run_gt.sh`/`--mode gt,pred,paired`，仅实例化活动模型；两路是公平对照，无权重共享/互相学习收益。
融合稀疏有序合成+Lovász解析一阶backward；六帧renderer VJP在边界累积后一次multi-root回传WM/CCR，仍完整六帧全网格CE+Lovász、所有geometry/action链路不断梯度，硬推理未改。38项CPU数值/梯度/全部参数Adam更新及三模式resume测试通过；BF16汇总顺序不承诺bitwise，服务器速度待实测。
新版mode/backend/实现指纹锁入断点契约，旧paired8保留且不直接移作GT-only恢复；新独立GT目录8步后同目录`RESUME=.../last.pt bash tools/joint_occ_experiments/run_gt.sh`接续TRAIN20%3轮/DEV512，无重建bank或旧缓存写入。
本轮最终相关回归81 passed/7 CUDA skipped（非全仓CI）。CPU synthetic 18类×640000体素/24编辑的合成+CE/Lovász反传2.884→2.001s（1.442×），不含网络/几何，不能外推L40S整轮或FPS。

最新服务器 GT-only 8 步成功安全停：计算0.933024s/窗，含输入0.947234；render/loss0.416069、占据backward0.297100，约76%仍在全网格。
用户最新要求更多执行提速，先本地实验后服务器测。新增独立 `performance/` 子目录：prepared CE/Lovász 共用完整梯度buffer，以及零贡献Lovász背景尾部压缩；完整GT前景/漏检梯度与所有非零概率仍参与，密集时回退。
本地640000体素稀疏概率loss forward/backward 0.944→0.754/0.349s；密集负对照1.739→1.532/1.669s，数值/梯度一致；不外推L40S整轮、精度或FPS。
只读原GT-only断点的 A/B/C `performance/run_probe.sh`：临时真实forward/backward/Adam、同RNG八批两遍；新目录snapshot/speed/summary，无科学更新/旧缓存写入，无自动推广。
旧 top-level trainer/依赖源码不改，c57 GT-only恢复契约仍兼容；候选尚未接入正式续训。详见 `docs/JOINT_OCC_GT_SPEED_PROBE_CN.md`。
此次新loss/probe+原fused/模型专项89项CPU测试通过，CLI/Bash检查通过；本地CUDA环境无法完成导入，未代跑或声称真实服务器加速。
服务器 GT源目录已确定 `outputs/p0_f9_joint_occ_ego/gt3_fast_20261011_123726`；probe_20261011_131016在combined八步累计检查首个报告的128维张量中1元素差2.2003544e-5>2e-5退出，余下未统计；原版0.636530s/窗，原断点未改。
修复新loss导数归一化顺序：先合CE/Lovász，再统一除，匹配旧计算图；非单位upstream逐体素梯度CPU精确一致。仍不宣称CUDA通过。
新probe数值超标不再丢剩余速度：报告参数名/count/max、warm真实1s概率loss梯度、两遍原版replay及原预算；任一parity失败候选gate仍False，结构/数据/非有限错误仍停止。
只改performance子目录与其测试/文档，旧GT resume指纹不变；新progress保留每阶段结果，不自动接入训练。下一步新目录重测同一已知GT source，不重建cache。
修复后新loss/probe及原fused/模型专项92项CPU通过（非全仓CI）；实际L40S仍待重测，旧失败目录保留。
服务器重测已完成：current_fused/combined/sparse分别0.636582/0.647168/0.663386s/窗，两候选均更慢，不采用、不自动接入训练。
两候选抽查真实1s概率loss及11520000个逐体素梯度全部一致；八步最终参数/Adam按原预算通过，maxabs分别7.934868e-7/2.928078e-6。
原版自身第二遍八步replay失败：62781个元素/34张量，maxabs0.001094939；保留FAIL，不能归咎新loss或据此宣称训练无效。具体原因未定位，不能把抽查loss梯度一致扩展为全训练确定性。
只读检查确认每遍重载model、深拷贝Adam及恢复Torch/CUDA/NumPy/Python RNG；未发现直接源state别名污染。CUDA归约/反传非确定性只是候选解释，未在当前数据定位，勿当结论。
旧GT训练源码相对c57未改，原last.pt未被测速更新；建议停止无收益loss后端测速，保留current_fused接续既有GT-only三轮试训，不重建cache、不重置LR/RNG。
用户已恢复原GT-only，贴L40S瞬时GPU19%/3507MiB；top显示的是约1TiB宿主机，不是容器80GiB/10核，未贴训练进程CPU占用，不能据此认定具体CPU瓶颈。
只读代码确认global batch4为逐窗口梯度累积，六未来逐horizon render/loss/backward；增加window_batch只延长这个循环且改变优化步数，不等于真实GPU合批。CPU历史/raw已经4worker、depth2预取；本地未改执行或运行中训练。
潜在下一步是同global batch4内真正合批WM/CCR/有界多horizon渲染，以及CPU模型依赖几何与GPU工作重叠；这仍需实现/实测，不保证提速或精确一致。先读取当前最近64更新的既有分段与input_wait，不另跑训练/大测速、不重启失败loss方案。

用户提供最近64更新56–119：输入等待0.0000044s/窗，计算0.762005；render/loss0.364405、occupancy backward0.262491，约82%。用户明确同意实现执行合批，并已停止服务器训练。
新增独立 `batching/`：同global batch4/source128合批WM和CCR，窗口actor偏移隔离；CPU投影最多4worker、一次XY/yaw读回；未来两帧并行full-grid soft renderer/CE/Lovasz，每帧类别均值仍独立；四窗边界VJP一次共享网络反传。
仅GT opt-in；原top-level GT trainer/依赖/缓存签名保持不变，不采用已慢的prepared/sparse loss。原hard推理/评估/FPS边界未改。
服务器入口 `bash tools/joint_occ_experiments/batching/run.sh probe`：同原last.pt/RNG八批，两遍原版/packed_h1/h2真实Adam只读测速。原容差、原版repeat、实际首窗概率/梯度、显存10%余量、两遍均快且≥1.1倍均需通过；未通过不允许新续训，不放宽容差/自动推广。
续训入口同script `resume`，显式成功speed.json且新目录导入；仅implementation更改，weights/Adam/RNG/cursor/LR/科学contract其余字段完全保持。新checkpoint绑定执行源码/group，后续同入口同目录恢复；原source从不写入，旧run_gt仍恢复旧断点。
详细命令及CUDA未验证限制见 `docs/JOINT_OCC_GT_BATCHING_CN.md`。本地17项新合批/恢复/只读测速+43项原fused/模型/probe回归共60项CPU通过，Bash语法检查通过；不是完整CI，尚未测L40S/BF16吞吐或精度，不保证真实服务器加速。
服务器合批测速已完成，source update149，未发布科学更新：原版0.703916、packed_h1 0.754875、packed_h2 0.475507s/窗；h2两遍均快、1.48035倍且显存余量通过，h1反而慢。
首窗固定边界原/组renderer概率、loss、poses/targets/yaws/mapped/actions梯度全部通过；不代表packed网络或多步Adam一致。两合批版本warm/八步model+Adam失败，h2 maxabs0.00184412；原版自身repeat仍失败maxabs0.00071803，失败门槛不放宽、不自动续训推广。
收益在h1→h2增加未来帧分组时出现；不能将数值漂移直接等同精度下降或无害，不采用失败packed版本。

用户批准窄版实现：新增独立 `horizons/`，保留原逐窗口WM/ego/CCR forward与逐窗口shared backward交替顺序，仅0/1、2/3、4/5两未来帧并行render/full-grid CE+Lovasz/occupancy backward。
batch4/source128、每帧loss归一化、source/existence/R10辅助、clip/Adam/全程余弦/RNG/cursor不变；原top-level训练源码及硬推理保持原样，没有使用慢的prepared/sparse候选。
只读两路线probe（current_fused/sequential_h2），同source真实八批、AB/BA两遍，复用原概率/梯度/Adam与baseline repeat严预算/≥1.1倍/显存10%门槛；不重跑packed_h1/h2。原始断点仍为gt3_fast_20261011_123726/last.pt实际cursor，未发布新训练更新。
独立成功receipt与代码/source SHA才允许新目录导入；失败packed receipt不能授权窄版，后续同目录resume不重置权重/Adam/LR/RNG。原GT入口可继续原样恢复。
7项窄版+60项既有CPU回归共67 passed，包含两真实Adam更新、原逐窗口前后向交替、空物体、只读两路线reset与显式导入/恢复；CLI help/Bash/diff检查通过。不是完整CI/真实CUDA/BF16吞吐或精度结论，窄版L40S验收仍待用户运行。
入口 `bash tools/joint_occ_experiments/horizons/run.sh probe`，命令及边界见 `docs/JOINT_OCC_GT_HORIZONS_CN.md`。以上为严格验收实施历史；最新决定见下段，不再测速。

服务器窄版已测：原0.741306→sequential_h2 0.504552s/窗，1.469237倍、两遍均快且显存余量通过；首窗renderer概率/loss/梯度及warm单步更新通过。八步候选maxabs0.000817965、原版repeat0.000402516，原严门槛FAIL保留，不称无损。
用户明确接受此小幅浮点差异，授权直接接续GT三轮，不追加128步A/B。新增仅在 `horizons/continuation/` 的显式accept入口，原horizons顶层实现/内核/测得fingerprint均未改，复用已有speed.json，无重测/新bank。
初次必须新目录、同source SHA/执行fingerprint、单步/梯度通过、≥1.1倍/两遍快/显存余量；独立接受策略仅容许已测多步与repeat状态maxabs≤0.001，不改原数值预算或pass_gate。float_drift_acceptance.json和训练结果保留原FAIL/未知质量；不承诺最终精度不变。
原weights/Adam/RNG/cursor/数据/batch4/source128/原三轮余弦不变，完成后原dev512硬输出评估；原GT断点与旧缓存只读。新目录同入口Ctrl-C完成update保存、同目录resume，不是额外三轮。
本地新增10项+窄版7项共17 passed，含原149游标/Adam/RNG导入、两次续训、源/失败report不改、拒绝未显式接受/过大漂移/单步/梯度/数据/执行不匹配；CLI/Bash检查通过。只测必要CPU恢复，不跑完整CI或新CUDA测速；服务器由用户运行。
直接入口 `bash tools/joint_occ_experiments/horizons/continuation/run.sh`；命令见 `docs/JOINT_OCC_GT_FLOAT_DRIFT_CONTINUE_CN.md`。

## 当前目标

Clean Joint Surface CCR 已完整随机联合训练20轮；固定5/6/8/12/14整网等权平均已完成DEV512选择、full4369质量和256×3正式FPS验证。
当前主候选mIoU44.153241 / MovingMicro32.214410；同256×3无损并行Strong多数投票正式FPS已到54.945，详见下文，旧路径保留默认。
最新已完成：用户服务器跑完static_carry全集，明确确认作为最终论文6秒长时预测路线；avg4–6 IoU41.218153 / mIoU29.117864。
记录在 `docs/SURFACE_STATIC_CARRY_LONG6S_FINAL_RESULTS_20261009_CN.md` 与 `docs/results/` 原始摘要/结构化转录。
Waymo两套zero-shot已全部完成并归档：2Hz 7998/7998，joint平均IoU61.209875 / mIoU52.152906；10Hz 39987/39987，joint平均IoU75.100701 / mIoU65.997882。2Hz相对同模型Transport为+0.957959 / +0.418007 pp；10Hz为+0.337958 / −0.113556 pp，不能称两协议全面提升。10Hz按原native+2/+4/+6，即0.2/0.4/0.6s，不是物理1/2/3s；不与2Hz绝对分数直接比较。档案`docs/WAYMO_ZERO_SHOT_FINAL_RESULTS_20261010_CN.md`及`docs/results/surface_ccr_waymo_2hz_10hz_20261010.json`；无需再resume或重跑Waymo。

## 核心决策与约束

- 4 历史→6 未来。固定 weighted ADD raw sigmoid@0.5，REMOVE-off；没有 KD/AE。
- Surface 描述进入 CCR 编码，实时投影高度/相位进入静态条件读出；不是外挂后处理修正器。
- 正式模型应运动/静态/动态一起干净随机初始化训练；当前 20 轮正是这一版，不能混称早先冻结验证。
- 原 checkpoint、optimizer/RNG、缓存、实验目录保留。平均文件只用于评估，不能拿它当 resume。
- 同训练契约、同模型配置、同 TRAIN 正权重；只平均可学习参数，固定 buffers 不平均。
- 评估共享确定性历史/几何和 Moving 区域，不能共享候选之间 learned poses/features/输出。
- 固定 Dense Forecast FPS 边界：CausalHistoryState→fresh Strong/KTA+live motion+CCR+实时投影+六帧 dense。历史表示准备另报，不称 raw-input E2E。
- 新平均正式FPS已测40.250（256窗口/150场景×3）；不能沿用旧冻结Surface/Frozen B的FPS。
- DEV64/DEV512 和 VAL 扩大人口均已参与研究，不称独立测试；本次已授权固定平均的 full4369，不自动重训/部署。

## 当前进度与服务器锚点

完整 TRAIN20430×20，102893 updates，whole-cycle cosine floor0.1，无 tail。所有 `epoch_XXXX.pt` 保留在原及续训目录。
当前完成锚点（文件在 run 根目录，不是 model 子目录）：

```
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787
  last.pt
  training.json
  epoch_XXXX.pt
  summary.txt
```

其他轮次按相同 contract 在 `/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr` 下查找。
VAL 只读缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1`。
TRAIN 缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_v1`。
单卡 L40S，10 核、80 GiB；不再自动尝试多卡/改变 batch4/source128。

DEV64 关键 Joint 结果：

| epoch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| 6 | 53.476777 | 41.666093 | 33.921350 | 28.234393 |
| 8 | 53.531044 | 41.574074 | 33.866847 | 28.646939 |
| 12 | 53.574938 | 41.553765 | 34.735137 | 30.010121 |
| 20 | 53.499071 | 41.172214 | 33.323547 | 27.574565 |

第 6 轮最高 mIoU；第 12 轮最高 IoU/Macro/Micro。不应只按最后一轮判断效果。
按 mIoU top5 的平均来源固定为 5/6/8/12/14，跨较大训练间隔，不保证平均更好。

原第 20 轮 DEV512：

| branch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| static_repair | 52.541410 | 39.619555 | 23.362541 | 28.837789 |
| dynamic_repair | 50.628250 | 39.324149 | 24.779588 | 30.681102 |
| joint | 52.543060 | 40.043394 | 24.779588 | 30.681102 |

## 最新已完成 DEV512 对照与冻结选择

| candidate | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| epoch6 | 52.451547 | 40.248853 | 24.912767 | 30.599135 |
| epoch8 | 52.488365 | 40.300115 | 25.243657 | 30.535807 |
| epoch12 | 52.506929 | 40.199564 | 25.281199 | 30.875808 |
| mean5/6/8/12/14 | 52.372722 | 40.432276 | 25.718056 | 31.486519 |

平均的三个时距 mIoU/Macro/Micro均最高，但IoU略低，不称四项全面提升。
收益主要体现于Transport；平均Transport mIoU39.362646 / Micro30.014088，CCR再增+1.069630 / +1.472431 pp。
四候选512窗口合计705.52s，这是质量eval耗时，不是正式FPS。
新平均 full4369 和正式FPS均已完成。

2026-10-09 服务器全集报告：固定平均（5/6/8/12/14）在 full4369、150 场景上
IoU55.211054 / mIoU44.153241 / MovingMacro28.256901 / MovingMicro32.214410。
同模型 Transport：53.631093 / 43.218661 / 26.889216 / 30.729273；CCR 增益分别
+1.579960 / +0.934580 / +1.367684 / +1.485137 pp。
相对旧 Local full4369：IoU-0.347217 / mIoU+0.124314 / Macro+0.968314 / Micro+0.974187 pp。
耗时1649.02s是带GT/整数指标的质量评估，不是FPS；仍不称独立测试。
结果：`outputs/p0_f9_joint_surface_ccr/mean_full4369_20261009_104036_848/full_validation.json`（服务器）。

正式FPS（同均值，256窗口/150场景×3，actualCUDA）：fused_eager152.852ms/39.254FPS；fused_graph149.067ms/40.250FPS，P90=186.348ms，256/256六帧和概率byte parity。
仅2个图捕获，计时外；无拒绝。Strong82.109ms是主瓶颈；读出24.515ms已包含实时phase14.053ms，不能重复相加。峰值allocated253.885MiB。
历史准备149.995ms、surface descriptor104.183ms单独排除，不称raw-input端到端FPS。
服务器结果：`outputs/p0_f9_joint_surface_ccr/mean_fps_20261009_113139_838/speed.json`。

## 本轮实现与验收

独立分支 `feature/v22-surface-aware-ccr`，工作副本 `swfm-surface-ccr`。原 `swfm-v20utc` 的用户未提交改动不动。
用户已授权该新分支提交/推送；不能推断新服务器训练/部署授权。

新增选择/平均模块、共享历史批量评估模块、CLI 与 Bash wrapper。
跨续训目录严格校验 lineage、完成 epoch、报告、源 SHA、原 TRAIN 权重，缺文件/重复冲突停止。
新平均权重标记 evaluation_only，不含原 optimizer/RNG。
四候选共用一次窗口准备，分别实时运行模型；每 8 个完整四模型窗口保存整数状态，严格 --resume。
既有 trainer/model/cache 依赖文件未改，避免破坏原训练 resume 或大缓存 namespace。
120 项相关本地回归通过，含真实 CPU/CUDA 单路径 vs 批量计数一致、中断恢复、平均/源文件只读与空类别指标，以及原联合恢复/表面读出回归。
增加只读缓存 namespace 预检和明确四窗口 CPU 预取后，10 项专项复查也通过；新文件 AST 和 Bash 语法检查通过。未声称全仓 CI 或真实 nuScenes 质量通过，本地没有完整服务器数据。

操作与风险见 `docs/JOINT_SURFACE_CHECKPOINT_AVERAGING_CN.md`。
服务器运行 `bash tools/real_motion/run_p0_f9_joint_surface_checkpoint_comparison.sh`；结果在新 comparison 目录，不覆盖训练。
中断后指定原 comparison 输出目录和 `SURFACE_COMPARE_RESUME=1`，不是恢复训练。
本次增加 `eval_p0_f9_joint_surface_mean_full.py` 与 wrapper，复用同一评估引擎，仅加载已完成对照中的平均文件。
只读已选来源，完整VAL4369不做DEV512筛选、不导入epoch20子集报告；新结果写full_validation.json，严格同目录整数resume。
操作见 `docs/JOINT_SURFACE_MEAN_FULL_VALIDATION_CN.md`。121 项相关本地回归通过，包含完整人口、来源冻结/只读、真实共享CPU/CUDA路径和CLI续评等价；AST/Bash语法通过。模型/trainer/缓存依赖未改。服务器未运行，不冒充全集结果或完整CI。

新增平均权重正式FPS入口；复用原 `paired_speed`（旧20×3默认兼容），不改训练/model/cache依赖。
19项相关本地回归通过，含真实CPU/CUDA读出及六帧合成一致、均值FPS算法、权重只读、缺帧拒绝、计时内建图拒绝、旧入口回归；CLI help/Bash语法通过。
这不是服务器实测FPS或全仓CI。服务器命令：`bash tools/real_motion/run_p0_f9_joint_surface_mean_fps.sh`；输出新 `mean_fps_*` 目录。
可用 `SURFACE_MEAN_FPS_WINDOWS=512` 扩大人口；中断保留partial日志，不改断点，重测使用新输出，不混拼前次计时。

本轮简单无损候选：`real_motion/strong_warp_execution.py`。保留原FP32逐时距GEMM、5e-3边界判定及FP64 NumPy纠错；合并边界索引与标签/known回传，去掉每时距any/变长gather同步。只保留本次最多6网格的临时buffer，不缓存预测。
默认仍reference；ContextVar显式scope可选buffered，异常自动恢复、不会改其他线程。7个历史几何namespace文件、10个训练指纹文件及native ABI都未改；旧权重/缓存继续兼容。
RTX3050上只读真实replay.zip：6窗口×5，六帧逆变换32.848→24.410ms（1.346×），与旧CUDA及CPU FP64参考逐字节一致。这不是整网FPS或L40S预测。
33项相关本地CPU/CUDA回归通过，含越界/边界、跨6网格批次、Strong anchor/components/CLEAR、旧FPS和graph读出；Bash语法通过。
同一正式入口设`SURFACE_MEAN_FPS_COMPARE_STRONG=1`，成对测原fused_graph与buffered Strong/fused_graph，默认仍256×3同seed/同人口，不自动采用候选。

## 新增：冻结平均 Surface CCR 的6秒递推（2026-10-09）

用户已授权实现新版6s评估；旧Local epoch19入口原样保留。不重训，不改变权重/阈值。
新入口 `run_p0_f9_joint_surface_long6s.sh`（默认dev64）及 `run_p0_f9_joint_surface_geniedrive_long6s.sh`（官方公开代码2569人口）。
只读同一已完成DEV512对照的5/6/8/12/14平均文件；严格4→6→最后4预测→6，报告1–6秒。
固定接续reconciled为主结果、redetect同人口对照，共享第一段；无GT身份接续、不按分数挑路线。
每段实时Surface Atlas/phase/source读出，ADD raw0.5/REMOVEoff；第二段禁止真实未来缓存/adapter读取，保留handed velocity。
第一段已有VAL cache只读核验；官方早期300窗口实时四历史重建，不丢样。省掉旧Local memory/frontier计算；native融合投影、4窗口预取、8192完整块静态CUDA graph有界复用。
未来标签/Moving support仅在所有路线预测完成后读取，Moving保留原始t0；第一段四历史/六张Transport/六张Surface逐字节gate，第二段完整概率/六帧gate。
同目录整数断点恢复：每8个完整窗口周期保存，SIGINT/SIGTERM窗口边界保存；模型/人口/实现/执行开关变更拒绝resume。
159项相关本地回归通过、28项跳过（CUDA相关未在本次CPU环境复现），包含实际Surface网络、native CPU/NumPy两段一致、早期起点、预测历史cache隔离、完整CLI snapshot/中断恢复。Bash语法和CLI help通过；不冒充全仓CI或服务器6s精度。
操作见 `docs/JOINT_SURFACE_MEAN_LONG6S_CN.md`。服务器下一步跑新dev64入口，发回summary；不用重建prototype/cache、重新平均或恢复训练。

## 6秒服务器验收与下载修复（2026-10-09）

Surface平均dev64真实6秒已完成，first/second exactness通过。接续主路径4–6s平均mIoU23.305861/Micro8.004605，
redetect对照23.148164/7.179412，接续增益+0.157697/+0.825193 pp；64窗口103.21s，不称FPS。
601/630当前source成功接续；长期预测仍衰减，不与旧dev512不同人口直接比较。
结果服务器：`outputs/p0_f9_joint_surface_ccr/long6s_mean_dev64_20261009_170347_837`。

GenieDrive全集入口下载官方metadata时报网络Errno101，尚未进入评估、未创建评估状态。
修复下载器：默认官网→网络失败尝试HF-Mirror；可显式`GENIEDRIVE_DOWNLOAD_ENDPOINT`或`HF_ENDPOINT`，
仍固定revision/size/SHA，校验错误不重试、不发布/反序列化、不覆盖原文件。标准HTTPS_PROXY兼容，
所有地址不可达时明确离线上传指引，无长traceback；只移除本次自己创建的.part。
76项下载/人口/Surface两段/接续相关回归通过，4项CUDA跳过；服务器镜像连通性未验证。
下一步更新后显式镜像重跑全集，`SURFACE_LONG_REDETECT=0`可省对照；下载阶段失败不使用resume。
完全离线时上传同一官方固定metadata到`data/geniedrive/world-nuscenes_infos_val.pkl`，再离线校验。
无模型/权重/阈值/训练/cache namespace变更；约2569窗口60–80分钟仅是dev64外推估算。

## 未解决问题与下一步

1. buffered Strong已提交并实测：L40S151.236→149.391ms，整网仅1.0124×；逆变换12.061→10.407ms，多数投票66.047→65.949ms。该路线收益小，不继续单独抠warp。
2. 用户授权试多数投票：新增显式4线程分块+紧凑临界坐标+仅在回退邻域取类别，仍原native整数投票及SciPy float32决胜。原路径保留默认，训练/缓存指纹及native ABI未改。
3. 本地真实6窗口×7六帧多数投票59.873→19.949ms（3.001×），完整SciPy/native/新路径byte parity；仅内部CPU阶段，不声称整网FPS、L40S或训练提速。详情`docs/STRONG_MAJORITY_EXECUTION_CN.md`。
   93项相关CPU/CUDA回归通过（含旧FPS及最新6秒递推），CLI/Bash/diff检查通过；不冒充全仓CI或服务器结果。
4. 服务器同256×3成对验收已完成：原155.390ms/38.613FPS，新109.200ms/54.945FPS、P90=145.082ms，整网1.423×；概率及六帧dense byte parity256/256。多数投票67.410→21.910ms，Strong84.759→39.042ms。显存allocated253.885MiB不变，不改权重/阈值，不自动修改旧默认或重训。
5. 全集已追回mIoU/Moving，IoU仍略低；不增加重模块，不改结构/阈值。

## 新增：I²-World Waymo 2Hz zero-shot（2026-10-09）

用户选择按I²-World对齐并授权实现。新入口 `run_p0_f9_joint_surface_waymo.sh`，文档 `docs/WAYMO_I2WORLD_ZERO_SHOT_CN.md`。
官方源码固定 `II-World@661d830f9b34ee03ce368db164a72753ab8764a3`；全局timestamp排序后stride5、202validation场景，未来index1/3/5报告名义1/2/3s，场景边界重复有效历史/未来，不静默丢样。实际时间/补帧数量另报。
固定同一5/6/8/12/14均值，四总历史（含t0），ADD raw0.5/REMOVEoff；用原Surface预测路径及无损native/majority执行，逐进程重做输入/Transport/概率/六帧exactness。
只读历史voxel_label及ego poses；未来pose是显式条件，未来occupancy在六帧预测完成后才读取。无未来annotations/mask，历史dense visibility全真与官方输入一致，不加额外mask信息。
复现官方Waymo→nuScenes18类映射；默认raw free23，已确认文档发布free15时显式切换并记报告，未知/混合编码拒绝。不能再映射模型预测。
同时报告官方零IoU排除的mIoU和union有效类保留零分的标准mIoU；binary occupied IoU，无camera/lidar mask，per-horizon累计后求均值，官方逐时距round2均值另给。不硬搬nuScenes Moving指标，不把eval时间称FPS。
默认256MiB不可变原始帧LRU；不使用nuScenes几何cache、不存learned features/预测。每8完整窗口整数checkpoint，Ctrl-C/SIGTERM边界保存；严格同目录、模型/metadata/人口/NPZstat/实现/执行契约resume。
数据仅需validation0.4m NPZ、可信官方 `waymo_infos_val.pkl`、`cam_infos_vali.pkl`；下载链接、目录和audit/full/resume命令在上述文档，不自动下载大数据。
本地新协议与旧Surface/四历史递推相关回归运行，真实小网格CPU模型预测覆盖早期/中间/末端；未跑真实Waymo或本次CUDA服务器，不宣称zero-shot精度/速度或全仓CI。
数据已下载并解压；服务器只读metadata统计39987原始帧→7998锚点，7796同场景链接全部stride5，15个timestamp跳变（最大1.199943s）。原逐对0.35–0.65s硬检查误拒绝官方数据；现保留官方人口/实际pose/timestamp，将跳变记入`timestamp_gap_audit`，仍拒绝非正时间/倒退帧及总体错频率/错单位。不重采样、不丢窗、不改模型时间步。
时间检查修复的Waymo/Surface/FPS CPU回归42通过、12项GPU/native相关跳过；2Hz服务器完整结果已收到，见下节。

### Waymo 2Hz完整服务器结果（2026-10-10）

固定nuScenes均值、无Waymo训练/校准/阈值搜索；全部7998锚点保留，无重采样/丢窗。

| 名义时距 | Transport IoU | Joint IoU | Transport mIoU | Joint mIoU | ΔIoU pp | ΔmIoU pp |
| --- | --- | --- | --- | --- | --- | --- |
| 平均 | 60.251916 | 61.209875 | 51.734899 | 52.152906 | +0.957959 | +0.418007 |
| 1s | 67.552118 | 68.209418 | 59.487389 | 59.602591 | +0.657300 | +0.115202 |
| 2s | 59.659763 | 60.669601 | 50.866556 | 51.381103 | +1.009838 | +0.514547 |
| 3s | 53.543867 | 54.750606 | 44.850752 | 45.475025 | +1.206739 | +0.624273 |

本次I²-World零IoU排除口径与标准mIoU数值恰好相同，不代表两种定义总是等价。
7796同场景stride5链接中15个timestamp异常（0.1924%，最大1.199943s），实际跨度另报；上表仍是index-based名义时距。
raw free23、无camera/lidar metric mask，未来ego pose显式条件；未来occupancy仅预测后读取，Moving未测。
可确认CCR在该zero-shot人口上三个时距有正增益；没有逐场景统计/置信区间，不能声称所有场景提升、动态指标提升或优于I²-World论文。
服务器摘要：`/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo_i2world_2hz_20261009_225100_83458/summary.txt`，详细`waymo_validation.json`。
这是用户粘贴服务器摘要，不是本地原JSON；未提供累计wall time或snapshot SHA，不猜测。评估wall time不等于正式Dense Forecast FPS。

### Waymo两协议最终归档（2026-10-10）

用户提供10Hz完整摘要：status=complete、39987/39987；四进程同目录恢复后完成，最后invocation复用保存前缀35959。原BrokenProcessPool原因未确认，不记录为OOM。
2Hz/10Hz完整Transport+Joint、全部三个时距/两种mIoU口径、timestamp gap/目标跨度和源目录已记录到独立结果文档、两份摘要及结构化转录JSON；只是用户服务器结果归档，未本地复跑或下载原counts/contract。
2Hz三时距IoU/mIoU全正收益。10Hz Joint平均75.100701/65.997882，Transport74.762743/66.111438；最短0.2s mIoU−0.557557、0.4s−0.006147、0.6s+0.223036pp。不据此再调阈值/重训/挑权重，不称SOTA。
最终10Hz目录`outputs/p0_f9_joint_surface_ccr/waymo10_parallel4_20261010_055012`。两协议均无Moving或正式FPS；没有跨全部invocation总wall time或服务器实际HEAD/snapshot SHA，未知保持null。
详细索引`docs/WAYMO_ZERO_SHOT_FINAL_RESULTS_20261010_CN.md`。以下Waymo入口/测速/接续内容为已完成阶段的实施历史，不是新的待跑任务；后续仅按需收集既有原JSON/契约作复现记录。

用户明确要求10Hz按I²-World代码做：同固定提交load_interval=1/eval_metric=miou，模型forward_test取六未来零基下标eval_time=1/3/5，即native第2/4/6帧，实际名义0.2/0.4/0.6s，不照抄配置注释的错误物理1/2/3s标签。
独立`waymo_i2world_10hz.py`及`run_p0_f9_joint_surface_waymo_10hz.sh`，原2Hz实现/指纹不改，运行/续评保持兼容。全native人口按实际metadata计算（当前39987），共用一次六帧模型预测评分三下标，不重复三次。保留训练slot clock、同冻结均值/四历史/阈值；未训练或插值适配10Hz，不能称物理长时距或同历史预算比较。
同一份metadata/NPZ，无新下载/大缓存；只读源数据/权重，六预测后才读未来GT，完整窗口整数resume拒绝2Hz/10Hz混拼。默认10Hz进程2worker，用户可同卡并发但资源竞争不保证总耗时更短；不干预现有2Hz进程。
本地合成元数据/实际小网格CPU及native模型、旧2Hz/Surface/FPS/递推回归78通过、16项CUDA跳过；Bash语法和CLI help通过，未跑真实10Hz或正式测速。服务器下一步更新后独立后台启动10Hz，发回各自summary。协议解释和audit/full/nohup/resume命令见Waymo文档。

最新服务器分段：2Hz约0.800s/窗，其中历史/搬运准备0.290s、证据构建与投影0.437s、head0.020s；10Hz约1.004s/窗，对应0.479/0.448/0.020s。前两段占总耗时91%/92%，I/O约0.010s，不应继续把原始帧读盘或显存当首要瓶颈。`evidence_projection`包含canonical support、Surface Atlas描述及实时投影，尚无内部细分，不能宣称全部耗时来自投影。
历史component在state构建和causal_source_history内重复提取，相邻窗口四历史亦重叠。用户授权后已实现独立10Hz fast入口，旧2Hz/10Hz实现文件及指纹不改；不干预正在跑的2Hz。

## 新增：Waymo 10Hz无损执行与接续（2026-10-10）

`run_p0_f9_joint_surface_waymo_10hz_fast.sh`：1024MiB/32帧只读内容键component/世界点LRU、取消同窗口重复提取、相同K16/半径2.5/FP64算式的4096行Surface描述并行、最多一个next-history纯CPU预取。
配准/Strong/support/Atlas/learned features/未来投影均live，不写大缓存；预取只在当前六预测完成后调度，无未来GT进入预测。缓存single-flight短锁，日志不能隐式阻塞worker；线程累计耗时不与wall time相加。
默认先做同连续16窗、两遍交替旧/new实际六帧+概率byte gate及测速；每计时pass清几何LRU只暖首窗，不拿全小人口暖命中冒充真实吞吐，再开始全native39987固定评估。
`WAYMO10_CONTINUE_FROM`显式读原停止目录，原contract/state SHA+JSON快照保存新输出；原实现/数据/权重/语义全同才迁移整数前缀，允许仅CPU worker与新增执行信息变化。新目录常规resume仍严格契约；kill -9只恢复最后周期保存窗口，不假装恢复progress尾条。
本地相关CPU回归135通过/18跳过，RTX3050 CUDA/native/真实静态Graph重放及Surface/Strong/递推回归105通过；实际模型运动输入/配准/evidence/plan/概率/六帧/整数指标一致、CLI原结果只读和迁移/续评验收。Bash语法/CLI help通过，不声称全仓CI或真实Waymo/L40S验收。
本地12万合成表面点相同4worker：describe625.588→207.769ms、3.011×、全部bytes相同；只是内部CPU微基准，不是完整eval/FPS。
服务器新版`outputs/p0_f9_joint_surface_ccr/waymo10_fast_20261010_040604`已接续：同窗口旧0.522039339→新0.396463246s，1.316741×、耗时减少24.05%。最近1255–1510共256窗平均0.4739s，不与成对16窗不同负载直接算倍率，也不与原并发旧10Hz约1.004s/窗硬比。
父级prepare0.2189s、evidence_projection0.1858s；子级state/tracks/tubes/Strong0.1695s（总窗35.8%）、surface邻域拟合0.0850s（17.9%），head仅0.0153s。历史就绪0.0013s，缓存没有明显等待；打印每32窗hits+96符合三历史帧复用。不要继续优先加缓存容量或改网络。
上述是暂停前的server日志；用户现在要求继续无损加速并利用闲置资源，新增实现见下节。不把父子计时相加、不自动切换运行中代码。按旧均值从1510算剩余约5.1h仅外推，不是新并行ETA。
操作见`docs/WAYMO_10HZ_FAST_EXECUTION_CN.md`。原10Hz锚点`outputs/p0_f9_joint_surface_ccr/waymo10_20261009_233328`保持只读；继续等待新10Hz完整质量，不重跑/调参2Hz，不改旧权重、nuScenes缓存或长6s结果。

### Waymo暂停后的无损内核与有界并行（2026-10-10）

新增文件，不修改旧2Hz/10Hz/fast-v1实现指纹；native独立ABI1/编译缓存，旧column ABI保持原样。
纯历史占据点齐次坐标/配准排序只读LRU；同FP64整批变换+稳定最小距离C++冲突选点；四帧tube线程与fresh Strong重叠；原trimmed ICP同算式重用当前source树；相同K16/2.5半径/归约顺序的C++表面fit，不缓存learned state/注册矩阵/未来相位。
serial-v2本地整体仅1.035–1.047×，用户指出收益小，遂增加独立spawn多窗口并行，不继续堆局部微优化。
默认2进程×2线程、同一GPU，各保留独立模型/Graph/历史LRU；连续8窗/任务、最多2任务在途。仅传小整数指标，主进程有序连续提交，每8窗周期保存；Ctrl-C/SIGTERM收完在途块，无指标洞/重计数。整块错误不部分提交，旧源目录contract/state及权重只读。
本地真实RTX3050、200×200×16/16请求source模拟、随机小模型、相同16窗×2交替：旧fast-v1单进程4线程0.723398→新双进程各2线程0.372681s/窗，1.941067×，含历史/六帧/GT/指标/IPC；初始化/首窗warm单列，只暖各worker首窗。不能称真实Waymo精度、L40S倍率或单窗口正式FPS。
CPU相关回归97通过/3CUDA跳过；真实CUDA/native/双进程/Graph/Surface/Strong/旧递推回归147通过；Bash语法/CLI help/diff检查通过，不称全仓CI。含实际双进程概率/六帧/整数gate、source只读迁移、speed-only不推进计数、保存权重路径resume及拒绝更改参数。
入口`run_p0_f9_joint_surface_waymo_10hz_parallel.sh`；文档`docs/WAYMO_10HZ_PARALLEL_EXECUTION_CN.md`。下一步显式`WAYMO10_PARALLEL_CONTINUE_FROM=.../waymo10_fast_20261010_040604` + `WAYMO10_PARALLEL_SPEED_ONLY=1`，先新目录短程比单/双进程。原前缀取真实保存cursor；用户发回`WAYMO10_PARALLEL_PAIRED_SPEED`后才判断是否值得完整resume；未代跑服务器或重启评估。
并行progress打印吞吐seconds/window，worker_seconds及stage重叠不相加；平均128/256新窗口才判断稳定吞吐。独立单模型Dense Forecast FPS口径/论文方法不变。

### 服务器双进程已验收与同v2布局探针（2026-10-10）

服务器`waymo10_parallel_20261010_050832`短测单进程0.3506→双进程0.1701s/窗、2.062×，SIX/概率/整数gate通过；随后恢复原计数，最新5088/39987、累计吞吐0.235s/窗。剩余约2小时17分只是按当前负载外推，worker0.486–0.621s不是总吞吐。不拿16窗和全集累计不同人口算退化。
用户授权比较4进程×1线程；新增独立`run_p0_f9_waymo10_worker_layout_probe.sh`，原并行实现/指纹一个文件都未改。新入口只读停止目录完整契约/实际cursor，在整个探针期间持有原kernel lease阻止并发resume。相同v2、相同64连续窗、AB/BA两遍，首窗warm与全六帧/概率/运动/整数gate计时外，每遍重置历史LRU，只写新receipt/speed/summary，不写科学state、不自动全评。两布局共6个worker常驻但一次只算一组，idle模型显存需公开，不把评估吞吐称FPS。
4×1建议门槛为平均至少快10%且每遍快；不满足保留2×2。若换布局，新目录显式迁移原保存整数前缀；普通旧目录resume仍严格2×2。文档`docs/WAYMO_10HZ_WORKER_LAYOUT_PROBE_CN.md`。
本地CPU相关47通过/2CUDA跳过；真实RTX3050 CUDA相关30通过，含实际2/4进程Graph/概率/六帧/计数一致、旧接续、源目录并发锁、只测不推进计数；CLI help/Bash语法通过。不是服务器布局提速证据，不称全仓CI。下一步服务器安全暂停后跑探针，发回最终summary，不动权重/阈值/训练/正式FPS/其他Camera任务。

## 新增：冻结6秒几何接续四路对照（2026-10-09）

用户已授权尝试提高6秒精度；保留并行 Waymo 工作和旧6秒入口，单独新增 `run_p0_f9_surface_geometry_carry.sh`。
真实 GenieDrive 公开代码人口2569/150已跑完：4/5/6s mIoU30.572013/27.095679/24.488694、平均27.385462；IoU42.032229/38.331931/35.324381、平均38.562847；2330.31s，不是FPS。
既有接续匹配31783/33451来源，不能再把身份匹配率低当作主要原因；论文Table2实际人口仍未独立确认。

新实验一趟 TRAIN64+dev64 跑 baseline/static_carry/se2_carry/combined，冻结同均值、阈值和原第二段 motion forward。
static_carry 原始t0静态直接投影到第二段，保留首段CCR静态新增；动态foreground/重叠动态fallback不变。
se2_carry 只将过去预测形状按首段预测中心/yaw对齐到3s，可靠唯一身份才替换ICP；split/merge/错语义/4m质心异常回退。
这是改变第二段输出的推理算法候选，不是无损后端；相对baseline可能删除/重标静态背景，单独报告corrected/damaged/removed。
四张初始真实历史之外不新增观测；保留causal几何状态必须公开说明stateful rollout；未来GT ego poses仍显式条件。

仅TRAIN64固定规则选择：avg4–6 mIoU+0.05pp、avgIoU/Micro非负、4/5/6mIoU分别非负；DEV64全报告不选路线。
all需要同权重/代码、内容指纹完整的screen recipe，仅冻结候选+baseline官方2569人口，默认不自动启动。
第一段共享且不变；未来GT occupancy/Moving仅全部候选预测之后读取。每8完整窗口整数状态，Ctrl-C/SIGTERM边界保存，严格同目录resume。
native fused projection/graph+4线程无损多数投票；现有VAL几何只读，不写大缓存，原权重/训练/cache namespace不改。
138项相关本地回归通过、20项CUDA跳过，包含真实CPU Surface四路参考字节一致、动态保护/歧义回退、CLI snapshot只读/中断恢复、全集入口和旧long/projection/graph/FPS回归。
CLI help、Bash语法通过；真实服务器候选精度及本次CUDA路径尚未跑，不宣称提分/正式FPS/全仓CI。
操作与比较边界见 `docs/SURFACE_LONG_GEOMETRY_CARRY_CN.md`。下一步服务器跑screen并发summary，无需重建prototype/重训。

服务器首次screen在TRAIN首窗口 four-history input4(KTA)差1.9967556e-6退出，未完成窗口。
本次补旧缓存升级算术兼容：仅KTA严格等于 `float64(FP32 anchor)-float64(FP32 normalized xy)*40` 后转FP32时接受。
诊断副本使用实时KTA，实际prep/record输入、权重不改；其他输入原严检、六帧Transport及Surface byte gate继续。
本地真实Surface大坐标fixture复现微米级缓存舍入；误改KTA/类别/mask拒绝。扩大相关回归142项通过、20项CUDA跳过。
代码指纹改变，原失败目录保留，新输出重跑，不使用旧目录resume，不重建缓存或训练。

## 用户选择静态接续并授权全集复测（2026-10-09）

真实四路screen：static_carry TRAIN64 avg4–6 mIoU+1.469280pp/IoU+2.499484pp/Micro-0.025451pp；
dev64 +1.553857/+2.611150/-0.022158pp。combined略差，SE2无益；原TRAIN Moving非负规则未通过。
用户明确接受小幅Moving下降，固定static_carry，要求跑同官方人口全集并只报告IoU/mIoU。
新增显式 `SURFACE_CARRY_APPROVED_ROUTE=static_carry` all入口：记录用户选择，不改旧screen/门槛，selected_train_route仍null。
同冻结均值/阈值/官方metadata，仅baseline+static，第一段共享不改；不训练、不用未来标签预测。
all默认metrics-only，不读/计算Moving支持，结果标记未评估/null，摘要报告1–6s及1–3/4–6平均的IoU/mIoU。
不计算未用SE2分支；每时距dense整数指标只扫描一次复用dataset/scene。旧默认screen原样，严格同目录resume。
本地专项和旧Surface/两段递推/官方人口/Strong/FPS回归130项通过、20项CUDA跳过；CLI help/Bash语法/diff检查通过。
含metrics-only省略Moving标签访问、整数中断恢复、源screen/权重只读、固定静态路线逐字节一致；不冒充全仓CI/真实GPU或全集收益。
本地不具备完整服务器数据，未代跑或下载完整评估产物；服务器结果现已收到，见下节。

## 最终6秒论文指标已由用户确认（2026-10-09）

用户粘贴完整all摘要：2569窗口/150场景，static_carry 4/5/6s IoU44.768673/41.083833/37.801952，
mIoU32.350774/28.876602/26.126216；avg4–6 IoU41.218153 / mIoU29.117864。
相对同人口baseline avg4–6增益+2.655306 IoU / +1.732402 mIoU pp；1–3s逐项不变。
用户明确要求记录并将static_carry作为最终论文长时预测结果，baseline保留消融，不再加入SE2/combined。
原full4369主表IoU55.211054/mIoU44.153241不变，不拿2569长人口1–3s均值替换。
未来GT ego poses显式条件，保留历史静态几何的stateful causal rollout；不输入未来occupancy/mask/annotation。
这是推理方法改动，不是无损执行；没有6s重训/阈值校准。全人口Moving/FPS未测，不借用screen或6帧FPS值。
按GenieDrive公开代码metadata人口对齐；论文Table2实际起点集合未独立确认，不直接声称严格SOTA。
档案为用户服务器摘要转录，非本地原evaluation.json；实际运行目录/HEAD/snapshot SHA未收到，不猜路径或指纹。
此次仅文档/指标归档，不改变模型、训练、旧默认入口或其他Waymo改动。

## Camera / Pred ego 当前决定（2026-10-10）

用户只做 STCOcc Camera，不做 BEVStereo。按 I²-World-STC 官方历史完整 semantics + 无 camera/lidar metric mask，不额外借用 GT 历史 visibility，不进行 mask 选优。需要的 known 是预测网格有效性，不是真实 sensor visibility。
服务器旧 `new_code/cache/come_main_table/camera_pred` 与 `camera_gt` 各4219窗/150场景，四历史及六 `future_e2g` 保留；原 BEVStereo/planner 路径已失效。仅考虑复用严格对齐的 camera_pred 预测世界 ego pose，绝不复用 BEVStereo 历史/latent/搬运结果；六未来 XY 世界坐标、yaw 相对 t0 非累积。四设置使用共同人口并同人口重算 Occ+GT，不混旧4369分数。
STC包扫描尚未确认存在，官方 I²-World 下载链接已核实；现已实现官方ZIP下载检查、逐值无损uint8紧凑解压、冻结Surface四设置 evaluator。四设置同planner-covered人口（all预计4219），一趟完整六帧后才读未来GT语义，报告1/2/3s与均值标准IoU/mIoU并附官方排除零类诊断，不混旧4369数字。
严格身份/order/tag、当前z/tilt与独立yaw、文件/pose指纹及固定mean检查；Pred全部未来几何用预测pose，无GT回退/事后对齐；Camera无GT visibility/cache复用。每8完整窗口整数保存，Ctrl-C/SIGTERM安全停、同输出目录严格resume，旧训练与缓存只读。
本地STC+Waymo+geometry-carry相关回归63 passed/4 CUDA skipped；实际小模型NumPy/native四设置概率/六帧exactness、无future GT读入、错序/缺失fail-closed、compact逐值相等和中断不重计通过。CLI help/Bash语法通过，不称全仓CI/真实STC精度。
未下载完整STC数据、未训练/代跑服务器。入口 `run_p0_f9_joint_surface_stc.sh`；顺序命令见 `docs/SURFACE_STC_CAMERA_PRED_EGO_CN.md`，调研见 `docs/CAMERA_PRED_EGO_RESEARCH_20261010_CN.md`。

## STC四设置执行提速（2026-10-10）

用户服务器421–484窗四设置1.7103s/窗；准备0.8247s、证据/投影0.7111s、head0.0406s。
新增独立 `run_p0_f9_joint_surface_stc_shared.sh`：复用原生CPU warp/Surface fit；
同一实际历史GT/Pred共享history-only tube/运动与canonical Surface证据，单bundle、内容/pose/visibility key。
Strong/未来变换/owner/fallback/phase/prob/dense各自重算；不训练/改阈值，不跨模型/历史复用learned输出。
初次8相同窗口四路六帧逐字节检查+AB/BA实测，>1%提速才接着剩余窗口；不能预告真实L40S倍数。
显式只读旧STC状态桥接到新输出，源/新租约全程持有、契约与integer counts校验，原入口/源码/旧结果不改。
本地CPU小模型byte与整数恢复、不同历史失效、科学变更拒绝、CLI源只读/严格恢复、租约及篡改检查通过。
相关回归69 passed/7 CUDA skipped；含新终端launcher原SWFM环境安全恢复，无shell eval。
真实CUDA/STC数据速度与质量仍待服务器内建检查；详见 `docs/SURFACE_STC_SHARED_EXECUTION_CN.md`。

## 冻结 Pred ego / STC 因果适配集中对照（2026-10-10）

用户明确不做 aligned 修分；优先查 Pred 几何误差与 STC 时序噪声，不改网络、不重训。
最新约千窗前缀四设置平均 mIoU：OccGT42.723579 / OccPred21.645784 / STCGT20.107599 / STCPred14.205545；未完成全集，不能混早期64窗数字。
新增独立 `run_p0_f9_stc_causal_geometry.sh`，只跑固定dev64：四基线、各自历史路面z/tilt补偿、STC地面列稳定及组合，共九路。
路面补偿保留planner世界XY/yaw；历史区域不足、查询在内点凸包外、>0.8m高度或>3度倾斜修正均回退。不修复真实XY/yaw规划误差，不使用未来GT位姿对齐。
地面稳定只改历史一致支持的class11/12/13/14既有1–3格地面列，最大1格；不把缺失当free、不新增列、动态/其他占据逐字节保护，history masks不变。
动态组件速度暂不覆盖，只报告同类因果匹配/初速/质心抖动，避免与网络residual冲突；不宣称已解决动态噪声或能恢复二十点落差。
同mean5/6/8/12/14、ADD0.5/REMOVEoff、4→6、无mask/阈值/epoch搜索；GT-conditioned基线仍仅按原协议读GT ego。
全部九路预测后才读未来语义与审计GT；每8完整窗口整数保存、同目录严格resume，新终端恢复本次SWFM flags。旧四设置入口/数据/cache/权重/训练/前缀不改。
摘要一趟输出九路1/2/3s及均值IoU/mIoU、位姿误差分解、STC t0质量、修正支持/拒绝和动态抖动；详细契约/原始计数保留。
本地相关回归109 passed/2 CUDA skipped，含真实CPU九路推理、原四基线六帧字节一致、权重不变、GT读取边界、地面向量化参考一致、鲁棒平面拒绝、CLI只读/中断恢复；Bash语法/help通过。
操作与限制见 `docs/STC_CAUSAL_GEOMETRY_SCREEN_CN.md`。本地CPU回归不代表真实服务器候选增益；尚未跑L40S/dev64，不自动扩大到4219全集。

## 最新：失败适配后的单次 yaw 一致性 screen（2026-10-10）

服务器九路dev64已完成：历史路面z/tilt OccPred mIoU−1.212540pp、STCPred−0.403252pp；STC时序地面Pred−0.306691pp、组合−0.644499pp，均失败。156.23s/64包含九路，不是单网络FPS变慢。基线OccPred18.350472/28.307956、STCPred11.960050/21.202278（mIoU/IoU），不能与此前约千窗混拼。
用户授权再试一次无训练方案：新增独立 `run_p0_f9_stc_trajectory_yaw.sh`，固定dev64 OccPred/STCPred原版与候选共四路；原网络、mean、XY/z/body tilt/阈值不变，可靠历史heading校准预测XY切线yaw，低速/倒车/不稳定/大修正原pose回退。
仅四历史pose/timestamp与六预测pose进入适配；全部预测完成后才读未来GT用于审计。没有aligned、GT mask、未来真值校正，不重复失败ground分支。不改旧STC实现指纹/结果，单独整数状态/严格resume；候选全pose不变复用原预测，首次真实改yaw仍验完整六帧/概率。
新规则是待验证因果推理候选，不是COME已有方法或官方原planner reproduction；不保证收益、不自动全集或调参。无清晰增益就停止此路线。详见 `docs/STC_TRAJECTORY_YAW_SCREEN_CN.md`；下一步只需服务器运行新入口、发回summary。
本地新yaw+旧STC协议/共享执行/失败geometry回归57 passed、1 CUDA skipped；实际CPU小模型原六帧/概率byte parity及权重不变、未来GT隔离、首次延后真实修正仍exactness、严格整数续评/CLI输出保护通过，Bash语法/help/diff通过。不声称全仓CI、CUDA或真实dev64收益；windows-python-env-guard仅使用已验证项目解释器。

## 最新：四设置完整结果后的单次分支归因（2026-10-10）

用户 full4219/150 场景四设置均值 mIoU/IoU：OccGT44.105025/55.113841、
OccPred22.418912/30.061409、STCGT18.231871/27.683236、STCPred12.900742/21.671181。
同 planner-covered 人口、full-grid、原 mean；不混旧4369或早期约千窗前缀。
之前路面/时序适配失败；yaw screen OccPred mIoU+0.183663但IoU−0.334617、
STCPred+0.023148/−0.063691，无足够增益，均不采用/不重启。
用户最新授权一次集中诊断，不换网络、不重训、不aligned修分：新增独立
`run_p0_f9_stc_branch_diagnostic.sh`，固定dev64四设置每窗四次forward，
直接取同次Strong anchors / Transport baseline / Joint六帧，报告12路指标、类组FP/FN、
CCR改对/损坏、t0感知质量、pose误差、因果track统计及GT/Pred motion字节一致。
未来真值只在所有预测后用于评分/审计；原始planner JSON有则核查原ordinal/row0/非累积yaw重建，
缺失明确未核验，不把cache声明当独立证据，不修改/纠正planner或历史。
另建输出/契约/整数窗口状态，支持严格同目录resume，旧四设置源码及full结果指纹不变。
操作与统计边界见 `docs/STC_BRANCH_DIAGNOSTIC_CN.md`；服务器只需运行此入口发回summary。
本地新专项+既有STC协议/共享执行/geometry/yaw回归70 passed、1 CUDA skipped；
包含真实CPU小模型原概率/六帧字节一致、权重不变、GT读取边界、原JSON非累积yaw重建、
错误整窗原子提交及整数/身份/order续评；CLI help、Bash语法通过。
没有真实服务器输入，不宣称已找到掉分的量化主因或真实增益；未改旧STC实现指纹。

服务器首次分支诊断在 motion fingerprint 直接 `.numpy()` BF16 时报不支持dtype；
这是新诊断兼容性问题，不是模型预测/数据报错，尚未发布完整首窗计数。
已改 detach/CPU/contiguous/flatten 后 uint8 reinterpret，哈希包含原dtype/shape/bytes，
不转FP32、不修改模型/原推理路径。追加BF16/FP16/FP32/FP64/int/bool、标量/空/非连续、
signed-zero/NaN位模式及四设置BF16读出回归；相关90 passed、1 CUDA skipped。
代码指纹已变，保留失败目录，使用新STC_DIAG_OUT重跑，不迁移旧契约或重建输入缓存。

## 原始 planner JSON 的 t0 起点审计（2026-10-10）

服务器dev64分支诊断：Strong→Transport→Joint均值mIoU：OccGT38.616→40.848→41.896，
OccPred17.477→17.834→18.350，STCGT15.970→16.770→16.995，STCPred10.930→11.825→11.960。
各设置CCR增益正，但主要掉分在Strong之前。STC t0占据precision51.542%/recall47.295%；
Pred 3s XY median/P90=1.237/2.371m，yaw=2.247/12.717°；z误差0，不继续把z当主要掉分来源。
GT/Pred motion bytes一致64窗；原始JSON当时缺失，起点审计未核验，不据缓存声明判定无错位。

本地父目录`_received/bevplanner_ego_in_bev_with_yaw.json`已找回：2,651,124 bytes、
6019行/150场景/每行7×3；官方SHA与旧cache声明一致（19c04eaf…6783afb）。
数字suffix是排序索引，非场景内零基ordinal。旧builder/COME皆按scene排序后用当前帧ordinal取行，
但原构建器没有独立核对row0 XY。不能仅从六缓存位姿可重建宣称t0正确。

新增独立`run_p0_f9_original_planner_t0_audit.sh`：只读全部4219缓存identity/6位姿、
完整场景row0及nuScenes ego/identity元数据，不需STC/GT occupancy/模型/GPU。
行排序/全链→t0 row0 XY（2cm）及邻行诊断→非累积yaw重建（1e-6）三重核验；
停车/重访XY显式标兼容有歧义，JSON无sample token，不冒充唯一身份证明。
严格官方SHA，重复key/suffix/场景行数/缓存order拒绝；源文件读前后核验，报告新目录，
不自动重索引、改pose或aligned修分；旧四设置/shared/branch/protocol实现指纹保持不变。
本地相关回归112 passed / 1 CUDA skipped（新专项22项），Bash/CLI help通过；
最初旧native默认缓存权限失败，显式复用tmp验证缓存后已通过，不是诊断实现故障。
实际服务器t0核验已完成：4219/4219缓存pose重建通过；3600 unique-XY、619 compatible-ambiguous。
完整6019行origin均兼容，最大XY误差0.000171m，没有row-shift证据；停车/重访不冒充唯一身份。
不重启失败的无训练适配。详情 `docs/ORIGINAL_PLANNER_T0_AUDIT_CN.md`。

## 冻结共享历史特征的内部ego头（2026-10-10）

用户已授权实现；本轮不自动启动服务器训练。现有mean5/6/8/12/14、GT版本、原四设置和原代码路径不改。
新增独立0.43M ego头：V18历史物体context（最多64，只限头池化）、CCR历史encode的8×8静态空间token、
四帧实际历史ego运动→6时间query×right/left/straight→相对t0绝对XY/yaw；不是物体WM残差。
只训头，XY Smooth-L1+周期yaw，特征先于未来投影生成；训练bank明确绑定冻结权重/历史数据/代码指纹。
输出pose统一进入Strong/搬运/CCR投影及phase；保留t0 z/tilt，无future GT occupancy/mask/CANbus输入或aligned评分。
导航采用I² converter的LiDAR末端x±2m规则、GAST未来destination行；显式GT派生navigation-conditioned，不称无导航预测。
新入口 `run_p0_f9_surface_ego_head.sh` 默认TRAIN1024头20轮余弦 + 一次固定dev64六路（OCC/STC×GT/external/internal）。
最终轮头、不选dev-best、不自动重试/扩训；bank、Adam/LR/RNG/游标/整窗计数严格续训续评，输出互斥lease。
本地全量非集成验收1789 passed / 75 skipped / 4 deselected；CUDA专项本地跳过，不是实际L40S精度/FPS结论。
文档 `docs/SURFACE_FROZEN_EGO_HEAD_CN.md`。

## ego头失败后的单次预算/几何监督对照（2026-10-10）

原头服务器已完成：TRAIN1024、batch64、20轮共320步，330.11s；WM/CCR未更新。
固定dev64：Occ外部/内部mIoU18.350472/16.760100、STC11.960050/10.928355，内部头不采用。
OCC内部1/2/3s平均XY误差0.565/1.668/2.875m，明显逊于外部0.339/0.611/1.341m；
末batch损失不证明收敛，320步不足与yaw尺度问题都是待检假说，不据此保证扩训会成功。
原来源 `outputs/p0_f9_joint_surface_ccr/ego_head_screen_20261010_194309_837`。

用户授权同bank单次对照：历史外推prior/已有old320/原目标A2000步/几何目标B2000步；只训头，不重提特征或重训WM/CCR。
A/B同原初始化（原seed+冻结模型构造，CPU RNG见证严格核验）、逐步同样本/Adam/余弦/clip；batch64约125轮。
B单项五个固定中心/±10m轴参考点的SE(2)坐标Smooth-L1，无GT形状/体素反传；不是再加多个loss。
整轮加权均值、每200步完整TRAIN1024误差与cmd计数；TRAIN是IN-SAMPLE。最终一趟dev64共12路：OCC/STC×GT/external/prior/old320/A/B。
只在原代码完整匹配时排除新增三份工具文件复原旧manifest，逐shard/SHA核验只读bank；新实验锁定当前完整指纹。
新成对断点存两头Adam/游标/RNG/均值累积；完成训练resume不改权重SHA；dev整窗原子续评，无aligned/阈值选优/自动推广。
入口 `run_p0_f9_surface_ego_ablation.sh`，最终`_eval_dev64/summary.txt`含训练与评估，不启动服务器。
本地专项及相关回归156 passed / 9 skipped；含实际小网格冻结WM/CCR的CPU几何路径、成对Adam/RNG续训、整窗续评与只读保护。
两个CLI帮助及Bash语法通过；CUDA专项本地跳过。文档 `docs/SURFACE_EGO_BUDGET_GEOMETRY_ABLATION_CN.md`。

## A/B服务器失败结果与单次TRAIN接口复核（2026-10-10）

真实A/B完成：各2000步/125轮，同TRAIN1024。原/A/B的TRAIN 3s XY2.285891/0.257947/0.292980m，
dev64 OCC 2.875380/4.087419/3.167395m；外部planner1.341432m。B的TRAIN/dev yaw3s0.880/8.747deg。
OCC mIoU external/old320/A/B18.350472/16.760100/12.844141/13.600391；STC11.960050/10.928355/9.243519/10.042540。
预算扩大只改善IN-SAMPLE，泛化变差；不采用新头，不自动继续扩训。原四设置/GT权重不变。
成对源目录 `ego_ablation_20261010_214909_837`，已有评估同名`_eval_dev64`。

用户授权只读同TRAIN64接口诊断：缓存 vs TRAIN式实时提取 vs dev OCC式实时提取，
逐字段/标签/导航核验；batch64 vs 单窗、train/eval no_grad前向；完整TRAIN1024报告复现并引用现有dev64报告。
固定导航/scene均衡64窗来自原bank，不以误差筛选，不称代表性总体；不新训练、不跑稠密预测、不读未来occupancy。
原Torch/CUDA/精度/SWFM环境与旧实现必须匹配；源bank/元数据/历史NPZ/权重SHA核验，新输出支持完整窗口resume。
新Python独立放 `tools/ego_diagnostics/surface_train_replay.py`，不影响旧bank绑定的顶层工具/real_motion指纹。
入口 `run_p0_f9_surface_ego_train_replay.sh`，发新输出`summary.txt`即可；门槛固定，不自动修缓存/推广头。
本地专项及相关回归166 passed / 4 skipped（CUDA），尚未跑真实服务器TRAIN64，不能据合成测试宣布已排除全部接口问题。
文档 `docs/SURFACE_EGO_TRAIN_REPLAY_CN.md`。

## ego 头扩大 TRAIN 的单轮试训（2026-10-10）

服务器TRAIN接口复核完成：bank/TRAIN实时/eval实时的七字段64窗逐字节一致，标签/cmd/train-eval一致，完整TRAIN1024报告复现。
原batch64/单窗严预算仍失败：maxXY2.09mm、yaw0.010deg，原失败保留；不能解释米级TRAIN/dev差距，不据此保证扩训成功。
用户授权扩大规模，随后明确只训一轮；不执行原先拟议20轮，不自动续训。
新 `tools/ego_experiments/` 独立入口覆盖完整TRAIN20430身份人口，固定SHA256(seed:scene)留出10%场景，其余fit窗口只优化一次。
WM已见过这些TRAIN留出场景，明确仅ego优化器场景不重叠。原约0.43M头/同原初始化/单项R10m几何，batch64单轮余弦；WM/CCR完全冻结。
只读复用旧1024 bank，其余历史特征一次提取；有界4线程/原帧及纯几何各512MiB，GPU一次打包，不运行未来Strong，不写旧缓存。
固定第1轮 `head_epoch1.pt`（不按留出/dev挑best），TRAIN留出仅诊断；最终固定dev64六路OCC/STC×GT/external/internal。
`last.pt` 含Adam/余弦/游标/RNG/整轮均值；bank逐shard/评估整窗恢复；完成resume不再训练、不重写导出SHA。
Python不放旧bank哈希覆盖的顶层目录，不改变旧head/提取/协议。入口 `run_p0_f9_surface_ego_full.sh`，输出新目录，恢复须同一EGO_FULL_OUT。
本地180 passed / 5 CUDA skipped，含实际CPU冻结WM历史bank、原文件保护、部分恢复/破坏拒绝；未启动服务器、没有真实扩大训练指标。
只需发 `_eval_dev64/summary.txt`，训练/构建耗时分开；文档 `docs/SURFACE_EGO_EXPANDED_ONE_EPOCH_CN.md`。

用户服务器已建到7168/20430，尚未开始ego优化；最新要求先用半量试训，有效才处理余量，已同意Ctrl-C安全暂停。
新增独立 `screen_surface_ego_partial.py` / `run_p0_f9_surface_ego_partial.sh`，原full三份指纹文件完全不改。
只读连续已完成bank前缀、最多10240，少于此数量则直接用现有，不补特征、不复制大bank；原全量场景fit/holdout身份保持。
一轮原初始化/R10m/batch64余弦，固定第1轮导出，再复用原dev64六路 evaluator。前缀按scene排序，不称代表性完整TRAIN。
源/新目录lease隔离、原契约/receipt/权重只读，pilot恢复人口不随源bank增长；完成不自动恢复全量。
原full可继续复用所有bank，但pilot头/Adam不是full断点；不静默改变原一轮的总步数。入口唯一父任务自动识别，多任务须显式EGO_PARTIAL_PARENT。
专项partial+full本地22 passed / 1 CUDA skipped；含active-parent拒绝、无新提取、来源字节不变、部分训练恢复、父bank增长而人口冻结，以及原eval导出兼容。
加入历史ego/STC/planner相关回归后188 passed / 5 CUDA skipped；真实服务器半量训练尚未运行，未承诺效果。

最新半量服务器已完成：10129窗/fit8998/holdout1131，1轮141步，读取16.76s、训练6.47s；留出prior/learned 3s XY2.268/2.242m。
dev64 Occ内部/外部mIoU18.861/18.350但IoU26.824/28.308；STC11.133/11.960，内部头尚不采用。
用户明确要求相同缓存训3轮并评dev512，且快速实现、不做大量额外验证。
新 `train_surface_ego_three.py` / `eval_surface_ego_three.py` / `run_p0_f9_surface_ego_three.sh`，旧full/partial绑定代码完全不改。
只读上述completed pilot固定receipt/population/split，从同随机初始化训完整3轮余弦（423步），WM/CCR冻结，无新增提取；固定epoch3。
复用旧六路评估engine，选manifest完整parent512 keys并严格512窗；无aligned/mask/阈值选优。
新输出 `_three_epochs` + `_eval_dev512`，支持同目录EGO_THREE_RESUME=1，源bank/原一轮结果不写。
仅3项针对性CPU测试通过（3轮最终导出/精确Adam-RNG恢复/来源不改/512人口），CLI/Bash通过；未代跑服务器或声称效果。

2026-10-11服务器3轮423步已完成，TRAIN holdout FDE2.219904（prior2.267768），WM/CCR未更新。
dev评估被我新增的硬512检查拒绝，尚未预测/写评估目录；日志没有parent/覆盖数，具体原因未可区分。
新增独立 `eval_surface_ego_three_population.py` / `run_p0_f9_surface_ego_three_eval.sh`，只评既有head_epoch3，不跑fit/重缓存。
保留原three绑定3文件字节不变以避免checkpoint SHA拒绝；修复evaluator自身SHA纳入新eval契约。
选manifest已有selected512，否则parent>=512按固定round-robin请求512；先固定名单再取planner交集，不补样本。
公开parent/requested/covered/missing和selection_rule，缺行则标dev512_planner_covered_intersection；parent不足512仍失败。
5项人口小测试通过，CLI/Bash通过，原three三文件无diff；当前只需服务器运行新增eval-only入口，发summary。

## 内部ego三轮结果与运动学控制集中试验（2026-10-11）

最新服务器完成：原parent512取planner交集495、missing17，所有六设置同人口。
内部/外部平均mIoU：OCC18.040683/19.990703，STC11.293439/12.210279；
平均IoU26.548403/28.578451及21.598709/22.622076。OCC 3s XY2.465m，外部1.384m。
旧内部头不替换planner；不继续自动加轮数，正式WM/CCR/GT/Waymo/6s指标不变。

用户授权认真改内部头、先本地模拟后给服务器集中试验。新增仅在tools/ego_experiments：
history_state从三历史secant拟合瞬时速度/一致加速度/角速度；小MLP读同七字段/六导航，
输出三个accel/yaw-rate knot并统一积分六XY/yaw；历史8710参数，场景attention版34502。
仍固定R10m单项loss，WM/CCR冻结，无新CANbus/地图/未来连续GT/visibility/aligned。
导航同原GT派生destination命令，不能称navigation-free。静止启动使用ego+X，不混LiDAR导航轴。

本地独立NumPy模拟、固定两seed、各2048train/512same/512shift、各3轮96步：
原CV/旧Transformer/纯运动学/历史控制/场景控制 FDEsame约3.737/3.510/1.103/1.113/1.105m，
shift约3.915/3.655/1.355/1.350/1.348m。收益主要是运动学初值，不是神经头已超过prior；
未来不可观测突变仍5m级，匀速带噪声仍会小幅退化，不保证真实planner收益。
初始无约束加速度拟合放大噪声，故增加相邻趋势一致保护；全部原始模拟试次/负对照保留。

服务器入口 `bash tools/ego_experiments/run_ego_kinematic.sh`：复用已有三轮来源和10129窗bank，
不补缓存；同fit8998/holdout1131，两头各3轮423步整周期余弦，固定最终，两头均报告。
默认一次DEV64共12路：OCC/STC×GT/external/old_epoch3/kinematic_prior/control_history/control_scene。
自动同目录两Adam/RNG训练恢复和整窗整数续评；新代码单独绑定，不改旧指纹/权重/银行。
新输出原three_epochs目录同级后缀_kinematic，再_eval_dev64；用户显式可改dev512取planner交集。
本地31项CPU验收通过（含实际小网格12路线、输入未来隔离、两头精确恢复、旧三轮/partial/人口），
不宣称实际CUDA/L40S、真实nuScenes涨点或完整CI。详细模拟限制/命令见 `docs/EGO_KINEMATIC_CONTROL_SCREEN_CN.md`。
本轮未代跑服务器、没有自动推广/扩训；下一步只需上述集中screen摘要。

## 仅按需检索的历史

冻结 Surface CCR 完整 TRAIN×3 DEV512 mIoU40.418031/Micro31.052754；扩大 VAL4369 mIoU43.991199/Micro32.043717。
Frozen B 与旧 Local 的静态差距集中 road11/sidewalk13；Surface 内部几何条件读出已追回大部分冻结静态差距。
这些是早先冻结实验，不是新联合精度。旧阈值、halo、共享场、点 CCR 支线不重启。
详细历史仍保留在 `docs/CLEAN_JOINT_SURFACE_CCR_CN.md`、`docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`，需要时检索，不复制旧日志。
