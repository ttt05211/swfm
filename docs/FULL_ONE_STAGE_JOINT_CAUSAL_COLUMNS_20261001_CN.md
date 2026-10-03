# 全量一阶段：15 / 20 轮统一余弦训练

## 目的与最终定义

用户明确要求：**从头在完整训练集训练一阶段网络，15 或20轮，整个周期统一余弦下降，不采用“10轮余弦＋最低LR tail”**。原 E14、冻结 V18+columns 成功模型、小规模 screen 均保留。

模型结构不改：原 V18时空 encoder/query decoder + 原XY/yaw/existence heads + source-linked shared column encoder/decoder。只有硬几何/ownership/candidate索引 stop-gradient，动态 refine 的 continuous source query 不 detach。所有 source 始终来自因果历史；GT只用于监督、校准或指标。

新的 `p0_f9_joint_causal_columns_full_train_v1` 与之前screen协议分开；不能拿失败screen的last/candidate作为全量初始化。默认全部随机初始化，E14只提供architecture config及冻结评估参考。

## 全量 population、合批及学习率

- 每轮 **ALL20430** 个训练窗口，每个窗口正好一次。不把32个calibration scenes从优化集拿走，不按前1024条替代全量。
- 独立、确定性的每轮window shuffle；窗口内所有source保留。
- 默认最多4窗口、128source组成一次optimizer update，单窗口source超预算时单独保留全部source，不截断。实际每轮更新次数由source数决定，不等于20430，也不固定等于ceil(20430/4)。启动时打印精确计划。
- V18将batch内source合并做真正的forward，之后按原窗口/source顺序切回live latents；每窗最多256个在线column，四窗口最多1024列，共享一次column forward。original motion loss按batch全部有效source/horizon归一化，column loss按类型及importance weights归一化。不是每个窗口各更新一次。
- 原损失/权重保留：XY SmoothL1 + existence BCE +19×yaw +.25×SE(2) overlap；column生成BCE/refine action CE。motion/column gradient clip分别5/1。
- AdamW：motion初始5e-4、WD1e-4，column初始3e-4、WD.01。
- 总步数 `T` 为声明的15/20轮所有planned batches之和。第s步使用 `0.1 + 0.9 × (1+cos(pi*s/T))/2`；最后约到初始LR的0.1（motion5e-5、column3e-5）。**没有第10轮切换或最低LR续训阶段**。
- 必须启动时选定15或20轮；两者的整个LR曲线不同。断点恢复不能改总轮数/batch/seed/数据/配置，15轮跑完也不能直接把last.pt接成20轮统一余弦方案。
- 默认不同时训练第二套V18-only对照以节省长训时间；E14作为评估reference。若确实需要同预算公平control，用 `FULL_JOINT_PAIRED_CONTROL=1`，同一批全部source、相同原运动损失和同一个全程schedule训练独立V18-only参数。

## 速度和空间

新增 **一整批NEXT raw/history evidence的CPU prefetch**，可与主线程当批的renderer、online sampling和GPU训练重叠。只预取下一批，不提前装下整个epoch。

预计算包含：因果history-source association/ICP、历史footprint、static memory及静态frontier。worker不调用model/CUDA、不读取GT构建证据；current source与记录的类别/中心及prefetched source逐元素核验。Strong保持主线程原GPU路径，完成后只将其CPU数组保存供后续复用。

以下内容永不跨更新缓存：learned poses、owner/fallback、完整候选、action labels、learned features。运动更新后全部在线重建，不出现旧bank错配。来源帧RAM缓存默认TRAIN/dev各256MiB，保持原精确inverse-map采样；持久几何采用下文新增的有界压缩cache，不构造learned dense feature bank。

新日志包含windows/sources/query数、prepare与sampling耗时、input wait、LR及CUDA peak memory。每32个batch显示滚动实测seconds/window和remaining training hours；ETA只涵盖训练，不假称包括所有评估/初始化。

不能以小规模原循环0.84s/window承诺新的全量耗时。未优化循环线性放大15轮约70小时，本版新增合批/prefetch/移除默认control，实际时间应以服务器前几十batch为准。不是缩减数据/监督来提速。

## CPU采样提速补丁（兼容b700519断点）

服务器实测每批约2.22–2.36s，`online_sampling_seconds`占1.60–1.74s；input wait约0.0005s。优化针对CPU在线候选/历史patch，不将GPU低占用误诊为显存不足。

- 每批最多4个CPU线程：candidate/label按窗口并行，patch映射跨horizon并行；RNG仍在主线程按原窗口/horizon顺序抽样，GT标签与importance weights不变。
- worker只访问NumPy证据；live source feature gather、Torch/CUDA和梯度仍在主线程。GPU motion loss与CPU patch任务重叠。
- frontier多个query读取同一历史anchor时，稀疏映射只计算一次，恢复原query顺序后按各自class设置membership bits。
- registration后的历史世界坐标只在因果prefetch中算一次，六个horizon复用；learned未来位姿仍逐更新应用。
- 保留完整标签/ownership/duplicate检查，用等价整数范围及字节键实现替代较慢的isin/结构化排序；移除单worker sampler的不必要嵌套线程。
- 日志拆分`online_selection_seconds`、`online_feature_wait_seconds`及`online_worker_seconds_sum`（多线程CPU时间之和，不是墙钟）；总速度继续看`seconds/window`。未在服务器实测前不承诺倍数。
- full训练协议、batch、scheduler及模型不变，可从b700519的last恢复，跳过prior1024重计数。以后SIGTERM/SIGINT只设置停止请求，完成当前batch后原子保存last并以130退出，不假报训练完成。

旧b700519进程本身没有新信号处理，使用安全切换器：

```bash
bash tools/real_motion/switch_p0_f9_joint_causal_columns_fast.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_20261001_180940_b700519 975
```

975来自用户此次nvidia-smi，切换前仍需核验：脚本读取/proc命令行，严格匹配full入口/原输出目录/config/cache/info/checkpoint/seed；不是目标则拒绝。脚本最多等待30min内下一次atomic last替换，完成后才TERM旧Python，确认退出后后台启动新版并打印新日志。原epochs/window/source/paired-control配置保持不变；不会删除旧目录或影响其他GPU进程。新旧版本数学计算的synthetic输入、RNG及优化更新一致性已检查，真实CUDA最终数值仍以服务器运行为准。

## 固定几何复用补丁（兼容第1422步断点）

用户确认资源：10核/100GB RAM/48GB显存，训练已安全停止。新profile每批4窗口：prepare主线程约0.68–0.79s、online约0.76s（候选等待0.41–0.61s，feature等待0.10–0.27s），总计1.62–2.12s。GPU低占用与CPU准备吻合；不是单纯把batch加大可解决的问题。

本补丁保持模型、loss、GT标签、采样顺序、source/window budget、梯度路径和整段余弦不变，合并以下计算优化：

- 每窗因果固定几何按需持久化：source extraction/association/ICP、aligned history points、footprint、static memory，以及**主线程GPU原路径完成后**的bit-exact Strong anchor/KTA/CLEAR和无learned replacement背景。不得在cold history worker里改用CPU重算六帧Strong。
- 六个horizon的history-only frontier、nearest anchor、road/sidewalk dominant class与Z support只算一次。完整合法候选仍由**当前**预测baseline/ownership/pose在线生成；不能缓存上一轮的sampled plan或labels。
- 动态source的BEV padding在source-local bbox内做精确dilation，保持原整网格argwhere顺序，不改变padding或裁掉候选。
- 下一批最多两窗并行CPU预备，每窗history几何workers不超过3；当批候选/特征最多4worker，RNG串行。Torch/CUDA及live source queries仅在主线程执行。
- 外层loader完成后才关闭内层pool，避免安全停止时预取线程提交到已关闭pool。

缓存默认位置 `outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1`，所有provenance namespace合计最多**48GiB磁盘**、单实例**4GiB RAM LRU**，写入前保留至少1GiB空闲磁盘；满额/空间不足继续精确重算，不删已有文件。原始输入/batch/optimizer及最多8窗writer queue仍占额外RAM，4GiB不是整个训练进程的内存上限。一个cache根目录只允许一个训练写入进程；同进程worker/实例共享quota核算。

每次lookup对history occupancy/observed/poses及future ego poses做SHA256；namespace额外绑定runtime config、Strong/column config、train/dev cache和info SHA及dataroot。artifact压缩后有内容checksum，损坏或provenance不一致直接报错，不静默用错缓存。只有本机可信cache可加载，不能从不可信来源导入pickle。cache不保存future occupancy GT、record训练标签、Tensor、模型输出或learned geometry。初次真实renderer/Strong exactness检查仍执行。

这是lazy复用：没有独立预构建任务，训练正常进行时填充，后续窗口重访/epoch复用。**冷缓存第一轮仍有计算和写盘开销，不能凭本地unit tests承诺服务器倍数或V18的20min/epoch。** 从第1422步恢复，第一轮已处理过的窗口不会为cache重跑；它们下轮首次访问才填充。日志每32batch打印 `GEOMETRY_CACHE` hits/misses/disk/RAM，以及每batch `causal_geometry_cache_hits` 和worker耗时。看warm-cache后的真实`seconds/window`，不拿一次nvidia-smi快照判定全程吞吐。

可用 `FULL_JOINT_GEOMETRY_CACHE_GIB` / `FULL_JOINT_GEOMETRY_CACHE_RAM_MIB` 修改工程budget，不影响resume scientific identity。零磁盘budget只保留有界RAM复用；Python入口不传`--causal-geometry-cache`则维持原无持久cache路径。更改算法需更新cache protocol，不能复用不同数学实现的旧artifact。

用户此次确认的最新安全断点：

```text
/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full15_resume_fix_20261001_195429/model/last.pt
```

恢复仍设15轮/window<=4/source<=128/paired-control=0，跳过prior，下一更新为1423，不重新训练前1422步。

### 6b9ee9a冷缓存回退修正

真实服务器后续日志：cold hits=0，约0.96–1.02s/window，较此前0.47s/window慢约一倍。代码审计发现6b9ee9a的完整CPU缓存builder把原本CUDA inverse-warp/majority-fill的六帧Strong搬到了CPU，压缩/读盘又占用cache共享锁。这是实现上的性能回退，不是模型或loss导致，不能要求用户只等到下一轮。

修正后：history worker只准备原因果证据和frontier；cold Strong保持**主线程原device**；renderer顺带保留固定background，不重复compose；主线程完成后保存CPU-only prepared state，绝不保存完整PreparedColumns/live输出或GT。最多8个待写条目、单writer异步pickle/zlib/atomic write，队列满则跳过该次写盘，不阻塞训练。读取/解压/压缩均在LRU锁外，关闭时drain有限队列并传播writer错误。实际占用按ndarray backing allocation核算，避免view只按slice计数。

324个文件约493.94MiB，即约1.52MiB/window；按当前样本均值线性推算全集约30.4GiB（估算，不保证所有scene大小相同）。所以default cap从16改为48GiB，仍小于用户允许的100GB存储额度，写盘仍保留1GiB空闲。namespace/文件协议及数学内容不变，已经构建的CPU artifact可直接复用，不删除或重建旧缓存。日志ETA更名为`remaining_train_hours_if_current_speed`，明确按当前吞吐外推，不当作后续warm阶段的已知耗时。

当前cache版已经安装safe SIGTERM handler。新增PID-only切换模式读取完整/proc参数，校验脚本、输出路径、全部data/config/seed/batch后才TERM；完成当前batch保存新的last、退出并观察到checkpoint atomic替换才启动新版。不再等下一个256batch checkpoint；PID消失/身份不符/没有新保存则拒绝自动启动，绝不KILL或删除历史输出：

```bash
bash tools/real_motion/switch_p0_f9_joint_causal_columns_fast.sh --graceful-from-pid 1043
```

1043是此次用户确认的当前PID；执行前脚本仍重新核验，不假定它永远有效。恢复本次运行最新断点，**不是倒退到旧1422 checkpoint**。CPU-only测试验证冷缓存不调用CPU完整Strong builder、Strong仅主线程原device执行一次、warm disk hit不重算、候选/特征及live gradient一致、writer不持有读锁、队列有界且错误不丢失；CUDA实测吞吐仍需服务器正常训练日志确认。

## 校准和评估

- 类别先验只计数一次固定、scene-balanced TRAIN1024的**未采样合法proposals**，然后固定。明确是prior subset，绝不称full20430统计；不是每轮加一次3小时以上全量audit。
- 全量20430都优化，所以最终TRAIN64校准是 **in-sample TRAIN-only**，不是held-out，也不使用dev标签/阈值扫参。报告中 `held_out=False`。
- 每轮固定dev64监控，固定GEN/REF threshold=.5、REMOVE-off，记录current transport/GEN/REF/JOINT及E14比较；只看趋势，不自动early stop、不按dev挑best、不自动扩训。
- 最终声明轮数的checkpoint经过TRAIN64固定grid校准，固定后仅一趟dev512，并同时统计其中的dev64。默认不跑full4369。
- 这回答的是「足够训练后，一阶段是否能超E14、补全是否有效」。默认没训paired control，不能声称额外收益完全来自joint gradient。与以前两阶段的.3269pp比较时应注意训练预算和population/阈值校准差异。
- 只有真实有效编辑、全体分支/各horizon非退化、joint相对E14非退化及link存在等门槛全通过，candidate才允许默认deploy。身份变换/全关闭不叫生成成功，失败不自动retry。

## 保存、恢复

默认每128个完整update原子替换 `last.pt`（`FULL_JOINT_CHECKPOINT_EVERY`可调），每轮结束前（开始eval之前）也保存，eval后再保存epoch cursor/history。先写独立临时文件、flush/fsync，再发布；保留一个 `last.previous.pt` 备份。last包含模型、optimizer、sampling/Torch/CUDA/Python/NumPy RNG和精确epoch/batch/window计数；若paired control开启也保存它。

启动时先保存零update断点；TRAIN prior保存已完成窗口和原计数，暂停后不从头统计。每轮loss sum/count也跨暂停累加，避免只报告恢复后的半轮均值。监控/校准和评估模型重新加载均保护训练RNG。旧断点没有累积统计时明确标注不完整，不伪造完整轮均值。

weight-only `epoch_*.pt` 只保留最新三轮以及10/14/15/20关键轮次（旧快照在本次新输出目录内轮转，不碰历史实验）。epoch snapshot没有optimizer，不可用于resume。`candidate.pt`是最终固定阈值artifact，也不可恢复训练。

恢复只支持本full protocol的last（或显式指定last.previous），使用 **新输出目录**。SHA256校验train/dev cache与info，核验config/model、reference、population order、batch和原schedule；完成的prior不重计数，未完成的prior从保存位置续接；不重新随机初始化或重置学习率。原本已完成的batch不会再训。轮末监控或最终评估中断时，最多重做被打断的诊断，不重训完成的batch。

SIGINT/SIGTERM只设置停止请求：完成当前optimizer update或当前评估窗口后保存，再退出130（预期的安全停止状态）。shell使用 `tee -i`，避免Ctrl-C先关闭日志pipe使模型来不及保存。不要Ctrl-Z或kill -9；强制杀进程只能恢复上次完整发布的定期断点，不能保证最后未保存的update。未完成初始化时也不能声称已有可恢复模型。

`runtime_status.json`记录PID、Linux启动时间token、输出目录和阶段。管理入口核验/proc实际脚本和out-dir后才发送一次TERM并等待，拒绝PID复用/错误目录，不升级到KILL。发生异常不序列化未完成的optimizer update，只标记failed并使用原完整断点。

```bash
# RUN明确指定此次完整训练的根目录，不猜“最近的”实验。
PY="$(command -v python)"
"$PY" tools/real_motion/manage_p0_f9_joint_training.py status --run-dir "$RUN"
"$PY" tools/real_motion/manage_p0_f9_joint_training.py stop --run-dir "$RUN"
bash tools/real_motion/run_p0_f9_joint_interim_eval.sh "$RUN" dev64
# dev512可替代dev64；不跑full4369，不用dev重校准或选择best。
"$PY" -u tools/real_motion/manage_p0_f9_joint_training.py resume --run-dir "$RUN"
```

管理入口从原execution_contract完整重放训练参数，自动设置新输出目录；也可 `--out-dir NEW_RUN_ROOT`。不会默默改变batch/source/epochs/history。旧六历史断点只能续旧六历史；本次正式新实验严格四历史→六未来。改变观察预算必须从头启动。

中途评估先复制**单次打开的完整checkpoint**并计算内容SHA，以快照运行真正的learned joint transport+columns；原last可继续被训练原子轮换，不再用“原路径SHA变了”误报只读评估失败。独立进程不修改训练模型、optimizer/RNG、阈值或cache，不自动promote。last/epoch使用固定GEN=.5、REF=.5、REMOVE-off，与轮末监控一致；candidate使用其原TRAIN-only校准阈值。评估期间中断不生成完成summary。

此独立评估仍使用CPU/GPU，大范围dev512建议先安全暂停训练。legacy E14有六历史观察，仅作旧参考，不是严格四历史的公平同预算baseline。正常小规模读快照可以并行，但可能降低训练吞吐。

## 运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_joint_causal_columns_full.sh 15
```

想训20轮，把最后的15换成20，从一开始声明20轮即可。长训可用nohup保持SSH断开后继续：

```bash
mkdir -p /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns
FULL_LOG=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/launch_full15_$(date +%Y%m%d_%H%M%S).log
nohup bash tools/real_motion/run_p0_f9_joint_causal_columns_full.sh 15 > "$FULL_LOG" 2>&1 &
echo "$FULL_LOG"
tail -f "$FULL_LOG"
```

此时Ctrl-C只结束tail，不结束后台训练。查看输出日志打印的实际带timestamp目录，发回最终 `model/summary.txt`、`model/epoch_history.json` 和 `model/progress.jsonl`。

中断恢复：`FULL_JOINT_RESUME=/实际旧输出/model/last.pt bash tools/real_motion/run_p0_f9_joint_causal_columns_full.sh 15`。总轮数及原batch等参数必须保持一致；这里不猜测旧实验路径。

## 本地验收边界

测试覆盖所有window每轮唯一且完整、empty/oversized source保留、CPU worker顺序及错误传播、历史预计算与原路径一致/不读GT、单窗口与旧训练更新一致、多窗口source link、完整入口/中途恢复权重与RNG一致、拒绝变更schedule或用candidate/screen做初始化。真实nuScenes全量15/20轮效果和CUDA提速未在本地伪造。
