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

预计算仅包含：因果history-source association/ICP、历史footprint和static memory。worker不访问Torch/CUDA/model或GT来构建这些证据；主线程重新提取的current source与worker逐元素核验。

以下内容永不跨更新缓存：learned poses、owner/fallback、候选、action labels、learned features。运动更新后全部在线重建，不出现旧bank错配。来源帧RAM缓存默认TRAIN/dev各256MiB，保持原精确inverse-map采样，不创建新的dense磁盘cache。

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

## 校准和评估

- 类别先验只计数一次固定、scene-balanced TRAIN1024的**未采样合法proposals**，然后固定。明确是prior subset，绝不称full20430统计；不是每轮加一次3小时以上全量audit。
- 全量20430都优化，所以最终TRAIN64校准是 **in-sample TRAIN-only**，不是held-out，也不使用dev标签/阈值扫参。报告中 `held_out=False`。
- 每轮固定dev64监控，固定GEN/REF threshold=.5、REMOVE-off，记录current transport/GEN/REF/JOINT及E14比较；只看趋势，不自动early stop、不按dev挑best、不自动扩训。
- 最终声明轮数的checkpoint经过TRAIN64固定grid校准，固定后仅一趟dev512，并同时统计其中的dev64。默认不跑full4369。
- 这回答的是「足够训练后，一阶段是否能超E14、补全是否有效」。默认没训paired control，不能声称额外收益完全来自joint gradient。与以前两阶段的.3269pp比较时应注意训练预算和population/阈值校准差异。
- 只有真实有效编辑、全体分支/各horizon非退化、joint相对E14非退化及link存在等门槛全通过，candidate才允许默认deploy。身份变换/全关闭不叫生成成功，失败不自动retry。

## 保存、恢复

每256batch原子替换 `last.pt`，每轮结束前（开始eval之前）也保存，eval后再保存epoch cursor/history。last包含模型、optimizer、sampling/Torch/CUDA RNG和精确epoch/batch/window计数；若paired control开启也保存它。

weight-only `epoch_*.pt` 只保留最新三轮以及10/14/15/20关键轮次（旧快照在本次新输出目录内轮转，不碰历史实验）。epoch snapshot没有optimizer，不可用于resume。`candidate.pt`是最终固定阈值artifact，也不可恢复训练。

恢复只支持本full protocol的last，使用 **新输出目录**。SHA256校验train/dev cache与info，核验config/model、reference、population order、batch和原schedule；恢复后不重做prior audit，不重新随机初始化或重置学习率。原本已完成的epoch不会再训。

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
