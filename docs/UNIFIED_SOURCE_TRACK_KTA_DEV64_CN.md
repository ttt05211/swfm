# 统一Source Track＋多帧KTA：独立冻结DEV64对照

独立分支 `experiment/unified-source-track-kta-dev64`，基于 `88d8dac`。只新增本实验目录和测试、本文档，不改正式模型、训练器、最终评估、STC适配或旧缓存源码。此实验不训练、不自动选择、扩训或部署。

## 三路与边界

固定完整GT epoch2/update10292，原DEV512内固定scene-balanced DEV64，OCC+GT ego和STC+GT ego同64窗。默认两种输入；`TRACK_INPUTS=occ`可预先只做OCC，不按跑分切换。STC缺四历史帧失败，不补窗或裁剪人口。

1. baseline：当前正式预测器的旧两帧KTA＋旧四帧历史配准。
2. unified_two：单一四帧关联表给KTA、STWM patch、feature memory，最近两个有效点估速度。
3. unified_fit：与第二路完全同一关联和配准，全部有效原始世界质心线性拟合速度。

保留已有associate_backwards的同类别、运动预测距离/物理距离、25m/s、贪心一对一、0.4m歧义拒绝和缺帧接续；不新增shape IoU、类组、静止阈值或tracking网络。旧两帧速度仅初始化已有关联算法，不再次生成各消费方的身份。历史形状ICP仍原样，它用于空间配准，不是新增shape匹配代价。track记录关联有效性；ICP拒绝的帧仍不供STWM/动态memory，不把“身份匹配到”误当“空间证据可对齐”。

关联仍原nominal 0.5s时钟以隔离匹配规则变化；新两点/多点速度用实际历史时间戳，未来仍六个名义0.5s槽。只有一个点速度零，两个点fit等于两点差分。拟合原始世界XY，不拟合ICP对齐后的质心；Z速度零。只换斜率，当前source中心不变，`anchors=center+KTA`同步更新。25m/s是已有匹配门限，不对fit新增截断；超25速度单独记录。

三路固定baseline fresh Strong/CLEAR背景，不把额外Strong算法变化混进这次对照。网络motion及memory/prior坐标重新计算；允许共享同一冻结Encoder对相同历史语义/known的live F，没有共享跨模态或learned trajectory。此screen不是“整个Strong也已经统一”的正式部署版本。使用现有确定性scope做受控BF16质量比较，不称正式FPS。

## 缓存是否要重建？

旧磁盘geometry缓存确实含source组件、历史registrations，以及依赖旧速度/未来ego的Strong背景等。旧record缓存也含旧KTA与anchors。因此不能拿旧registrations或旧KTA字段直接当新版。

本次不重建整套大缓存：三个版本都从只读原始四历史准备，当前/前一帧的纯组件提取只在同窗复用；新版关联、配准、KTA、source读取和memory坐标在RAM现场构造，完整窗口后释放。旧磁盘cache不打开作几何输入、不写新geometry bank。旧VAL record文件仅用于固定人口和baseline入口核验，不把未来标签字段传给预测器。

如果后续采用新方法重训：可复用语义、known、ego位姿和规则不变的逐帧组件；需为新版track/registration/KTA/读取坐标等派生数据建立独立版本/namespace或继续live构造。Strong/CLEAR是否一并采用新速度是另一明确方法选择，不能静默混旧背景；原缓存及实验保留。无需重新下载/提取STC数据或重建学习特征bank。

## 指标

完整六帧预测，报告1/2/3秒及平均标准full-grid IoU/mIoU/MovingMicro，Transport和SemanticRefiner均列。全部模态/三路预测结束后才读未来标签、t0/future annotation与Moving support。

KTA误差遵守旧displacement-preserving目标：比较预测位移与同GT instance的future_center−t0_center，不把观测组件直接对齐GT框中心。离线GT标签使用原同类别一对一4m匹配约定，**该4m只属于评估标签，不进入历史track规则/预测**。同一模态三路共用source和目标有效mask；报告1/2/3秒均值/P90/支持量，并按1点、2点、≥3点track拆分，避免不同可评分source造成假收益。GT/STC source数量不同，不冒称逐物体跨模态成对。

输出evaluation.json、完整窗整数state.json、summary.txt、contract/population和冻结GT快照。Ctrl-C保存完整窗口，半窗重算；source SHA/元数据/历史文件stat/源码/预算变化拒绝恢复。原Adam/RNG、权重和缓存不写，没有新训练断点。

## 服务器运行

本地38项专项及原GT/正式评估回归通过（11.56s），含真实CPU/CUDA小网络、非均匀时间戳、歧义/缺帧、ICP前质心、统一source patch/memory、常速无变化六帧对照、冻结参数、GT标签延后及整窗恢复。没有真实nuScenes数据，不宣称新方案已提升指标。

代码应在独立worktree运行，避免切换正在开发/运行的主目录分支。**以下命令需在该独立分支推送后使用**，当前实现不自动执行远端推送或服务器评估：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin experiment/unified-source-track-kta-dev64
git worktree add /root/nas/occ/swfm-unified-source-tracks FETCH_HEAD
cd /root/nas/occ/swfm-unified-source-tracks
export GT_CKPT=/root/nas/occ/swfm/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt
export TRACK_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/unified_tracks_dev64_$(date +%Y%m%d_%H%M%S)
bash tools/unified_source_tracks/run.sh
cat "$TRACK_DIR/summary.txt"
```

新worktree不会带原upstream_occfm目录内容；若服务器upstream依赖不是已安装包，运行前设置 `export PYTHONPATH=/root/nas/occ/swfm/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}`，不复制模型/数据或大型cache。主训练目录分支不变。若目录已有，请进入其实际目录，不重复worktree add、不覆盖目录。

中断后，在这个独立worktree内指定原实际TRACK_DIR，再 `export TRACK_RESUME=1; bash tools/unified_source_tracks/run.sh`。脚本不停止其他GPU任务；正式测速或质量吞吐比较需用户自行避免资源争用。本地小图测试不等于真实nuScenes收益，真实指标以服务器summary为准。
