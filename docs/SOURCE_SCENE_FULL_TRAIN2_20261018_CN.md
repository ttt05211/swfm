# 第二阶段结构 / 完整TRAIN ×2轮

用户明确选择完整20430窗×2轮，而不是TRAIN20%。本次仅增加 `tools/source_scene_full` 新入口和显式模型loader；旧Stage1/Stage2 pilot/测速源码不修改，旧一轮断点及其指纹不受影响。新入口采用新协议、新输出目录，不接受旧pilot断点。

“在第一阶段基础上”指架构继承：Shared Encoder→Scene Reader→Temporal→Future queries，第二阶段再去ICP、改track_valid、移除KTA Query投影并学习历史相对yaw；**不是加载第一阶段训练权重续训**。完整模型从随机初始化开始，不导入原GT、Stage1/Stage2 pilot或norm192权重。E14仅供架构及旧缓存协议读取。

## 固定配置

- 全部20430窗，每轮每窗恰好一次，两个不同epoch shuffle，沿用原scene round-robin及global B4/source128分组，不截source、不padding/drop。
- 整个两轮一次余弦；motion LR5e-4、Refiner3e-4，floor0.01，原AdamW/clip5。只六未来帧CE+Lovász；无ego/motion/existence/对齐辅助监督。
- TRAIN/OCC/STC都是dense-valid；GT future ego明确条件，无导航/内部ego。原两帧KTA仅skip/Strong，不作Query输入。
- 默认两张卡训练，每2000个**全局optimizer update**保存命名checkpoint；每轮结束额外保存epoch checkpoint。不是每个rank分别计2000、也不是2000窗。
- 同时在每2000步和每轮结束，rank0单卡评固定remaining64 OCC/STC，另一rank等待；完整6预测、1/2/3s全网格IoU/mIoU/Moving。所有评估均报告，不按dev选best；不自动跑全集/扩训。
- 旧geometry缓存require/read-only，新identity-only缓存只写本次 `identity_history/rank*`，不缓存学习特征。不覆盖旧输出。

## 启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export SCENE_FULL_RESUME=0
export SCENE_FULL_DIR="$PWD/outputs/p0_f9_shared_history/scene_stage2_full_2ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_full/run.sh > "${SCENE_FULL_DIR}.log" 2>&1 &
tail -F "${SCENE_FULL_DIR}.log"
```

不需要SCENE_PILOT_DIR或Reader结果文件。REFERENCE_DIR沿用原 `stc_empirical_refiner500_20261014_194954`，只读其clean/results.json中的64窗口keys，不读权重或指标用于训练。

## 输出与恢复

命名文件 `step_0002000.pt`、`step_0004000.pt`……，以及 `epoch_0001.pt`、`epoch_0002.pt`、`last.pt`。step和epoch快照在评估前保存模型/Adam/RNG/游标，防止评估中断影响训练；评估结束后的最新计数与状态存last及results。原权重没有根据DEV切换或改写。

`summary.txt`列全部已完成评估；`eval_step_0002000.json`等保存对应报告，state文件保存整窗整数计数。last保存原Adam状态、原两轮余弦游标、每rank RNG，恢复不重启学习率、不重复已完成epoch/评估。

中断后新终端明确填入本次目录、同两张卡：

```bash
export SCENE_TRAIN_GPUS=0,1
export SCENE_FULL_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/scene_stage2_full_2ep_实际时间戳"
export SCENE_FULL_RESUME=1
nohup bash tools/source_scene_full/run.sh >> "${SCENE_FULL_DIR}.log" 2>&1 &
```

正常Ctrl-C/SIGTERM在完整update或评估窗口之后保存；SIGKILL/断电只能回到最近已保存断点。改卡数/数据/代码/人口/日程时拒绝恢复。新checkpoint推理使用 `tools.source_scene_full.model.checkpoint_model`，不是旧Stage1/Stage2 pilot loader。

本地专项及第二阶段/原DDP归约回归共18tests passed28.47s，含20430人口每epoch不漏不重复、跨两轮游标、2000步命名规则、小网格CUDA跨epoch停止/恢复/epoch自动评估/完成后resume不重训及严格新模型loader；Bash语法通过。Windows按环境守卫使用已有安全非base yoloe环境，未修改Python环境。

没有代启动服务器。双卡不保证2倍加速；旧小预算指标和FPS不作为本次全量质量或速度结论。本地小网格不冒充真实完整两轮训练或L40S性能。
