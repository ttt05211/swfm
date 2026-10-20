# 新ego Query／纯历史Memory／Cross-first epoch2：完整VAL4369

独立入口 `tools/source_scene_ego_all/run.sh`，只加载 `tools/source_scene_ego_full` 完整TRAIN×2的epoch_0002.pt/update10292。不接受小实验、早期GT/Pred/ICP、mean或其他更新断点；不修改旧训练/评估文件和结果。

真实模型由新full checkpoint_model构造，EgoPredictor显式调用新candidate adapter.forward_window，不调用继承的旧无egoforward。新结构的相对future ego Query、pure-history Memory、cross→self→FFN、KTA Query/skip、无ICP和三AdaGN全部保留。GT future ego同一轨迹条件化Decoder和未来几何。

## 口径

- ALL4369/150 scenes，完整缓存键集合，无planner trimming或挑窗口。
- OCC＋GT、STC＋GT，历史均dense-valid；full6预测，1/2/3s full-grid IoU/mIoU/MovingMicro/Macro；不使用可见性评分mask，不调阈值。
- 仅质量评估，不训练、无ego预测头/导航；不引入旧norm192/calibration/halfKTA/age/support等后处理。
- 一张物理GPU上两个独立模型进程，各处理按原顺序stride2切分的2185/2184窗，无补齐、无重叠。每个进程依次评OCC和STC；全部完成后先SUM整数混淆计数，再算全局IoU，不平均分片分数。
- 元数据、模型/实现SHA、source contract、分片身份、完整人口和整数计数总量严格检查。每进程独立checkpoint字节快照、完整窗口恢复；原checkpoint/Adam/RNG/history caches只读。
- 普通BF16执行，非bitwise parity声明、非FPS，不承诺双进程2倍加速。协调器不执行模型推理；两个推理进程使用相同CUDA_VISIBLE_DEVICES。

## 运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
export EGO_FULL_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_query_full_2ep_20261019_231946"
export EGO_ALL_GPU=0
unset EGO_ALL_RESUME
export EGO_ALL_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_query_epoch2_all_2proc_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_ego_all/run.sh > "${EGO_ALL_DIR}.log" 2>&1 &
tail -f "${EGO_ALL_DIR}.log"
```

总日志每约30s输出已保存进度：`EGO_ALL_PROGRESS occ=n/4369 stc=n/4369`。文件暂时写入中显示saving，不伪造0进度。每shard每8窗保存一次，刚启动提取/检查输入阶段尚未计窗。完整子日志在 `shard_00.log`、`shard_01.log`。

完成看 `cat "$EGO_ALL_DIR/summary.txt"`。结果四条Transport/Refiner路线，含各时距和Avg；另存合并后的OCC/STC原始计数JSON。

中断后设回相同EGO_ALL_DIR、EGO_FULL_DIR、GPU编号和CPU预算，`export EGO_ALL_RESUME=1`，同入口追加日志续评。不能同时启动两个相同输出任务；状态/代码/源变更拒绝混合。某worker失败会通知另一个保存完整窗口后退出，不把部分结果当全集报告。不要为恢复改checkpoint或known口径。

本地测试包含真实CPU/CUDA新模型forward与adapter完全一致（仅精确身份审计用既有deterministic上下文）、旧forward trap、完整人口/源拒绝、整数SUM而非平均分数、同GPU子进程启动参数、双shard恢复、进度文件并发写保护。进程编排为模拟子进程测试，未声称本地4GB卡实测两个完整模型并发或服务器速度。没有可用服务器连接/最新权重，需用户执行命令；此文档不是新全集得分。

新入口8项＋新full/旧全集回归8项，共16 passed（34.63s）；Bash-n与diff检查通过。
