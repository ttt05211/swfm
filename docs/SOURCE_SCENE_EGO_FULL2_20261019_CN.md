# 新三项结构：完整TRAIN×2轮

独立入口 `tools/source_scene_ego_full/run.sh`，完全复用新pilot的EgoBody/make_model和adapter/evaluation，不修改任何旧2+2、小实验或只读检查文件。

- 新结构：ego→Future Query、纯历史Memory（无source_query参数/内容/importance连接）、Decoder cross→self→FFN；保留2+2深度、三AdaGN、六learned queries、KTA query+skip、无ICP、track_valid、历史相对yaw及direct memory。
- RANDOM，完整原TRAIN20430×2，不续用pilot/旧GT/mean/head权重或Adam。旧E14仅供原架构元数据和兼容纯历史缓存读取；不继承科学模型权重。
- GT future ego同一轨迹同时条件化Decoder/Strong/Transport/Memory，dense-valid用于TRAIN/OCC/STC。原source motion坐标不变；没有ego预测头、导航、额外监督或真实STC训练。
- full6 CE＋Lovász ONLY；global B4/source128分组、不截断source；默认双卡同步梯度；原motion LR5e-4、Refiner LR3e-4、AdamW/clip及整两轮cosine floor0.01。预计更新数10292，与旧full2分组相同。
- 每2000 GLOBAL更新及每轮结束：保存checkpoint、rank0单GPU评估相同remaining64 OCC/STC和learned−KTA/运动诊断。不是全VAL评估，不选best；原模型保留作同预算对照。

## 启动（后台，关终端不依赖前台会话）

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
export SCENE_TRAIN_GPUS=0,1
unset EGO_FULL_RESUME
export EGO_FULL_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_query_full_2ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_ego_full/run.sh > "${EGO_FULL_DIR}.log" 2>&1 &
tail -f "${EGO_FULL_DIR}.log"
```

`Ctrl+C`退出tail不终止nohup训练。完成或中途看已有评估：`cat "$EGO_FULL_DIR/summary.txt"`，同样在日志打印。

checkpoint：`last.pt`、`step_0002000.pt`等、`epoch_0001.pt`、`epoch_0002.pt`。独立新协议，只允许本入口同目录last恢复，不能把pilot/旧full checkpoint当作续训。

## 续训

```bash
export SCENE_TRAIN_GPUS=0,1
export EGO_FULL_DIR=/实际原完整训练输出目录
export EGO_FULL_RESUME=1
nohup bash tools/source_scene_ego_full/run.sh >> "${EGO_FULL_DIR}.log" 2>&1 &
tail -f "${EGO_FULL_DIR}.log"
```

同world-size/数据/实现/contract，恢复model＋Adam＋各rank RNG＋更新游标；LR不重启，沿整两轮cosine继续。中断评估按完整窗口恢复，不重跑已完成评估。

本地非base安全环境：新full4项＋新pilot/ego检查/原full回归20项，共24 passed（25.62s），包括完整TRAIN分组、cadence、实际CUDA微型两轮保存/续训、同fresh权重和loader隔离。Bash语法通过；没有启动服务器科学训练，不代表质量已验证。
