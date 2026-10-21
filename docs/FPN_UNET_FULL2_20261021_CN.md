# FPN＋3D U-Net：完整 TRAIN 两轮

沿用组合试训已验证的架构，**完整TRAIN20430×2轮，从随机初始化开始**，不是20%×2，也不是加载20%一轮权重继续微调。单一GT ego条件配置，不加结构对照或新loss。

默认两张GPU、global B4/source128，原seed20261011/AdamW（WM含Encoder 5e-4，Refiner含lift 3e-4）、整两轮cosine floor.01、clip5、BF16网络/FP64几何、full6 CE+Lovasz ONLY。Source WM/GTego Query/history-only Memory/pinned tracks/两帧KTA/无ICP完全沿用试训；Refiner仍8/16/32 U-Net，参数2520442。STC只用于评估，不参与训练。TRAIN/OCC/STC dense-valid，全网格评分。

复用已有原始历史和纯几何cache，require只读；新Encoder特征/Memory现场重算，不提取或训练新的冻结特征bank。本run独立目录内可构建纯历史identity缓存。旧试训、扩展512评估以及其他模型文件均不修改。

每2000 GLOBAL updates和轮末保存；**仅每轮结束评一次同remaining64 OCC/STC Transport与Refiner**，不在每2000步额外评、不自动跑512或搜索best。最终固定epoch2。按既有真实数据grouping预计约10292次global updates，启动时以实际两轮group总数为准。单卡也支持，但恢复时卡数、参数、数据和代码必须保持不变。

## 更新代码

```bash
cd /root/nas/occ/swfm
git switch feature/shared-history-execution-speed-v2 &&
git pull --ff-only https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
```

## 启动

```bash
conda activate OccFM
export SCENE_TRAIN_GPUS=0,1  # 若仅一张空闲GPU，设为0
export FPN_UNET_FULL_RESUME=0
export FPN_UNET_FULL_DIR="$PWD/outputs/p0_f9_shared_history/fpn_unet_full_2ep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$FPN_UNET_FULL_DIR")"
nohup bash tools/fpn_unet_full/run.sh >"${FPN_UNET_FULL_DIR}.log" 2>&1 &
export FPN_UNET_FULL_PID=$!
echo "PID=$FPN_UNET_FULL_PID  DIR=$FPN_UNET_FULL_DIR"
tail -f "${FPN_UNET_FULL_DIR}.log"
```

如果希望前台跑，把nohup及后面PID/tail行替换为 `bash tools/fpn_unet_full/run.sh`。前台Ctrl-C会在完整update/eval window后保存；后台的tail上按Ctrl-C只是停止看日志，不会停止训练。

## 恢复

仅恢复这次完整两轮run的`last.pt`，保留Adam、两轮cosine、每卡RNG和游标，不从试训接续、不重启学习率。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export SCENE_TRAIN_GPUS=0,1
export FPN_UNET_FULL_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_full_2ep_实际时间
export FPN_UNET_FULL_RESUME=1
bash tools/fpn_unet_full/run.sh
```

查看结果：`cat "$FPN_UNET_FULL_DIR/summary.txt"`。checkpoint有`last.pt`、`epoch_0001.pt`、`epoch_0002.pt`、每2000步的`step_*.pt`；独立full-protocol loader拒绝pilot。

该入口只实现并验证恢复流程，本地没有全量服务器数据/执行连接，不声称已启动或完成完整训练。局部CUDA测试不是科学质量结果；一轮试训测得的FPS不等于两轮权重已验证FPS。

本地检查：新增4项＋原组合8项共12 passed（13.08s）；补充CPU/CUDA RNG逐字节恢复断言后新4项再过（7.58s）。覆盖全20430窗口两轮计划无丢弃、固定同64、checkpoint/评估cadence、真实小网格CUDA跨轮停续、Adam矩和每卡随机状态、loader隔离、完成后不重复评估。Bash-n及CLI帮助通过。旧试训和冻结512代码零diff，未在本地实测双GPU正式训练。
