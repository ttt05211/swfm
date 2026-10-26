# 当前前向关联 AdaLN：完整TRAIN×2轮

用户最终确定保留AdaLN-Zero与前向关联，关闭FREE−1及固定历史衰减。小实验C的已知路径为 `adaln_forward_single_free20_20261025_161644/C_train`。历史衰减在同64窗的OCC ΔmIoU/IoU/Moving为−0.057574/−0.146928/+0.018811，STC为−0.001587/−0.022261/+0.007304，无采用依据。随后用户明确“大训练用双卡”；本入口默认两卡训练，评估仍只有rank0使用一张卡。

## 固定设置

- RANDOM完整TRAIN20430×2轮；不续训pilot，不导入pilot/旧模型权重或Adam。
- FPN、Scene–Temporal Encoder2、cross-first Future Decoder2/AdaLN-Zero、KTA query+skip、纯历史Memory/direct history yaw、plain3D U-Net；2,816,122参数，与验证过的C结构相同。
- 新前向身份/KTA用于TRAIN/OCC/STC全路径；无ICP；末四帧语义/known/ego历史→六帧未来，GT future ego一致用于Decoder/render。
- 只用原TRAIN GT训练；STC仅评估；dense-valid，完整六帧CE+Lovasz，无ego/motion/existence辅助。
- 全局B4/source128预算；双卡分配同一全局分组、不扩大全局batch；沿原完整循环的全局归一化梯度求和、clip和AdamW。
- 原LR为motion/Encoder `5e-4`、Refiner `3e-4`，两轮整周期cosine，floor0.01。完成更新的Adam、各rank RNG、batch/epoch游标一并恢复；恢复必须保持训练卡数。
- 每2000 GLOBAL更新及轮末保存，并在同64窗做OCC+GT/STC+GT Transport/Refiner评估，评分1/2/3s和Moving；rank1等待，rank0单卡评估。
- FREE偏置0，固定age关闭，无旧norm/calibration/half-KTA/static-support、内部ego头、噪声或STC监督适配。

## 实现与来源

独立 `tools/causal_forward_full`；旧pilot、已完成评估、原full入口及缓存均不覆盖。复用原完整两轮训练/保存/Adam-RNG恢复循环的私有命名空间，仅修正关联策略/评估频率元数据与日志，设置实际新Provider/前向函数/评估器。两轮完整分组与原全量planner相同，pilot训练key必须属于完整TRAIN，评估64顺序及STC清单严格匹配原C。

只读原C的 `contract.json` 固定结构/seed/LR/数据/实现，不读取C训练权重。E14路径仍是原循环需要的结构/缓存来源，新的trainable body从随机初始化构造；不将E14权重导入新body。源实现、配置/原TRAIN/VAL元数据与STC库存保持校验。

新raw评估器使用独立命名空间，不接FREE读出；历史元数据通过已验证的末4适配再调用原前向构建器，兼容旧6-token记录，实际数组/poses仍严格4帧。避免再次出现训练完成后STC评估的长度误拒。四路六帧预测后才使用未来占据标签/Moving支持。

## 服务器启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_FULL_GPUS=0,1
export FORWARD_FULL_RESUME=0
export FORWARD_FULL_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_full_2ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_full/run.sh > "${FORWARD_FULL_DIR}.log" 2>&1 &
tail -n 30 -f "${FORWARD_FULL_DIR}.log"
```

`Ctrl+C`只退出日志查看。进程在后台。结果保存在 `$FORWARD_FULL_DIR/summary.txt`、`results.json`，训练进度为 `progress.jsonl`，断点为 `last.pt`、`step_0002000.pt`等及 `epoch_0001.pt`、`epoch_0002.pt`。保留该目录路径；不按分数或修改时间自动找断点。

若中断，保持相同的 `FORWARD_FULL_DIR` 和两卡设置，设置 `FORWARD_FULL_RESUME=1` 后重跑同脚本，只从该目录 `last.pt` 续训，不重启cosine。若需切换物理GPU编号仍需保持两卡及相同软件/数据/代码契约。此入口独立协议拒绝pilot/旧full/不同卡数与日程的断点。

本地验证覆盖完整两轮planner、初始化与原C权重逐tensor及RNG一致、协议/来源隔离、CPU/CUDA实际raw四路评估含6-token元数据、GT标签后读、保存/评估频率、真实小CUDA优化与Adam/RNG停续。没有本地双GPU硬件，不声称已实测两卡完整训练速度或新全集质量；分布式训练核心沿用原已用过的全局梯度路径。服务器训练尚待用户启动。
