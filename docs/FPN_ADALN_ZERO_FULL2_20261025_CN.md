# AdaLN-Zero：完整 TRAIN×2 轮

## 依据与范围

用户回传匹配 TRAIN4086×1 / remaining64：原 FPN 加法 Ego → AdaLN-Zero 的最终 OCC mIoU 46.810713→47.776497（+0.965784）、IoU +0.149041、Moving −0.533033；STC mIoU 17.202257→17.470307（+0.268050）、IoU +0.011189、Moving −0.283255。当前仅支持扩大验证，不能宣称动态运动改善、全集收益或 AdaLN 单个机制的证明。

本次只接完整 TRAIN20430×2 轮，随机初始化、不续训20%小实验，不加载原加法主体权重。共同 FRESH 参数、原 seed/order/globalB4-source128/H6 CE+Lovasz/AdamW/LR/两轮全程 cosine floor0.01 与已完成加法完整两轮匹配。参数仍为2,816,122，模型完全复用 `tools/fpn_adaln_zero/model.py`，不重写网络。

GT ego 条件明确使用同一条相对未来轨迹，Decoder三分支AdaLN-Zero、renderer未来坐标投影；纯历史Memory、学习历史yaw、KTA Query+skip、无ICP、FPN/U-Net不变。真实 TRAIN GT only，STC只评估；dense-valid TRAIN/OCC/STC。无ego/motion/existence loss、age衰减、FREE bias、校准、内部头或额外消融。

## 实现与匹配

新增独立 `tools/fpn_adaln_zero_full`，在私有模块命名空间复用原 `tools/fpn_unet_full/run.py` 的完整训练循环/梯度归约/保存恢复/evaluator。不修改旧FPN、AdaLN一轮或完整两轮文件，因此旧实现SHA、训练、checkpoint与历史结果不变。独立protocol/输出目录/loader拒绝混入pilot或加法checkpoint。

控制默认 `outputs/p0_f9_shared_history/fpn_unet_full_2ep_20261021_112202/results.json`，只读报告与契约，不加载权重。训练前匹配完整20430和同64的有序keys（显式规范化JSON容器、不排序）、每轮组数与总预算、数据/旧实现SHA、runtime、LR/日程、STC清单。新入口和旧AdaLN模型SHA存入模型experiment契约；恢复严格保留模型/Adam/每rank RNG/cursor，不重新启动cosine。两轮评估正负delta全报，不选best。

双卡训练默认0,1，global batch仍4；每2000 GLOBAL updates及每轮结束保存checkpoint。为延续原FPN完整两轮协议，只在每轮结束rank0单卡评同64 OCC/STC，而非每2000步都eval。训练不依赖多进程评估。约10292 updates以实际原source分组为准，不是20430/4简单向上取整。

## 服务器命令

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export FPN_ADALN_FULL_CONTROL_DIR="$PWD/outputs/p0_f9_shared_history/fpn_unet_full_2ep_20261021_112202"
export FPN_ADALN_FULL_DIR="$PWD/outputs/p0_f9_shared_history/fpn_adaln_zero_full_2ep_$(date +%Y%m%d_%H%M%S)"
export FPN_ADALN_FULL_RESUME=0
nohup bash tools/fpn_adaln_zero_full/run.sh > "${FPN_ADALN_FULL_DIR}.log" 2>&1 &
tail -f "${FPN_ADALN_FULL_DIR}.log"
```

训练日志保留复用原入口的 `FPN_UNET_FULL` 前缀；首行、独立目录、protocol与summary明确AdaLN-Zero。后台运行，退出tail不会停止训练。每轮结束会打印四路指标与控制差值；完整结果见 `summary.txt` / `results.json`。`last.pt`用于恢复，`step_0002000.pt`等及 `epoch_0001.pt`、`epoch_0002.pt`各自保留。

中断后先确认无旧任务，再设回原完整输出目录与 `FPN_ADALN_FULL_RESUME=1` 运行同脚本；同卡数、数据、代码、控制报告不可变。不要用一轮pilot的目录或ckpt，不覆盖旧实验。没有服务器执行连接，本地功能验证不能冒充已远程启动、两轮质量或双卡吞吐结果。

## 本地验证

安全非base yoloe环境新旧25项回归通过（21.54s），包含实际完整20430/4369规划＋JSON有序key检查、每轮覆盖全样本不重复丢失、预算/顺序/STC清单变化拒绝、两轮控制及所有正负差值报告、loader协议隔离与实现指纹、实际CUDA H6前后向＋epoch1停机/epoch2恢复Adam和rank RNG＋最终评估不重复。原网络/一轮入口/完整FPN循环零diff，Bash语法检查通过。首次新测试预期只报total_updates，但保持合同一致需同时改变每轮预算，实际先由epochs_updates检查拒绝；只修正测试匹配字段，不改变生产预算检查。未执行服务器完整科学训练。
