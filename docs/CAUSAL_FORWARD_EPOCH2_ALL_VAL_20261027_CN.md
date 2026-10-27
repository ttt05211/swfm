# 当前前向关联 AdaLN epoch2：完整VAL4369

用户在同64与另一批256窗比较后同意以epoch2评全集。本入口仅评当前 `causal_forward_full` 的 `epoch_0002.pt`（10292更新），不是mean3，也不评其他训练步或旧网络。现有权重、优化器、RNG、原报告/缓存不写入；新结果独立目录，无训练、重新融合或参数选择。

## 配置与执行

- 当前AdaLN-Zero/FPN/U-Net、Scene–Temporal2/Future Decoder2、KTA query+skip、前向历史身份、无ICP、纯历史Memory/direct历史yaw。
- 四历史→完整六未来；GT future ego一致用于Decoder与renderer；OCC和STC都dense-valid。FREE−1、固定历史衰减关闭，无内部ego头/导航/额外后处理或底层执行后端修改。
- 全部VAL4369、150场景，包含之前64及256，不再排除场景。完整网格1/2/3s与Moving评分，无可见性指标mask、GT pose alignment或planner覆盖裁剪；缺输入报错，不缩小人口。
- **一张物理GPU、两个独立推理进程**，两个进程继承相同CUDA_VISIBLE_DEVICES。按父有序keys的rank stride2分成2185和2184窗，分别保存完整窗整数计数。父进程校验身份、来源SHA、known、窗口总数、重复/遗漏与Moving支持，先累加confusion整数，再算IoU/mIoU，绝不平均两个进程的分数。
- 原有单卡并行/恢复框架用私有命名空间复用，重绑定到当前full loader与当前四帧前向预测器；旧脚本与旧实验均不改。新末4元数据适配保留，兼容缓存记录6 tokens、实际输入4帧。
- 原训练协议/模型配置/数据/代码SHA保留校验；只接受完整epoch2/batch0/update10292。单模型权重由两个worker分别读入，源SHA冻结到各自新快照；不读取mean3或其来源断点来评估。

质量评估耗时不是FPS；并发也不保证固定倍数提速。旧64/256计数不导入新结果，全集统一重评同一入口。

## 启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_ALL_GPU=0
export FORWARD_ALL_RESUME=0
export FORWARD_ALL_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_epoch2_all_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_all/run.sh > "${FORWARD_ALL_DIR}.log" 2>&1 &
tail -n 30 -f "${FORWARD_ALL_DIR}.log"
```

保留训练时的 `FORWARD_FULL_DIR` 即读该目录；未设置则只允许找到唯一已完成当前协议的 `adaln_forward_full_2ep_*`。多候选需显式指定源目录，不按分数或时间自动挑。GPU默认0，每进程CPU workers默认2，可显式 `FORWARD_ALL_CPU_WORKERS=4`。

父日志每约30秒打印 `FORWARD_ALL_PROGRESS occ=…/4369 stc=…/4369`。若某个状态文件正写入，显示saving，不虚构进度。每个worker细节在 `$FORWARD_ALL_DIR/shard_00.log`、`shard_01.log`。

```bash
cat "$FORWARD_ALL_DIR/summary.txt"
```

最终 `evaluation.json`、`summary.txt` 含四路线在1/2/3s与Avg的IoU/mIoU/Moving；`occ_merged_counts.json`、`stc_merged_counts.json`保存合并整数计数。中断时保持原 `FORWARD_ALL_DIR`，设 `FORWARD_ALL_RESUME=1` 重跑同脚本，自动恢复两worker完整窗状态；完成窗口不重新推理。

本地验证包含当前完整模型loader/旧与mean断点拒绝、完整4369/150人口、两worker相同GPU及当前子模块调用、奇数分片整数合并与重复/缺失计数拒绝、真实指标引擎中断恢复不重评OCC、当前CPU/CUDA H6预测及6-token兼容、只读源SHA与进度文件半写处理。没有本地真实nuScenes全集或服务器执行连接，不宣称已代跑完或有新全集分数。
