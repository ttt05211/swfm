# 当前前向关联 AdaLN 完整两轮：固定前三断点等权融合

用户要求“最好的三个”融合后在原val64对比。按当前主要关注的最终Refiner **STC mIoU**，用户刚给出的报告前三为10292（18.229030）、10000（18.217580）、4000（18.135422）。固定这一组，按时间顺序保存为 `[4000,10000,10292]`，各占1/3。不搜索其他组合或系数。这是根据已经反复检查的开发64标签选出的配方，不是预先固定或独立测试。

## 实现

新增独立 `tools/causal_forward_average`，复用既有参数平均工具和冻结均值评估循环的私有命名空间；不改旧融合入口、原训练/数据/缓存或当前full源码。

- 只接受当前 `causal_forward_full` 完整两轮协议/模型、2,816,122参数、前向关联、dense-valid、原20430/64顺序、GT ego、FREE=0、age=False。检查两轮预算5145/5147，总10292。
- 三个源文件为 `step_0004000.pt`、`step_0010000.pt`、`epoch_0002.pt`。各自epoch/batch必须为 `(0,4000)`、`(1,4855)`、`(2,0)`；不能套旧脚本把4000当成第二轮。
- 模型参数以float64等权求均值再转原dtype；非参数buffer必须完全相同，复制一份；BatchNorm或不兼容/非有限状态拒绝。只保存网络，不融合或写入Adam/RNG，不允许该产物作为训练resume。
- 使用当前full模型loader和当前前向关联预测器，兼容旧6-token元数据与实际末4帧。GT ego、两路dense-valid、KTA query+skip、无ICP、纯历史Memory/AdaLN-Zero/FPN/U-Net均保持；不导入旧Integrated网络/旧后处理。
- 单卡/单模型进程，原同64窗OCC+GT/STC+GT，完整H6前向后评分1/2/3s full-grid和Moving。FREE−1和固定历史衰减关闭。
- 原三组单模型指标只读源 `results.json` 中已完成的对应评估；只新评mean3。汇总包含三个原断点和均值模型的四路分数，以及均值相对每个源的正负差值，不自动判优或推广。

新目录保存 `mean3.pt`、`mean_receipt.json`、`contract.json`、`results.json`、`summary.txt`、`comparison.md`，以及OCC/STC各自完整窗口整数状态。原源ckpt/Adam/RNG、报告和历史缓存不写入；源及模型指纹前后检查。当前DEV有选择偏差，不承诺融合涨点或全集收益。

## 服务器命令

仓库改回公有后沿用原代理同步：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_MEAN_GPU=0
export FORWARD_MEAN_RESUME=0
export FORWARD_MEAN_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_mean3_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_average/run.sh > "${FORWARD_MEAN_DIR}.log" 2>&1 &
tail -n 30 -f "${FORWARD_MEAN_DIR}.log"
```

若保留训练时的 `FORWARD_FULL_DIR`，直接读该源目录；未设置时只允许自动找到唯一已完成的 `adaln_forward_full_2ep_*` 当前协议目录。多个符合目录时显式设置原训练目录，不按日期/分数自动挑源。

```bash
cat "$FORWARD_MEAN_DIR/summary.txt"
```

融合权重为 `$FORWARD_MEAN_DIR/mean3.pt`。中断后保持原输出目录，设置 `FORWARD_MEAN_RESUME=1` 再跑同脚本；已完成模态/窗口不会重评。原三个对照不重评，fresh与cached对照不承诺逐字节一致，NOT FPS。

本地覆盖当前full真实参数均值、buffer/有限性/协议与游标拒绝、源目录歧义拒绝、实际冻结workflow完整窗恢复/不重评单模型和已完成模态、源文件SHA不变、当前前向预测器CPU/CUDA H6及6-token元数据/未来标签隔离。暂无服务器新融合分数。
