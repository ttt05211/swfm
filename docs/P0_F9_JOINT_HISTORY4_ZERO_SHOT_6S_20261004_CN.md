# 冻结 epoch19 Local 联合模型：nuScenes 6秒评估

只做推理，不训练、不重设查询数量、不调阈值，不改变既有1–3秒主表。
入口：`tools/real_motion/run_p0_f9_joint_long_rollout.sh`。

## 数据流与信息边界

1. 起始输入为四张真实历史 occupancy（t0−1.5/−1/−0.5/t0），第一次
   完整 transport + generation + refine 推理输出0.5–3秒六帧。
2. 把1.5/2/2.5/3秒四张**完整联合预测**作为下一段历史，以3秒为新原点。
   从预测 occupancy 重建 Strong source、因果关联、KTA、局部tube、历史注册和静态证据。
3. 复用同一个模型，输出3.5–6秒。没有GT物体运动、未来occupancy或未来mask作为预测输入。
4. 保留已知未来ego pose至6秒的信息合同；这是trajectory-conditioned forecasting，
   不是未来ego也由模型预测。Moving统计始终相对于**最初t0**，第二段不重置GT support原点。

所有层均严格四历史帧。为复用旧V18的flat-feature ABI，重建时有两个空白前缀slot，
只用free网格和无效track填充；transport encoder只取last4，column网络也只见四张真实/预测帧。
这不是六历史输入。Synthetic record只包含推理字段，不含source supervision或目标标签。

## 合成历史的可见性

预测帧没有真实LiDAR观测。本实现把**起始四张真实历史**的观测mask，分别按ego pose
前向投影到四张合成历史网格并取并集；空间范围外为unobserved。
它是继承的历史证据support，不是真实未来LiDAR可见性，也不把预测为occupied的体素自动当作观测。
该规则固定为 `initial_real_history_observation_union_forward_warp_no_future_sensor_v1`。
generation和refine按原网络运行，但预测历史及继承mask存在分布偏移；性能不能预先保证。

不允许为了提高长时结果读未来mask或GT occupancy。GT只在两个预测块均结束后用于指标。

## 权重和阈值

使用已选定的 `checkpoint_selection_20261004_220555/epoch_0019.pt`，
独立复制为只读评估snapshot；合同校验包括epoch19完成边界、strict4协议、配置、
验证cache/info和冻结dev manifest指纹、TRAIN/dev场景隔离。
阈值固定 `(0.5,0.5,REMOVE-off)`，与选epoch19时一致；不另做TRAIN校准或dev扫描。
不加载optimizer，不保存训练checkpoint，不把长时结果用于挑选新的epoch。

## 人口与报告

- `dev64`：冻结dev512 keys中拥有完整4历史+12未来的窗口，按既有确定性
  scene-balanced round-robin取64，单独固化到contract；**不是原dev64原封不动**。
- `dev512`：原冻结dev512中全部long-eligible窗口，不保证剩余恰好512；记录排除key。
- `all`：full4369 V18 cache中全部long-eligible窗口；没有cache记录的不加入。

所有1–6秒指标用**同一人口**，报告1/2/3/4/5/6秒IoU/mIoU、MovingMacro/Micro，
以及1/2/3与4/5/6平均。逐horizon累加intersection/union后算分；空Moving support为NA，
不把缺失支持计为0。半秒中间帧参与rollout，但不包含在整数秒平均中。

窗口token必须连续、同一scene，时间戳与2Hz相符（允许最多0.06秒累计keyframe抖动）。
不能把此人口上的1–3秒数字与完整dev512直接相减，亦不能直接与论文不同ego/人口/mask合同的数字排名。

## 首窗口与中断安全

首窗口检查：四历史重建模型输入一致；六张transport预测逐voxel一致；
六张完整联合预测与原 `forecast_columns` 部署路径逐voxel一致。
优化推理还有已有的首次概率exactness检查。失败即停止，不静默切换输入合同。

每个完整窗口原子保存`evaluation_state.json`（原始计数、时间、编辑统计、合同指纹）。
Ctrl-C/SIGTERM等到窗口完成后退出。崩溃可恢复到最后完整窗口；不给部分结果打complete标记。
同输出目录受评估锁保护；`--resume`校验快照和同一人口/参数合同，不重复累加前缀窗口。
无持久几何缓存写入；只用256MiB有界原始frame cache和当前/下一窗口预取。

## 服务器命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_joint_long_rollout.sh dev64
# 中断后：保留打印的实际OUTPUT_DIR
bash tools/real_motion/run_p0_f9_joint_long_rollout.sh dev64 "$OUTPUT_DIR" --resume
# 扩大评估（会创建独立输出，不自动串联/重训）
bash tools/real_motion/run_p0_f9_joint_long_rollout.sh dev512
```

`summary.txt`可直接粘贴；`evaluation.json`保存完整数字和raw counts，
`progress.jsonl`拆分first_prepare/forecast、second_prepare/forecast、metrics耗时。
本地synthetic测试不代表真实nuScenes性能通过；真实GPU测试必须在服务器数据上执行。
