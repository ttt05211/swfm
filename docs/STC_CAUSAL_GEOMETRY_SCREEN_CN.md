# 冻结模型：Pred ego 与 STC 历史几何集中对照

用户约束：不改网络、不重训、不使用 aligned 分数修饰主实验。此入口是一个改变推理输入的候选对照，**不是无损执行后端，也不是已验证的新论文指标**。

## 问题边界

当前 Pred 缓存包含六个世界 XY 与相对当前姿态的独立 yaw，未来 z/tilt 沿用 t0；评分标签仍在真实未来 ego 坐标系，不作事后 GT 对齐。
STC 输入完整发布预测，不额外使用 GT camera/lidar visibility。既有 Strong 的动态初速来自最后两帧组件质心差，感知形状抖动可能影响速度；这里不能断言它就是主要掉分原因。
CCR 的 ADD-only 规则不能清除或重标已经继承的假占据。直接更改动态速度还可能与已学习的 residual 重复修正。

用户最近粘贴的是约千窗未完成前缀：Occ+GT / Occ+Pred / STC+GT / STC+Pred 平均 mIoU 分别为 42.723579 / 21.645784 / 20.107599 / 14.205545。不能混用较早64窗或 full4369 数字。

## 一趟固定 dev64，九个配置

| 配置 | 历史 | 未来 ego | 输入适配 |
| --- | --- | --- | --- |
| b_occ_gt / b_occ_pred | 原 OCC | GT / 原 planner | 不变 |
| b_stc_gt / b_stc_pred | 原 STC | GT / 原 planner | 不变 |
| pose_occ_pred / pose_stc_pred | 各自原历史 | planner + 因果路面高度/倾斜 | XY/yaw 不变 |
| temporal_stc_gt / temporal_stc_pred | 稳定后的 STC | GT / 原 planner | 只稳定既有地面列 |
| combined_stc_pred | 稳定后的 STC | planner + 原 STC 路面补偿 | 两项组合 |

相同历史的配置连续执行，复用既有已验收的 history-only 运动/证据准备；未来 Strong、渲染和投影仍各自计算。
模型固定为 5/6/8/12/14 等权平均，四历史六未来，ADD0.5 / REMOVE-off。无阈值搜索、checkpoint 搜索、自动选择或全集重跑。

### 历史路面位姿补偿

仅使用四张实际历史的 road11 体素和历史 odometry，变到世界系。相同0.2m XY列重复观测不重复计空间支持；每历史最多3000个确定性采样点。
在当前位置和各 planner XY 的8m邻域拟合鲁棒地面，最多256邻居、至少32唯一列、75%内点，残差上限0.2m、坡度上限0.15；退化协方差或查询落在内点凸包外时拒绝。
用 t0 的 ego 离地高度和车体相对路面姿态作校准，估计 planner 位置处的高度/倾斜；严格保留 planner 世界 XY 和 yaw。
高度变化超过0.8m、倾斜变化超过3度或历史区域不支持时，逐 horizon 回退原 planner。平坦恒定路面应逐值不变。
**不能修复真正的 XY/yaw 规划错误，也不保证恢复主要掉分；更不是预测完整未知地形。**

### STC 历史地面列稳定

四张历史各自在自己的原 ego 坐标系处理：把其他三张真实历史投影过去，至少两张对地面类别和高度一致才改。
仅 class11/12/13/14 的单类别连续1–3体素厚地面列；历史高度分歧≤1格，当前最多移动1格；中位高度的半格平局向当前高度取整。
保留现有厚度；允许有历史一致支持的地面语义重标，拒绝越界和目标位置与非地面占据冲突。
不把历史缺失当 free、不新增占据列；所有动态及其他非地面占据逐字节保护。观察掩码原样保留，不使用 GT visibility。
修改较早历史槽时可用窗口内较晚历史，但所有输入都≤t0，因此是窗口因果而不是每槽实时因果。

动态方面只报告同类因果匹配、初速p90、三帧以上质心线性拟合残差；**不修改速度、不清除动态物体，也不宣称已解决动态时序噪声。**

## 运行

无需重新下载、解压、生成缓存或训练。已有 STC compact、planner v4缓存和冻结 dev64 manifest 必须保留。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_stc_causal_geometry.sh
```

脚本使用已确认的数据路径，输出到新的 `outputs/p0_f9_joint_surface_ccr/stc_causal_dev64_*`。
若冻结 manifest 不在默认 `data/p0_f9_v21_dev64_manifest.json`，指定真实的 `STC_POPULATION_MANIFEST`；不随意重抽人口。
`STC_CHECKPOINT` 可显式指定已有平均权重；否则沿用已有 frozen bundle 发现流程，绝不平均新权重或选新轮次。

同目录安全续评（将路径替换为这次实际打印的新目录，不是旧四设置目录）：

```bash
export STC_CAUSAL_OUT=/实际的/stc_causal_dev64_目录
STC_CAUSAL_RESUME=1 bash tools/real_motion/run_p0_f9_stc_causal_geometry.sh
```

一次窗口完成九路预测后才读取未来语义评分和实际位姿误差；GT-conditioned 基线原本允许未来 GT ego 条件，但不向 Pred 适配器传递它。
每8完整窗口原子保存整数计数，Ctrl-C/SIGTERM在窗口边界停；异常只提交此前完整窗口。源码/数据/权重/规则变化拒绝续评，不迁移旧实验前缀。
新终端续评自动恢复本次契约的 SWFM flags；已有训练与四设置结果只读、不清理或覆盖。

## 如何判断

`summary.txt` 一次给出九路的1/2/3s与均值 IoU/mIoU、相对各自原版变化、原/补偿位姿 XY/yaw/z/tilt 误差、适配支持/拒绝计数、STC t0 假占据/漏占据/错类、动态匹配抖动。
原始整数与每窗口统计在 `evaluation.json`，数据/权重/规则契约在 `contract.json`。
若补偿主要回退或真实误差仍以 XY/yaw 为主，不继续“调高度”追分；若 STC 动态误差占主导，地面稳定不能冒称解决它。
此候选属于新因果推理协议，不能冒充未经修改的官方 planner 复现；人口、外部规划输入和公开缓存前端需继续披露。
本地单元与 CPU 小模型验证不代表真实 L40S 精度或 FPS；服务器结果未跑，不保证涨分。
