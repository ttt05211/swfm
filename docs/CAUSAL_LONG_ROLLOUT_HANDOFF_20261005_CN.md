# 冻结 epoch19 的因果跨段轨迹接续

本次只修改长时推理，不训练、不修改 checkpoint，不改变 1–3s 主表或默认旧递推。
评估属于 trajectory-conditioned forecasting：保留原协议的未来 ego pose 条件。
不使用未来 occupancy、未来 sensor visibility mask、未来 annotation/GT identity。
过去四帧真实 occupancy/mask 是原始任务输入，不能把它们与未来 GT 混淆。

## 改进与边界

第一段保留 original source index、六帧实际 renderer target 的世界坐标中心、
最终可见 owner grid。第二段仍从完整预测中提取 Strong components，保留其类别、
形状、顺序、质心、当前 yaw 外观；通过同类别的可见 owner voxel 找到旧身份。

仅在 owner 对应唯一、没有 split/merge、最后两帧有预测 source footprint、
所有保留轨迹段不超过原 Strong 25m/s gate、形状原点偏差不超过固定 4m 时接续。
新组件、关联不明确、合并/分裂、越界丢失和不可靠轨迹都走原 redetection 路径。
不 resurrect 缺失 source，不做 memory-only 几何写入，不将 prediction 当成真实观测。

对匹配的 source，将第一段的最后四个预测中心统一平移到第二段当前形状原点。
这个常量平移保留每个时段的位移，消除帧间补全质心变化造成的伪速度。
按最后两个预测中心计算终端速度，而不是从 refine 后两个质心求差。
同步重建 history motion features、local tubes/source masks、KTA 和 Strong prior。
缺少 predicted footprint 的历史帧不伪造 valid 标记。
第一段 yaw 已体现在第二段形状和历史语义中；没有额外添加 yaw-rate 外推头。

这不同于 V19 失败过的 pure persistent-source replacement：检测几何仍 authoritative，
只修正明确对应 source 的运动输入。仍不能保证无负增益，需要真实数据验证。
它也不能消除第一段本身的 XY/yaw 误差、source 截断或未见过的物体预测困难。

## 一次比较，不逐次试超参数

`run_p0_f9_joint_long_handoff_compare.sh` 固定以下四条路由：

- `redetect`：原完整模型预测重新提取组件/速度。
- `reconciled`：同样的完整预测形状，加上述因果身份/轨迹接续。
- `transport_history`：第一段最终输出仍为完整模型，但第二段只反馈纯搬运历史。
  这是隔离补全反馈的机制消融，不是等价的完整历史方案。
- `clean_E14_native6`：同窗口、原生六历史 E14，自身六预测帧反馈到第二段。
  明确是六历史对照，不冒充 matched-four-history baseline。

三个 Joint 路由共享第一段；reconciled 复用 redetect 的组件提取；
所有路由共享未来 GT 计数与 Moving support。未来标签只在全部路由预测完成后读取。
E14 两个额外输入必须是连续过去帧，第一次将 rebuilt E14 与原 cached forward/renderer
逐 voxel 对照。第二段 E14 不用 Joint 输出，也不使用未来 GT history。

所有路由使用同一个 frozen population identity/order；报告每个 1–6s 的
mIoU/IoU/MovingMacro/MovingMicro 及累计 raw counts。输出每窗口关联审计，
包括 matched、split、merge、缺失 source、速度修正、原点偏差和安全回退数。
固定阈值 0.5/0.5/REMOVE-off，不挑 dev 最佳参数，不自动部署获胜路由。
这里比较的是可实现的冻结推理候选，不是数学意义的无 GT 上限。
若据 dev64 选择了路由，随后 dev512 仍是开发验证；不要称为独立最终测试。

## 服务器运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_joint_long_handoff_compare.sh dev64
```

先用固定 scene-balanced long-dev64，一次得到全部路线。没有训练或重建缓存。
需要扩大时将 `dev64` 改成 `dev512`；只有其中完整六秒的窗口进入评估（此前为410）。
第二个参数可指定新的输出目录；使用同一输出目录加第三个参数 `--resume`
会从完整窗口边界接续所有路由的计数。路由/权重/参数/population 变化拒绝续接。
旧默认单路由调用和旧单路由 evaluation_state 保持兼容。

可设 `LONG_FEATURE_BACKEND=gpu` 使用已有可核验 GPU 特征采样器；默认 CPU 路径保留。
不把需要边界回退的 GPU 后端假称为全 GPU。
