# 冻结四设置的集中误差归因

入口：`tools/real_motion/run_p0_f9_stc_branch_diagnostic.sh`。只运行冻结 dev64，
使用原 mean5/6/8/12/14、4 历史 → 6 未来、ADD 原 sigmoid@0.5、REMOVE off。
不训练、不调整位姿/历史、不给预测结果事后 GT 对齐、不引入 GT 可见性、不选阈值。
这不是新的改善方法；它检查当前掉分发生在哪个已有环节。

每窗四次完整预测，分别对应 Occ+GT、Occ+Pred、STC+GT、STC+Pred。
同一次预测直接取出 Strong 的完整六帧 `state['anchors']`、learned Transport 的
`prep.baseline`、完整 Joint 输出；不是把稀疏 source 列表误当 Strong，不做额外中间模型 forward。
复用既有共享历史/native Surface 执行及参考六帧 exactness。

## 一次输出哪些证据

- 12 路的 1/2/3 秒和平均标准 IoU/mIoU、18 类混淆整数计数及每类 IoU。
- road/sidewalk、ground、building/vegetation、dynamic、other_static 的 FP/FN、
  组成员 IoU、组内类别混淆。road/sidewalk 包含在 ground 内，不能把这些相加作 mIoU 贡献。
  dynamic 指八个语义类别，不是 GT 实例/速度定义的 Moving 指标。
- Transport→Joint 的新增、移除、重标记、改对、损坏、新增占据/语义精度。
  当前契约不允许移除或修改已占据 voxel；若实际输出违反这一点立即报错。
- t0 STC 全网格输入的错占据、漏占据、语义错误，以及历史有效比例；
  不会把 STC 的全网格有效位冒充真实观测可信度。
- 原 planner 的 XY/yaw/z/tilt 误差，以及因果历史组件的数量、匹配数、初速、线性质心抖动。
  后者是不同感知人口的统计，不是同一个物体配对后的因果贡献证明。
- GT/Pred 两设置的历史-only learned motion 输出逐字节一致检查。
  这样可区分正常的最终投影差异与实际不应存在的 motion 条件差异。
- 摘要列出三种设置转换中下降最大的六个语义类别；完整类别与逐窗信息在 `evaluation.json`。

未来语义仅在四设置、十二路完整预测都完成后读入做评分。
未来 GT pose 只供 GT-conditioned 两设置，或所有预测完成后的误差审计；不反馈给 Pred 分支。
不是“GT aligned”修分，也不重复已失败的历史路面/yaw 适配。

## 原始 planner 起点审计

现有 cache 的 `come_trajectory_sha256` 字段只是声明，不足以独立证明起点映射。
若原始 JSON 不在缓存 manifest 记录的路径，本次明确报告 `unverified_missing_original_json`，
仍可进行分支归因。可以提供原始文件：

```bash
export STC_PLANNER_JSON=/actual/path/bevplanner_ego_in_bev_with_yaw.json
```

提供的 JSON 必须匹配已冻结官方 SHA256。不猜格式、不替换 planner、不下载未知镜像。
按原构建器的 scene 数字后缀排序、完整场景帧 ordinal 找到 t0 对应行，核对：

1. 每个 scene 的原始条目数与完整 metadata 帧数一致，数字索引不重复。
2. 当前行 XY 与 t0 measured pose 误差 ≤ 2 cm。
3. 六未来 XY 与相对当前姿态的**非累积** yaw 独立重建 SE(3)，与 cache 最大元素误差 ≤ 1e-6。

不一致会报告 mismatch；不会自动换行、重估 yaw 或覆盖 cache。
原地/重复 XY 无法单独证明帧身份，所以必须结合原始 ordinal 与六 pose 检查。
即使接口核验通过，也不意味着 planner 的真实预测误差不存在。

## 服务器运行及续评

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export STC_DIAG_OUT="$PWD/outputs/p0_f9_joint_surface_ccr/stc_branch_dev64_$(date +%Y%m%d_%H%M%S)"
bash tools/real_motion/run_p0_f9_stc_branch_diagnostic.sh
cat "$STC_DIAG_OUT/summary.txt"
```

默认沿用此前的数据/compact STC/cache/固定平均权重发现路径；无需下载、解压、重建输入。
支持 `STC_CHECKPOINT` 显式指定同一冻结均值文件。
原始 JSON 缺失不阻塞；显式提供不存在/错误 SHA256 的 JSON 则 fail-closed。

每 8 完整窗口保存整数统计；Ctrl-C/SIGTERM 在完整窗口边界停止并写摘要。
硬中断只能恢复最后落盘窗口。保存相同 `STC_DIAG_OUT`，设置 `STC_DIAG_RESUME=1` 重跑。
契约锁定 population、weights、实现、输入指纹和 SWFM 环境，不能迁移旧四设置结果到这里。
全部旧模型、训练 RNG、数据、失败实验以及 full4219 指标保持只读。

本地单元/实际小模型回归不能代替服务器 dev64 精度。没有自动重训/全集/选优。
