# Clean Joint Surface CCR：固定权重平均与 DEV512 对照

## 本次固定方案

完整随机联合训练已完成 20 轮。本次不重训，也不改阈值或网络：

- 按已保存 DEV64 Joint mIoU 选出第 5、6、8、12、14 轮，全网络参数等权平均。
- 同一 DEV512 比较第 6、8、12 轮和这一个平均模型。
- 第 20 轮只引用原 `training.json` / `last.pt` 中已经完成的同人口结果，不重复推理。
- 报告 IoU、mIoU、MovingMacro、MovingMicro，以及 1/2/3s、Transport、各分支、类别和场景结果。

这是基于开发集选权重的实验，不是独立测试，也不保证平均优于单轮。
这些 checkpoint 来自同一次训练，但第 5 到 14 轮间隔较大；参数平均仍可能损害效果，因此必须与优秀单轮对照。
平均结果是一套网络，不是五次预测的 ensemble；不能直接沿用旧模型的 FPS。

## 原文件安全与选择检查

入口跨续训目录查找 `epoch_XXXX.pt`，只接受完全一致的训练契约、模型结构、已完成 epoch cursor、DEV64 报告和 TRAIN 正权重。
同轮重复文件的权重不同则报错；缺文件不替换、不给半轮 `last.pt` 冒充整轮。
如果实际 DEV64 top5 不是预定 5/6/8/12/14，就停止，而不是自动改变方案。

所有可学习参数（Transport、静态和动态 CCR）在 CPU float64 等权平均后回到原 dtype。
正权重等固定 buffer 必须完全相同，只复制，不平均。当前网络没有 BatchNorm；若以后加入，不静默平均其运行统计。
保存前后核对所有原 checkpoint 的 SHA256，不写原 optimizer、RNG、缓存或实验报告。

新权重文件标记 `evaluation_only` / `resume_allowed=False`，不包含 optimizer 或训练 RNG。
不能拿平均文件去原训练入口断点恢复。

## 服务器运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_joint_surface_checkpoint_comparison.sh
```

默认完整训练锚点：

```
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787
```

默认只读 VAL 历史缓存：

```
/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1
```

若服务器路径不同，用 `SURFACE_COMPARE_RUN` 和 `CCR_VAL_CACHE` 覆盖。输出为新的 `checkpoint_comparison_时间_PID` 目录。

## 评估与中断恢复

按窗口评估四个候选，共享历史加载、只读因果几何、历史表面表示和 Moving 评估区域。
每个候选单独运行自己的运动模型、实时投影、完整 CCR 和合成；不复用另一个模型的 learned pose、feature 或预测。
保持 4 历史→6 未来、weighted ADD raw sigmoid@0.5、REMOVE-off。
核对原数据 SHA、人口、Torch 环境、配置和训练关键实现；不兼容就停止。

每 8 个“四个候选均完成”的窗口保存整数累计计数。
Ctrl+C/SIGTERM 可停止；强制退出只能回到最近完整保存的组边界。
接续必须指定**本次对照输出目录**，不是原训练目录：

```bash
SURFACE_COMPARE_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/checkpoint_comparison_本次时间_PID \
SURFACE_COMPARE_RESUME=1 bash tools/real_motion/run_p0_f9_joint_surface_checkpoint_comparison.sh
```

接续核对权重、人口、执行方式和实现指纹。已完成的 comparison 拒绝覆盖。
中断可能重做最后不足 8 个窗口，不会混入只完成部分候选的计数。

## 结果文件与边界

- `dev64_audit.txt`：原 20 轮四指标及固定平均配方。
- `bundle.json`：原 checkpoint 来源、SHA、配置与平均来源。
- `epoch_0006.pt` / `epoch_0008.pt` / `epoch_0012.pt` / `mean_top5_dev64_mIoU.pt`：新评估权重。
- `summary.txt`：五个候选四指标、各时距、Transport 和计时。
- `comparison.json`：分支、类别、场景、编辑质量及相对 epoch20 的差值。
- `evaluation_state.json` / `progress.jsonl`：本次评估中断状态和进度。

DEV512 已用于研究/选择；不能把均值或选优分数称为独立泛化证据。
不启动 full4369、训练、阈值/融合系数搜索或自动部署。
计时是多模型质量评估吞吐，不是 Dense Forecast FPS。
本地模拟测试不能替代完整 nuScenes 服务器精度验证。
