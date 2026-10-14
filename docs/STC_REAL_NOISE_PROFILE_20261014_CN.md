# 真实STC噪声统计：开发集拟合，暂不训练

固定已完成prior-strength DEV64的所属场景作为fit部分（预期18）；其余STC发布场景（预期132）不参与本次拟合，不读其GT/STC语义文件。不是随机拆重叠窗口。fit场景所有发布keyframe各统计一次，按时间链使用四历史做时序统计；读标准metadata/同帧annotation，不预测、不训练、不读forecast未来target接口、不改正式GT/STC评分或模型。CPU NumPy/SciPy，无Torch、SDK、CUDA依赖，默认两个场景并发。

## 输出什么

- 同token未变轴/未重映射全网格STC vs GT混淆；真实GT历史known内外仅分析，不提供mask给STC模型。
- FP/FN/occupied错类的类别与水平距离分布，GT对立占据边界距离。无对立表面时进入overflow，不伪造有限距离。
- 每错误类型、每语义类单独六邻接3D组件；保留单体素错误。joint表保留类别/距离bin/XYZ包围盒extent/voxel数的相关性，不复制真实VAL体素图或世界坐标。axis-aligned bbox不等于真实形状或物体尺寸。
- 四历史ego配准，最近体素中心采样；只在四帧共同FOV、GT类别不变且非动态GT处统计16种错误bit pattern。bit0最旧/bit3当前，不能把GT变化或动态运动叫噪声；GT-free上的假动态占据仍可作为静态世界错误统计。grid量化与条件筛选仍影响统计，不是纯噪声或完整寿命。
- 同帧GT instance annotation box内STC与GT同类体素，在object-local坐标比较centroid残差/支持量比。跨历史用同instance ID的残差变化，扣除实际物体运动，不是把连续帧质心差当噪声。缺STC支持显式计数不填零。GT支持四帧时统计缺失pattern。crop可受邻车、遮挡、边界/形状变化影响，不是预测组件匹配或真速度。

noise_model.json输出经验条件分布及原始支持数/联合表，未实现污染生成器。无支持条件不能当成零错误概率；不能声称只用这些低阶统计就完整模拟STC。动态偏移/时序统计用于后续讨论，未擅自添加多种增广或新loss/gate。

## 数据使用声明

本步骤使用fit场景同帧VAL GT标签校准模拟误差，属于验证集参与方法开发。先前完整VAL分数已被查看，剩余场景只能称“未参与本次拟合”，不是从未接触过的独立test。未来正式训练只用TRAIN样本/未来TRAIN目标；不直接拿VAL网格或实例轨迹训练，不声称完全无目标域信息的zero-shot。剩余场景模型质量评估本轮不跑，需后续冻结配方/同预算干净控制后明确报告。

## 服务器命令

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
export STC_ROOT="$PWD/data/stc_camera/compact"
export NOISE_FIT_DIR="$PWD/outputs/p0_f9_shared_history/stc_noise_fit_$(date +%Y%m%d_%H%M%S)"
unset NOISE_FIT_RESUME
bash tools/stc_noise_profile/run.sh && cat "$NOISE_FIT_DIR/summary.txt"
```

通过PRIOR_DIAG_DIR读取原prior结果；变量不在时自动找最近完整同协议结果（不按分数）。可指定PRIOR_RESULT实际evaluation.json路径。输出打印fit/remaining场景及实际frame数；split.json冻结场景，evaluation.json是完整统计，noise_model.json是后续模拟器输入，不是训练权重。

Ctrl-C最多保留完成的scene，中断scene下次重算；保留原NOISE_FIT_DIR，设置NOISE_FIT_RESUME=1再同入口。不跨scene重复统计帧；恢复核scene/token prefix、源码/原prior/receipt/meta/fit文件stat及整数计数。新输出必须在原数据/结果之外；系统文件锁阻止同目录并发。原模型、权重、optimizer、历史缓存不改、不复制；旧评估/trainer fingerprint不变。

本地小网格/metadata夹具只验证算法与隔离，不提供真实nuScenes噪声分布、运行耗时或涨点结论。用户回传summary后再定固定污染配方与训练预算。
