# 四帧向前关联：冻结权重、免训练 A/B（2026-10-25）

## 范围

独立入口 `tools/causal_forward_track`，不改原训练、模型、Stage2 关联、缓存或默认评估。没有优化器、参数更新、形状代价、ICP、卡尔曼或额外网络。不会自动扩训、选阈值或切换默认。

支持显式指定当前 FPN/U-Net 或 AdaLN-Zero 的已完成 20%×1／完整两轮 checkpoint，按其协议严格加载；不搜索“分数最好”的权重，也不回退旧 Surface/norm192 模型。服务器若正在训练 AdaLN，只有已完成且通过加载契约的 checkpoint 可用；实验不修改该训练任务。

## 原关联与新关联

- baseline：原最近两帧同类、3D 质心互为最近邻＋25m/s配置门限，固定身份和两帧速度，再向过去扩展。
- forward：按最早到最新顺序处理四帧。每帧先匹配有速度轨迹，再处理无速度轨迹；各组连续观测优先于短暂失配。
- 有速度：在共同世界坐标中按真实历史时间差恒速预测，用 XY 距离做带门控的全局最近对贪心；不同类或超过物理／预测距离范围的边预先禁止。
- 无速度：速度未知不等于静止；用同类、距离门控、互为最近邻初始化／重连，成功后才得到速度。
- 当前原型复用现有最大速度配置；不复制 CenterPoint 检测器的类别门限，不乘1.5、不禁止横向运动。沿用旧较早历史扩展的0.4m歧义差距，作用于新关联的候选行；没有在DEV拟合门限。
- 失配轨迹在本四帧窗口中保留，允许跳帧重连；不生成缺失帧的位置、组件或占据点。最终按原始 t0 组件顺序输出，只在历史出现的轨迹不变成新的当前几何。
- 关联速度使用真实历史时间差；最终 KTA 仍仅由所匹配的 t−1/t0 差分、原名义0.5秒计算。缺少 t−1 就不给 KTA 速度，不用更早轨迹代填。

这是一组明确的关联策略变化，不是仅替换求解器或无损执行优化；第一对仍可能错，片段/合并问题仍存在。

## 推理接线与因果性

每模态每窗只提取一次四帧 CC，两路共享完全相同的组件集合、体素和 t0 顺序；各自独立构建身份、KTA/anchor、STWM读取位置和历史 memory。每一路 Strong/CLEAR 都用本路速度重新计算，所有未来几何用同一条显式 GT future ego。网络共享同一组冻结权重，没有只改配对表却复用旧背景的混合。

关联输入仅四帧历史占据、历史有效性、历史 ego poses/timestamps；未来占据、实例身份、GT可见性都不进入匹配。OCC/STC 均 dense-valid。两模态×两策略都完成全部六帧预测之后，才读未来 GT 占据和（服务器）真实 Moving 标注评分。输入的未来 ego 是本实验明确声明的 GT 条件，不是 navigation-free/Pred 结果。

本地 H1 仅减少一次读出的 horizon 数，仍全部六帧；服务器默认 H6。不是跨平台逐字节一致性/FPS承诺。STC age0.5 可显式 `--stc-age`，若开启，两路完全相同；默认关闭，不自动套到 AdaLN。没有 FREE−1 或 norm 校准。

## 已完成本地结果

本地可用的较新真实权重是用户提供的完整两轮 **FPN＋U-Net（additive ego）**，源SHA `cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f`，**不是尚未下载的 AdaLN 两轮结果**。

只复用旧 portable ZIP 的真实 GT/STC 数据，不加载其中旧模型。原64个key与当前权重的 eval64 对齐；同GT future ego、dense-valid、STC两路相同age0.5，未使用未来标签确定关联。固定前16烟测后按相同算法跑完64，没有据此调参。

最终同64、1/2/3s全网格平均：

| 输入／最终Refiner | 原关联 mIoU / IoU | 向前关联 mIoU / IoU | 增量 pp |
|---|---:|---:|---:|
| OCC＋GT | 50.863900 / 63.829585 | 51.078689 / 63.906899 | +0.214789 / +0.077314 |
| STC＋GT | 18.279316 / 28.620714 | 18.309313 / 28.641654 | +0.029997 / +0.020940 |

Transport 增量：OCC +0.233761mIoU / +0.063332IoU，STC +0.021105 / +0.001404。

OCC共1313个t0 source中81个最近两帧配对改变，279个历史索引改变；STC共706个中25个最近配对改变、112个历史索引改变。这是变化数，不是正确率。portable没有实例Moving支持，不造Moving分数。耗时187.7秒为整段质量检查墙时，不是FPS；同时的小型回归测试也可能竞争资源。

结果文件：`outputs/forward_track_20261025_fpn64_v2/{summary.txt,results.json,contract.json,progress.json}`。前16结果在 `forward_track_20261025_fpn16_v2`，最终mIoU增量OCC+.688483/STC+.107064，不能只报前缀而隐去64增幅缩小。

结论：免训练接线可运行，OCC小幅正，STC增幅很小；不证明稳定提升或匹配正确率提高，不替换默认，不外推全集或AdaLN收益。没有因此启动新训练/全集。

## 来源与回归

权重、输入ZIP和参数前后指纹核验；原训练/推理源码零修改。权重记录中两项训练调度文件与当前不同（`joint_training_distributed.py`、`joint_training_recovery.py`），明确记录旧/新SHA及本实验不执行这些训练函数，其他记录的推理源码仍严格拒绝不匹配。配置仅允许精确CRLF→LF规范化后与原SHA相等，不能绕过实质内容变化。

本次新实验与原AdaLN/FPN评估合计32项测试通过，含实际CPU/CUDA六帧、四种协议加载、初始未知速度、交叉、缺帧、旧baseline准备一致、两路各自fresh Strong、未来输入拒绝、完整窗停机/恢复、无优化器和参数不变。CLI帮助、Bash语法与原模型文件零diff检查也通过。没有安装依赖或使用Anaconda base；本地安全yoloe＋已有隔离pytest/OccFM源码。

## 后续可选固定读出（2026-10-25）

新增显式`--free-minus1`：每个模态/关联的同一次网络前向同时报告raw Refiner和固定FREE−1，Transport不变；不启用时原8路、默认行为不变，启用时额外4路。原系数来自此前FPN/STC前32的GT标签选择，本次不拟合、不自动加age；OCC/STC同系数，所有六帧读出后再评分。合同记录系数来源与实现SHA，raw-only旧状态不能混成含校准状态。当前A/B/C单卡工作流显式启用它，不重复网络forward；旧本地64质量结果仍是未启用此选项的结果，不追改旧分数。

## 服务器命令（代码同步后；只评固定64，不训练）

明确选择你想验证的当前checkpoint，不能把下面占位路径原样执行：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export TRACK_CKPT="/实际当前模型目录/epoch_0002.pt"
# 若使用已完成的AdaLN 20%一轮pilot，则明确指定它的epoch_0001.pt。
export TRACK_DIR="$PWD/outputs/p0_f9_shared_history/forward_track64_$(date +%Y%m%d_%H%M%S)"
export TRACK_GPU=0
unset TRACK_RESUME
nohup bash tools/causal_forward_track/run.sh > "${TRACK_DIR}.log" 2>&1 &
tail -f "${TRACK_DIR}.log"
```

如要匹配已有 FPN age0.5 路线，在脚本后加 `--stc-age`，两策略同时启用，不扫其它组合。查看 `cat "$TRACK_DIR/progress.json"` 和 `cat "$TRACK_DIR/summary.txt"`。恢复必须使用原 `TRACK_CKPT`、`TRACK_DIR`、相同参数并 `export TRACK_RESUME=1`；仅从完整的两模态／两策略窗口恢复，原累计整数counts与key严格检查。结果不是全集或FPS，也不自动选择新关联。
