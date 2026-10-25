# FPN + U-Net：AdaLN-Zero 与 Query 加法的匹配一轮实验

只新增一个候选，不启动主组件/池化/Memory/Pred 的其他实验。原模型、trainer、evaluation、checkpoint 与旧结果不修改。

## 唯一研究改动

保留 FPN、2层 Scene–Temporal Encoder、2层 cross→self→FFN Decoder、KTA Query 和 skip、无ICP/shared-pinned、学习历史yaw、纯历史Memory及普通3D U-Net。删除入口 additive ego 模块，改为每层 Decoder 三个独立子分支的 AdaLN-Zero：每时距ego embedding生成cross-query/self-attention/FFN各自的shift/scale/gate，共9d输出。三个受调制LN不带固定affine，保留原eps；历史K/V的归一化不接受未来ego条件。

调制投影末层weight/bias全零，gate是无界线性输出而不是sigmoid。ego encoder正常初始化，避免双零阻断条件学习。初始Decoder块为identity；这不等价于原additive模型的中间特征。原零运动/Refiner输出头仍保留，初始最终argmax为Transport。此实验比较完整条件注入方案（包括零门控初始化和LN affine差异），不是等参数单一加法位置消融。

参照 [DiT官方代码](https://github.com/facebookresearch/DiT/blob/main/models.py)，不增加diffusion/noise/采样或辅助监督。参数：原2,520,442；候选2,816,122。GT future ego仍是显式条件，Decoder及全部未来几何使用同一轨迹，无内部ego头/导航/Pred运行。Memory没有Q或ego内容注入，ego仅通过运动改变对齐位置。

## 匹配与复用

RANDOM TRAIN4086×1、约1031 updates、seed20261011、globalB4/source128、H6完整CE+Lovasz、原AdamW/LR/整一轮cosine/clip，dense-valid TRAIN/OCC/STC，最终同remaining64评估一次。无age衰减、FREE bias、norm校准、STC训练、motion/existence/ego loss。

默认控制：`outputs/p0_f9_shared_history/fpn_unet_20pct_1ep_20261021_100854/results.json`。只读控制报告，不加载其训练权重/Adam。共同参数从相同FRESH原模型拷贝、额外构造隔离CPU RNG；模型结构/数据SHA、原实现SHA、world size、LR、顺序key/更新数与最终运行设置严格核对，控制不匹配时停止，不自动重训或选择新控制。共同参数不包含已删除的固定LN affine及新的ego/modulation参数。

复用原FPN trainer和四route evaluator：在独立私有模块命名空间载入原循环，只替换该命名空间的模型工厂/协议/报告及匹配检查，不修改原模块全局或复制优化/归约/恢复循环。新代码SHA与控制报告SHA存入新模型实验契约，原trainer已有SHA继续覆盖所有旧依赖。新目录只接受同协议last.pt恢复完整model/Adam/每rank RNG/cursor，最终rank0单卡评OCC/STC；比较正负delta全报，不选best或自动延长。尚无新科学分数/FPS。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export FPN_ADALN_CONTROL_DIR="$PWD/outputs/p0_f9_shared_history/fpn_unet_20pct_1ep_20261021_100854"
export FPN_ADALN_DIR="$PWD/outputs/p0_f9_shared_history/fpn_adaln_zero_20pct_1ep_$(date +%Y%m%d_%H%M%S)"
export FPN_ADALN_RESUME=0
nohup bash tools/fpn_adaln_zero/run.sh > "${FPN_ADALN_DIR}.log" 2>&1 &
tail -f "${FPN_ADALN_DIR}.log"
```

控制原来双卡，候选也用双卡，global batch仍为4。除最终summary标题外，训练与eval进度保留复用入口的`FPN_UNET`/`FPN_UNET_EVAL`前缀；独立输出协议明确区分。

完成：`cat "$FPN_ADALN_DIR/summary.txt"`，包含控制和候选四route及delta。完整科学记录在`results.json`，候选权重`epoch_0001.pt`，可恢复权重/Adam/RNG在`last.pt`。关闭tail不会结束nohup任务。

中断后设置原`FPN_ADALN_DIR`、相同控制和卡数、`FPN_ADALN_RESUME=1`，重新运行同脚本；不换目录、不重启学习率，先确认无同目录旧进程运行。不在此实验完成后自动训全集或评Pred。

## 本地验证边界

2026-10-25启动检查修复：原`record_keys(evaluation)`返回tuple-of-tuples，控制报告JSON读回是list-of-lists，旧检查误判同一有序64窗口不匹配。现在TRAIN/EVAL均仅规范化容器为列表，不排序、不修改样本、更新数、容差或控制；真实差异仍拒绝，并分别打印字段、数量与首个差异索引。该失败发生在构造模型/Adam之前，未训练任何一步，旧结果未受影响。同步修复后新建输出目录运行，不对空失败目录使用resume。

新增生产launch回归直接使用原TRAIN20430/VAL4369规划函数、实际record_keys和JSON控制报告，覆盖同序通过、TRAIN/EVAL换序及更新数/STC清单差异仍拒绝；模型/训练循环不变。此前CUDA恢复测试只覆盖复用循环，未覆盖这层新控制检查，此次补齐。修复后新旧测试合计17 passed（14.50s），Bash语法与diff检查通过；没有代跑服务器科学训练。

安全非base yoloe环境完成16项新旧CPU/CUDA测试：三分支精确算式和历史mask、零门控identity、共同FRESH初始化/RNG及参数量、真实几何/纯历史memory/六帧CE+Lovasz反传、零初始化整网四步后ego学习自然打开、独立checkpoint、复用训练循环的实际CUDA保存/恢复及最终只评一次。早期测试配置类名和autocast夹具问题已修正，不修改生产精度或数值容差。CLI帮助中的百分号转义问题也已加回归检查。

这些是小网格功能/恢复测试，不是本地完整4086窗科学训练或双GPU吞吐实测；没有服务器连接，需运行上面的命令得到实际同64质量。
