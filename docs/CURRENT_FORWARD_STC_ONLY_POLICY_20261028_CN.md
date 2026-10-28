# 同一epoch2权重：OCC原版，STC专用适配

用户希望GT输入不启用去抖动/距离校准，STC输入启用，保留GT得分。新入口为 `tools/causal_forward_stc_only/run.sh`，不修改原baseline和两模态共同适配的candidate入口。原源checkpoint SHA固定为 `7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0`，2816122参数权重共用，无训练或重新拟合。

| 输入来源（预测前指定） | 历史中心/速度估计 | FREE输出校准 |
|---|---|---|
| OCC/GT四帧历史 | 原始forward关联及KTA，完全沿原baseline推理 | 关闭 |
| STC四帧历史 | 已有固定candidate：原值与≥3帧直线拟合各半混合 | 已有固定candidate：FREE加−2+min(t0水平距离/40m,1) |

输入适配器依据调用方声明的occ/stc来源给历史标记；关联只接收原三项历史数组，标记不作为网络特征。在每次forward前选择原baseline或原candidate方法，去掉来源标记，网络权重不切换。没有按窗口预测质量、标签、Moving或GT未来占据决定是否启用；来源缺失/未知直接报错，防止静默误用。GT future ego仍统一明确用于两路线Decoder和renderer。

这是同一网络权重下的两种输入专用推理配置。论文应明确STC专用适配，不能宣称两模态完全相同推理策略。STC距离校准的两个系数原先使用了32个VAL场景的GT未来标签，仍属于有监督目标域输出校准；新切换不消除这部分标签使用。原始未经目标域校准结果和校准结果保留。

用户已有两次ALL4369最终Refiner分数可分别对应下表；这是不同既有运行的记录，新入口尚未在服务器重新完成一次全集：

| 既有运行/输入配置 | mIoU | IoU | MovingMicro |
|---|---:|---:|---:|
| 原epoch2 / OCC原版 | 48.711339 | 62.809965 | 30.593620 |
| 固定candidate / STC适配 | 19.073191 | 29.146352 | 20.090677 |

新入口单卡双进程，OCC/STC完整4369/150场景，各自完整窗整数confusion后合并，正确记录两个input_policies和共享源SHA。新protocol与独立输出目录拒绝混入旧统一candidate的状态文件，不会修改既有score/cache/权重。不需要重新训练，也无需因为切换而重新搜索系数或重复整个开发流程。

如需一次新的统一运行记录，服务器执行：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_FULL_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351"
export STC_ONLY_GPU=0
export STC_ONLY_RESUME=0
export STC_ONLY_DIR="$PWD/outputs/p0_f9_shared_history/occ_raw_stc_adapt_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_stc_only/run.sh >"${STC_ONLY_DIR}.log" 2>&1 &
tail -n 25 -f "${STC_ONLY_DIR}.log"
```

结果 `$STC_ONLY_DIR/summary.txt`；完整窗中断恢复使用原目录与 `STC_ONLY_RESUME=1`。21项新旧回归38.56s通过，含CPU/CUDA真实小模型六时距、GT/STC交替调用对应原方法、共享权重和历史读取不变、来源拒绝、实际指标引擎中断续评、单卡双进程与整数合并/策略混入拒绝。两CLIhelp和Bash语法检查通过，原模型/训练/两个既有全集入口零diff。

另用下载的原2816122参数和真实2窗GT＋2窗STC核对旧路径和新入口：Transport均0差异；Refiner第一次最多28体素差异，带原重复对照时最多30体素差异，原路径重复也最多30体素。严格byte parity仍为False，未修改容差或丢弃窗口，不能将这些检查声称为逐字节等价；CPU/CUDA小模型单元路径精确对照通过。真实检查不读未来GT占据，不产生质量分数。收据保留在本地outputs/forward_local_20261028/stc_only_policy_smoke和stc_only_policy_with_repeat。没有本次新全集或FPS结论。
