# 冻结epoch2：编辑质量与完整六帧推理诊断

## 当前服务器结果（用户日志转录）

GT未来ego条件，4历史→6未来；新Refiner独立随机初始化、TRAIN20%4086窗×2、2060 updates、固定DEV512。

| 轮 | 分支 | mIoU | IoU | MovingMicro |
| --- | --- | --- | --- | --- |
| 1 | Transport | 38.213947 | 51.306206 | 27.234976 |
| 1 | semantic_refine | 41.492979 | 57.080235 | 28.246262 |
| 2 | Transport | 39.792319 | 51.556657 | 27.109999 |
| 2 | semantic_refine | 43.170140 | 57.664138 | 27.909835 |

第2轮修复增益+3.377821 mIoU/+6.107481 IoU/+0.799836 Moving；相对第1轮最终输出+1.677161/+0.583903/−0.336427。后者是动态警示，不据此自动选epoch1/扩训/改loss。第2轮最近255窗纯计算0.399315s/窗，非推理FPS。旧Shared ADD-only同两轮DEV512为40.553310/52.881533/28.035021，结构/初始化/执行不同，非严格单变量消融。这里只转录摘要，尚无本地原始counts/服务器HEAD核验。

固定来源默认：`outputs/p0_f9_shared_history/refiner_probe_2ep_20261012_184856/train/epoch_0002.pt`。
新入口独立 `tools/shared_history_experiments/refiner_diagnostics/`，放在refine树外以保持原trainer和递归workflow指纹；不替换训练/模型源码。无服务器执行连接，用户运行命令。

## 一趟集中诊断

- 冻结epoch2/固定DEV512，复现Transport和Refiner全部1/2/3s IoU/mIoU/Moving；与原epoch2 results逐项比较。
- 只根据已完成预测与GT差分统计，不重新引入ADD/REMOVE头、候选/权限/阈值。18类输出的新增、删成FREE、占据重分类互斥。
- 每类编辑报告corrected/damaged/wrong-to-wrong语义计数，另外统计binary占据改对/损坏。invalid GT忽略。
- 分别统计全网格、GT动态语义（不含动态FP）、GT/前后动态类别并集（包含FP、并集是预测相关人口）、原始t0 Moving支持区、Moving支持区×动态并集。后两者是空间编辑计数，不直接等于MovingIoU贡献。
- 6个时距分别存计数，summary额外合计1/2/3s；没有按分数筛窗口、epoch或模型。
- 固定DEV512内scene-balanced32窗×3次暖测速：每窗1次不计时warm与gate、3次完整计时、1次额外串行profile（不计吞吐）。这是小范围延迟诊断，不冒充旧256×3正式FPS。

## 计时边界

历史读取/CC匹配/坐标读取表/初始history H2D在外，Shared Encoder及height lift在内，不能提前缓存learned F/M。现场Strong/CLEAR、现场KTA（包含已验证的旧upgrade精确算术）、STWM、连续Transport投影、学习特征搬运、Refiner、完整六帧argmax/CPU输出都在内。GT配置辅助ego头不参与部署，不计入该计时。投影算子自身临时张量/搬运仍在内，不扣掉执行成本；不是raw-input E2E。

首次每窗fresh与缓存质量两路Transport/Refiner完整六帧逐字节检查；计时内仅输出最终Refiner六帧，不增加Transport诊断CPU读出。每次计时及单独profile仍核最终六帧。不放宽parity，不宣称BF16一定逐字节稳定。失败保留计数，valid=false/FPS=null；原epoch2质量复现失败或无原结果也不发布有效FPS。

报告六帧ms、P90、window_fps与output_frame_fps=6×window_fps，显式区分，旧54.945是输出帧FPS。Strong缓存可以用于质量基线，但绝不用于计时；未来labels/annotations/visibility不进入新fresh函数，只在所有预测后做评分。未来GT ego poses仍是声明的GT条件。

## 只读与恢复

来源checkpoint复制到新输出frozen/epoch2.pt，固定source SHA；仅evaluate-only，原Adam/RNG/权重/缓存不写，不扩大训练。每个完整窗口后原子保存整数metric/edit与完整测速记录，同目录严格resume；已完成前缀不重计。原source结果仍在原目录，summary/speed/edit_audit/results写独立输出。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/shared_history_experiments/refiner_diagnostics/run.sh
```

显式其它来源：`REFINER_CHECKPOINT=/实际epoch_0002.pt bash .../run.sh`。

恢复：

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/原refiner_epoch2_audit目录
AUDIT_RESUME=1 bash tools/shared_history_experiments/refiner_diagnostics/run.sh
```

最终只需回传新目录summary.txt，speed.json/edit_audit.json保留详细计数。候选有效后扩大TRAIN是后续任务，本次不启动、不升级网络/ego、不更改评分协议，也不直接将新指标作为完整论文主表。

## 本地验收边界

本地专项7项与原Refiner/工作流13项共20passed，含真实小网格CPU/CUDA/BF16网络、真实KTA重建与严格输入识别、六帧byte核验、encoder在每次计时内调用、未来GT/缓存Strong毒化、ego辅助不进入GT部署、权重不变、互斥编辑与invalid忽略、完整窗口整数恢复不重计、epoch2快照与协议拒绝、质量复现失败不报FPS。Strong/Moving支持与数据人口使用测试夹具，不代表实际nuScenes/L40S已验收；服务器真实全六帧Strong parity和吞吐仍需运行新入口。无Moving支持的负测试产生empty-slice warning，保留原指标定义未改。
