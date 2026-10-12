# Refiner测速v2：不按一致性筛样本

用户明确授权修复测量方式，不修改预测算法/权重/loss。旧严byte FAIL保留，不变相放宽容差。

原audit32窗中28计时、4在warmup输出差异后被跳过；6窗不同，最多Refiner4体素、Transport0。全6帧384万体素，max约0.000104%。原147.46ms来自28窗子集，不能声称完整32窗或正式FPS；可能是argmax临界漂移，但未证实归因BF16/归约。

## v2行为

- 每个选定窗口都完成warmup、3次完整实时推理及额外serialized profile，byte差异不提前return、不筛掉窗口。
- 计时人口完整/重复次数/finite时延与严格byte parity分开。`timing_complete`要求所有预定窗口/重复均有真实完整6帧计时。
- `six_frame_byte_parity`和`valid_same_quality_route_timing`仍保留原严格意义；任何差异/原分数非精确复现都不自动变PASS。
- 完整时延可报告`window_fps`/`output_frame_fps`，标记WARM actual-execution diagnostic。byte失败时不称bit-identical-reference/无损FPS。人口不全、CPU或时延非finite仍FPS=null。
- 不容忍形状/标签契约错误、nonfinite logits/错误KTA；这些是结构/数值异常，不是微小输出漂移，仍失败并保留先前完整窗口。
- 差异每时距计数与差异体素（最多16个）reference/fresh top2 margin及max logit差均在计时外采集，不读GT挑模型/阈值，不设新的自动数值接受门槛。
- allocator峰值包含不计时debug/profile保留，公开说明，不包装成独立最小部署显存。

## 这次只补32窗，不重评DEV512

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/shared_history_experiments/refiner_diagnostics/retime.sh
```

默认原audit `outputs/p0_f9_shared_history/refiner_epoch2_audit_20261012_204921`；显式其他完成audit用`REFINER_AUDIT_DIR=/实际audit目录`。只接受完整epoch2/DEV512、source snapshot SHA、population、原整数metric复算与edit计数一致、原Torch/CUDA/NumPy/fresh helper字节相同。只读取原固定32行raw/history并实时前向，未选的480行不做prepared replay、没有Moving/GT质量重新评分。

输出新`refiner_epoch2_retime_TIMESTAMP`，旧result/state/contract/权重不写。完整窗口后保存；中断恢复：

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/原refiner_epoch2_retime目录
RETIME_RESUME=1 bash tools/shared_history_experiments/refiner_diagnostics/retime.sh
```

summary/speed保留完整人口与严FAIL，results.evaluation保留旧质量/编辑并用新完整32计时替换其speed字段；旧基线文件原样。部分窗口旧偏置计时不导入；普通原audit入口变v2，旧audit不要用AUDIT_RESUME恢复，请使用该retime入口或另开全新audit。

本地测试包含少量/大量warmup差异仍计时、完整与28/32人口、GPU/BF16小模型、logits flip/非finite/shape拒绝、原整数计数校验、32行预取而非512、源文件不变、时间窗口恢复；不等同真实L40S已完成补测，也不扩大训练或直接推广新算法。
