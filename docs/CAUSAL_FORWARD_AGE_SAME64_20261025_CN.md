# 当前前向关联 AdaLN：冻结历史Memory衰减开／关对照

## 固定实验

源为本次已完成的C：`adaln_forward_single_free20_20261025_161644/C_train/epoch_0001.pt`，AdaLN-Zero/FPN/U-Net、前向关联、TRAIN4086×1、1031更新。严格复用完成断点加载器与来源检查，不调用训练循环、不创建优化器，不加载旧FPN或其他试错权重。

只评原契约的有序64窗，单GPU/单模型进程。OCC和STC各做衰减关闭／开启，以便评估一套配置是否能用于两种输入。GT future ego用于Decoder/render，dense-valid，全网格1/2/3s及Moving评分。FREE−1、历史拟合、norm适配和其他后处理全部关闭。

沿用已有 `tools/fpn_unet_stc_age/processing.py` 固定半衰期0.5s：

`weight(t) = 2 ** ((t - t0) / 0.5)`

按实际历史时间戳计算；常规0.5s间隔时从老到新为 `[1/8, 1/4, 1/2, 1]`。不搜索系数。

它仅乘在Memory读取的静态 `static_valid` 与动态 `coverage` 上；融合分子、分母都随权重变化。不是把特征值单独缩小，也不改Reader token/known、关联有效性、KTA、Decoder、source运动或历史yaw。两个分支共用一次Encoder/运动预测/Strong/Transport，开启组只重算Memory和Refiner。Transport共用，因此汇总每种输入只列一次Transport。

两种输入、两路六帧预测全部完成后再读取未来GT占据和Moving标签。源码不调整网络、损失、精度或原数值门槛；复用前向关联恢复适配，兼容旧6-token元数据与实际末4帧输入。它是只读推理干预，不是“衰减参与训练”的消融，也不保证完整版收益。

## 运行

同步本分支后：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_DIR="$PWD/outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644"
export FORWARD_EVAL_GPU=0
export FORWARD_AGE_RESUME=0
export FORWARD_AGE_DIR="$FORWARD_DIR/age_off_on_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_age/run.sh > "${FORWARD_AGE_DIR}.log" 2>&1 &
tail -n 30 -f "${FORWARD_AGE_DIR}.log"
```

`Ctrl+C`只退出日志查看。完成后：

```bash
cat "$FORWARD_AGE_DIR/summary.txt"
```

输出同时包含两模态的Transport、age_off Refiner、age_on Refiner，mIoU/IoU/Moving及逐时距和均值差值。保留正负结果，不自动切换正式默认或启动完整训练。

新输出独立于源 `C_train`、旧B/配对审计、补评目录和缓存；不复制或改写源权重。完整窗整数计数、实际权重、源/输入/实现指纹入状态，每窗保存。若进程中断，保持原 `FORWARD_AGE_DIR`，设置 `FORWARD_AGE_RESUME=1` 后再次运行同脚本即可；禁止跨断点/代码/数据混用状态。

## 本地验证范围

覆盖当前AdaLN CPU/CUDA真实六帧推理、候选不重跑Encoder/运动且不改原tensor/权重/历史读取、等权CPU读出复现原结果、4帧时间权重、旧6-token元数据兼容、原指标精确整数统计对应；实际CLI使用严格完成断点加载，验证64窗完整窗中断恢复/无重复计数、未来标签晚于所有预测、无优化器创建、源checkpoint SHA不变、损害结果仍输出以及无效窗口不提交。

暂无本次真实服务器质量结果。用户此前C原64 raw Refiner为OCC `48.242860/61.103879/34.338730`、STC `17.528190/28.531593/21.240331`（mIoU/IoU/Moving）；本入口重新评估关闭组供同次对照，不将历史数字硬塞入新计数，也不承诺逐字节一致。反复查看的开发64不是独立测试或FPS。
