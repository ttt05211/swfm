# 原 GT epoch2 全解冻：模拟污染 500 步单路试验

用户决定：从**原始 GT epoch2**重新开始，不从上一轮已适配 Refiner 初始化；Shared Encoder / STWM / SemanticRefiner 全部解冻，固定 500 更新，评估**当前 paired 实验相同 remaining64**，不换回旧 fit18 的 DEV64。

只新增独立 `tools/stc_noise_full/`，旧正式 GT trainer、refiner-only 对照与提速续训入口都不改。

## 固定配置

- 初始化来自 reference workflow 的 `empirical/frozen/GTepoch2.pt`。它是原完整 TRAIN20430×2、update10292 的 GT 快照，严格核协议 / SHA；**不是 empirical/train/last.pt**，不导旧微调权重、Adam或RNG。
- 新 Adam / seed20261011、主体 LR1e-5、Refiner LR1e-4、500更新整周期 cosine floor .01。空 ego 兼容槽无参数，不引入 ego 头。
- 仍原 TRAIN20%=4086 的首500分组、B4/source128/H6、50%历史污染、TRAIN dense-valid、GT future ego条件、未来干净GT监督。只 CE＋Lovász，motion / existence / ego 辅助仍关闭。existence结构可以存在但无直接loss，不声称每个未被目标使用的参数都有梯度。
- c=1/epsilon1e-4不改。纯历史几何缓存继续只读复用；污染后 source/匹配/KTA/Strong/prior/memory重建。学习特征实时生成，不缓存旧冻结Encoder特征。
- 复用有界CPU预取和历史Strong匹配复用；第一真实dirty TRAIN窗完整六帧Strong byte核验。不改GPU前向数学、精度、loss或noise draw；全解冻会增加反向成本，不继承上轮Refiner-only耗时/FPS。

500更新只是部分TRAIN遍历，不称完整epoch。共用driver `RANDOM/epoch1` 启动文字不是实际初始化 / 完整epoch，以独立summary/contract/audit为准。

## 评估与参考

要求上一 paired workflow 的 clean / empirical 都已完成500和64窗。绑定这份workflow参数、统计profile、源GT SHA、STC inventory与确切keys；新选择与上一名单不一致就拒绝，不补窗或重新挑简单场景。

仅新增这一路全解冻训练，最终自动评 OCC+GT、STC+GT 的 Transport/Refiner、IoU/mIoU/Moving，表中只读列上一 clean / refiner-only empirical，并打印新结果相对两者差值。OCC原lidar-known输入、STC完整dense预测语义，完整网格评分，不用GTmask给STC。

新 full empirical vs 旧 refiner-only empirical 主要比较可训练范围（主体LR从未生效到生效）。与旧clean比较**同时改变解冻范围和污染**，不是全解冻下的干净/污染单变量控制；没有偷偷新增一趟 full-clean 训练。Transport现在会学习，不能再要求它与旧两路输出不变。

继续声明VAL校准噪声 / remaining64不是pristine独立test；没有自动扩大 / 选择best / promotion。

## 服务器命令

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only

export REFERENCE_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
export FULL_NOISE_DIR="$PWD/outputs/p0_f9_shared_history/stc_full_unfreeze500_$(date +%Y%m%d_%H%M%S)"
unset FULL_NOISE_RESUME

bash tools/stc_noise_full/run.sh && cat "$FULL_NOISE_DIR/summary.txt"
```

不需要再次下载、生成旧缓存或配置GT_CKPT/NOISE_MODEL。来源是原paired任务绑定的原始GT快照、同一profile；代码在新目录保存独立快照与reference hashes，旧结果/模型不写。

Ctrl-C保存完整update / eval window后，可同目录继续原500周期：

```bash
export FULL_NOISE_DIR="/root/nas/occ/swfm/outputs/p0_f9_shared_history/本次实际新目录"
export FULL_NOISE_RESUME=1
bash tools/stc_noise_full/run.sh && cat "$FULL_NOISE_DIR/summary.txt"
```

`train/last.pt` 为新实验断点；新协议拒绝旧refiner-only断点。只报告固定最后step500，不选中途best。`initialization_audit.json`声明未导适配权重、`trainable_audit.json`记录是否全部requires_grad及主体/Refiner是否更新。

## 本地小验证

遵守 windows-python-env-guard，安全非base yoloe和已有依赖。3项通过（10.19s）：CPU/CUDA真实完整6帧 CE/Lovász 梯度到F/XY，Encoder/STWM/Refiner全部更新、原GT权重不变；实际单路CLI500日程/同64键、1→2 Adam/RNG恢复、新协议与旧目录拒绝、旧报告/统计/source bytes不改。Bash语法/diff通过。没有服务器训练或真实涨点结论，不声称全仓CI / L40S吞吐。
