# 最新 epoch2 本地 STC 改进：小包导出

用户目标：在当前 forward AdaLN-Zero / FPN / U-Net epoch2 的基础上尝试 STC mIoU≥19、IoU≥29，允许 OCC+GT 适度下降。当前 ALL4369 原始最终 Refiner 为 STC 18.721545 / 28.446166 / Moving18.725074，OCC48.711339 / 62.809965 / Moving30.593620。

本地已有 `stc_overnight_bundle.zip`（57,875,829字节）：原64窗真实STC/GT历史及GT未来，另有32个TRAIN窗口。本轮只复用其中数据，不加载其旧模型；不重复传数据。新导出包只有当前 epoch2 权重、原始配置/来源、原同64报告；排除Adam/RNG和原始传感器数据。

旧本地数据包没有Moving实例标注；本地筛查只报标准mIoU/IoU，Moving需后续服务器验证。

服务器已激活 OccFM 后：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
export FORWARD_LOCAL_EXPORT_DIR="$PWD/outputs/p0_f9_shared_history/forward_epoch2_local_pack_$(date +%Y%m%d_%H%M%S)"
bash tools/causal_forward_local/export.sh
```

若 `FORWARD_FULL_DIR` 已设置则用它；否则只接受唯一已完成、协议相符的 `adaln_forward_full_2ep_*`。存在多个候选时，显式设置为本次原训练目录，不按日期或分数自动选。

下载最后打印的 `DOWNLOAD_THIS_FILE` 路径对应 ZIP，放入本地工作区 `_received`。实际体积和 SHA256 由脚本打印。CPU导出，不训练、不评估，也不修改源权重或缓存。

本地 `tools.causal_forward_local.portable.load_weights` 严格加载本次2816122参数模型及训练来源；允许Windows CRLF与服务器LF的纯换行表示差异，其他源码变化仍拒绝。权重payload和原checkpoint的SHA分别记录，不混淆；新权重包只用于推理/独立适配研究，不能用来续原训练。

下载后先对齐同64原基线，再做小规模改进筛查，同时报告GT代价。若使用VAL标签校准，明确标注有监督开发并保留场景分开的验证；原64已反复查看，不作为独立测试。新子集有收益仍需服务器固定全集验证，不能将子集19/29声称为全集达标。不得继承旧clean1000/norm192/half-KTA结果或权重。
