# 固定原两帧身份，统一四帧track：只新训一条

同一独立分支，不开新tracking设计分支；新增tools/pinned_source_track，不改已完成baseline/unified_fit或旧screen源码/指纹。用户允许只做最后一项同预算验证；不把平均速度稀释当作已证实退化主因。

## 与之前unified_two不同

旧unified_two会先用两帧速度引导四帧重新关联t-1，再取最近两个有效点，采用实际时间；不等价于保原KTA。本版严格原t-1/t0同类别mutual-nearest的XYZ距离/25m/s/.5s，只算一次，直接保留选中索引。四帧track的最后一段锁定，不再关联t-1；只延伸t-2/t-3，沿用此前older关联的一对一/0.4m歧义保护/缺帧接续，不增新shape、加速、平滑、静止或类别阈值。

KTA严格原最后两帧差分/名义.5秒，若原最后一对未匹配，速度仍零，即便更老帧有匹配；不退回最近两个有效点、不做多帧拟合。TRAIN缓存record、anchors/KTA直接保原对象/bytes，不重新编码；仅数值核验原KTA属于旧nominal pair规则（兼容原anchor-upgrade微米舍入）。STWM/ICP/feature memory引用唯一新track表，ICP仅空间配准，不选择身份。ICP失败仍不供空间特征，不把匹配与registration可用性混淆。

OCC/STC评估使用独立PinnedPredictor，四帧组件各提取一次→pair一次→older-only track→所有消费方共享；不先调用旧four-frame prepare。Strong复用同一pair速度，不再匹配；仍是原Strong算法和KTA数值，只有历史对应改变。本地六帧Strong与原builder parity检查，不通过不冒称等价。TRAIN继续只读旧Strong背景，因为原pair数值完全保持；旧cache原有source构建不重写，新older关联/配准/reads在有界RAM现场构造。

## 预算和baseline复用

只新训unified_pair：RANDOM同初始化、原TRAIN4086×1、同seed/order/B4-source128/六帧CE+Lovasz/整轮cosine；无ego头/运动existence辅助，无baseline重训、不跑unified_fit。读取刚完成matched目录的workflow和baseline结果/last校验，不导入其参数、Adam或RNG。新模型初始化hash在第1次optimizer前必须与baseline一致，TRAIN keys/order与更新数也在训练前核验；最终科学契约除protocol/implementation全部一致。

新训练最终同DEV64 OCCGT/STCGT、fullgrid/no mask/alignment、Transport/Refiner及Moving全报告，baseline用原已完成数字，不根据分数挑权重。单seed/DEV64不足证明稳定收益，近似持平也不自动升级正式模型；不改变旧完整GT模型或已跑结果。

REFERENCE原完整结果与权重SHA绑定，新输出独立root/unified_pair，Adam/RNG/游标整update及eval整窗恢复。已完成resume只重出对照，不多训；不接受原baseline/fit断点。未显式指定reference时按目录时间找到最近完成同协议baseline，不按分数；首次路径打印并冻结，resume用已绑定路径，不换成后来的目录。PINNED_REFERENCE_DIR可明确指定原TRACK_TRAIN_DIR。

## 本地验证

51项本地回归通过，21.62s，包括原GT/冻结screen/前一轮matched训练与本次6项专项。实际CPU/CUDA小图检查：原mutual-nearest速度逐元素一致、最新一对不被歧义重匹配推翻、未配对KTA仍零、四帧只提取一次/最后一对只匹配一次、Strong六帧背景与原builder字节一致、缓存KTA/anchors对象不变。实际小网格baseline完成后只新训pair、部分update恢复至同一轮末、source文件bytes不变、外层入口复用来源/同目录完成恢复不再训练、初始化不符在optimizer前拒绝、临时模块绑定异常恢复均通过。CLI help/Bash通过；不是nuScenes精度、L40S吞吐或最终方法采用证明。

## 运行（代码发布后）

```bash
conda activate OccFM
cd /root/nas/occ/swfm-unified-source-tracks
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git experiment/unified-source-track-kta-dev64
git merge --ff-only FETCH_HEAD
# 如原终端仍保留TRACK_TRAIN_DIR，可显式指定；否则入口自动定位并打印。
# export PINNED_REFERENCE_DIR="$TRACK_TRAIN_DIR"
unset PAIR_RESUME
export PAIR_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/pinned_pair_20pct_1ep_$(date +%Y%m%d_%H%M%S)
bash tools/pinned_source_track/run.sh
cat "$PAIR_DIR/summary.txt"
```

中断后指定原PAIR_DIR，`export PAIR_RESUME=1; bash tools/pinned_source_track/run.sh`。root summary会同时列旧baseline与新unified_pair；没有“第二组又开始一轮”。不代启动服务器、不自动扩训、调参或推广。
