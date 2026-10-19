# 新三项联合候选：只读ego Query贡献检查

入口 `tools/source_scene_ego_check/run.sh`。只接受已完成的新 `scene_ego_query_1ep_*` RANDOM TRAIN4086×1 checkpoint，不接受旧Integrated/full epoch2/mean3/旧ego头。

## 比较什么

- 同一个冻结 `epoch_0001.pt`，同一ordered remaining64，OCC/STC均dense-valid，GT future ego条件。
- ego_on：保留训练后的ego projection。
- ego_off：临时hook将 `ego_future_proj` 的整个**输出**替换成同shape/dtype的零，包括学到的bias；不是改输入pose、不是输入零向量、更不是修改或保存模型权重。
- 每个modality/window共用同一历史F/reads/lift、KTA、同一份fresh Strong背景和同一个future pose tensor；两个transport读出分别计算，Memory继续纯历史内容＋预测运动对齐，Refiner/坐标/阈值/精度不变。
- 两个modality、两个模式、完整六帧全部预测完才读取future occupancy及Moving标注，full-grid统计1/2/3s。
- 前固定8窗/modality额外重复ego_on，记录原版byte失败窗/最大差异体素。所有窗口都保留，失败不豁免、不放宽阈值；这不是FPS/bitwise parity声明。

输出：八条Transport/Refiner指标、on−off、fresh on−训练时原报告（同权重不同执行路径的sanity检查）；embedding范数、Q变化、世界空间source XY/yaw变化、历史context/yaw是否保持一致、语义差异体素、重复漂移。

此检查衡量**这个已训练checkpoint对ego分支的依赖**，不是重新训练无ego模型的公平消融。关闭后变差说明当前模型依赖它，但不能独自证明ego信息提高了可学习上限；差异近零也不等于充分训练后永远无用。不进行系数搜索、route/epoch选择、训练、自动推广。

## 服务器单卡运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
export EGO_CHECK_GPU=0
unset EGO_CHECK_RESUME
# 复用 EGO_QUERY_DIR；未设置时只接受唯一已完成的 scene_ego_query_1ep_*。
export EGO_CHECK_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_branch_check_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_ego_check/run.sh > "${EGO_CHECK_DIR}.log" 2>&1 &
tail -f "${EGO_CHECK_DIR}.log"
```

进度 `EGO_BRANCH_CHECK 8/64`。完成 `cat "$EGO_CHECK_DIR/summary.txt"`。
中断时设回同一个EGO_CHECK_DIR，`export EGO_CHECK_RESUME=1`，同入口恢复；state.json只保存完整paired-window整数计数，不能混用其他源/实现/人口。已完成恢复不重新预测。

检查源报告/weights SHA、数据/训练实现fingerprint和STC库存；没有创建历史缓存writer、不载入/运行Adam、不写checkpoint。新结果独立目录，原文件不覆盖；结束核验权重fingerprint、无梯度且临时hook已移除，即使异常也移除hook。

本地验证：新6项＋原三项候选/Integrated/Full回归26项，共32 passed（26.29s）。真实CPU/CUDA冻结forward、零输出hook异常清理、同pose/背景、初始零分支一致性、完整pair先预测后读标注、整数恢复/重复不再预测、source SHA和summary符号均检查。精确结构身份测试仅在既有deterministic审计上下文中进行，不改变服务器普通BF16执行。真实科学权重仍在服务器，尚无本次on/off质量结果。
