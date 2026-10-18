# 最新 Integrated 2+2 固定三断点融合

仅融合本次完整 TRAIN20430×2 同一训练的 update 8000、10000、10292(epoch2)，各 1/3。
只对 named_parameters 用 CPU FP64 累加再转回原 dtype；buffer 必须逐元素相同，只复制，不平均 Adam/RNG。
保留 KTA query+skip、无 ICP、共享 pinned 身份、历史相对 yaw/direct memory、三块 AdaGN；不引入旧模型或后处理。

输入必须是 `source_scene_integrated_full` 已完成输出：`step_0008000.pt`、`step_0010000.pt`、`epoch_0002.pt`、`results.json`、`contract.json`。
校验同一 contract、更新游标、模型深度/配置、全 TRAIN、same remaining64、dense-valid、数据/实现哈希和 STC 文件统计。
原三个断点的四路已完成报告直接复用；只对新 mean3 单卡单进程重新评估 OCC/STC（全部六帧输出，1/2/3s full-grid＋Moving）。
GT future ego 为显式条件；未来占据/Moving 标注只用于预测后的统计。新目录保存评估专用 `mean3.pt`，不能当训练断点续训；原文件不覆盖。

## 服务器运行

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD
# 若 INTEGRATED_FULL_DIR 还保留着刚才的训练目录，直接复用。
# 未设置时只自动接受唯一已完成 scene_integrated22_full_2ep_*，多于一个会停止，要求显式指定。
export INTEGRATED_MEAN_GPU=0
unset INTEGRATED_MEAN_RESUME
export INTEGRATED_MEAN_DIR="$PWD/outputs/p0_f9_shared_history/scene_integrated22_mean3_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_integrated_average/run.sh > "${INTEGRATED_MEAN_DIR}.log" 2>&1 &
tail -f "${INTEGRATED_MEAN_DIR}.log"
```

完成后 `cat "$INTEGRATED_MEAN_DIR/summary.txt"`。列出原三个断点与 mean3 的 OCC/STC Transport/Refiner，以及相对 8000、epoch2 的差值。
中断后设回相同 `INTEGRATED_MEAN_DIR`，`export INTEGRATED_MEAN_RESUME=1`，再运行相同入口；保留完整窗口整数计数，不重跑已完成 modality。

这是固定配方参数平均，不是预测投票、不新增训练、不搜索系数、不自动选模/扩大实验，不保证涨点。
同64已反复用于开发，不是独立测试或全集证据；原缓存评分与新实时推理不声称逐字节一致；非 FPS。

本地验证：新融合测试6项（含真实CPU/CUDA同口径计数与恢复）＋原full2+2回归4项，共10 passed（17.53s）；Bash语法检查通过。本地无最新三个服务器权重，这不是实际mean3质量报告。
