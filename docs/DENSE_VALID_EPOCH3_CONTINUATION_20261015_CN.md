# Dense-valid 双卡第三轮续训

独立入口 `tools/dense_valid_epoch3/run.sh`，不修改旧两轮 trainer 或其 fingerprint 依赖文件，不覆盖旧目录。前两轮完整结束并退出后执行。

- 自动选择最近完成的 `dense_valid_shared_full2_ddp_*/epoch_0002.pt`，也可用 `DENSE_DDP_DIR` 显式指定旧目录。
- 导入 epoch2 权重、Adam steps/moments、两 rank RNG；按原 seed 的 epoch index2 完整遍历 TRAIN20430 一次，保留全局 B4/source128、双卡/H6/full6 CE+Lovasz、shared-pinned source。
- 学习率从断点**实际**末值开始，在新增一轮内余弦降到其十分之一；不重启初始高 LR，不清空 Adam。不是从头规划的三轮余弦。原末值约为主体5e-6、Refiner3e-6，精确值打印并存入 contract。
- TRAIN/OCC/STC 均为 dense-valid，评估仍为同一 remaining64、full-grid、GT future ego。首先评 source epoch2 的正确 dense 基线，随后每2000**累计** update（通常12000/14000）及最终评估；不是每2000新增 update。
- 每64新增 update 保存 last.pt；评估点保留 update 快照，结束保留 epoch_0003.pt。评估保存分 rank 整数计数；中断时恢复同目录，保留 Adam/RNG/数据游标和评估进度。
- 不自动选 best，不增加 Pred/ego 或重启噪声适配。第三轮能否涨分需实测；尤其 STC 不保证改善。

服务器执行（代码同步后）：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export CUDA_VISIBLE_DEVICES=0,1
# 如需指定旧目录：export DENSE_DDP_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/dense_valid_shared_full2_ddp_实际时间
export DENSE_EPOCH3_DIR="$PWD/outputs/p0_f9_shared_history/dense_valid_shared_epoch3_$(date +%Y%m%d_%H%M%S)"
unset DENSE_EPOCH3_RESUME
nohup bash tools/dense_valid_epoch3/run.sh > "${DENSE_EPOCH3_DIR}.log" 2>&1 < /dev/null &
echo "输出: $DENSE_EPOCH3_DIR；日志: ${DENSE_EPOCH3_DIR}.log"
tail -f "${DENSE_EPOCH3_DIR}.log"
```

新终端续训这次第三轮：激活相同环境、显式设置原 `DENSE_EPOCH3_DIR` 与双卡，再设 `DENSE_EPOCH3_RESUME=1`，同一 nohup 命令即可。不需要重新指定旧目录；从新 contract 恢复 source 路径。

本地小数据验证仅覆盖接口、真实 Adam/backward、双进程 CPU/Gloo 同步/恢复和 dense 评估接线；不是服务器双 CUDA 速度或真实数据质量验证。
