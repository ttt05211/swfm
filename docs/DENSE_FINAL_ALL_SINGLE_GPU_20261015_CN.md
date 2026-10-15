# 固定 dense epoch2 全集：单卡双评估进程

两组融合均未提高主要mIoU；第二组wide相对epoch2 OCC m−0.175543、STC m−0.156841，但Moving +0.929205/+0.460116，是指标取舍，不称全部无效。按此前约定不继续组合，固定原dense-valid双卡训练的epoch2/update10292，完整评OCC+GT和STC+GT。

用户特别明确只用单卡。新增 `tools/dense_final_all`，不修改原trainer/model/缓存和训练fingerprint。`run_parallel.sh` 默认 `DENSE_ALL_GPU=0`，协调器只接受一个可见GPU，两个子进程继承完全相同的CUDA_VISIBLE_DEVICES；每个子进程只看到cuda:0，不使用物理GPU1。GPU模型副本/显存会增加，CPU准备可交错，GPU会争用，不保证快于单进程、更不保证2倍；不自动开MPS/改驱动/混精/网络/阈值。

- 固定权重：只接受完整原dense/shared-pinned epoch2/update10292，no mean/no third/no Pred。可设DENSE_DDP_DIR，缺省找最近具有epoch_0002.pt的dense_valid_shared_full2_ddp_*。
- 完整有序4369窗口、150scenes；每模态按keys[0::2]/keys[1::2]分2185/2184窗，无重复、padding、drop或planner覆盖裁剪。
- OCC/STC均dense-valid，四历史至六未来，GT future ego明确条件，full-grid在1/2/3s计IoU/mIoU/Moving。STC不读GT历史mask，无内部ego、导航或训练。
- 每个进程各自缓存有限历史和pure preparation，不跨模型复用学习特征，不申请旧training track cache writer。
- 每8完整窗口及退出保存整数state。Ctrl-C/SIGTERM协调器只通知自己创建的两个子进程，等待其完整窗保存；同目录恢复各自游标，已完成OCC不重复预测。
- 全部完成后核验每片keys/来源SHA/known/计数总数，SUM原始confusion与moving计数后再算指标，绝不平均两片IoU；失败不丢片或自动切后端。

服务器同步后：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export DENSE_ALL_GPU=0
export DENSE_ALL_DIR="$PWD/outputs/p0_f9_shared_history/dense_epoch2_all_singleGPU2proc_$(date +%Y%m%d_%H%M%S)"
unset DENSE_ALL_RESUME
nohup bash tools/dense_final_all/run_parallel.sh > "${DENSE_ALL_DIR}.log" 2>&1 < /dev/null &
echo "输出: $DENSE_ALL_DIR"
tail -F "${DENSE_ALL_DIR}.log" "$DENSE_ALL_DIR/shard_00.log" "$DENSE_ALL_DIR/shard_01.log"
```

tail可Ctrl-C退出，后台评估不停止。输出parent/summary.txt、evaluation.json、occ/stc_merged_counts.json、两份子进程日志及各自子目录state/contract/冻结源快照。读取总结果：`cat "$DENSE_ALL_DIR/summary.txt"`。

续评同目录：激活相同环境并设置原DENSE_ALL_DIR、DENSE_ALL_GPU=0、DENSE_ALL_RESUME=1，再执行相同nohup命令。若需单进程对照，使用**新目录**运行run.sh，勿将双进程parent目录当单进程断点；不自动将结果差异选成best。

本地6项检查通过：真实小模型epoch2加载/来源不变/拒绝其他step、完整人口、同一GPU可见性启动接线（模拟子进程，不是双CUDA实测）、不等长分片整数汇总与错片拒绝、实际旧metric engine微型OCC完成/STC部分/恢复和完成后零重预测、known编码及memory旧回归。未在真实nuScenes上执行，未测服务器单卡双进程加速或峰值显存。安全环境技能仅选择本地非base解释器，不改服务器OccFM。
