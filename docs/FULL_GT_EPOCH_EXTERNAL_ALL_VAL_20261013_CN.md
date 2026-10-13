# 完整GT主体第一轮：外部planner全VAL只读评估

独立`full_gt_external/`入口；旧`full_gt/*.py`、训练workflow fingerprint不变，不接测速候选。
原GT模型严格加载原contract/权重，再在评估进程内改用外部BEV-Planner+yaw，预测六帧、报告1/2/3s及平均IoU/mIoU。
不运行GT参考或内部ego模型，不训练、不改阈值、不事后对齐。Moving本轮不计算以省去annotation评估。

从VAL4369记录完整集合取与planner缓存的交集（既有缓存预计4219窗/150场景，以实际打印为准）；不把缺失预测轨迹的窗补GT，不声称评了4369个Pred窗口或与论文人口完全一致。
`population.json`记录请求数、交集keys、全部缺失keys、planner来源和pose fingerprint。
`results.json.actual_eval_population`为all_val_planner_covered；`contract`是为严格加载保留的源训练/DEV512合同，不能误读成只评DEV512。

未来外部poses用于Strong背景、Transport prior、memory及Refiner；历史缓存只复用因果历史字段，GT未来背景不使用。
预取原数据会读未来标签，但模型forward不读取标签/未来GT poses；只有预测完成后指标代码读取标签。外部planner的自身输入/训练来源不改。
缓存中的z/tilt沿用t0，不用未来GT补全。

## 新终端

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2
export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0001.pt"
export EVAL_DIR="$PWD/outputs/p0_f9_shared_history/epoch1_external_all_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/full_gt_external/run.sh && cat "$EVAL_DIR/summary.txt"
```

不需要停止原训练，但同一GPU并行会争抢算力/显存，可能让两边都变慢或评估OOM；这不是FPS测速。
评估一次只处理一窗、CPU workers2/prefetch2、缓存预算256/128/128MiB，但仍会加载原元数据，实际RSS不能保证只用这些缓存预算。
脚本不停止任何已有任务，也不修改GPU环境或训练线程数。
对正在运行的Python进程，git更新只新增目录，不改原trainer源码。

按一次Ctrl-C保存完整评估窗口计数，之后同GT_CKPT和原EVAL_DIR：

```bash
EXTERNAL_EVAL_RESUME=1 bash tools/shared_history_experiments/full_gt_external/run.sh
```

只读snapshot写在新eval目录；不是原train/last.pt，不载入新科学更新，不覆盖原训练summary/epoch文件。
数据/runtime/实现不符合原checkpoint合同明确失败，不绕过合同。报告为标准full-grid mIoU，保留IoU=0但union>0的类别，无mask/阈值扫描/选checkpoint。

本地验证：新增CPU/CUDA小网格真实前向、GT未来pose/标签/背景毒化不进入模型、完整VAL交集（包含DEV512之外窗口）、完整窗口计数中断恢复、源文件字节不变及原GT训练回归，共8passed in13.63s；CLI帮助、Bash语法及diff检查通过。没有本地nuScenes全集或L40S结果，实际覆盖和资源占用以服务器为准。
