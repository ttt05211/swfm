# STC 全集评估：单卡双进程接手

用户当前 `STC_VERIFY 432/4369 seconds411.6`。旧入口是窗口串行，并沿用了本地4GiB显存的H1读出；provider的workers并不等于两个GPU推理进程。本次仅新增独立执行入口，不修改旧verify_all/replay/bundle/calibration、训练器、权重或校准。

## 接手方式

同一块GPU启动两个评估进程，只处理剩余窗口的rank::2分片，无padding。旧任务在完整窗口边界保存退出后，只读复制其完整prefix和整数confusion；父进程将旧prefix计入一次，加上两个分片的原始整数计数，再统一计算IoU/mIoU。不是平均两份mIoU，也不重新拟合校准。

`--stop-source`只对Linux /proc中模块名和输出路径均精确匹配的旧评估进程发送一次SIGINT，等待其保存释放锁；不强杀，不停止训练或其他输出目录。旧目录和新目录相互独立。失败时保留原任务文件，不自动改阈值、切模型或重试。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

# 如当前终端仍保留STC_VERIFY_DIR则自动使用；否则默认查找最新stc_verify_all_*。
# 有多个旧任务时，请显式指定原始目录，避免接手错误任务：
# export STC_VERIFY_SOURCE="/root/nas/occ/swfm/outputs/p0_f9_shared_history/stc_verify_all_原时间"
export STC_PARALLEL_DIR="$PWD/outputs/p0_f9_shared_history/stc_verify_parallel_$(date +%Y%m%d_%H%M%S)"
unset STC_PARALLEL_RESUME
export STC_VERIFY_HORIZON_GROUP=6
nohup bash tools/stc_overnight_parallel/run.sh --stop-source >"${STC_PARALLEL_DIR}.log" 2>&1 &
tail -n 15 -f "${STC_PARALLEL_DIR}.log"
```

默认同一可见GPU0，`STC_VERIFY_GPU`可指定物理卡。不要同时手动启动另一份相同旧评估。新入口自动导入接手时最新的已保存完整窗口，而不是固定导入432。如果旧全集已经完成，则直接导入完整计数，不重跑。

进度：`STC_PARALLEL 总完成/4369 imported=旧完整窗口 shards=[分片0,分片1]`。最终新目录`summary.txt/results.json`，子进程诊断日志`shard_00.log/shard_01.log`。评估进程结束前父进程只合并各分片已保存的完整窗口。

恢复新入口时显式填写两条原路径：

```bash
export STC_VERIFY_SOURCE="原串行输出目录"
export STC_PARALLEL_DIR="已创建的并行输出目录"
export STC_PARALLEL_RESUME=1
export STC_VERIFY_HORIZON_GROUP=6
nohup bash tools/stc_overnight_parallel/run.sh >"${STC_PARALLEL_DIR}.log" 2>&1 &
```

## 数值与验证边界

H6保留完整六帧、相同权重/输入/公式/校准/指标，但不是BF16逐字节相同。真实CUDA单窗比较中，Transport六帧完全相同，Refiner各路有4–9个argmax体素变化（完整六帧384万体素）；严格byte检查仍FAIL，不放宽或称PASS。结果明确记录H1旧prefix加H6新tail的执行方式。希望保持原horizon分组时，可设置`STC_VERIFY_HORIZON_GROUP=1`，仍是单卡双进程，但不做H6合批。即便H1，原CUDA归约重复本身也不保证byte相同。

本地RTX3050 4GiB仅能用H1进行真实双进程64窗联调；单H6实测reserved2436MiB，不在小卡强行并行两H6。该检查不能作为服务器L40S的H6并发显存/加速结论；实际加速不保证两倍。单卡进程数固定2，不盲目堆进程。

验证：19项定向测试通过；真实同卡两个CUDA子进程各32窗、合并64窗完整成功，8条route齐全，三个Transport路由和旧参考指标完全相同，其余mIoU最大差0.000114pp、IoU最大差0.000078pp（保留浮点重复差异，不称byte相同）。组合路由18.508161/28.760863。旧prefix导入/恢复不重复计数、校准更改拒绝、GT标签在完整预测后访问另有定向测试。结果位置`outputs/stc_overnight/parallel_cuda64_smoke_v2_20261016/smoke_report.json`；这仍然仅是DEV64联调，不是服务器新FPS或全集质量证明。

目标仍须真实完整4369窗验证，不把DEV64收益加到历史全集分数，也不把质量评估耗时称为模型FPS。
