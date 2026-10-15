# 累计干净适配1000步 checkpoint：完整VAL只读评估

输入必须为已完成 `tools/stc_clean_continue/run.sh` 的累计1000步结果，不是原GTepoch2、
旧clean500、污染版或internal ego版。只读snapshot至新输出，从checkpoint原model配置构造GTBody并严格load。
不创建优化器，不再训练，不改旧checkpoint/Adam/RNG/缓存，不推广或挑best。

固定完整VAL4369窗口/150场景，OCC+GT和STC+GT各评Transport/Refiner，标准full-grid IoU/mIoU/Moving及1/2/3s/平均。
GTfutureego是条件；无内部头/外部planner、不按planner覆盖裁剪、不读未来visibility输入、不改先验系数。
保持原适配评估known：OCC历史lidar-known，STC全发布sem/dense-valid、绝不GTmask。
复用正式final_semantic_evaluation的Predictor/engine/metrics，GT-ego引擎仅切历史数据源；不修改任何旧依赖文件。
完整人口两种历史均preflight，缺文件直接拒绝，不静默丢窗口。预测后才读未来标签/Moving。

模态顺序OCC再STC，共用冻结模型；各自整数指标/窗口游标保存，可Ctrl-C后同目录续评。
完成OCC后续STC时，OCC完整state会直接还原报告，不重复预测。checkpoint/hash/数据inventory/代码/配置绑定。
完整VAL已反复用于开发，不能称pristine independent test；质量评估时间不是FPS。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
unset CLEAN_ALL_SOURCE CLEAN_CONT_DIR CLEAN_ALL_RESUME
export CLEAN_ALL_DIR="$PWD/outputs/p0_f9_shared_history/clean1000_all_$(date +%Y%m%d_%H%M%S)"
bash tools/stc_clean_all/run.sh && cat "$CLEAN_ALL_DIR/summary.txt"
```

默认只接受outputs下唯一已完成stc_clean_continue1000_*；多个则显式设置CLEAN_ALL_SOURCE，绝不按分数或时间选。
恢复时指定原CLEAN_ALL_DIR、同源，`export CLEAN_ALL_RESUME=1`后运行同脚本。
