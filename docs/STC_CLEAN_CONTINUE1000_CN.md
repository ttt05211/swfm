# 干净 dense-valid 全解冻500后，再接500步

仅从已完成的 `tools/stc_clean_full/run.sh` 结果继续，不从原GTepoch2重来。
保留权重、Adam step/一阶二阶矩、RNG，原纯TRAIN20%日程接第501--1000批，绝不重复前500批。
固定新增500步，累计适配1000，不是原GT主体训练以来的总步数，也不是完整dataset epoch。

上一阶段500步余弦已经结束：新增500步明确重启body1e-5/refiner1e-4、floor0.01的余弦，
Adam不重置；这是两段500，不是一开始就配置的1000步整周期，也不是沿终点极低LR恒定训练。
不回改旧500权重、优化器或结果。新输出保存本阶段0--500游标及累计500--1000字段。

纯干净TRAIN GT、TRAIN dense_valid，Encoder/STWM/Refiner全解冻，H6/B4/source128分组、
完整6时距CE+Lovasz/c1/原epsilon，不加污染、真实STC训练、ego/motion/existence loss或新source。
只读复用兼容历史几何，不建新特征bank。最终固定同remaining64的OCCGT(原lidar-known)/STCGT(dense-valid)，
标准full-grid IoU/mIoU/Moving，并列源500与累计1000，不重训旧组，不自动推广/扩训/挑best。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
# 单个已完成clean_full500目录自动识别；多个时必须显式CLEAN_CONT_SOURCE，不按时间或分数选。
unset CLEAN_CONT_SOURCE CLEAN_FULL_DIR CLEAN_CONT_RESUME
export CLEAN_CONT_DIR="$PWD/outputs/p0_f9_shared_history/stc_clean_continue1000_$(date +%Y%m%d_%H%M%S)"
bash tools/stc_clean_continue/run.sh && cat "$CLEAN_CONT_DIR/summary.txt"
```

如有多个源：`export CLEAN_CONT_SOURCE=/root/nas/occ/swfm/outputs/p0_f9_shared_history/stc_clean_full500_实际目录`。
Ctrl-C整update保存；恢复指定同CLEAN_CONT_DIR与同源，再`export CLEAN_CONT_RESUME=1`运行原脚本。
恢复本阶段Adam/RNG/游标，总共仍只新增500。若已训练完成，只续未完成评估。
