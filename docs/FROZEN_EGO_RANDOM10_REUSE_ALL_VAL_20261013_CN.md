# 冻结epoch2，历史缓存只读复用，随机ego头10轮＋全集VAL

用户明确选择：头从头训练10轮，不接一轮head/Adam；主体仍是已完成的GT epoch2/update10292，Encoder/STWM/Refiner完全冻结。TRAIN20430/batch64/seed21/AdamW3e-4/R10几何监督，完整10轮cosine最低1%，每轮320updates，共3200。导航依然是GT派生destination command；不是navigation-free、不是独立随机Pred整网训练。

## 复用旧bank而不复用旧头

旧history_tokens仍保留，只读加载，不复制、重提取或改manifest。导入程序只检查旧head_last.pt中的contract作为bank出处，不把head/Adam/RNG载入新头。
缓存身份单独提取主体SHA/digest、历史提取源码、TRAIN键及顺序、数据配置、运行时/精度、pool尺寸/特征宽度。训练轮数/LR/头权重/新目录不属于特征身份；旧chunk仍按原contract+SHA验证，不重写旧指纹。记录新history_reuse_receipt.json。
缺失、不完整、损坏或不兼容则报错，**不自动重新提取**。只读查找按manifest时间选最新兼容完整bank，不读取DEV分数来选缓存。HEAD初始化仍是独立seed21随机权重，绝不是旧头继续训练。

## 全集范围

恢复主体仍使用原DEV512 checkpoint合同；actual_eval_population明确记录ALL_VAL4369。
最终epoch10仅内部Pred，完整VAL4369（预期150场景），没有planner覆盖裁剪；完整六帧前向，报告1/2/3s及平均IoU/mIoU/MovingMicro、head/prior轨迹误差。population.json及结果保存实际全部键，检查TRAIN/VAL场景不交叉。两种readout是同一模型的Transport和SemanticRefine，不是另训模型。
入口继续v2严格确定性同位姿审计，仅诊断使用GT；实际Pred的所有未来Strong/prior/memory用预测ego，未来GT只用于导航声明和监督/评估，无对齐、未来占据或可见性输入、无best-by-DEV。

## 服务器运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export GT_CKPT="$PWD/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt"
export EGO_RUN_DIR="$PWD/outputs/p0_f9_shared_history/frozen_gt_scene_ego_10ep_all_$(date +%Y%m%d_%H%M%S)"
unset FROZEN_EGO_RESUME EGO_REUSE_DIR
bash tools/scene_ego_head/frozen_gt/run10_all.sh && cat "$EGO_RUN_DIR/summary.txt"
```

自动打印History source，必须出现REUSED_HISTORY ... fresh=0；不会出现新提取FROZEN_EGO_BANK。若自动找不到，export EGO_REUSE_DIR为这次成功一轮任务的目录（包含head_last.pt与history_tokens/），再运行同脚本。初次启动已建新目录失败时须另一个NEW EGO_RUN_DIR；不要删除旧结果。

中断后在同一新任务目录恢复（不改10轮、数据、缓存来源或代码）：

```bash
export EGO_RUN_DIR="实际十轮任务目录"
export EGO_REUSE_DIR="首次打印的History source目录"
FROZEN_EGO_RESUME=1 bash tools/scene_ego_head/frozen_gt/run10_all.sh
```

仅恢复这次十轮任务，旧一轮任务不能直接改周期续接。全集可按完整窗口保存并恢复，旧源权重/缓存均不写。复用消除首次约14.5分钟特征提取，但仍要载入/校验bank、初始化数据和评全集；10轮耗时不可只保证为9秒×10。

本地17项测试通过（35.35s）：真实只读chunk导入/源文件字节保持/新预算兼容及源变化损坏拒绝，随机头十轮Adam，模拟完整VAL4369接线及恢复，旧冻结头CPU/CUDA前向和一窗诊断。旧CUDA未来毒化测试一次普通散射重跑max5.96e-8失败，现采用已有确定性审计上下文隔离输入差异，没有放宽容差或修改普通推理。Bash/CLI/diff检查通过；模拟数据仅证明接线，无实际nuScenes/L40S质量或吞吐结论。
