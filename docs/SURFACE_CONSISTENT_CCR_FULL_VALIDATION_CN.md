# 表面一致性 CCR：完整训练集验证

## 目的与结构

保持 V18 Transport 与快速 CCR 数据流，只替换 CCR 内部静态表面的表示和条件读出；不是在 WM → CCR 后继续外挂一个修正器。当前证据说明额外 FP 集中于路面/人行道，且许多来自真实历史证据，但尚未证明是高度、量化还是配准误差主导。本实验用新信息检验这一假设，不宣称原因已确定。

历史表面证据 → 一次共享编码 → 六个未来条件读出 → 原约束合成。

- 在完整 Direct 静态点集合上建立历史局部表面索引。只查询至多 16 个近邻计算坡度、高度残差、粗糙度、支撑程度及异类边界距离；16 是描述子上下文预算，不裁剪候选。使用三维距离，保留不同高度层，未知支撑显式标记。
- 12 维几何描述子直接进入 CCR 编码。6 维未来投影高度/体素小数相位进入静态条件读出，允许各未来帧采取不同修复动作。静态读出替换原读出，不给原 logits 再叠一个修正分数。
- 不增加候选、修改 halo/ownership、改变采样或 compositor。动态路径仍为原 CCR。无需 AE、蒸馏、新 Transformer、逐点大 patch 或六个密集编码器。
- 执行时根据现有候选分块的CPU角色元信息选择读出：纯静态块只算新静态读出，纯动态块只算原动态读出；混合块保留两个路径。不重排候选或改变矩阵批次大小，CPU/CUDA逐字节等价已测试。冻结验证还去掉无用原静态 logits 的反传图；正式联合训练保持运动上下文梯度。

最终可联合训练同一模型的全部参数；当前先冻结运动和原动态 CCR，隔离静态收益。能反传到运动上下文并不意味着从头联合训练的精度已得到验证。

## 固定实验

复用明确确认的 Frozen B：

```
/root/nas/occ/swfm/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt
SHA256 fb6f7bdfa9e7ebca8f479d7005372ad328e09a2ed44e84514cfb180722942204
```

新静态读出复制 B 权重，新增几何输入投影置零；基线参数/动态参数冻结。完整 TRAIN20430 × 3 轮，新 AdamW，LR=3e-4，三轮全程余弦至 0.1 倍，无 tail。保留 TRAIN 正权重；ADD-only 加权 BCE，raw sigmoid≥0.5，REMOVE 关闭。不使用前次 halo 规则候选，不调阈值。

B输入在新输出目录快照后加载，不持续依赖服务器可能变化的`last.pt`。

每轮 dev64，最后 dev512 同窗口对比 Transport、Frozen B、Old Local（REMOVE-off）和新 CCR，包含 IoU/mIoU/MovingMicro、分时距、分静动态质量与场景统计。dev512 是开发集合，不能冒称独立测试。最终轮选择，不自动挑 dev-best，不自动扩训/全量评估/升级 Frozen B。

## 缓存与速度

```
TRAIN /root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_v1
VAL   /root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1
```

保持 v2 几何缓存 namespace 依赖文件原样。训练 mode=require，缺失/不匹配即停，不自动重建。缓存只含旧固定几何。新增表面索引及采样描述子在窗口生命周期内计算，不写入持久缓存；训练仍先采样再算描述子，不全量编码无用候选。复用逻辑 batch4/source128、superbatch4、CUDA streams4、6 个预取与 4 个采样 worker 的现有快路径。

正式 FPS 从全部固定 dev64 keys 确定 18 个 scene-balanced + 2 个 source-count-only 压力窗口，交替对比 Frozen B 和新 CCR，各重复三遍。边界不变：CausalHistoryState → fresh Strong/KTA + live motion + learned CCR + **实时未来相位计算** + 六帧 dense 合成。历史-only 表面索引/描述子归入确定性历史表示，单独报告其构建时间；不缓存 learned encoding、未来相位或 learned poses。FPS 不代表原始历史输入端到端速度。

动态概率逐字节核对；重复六帧输出一致性在计时外检查。是否仍达 ≥40 FPS 要由 L40S 实测，不能拿本地小 GPU 或旧候选数字代替。完整轮十几分钟是用户既有缓存版本的经验，不是这版承诺。

## 启动与恢复

服务器代码更新到本实现后：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_surface_ccr_full.sh
```

中断：Ctrl+C 或 SIGTERM，在完整 optimizer 更新后保存。断点为新输出目录下 `last.pt`；旧 B 是 warm-start 输入，**不能**当作本实验 resume。

```bash
CCR_RESUME=/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/本次目录/last.pt \
  bash tools/real_motion/run_p0_f9_surface_ccr_full.sh
```

恢复使用新输出目录，严格恢复模型、optimizer、Torch/CUDA/NumPy RNG、批次游标及原三轮 cosine。不得静默改变 LR、轮数、采样、实现或 batch。kill -9 只能恢复最后周期 checkpoint（每256更新）；正常停止按完整更新保存。训练结束后的评估/FPS中断可从完成训练的断点续做，不重训。

## 本地验收

124 passed / 1 skipped（实际CPU/CUDA集中回归）；601 Python文件AST检查、Bash启动脚本语法检查通过。完整1437项旧方案重回归未跑完，不能称仓库CI全绿。旧测试fixture和测试入口环境的修复不改变模型数值容差或持久缓存合同。

`benchmark_surface_ccr_head.py`仅用本地RTX3050合成48000候选检查MLP六读出/上传/回读的开销和路由等价，不包括几何、Strong、运动、合成，不代表正式FPS或精度。不拿它代替L40S 20×3 paired测试。

本次5遍交替测量：Frozen B head81.28ms，未路由新head120.90ms，路由新head91.77ms。路由消除了约24%的新head开销；新head仍比B head多约13%。初始化概率逐字节一致。数字只属于这次合成head口径，没有声称全模型速度相同或精度已经追回。

## 收口

先看静态 mIoU 是否真实追回、Moving 是否不退、同口径 FPS 是否达标。若无效，不自动再跑 E4/E5、不追加修正器。若有效，再用同一结构干净随机初始化、完整数据联合训练；这一正式训练仍是后续独立实验，当前命令不会启动。
