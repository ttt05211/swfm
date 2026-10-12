# 共享历史：连续 ADD＋REMOVE 与执行修复

## 范围

修复提速分支 scene-packed ADD 漏掉静态 actor=-2 的错误。每次 forward 预建实际 actor/class 的六时距合法行索引，保持 actor→class 顺序和 scene slot 隔离；每 scene/group 的 pose inverse 只计算一次。索引不跨 forward 缓存，预测坐标/合法域不进入持久几何缓存。

原 `tools/shared_history_experiments/model.py`、`engine.py`、`run.py` 及 joint_occ trainer 未修改。原 ADD-only 断点实施指纹不包含嵌套 actions/speed 文件，继续兼容；旧 speed receipt 绑定 speed 源码，修复后必须重新测速，不能沿用旧推荐。

## 新学习配置

独立 `tools/shared_history_experiments/actions/`，默认 GT、4 历史→6 未来、相同 TRAIN20%4086 窗×2 轮、seed20261011、batch4/source128、整周期余弦。从 RANDOM 初始化，不导入旧模型或 Adam。基础配置与数据顺序一致，但新增层改变初始化 RNG 消耗，**不声称与 ADD-only 所有参数逐字节同初始化**。

- 同一共享 encoder/STWM/CCR，无第二个修复网络。
- CCR 原1维 ADD 改为2维 sigmoid（ADD、REMOVE），增加 REMOVE 合法性与 fallback 类别 embedding 条件，复用19类 embedding。width64 时相比原 head 增加609个参数。
- 固定初始 bias：ADD=-2、REMOVE=-4。初始偏 KEEP，但 REMOVE 非常数零，能接受最终占据梯度。
- 训练与推理都连续删除，最终 argmax；不使用硬0.5/0.95动作阈值，不扫描阈值。旧通用 trainer 的 contract 为兼容仍存 thresholds 字段，但本入口软输出不消费它们。
- 原全六future full-grid CE＋Lovasz、source SE(2)/existence、ego R10 不变；不再新增动作 BCE。
- 删除把被编辑 source-class 概率质量转移到 fallback。动态回退下层背景；静态 road11/sidewalk13 回 free17。保留 ownership/合法域，不任意删除建筑植被、不任意重标语义，不能撤销同次 ADD。
- 2026-10-12 用户明确接受报告中的浮点执行差异，后续新实验默认逐 scene **H6** 渲染/梯度累积，六future全部监督，不采用Packed、不降低Lovasz频率。旧ADD-only测速H2→H6为0.591009→0.353608s/窗，8步严格numerical/replay门槛均FAIL，保留失败记录，**不是无损/逐步数值等价声明**。`last.pt` 与轮次快照保存 `execution_policy` 记录该授权和原报告路径。新双动作的L40S吞吐/质量仍未验证，不能借用旧速度。

DEV512 固定报告同一双动作模型的 Transport、关闭 REMOVE 的 ADD-only 消融、ADD＋REMOVE；预指定主读出为 ADD＋REMOVE，不根据 dev 选择路线。它的 ADD-only 消融与之前独立训练的 ADD-only checkpoint 是不同对照。

## 命令（代码更新到服务器以后）

先仅8步接口检查，配置仍是完整两轮余弦：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/shared_history_experiments/actions/run.sh --max-updates 8
```

接续同一新实验（用启动时打印的实际目录，不用旧 ADD-only last.pt）：

```bash
RESUME=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际shared_addremove目录/last.pt \
  EPOCHS=2 bash tools/shared_history_experiments/actions/run.sh
```

也可直接运行完整固定预算：

```bash
bash tools/shared_history_experiments/actions/run.sh
```

Ctrl-C/SIGTERM 完整 update/window 边界保存；kill -9 只恢复最近周期保存断点。`last.pt` 保存 model/Adam/RNG/cursor/完整契约；完成轮额外保存 epoch_0001/0002.pt。同目录普通恢复严格核对模型、实现、人口、配置、周期，拒绝旧 ADD-only 协议。

中途只读评估写独立目录，不覆盖训练 summary/last.pt：

```bash
TRAIN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际shared_addremove目录
RESUME="$TRAIN_DIR/last.pt" RUN_DIR="$TRAIN_DIR/eval_latest_$(date +%Y%m%d_%H%M%S)" \
  EPOCHS=2 bash tools/shared_history_experiments/actions/run.sh --evaluate-only
```

固定 DEV512 与训练契约一致；不恢复时修改人口/阈值/epoch 总数。历史几何缓存只读复用，不重建 learned bank。

H6默认仅适用于这个新actions入口。旧ADD-only原入口保持H2和原代码指纹；不静默修改已存在断点的执行路径，旧两轮结果保留。新actions源码指纹已变化，如果此前使用H2 actions试跑了断点，必须明确迁移后才能续训（当前未实现该迁移），不能直接套用。

## 验证边界

本地专项覆盖：静态 actor=-2 ADD/scene-packed 输出与梯度；每forward行索引及逆矩阵复用；baseline独立回放漂移/OOM拒绝；REMOVE 概率质量守恒和 fallback 梯度；真实全六占据损失更新删除权重；CPU/CUDA-BF16 小网格六帧快速/参考一致；两步 Adam/RNG精确恢复；真实CLI固定4086人口、epoch快照、同目录续训及独立只读评估、旧协议拒绝；原ADD-only回归。

这是小网格实现验证，不是 nuScenes 精度、L40S速度或论文FPS。未自动启动服务器、扩大预算、推广模型或推送。
