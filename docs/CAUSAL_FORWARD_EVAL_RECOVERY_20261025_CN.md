# 前向关联单卡一轮：完成训练后的只读评估恢复

服务器在 `1024/1031` 后进入轮末评估，报 `four histories ending t0 -> six futures`。日志不逐步打印；原循环在完成第1031次更新时先保存 `C_train/last.pt`、`C_train/epoch_0001.pt`，再运行评估。是否确实完成，由恢复入口读取断点并严格核验，不能仅凭最后一条日志猜测。

## 根因与修复范围

旧缓存记录可以保留六个历史 token，实际网络输入为末四帧语义、known 和位姿；训练构建器始终使用末四帧。新评估器把旧记录传给 `ForwardPredictor.prepare`，后者错误要求元数据恰好四个 token。训练更新与网络结构无需改变。

新包 `tools/causal_forward_recovery` 只把评估记录的历史 token 视图规范为末四个，并验证实际数组/位姿为四帧、末 token 等于 t0、未来恰好六帧。保持未来 token、实际历史输入、时间戳、GT ego、dense-valid、前向关联、KTA、Strong、Memory、网络、精度和指标定义不变；不放宽数值容差或原来源检查。

原训练实现源码已被完整哈希写进断点，所以本次不直接修改旧 `causal_forward_train` 或 `causal_forward_track` 文件。新入口在独立命名空间加载完成的原断点，保留原实现 SHA 校验；断点必须为原单卡协议、epoch1/batch0、已完成全部更新、TRAIN4086/同64、原参数量与配置。config、DEV cache/info、STC历史清单、控制报告均核验。原断点和旧失败报告不会被重写。

## 服务器命令

保留原来的 `FORWARD_DIR`，它应指向包含 `C_train`、`frozen_B`、`pair_audit` 的本次工作流目录。不要重新创建训练任务，也不要继续调用会重新进入旧评估器的原 workflow。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

: "${FORWARD_DIR:?请设置为刚才失败实验的原目录}"
test -f "$FORWARD_DIR/C_train/epoch_0001.pt" || exit 1
export FORWARD_EVAL_GPU=0
export RECOVERY_RESUME=0
export RECOVERY_DIR="$FORWARD_DIR/recovered_eval_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_recovery/run.sh > "${RECOVERY_DIR}.log" 2>&1 &
tail -f "${RECOVERY_DIR}.log"
```

只读补评单卡、原64窗的 OCC/STC，六个未来预测完成后评分1/2/3s。raw 和固定 FREE−1 从同一前向读出，不新增历史衰减或重拟合系数。固定 FREE−1 仍是此前使用过目标标签确定的输出校准，不能称为无标签方法。完整窗口整数计数每窗写入新目录，可用原 `RECOVERY_DIR` 加 `RECOVERY_RESUME=1` 中断恢复。

```bash
cat "$RECOVERY_DIR/summary.txt"
```

汇总会在新目录复用原 B/框配对审计证据，生成新的 C 冻结评估视图及 A/B/C、FREE−1 表。`recovery_receipt.json` 记录源断点 SHA、源1031步和本次零训练更新；原 `C_train/results.json` 的失败前状态保留。若完整断点/来源校验失败，应先查看报错，不得绕过后重训或冒用别的断点。

后续新训练应显式采用一致的四帧元数据适配；本恢复入口用于救回这次已完成训练的评估，不修改旧训练默认实现。

## 验证

本地 CPU/CUDA 覆盖原6-token检查复现、四帧关联/读取/六帧推理、真实多余输入及错误t0/未来长度拒绝、完成断点严格加载/旧SHA拒绝、无优化器创建、全部六帧预测先于标签、完整窗停续与旧文件不变。真实服务器同64质量结果尚待运行。
