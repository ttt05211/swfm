# 当前epoch2与已有mean3：固定另一批256窗

用户希望在更多窗口上比较10292步与刚生成的mean3。已有64窗Refiner结果：epoch2 OCC51.260686/64.055743/34.764747、STC18.229030/28.514311/21.272111；mean3 OCC50.954636/63.959331/34.626334、STC18.238343/28.642992/21.156248（mIoU/IoU/Moving）。融合只带来STC IoU小涨，需更大人口成对验证，不再搜索融合组合。

## 固定对象与人口

只读现有的源 `epoch_0002.pt` 与已有 `mean3.pt`（4000/10000/10292等权1/3）。不重新融合、不训练、不修改原报告或断点。完成mean3目录/receipt/SHA/protocol/coefficients/source contract、三源断点SHA、源full报告、epoch2完成游标、网络shape/dtype/有限性与相同buffer均检查；原训练及mean执行实现指纹不放宽。

从原完整VAL4369/150场景身份列表中，**排除原64窗所在的全部场景**，余下按 `(scene,t0)` 排序，再用现有scene-balanced round-robin固定取256。只用身份元数据，不读未来标签或分数，不按类别/效果/可见性/外部planner覆盖挑窗；不足或文件缺失则报错，不补回旧场景或偷偷缩小人口。两个模型完全相同有序256窗，TRAIN/VAL场景检查保留。

因此这不是把64窗分数与另256窗绝对值比较，也不把旧64并入均值。新的均值与差值只来自当前两模型在这同256上的新预测。排除了直接用于选融合配方的场景，但先前其他版本完整VAL已被查看，仍不称全新独立测试/标准全集分数。

## 推理

单GPU、单进程，内存中保留两份冻结参数。每模态每窗口只做一次纯历史component/前向身份/KTA/读取坐标构建，两个模型共享这些纯几何输入；各自运行自己的Encoder、运动/历史yaw、fresh Strong、Transport、对齐Memory、Refiner。禁止跨模型共享learned field、motion、future投影或Memory结果。

GT future ego、dense-valid、原FPN/AdaLN-Zero/U-Net及前向关联保持；FREE−1与固定历史衰减关闭。两模态×两模型的全部六帧预测后再读未来GT占据和Moving，在完整网格上评分1/2/3s。每窗8路计数全部有效后才合并及保存；恢复严格同源/人口/代码/输入，已完成窗口不重评。源权重/Adam/RNG/cache只读。

`summary.txt`只列八行两模型×两模态×Transport/Refiner均值和mean3−epoch2四行差值。逐时距和完整名单留在 `results.json`、`contract.json`，避免终端打印巨大列表。当前只是质量评估，不是FPS。

## 服务器命令

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export FORWARD_COMPARE_GPU=0
export FORWARD_COMPARE_RESUME=0
export FORWARD_COMPARE_DIR="$PWD/outputs/p0_f9_shared_history/adaln_epoch2_mean3_other256_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/causal_forward_compare256/run.sh > "${FORWARD_COMPARE_DIR}.log" 2>&1 &
tail -n 30 -f "${FORWARD_COMPARE_DIR}.log"
```

若保留 `FORWARD_MEAN_DIR`，读该已完成融合目录；未设置则仅允许找到唯一已完成的当前mean3目录。原epoch2目录从该mean3来源直接取得；若还设置了 `FORWARD_FULL_DIR`，必须与其一致。多候选时显式指定目录，不根据日期或分数选择。

```bash
cat "$FORWARD_COMPARE_DIR/summary.txt"
```

中断后保持原 `FORWARD_COMPARE_DIR`、设置 `FORWARD_COMPARE_RESUME=1` 后重跑同脚本。`progress.json`中的windows是两模型、两模态均已完成的配对窗数，总计256，不是1024次单模型调用的数量。

本地覆盖4369/150身份上的固定选窗及乱序输入不变、原64场景全排除、现有真实参数artifact精确加载且无重融合、buffer/receipt/lineage拒绝、CPU/CUDA共享纯几何与各模型独立编码一致、实际256窗CLI完整窗中断恢复/标签后读/已完成不重评及源SHA不变。尚无本次服务器256质量分数。
