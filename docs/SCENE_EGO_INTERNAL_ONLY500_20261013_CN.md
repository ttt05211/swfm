# 场景 ego：只训练及评估内部 Pred

最新用户明确：外接 planner 不跑，GT reference 不跑，只跑内部 ego。
新增 `tools/scene_ego_head/experiment/internal_only/`；旧三路入口源文件不修改，
避免破坏可能已经启动的旧任务/断点实现指纹。**今后本次试验使用下面新入口。**

训练完全沿用已审查的随机全模型500步、原TRAIN4086前500groups、seed20261011、
batch4/source128、500步完整余弦、H6 CE+Lovasz+ego。直接motion/existence损失关闭。
内部pose用于全部未来几何，硬Strong/CLEAR背景重建且detach；连续prior/memory
向ego/F/物体XY/yaw反传。GT连续pose只做标签，GT派生navigation仍是声明的条件。

最后只跑内部ego的Transport与SemanticRefiner两个同次读出，报告完整DEV512
的1/2/3s及平均mIoU/IoU/MovingMicro和六时刻XY/yaw误差。不读取planner缓存、
不取planner人口交集、不调用external或GT pose-conditioned forward。
复用原整数评分代码但明确注入内部Pred forward，独立eval协议/route标签；
退出/异常后GT评分函数及协议均恢复。无aligned、阈值/轮次选优或扩预算。

## 运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

export RUN_DIR="$PWD/outputs/p0_f9_shared_history/scene_ego_internal500_$(date +%Y%m%d_%H%M%S)"
bash tools/scene_ego_head/experiment/internal_only/run.sh && cat "$RUN_DIR/summary.txt"
```

不用设置PLAN_CACHE。新入口同目录中断恢复：

```bash
# 保持RUN_DIR为实际原目录，不生成新时间戳。
SCENE_EGO_RESUME=1 bash tools/scene_ego_head/experiment/internal_only/run.sh
cat "$RUN_DIR/summary.txt"
```

若已启动旧三路入口，不要另起第二个GPU训练，也不静默把旧checkpoint当新协议续训。
旧任务不会因拉取这几个新增文件改变；按需另行只读评估已有权重，而非擅自重训。

## 对GT版本的影响

旧GT trainer、结构、loss、cache、checkpoint均不改变，因此旧GT模型与已经取得的
结果不受本次代码影响。新场景头只接入本次独立Pred训练。
如果未来将场景ego头也接入GT模型并让ego loss反传共享F，则可能改变GT训练轨迹；
不能声称加头天然不影响精度。GT条件的未来几何仍需直接使用GT，不使用预测头替代。

本地针对性测试覆盖：内部-only无planner/未来GTpose输入、同次两个readout、
完整整数恢复、原GT函数恢复、实际CLI固定500/安全停止恢复/只读eval、旧协议拒绝。
只是真实CPU/CUDA小网格测试，不是服务器nuScenes指标、L40S性能或收敛保证。
