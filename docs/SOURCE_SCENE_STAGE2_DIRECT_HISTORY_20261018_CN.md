# 第二阶段：KTA只作残差锚点，移除ICP的直接历史搬运

## 起点和边界

本轮基于**刚完成的一轮Scene Reader**，不回到旧clean1000/norm192方案。用户回传：Reader总参数1,573,186，比baseline少23.4%；质量短训基本持平，训练约慢1.6%、GT/STC FPS约慢2.3%/2.7%。这是表示重构候选，不是已证明提分/提速的新正式模型。

第二阶段仅新增 `tools/source_scene_stage2`，不改任何Stage1、正式GT、旧缓存/检查点/FPS实现或默认入口。回退直接使用原Stage1 Reader入口和权重，不做git reset、不覆盖旧文件。无全局monkeypatch或实验入口临时替换；replay/test中的ICP trap只用于检查没有调用旧算法。

本轮三个关联改动：

1. 删除 `kta_future_proj`。未来Query只含learned queries并读取历史context；网络完全不读取KTA参数。原两帧KTA仍在最终source_targets残差锚点和原Strong背景中使用，初始化未来XY残差为零，仍退化为KTA。
2. 同一共享source身份，历史有效性由registration_valid改为track_valid；新train/live数据路径不计算ICP、不读取旧ICP矩阵/接受门限。原类约束/互近邻/速度门限及较早帧歧义规则、两帧KTA完全保留，不借此更换关联算法。
3. 历史memory保留各帧原world点及各自source中心，网络从四历史temporal tokens预测历史到t0的相对yaw，再与同一未来source运动组合，直接搬到未来。静态ego warp、importance/coverage融合和Semantic Refiner保持原逻辑。

这是一组联合结构变化，**不是单变量消融，也不保证比原registration好**。不同时加入GT框监督、姿态标签、motion/existence/ego loss、新阈值、速度拟合、噪声增强、校准或修补规则。

## 为什么多一个历史yaw输出

身份/中心不能确定物体朝向。只做 `p_t-c_t+c_future` 会忽略历史转弯，并不等价于ICP。新版本不隐瞒这项差别：

```
四帧F → 同一个Scene Reader → Temporal tokens
                               ├─ Future queries → future XY/yaw/Q
                               └─ shared Linear(128,1,bias=False) → history yaw to t0
```

历史yaw头读 `context_t-context_t0`，t0严格为0；无效token仅以attention mask处理，输出yaw也置0。新头零初始化，无额外loss。运动网络仍不消费类别embedding、显式速度/位移、mask数值特征或KTA Query投影；标准XY/真实时间PE保留，否则中心跟随采样会隐藏运动。

未来yaw `theta_h0` 与历史相对yaw `theta_0t` 组合：

`p_future.xy = R(theta_h0 + theta_0t) * (p_history.xy - c_history.xy) + c_future.xy`

历史world Z保持原planar renderer约定，随后完整future ego pose转网格；不宣称补了物体z/roll/pitch运动。t0观测几何仍由原Transport搬运；新历史yaw只影响memory，不篡改t0几何或KTA。静态点仍按历史ego→未来ego warp。

这不是“无配准信息也能精确对齐”的等价优化，是让最终占据监督学习历史相对朝向的研究改动。只有CE+Lovász通过memory→采样权重/坐标反传；初始Refiner最后层为零时memory/历史yaw梯度可能还未建立，后续随读出学习打开。短预算不能保证相对yaw有唯一物理解释或足够可辨识性。

## 训练/对照

- **只新训stage2一组**，TRAIN4086（20%）×1轮，原seed20261011、顺序、global B4/source128、AdamW、motion5e-4/refiner3e-4、完整一轮余弦、原BF16和六未来CE+Lovász。
- 从RANDOM开始，不导入Stage1训练权重。构造同seed的Stage1新鲜Reader，把共同名称/形状的随机参数复制到stage2；Encoder/Reader/Temporal/FutureDecoder/Refiner初始参数一致。新history-yaw权重为零；删KTA投影，生产架构总参数预计少512，实际由checkpoint配置核对。
- 复用已完成Stage1 Reader的results作为对照，不重训baseline/Reader、不按dev挑checkpoint。代码拒绝人口/known/日程/数据/精度/world size等不同的结果；架构和validity变化明确披露。
- TRAIN、OCC、STC都dense-valid；GT future ego为明确条件，无导航/内部ego头。结束同remaining64评完整六帧，只按1/2/3s全网格计IoU/mIoU/Moving；双卡训练，评估仅rank0一张GPU，另一rank等待。
- 新结果为development证据，不是独立test或直接继承旧全集/FPS。匹配正确率审计仍沿用原独立Stage1 audit.sh，不会自动执行。

## 缓存和恢复

旧history geometry只require/read-only。为兼容数据读取，旧压缩blob可能包含旧registration字节，但新Provider仅取t0纯几何和原GT-conditioned Strong背景；registration/support/Surface等不交给新消费者，错source不允许退回旧builder。训练GT背景复用与上一阶段相同；live OCC/STC按各自历史/未来ego重新建Strong，不能借未来GT occupancy。

新identity-only payload不含registration字段，用独立协议和**本次新目录**下 `identity_history/rank*`，不复用旧track派生缓存。原组件提取/两帧匹配/KTA没有重复构建后覆盖；当前缓存t0组件复用，前三帧原提取一次。

last.pt保存新协议/模型/Adam/完成游标/各rank RNG/评估状态；旧Stage1 checkpoint不能作为stage2 resume。中断只恢复相同新目录、相同卡数和完整契约。六帧损失/末小batch均保留。评估整数按整窗提交；原Stage1评估状态不可冒充新状态。

## 服务器命令

同步分支后，`SCENE_PILOT_DIR`指向**刚完成baseline+Reader一轮**的父目录，不是speed目录，也不是reader子目录。若环境没保留，脚本只在找到唯一完成的一轮任务时自动识别；多个则列出候选，要求明确指定。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export SCENE_STAGE2_DIR="$PWD/outputs/p0_f9_shared_history/scene_stage2_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_stage2/run.sh > "${SCENE_STAGE2_DIR}.log" 2>&1 &
tail -F "${SCENE_STAGE2_DIR}.log"
```

完成后 `cat "$SCENE_STAGE2_DIR/comparison.txt"`。原结果引用只读；新的summary/results和epoch_0001.pt在本次目录。续训设同 `SCENE_STAGE2_DIR`、同 `SCENE_PILOT_DIR` 和卡数，再 `export SCENE_STAGE2_RESUME=1` 重跑run.sh；不是从旧Reader权重继续训练。不会代启动服务器或自动扩训/推广。

## 本地验证记录

新阶段与原Reader、测速、shared-source、DDP归约、Refiner、GT训练及匹配审计相关回归共 **60 passed / 39.56s**。含CPU/CUDA全六帧CE+Lovász梯度（历史yaw/未来XY/yaw/Reader/Encoder）、精确CPU Adam/RNG恢复、CUDA小网格新trainer停止续训/只读旧对照、实际OCC/STC整窗整数续评、身份/KTA逐数组一致、旧ICP全部拒绝时track-valid仍可读取、旧缓存配准字段和未来标签poison检查；CLI与Bash语法通过。

真实已下载两窗GT/STC完整200×200×16 H6前向通过；GT分别17/24个source、STC分别9/16个source，identity有效历史token分别64/84和29/57。初始future残差与history yaw均零，初始最终argmax等于Transport；memory为6×16×100×100×16。ICP入口被trap时全程通过，没有读未来占据、优化或产生新质量/FPS。生产参数实数 **1,572,674**（比Stage1 Reader少512）。记录 `outputs/stc_overnight/scene_stage2_real2_plumbing_20261018_v1.json`。

本地Windows按windows-python-env-guard使用安全非base yoloe环境。完整尺寸只验证前向；小网格全6反向不冒充L40S完整训练显存/性能，不能继承上一阶段FPS。服务器训练和真实质量尚未运行。
