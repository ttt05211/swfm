# 当前主架构审查与收口：Shared History / Source Transport / Semantic Refiner

日期2026-10-12。以实际实现为准，不恢复旧Surface CCR或旧ADD-only方案。

## 结论

建议固定当前**GT-ego条件下的主架构**作为后续完整训练、消融与跨输入验证的研究主线，暂停继续更换核心模块。它已有合理可解释的数据流、共同占据监督和明显的短预算真实增益，可以作为正规方法原型；这不等于已证明顶会创新/SOTA、完整收敛、Pred/Camera/Waymo/6秒能力或正式FPS。本次只调整测量，不改模型/loss或扩训练。

主句：**用同一套历史表示预测source运动，并用同一运动搬运观测几何和局部空间证据，再对几何先验作全域语义残差修复。**

## 三个学习模块，一个公共传输算子

```text
四帧历史语义占据 + 历史known
       ├── Shared History Encoder ── F[4,64,100,100]
       │          ├── source局部读取 + G事实 ── STWM ── 六组运动 / Q
       │          └── height lift ── 四帧低通道XYZ历史特征
       └── 固定source/历史关联几何G ← 历史ego poses
                                  │
            KTA/Strong先验         │     未来ego poses（GT显式条件）
                    └─────────────┴────────────────┘
                   Common Source/Ego Transport
                ├──观测几何 ── P_T[6,18,200,200,16]
                └──历史特征+Q ── M[6,16,100,100,16]
                          │
                Full-grid Semantic Refiner
                Z = log(smoothed P_T) + ΔZ
                          │
                  六张18类语义占据图
```

### 1. 共享历史编码

实际`SharedEncoder`只直接读取四帧semantics与observed/known；**历史ego poses不是直接拼入该CNN**，而是用于G、历史source读取/注册与后续空间配准。

输入[4,200,200,16]；逐高度semantic embedding与known拼通道，XY stride2轻量2D卷积/局部残差→F[4,64,100,100]。不对Z直接max/mean，但64维压缩仍是学习的压缩，不能称高度信息无损。四帧共享权重；历史/未来ego几何事实不要求网络猜。

### 2. Source STWM

G由历史语义连通域、因果关联/配准、观测点云、source mask、历史中心与valid等固定操作构成。它是一份几何事实，不是另一个学习encoder。

从F按source各历史实际位置读取局部crop；加入同G里的mask、中心位移/速度、时间、有效性与class。ST块和六个未来query学习运动，输出KTA初值上的XY残差及yaw变化、未来context Q。真实几何不被decoder重新生成，而是按该运动变换。

H是历史source context，Q由同一STWM decoder产生，不是第三套历史encoder。**新Refiner直接使用Q，不再像旧CCR那样另接H投影支路**；H的池化输出仍在接口中返回，但不被新主读出消费。

### 3. 公共几何/特征传输

历史几何的source-SE(2)与feature source-SE(2)共享预测运动；未来ego将它们投影到各未来网格。

- P_T：t0静态类别连续ego warp、Strong/CLEAR确定性背景fill、动态source观测点continuous splat及source-order overlay。
- M：同F经学习1×1 lift成保Z低通道特征；静态证据inverse warp，历史source对应证据通过注册后dynamic forward splat，Q加入对应source局部证据。
- 多来源M通过学习importance与归一化汇聚，不按winner owner屏蔽；没有证据的地方使用可学习empty token，不把未知当可靠FREE观测。

代码上`TransportPrior`与`aligned_memory`是两条算子调用，不声称已将所有坐标/矩阵计算完全融合。统一的是运动定义与数据契约；后续共享执行可以优化，不能把未实现的计算复用当贡献。

### 4. 全域语义残差修复

这不是另一个独立encoder反复读取原始四帧；它读取已经搬运的**学习证据M与几何概率P_T**。

输入M16通道+XY池化的P_T18通道→32维；3个XY/Z可分离局部残差块（XY dilation1/2/3）与时距FiLM；原200×200×16全网格分块读取局部32维、fine P_T18维、采样phase3维，经小MLP输出18类ΔZ。

epsilon1e-4有限log prior，末层零初始化；初始argmax等于Transport，不保证训练后永不改坏。训练用完整概率，推理仅最终argmax，同一个连续前向。新增Refiner及其memory适配/融合46,483参数。

Refine的卷积主要在低分辨率XYZ**特征空间**上运行，最后在全部原始**体素位置**输出语义logit；新增/删除/重分类是最终语义与Transport的差分，不再是动作头/阈值或互斥编辑权限。

## 仍保留与已经移除的内容

保留：CC/source关联/注册、刚性SE(2)假设、KTA、Strong/CLEAR、静动态语义划分、P_T的source有序碰撞覆盖、已知/未知与网格边界约定；行人yaw禁用等原类别几何约定仍在。不是全链无规则、可学习检测或全流程可微。

移除的是**修复阶段**的canonical road/source候选白名单、halo、Surface KD-tree描述、静态冲突禁止编辑、ADD/REMOVE阈值、owner/fallback编辑权限。当前全域decoder能够删除与重分类，不需要恢复旧REMOVE头。

缓存兼容复用只读原artifact。新F/M/Q/logits必须随权重重算，不能永久缓存。实际GT路径还复用artifact中**GT-ego条件的确定性Strong背景**，不能笼统称全部缓存字段都是纯历史坐标；它不包含未来GT占据标签，但不能直接迁移到Pred、改未来ego或自预测第二段。测速现场重算所有Strong/KTA。

## 实际监督与两个辅助接口

完整6帧：

`L = mean_h(CE(Z_h,Y_h) + Lovasz(softmax Z_h,Y_h)) + λsource(geometry + existence BCE) + λego R10`

当前默认三个λ均1；FREE17参与CE，不参与0..16语义Lovasz类平均；invalid忽略。

- source geometry：32个历史观测source点按预测/GT位移与有效yaw变换的Smooth-L1；没有有效yaw时translation-only，supervised/valid来自训练标签契约。
- existence BCE：存在性输出仍辅助共同STWM表示学习。**当前P_T与M没有用existence概率作gate，不能画成已控制消失的模块**。
- ego R10：历史ego+导航的小控制头，五个中心/±10m参考点SE(2)误差Smooth-L1。**GT版本poses直接取显式GT条件，所以occupancy loss不经过这个ego头**；它是辅助支路，不是场景planner。GT派生导航只进辅助头，不进scene。

最终占据loss可经连续P_T/M传到F和运动XY/yaw；固定CC、source身份、采样索引/落格选择不微分。existence仍主要依赖其直接监督。新架构只实现GT-only，Pred仍需另立前向/缓存失效协议和场景ego头验证；不要把旧Pred pilot指标迁移过来。

## 证据与应对审稿质疑

TRAIN4086×2/DEV512：Transport39.792319/51.556657/27.109999，Refiner43.170140/57.664138/27.909835（mIoU/IoU/Moving）。只读复测43.170088/57.664131/27.909839。同模型修复收益明显，第二轮Moving相对第一轮小降仍需监测。

累计1/2/3s输出编辑约1.32%体素；新增净语义修正956,651，删除1,335,885，重标13,526。这证明输出不是任意大幅改写且删除有正收益，不等于证明网络不依赖先验、精确因果分解或所有动态编辑有效。Moving区动态新增误报仍是局限。

整体框架合理，但“warp后refine”不是新颖性本身：[DFIT-OccWorld §3.5](https://arxiv.org/html/2412.13772v1#S3.SS5)已有动态flow/静态pose变换与CNN修复；[OccProphet](https://arxiv.org/abs/2502.15180)已有Observer–Forecaster–Refiner分工。

更值得验证的差异是：**source低维运动共同运输观测几何与局部证据，让网络学习少量全域语义残差，而非另预测互不约束的dense flow或受限编辑动作。**这是候选贡献，不宣称首次或已证明不可替代。

最小必要消融：同Refiner/同预算去掉M（geometry-only）；M不按source运动对齐 vs 当前同SE(2)对齐；完整训练和适当种子稳定性。已有同模型Transport读出是功能净增益，不替代独立训练的必要性消融。最终是否适合顶会取决于完整协议对比、差异消融、质量/速度与局限披露，不取决于换更多模块或把规则藏进小框。

当前工程仍是研究原型：`columns`沿用旧命名，trainer复用legacy adapter，H/existence/GT辅助ego含未部署输出。可以后续按冻结契约整理接口，不必现在改网络、扰动断点或重新推倒。GT主结构建议定下来，完整Pred/跨域/6s成果和正式FPS尚未定。

## 最新测速修复

原32窗28计时、4预热fail被跳过；最多4/3,840,000体素差异，仅Refiner，Transport0；旧147ms不能作为全32窗均值。v2分开报告完整人口测量与严格byte身份：每窗完整三次都测，FAIL不筛样本，不放宽阈值；含warm/timed/profile差异与最多16个差异体素logits间隔（检查在计时外）。实际执行诊断FPS可报告，但byte-verified-reference有效标志仍False，不称无损/正式FPS。

新retime.sh只复用旧已完成DEV512/编辑整数计数并重测原固定32窗，不重评分512；原结果/模型/数据只读，新目录保存整窗恢复。布局和旧FPS计时边界保持，新增学习模块全部在内。
