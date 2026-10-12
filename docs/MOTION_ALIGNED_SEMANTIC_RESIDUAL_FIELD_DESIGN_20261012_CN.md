# CCR整体替换候选：运动对齐语义残差场（2026-10-12，仅设计）

## 用户要求与范围

用户提供跨领域refinement附件及Transport-then-Refine建议，希望去掉CCR中ADD/REMOVE、owner/fallback、手工Surface和类别候选约束，优先质量，不照抄GAST。附件作为参考，不作为自动实施指令。本轮阅读附件、核查当前代码与原论文，未修改模型/训练/缓存/评估入口、未启动新实验或推送。本设计是下一版独立候选，不改变正在进行的两轮REMOVE验证，也不把旧论文分数/FPS移植过来。

## 原论文支持的借鉴与边界

- [PointRend](https://arxiv.org/html/1912.08193)：fine features+coarse predictions进行point分类，推理迭代细分并选择不确定点；训练保留均匀覆盖，过度只采不确定点可能更差。不是“无需任何采样设计”，也不保证高置信错误会被选中。
- [IF-Net](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chibane_Implicit_Functions_in_Feature_Space_for_3D_Shape_Reconstruction_and_CVPR_2020_paper.pdf)：多尺度三维feature space中query occupancy，借局部几何表达而非单全局向量。任意位置可查询不等于全网格查询免费。
- [Softmax Splatting](https://openaccess.thecvf.com/content_CVPR_2020/html/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.html)：可微forward warp及带重要性的归一化解决多来源碰撞，可借为运动证据融合，而非保证复原三维遮挡真值。
- [Mask Transfiner](https://arxiv.org/html/2111.13673)：学习incoherent regions、quadtree与节点分类；论文COCO实验中区域占14%但覆盖43%错误，不能外推我们“99%体素不用改”。
- [OccProphet](https://arxiv.org/html/2502.15180)：历史/未来feature交互的残差refiner，Table5去Refiner掉0.8为其任务的IoU指标，不是我们Occ3D标准mIoU预计收益。
- [DFIT-OccWorld](https://arxiv.org/html/2412.13772)：动态/静态flow解耦、warp及feature refinement；[GAST](https://arxiv.org/html/2608.15279)已有pose warp/modulation/deformable cross-attention。因此“warp+refine”不能单独作为新的贡献。
- [GSDN](https://arxiv.org/abs/2006.12356)是3D detection的generative sparse decoder，不是occupancy forecasting证明；[SG-NN](https://openaccess.thecvf.com/content_CVPR_2020/html/Dai_SG-NN_Sparse_Generative_Neural_Networks_for_Self-Supervised_Scene_Completion_of_CVPR_2020_paper.html)才是场景补全方向。生成/剪枝支持依然包含stride/kernel/阈值或预算、非可微离散选择，不是消灭所有设计。

## 对附件的关键纠正

当前Shared联合ADD-only/新REMOVE均采用连续composition→argmax，不再使用ADD0.5/REMOVE0.95硬动作阈值。阈值字段只是旧接口metadata；真正剩余负担是候选/类别限制、Surface统计、owner/fallback和有序动作合成。

仅将两输出改为18类并保留原source+road/sidewalk+halo域，仍保留候选上限；移除fallback却仍按source逐个覆盖，会保留last-write冲突。未来voxel汇聚后一次预测是有价值的职责改变。

“车A占据而GT车B占据”的例子若语义都是car，semantic occupancy已正确；18类头解决不了实例身份。真实新能力是car→FREE、road→sidewalk、car→truck等最终语义重新判别，不是自动实例跟踪。

零初始化保证初始残差零，不保证训练后不损坏正确区域，也不限制最终一定不重生成。不额外默认增加一致性loss/门控/纠错器来保证这个命题。历史支持只是输入证据，不能宣称所有输出都来自真实观测；全域可修正后允许学习补全未知区域，需要承认其不确定性。

## 推荐首版：保留运动底座，完全替换修复语义

```text
4帧历史occupancy+known+历史ego poses
                    ↓
            共享height-aware F
           ↙                  ↘
 source局部STWM          纯历史空间证据
 SE(2)运动+未来Q          （不另做KD-tree Surface12）
           └───────┬──────────┘
        同一运动/ego坐标投影
             ↙             ↘
      Transport概率P_T    运动对齐空间memory M_h
             └──────┬──────┘
          统一语义残差解码器
     每个future voxel输出18类Δz
                    ↓
        softmax(log P_T + Δz)
                    ↓
             最终六帧occupancy
```

### 1. 不另造历史encoder

保留现有共享F和source geometry/历史关联。用轻量learned projection将F中的高度编码显式展开为低通道三维特征；初始候选100×100×16、16channels，不max/mean掉高度。它只是共享特征的适配，不新增一套独立原始历史编码器。

### 2. 几何和学习证据使用同一source motion

动态source几何与关联局部feature都使用STWM的同一预测SE(2)；静态使用ego变换。历史ego配准、t0 source关联保持因果，没有未来GT source/annotation参与。动态时间Q随对应证据投影，不把整个source向量无位置地广播到全场景。

多个来源投到同一future位置，用共享重要性/attention及可微归一化聚合，形成M_h。不得“先选owner再只取那个source”。未知/无证据位置使用明确空证据状态；unknown不等于observed FREE。这个融合器借鉴soft splatting，不声称几何碰撞/遮挡从此已解决。

首版P_T可保留已验证的连续Transport概率作为粗先验。因此Transport内部原物理碰撞约定未必全删除；必须清楚区分“去掉修复权限/owner/fallback”与“整个系统完全没有source顺序”。后续如将粗先验也改为learned融合，要单独比较，不暗改基线。

### 3. 一个future voxel一次修正，可覆盖整个grid和全部类别

M_h上轻量局部三维/XY+Z可分离残差块聚合邻域；同一query decoder读取局部M_h、P_T、位置/高度和时距，输出18类修正。首版不按road/sidewalk筛点、不用canonical halo作许可、不输入REMOVElegal/fallback、不冻结原正确占据。

全域读出用低通道场+分块decoder，不建六张200×200×16高通道3D U-Net。它是全网格语义输出，不是全网格高维计算。每个query有空间邻域上下文，不退回“手工descriptor+独立二分类器”。

输出：P_final=softmax(z_T+Δz)，z_T=log(P_bar)，P_bar=(P_T+epsilon)/(sum_c P_T+18epsilon)。必须有有限平滑，因为hard/soft prior可能零概率。Δz末层零初始化，则初始P_final=P_bar、argmax与原P_T一致；不能宣称raw概率未经平滑也逐字节相同。epsilon需预先固定，不从dev调参。

无需动作阈值：FREE→car是新增、car→FREE是删除、road→sidewalk是重标，都是最终argmax前后的差分统计。空间碰撞在evidence汇聚解决，refiner最终只预测一份voxel类别，不回退固定fallback。

### 4. 首版保持原最终监督，不新增目标堆叠

CE+Lovasz监督完整六future，运动几何/existence辅助保持；当前ego R10辅助也先保持预算可比，GT主配置使用GT future ego。Pred后续使用内部/外部预测pose，严禁GT future pose前向对齐。连续feature投影、概率prior与残差可将梯度传回F/STWM；离散CC/source关联/整数索引仍不微分，不包装为全部离散准备可微。

先不加动作BCE、修复区域BCE、删除阈值搜索、teacher/蒸馏、默认Transport一致性loss。新架构需要独立权重/optimizer/协议，不能把旧CCR断点直接当resume。

## 为什么首版不直接Sparse+pruning

Sparse预算会重新引入候选漏检；Transport把错误FREE/road预测得非常自信时，entropy/边界采样可能根本不访问该位置。先全域query验证能力，是算法质量对照；若质量成立，再借PointRend/Mask Transfiner做adaptive细查询，同时保留全域粗修正。后续不能悄悄改成仅旧halo内的小分类器，必须报告error-region coverage与质量/速度差。

生成稀疏转置卷积/剪枝可列候选，但首版不增加剪枝头、阈值和额外selection loss，以免一起改动太多。

## 与已有方法的差异是假说，不是新颖性承诺

拟验证的是：由同一低维source motion共同运输观测几何及局部空间证据，比仅ego warp/通用flow refinement更可靠，统一voxel解码比owner/fallback动作链更能处理碰撞和错语义。source-aware feature warp、隐式query和残差均已有工作，组合不能保证创新。

至少对照：相同共享encoder/STWM与预算的旧CCR、新残差decoder；新decoder only P_T versus P_T+aligned spatial features；必要时纯ego对齐 versus共同source motion。比较Transport、最终结果、改对/损坏和静态/动态类，避免一次加很多模块再宣称每项有效。

## 性能与数据风险

200×200×16×18×6的FP32输出约264MiB，仅是一个概率/logit张量，不含grad/activations/排序。不能因为Z=16就承诺dense够快；分块、六时距合批范围及真实后向都须replay测。新头可省KD-tree/候选plan/动作loop，但会增加learned volume/query成本，可能更快也可能更慢；旧54.945FPS/H6训练0.354秒窗不移植。

训练与推理使用同一残差概率读出，首版没有只训练soft部署hard的替换。新learned投影/feature aggregation/decoder全部计时，不移到预处理外。兼容的历史原始数据/纯几何可只读复用；新learned features不能持久缓存。

全域refiner提供更大的可修复空间，不保证STC/Pred差距自动消失；真实感知输入域迁移、未来ego位置误差、未知新生物体仍是独立难点。不能用未来GT visibility/mask/occupancy作特征补救。

## 当前状态

仅调研与设计，没有实现、训练、验证收益、修改正在跑的REMOVE实验或推送。是否实施以及是否沿用20%两轮预算，等待用户明确决定；原已完成ADD-only及正式Surface结果保留。
