# Camera 输入与 Pred. ego 轨迹：调研与拟定评估设置

日期：2026-10-10。本文是调研和接口建议，不代表新增实验已经运行。不会修改冻结模型、现有指标或训练状态。

## 结论

我们的 V18 Transport + Surface-aware CCR 没有 ego 预测头，但可以接外部规划器，作为轨迹条件世界模型评估。最贴合的先例是 COME：外部感知提供历史 occupancy，外部 BEV-Planner 提供未来 ego trajectory；官方模型表对这些设置列出相同的世界模型 checkpoint。因此建议先冻结现有平均网络 `[5,6,8,12,14]`，不重新训练主网络。

这不保证 Camera 输入效果好：感知错误可能影响物体组件、关联、速度及静态证据。先测输入域迁移，再决定是否有必要训练适配版本，不把不同版本混为同一个冻结 checkpoint。

## 主要工作的实际做法

| 工作 | Camera 路径 | Pred. ego 路径 | 对我们的启示 |
|---|---|---|---|
| GAST | STCOcc 先生成 semantic occupancy，然后输入世界模型 | 自己的历史场景特征与 ego-motion 预测未来平移及旋转 | 截图中的 Camera 行不是要求世界模型直接编码 RGB；但其 ego 预测头不等同于外部 planner |
| I²-World | 官方仓库提供 STCOcc-Res 预测缓存下载 | 此处不据此推断其具有预测 ego 能力 | 可优先复用公开 Camera occupancy，避免重新运行或训练感知前端 |
| COME | 官方 BEVStereo Swin-B；EFFOcc 是另一条 LiDAR-camera 融合路线，不能标为纯 Camera | 外部 BEV-Planner，增加 yaw 回归，输出六个 `(x,y,yaw)` waypoint | 没有内置 ego 预测器也能评估 Pred.；其世界模型 zoo 明确复用 checkpoint |

来源：

- [GAST 论文，§3.2、§4.2](https://arxiv.org/html/2608.15279)
- [I²-World 官方仓库，Prepare Dataset](https://github.com/lzzzzzm/II-World)
- [STCOcc-Res 官方仓库链接指向的下载文件](https://drive.google.com/file/d/1dXB9mtROLWChycBZlhYIf_JBLshXogBs/view)
- [COME 论文，Appendix A.2、A.3.4](https://arxiv.org/html/2506.13260)
- [COME 官方仓库，Model Zoos 与数据准备](https://github.com/synsin0/COME)
- [BEV-Planner 官方仓库](https://github.com/NVlabs/BEV-Planner)

COME 的官网更新说明下载迁移到 [Hugging Face](https://huggingface.co/synsin0/come)。本次未下载大型缓存；各 artifact 的实际可访问性、大小、身份覆盖率及格式仍需下载前检查。不要据 README 的链接就假定本地已有文件。

## 四个设置与数据流

```text
四张历史 Camera → 冻结 STCOcc → 四张预测 occupancy ─┐
四张历史 GT occupancy ────────────────────────────┤ 历史输入（二选一）
                                                  ↓
                                      冻结 Transport + Surface CCR
                                                  ↑
外部 planner → 六张未来预测 ego pose ──────────────┤ 轨迹条件（二选一）
六张未来 GT ego pose ──────────────────────────────┘
                                                  ↓
                                  六张 occupancy；报告 1/2/3s 及均值
```

用户已确定 Camera 路线只选 STCOcc，以接近 GAST-STC / I²-World-STC 的前端设置；不实现 BEVStereo 实验，不把不同前端的结果放在同一行。

| 方法/设置 | 历史输入 | Future ego | mIoU 1s / 2s / 3s / Avg. | IoU 1s / 2s / 3s / Avg. |
|---|---|---|---|---|
| Surface CCR-STC | Camera | Pred. (external planner) | — / — / — / — | — / — / — / — |
| Surface CCR-STC | Camera | GT | — / — / — / — | — / — / — / — |
| Surface CCR | 3D-Occ | Pred. (external planner) | — / — / — / — | — / — / — / — |
| Surface CCR | 3D-Occ | GT | 待提取 / 待提取 / 待提取 / 44.153241 | 待提取 / 待提取 / 待提取 / 55.211054 |

最后一行均值来自既有 full4369 记录，不是本次重测。其余数字尚未获得。不能拿 6s 实验的 2569 窗口前段数字填入这张 3s 主表。对应 SVG 只是排版模板，不是可投稿的完整结果表。

## 没有内置 ego predictor 时的具体接口

优先接 COME 发布的带 yaw planner 输出（若能确认来源、格式及样本覆盖），而不是给我们的主网络追加预测头。它用 expert trajectory/yaw 作训练监督不等于测试时输入 future GT；重点是推理输入只能来自当前/历史信息。

实现时必须明确：

1. 每个窗口六个 waypoint 对应 0.5/1/1.5/2/2.5/3s。已核对 COME 官方 loader 和公开 JSON：XY 为世界坐标，yaw 为相对当前 t0 的偏移，非相邻步增量。loader 对每个未来姿态分别使用 t0 旋转，不累积 yaw。旧 `camera_pred` 缓存的 `future_e2g` 已保存该六帧世界姿态，可经身份/数值审计后复用；不复用旧 latent/anchor/proposal。
2. 所有依赖未来 ego 的投影、query footprint、Strong/KTA、CCR 候选和合成统一使用预测 pose。不能只在最终 renderer 改 pose，却保留 GT trajectory 构建的中间输入。
3. 只有 XY 的规划器不构成完整预测 pose；优先使用公开带 yaw 的版本。若只预测 SE(2)，z/roll/pitch 从当前状态保持的近似要明确披露，不使用未来 GT 补齐。
4. 预测轨迹必须按 `(scene_name,t0_token)` 严格对齐，缺失/重复直接报告，不按列表序号猜测，更不能回退到 GT。
5. 主 Pred. 评估不拿 GT future pose 将输出事后对齐，也不偷偷重定位 GT 标签。若另做 coordinate-normalized 诊断，与主表分开。
6. Planner 的图像、ego status、导航命令等输入必须审计并披露。特别是 3D-Occ + Pred. 行若 planner 使用 Camera，表下注明额外规划输入；不能声称整个系统仅依赖 occupancy，也不能按未来 GT 选择最佳轨迹 mode。

历史 ego pose、相机标定等测量信息允许使用，但不能把未来 pose 藏入特征或缓存。外部 planner 的精度与条件和 GAST 自带预测头不同，因此这些行反映完整系统而非相同 planner 下的世界模型单因素比较。

## Camera 路径的防泄漏与公平比较

- 必须替换全部四张历史 occupancy；组件、类别、形状、关联、静态记忆和速度从 Camera 预测重新提取，不能复用 GT occupancy 历史特征缓存。
- 不读取未来 RGB 或未来 sensor visibility。公开 STCOcc 缓存也要审计时间因果性、网格范围、坐标轴、类别映射和 token 对齐。
- GT occupancy/annotation 只用于评估目标和指标支持，不作为 Camera 路径的建模输入。历史 GT `mask_lidar` 不能伪装成 Camera 自身可见性证据；Camera 历史有效性规则必须明确且只依赖允许输入。
- COME 明确在 sensor-input occupancy generation 评估中使用 `mask_camera`，当前我们的主表无 camera/lidar mask。这是评分口径差异，必须分开报告或在明确对齐后比较，不能混用数字宣称同协议优胜。仅作为 scorer 使用的 GT mask 与作为模型输入的 GT mask 是两回事。
- 若预计算 Camera occupancy，质量评估可使用缓存，但端到端延迟必须包含 Camera 前端，不能直接沿用 occupancy-input 快速后端的 FPS。

### 用户冻结的 STC mask 决策（2026-10-10）

按 I²-World-STC 公开实现对齐，不另开 mask 调参实验。官方 `LoadStreamOcc3D(dataset_type='stcocc')` 读取 `stc-results/.../labels.npz` 的 `semantics`，不读取或应用 camera/lidar mask；官方 forecasting scorer 的 `use_lidar_mask` 与 `use_image_mask` 均为 False，评分调用传入 None。
因此使用发布的完整四帧 STC 语义预测，不借用历史 GT mask_lidar/mask_camera；适配器所需 known 标志表示完整预测网格有效，不声称真实传感器可见性。未来评分也不加 camera/lidar mask，不复制 COME/BEVStereo 的历史置 free 规则。STC 包若在生成时已有处理，保留其发布内容，不擅自恢复或额外遮罩。
这只确认公开 mask 接口，不代表论文各行的精确人口/前端权重完全相同。人口、token、类别、网格以及预测姿态仍要自动核验。
官方依据：[loading.py](https://github.com/lzzzzzm/II-World/blob/main/mmdet3d/datasets/pipelines/loading.py)、[nuscenes_world_dataset.py](https://github.com/lzzzzzm/II-World/blob/main/mmdet3d/datasets/nuscenes_world_dataset.py)。四设置 evaluator 已实现及本地回归，服务器真实 Camera 质量未跑；下载、紧凑解压、评估与续评见 [操作文档](SURFACE_STC_CAMERA_PRED_EGO_CN.md)。

## 最省事的验证顺序

先检查两份公开 artifact（STCOcc 历史预测、带 yaw 的 planner 结果）及身份覆盖。接着共享同一 dev64 population，一趟检查 Camera+GT、Occ+Pred、Camera+Pred，并复核 Occ+GT 不变，不进行阈值搜索或 checkpoint 选择。确认接口没有泄漏/错序后，再在同一完整 3s population 跑四种设置。

若某份公开 artifact 缺失窗口，先打印缺失及共同交集，不悄悄改变人口；共同交集上的既有 Occ+GT 必须重测。小样本用于接口诊断，不冒充论文全集结果。是否训练新的适配器/ego head，留待这些冻结评估结果出来后另行决定。
