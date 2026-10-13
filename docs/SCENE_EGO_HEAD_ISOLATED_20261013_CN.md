# 独立场景感知 ego 头（2026-10-13）

## 范围与隔离

按用户要求先独立实现，后续融合。不修改 `shared_history_experiments/refine/`、
`refiner_pipeline/`、`loss_minimal/`、旧 ego 头或任何 trainer；无全局 hook/monkeypatch。
新代码仅在 `tools/scene_ego_head/`，测试仅 `tests/test_scene_ego_head.py`。
未启动训练、评估服务器数据、推送或写入原权重/optimizer/缓存。

这是一份可执行的组件及 Pred 前向适配器，不是已接入的正式训练入口。
不能把旧 GT/Pred 断点直接 resume 成新结构，也不能沿用旧正式 FPS/质量数字。

## 结构与实际规模

复用 SharedEncoder 已算出的 `F[4,64,X/2,Y/2]`，不新增感知编码器。
四帧各 coverage-weighted 池化到8×8，共256 tokens。保留逐帧特征、
t0坐标位置、实际历史时间、有效观测比例；不将场景全部平均为一向量。
坐标由历史 ego pose 转到 t0；这是历史坐标编码，不是未来 GT 对齐。
无效 tokens 被 attention mask 排除；null key 保证全未知场景仍能有限预测。

ego四帧自身历史及历史运动学状态形成128维query，一层cross-attention读取场景，
随后轻量融合和 `Linear(128,64) → Softplus → Linear(64,54)`。
默认整个新增头 **143,094 参数**，不包含已共享的历史 Encoder。

54个输出为6步×3模式×(Δx,Δy,Δyaw)的运动学初值残差。末层零初始化，
初始轨迹为已有历史运动学先验；第一步末层可学习，更新后场景读取的上游梯度打开。
这不是保证后续不退化或一定超过外部 planner。

模式顺序沿用right=0/left=1/straight=2，支持严格整数及one-hot。
每一步先按该步命令选择增量，再累计；允许命令变化，但不会拼接不同模式的绝对轨迹。
XY增量统一在固定t0轴中，不能再按逐步yaw重复旋转；yaw累计不强制wrap，几何loss用周期旋转。
轨迹为0.5/1/1.5/2/2.5/3s，保留t0世界高度与tilt，输出SE(2)，不是GAST的完整四元数旋转头。
导航仍是显式benchmark条件，若用既有GT派生destination命令，必须报告navigation-conditioned。

## 接口

```python
from tools.scene_ego_head.model import SceneEgoHead, prepare_scene

head = SceneEgoHead().to(field.device)
scene = prepare_scene(
    field,                         # [4,64,X/2,Y/2], 本次live历史Encoder输出
    observed=historical_validity,   # [4,X,Y,Z] bool，需排除未知语义编码
    history_poses=history_poses,    # 只包含四张真实历史
    ego_history=ego_history,        # [4,10]，已有单位/时钟约定
    grid=grid,
    pool_side=8,
)
result = head(ego_history_tensor[None], commands_tensor[None], scene)
# result['se2']: [B,6,3]，t0相对绝对XY/yaw
# result['branch_increments']: [B,6,3,3]，不是绝对位姿分支
```

`prepare_scene`当前一次窗口；头本身支持同规格tokens组批。池化坐标按真实adaptive
bin边界计算，100→8非整除不假设均匀12.5cell中心。高度由原Encoder的逐高度输入编码进F，
token坐标的z=0是该帧BEV参考平面，不宣称每个token含独立3D物体高度。
没有learned feature bank、persistent scene cache或未来条件特征输入。

## 独立 Pred 接线

`integration.SceneEgoRefiner`复用现有SharedTransport和SemanticRefiner类，构建独立新模型。
`forward_pred_window(model, provider, row)`只消费现有纯历史source/memory读取和ego历史/cmd。
它显式忽略row中的ego target、未来GT pose/占据/annotation/mask及旧GT背景。
只计算一次共享Encoder，先预测ego，再让同一预测pose进入连续Transport prior、
source投影与aligned feature memory、最终全六帧Refiner读出。

背景由`rebuild_pred_backgrounds`使用历史+**本次预测pose**重新运行Strong/KTA/CLEAR，
不复用GT未来缓存。这一离散步骤显式detach，不声称全链可微；连续inversewarp/splat
仍向ego XY/yaw和共享F反传。未来GT只能在forward返回后进入直接ego监督或CE/Lovasz。
函数允许显式注入背景builder用于融合与测试，不能给正式Pred注入GT缓存或oracle背景。

后续融合时：保原共享Encoder/STWM/Refiner，替换ego并传入历史scene；把目前GT-only
future背景准备改为上述Pred规则；沿用完整6 CE+Lovasz与直接ego R10，辅助loss策略由
另一实验确定，不在本组件中擅自改。新结构/协议/配置/依赖SHA进入独立断点契约，
保存Adam/RNG/余弦/游标；现有GT入口、运行中的提速和损失控制实验保持不变。

## 验证与限制

```bash
# 已具备本仓库依赖的OccFM环境；不启动训练
python -m pytest -q tests/test_scene_ego_head.py
```

本地最终新组件13项、与原Shared/Refiner合计36项通过，含真实CPU/CUDA小网格
完整occ反传、头/位姿/F梯度、未知mask、空间敏感性、先选模式后积分、
零初始化→上游梯度打开、CPU头Adam精确恢复、GT未来及旧背景毒化、历史只编码一次。
包含真实CPU Strong背景的完整Pred接线验收；没有用未来GT背景或预测轨迹oracle。
默认头参数实测143094。使用windows-python-env-guard的非base yoloe环境，
缺失纯Python测试依赖仅放workspace `tmp/scene_ego_test_deps/`；旧upstream OccFM
只读路径用于本地旧模块导入，没有改Python环境或旧源码。

这些不是nuScenes质量、L40S吞吐、正式FPS或全仓CI；新头尚未训练，不能承诺指标收益。
尚无正式训练/评估/续训runner；等另一实现稳定后融合，不在这里重建第二套trainer。
