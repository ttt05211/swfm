# 感知噪声鲁棒训练：独立两轮快速对照

## 与 e9a3578 诊断不重复

该提交冻结正式 GT epoch2 和 head10，查 STC 身份/轴/known/错误先验/ego 场景敏感性。
本实验**训练新模型**验证历史污染增强，不重新做这些反事实或根因统计。
只新增 `tools/perception_robustness/`、本说明和专项测试；正式模型、full_gt/refine trainer、final_semantic_evaluation/diagnostics 文件不改。
不并行启动 GPU 实验，不修改服务器正在运行的程序，也不自动融合另一分支。

## 固定定义

- A clean / B robust：同一当前 GTBody，SharedEncoder + STWM + 18 类 SemanticRefiner，没有 ego 头。
- 各自相同 RANDOM seed=20261011、scene-balanced TRAIN4086（20%）、完整两轮、全周期余弦、全六未来 CE+Lovász，辅助运动/existence/ego 全关。
- 不是原正式完整 TRAIN20430 模型的微调或同预算对照；正式 checkpoint/分数不变。
- 同原 batch4、按干净 record 的 source128 **名义分组**，完全相同顺序/更新次数。污染后实际 source 数可能变化，单独记录；不为维持上限偷偷删输入或重分组，OOM 则停留在旧安全 checkpoint，不跳窗。
- B 每窗50%保干净、50%选择一种固定错误：区域漏占据 / 区域类别混淆与局部假表面 / 小几何错位与帧间抖动。四个 t0 世界空间 patch，XY边长2–6m、Z边长0.8–2.4m，持久XY偏移±1体素、逐帧XYZ抖动±1体素。
- 每帧改动最多原已知 occupied 数15%，超限整patch拒绝，不随机散点裁剪；选择但未实际改变的窗口另计，不冒称污染成功。配方预先固定，不根据 DEV 挑强度/类别/阈值。
- 修改历史已知有效体素；known 原值不变，不送污染位置提示。它是 **GT known 下的语义鲁棒性首轮实验**，未同时解决 STC all-True 的 known 域差；该变量由另一诊断单独检查，不宣称模拟真实 STC 全部误差。
- patch 在 t0 坐标定义并按每帧实际历史姿态投影，有时序相关性，不给整数 class ID 加高斯噪声。
- 独立 SHA256(seed/epoch/scene/t0) RNG，不消耗训练 Torch/NumPy/Python RNG；恢复同epoch/window复现增强。
- future GT占据始终干净，只作监督；GT未来ego是显式条件；不读取未来annotations/masks作预测。

## 缓存与效率

旧 frame/metadata/干净 geometry 只读复用。实际改变的 B 窗口从污染历史重新提取当前/过去组件、关联/配准、KTA、source crops和memory；主线程重算六时距 Strong/CLEAR背景。
不会把干净缓存的source/背景给污染路径，不把随机污染几何写入旧缓存或建立新feature bank。CPU最多4个窗口线程；只缓存原确定性历史。
训练增强新增准备成本，记录 CPU geometry/augmentation wait/fresh Strong 与实际 source 数；不承诺训练无损提速。网络/部署路径不变，但新权重精度与FPS需独立验证，不能继承旧FPS。

## 评估与恢复

每路固定最终epoch2，只评原DEV64同一人口的 OCC+GT / STC+GT，两者均报告Transport/Refiner、1/2/3s与平均mIoU/IoU/Moving。
真实STC仅评估，所有64窗必须有四历史，不丢/补窗；STC全semantics/all-True known，无GT camera/lidar mask或alignment。未来标签只进入loss/metrics；不训练或校准验证集。
根comparison列robust-clean及Refiner相对Transport增益，不自动挑路线/扩大DEV512或完整训练。
两路顺序运行，不互相学习。Ctrl-C保存完整更新/评估窗口；同根目录恢复，不重置Adam/RNG/LR，完成一路不重跑。硬kill只能恢复最近周期checkpoint。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
unset RESUME ROBUST_RESUME
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/robust_history_20pct_2ep_$(date +%Y%m%d_%H%M%S)"
export STC_ROOT="$PWD/data/stc_camera/compact"
bash tools/perception_robustness/run.sh
cat "$RUN_DIR/summary.txt"
```

可以先 `bash tools/perception_robustness/run.sh --max-updates 8` 验证启动；A/B各跑前8步并保存，均属于原两轮预算，不改变余弦周期、不重复随机初始化。然后用同目录恢复：

```bash
export RUN_DIR=/刚才原来的实验根目录
export ROBUST_RESUME=1
unset RESUME
bash tools/perception_robustness/run.sh
cat "$RUN_DIR/summary.txt"
```

不修改正式模型/其他诊断目录。请在其他 GPU 训练/诊断结束或安全暂停后运行，避免资源争用；脚本不会停止其他进程。

## 本地验证边界

专项测试覆盖原数据不变、known不变/修改上限、历史标签毒化、局部RNG、实际组件重建/Strong/完整网络反传、整数续评、两路调度和实际小网格训练游标恢复。
本地没有真实nuScenes/STC全集，不能据合成测试承诺STC涨点或L40S吞吐。真实质量由上述一次固定预算实验判断。

新专项12项＋既有full GT/正式评估接口28项，40 passed in12.72s；含实际CPU/CUDA小网格、改变后无source与原source梯度链回归、真实小网格CLI安全停/恢复及A/B首8步流程。CLI help、Bash语法检查通过。未执行服务器训练或改正式权重。
