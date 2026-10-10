# 冻结 ego 头：原 TRAIN 缓存与实时接口的一次只读复核

## 已出现的现象，不提前把原因定死

服务器已完成同一 TRAIN1024 bank 的 A/B 各2000步、125轮对照。WM/CCR冻结。

| 候选 | TRAIN ADE / 3s XY (m) | dev64 OCC ADE / 3s XY (m) | TRAIN / dev OCC 3s yaw (deg) |
|---|---|---|---|
| prior | 1.241403 / 2.664260 | 1.416069 / 2.938026 | 5.810 / 6.058 |
| old320 | 1.066669 / 2.285891 | 1.437626 / 2.875380 | 5.711 / 6.042 |
| A_original | 0.178207 / 0.257947 | 1.921248 / 4.087419 | 4.965 / 7.278 |
| B_geometry | 0.192509 / 0.292980 | 1.555724 / 3.167395 | 0.880 / 8.747 |

dev64 OCC外部planner ADE0.645451m、3s XY1.341432m；原头和A/B均未超过外部planner。
OCC平均mIoU external/old320/A/B = 18.350472/16.760100/12.844141/13.600391；
STC = 11.960050/10.928355/9.243519/10.042540。
TRAIN小、重复125轮和绝大多数straight导航，符合明显的泛化失败模式；但先排除缓存/实时提取或batch/模式差异，再判断是否真实过拟合。
本复核不重训、不扩大训练、不自动修正或选头，不重新跑稠密占据预测。

## 一趟检查什么

先在原缓存上复现最终完整TRAIN1024轨迹报告。再从**原来训练过的窗口**确定性选64个，
以3s destination导航right/left/straight轮询，各组内部按scene轮询，不读取任何模型误差来选样本。
因此64是诊断子集，不把其均值冒充原TRAIN总体均值；总体对照仍用完整1024报告。

同一批窗口依次比较：

1. 原bank vs 按原TRAIN reader实时重提的七项历史特征。
2. 原bank vs 按实际dev OCC reader约定实时重提的七项历史特征。
3. 两种实时reader的语义、lidar历史mask、ego pose、实际历史timestamp，以及其特征输出。
4. 原bank标签 vs NuScenes pose重建标签 vs独立JSON catalog重建标签；导航命令逐值核验。
5. 所有既有头的batch64 vs batch1；同一批特征的eval vs train-mode前向。
6. 实时特征/实时导航下的头输出是否重现缓存输出，并报告对误差的实际影响。

train-mode仅是`no_grad`前向，原头dropout=0、无BatchNorm；绝不运行优化器或反向。
非零dropout/BatchNorm将拒绝这一检查，避免把状态变更当只读诊断。
原训练式与dev式提取都只输入四张历史语义/历史mask/历史ego pose，历史timestamps按各自reader读取。
未来pose和离散导航只在两次历史特征提取之后用于**核验监督**；不读取未来occupancy或可见性mask。
本检查不是aligned占据评分，也不输出新的占据指标。

## 保护和判定

原头、A/B、WM/CCR、bank、原contract和已有dev JSON只读；读取前后SHA与冻结模型state核验。
六张必要nuScenes元数据表严格匹配原bank契约，原TRAIN历史文件人口/size/mtime严格匹配，
所选历史NPZ额外SHA核验。设备/Torch/CUDA/精度与SWFM环境使用原bank配置，不混用CPU/F32来审计CUDA/bf16 bank。
几何RAM缓存设0，避免第二种reader复用第一种的中间特征；不写持久几何bank，不运行未来Strong/renderer。

原实验指纹递归包含real_motion以及tools/real_motion顶层Python。新诊断Python放在独立的
`tools/ego_diagnostics/`，Bash/tests/docs不影响原指纹；旧代码一个字不改，也不放宽原bank守卫。
新诊断自身源码和Bash SHA另绑新契约。

七项特征同时报告字节exact和声明的数值误差：float atol1e-6、rtol1e-5；布尔mask严格同dtype/同值。
标签和cmd严格同dtype/同值；头输出容许XY1e-4m、wrapped yaw1e-5rad，所有门槛预先固定，不能看结果调宽。

全部门槛通过：在检查的TRAIN窗口上未发现这些接口不一致，原TRAIN/dev泛化差距仍在；
不等于排除了所有DEV分布变化，也不保证扩大数据一定成功。
有门槛失败：报告具体字段和窗口，停止扩大训练；不自动替换缓存、标签或姿态来提高分数。
每8个完整窗口保存新诊断状态，正常SIGINT/SIGTERM在完整窗口后保存；SIGKILL最多丢最近7个未存窗口。
新输出互斥lease，恢复严格核验同一契约和完整窗口前缀。完整TRAIN缓存复核较便宜，resume会再做，不重新提取已保存的窗口。

## 服务器命令

先更新`feature/v22-surface-aware-ccr`，在OccFM环境执行：

```bash
cd /root/nas/occ/swfm
export EGO_REPLAY_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_train_replay_$(date +%Y%m%d_%H%M%S)
bash tools/real_motion/run_p0_f9_surface_ego_train_replay.sh
cat "$EGO_REPLAY_OUT/summary.txt"
```

默认来源正是已完成的`ego_ablation_20261010_214909_837`，自动核验并引用同名`_eval_dev64/evaluation.json`。
这次只需发新的`summary.txt`；逐字段、逐窗口详情在`audit.json`，完整窗口恢复状态在`state.json`。
来源不靠mtime猜测，可显式设置`EGO_REPLAY_PAIR`；没找到原目录会停止，不启动替代实验。
原bank的SWFM环境在导入模块前恢复，不污染父shell。

```bash
# 中断后，仍指定这次实际输出目录，不生成另一个目录：
export EGO_REPLAY_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_train_replay_实际目录
EGO_REPLAY_RESUME=1 bash tools/real_motion/run_p0_f9_surface_ego_train_replay.sh
```

## 本地验收

相关回归166 passed / 4 skipped（CUDA无设备），包含非零readout头的batch/模式检查、
主动注入字段/标签/cmd/输入reader/报告误差与batch/mode依赖、半窗失败、恢复一致性、
CLI完成恢复和原权重只读保护，以及实际小网格冻结WM/CCR的两个历史reader路径。
本地测试不冒充真实TRAIN64结果，不宣称新的精度提升或FPS。
