# 冻结 Surface CCR：STC Camera / Pred ego 四设置

本版只做评估，不新增训练，不改 mean5/6/8/12/14 权重、ADD0.5/REMOVEoff 阈值或现有主表。
一趟运行 `occ_gt / occ_pred / stc_gt / stc_pred`，报告 1/2/3s 与均值 IoU、标准 mIoU。

## 固定协议

- 四张历史包含 t0，六个未来 keyframe，名义2Hz；不做6秒递推，不启用第二段 static carry。
- Camera 前端仅采用 [I²-World 发布的 STCOcc-Res](https://github.com/lzzzzzm/II-World#-prepare-dataset)。全部历史语义从 STC 重新提取组件、Strong、速度、关联及 CCR 证据，不复用 GT learned geometry 或旧 BEVStereo latent/history/proposal。
- 按 I²-World-STC 公开 loader 使用完整 STC `semantics`；全预测网格 valid 不代表真实 sensor visibility。Camera 路径不读取任何 GT 历史/未来 mask。评分四设置均无 camera/lidar mask。Occ 历史保留当前预测器既有历史 `mask_lidar`。
- Pred 只读取已存在 v4 `camera_pred` 缓存的身份和六张 `future_e2g`。官方 planner 原 JSON SHA必须匹配 `19c04eaf37f531148d5b5719e3cabf8caa35a7140bba9527df47c79aa6783afb`；该原文件路径失效不影响已序列化姿态。
- 六张预测世界XY、相对t0独立yaw用于全部未来投影/KTA/CCR/合成；z、tilt只能继承当前t0。严格核验姿态、token/order/tag及历史/未来完整性，不回退GT，不用GT未来姿态事后对齐。未来GT语义仅四设置的完整六帧预测都完成后读取。
- Planner自身来自外部Camera/ego/navigation系统，不是本方法ego预测头。`3D-Occ + Pred` 行也必须披露外部planner的额外输入，不能标成整个系统纯occupancy。
- 主列为标准mIoU，保留有union但IoU恰为0的类别。另列 `I2_code_mIoU` 按其公开代码排除恰零类别；不得择高混报。
- 四行使用同一 planner-covered population。旧缓存4219窗/150场景是旧6历史起点人口，本模型仍只读最后4张；不是旧full4369，也不声称论文起点集合完全一致。不能把旧4369的Occ+GT分数填入这次表格。
- dev64为冻结parent512与planner身份的明确交集再按场景round-robin取64，所有缺失身份记录到contract；all为完整缓存人口，任一必要STC/GT/planner文件缺失都报错，绝不静默丢窗。

## 1. 更新及下载（服务器 OccFM）

下面 `data/stc_camera` 和 `stc_four_*` 是本次**新建**路径，不是声称已存在的数据。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
(
  set -euo pipefail
  git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
  git merge --ff-only FETCH_HEAD
  git rev-parse HEAD
)
export PY="$(command -v python)"
export STC_ZIP=/root/nas/occ/swfm/data/stc_camera/stc-results.zip
export STC_ROOT=/root/nas/occ/swfm/data/stc_camera/compact
export STC_PLAN_CACHE=/root/nas/occ/new_code/cache/come_main_table/camera_pred
export DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes

"$PY" -c 'import gdown' || "$PY" -m pip install gdown
bash tools/real_motion/download_stc_camera_data.sh
```

官方Drive文件约68MB（67,779,055 bytes），普通展开约19.26GB。下载不覆盖已有ZIP，失败保留.part。服务器若不能访问Google Drive，用浏览器下载[同一官方文件](https://drive.google.com/file/d/1dXB9mtROLWChycBZlhYIf_JBLshXogBs/view)，上传到上述STC_ZIP，再运行下载脚本即可校验；不要把ghfast当作Drive代理。没有下载或运行STCOcc网络、raw RGB；这是离线前端预测缓存评估。

## 2. 无损紧凑解压

```bash
"$PY" -u tools/real_motion/prepare_stc_camera_data.py \
  --archive "$STC_ZIP" --out-root "$STC_ROOT"
```

解压逐文件检查外层ZIP CRC、成员路径、200×200×16整数0..17；仅保留uint8语义并重新压缩，逐值不变，不应用mask、不重映射类别。避免保存官方NPZ中无用大数组；原ZIP保留。首次必须新目录，已有未完成输出用同一命令加 `--resume`，按同ZIP SHA及逐帧语义检查，绝不悄悄覆盖用户已有数据。完成打印 `STC_READY`，目录下直接为scene-XXXX，不额外嵌套stc-results。

## 3. dev64一趟四设置

```bash
export STC_POPULATION=dev64
export STC_OUT="/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/stc_four_dev64_$(date +%Y%m%d_%H%M%S)_$$"
(
  set -euo pipefail
  bash tools/real_motion/run_p0_f9_joint_surface_stc.sh 2>&1 | tee "${STC_OUT}.log"
)
cat "$STC_OUT/summary.txt"
```

自动复用当前 `full20_nohup_resume_20261008_190410_787` 对应已完成的均值comparison bundle，不重新平均或选checkpoint。若用户移动该run，显式设置 `SURFACE_RUN_DIR`、`SURFACE_MEAN_SOURCE` 或 `STC_CHECKPOINT` 到**确认存在的冻结mean文件**，不要猜路径。DEV子集默认读 `$ROOT/data/p0_f9_v21_dev64_manifest.json` 的parent_keys；可显式 `STC_POPULATION_MANIFEST` 指向已有同协议manifest，缺失不造人口。

启动 `STC_SHARED_POPULATION` 后preflight逐窗核验身份及姿态，然后运行模型；首个窗口四设置分别做输入/Transport/probability/六帧输出exactness。每16窗打印一次进度。可先加 `STC_AUDIT_ONLY=1` 到新输出目录仅检查文件/首窗输入，无需CUDA或模型；非必要，不另强制一次验证。

## 4. 完整共同人口四设置

```bash
export STC_POPULATION=all
export STC_OUT="/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/stc_four_all_$(date +%Y%m%d_%H%M%S)_$$"
(
  set -euo pipefail
  bash tools/real_motion/run_p0_f9_joint_surface_stc.sh 2>&1 | tee "${STC_OUT}.log"
)
cat "$STC_OUT/summary.txt"
```

`all`不是2569个6秒窗口，预计现有planner4219个3秒窗口，数量以实际打印为准。保存summary、evaluation.json、contract.json（有精确人口/输入来源/权重与代码指纹）、progress及整数计数state。这里只是实际质量评估，不声称原始Camera端到端FPS。默认native并行投影/多数投票、验证Graph及512MiB帧LRU；不写几十GB几何缓存。

运行时资源/候选质量随STC假物体数量变化，不把旧GT速度硬当作STC ETA。可从16–64窗的实际 `seconds` 外推剩余时间；四设置总工作量不是单模型单设置的时长。

## 5. 中断续评

Ctrl-C/SIGTERM等四设置当前窗口完成后保存，不使用kill -9。默认每8完整窗周期保存，异常退出恢复最后实际state，日志里未持久化的尾部会重算但不重复计数。保留原 `STC_OUT` 和 `STC_POPULATION` 后：

```bash
(
  set -euo pipefail
  STC_RESUME=1 bash tools/real_motion/run_p0_f9_joint_surface_stc.sh 2>&1 | tee -a "${STC_OUT}.log"
)
cat "$STC_OUT/summary.txt"
```

新终端需要重新激活OccFM、设置STC_ROOT/PLAN_CACHE/DATAROOT及**原输出路径**、原population。不自动找“最新目录”；数据/权重/人口/代码/执行参数改变会拒绝续评。旧训练checkpoint、optimizer/RNG、旧缓存与结果保持只读。

## 验证边界

本地新增STC与既有Waymo/geometry-carry相关回归63 passed / 4 CUDA skipped；包含真实CPU Surface四设置、NumPy与native概率/完整六帧exactness、GT隔离、错序/缺失拒绝、压缩语义不变和整数续评。CLI help与Bash语法通过。未代跑服务器真实STC质量；未声称本地CUDA、全仓CI或论文提升。
