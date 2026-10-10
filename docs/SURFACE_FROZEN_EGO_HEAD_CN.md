# 冻结 Surface CCR 的内部 ego 轨迹头

## 范围

现有 mean5/6/8/12/14、V18 Transport 和 Surface CCR **均冻结**，原 GT / 外部 BEV-Planner / STC 四设置不修改。新实验仅训练一个独立轻量 ego 读出，不重新训练 GT 版本，不修改科学主表或旧缓存。性能和 FPS 尚未在服务器验证。

数据流：

```
四帧已知 occupancy + 历史观测 + 历史 ego pose
    ├─ 原 V18 历史编码 → 物体 context + 历史位置/速度
    ├─ 原 CCR 历史 encode → 8×8 空间静态 context
    └─ 已知 ego 历史位移/角速度
           ↓ 六时间 query × right/left/straight 三分支
       ego 头：各未来时刻相对 t0 的绝对 XY、yaw
           ↓ 明确的导航分支选择
       六个预测 ego pose（z/tilt 保持 t0）
           ↓ 所有未来几何统一使用预测 pose
       Strong + 原物体 WM/搬运 + CCR 投影/phase + 六帧输出
```

物体 WM 输出物体运动残差，ego 头输出自车坐标系运动，两者不是重复的预测对象。历史特征提取在未来投影之前完成；物体 WM 和 CCR 的 future decode/phase 不作为 ego 特征。静态 token 保留 BEV 空间位置；最多64个物体、每个静态格最多64个历史点仅限 **ego 读出池化**，不截断正式 WM 的物体或 CCR 候选。缺失静态/物体 token 明确 mask，四个历史 ego token 始终存在。

默认头宽128、两层decoder、4头attention，新增 **427,907参数（约0.43M）**。独立head checkpoint不包含或替换原网络。

当前验证入口优先正确性与共享历史复用；ego 特征抽取与正式第一路预测仍有部分重复编码。这不改变概率/原网络，但新增头与抽取的时间须另测，不沿用旧后端54.945 FPS声明。

## 指令和监督

导航类别依照 [I²-World 官方 converter](https://github.com/lzzzzzm/II-World/blob/661d830f9b34ee03ce368db164a72753ab8764a3/tools/data_converter/nuscenes_converter.py)：在各行当前 LiDAR 坐标系中，未来六个 keyframe 的末端 x≥2m 为right，x≤−2m为left，其余straight；场景末端重复最后可用 pose。六个 query 使用未来 **destination 行** 的指令，采用 [GAST 的数据组织方式](https://github.com/chenst27/GAST)。

这是一种 **未来 GT 派生、导航条件 Pred**，不是“无导航仅历史”设定；其指令可能参考 t0+3s之后的 metadata。离散指令作为公开条件，但不把连续未来位姿、GT occupancy、未来 mask 或 `gt_ego_lcf_feat` / 未来 CAN-bus 回退输入模型。自车历史特征仅由四帧已知 pose 和实际历史 timestamp 计算。

输出固定 t0 坐标系下的绝对 XY、yaw，不累加独立 yaw/XY 预测。监督为 XY Smooth-L1 + 周期 yaw 损失 `1-cos(Δyaw)`（权重可显式设置）；初始化为历史末段速度/角速度外推 + 零残差。硬几何不反传，也不阻断 ego 的直接监督；原 WM/CCR 无梯度。

内部头与外部无同等指令条件的 planner 比较只能称带导航条件的新设置，不冒充严格同条件公平比较。禁止 GT pose事后aligned评分；标准全网格 IoU/mIoU。

## 一次小规模实验

默认 TRAIN1024 scene-balanced、头训练20轮、batch64，整个头训练周期余弦下降。先一次性提取冻结历史特征，之后训练不重复几何准备。固定dev64同一人口一趟评估六路：OCC/STC × GT / 外部planner / 内部头，并报告各未来时刻轨迹误差。仅用最后轮头，不按dev选择best，不自动重试或扩大训练。STC属于该GT-history训练头的零样本输入迁移，未宣称做过Camera专用训练。

服务器代码更新后：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_surface_ego_head.sh
```

原始输入、缓存和冻结 mean 路径沿用已有环境变量；默认查找已完成 mean comparison。可指定 `STC_CHECKPOINT` 为现有冻结 mean（不可是resume/Local/E14 checkpoint）。`EGO_TRAIN_WINDOWS` / `EGO_EPOCHS` / `EGO_BATCH_SIZE` 可在**新实验**开始前显式设置；既有断点禁止静默改人口/轮数/batch/LR。

结束后发回训练目录和同名 `_eval_dev64` 目录的 `summary.txt`。若仅需头训练：`EGO_SKIP_EVAL=1`。更大训练、正式主表更新均须另行决定。

## 中断与恢复

Ctrl+C / SIGTERM 请求在已完成更新后停下并原子保存 `head_last.pt`。定期保存默认32个更新；SIGKILL/掉电只能恢复最后已写断点，不承诺保留未保存更新。特征提取被打断时复用已完成的新bank shard，未完成 shard重建，绝不复用旧冻结实验目录。

```bash
# 指向刚才打印的原训练目录；保持所有原参数和环境不变。
export EGO_HEAD_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/ego_head_screen_实际目录
EGO_HEAD_RESUME=1 bash tools/real_motion/run_p0_f9_surface_ego_head.sh
```

评估中断时不要重跑训练，直接恢复同一评估目录：

```bash
python -u tools/real_motion/eval_p0_f9_surface_ego_head.py \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --stc-root /root/nas/occ/swfm/data/stc_camera/compact \
  --plan-cache /root/nas/occ/new_code/cache/come_main_table/camera_pred \
  --head-checkpoint "$EGO_HEAD_OUT/head_last.pt" \
  --out-dir "${EGO_HEAD_OUT}_eval_dev64" \
  --population dev64 --population-manifest data/p0_f9_v21_dev64_manifest.json \
  --config configs/real_motion_occfm.yaml --resume
```

保持启动脚本中的PYTHONPATH、线程设置、SWFM环境一致。恢复严格核验权重SHA、源代码几何指纹、数据身份/order、历史NPZ size/mtime、nuScenes metadata SHA、特征内容、训练配置与随机/优化器状态。评估整窗六路预测和计数一起提交；不会重复计入半窗。没有源权重写操作、自动质量选头或未来 GT fallback。

## 本地验收

本地全量非集成测试：**1789 passed / 75 skipped / 4 deselected**。CUDA 专项在本地 CPU 环境跳过，不称服务器 CUDA 验收通过。

新增专项覆盖：四历史与空token、指令fail-closed、绝对SE(2)/t0 tilt、GT派生cmd末端padding、未来GT扰动下历史特征逐项一致、原输出六帧/概率不变、预测pose贯穿所有未来几何、原模型权重不变、Adam/LR/RNG逐字节续训、真实小网格bank/训练CLI和整窗续评。CPU通过不等价于L40S精度/吞吐验证；服务器只按上面小规模入口验证一次后再决策。
