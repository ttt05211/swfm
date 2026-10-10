# STC 四设置无损历史共享（2026-10-10）

用户服务器最近64窗（421–484）为1.7103秒/四设置窗口。四路历史与搬运准备合计
0.8247秒、证据与投影0.7111秒，约90%的总时间；网络head合计0.0406秒。
GPU显存1.2GiB不是瓶颈证据，更大的batch或占满显存不能直接消除这些CPU工作。

## 改动边界

新独立入口 `run_p0_f9_joint_surface_stc_shared.sh`，原STC入口及全部科学实现文件不改。
复用已经存在的原生CPU warp/局部平面拟合、历史frame几何LRU与并行Surface查询。
同一窗口/同一实际历史内容的GT/Pred两路共享以下因果、future-independent量：

- 物体提取、历史关联与注册、tube/KTA/运动输入；
- 同一个冻结整网的历史-only motion输出（非跨模型、非跨窗口持久缓存）；
- canonical evidence与其Surface描述。

缓存key包含scene/t0/history token，以及每帧semantics、observed mask、pose的完整内容摘要。
只有一个临时history bundle，历史、可见性或Camera/Occ内容改变立即失效。
每帧固定几何LRU另有512MiB上限；原始帧LRU512MiB；不建立新磁盘数据缓存。
不共享Strong、未来ego transforms、renderer owner/fallback、未来Surface相位、概率或六帧输出。
因此Pred使用预测轨迹的全部未来几何，Camera仍只有STC完整语义/all-valid grid，不借GT可见性。
固定平均5/6/8/12/14、ADD0.5、REMOVEoff、四历史六未来、人口与指标均不变。

## 一次命令：无损对照、测速、安全接续

先安全停止原评估（Ctrl-C或SIGTERM，不能kill -9），等待进程退出。
可以在新终端执行以下命令，原输出保持只读。新代码不自行发送进程信号。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export STC_ROOT=/root/nas/occ/swfm/data/stc_camera/compact
export STC_PLAN_CACHE=/root/nas/occ/new_code/cache/come_main_table/camera_pred
export STC_POPULATION=all
export STC_CONTINUE_FROM=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/stc_four_all_20261010_102802_1047
export STC_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/stc_shared_all_$(date +%Y%m%d_%H%M%S)
bash tools/real_motion/run_p0_f9_joint_surface_stc_shared.sh
```

首次执行默认取未完成前缀后的8个相同窗口：每个窗口全部四设置、全部六帧，
原版/新版运动输入、Transport、概率和dense逐字节对照；还保留原版live exactness。
随后相同窗口完整前向做两遍AB/BA计时；不读未来标签、不更新科学计数。
只有测得大于1%的速度提升才自动接着跑剩余窗口；无损或提速检查失败原状态不变。
该速度是四设置质量评估耗时，不是raw-camera端到端FPS。不在此预告L40S速度倍数。

新输出记录speed_comparison.json。contract记新旧全部源码指纹、数据与模型指纹和旧状态SHA。
桥接严格验证旧三时距18×18整数计数与cursor，旧计数原样复制，已完成窗口不会重跑。
旧/新两个输出的kernel lease持有到整个新评估结束，拒绝旧程序仍运行时接续；
每次恢复也确认原目录state/contract未变，不允许两边分别继续同一个前缀。
每8个完整四设置窗口原子保存；中途停止和新版普通恢复仍严格检查全部契约。

```bash
# 继续新版，不要重新迁移；STC_OUT指向上次打印的新版目录。
unset STC_CONTINUE_FROM
export STC_RESUME=1
bash tools/real_motion/run_p0_f9_joint_surface_stc_shared.sh
```

普通resume不重复小窗口测速，不改权重/阈值、不训练。
wrapper会从旧/新版contract恢复原SWFM执行环境，避免新终端遗失或残留这些开关；
只读JSON并校验变量名/字符串，不用shell eval。其余数据/模型/CPU/graphs契约仍严格核对。
如果初次测速未提速，使用原入口+原STC_OUT+STC_RESUME=1继续即可，原结果没有被覆盖。
保持其余原评估配置/环境一致；若原来关闭graphs或改变CPU线程数，迁移时也要显式保持一致。

## 本地验证范围

相关回归：69 passed / 7 CUDA skipped；Bash语法、CLI help、diff检查通过，非全仓CI。
实际小模型CPU/原生后端四路运动输入、Transport、probability、完整六帧byte gate及
整数指标相等；不同visibility、不同未来GT/Pred轨迹、变更历史内容失效均覆盖。
断点整数前缀迁移/恢复、科学契约与旧代码变更拒绝、源目录只读、租约互斥、
源state篡改检测、CLI恢复严格性均覆盖。CUDA用例在无CUDA本地环境明确跳过，
服务器入口会对真实CUDA与实际STC窗口重新检验，不能把CPU fixture当作服务器性能/精度结果。
