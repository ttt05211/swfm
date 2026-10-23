# STC小处理：冻结完整FPN＋U-Net，仅历史Memory年龄衰减

## 为什么只做这一项

旧clean1000完整VAL4369曾有处理收益：原18.819637 mIoU/28.352669 IoU，half-pair/LS KTA＋年龄记忆为19.098244/28.478163，再叠历史支持与后验校准为19.163111/28.731686；额外192个norm适配后19.190065/29.016647。后者是明确的无标签目标域适配，不是原始网络零适配分数。

现在FPN、纯历史Memory和U-Net都已改变，旧norm参数/后验bias绑定旧checkpoint，**不复制旧patch**。此前多帧KTA有Moving代价，不重启。去噪、形态学、TTA、额外小头等旧负结果不重跑。

本次只迁移最小的历史Memory年龄衰减，旧独立年龄项在同64窗约+.06～.07 mIoU、+.12～.24 IoU，不能承诺它在新模型仍有效，更不能把旧delta加到新分数上。

## 实际改动

只有`processing.py`中的固定权重处理：

`weight[t] = 2 ** ((history_time[t] - history_time[t0]) / 0.5s)`

标准四历史为 `[1/8,1/4,1/2,1]`。按真实历史秒数计算，半衰期固定0.5s，不扫参数。对static_valid和dynamic coverage乘权重，再调用**原有**aligned_memory融合，使分子和分母同时加权；不是把特征整体乘小导致幅度失真。

原Encoder/WM/source身份/两帧KTA/geometry/Strong/Transport Prior全部共享，候选只重算Memory和Refiner。模型所有权重冻结，不训练、不拟合校准、不读未来GT作修正，不用GT可见性mask。不改任何正式默认/原训练/原全量评估。只在本STC实验入口启用，GT分支不接这个开关。

## 一次固定64窗配对

加载完整TRAIN20430×2最后`epoch_0002.pt`，STC原同64窗、四历史六未来、GT future ego作为显式条件。一次Encoder/WM/geometry，输出baseline Transport、baseline Refiner和age Refiner后才读未来GT/Moving评分。三路用同整数confusion计算1/2/3s及均值，报告正负delta，重点同时看Moving。没有新结构/多个系数/自动best选择/自动扩大全集。

这64窗已被反复研究，只是开发检查；如果负增益，如实保留，不部署。若正增益，仍需用户决定是否在更多窗口验证。不是FPS验证，不继承40.6/44.7输出帧s。

## 服务器命令

先同步本次提交，再：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export STC_AGE_GPU=0
export STC_AGE_RESUME=0
export STC_AGE_DIR="$PWD/outputs/p0_f9_shared_history/fpn_stc_age64_$(date +%Y%m%d_%H%M%S)"
bash tools/fpn_unet_stc_age/run.sh
cat "$STC_AGE_DIR/summary.txt"
```

已有`FPN_UNET_FULL_DIR`就读取该目录；否则自动查找唯一已完成的`fpn_unet_full_2ep_*`。多个候选会停止并列出，不按分数或时间选。需要时显式设回正确原目录：

```bash
export FPN_UNET_FULL_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_full_2ep_实际时间
```

前台中断后，保留`STC_AGE_DIR`、`export STC_AGE_RESUME=1`再执行原入口，从完整窗口整数计数恢复，不重训。代码只写独立输出snapshot和指标，不修改原checkpoint/旧缓存。当前本地无最新科学权重，局部小网格功能检查不代表真实STC提升。

本地安全yoloe新4项＋旧冻结评估8项，共12 passed（11.08s），包括真实CPU/CUDA新Memory/Refiner前向、未来GT poison不读、原参数/Prior/targets/历史数组不变、真实时间权重、整数恢复与负增益照报；Bash-n通过。没有真实新模型质量或FPS增益声明。
