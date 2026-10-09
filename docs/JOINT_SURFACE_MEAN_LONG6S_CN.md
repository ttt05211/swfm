# 冻结 Surface CCR 平均模型的 1–6 秒预测

## 固定模型与口径

只加载已完成 DEV512 对照中冻结的 5/6/8/12/14 轮整网等权平均文件。
从来源 bundle 验证来源 SHA、训练契约、TRAIN positive_weight 与平均配方；不重新平均、不扫描阈值。
平均是一个网络，不是运行时 ensemble，也不是可续训 checkpoint。
旧 Local epoch19 的长预测入口完全保留，新结果不能混为旧网络结果。

每段严格四历史→六未来（0.5–3s）；第二段只读取第一段最后四张完整预测（1.5/2/2.5/3s）。
重新构建 canonical evidence、Surface Atlas、owner、live projection phase 和动态 source queries；不复用 learned activation。
固定 weighted ADD 原始 sigmoid ≥0.5、REMOVE-off，静态与动态都采用当前 Surface 网络。

主结果预先固定为 `reconciled`：以可见 owner/class 唯一匹配接续第一段预测轨迹，并保留重提取的当前形状/物体顺序。
分裂/合并、缺失或不可靠匹配退回 redetect，绝不复活消失 source。
默认同窗口报告 `redetect` 对照，共享第一段；不按 GT 指标自动选择路线。
没有 GT box/source ID 辅助接续，没有未来 GT occupancy/mask/annotation 作为模型输入。
GT ego pose 到6s是显式轨迹条件，不把这种设置称为未知自车轨迹预测。

未来 GT 和 Moving 注释 support 在全部路线的两段预测完成后才读取，仅用于统计。
Moving support始终以原始 t0 为参照，不在3s重新定义 population。
报告1/2/3/4/5/6s，分别汇总1–3s和4–6s；保持 nominal 2Hz keyframe 定义并保存实际时间戳审计。
标准 mIoU 保留有 union 的零 IoU 类；GenieDrive兼容分数另报，不能替换标准分数。

## 服务器运行

在 OccFM 环境及项目根目录：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_joint_surface_long6s.sh dev64
```

dev64 为原 dev512 中完整6秒窗口的确定性场景均衡64个窗口，不偷偷补短序列。
dev512 是冻结 dev512 的长时可用子集，未必有512个；`all` 是 VAL4369 cache 与完整6秒窗口的交集。
入口自动发现已完成对照中的同一平均权重；发现不同平均文件时停止，不按分数挑选。
如需固定来源，可设置 `SURFACE_MEAN_SOURCE=/实际/已完成/comparison目录`。
默认训练锚点为已确认的 `full20_nohup_resume_20261008_190410_787`，可显式 `SURFACE_COMPARE_RUN` 修改；契约仍强制校验。

按 GenieDrive 固定公开 metadata 起点集合评估：

```bash
bash tools/real_motion/run_p0_f9_joint_surface_geniedrive_long6s.sh
```

复用现有安全下载器的 pinned revision/size/SHA 校验；只有身份/时间/序列用于 population。
下载器默认先访问 Hugging Face，网络失败时尝试第三方 HF-Mirror；不发送 HF token，
固定 revision、103781328 bytes 与 SHA256 始终不变，任何不一致都停止、不反序列化。
可只在本次命令指定镜像（不修改 shell 全局代理）：

```bash
GENIEDRIVE_DOWNLOAD_ENDPOINT=https://hf-mirror.com SURFACE_LONG_REDETECT=0 \
bash tools/real_motion/run_p0_f9_joint_surface_geniedrive_long6s.sh
```

`--endpoint` 优先于 `GENIEDRIVE_DOWNLOAD_ENDPOINT`，其次 `HF_ENDPOINT`；标准 `HTTPS_PROXY` 由 urllib 使用。
镜像也不可达时不会启动评估，不能用 ghfast 的 Git URL 代理套在 HF 下载 URL 上。
完全离线服务器可在联网机器下载下列固定文件后上传，已有正确文件会离线校验并复用：

```
https://huggingface.co/ANIYA673/GenieDrive/resolve/17e37acfff5b10517393a669ecf471f75f34d43f/world-nuscenes_infos_val.pkl
默认服务器位置：/root/nas/occ/swfm/data/geniedrive/world-nuscenes_infos_val.pkl
SHA256：0426072260a908260625c6dd91b9f06919726f5265c10849b87156d282547ded
```

也可将 `GENIEDRIVE_INFO` 设为上传的实际文件位置；不能用当前 V18 info 或不同 revision 替代。
下载失败发生在新评估输出目录创建前，修复连接后重跑即可，不使用 `--resume`。
4历史+20未来 metadata 用于选起点，网络只预测12未来帧，不读取多出的八帧 labels/ego poses。
固定公开代码 population 为2569窗口/150场景，包括原六历史 cache 中缺少的300个早期起点。
这些起点只用四张真实历史重建，不能因缓存缺失删掉。仅称公开代码对齐，不能声称已验证论文 Table2 population。

## 加速与缓存

第一段优先读取现有只读 VAL geometry cache；相同因果输入 hash、namespace、manifest 都强制验证。
已知缓存记录缺失/损坏直接停止；没有缓存的 GenieDrive 早期起点明确实时重建。
第二段没有 source adapter、磁盘查找或真实未来 cache lookup；全部 geometry 来自预测历史，接续 velocity 不再被重建覆盖。
不计算旧 Local future memory、footprint、frontier候选。使用编译 CPU/融合 projection、有界4窗口预取。
复用逐字节验证的静态 CUDA graphs：只捕获完整8192块，变化尾块 eager；每次补入当前 source 值，最多四张图。
第一段核验四历史输入、六张 Transport、六张完整 Surface 输出；第二段再次核验概率与六张输出。
这些质量评估耗时包含 GT/Moving/整数统计，不是正式 Dense Forecast FPS。

可设 `SURFACE_LONG_VAL_CACHE=off` 明确关闭只读缓存，不创建/重建大缓存。
可设 `SURFACE_LONG_GRAPHS=0` 禁用 graphs；`SURFACE_LONG_REDETECT=0` 只跑固定接续主结果，减少对照耗时。
执行选项进入 resume contract，不能在同一结果中静默切换。

## 中断恢复与结果

SIGINT/SIGTERM 在完整所有路线窗口边界退出并原子保存整数状态；每8个窗口周期保存。
kill -9 只恢复最后一次周期保存。旧 Local、其他模型或不同人口的评估状态不能续到这里。

```bash
# 原终端打印的输出目录原样填入，保持同一人口与执行选项。
bash tools/real_motion/run_p0_f9_joint_surface_long6s.sh dev64 /实际/原输出目录 --resume
# GenieDrive:
bash tools/real_motion/run_p0_f9_joint_surface_geniedrive_long6s.sh /实际/原输出目录 --resume
```

写入新输出目录：`summary.txt`、`evaluation.json`、`evaluation_state.json`、`contract.json`、
`timestamp_audit.json`、`checkpoint_snapshot.pt`、`bundle.json`、`progress.jsonl`。
仅在完整所有窗口后发布 evaluation.json；训练 checkpoint、optimizer/RNG、缓存、旧结果都不修改。
本地单元/合成测试不等于真实 nuScenes 6s 精度通过；服务器实际结果另行验收。
