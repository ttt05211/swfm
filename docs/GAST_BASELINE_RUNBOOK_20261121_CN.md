# GAST对比图：按顺序执行的完整流程

本入口只做官方GT ego权重的冻结推理，不训练。输入为现有Occ3D GT四帧，输出与CarryWorld相同窗口的1–3s和1–6s对比。官方模型源码不改，配置、路径、数据格式、两段滚动和图册接入均由独立适配器处理。

**第1–7步可在没有GPU的CPU实例上完成。仅第8、9步需要一张GPU。** 所有脚本直接调用独立`gast`环境，不用反复手动激活Conda，不改原OccFM环境。安装使用预编译Torch/CUDA算子wheel，不调用nvcc。

默认路径：

| 内容 | 路径 |
|---|---|
| CarryWorld仓库 | `/root/nas/occ/swfm` |
| 官方GAST源码 | `/root/nas/occ/GAST` |
| 新输出及下载目录 | `/root/nas/occ/swfm/outputs/gast_baseline_gt` |
| 现有数据 | `/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes` |
| 官方GT权重 | 新输出下`downloads/gast_gt.pth` |

## CPU阶段

### 1. 同步适配脚本，并下载官方GAST代码

```bash
cd /root/nas/occ/swfm
git pull --ff-only https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
bash tools/gast_baseline/step.sh code
```

锁定官方提交 `c4687255132e580468e1007168c57b00ce36b76a`。已有GAST目录若有源码修改会明确停止，不覆盖它。**不要运行GAST原来的run.sh，它默认启动训练。** 本入口不需要手改官方YAML或Python。

### 2. 安装独立环境

```bash
bash tools/gast_baseline/step.sh env
```

自动建立Python3.10的`gast`，安装Torch2.0.1/cu118、torchvision0.15.2、MMCV-full1.7.2、torch-scatter2.1.2，以及小型数据适配与现有PyVista/VTK渲染依赖。numpy固定1.23.5，setuptools固定69.5.1以兼容旧MMCV。不会查询GPU是否可用；CPU实例也能装带CUDA的预编译包。使用`--no-cache-dir`避免保留第二份大wheel。安装完成打印`GAST_ENV_CPU_OK`。

### 3. 下载GT ego权重

```bash
bash tools/gast_baseline/step.sh weights
```

仅请求官方README的GT ego权重，Drive ID `1cV4c4zyoiitZcJEdvFhr6aiCWwT3mI4-`，不下载Pred权重或数据集。支持相同命令继续`.part`下载；完成后CPU检查state_dict和4历史/6未来参数形状，记录大小和SHA256。原作者未公布可信hash，因此记录的是本次下载身份与本地hash，不冒充独立验真。

### 4. 检查现有info，必要时仅补官方VAL info

```bash
bash tools/gast_baseline/step.sh infos
```

自动寻找之前定性图的完整候选ZIP，优先使用窗口数最多的匹配包（当前已知43窗）。验证其最终E权重身份、原生网格和短时/长时token。现有temporal_v3_scene.pkl中的真实ego速度/角速度与导航字段齐全时直接复用；字段不全才列出官方info目录，并只下载唯一VAL pickle，不下载TRAIN或Occ3D。目录有多份不明确的val时停止，请用后面的手动接入命令。

如果服务器没有自动找到之前的候选包，只需设置一次路径，再执行本步；不必重新导出数据：

```bash
export GAST_ASSETS=/完整路径/paper_qualitative_expanded_20261115_231750.zip
bash tools/gast_baseline/step.sh infos
```

### 5. 准备同窗口的小型数据清单

```bash
bash tools/gast_baseline/step.sh data
```

直接读取原`gts/scene/token/labels.npz`路径和nuScenes元数据，用ijson流式扫描sample_data/ego_pose，只保留选中窗口所需的LIDAR_TOP记录。支持现有按scene分组或官方flat info；不需要额外安装nuScenes SDK，也不需要图片或点云文件。

验证同scene、连续2Hz keyframe、ego2global、wxyz四元数、真实ego字段、与原CarryWorld包完全相同的未来位姿。只产生小型`prepared.json`，不复制标签、不生成dense cache，不改任何原数据。缺失字段不以零代替。

### 6. CPU严格加载权重

```bash
bash tools/gast_baseline/step.sh cpucheck
```

在CPU实例化原版GAST，去掉DDP前缀后`strict=True`完整加载权重，核对官方源码指纹，生成新目录下的`gast_infer.yaml`。不运行CUDA或完整场景推理。成功打印`GAST_CPU_CHECK_OK`。

### 7. 安装并检查纯CPU渲染支持

当前root终端执行：

```bash
bash tools/gast_baseline/step.sh render-libs
bash tools/gast_baseline/step.sh render-check
```

安装OSMesa/EGL系统运行库，默认固定`vtkOSOpenGLRenderWindow`软件渲染，支持没有GPU、没有DISPLAY的CPU节点。小方块检查图在新目录`rendered/render_check.png`。无需Mayavi或Xvfb；沿用已有COME相机/色表/原生体素渲染代码，不会自动换成另一种BEV示意图。

CPU阶段到这里就可以结束。所有数据和配置保存在固定输出目录，重新打开终端后仍可继续。

## GPU阶段

### 8. 分配一张GPU，检查算子与两个固定窗口

```bash
cd /root/nas/occ/swfm
bash tools/gast_baseline/step.sh check
```

默认物理GPU0；如需其他卡用`GAST_GPU=1 bash tools/gast_baseline/step.sh check`。检查实际CUDA scatter和deformable attention，再用官方权重跑固定第一个short和第一个turn窗口，分别核验3s和6s。成功结果直接保存，后续全候选导出复用，不重复推理。成功打印`GAST_CHECK_OK`。

### 9. 后台导出全部候选

```bash
nohup bash tools/gast_baseline/step.sh infer > outputs/gast_baseline_gt/infer.log 2>&1 &
tail -n 20 -f outputs/gast_baseline_gt/infer.log
```

单GPU/单模型进程，只跑选中的定性窗口，不评整个VAL。进度打印`GAST_EXPORT n/总数`。每窗保存压缩uint8预测，支持断点继续；终端关闭后nohup继续运行。若被停止，直接重跑同一条nohup命令，已完成窗口与已完成ZIP的字节保持不变。

1–3s先丢弃官方输出的4帧历史重建，再选未来1/2/3s。1–6s用发布的3s权重做两段滚动，第二段仅使用预测的1.5/2/2.5/3s四帧历史，条件于后续GT ego；不读取真实未来占据作为历史。所有6/12预测完成后才读取用于图示的未来GT。

完成产物是`comparison_assets.zip`，包含同窗GT/GAST/CarryWorld的原生语义数组和小清单，不含代码、权重或logits。明确标注released3s checkpoint / 2x3s rollout，不冒充官方论文的长时表复现。

## 渲染与查看（CPU即可）

### 10. 生成全部对比图

```bash
nohup bash tools/gast_baseline/step.sh render > outputs/gast_baseline_gt/render.log 2>&1 &
tail -n 20 -f outputs/gast_baseline_gt/render.log
```

短图三行GT/GAST/CarryWorld、四列t0/1/2/3s；长图三行GT/GAST/CarryWorld、六列1–6s，全部行使用同一1s参考坐标。两种方法共用同一COME相机、OccWorld/COME色表和原生体素尺度，无圈、平滑或后处理。每个窗口一张PNG，另有速览页与`rendered/index.html`，可以全部看完再选场景。

渲染按窗口恢复，不重新生成已经通过指纹核对的图片。若改变相机，应另指定新输出目录；已有预测不需重跑。也可以下载`comparison_assets.zip`回本机，用相同Python模块渲染，具体参数见`python -m tools.gast_baseline.render --help`。

### 11. 终端重开后查看进度

```bash
cd /root/nas/occ/swfm
bash tools/gast_baseline/step.sh status
tail -n 20 outputs/gast_baseline_gt/infer.log
tail -n 20 outputs/gast_baseline_gt/render.log
```

查看/下载：`outputs/gast_baseline_gt/comparison_assets.zip`和`outputs/gast_baseline_gt/rendered/`。前者可用于本地渲染，后者直接含全部图片及HTML。

## 下载受限、路径变化与恢复

如果服务器无法访问Google Drive，可在浏览器按官方链接下载，再用明确命令接入，代码无需编辑：

```bash
bash tools/gast_baseline/step.sh adopt-weights --file /下载位置/gast_gt.pth
bash tools/gast_baseline/step.sh adopt-info --file /下载位置/world-nuscenes_infos_val.pkl
```

然后继续`infos`、`data`、`cpucheck`。手动文件也记录hash和来源方式；已有不同文件不会被覆盖。

默认路径之外的配置使用环境变量：`GAST_REPO`、`GAST_WORK`、`GAST_ASSETS`、`GAST_DATAROOT`、`GAST_INFO`、`GAST_GPU`。显式`GAST_INFO`优先于自动下载的VAL文件。不改官方源码和旧实验目录。

恢复必须沿用同一`GAST_WORK`、原数据、原权重和代码。默认固定输出目录已经满足；不要每次恢复再拼一个新时间戳。若想真正开始另一套设置，才选择新目录。官方源码检查也接受Windows/Linux换行差异，但不接受模型代码内容变化。

## 验证与范围

本地测试包括：实际执行官方dataset函数并逐张量比对适配输入；真实ijson元数据反查与连续性检查；CPU阶段禁止GPU探测；raw state_dict/完整strict加载及错误拒绝；第二段预测历史隔离；1/2/3与1–6s索引；逐窗与打包断点恢复；全部角色共用坐标变换；实际COME/VTK小网格渲染和图册恢复。官方源码小副本只用于本地测试，不随交付上传，不分发官方权重。

仍须由服务器的cpucheck/check确认实际下载的权重与Linux/CUDA组合。当前没有服务器执行连接，不声称已产生GAST质量分数或已完成真实权重CUDA推理。官方eval的两帧合并区间口径也没有冒充我们单取1/2/3s的指标；本流程仅做定性导出。
