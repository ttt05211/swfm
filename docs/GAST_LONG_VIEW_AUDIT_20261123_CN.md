# 长时GT“短一截”的显示检查

本次用户截图匹配`turn07 / scene-1060`。原始预测包及全部语义数组只读，没有补全、掩码、阈值、模型或指标改动。

## 公开代码核对

- SparseWorld-TC：[vis_id.py](https://github.com/MrPicklesGG/SparseWorld-TC/blob/main/vis_id.py)的`vis_iter`与主循环直接把每个future的解码坐标送入`pyvista_vis_occ`；绘图时没有额外乘`inverse(pose_1s) @ pose_h`。`fut2cur`在该绘图循环中用于确定帧数，并非渲染变换。
- [loading.py](https://github.com/MrPicklesGG/SparseWorld-TC/blob/main/loaders/pipelines/loading.py)的`LoadOccFromFile`读取各future自身的`labels.npz`并保留原生语义网格，位姿矩阵另行交给网络。模型head以对应future语义作监督。
- 仓库目前只有通用`vis_id.py`和默认`[0,2,4,6]`的配置，没有找到与论文Fig.5十秒图一一对应的独立导出入口。通用入口会按future帧列表循环，不能把默认三秒配置称为已复现十秒实验。
- COME：[visualize_demo.py](https://github.com/synsin0/COME/blob/main/tools/visualize_demo.py)逐帧将输入/预测张量交给绘图函数，原点`[-40,-40,-1]`、体素0.4m；主绘图路径没有额外的世界固定参考变换。另有`export_pcd`分支使用绝对位姿，不能与主逐帧渲染混淆。

因此本次以`--view ego`提供逐帧future ego坐标的展示，同时保留旧固定参考图作运动与覆盖诊断。相机数字相对ego固定，等价于世界中的相机跟随自车平移、转向。使用COME相机和配色，不冒充逐像素复现SparseWorld-TC的相机或Fig.5。

## 截图场景的实测几何

每帧GT仍为200×200×16、0.4m，即自车周围80×80m局部网格。旧长图把这些不同future局部网格放到固定1s ego参考中。

| horizon | 相对1s平移XY/m | 相对yaw/度 | 与1s网格覆盖重合面积 |
|---|---|---:|---:|
| 1s | 0.00, 0.00 | 0.00 | 100.0% |
| 2s | 6.38, 1.10 | 19.51 | 85.8% |
| 3s | 12.59, 4.12 | 31.31 | 78.1% |
| 4s | 19.63, 8.48 | 32.51 | 69.2% |
| 5s | 26.64, 12.94 | 33.10 | 59.5% |
| 6s | 33.31, 17.27 | 33.23 | 49.6% |

六帧GT显示体素中心的近裁剪、远裁剪、画面外计数均为0；此检查统计点中心，不宣称所有边缘体素面像素完全无裁剪。6s原生道路X范围约`[-39.8,31.8]m`，并非原始语义数组在导出时被裁掉近半。移动/旋转的有限GT窗口加固定相机，会把较近处原来覆盖的区域留空，并改变透视下的主体大小。旧图把这种显示覆盖变化呈现成“场景缩短”，不适合作为与逐帧ego展示混用的论文图。

逐帧ego重画没有增加任何GT或预测内容，也不保证所有真实预测缺失消失。它恢复相同局部空间范围下的观察口径。若需要固定世界视角且固定空间ROI，应单独定义所有时刻共同有标签的范围，或声明世界地图构建协议，不能用GT静态背景补进模型预测来掩盖缺失。

## 重画命令

```bash
python -m tools.paper_assets.gast_readable_gallery render \
  --assets /path/comparison_assets.zip --out /NEW/long_ego --view ego \
  --cases turn01 turn02 turn04 turn05 turn06 turn07 turn08 turn09 turn10 turn11 turn12 turn13 turn14 turn15 turn16 turn17 turn18 turn19 turn20 turn21 turn22 turn23 turn24
```

`ego`对长时六帧不附加显示变换；原`native`长时沿用固定1s参考以保留旧结果，`motion`也为固定参考。所有方法与时距仍共享透明边缘裁剪、比例和同一COME相机。

## 两场景独立图片

```bash
python -m tools.paper_assets.gast_scene_panels \
  --assets /path/comparison_assets.zip --out /NEW/selected_scenes --scenes 1064 797
```

ZIP只含`scene-1064`和`scene-0797`两个目录，各`gt_0..3.png`、`carry_0..3.png`、`gast_0..3.png`，共24张。0/1/2/3对应0/1/2/3秒，原ego坐标，不附加future→固定参考变换。每个场景共用一次完整内容并集裁边，保持10个独立渲染（3行共用t0）之间相同比例。保留原生渲染像素，白底RGB、300dpi标记，没有文字或边框。接入记录在解压目录根下，用户要求的ZIP中不放额外文件。

长时仅导出GT与CarryWorld，例如scene-0345/scene-0916：

```bash
python -m tools.paper_assets.gast_scene_panels \
  --assets /path/comparison_assets.zip --out /NEW/long_selected_scenes \
  --scenes 0345 0916 --horizon long --models gt carry
```

每个场景`gt_1..6.png`与`carry_1..6.png`，数字直接对应1–6秒，共12张；不导出GAST，也不放入t0。长时逐帧ego坐标，不做固定世界参考显示变换。两模型六时距共用外边缘裁剪，场景内每张PNG尺寸和比例一致。默认不传`--horizon`/`--models`时保留原短时三模型导出行为。
