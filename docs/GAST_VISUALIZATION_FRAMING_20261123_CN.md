# GAST对比图：运动显示与留白修正

原始42窗预测和旧图册保留。这里只改显示与拼图，不改模型、语义数组、预测运动、评估分数或数据窗口。

## 短时车辆看起来慢的原因

旧短时四列t0/1/2/3s分别采用该时刻的ego坐标；同向同速的车相对自车位置可能几乎不变。原候选框元数据表明，short02的可见车辆t0到3s世界位移约30.15m。已有GT车辆最大连通域质心诊断：short02在1到3s的原ego显示位移约0.37m，转到固定1s参考后约19.84m；short03约0.86m与20.30m。这是体素质心的显示诊断，不是新的标注框速度测量或正式运动指标。

新增motion视图把三种方法的所有未来帧统一转换到第一张1s ego参考：`inverse(pose_1s) @ pose_h`。原生占据数组只读，变换只作用于显示几何。静态世界物体不应随自车运动平移，同向车辆真实的世界运动更容易观察。

comparison_assets.zip未保存t0的ego2global，不能凭空对齐输入帧。因此缺少额外位姿时，motion短时图只显示1/2/3s；原生四列图仍可生成。不能给t0套用1s位姿，或通过语义ICP猜一个位姿。

## 留白处理

COME参考相机原先固定2560×1440完整画布并关闭边缘裁剪。新增参数只移除所有方法、所有时刻的透明外边缘并集之外的共同留白，再统一缩放；面板高宽比跟随共同裁剪范围。保留原COME相机位置、方向、视角、裁剪面、配色和原生体素。

不能逐帧独立缩放或只按GT边界裁剪；GAST/CarryWorld多预测出来的区域也必须纳入公共边界，不能借裁剪藏掉误差。旧渲染默认参数不变，新图应使用新的输出目录。

## 本地重画

使用已安装可视化依赖的独立Python解释器（Windows遵循windows-python-env-guard，不使用Anaconda base）：

```bash
python -m tools.paper_assets.gast_readable_gallery render \
  --assets /path/comparison_assets.zip --out /NEW/output/native --view native

python -m tools.paper_assets.gast_readable_gallery render \
  --assets /path/comparison_assets.zip --out /NEW/output/motion --view motion \
  --cases short01 short02 short03
```

native短时仍用各帧ego坐标，长时仍用固定1s参考；两者都统一去留白。motion未来帧统一参考，不能把它与原生ego显示混淆。

## CPU补输入位姿，恢复固定坐标的完整四列

同步新脚本后，在服务器执行：

```bash
cd /root/nas/occ/swfm
bash tools/gast_baseline/step.sh display-poses
```

只读取既有`outputs/gast_baseline_gt/prepared.json`，输出同目录下`display_input_poses.json`。仅含42窗输入位姿、t0 token和源prepared文件SHA256，通常几十KB，无需GPU、权重、数据重新导出或模型推理。输入JSON必须与预测包中的prepared SHA和全部case/token完全吻合。

下载该小JSON后，固定参考图可包含t0：

```bash
python -m tools.paper_assets.gast_readable_gallery render \
  --assets /path/comparison_assets.zip --input-poses /path/display_input_poses.json \
  --out /NEW/output/motion_with_input --view motion
```

检查包括：静止世界点在固定参考中不随ego移动、同速车辆显现位移、输入位姿精确关联、公共裁剪包含全部方法边界和共用t0像素一致，以及原GAST/VTK回归。没有重新计算正式质量分数。
