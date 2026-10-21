# FPN Encoder + 常规 3D U-Net：单组快速验证

用户明确要求两个结构一起改、只训练一组，不跑拆分消融。旧模型与入口不修改。

## 结构

- 历史语义 embedding4维 + known1维；Z折叠入通道，四帧共享二维 Encoder。
- 轻量残差 Backbone：100/50/25，64/96/128通道，各stage两个普通残差块。
- 真正FPN：1×1 lateral + nearest top-down相加 + 3×3 smooth，三个输出均64通道；不再平均池化伪造金字塔。
- stride2使用偶数kernel，保持三尺度物理边界/格点中心约定。共享128维投影供原 Deformable Attention；P2给中心采样和Memory lift，P4给GAP。
- 原2+2 WM、relative GT ego Query、cross/self/FFN、pinned-track、两帧KTA、无ICP、纯历史Memory保持。
- Refiner：标准双Conv3D/GN/ReLU U-Net，8/16/32，XY-only AvgPool，Z=16保持。P18+时间进入fine；pool(E0)仍8通道，concat Memory16=24通道后Conv24→16→16。
- 解码采用trilinear上采样+skip concat+双卷积，最终1×1×1输出18类修正；有限log-prior残差连接。无AdaGN/phase/point MLP/ADD REMOVE动作门限。
- 六时刻共享权重，U-Net每组两个时刻；训练activation checkpoint重算同一网络，仍监督所有六时刻，不是抽两个future训练。推理不重算反向图。

## 固定预算与缓存

RANDOM TRAIN20%=4086×1，seed20261011，global B4/source128，原5e-4/3e-4两组AdamW、整一轮cosine floor.01、clip5、H6 full-grid CE+Lovasz ONLY。GT future ego条件，无ego预测头/辅助损失。最后一次same remaining64 OCC+GT/STC+GT、Transport/Refiner四行；不附加KTA反事实或额外模型训练。

原始历史/纯几何旧cache require只读复用；独立输出目录内可写纯历史identity cache。新Encoder特征/Memory现场可微计算，不能复用旧网络的固定特征或此前局部实验P/M缓存。base checkpoint只提供架构和旧几何provider，不导入其主体权重。

## 服务器运行（代码同步后）

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export SCENE_TRAIN_GPUS=0,1  # 只有一张空闲卡时改为0
export FPN_UNET_DIR="$PWD/outputs/p0_f9_shared_history/fpn_unet_20pct_1ep_$(date +%Y%m%d_%H%M%S)"
export FPN_UNET_RESUME=0
bash tools/fpn_unet/run.sh
cat "$FPN_UNET_DIR/summary.txt"
```

默认数据路径沿用现有实验。`REFERENCE_DIR`默认既有`stc_empirical_refiner500_20261014_194954`，仅从其`clean/results.json`取同64窗ID，不加载权重/分数、不另训baseline。

后台运行可把上面的bash行替换成：

```bash
mkdir -p "$(dirname "$FPN_UNET_DIR")"
nohup bash tools/fpn_unet/run.sh >"${FPN_UNET_DIR}.log" 2>&1 &
tail -f "${FPN_UNET_DIR}.log"
```

按一次Ctrl-C会在完整update/eval window后保存。恢复使用**同目录、相同卡数**：

```bash
export FPN_UNET_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_20pct_1ep_实际时间
export FPN_UNET_RESUME=1
bash tools/fpn_unet/run.sh
```

完整3D卷积的显存/延迟与旧Refiner不同，未实测不得沿用旧FPS。此组合只回答整体候选是否值得继续，不能分别归因两个组件；本地合成小网格测试不算正式质量结果。

## 本地检查结果

安全yoloe环境：新增8项＋原ego-conditioned路径10项，共18 passed（19.81s），包含实际CUDA BF16单步、完整预测/CE+Lovasz反向、小型入口训练停续、最终仅一次eval、checkpoint loader及整数指标恢复。第一次旧确定性测试缺少CUBLAS环境变量而失败；补齐与脚本相同的`:4096:8`后全过，不修改容差/旧代码。Bash `-n` 与 CLI `--help`通过。

默认四历史、d_model128配置总参数 **2,520,442**，其中FPN Encoder **1,357,068**，U-Net及Memory lift/importance/empty **116,299**。参数增大主要来自普通残差Backbone，不是Refiner。没有借用旧1.2M或旧FPS数字。

实现时本地没有完整TRAIN4086、旧geometry caches或服务器执行连接，未启动正式一轮训练，暂无新质量指标；上述检查仅是小网格功能验证。启动服务器前需同步本次提交的新package。
