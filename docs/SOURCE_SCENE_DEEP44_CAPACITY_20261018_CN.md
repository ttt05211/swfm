# Integrated 2+2 → 4+4：最后一组联合容量候选

## 用户决定与控制

仅将当前最新 Scene–Temporal Encoder 两块增至四块、Future Query Decoder 两块增至四块。保留共享历史 Encoder、三块 AdaGN Refiner、d128、三尺度读取/采样点数/PE/TE、初始 GAP、KTA Query/skip、无 ICP、track_valid、学习历史相对 yaw 和 direct-history memory。

控制为用户刚完成的 `tools/source_scene_integrated` RANDOM TRAIN4086×1，不是更早 KTA-query 独立 Scene/Temporal 结构、no-query Stage2、全数据旧两轮或 clean1000/norm192。控制结果只读复用，不重训、不导入其训练权重。

正式宽度参数：2+2 为 **1194146**，4+4 为 **2191042**（+996896）；其中新增两 Encoder 共467232，两 Decoder 共529664。Refiner 两边同47347参数，网络宽度不变。不是等参数/等算力实验，不能将效果单独归给 Encoder 或 Decoder。

## 实现范围

新增独立 `tools/source_scene_deep44`。DeepTransport 继承原 Integrated 的完整 forward，只向原 ModuleList 分别追加两个独立随机 Block，不改数据流、不复制四份历史/六份特征、不在每层额外做预测/重投影。Decoder 仍只在最后读出 XY/yaw 和 Q，不声称增加了逐层几何纠错机制。

四层仍然逐子层屏蔽缺失历史，历史时序在每个 source 内交互，完整四帧 H 进入未来 Decoder。未来 Decoder 的四层在六个 query 内交互并读取 H；没有未来占据输入，没有新增 ego/导航或运动监督。历史 yaw 的 t0 gauge 仍严格为零。

新模型先生成与2+2完全相同的 FRESH 随机控制模型，再在独立 RNG 中构造4+4；所有原有参数逐键直接拷贝新鲜随机值，新增加的四块保留各自独立随机初始化。原共享模块/Refiner/前两层/预测头参数和 CPU post-init RNG 与控制逐元素相同。不是从训练权重出发，也不是克隆已有层共享权重；新增随机层不保证中间特征与2+2相同。初始最终 Transport argmax 的保证仍来自原零初始化运动/残差读出。

旧模型/训练/测速/几何源码完全不修改，旧 fingerprint、断点、结果和缓存不覆盖。新 protocol、variant、loader、锁和 output/identity_history/rank*；旧纯历史缓存 require 只读。拒绝把2+2或更早 checkpoint 当新任务 resume。

## 同预算流程

- RANDOM TRAIN4086×1，原 seed20261011/keys/order/global B4-source128/H6。
- 完整六时距 CE+Lovász；motion/existence/ego 均关闭。
- TRAIN/OCC/STC 全 dense-valid，GT future ego 条件、原 remaining64。
- 原 AdamW、两组LR、裁剪、完整一轮 cosine、精度不变。
- 默认双卡训练，控制 world_size 必须相同；最终仅 rank0 单GPU评估 OCC/STC，另一 rank 等待。
- 两边模型元数据除两种深度外严格核对，控制原源码 SHA、数据/输入库存、人口/预算/日程绑定；不按最高分或最新时间选控制。
- 最终比较四个标准 route，并报告本模型 learned−KTA、条件 GT 框 moving/slow 位移误差、修正量。先预测后读标签。整数评估与完整 update Adam/rank RNG/cursor 可恢复。
- 日志保留真实训练秒/窗，不推算 FPS；没有自动扩训/全集评估/部署。反复使用的 DEV64 是开发人口，不是独立测试。

## 服务器

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_TRAIN_GPUS=0,1
export DEEP44_RESUME=0
export DEEP44_DIR="$PWD/outputs/p0_f9_shared_history/scene_deep44_1ep_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_deep44/run.sh > "${DEEP44_DIR}.log" 2>&1 &
tail -F "${DEEP44_DIR}.log"
```

控制默认使用明确的 `INTEGRATED_DIR`，否则只认唯一完成的 `scene_integrated_1ep_*`。若有多个完成目录会列出并要求设置，不自动选；`DEEP44_DIR` 必须是新的输出，不能设成控制目录。

结束：`cat "$DEEP44_DIR/comparison.txt"`。中断后设回原 `DEEP44_DIR`、`DEEP44_RESUME=1`，使用同入口。不会重训控制，不会因为完成1轮自动扩到全集。

## 验证边界

专项测试包括：只有新增层参数、原公共新鲜初始化/RNG严格一致、正式宽度参数量、CPU/CUDA完整六时距梯度到第四Encoder/第四Decoder及运动/历史yaw/AdaGN、四层 missing mask、完整4token上下文、KTA Query条件、future label poison、无ICP trap、空source、新loader/恢复隔离、真实CUDA两步停续/控制只读，以及OCC/STC整数续评。

本地新旧41项测试通过（24.32秒），含正式宽度参数量与实际小网格CPU/CUDA完整梯度/停续；CLI帮助、Bash语法检查通过，旧Integrated/Query/Stage2/full/资源代码零diff。

本地只验证实现，不代表已经在 L40S 完整网格训练或获得质量/FPS。没有服务器执行连接，需运行以上命令。是否选为最终结构看质量、动态运动与资源取舍，不能预先保证更深更好。
