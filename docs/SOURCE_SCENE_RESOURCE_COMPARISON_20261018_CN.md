# Baseline / Scene Reader 资源对比

独立 `tools/source_scene_speed`，不改训练入口或训练代码指纹、不重训、不修改 checkpoint/Adam/旧历史缓存。只读本次两个完成的 `last.pt` 及原 `progress.jsonl`，不是旧 clean1000/norm192 模型。

本地相同生产架构精确计数（服务器再从实际checkpoint核对）：baseline 2,054,435，Reader 1,573,186，总量减少481,249（23.4249%）。Encoder均116,876、Refiner均46,483；motion去Encoder分别1,891,076 / 1,409,827。未有服务器原日志或新权重，当前不能填写实际训练秒/窗或服务器FPS。

本地新专项＋原Stage1共19tests passed10.44s，含小网格CUDA完整入口两臂/GT和STC/只读文件/实际原入口CPU非零残差六帧一致；Bash语法和CLI通过。小网格不是实尺寸GPU吞吐结果。原Stage1训练文件无修改，因此不会因本轮测速入口引入resume指纹变化。

## 比较内容

- 参数总量，以及 Encoder、去 Encoder 的 motion、Refiner 分项；严格加载实际已训模型并核对 contract 参数数。
- 原训练日志 first16 update 之后和末256 update：按真实窗口数加权。是原两卡 global B4 的 `max(rank pipeline elapsed)/global windows`，含输入等待和梯度通信，不含 checkpoint/评估/启动；不是单卡训练速度。恢复导致重复 update 时固定取最后一行，不择快。两组覆盖 update 必须一致。
- 同 remaining64、GT/STC、单GPU单进程，每次仅一个模型驻GPU，两pass模型顺序相反。报告六帧延迟均值/P90、窗口FPS、输出帧FPS（六倍），不平均各窗口FPS。
- 独立串行分段 profile（前4窗，不纳入FPS），可以看 Reader 本身 `encoder_motion_lift` 时间和共同几何/Refiner 时间。

## FPS边界

历史I/O、CC、共享关联、registration、读坐标准备和初始语义/read/pose上传在外；原两帧KTA实时重算、fresh Strong/CLEAR、共享 Encoder、STWM/Reader、lift、Transport、memory、Refiner、argmax与完整六帧CPU输出在内。背景/特征/投影每call重建。无 half-KTA、age、静态支持、校准、norm192、packed优化、CUDA graph、精度切换。不能直接继承旧41.78FPS，也不是相机端到端FPS。

每窗另做原 live entrance 两次及新执行语义差异检查。BF16原重复和跨入口strict byte FAIL全部保留，失败样本不剔除；实际执行FPS与等价性分开报告，不自动推广。两组差异不能要求模型间输出相同，只在各模型内部检查原入口与测速入口。profile不加入FPS均值；显存 reserved 受allocator历史影响，不当作纯模型需求。

## 服务器

```bash
cd /root/nas/occ/swfm
conda activate OccFM
# SCENE_PILOT_DIR 若仍在环境中，直接复用。
# 若不存在，脚本仅在找到唯一完成的 scene_reader_stage1_1ep_* 时自动识别；多个则列出路径，明确指定后再跑。
export SCENE_SPEED_GPU=0
export SCENE_SPEED_DIR="$PWD/outputs/p0_f9_shared_history/scene_reader_speed_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_speed/run.sh > "${SCENE_SPEED_DIR}.log" 2>&1 &
tail -F "${SCENE_SPEED_DIR}.log"
```

完成读 `$SCENE_SPEED_DIR/summary.txt`；原始timings在samples.jsonl。新目录运行，无训练或多进程吞吐混入。中断后的部分timings保留，不充完整FPS；重测使用新目录。正式结果须服务器运行回传，本地没有本次训练日志/权重，不能编造训练速度或服务器FPS。
