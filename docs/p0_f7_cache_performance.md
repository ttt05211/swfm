# P0-F7 高吞吐 Cache 构建

这套脚本只做工程提速，不改变 P0-F7 的实验协议：Strong W2Det、冻结 MSP、Top-2、15% write support、occupancy-space repair endpoint、FP32 deterministic VAE mean 都保持不变。

## 1. 推荐环境

多线程 CPU preparation 时避免 NumPy/BLAS 每个 worker 再开一组线程造成 oversubscription：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

如果节点 CPU 核很多，`--workers/--prepare-workers 0` 会自动最多使用 16 个线程。可根据 `htop`、磁盘吞吐和 GPU 利用率手动提高或降低。

## 2. 4096 个 MSP windows

最快路径可复用旧 1024 probe records。复用前会检查 seed、stride、config、feature/target contract 和 matching distance；不兼容直接报错。

```bash
python tools/real_motion/p0_msp_build_dataset.py \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --mode train \
  --max-windows 4096 \
  --workers 16 \
  --prefetch-windows 64 \
  --reuse-cache /root/nas/occ/swfm/data/msp_probe_train_1024.pt \
  --output /root/nas/occ/swfm/data/msp_probe_train_4096.pt
```

如果旧文件名不同，只替换 `--reuse-cache` 路径。若希望完全重新构建所有 probe records，删掉 `--reuse-cache` 即可。

## 3. 高吞吐 WM cache

推荐先从下面配置开始：

```bash
python tools/real_motion/build_p0_f7_cache_fast.py \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_4096.pt \
  --msp-checkpoint /root/nas/occ/swfm/outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --route-batch-size 128 \
  --vae-batch-size 16 \
  --prepare-workers 16 \
  --prefetch-windows 64 \
  --shard-size 32 \
  --max-pending-shards 2 \
  --pin-memory \
  --async-write
```

提速机制：

- routing 一次处理 128 个窗口；
- 16 个 CPU worker 并行做 raw file I/O、Strong W2Det、repair endpoint；
- 最多 64 个 window bounded prefetch，CPU 在 GPU VAE encode 时继续准备后续数据；
- host semantic occupancy 保持 uint8，并用 pinned memory/non-blocking transfer；long cast 在 GPU 上完成，减少 PCIe host payload；
- VAE 每次 encode 16 个完整 6-frame windows；
- shard `torch.save` 在独立 writer thread 中执行，与下一批 preparation/encoding 重叠；
- 输出顺序仍与 serial builder 完全一致；
- `--resume` 只从已经 durable 的 shard/index 继续，未落盘的 in-flight batch 会安全重做。

### 显存足够时继续压榨 GPU

优先尝试：

```text
--vae-batch-size 16 -> 24 -> 32
```

每提升一次先观察显存和 GPU util。OOM 后只降低 VAE batch，不需要重建已有 shard；同目录加 `--resume`。

CPU/磁盘没有打满时：

```text
--prepare-workers 16 -> 24/32
--prefetch-windows 64 -> 96/128
```

但网络盘/共享盘并不是 worker 越多越快。若 iowait 很高，回退到 8~16 通常更快。

### 可选 reuse old 1024 WM cache

`build_p0_f7_cache_fast.py` 支持：

```bash
--reuse-cache /root/nas/occ/swfm/data/p0_f5_wm_train_top2
```

它会逐 sample 检查当前 frozen-MSP route 与旧 cache 的 `window_origins/window_valid/write_support` bit-exact 一致后才复用。

**主 P0-F7 科研实验默认建议不加这个参数。** 之前已经观察到 official VAE encode 对 batch path 存在轻微数值差异；完全重新 encode 4096 windows 可以保证新 train cache 内使用统一 batching path。`--reuse-cache` 更适合快速工程 smoke 或时间非常紧张时。

## 4. 快速 semantic sidecar

原 P0-F6 builder 每个小 batch 都把不断增长的完整 records list 重写一次，扩到 4096 后会产生大量重复 I/O。F7 fast builder 改为：未来 GT 多线程读取 + 大 batch frozen decoder + 每 512 sample 才保存一次 resume checkpoint。

```bash
python tools/real_motion/build_p0_f7_semantic_targets_fast.py \
  --source-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --output /root/nas/occ/swfm/data/p0_f7_semantic_train_4096.pt \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_4096.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --batch-size 32 \
  --workers 16 \
  --prefetch-samples 64 \
  --checkpoint-every 512 \
  --pin-memory
```

若 decoder OOM：`--batch-size 32 -> 16`。中断后加 `--resume`。

## 5. 训练规模

4096 train windows + batch size 8：

```text
4096 / 8 = 512 steps / epoch
```

第一轮 P0-F7 推荐 `1000~1200 steps`，约 2~2.3 epochs，而不是 4000 steps。
