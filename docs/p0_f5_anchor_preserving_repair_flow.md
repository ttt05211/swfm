# P0-F5：Anchor-Preserving Sparse Repair Flow

P0-F5 不重新打开 Real-Motion / MSP / Top-2 的诊断树，也不新增 loss、router 或 selector。它只修正 P0-F4 暴露出的训练 endpoint 不一致问题。

## 冻结不变

- Strong W2Det causal anchor。
- Frozen Real-Motion decomposition + MSP。
- Top-2 `20×20` future compute windows。
- `40×40` full-history latent context。
- OccFM-Fut epoch 196 初始化并 fine-tune。
- source noise = 0，sample NFE = 10。
- 最终 occupancy-space causal write protection。

## 唯一核心修改

P0-F4 训练的是

```text
Z_anchor -> Enc(full future GT)
```

再在 latent 上用 MSP hard mask 选择 MSE 区域。这隐含假设 latent cell 与局部 occupancy edit 一一对应，但 VAE encoder/decoder 有空间 receptive field，因此训练目标和最终 sparse semantic repair 并不一致。

P0-F5 先在 occupancy 空间定义真实部署 endpoint：

```text
repair_occ = Strong W2Det
outside causal MSP support: exact Strong W2Det
inside causal MSP support:
  clear Strong-W2Det dynamic voxels
  insert GT dynamic voxels
```

然后：

```text
Z_anchor = Enc(Strong W2Det)
Z_repair = Enc(repair_occ)
```

训练普通 anchor-centered flow：

```text
z_t = (1-t) Z_anchor + t Z_repair
v*  = Z_repair - Z_anchor
L   = MSE(v_pred, v*)
```

**没有 latent loss mask。** `msp_write_support_latent` 只用于定义 occupancy-space repair endpoint 和最终推理写权限。

## 服务器更新

```bash
cd /root/nas/occ/swfm
git pull https://gh-proxy.com/https://github.com/ttt05211/swfm.git main
```

## 1. 推荐：直接把已有 P0-F4 cache 增量升级为 P0-F5

不要重新跑 MSP，也不要重新编码 full history / Strong-W2Det anchor。`upgrade_p0_f4_cache_to_p0_f5.py` 会 bit-exact 复用已有 P0-F4 的：

```text
full_history_latent
anchor_future_latent
window_origins
window_valid
msp_write_support_latent
trajectory
```

GPU 只新增：

```text
repair endpoint occupancy -> Frozen VAE(mean) -> repair_target_latent
```

### Train cache

P0-F4 train cache 没保存 raw future GT / Strong-W2Det occupancy，因此仍需按 frozen MSP probe 的 exact sample IDs 回读 raw window，并在 CPU 上重建一次 Strong W2Det occupancy；**不会重跑 MSP，GPU 也不会重编码 history/anchor**。

```bash
python tools/real_motion/upgrade_p0_f4_cache_to_p0_f5.py \
  --source-cache /root/nas/occ/swfm/data/p0_f4_wm_train_top2 \
  --output /root/nas/occ/swfm/data/p0_f5_wm_train_top2 \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_1024.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --vae-batch-size 8 \
  --shard-size 16
```

如果 P0-F4 metadata 里记录的 VAE / MSP probe 路径在当前服务器仍有效，则 `--vae-ckpt` 和 `--msp-cache` 可以省略；脚本会同时校验 SHA256，防止错用文件。

中断后原命令追加：

```bash
--resume
```

### Val cache

P0-F4 val cache 已保存：

```text
eval_future_gt_occ
eval_strong_anchor_occ
```

所以升级 val 时**不读取 nuScenes、不重算 Strong W2Det、不需要 MSP probe**，只构造 repair endpoint 并做一次 VAE encode：

```bash
python tools/real_motion/upgrade_p0_f4_cache_to_p0_f5.py \
  --source-cache /root/nas/occ/swfm/data/p0_f4_wm_val_top2 \
  --output /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --vae-batch-size 8 \
  --shard-size 16
```

升级后的 v3 cache 会额外记录：

```text
incremental_upgrade = true
incremental_reused_tensor_keys = [...]
incremental_new_gpu_encode = [repair_target_latent]
```

验证 cache 还会新增 `eval_repair_target_occ`，供最终 evaluator 逐体素确认训练 endpoint 与 same-support GT repair oracle 完全一致。

## 2. 仅在没有 P0-F4 cache 时：从头构建 P0-F5 cache

完整 builder 仍保留：

```bash
python tools/real_motion/build_p0_f5_cache_direct.py ...
```

但已有 P0-F4 cache 时不推荐走这条路径，因为它会重复 MSP routing、history VAE encode 和 anchor VAE encode。

## 3. 训练

P0-F5 不从 P0-F4 checkpoint resume，因为 target contract 已改变。仍从官方 OccFM-Fut-196 初始化。

```bash
python tools/real_motion/train_p0_f5_sparse_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f5_wm_train_top2 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f5_top2_sparse_wm \
  --steps 3000 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-5 \
  --val-every 200 \
  --sample-steps 10 \
  --amp
```

## 4. 最终 occupancy evaluation

```bash
python tools/real_motion/eval_p0_f5_sparse_wm.py \
  --cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt /root/nas/occ/swfm/outputs/p0_f5_top2_sparse_wm/best.pt \
  --output /root/nas/occ/swfm/outputs/p0_f5_top2_sparse_wm/eval.json \
  --amp
```

最终报告仍只看同协议三方：

1. Strong W2Det。
2. Strong W2Det + trained P0-F5 Sparse WM。
3. Same-support GT dynamic repair oracle。

P0-F4 在 128-window 协议上的冻结 baseline 为 `Strong W2Det Moving=21.39`。P0-F5 是否成功只由 trained Sparse WM 是否在同协议上超过 Strong W2Det 决定。
