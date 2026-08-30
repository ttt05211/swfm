# P0-F6：Decoder-Aware Sparse Innovation

P0-F6 不再改 Real-Motion、MSP、Top-2、Strong W2Det、full-history context 或 P0-F5 occupancy repair endpoint。P0-F5 已经证明：即使 latent FM validation loss 很低，decoder 后的 Moving-mIoU 仍可能显著低于 Strong W2Det。因此 P0-F6 只补上 latent objective 与最终 occupancy semantics 之间的闭环。

## 冻结不变

- Strong W2Det causal anchor。
- Frozen Real-Motion decomposition + MSP。
- Top-2 `20×20` prediction windows。
- 15% causal MSP write budget。
- Full 6-frame history latent。
- `40×40` history context。
- P0-F5 occupancy-space sparse repair endpoint：`Z_repair = Enc(repair_occ)`。
- OccFM-Fut epoch 196 initialization。
- Source noise = 0，sampling NFE = 10。
- 最终 occupancy writeback：support 外 exact Strong W2Det；support 内先清 Strong-W2Det dynamic，再只写 decoded WM dynamic semantics。

## 唯一方法修改：decoder-aware semantic repair loss

仍保留 P0-F5 的 FM：

```text
z_t = (1-t) Z_anchor + t Z_repair
v*  = Z_repair - Z_anchor
L_FM = MSE(v_pred, v*)
```

同一次 WM forward 直接得到 endpoint estimate：

```text
Z_hat_repair = (z_t + (1-t) v_pred) / latent_rescale
```

两个 Top-2 local endpoint 先按最终 inference 同样的 overlap-average scatter 回完整 Strong-W2Det latent，然后走 frozen official VAE decoder。Decoder 参数始终 `requires_grad=False`，但 semantic gradient 可以穿过 decoder 回到 `Z_hat_repair` 与 WM transition。

### 9-way dynamic repair semantics

Occ3D 的 18 类被折成：

```text
0: background / any non-dynamic semantic
1..8: bicycle, bus, car, construction_vehicle,
      motorcycle, pedestrian, trailer, truck
```

background logit 是所有 non-dynamic logits 的 `logsumexp`，因为最终 fusion 对 road/free/barrier 等所有非动态 decoder 输出的处理完全相同：都不写动态。

semantic supervision 只放在 causal MSP write support 内，并且只监督：

```text
GT dynamic  OR  Strong-W2Det-anchor decode dynamic
```

因此一个 CE 同时覆盖：

```text
anchor dynamic + GT same dynamic    -> keep
anchor dynamic + GT non-dynamic     -> remove
anchor non-dynamic + GT dynamic     -> create
```

不会让整个 16-height column 的大量无关 background voxel 淹没动态监督。

最终：

```text
L = L_FM + lambda_sem * L_sem
```

没有第三个 loss。

## lambda_sem：首个 batch 梯度尺度自动校准

默认不手填 `lambda_sem`。在第一个包含 semantic supervision 的训练 batch 上分别计算：

```text
g_FM  = || grad_theta L_FM ||_2
g_sem = || grad_theta L_sem ||_2
```

然后固定：

```text
lambda_sem = 0.5 * g_FM / g_sem
```

默认 clamp 到 `[1e-4, 10]`。这一步只执行一次，之后整个训练不再动态调整，并写入 checkpoint。训练第一步会打印：

```text
semantic_lambda_calibration {
  fm_grad_norm,
  semantic_grad_norm_unweighted,
  raw_lambda,
  lambda,
  realized_semantic_to_fm_grad_ratio,
  ...
}
```

如果想做显式固定 lambda，可使用 `--semantic-lambda X`，但 P0-F6 主实验默认使用自动首-batch 校准。

## 不需要重建 P0-F5 cache

继续直接使用：

```text
/root/nas/occ/swfm/data/p0_f5_wm_train_top2
/root/nas/occ/swfm/data/p0_f5_wm_val_top2
```

只新增两个很小的 sparse semantic sidecar。

## 1. 构建 train semantic sidecar

train cache 没有 future GT eval payload，因此只读取 6 个 future Occ3D semantics；不会重算 Strong W2Det、MSP、VAE encode。Anchor semantic mask 只把已经缓存的 `anchor_future_latent` frozen-decode 一次。

```bash
python tools/real_motion/build_p0_f6_semantic_targets.py \
  --source-cache /root/nas/occ/swfm/data/p0_f5_wm_train_top2 \
  --output /root/nas/occ/swfm/data/p0_f6_semantic_train.pt \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_1024.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --batch-size 8
```

中断后同命令加 `--resume`。

## 2. 构建 val semantic sidecar

P0-F5 val cache 已保存 future GT，因此不需要 dataroot / MSP probe：

```bash
python tools/real_motion/build_p0_f6_semantic_targets.py \
  --source-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --output /root/nas/occ/swfm/data/p0_f6_semantic_val.pt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --batch-size 8
```

## 3. 训练 P0-F6

P0-F6 仍从官方 OccFM-Fut epoch 196 初始化，不从失败的 P0-F5 checkpoint resume。

```bash
python tools/real_motion/train_p0_f6_decoder_aware_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f5_wm_train_top2 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-semantic-targets /root/nas/occ/swfm/data/p0_f6_semantic_train.pt \
  --val-semantic-targets /root/nas/occ/swfm/data/p0_f6_semantic_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f6_decoder_aware \
  --steps 3000 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-5 \
  --val-every 200 \
  --sample-steps 10 \
  --semantic-grad-ratio 0.5 \
  --amp
```

Decoder-aware backward 会比 P0-F5 占更多显存。如果当前 GPU 上 `batch-size=2` OOM，只把 batch 改成 1；不要同时改 LR、Top-K、MSP budget 或 loss contract。

Checkpoint 不再按单独 latent FM MSE 选择，而按：

```text
val_objective = val_FM + lambda_sem * val_semantic_CE
```

选择 `best.pt`。

## 4. 最终 occupancy evaluation

```bash
python tools/real_motion/eval_p0_f6_decoder_aware_wm.py \
  --cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt /root/nas/occ/swfm/outputs/p0_f6_decoder_aware/best.pt \
  --output /root/nas/occ/swfm/outputs/p0_f6_decoder_aware/eval.json \
  --amp
```

最终仍只比较同协议三方：

1. Strong W2Det。
2. Strong W2Det + trained P0-F6 Sparse WM。
3. Same-support GT dynamic repair oracle。

冻结 baseline：

```text
Strong W2Det Overall = 39.75
Strong W2Det Moving  = 21.39
GT repair oracle Moving = 58.07
```

P0-F6 成功条件仍然只有一个：trained Sparse WM 的 Moving-mIoU 必须超过同协议 Strong W2Det，而不是只看 semantic CE 或 latent FM loss 是否下降。
