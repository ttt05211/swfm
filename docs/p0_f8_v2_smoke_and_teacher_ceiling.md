# P0-F8 v2 GPU smoke and teacher-endpoint decision gate

This document defines two different experiments. They must not be interpreted
as interchangeable results.

1. The GPU smoke checks that the real causal P0-F8 v2 training, validation, and
   checkpoint path works on the target GPU.
2. The teacher-endpoint ceiling bypasses causal prediction and asks whether the
   frozen VAE decoder plus P0-F8 action head can beat Strong W2Det when given the
   GT-derived cached repair endpoint.

The required caches and edit sidecars are reused. Neither experiment recaches
latents or rebuilds MSP.

## 1. Run the real P0-F8 v2 GPU smoke

Use a new output directory. The wrapper refuses to overwrite an existing run.

```bash
cd /root/nas/occ/swfm

python tools/real_motion/smoke_p0_f8_v2.py \
  --train-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --val-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f8_v2_smoke_20260903 \
  --steps 20 \
  --batch-size 8 \
  --num-workers 4 \
  --amp
```

The command runs exactly one validation at the final smoke step, reloads
`latest.pt` into the full model with `strict=True`, validates the v2 population
fields, and writes:

```text
outputs/p0_f8_v2_smoke_20260903/smoke_report.json
```

The terminal success marker is:

```text
P0-F8_V2_GPU_SMOKE_PASS
```

A passing smoke is an engineering gate only. It is not evidence that P0-F8
beats KTA.

## 2. Train the teacher-repair-endpoint ceiling

This diagnostic intentionally uses future GT through the cached
`repair_target_latent`. It is not causal and must not be reported as the final
method. No OccFM-Fut checkpoint is loaded and no transition parameters exist in
the checkpoint; only `AnchorRelativeEditHead` is optimized.

```bash
cd /root/nas/occ/swfm

python tools/real_motion/train_p0_f8_teacher_endpoint.py \
  --train-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --val-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f8_teacher_endpoint_600 \
  --steps 600 \
  --val-every 200 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 1e-3 \
  --keep-ratio 1.0 \
  --keep-when-no-edit 64 \
  --keep-bias 2.0 \
  --lovasz-weight 0.5 \
  --amp
```

This produces `step_0200.pt`, `step_0400.pt`, `step_0600.pt`, `latest.pt`,
`best.pt`, `last.pt`, and `training_report.json`.

## 3. Evaluate the fixed teacher checkpoints

Evaluate all three scheduled checkpoints. Do not select a checkpoint only after
looking at the test metrics; retain the complete curve in the experiment
record.

```bash
cd /root/nas/occ/swfm

OUT=/root/nas/occ/swfm/outputs/p0_f8_teacher_endpoint_600
VAL=/root/nas/occ/swfm/data/p0_f5_wm_val_top2
VAE=/root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt

for STEP in 200 400 600; do
  python tools/real_motion/eval_p0_f8_teacher_endpoint.py \
    --cache "$VAL" \
    --vae-ckpt "$VAE" \
    --teacher-ckpt "$OUT/step_$(printf '%04d' "$STEP").pt" \
    --output "$OUT/eval_${STEP}.json" \
    --min-delta-overall 0.0 \
    --min-delta-moving 3.0 \
    --min-delta-moving-1s -0.5 \
    --amp
done
```

Each report includes Strong-W2Det, teacher endpoint, and same-support GT oracle
metrics, plus a `decision_gate` block. The frozen point-estimate gate is:

```text
Delta Overall >= 0.0 pp
Delta Moving  >= 3.0 pp
Delta Moving @1s >= -0.5 pp
```

The gate is advisory by default so a scientifically negative result still
produces a normal report. Add `--fail-on-gate` only when a nonzero process exit
is useful for automation.

Interpretation:

- PASS: the VAE/action path has meaningful headroom; proceed to the 1200-step
  causal P0-F8 v2 run.
- FAIL: do not spend the full causal run yet; first inspect the VAE evidence,
  action prior/calibration, and edit representation.

The evaluator currently reports a point-estimate gate. A paired scene-bootstrap
confidence interval requires a separate per-scene metric export and is not
silently approximated here.
