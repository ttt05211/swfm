# P0-F8 v2: True-Motion Anchor-Relative Edit World Model

Before the full 1200-step causal run, execute the real-GPU smoke and the
non-causal teacher-endpoint decision gate documented in
[`p0_f8_v2_smoke_and_teacher_ceiling.md`](p0_f8_v2_smoke_and_teacher_ceiling.md).

P0-F8 does **not** add another external router. Strong W2Det, frozen Real-Motion/MSP Top-2 routing, 15% write support, full 6-frame history, 40×40 context, frozen VAE, and NFE=10 remain unchanged.

Current training protocol:

```text
p0_f8_anchor_relative_edit_wm_v2
```

The decoder-aware deployment parameterization is:

```text
Strong W2Det anchor + full-history World Model
                  ↓
         predicted future latent
                  ↓ frozen VAE
         sparse semantic evidence
                  + exact anchor class
                  + horizon embedding
                  ↓
          KEEP / CLEAR / WRITE
                  ↓
       MSP-protected sparse writeback
```

## Why this is different from P0-F6/F7

P0-F6 predicts absolute dynamic semantics. Even when the anchor is already correct, the WM must generate the correct class again. P0-F8 instead predicts the minimal action relative to the exact Strong-W2Det anchor:

```text
anchor car / GT car  -> KEEP
anchor car / GT free -> CLEAR
anchor free / GT car -> WRITE car
anchor car / GT bus  -> WRITE bus
```

The edit head starts with a positive KEEP bias, so untrained behavior is close to exact anchor preservation.

## Class imbalance handling in v2

Every EDIT voxel is always retained for action CE. The total KEEP budget remains controlled by `--keep-ratio` (default `EDIT:KEEP = 1:1`), but KEEP is now **stratified** so background/non-dynamic KEEP cannot disappear from training.

With the frozen default:

```text
KEEP budget
├─ 50% correct dynamic KEEP
│  ├─ true-moving correct dynamic KEEP first
│  └─ other correct dynamic KEEP second
└─ 50% background / non-dynamic KEEP
```

If one stratum does not contain enough candidates, the shortage is filled from the other stratum so the requested total KEEP budget is preserved exactly.

True-motion GT is used only to prioritize hard dynamic KEEP examples during training. It is never an inference input.

## Loss and population contract

The latent flow loss is the P0-F6 uniform MSE. P0-F7 innovation-energy weighting is not used.

```text
L = L_FM + lambda_edit * L_edit
```

with

```text
L_edit = CE_balanced(KEEP/CLEAR/WRITE)
       + beta * Lovasz_full_pool(result semantic)
```

Default:

```text
beta = 0.5
```

The two edit terms intentionally use different populations:

```text
CE population:
  all EDIT + stratified balanced KEEP

Lovasz population:
  complete compact sidecar pool
  = all EDIT + all dynamic KEEP + bounded background KEEP
```

Lovasz is not applied to action IDs directly. Action probabilities are marginalized through the exact anchor into the resulting 9-way semantic distribution:

```text
background + 8 dynamic classes
```

This makes the IoU surrogate closer to the occupancy produced after applying KEEP/CLEAR/WRITE.

`lambda_edit` is gradient-calibrated once on shared transition parameters and then frozen.

The randomly initialized edit head has its own optimizer group. Its default
learning rate is `1e-3`, while `--lr` remains the learning rate for new/unloaded
transition parameters and `--backbone-lr-scale` controls reused transition
parameters. This separation is required because using the causal WM rate
(`2e-5`) for the edit head can leave the positive KEEP initialization unchanged.
Because this changes the optimizer parameter-group topology, start this run in a
new output directory; checkpoints created before the independent edit-head group
are rejected by `--resume-from`.

## Validation aggregation in v2

Validation must respect the same population split. P0-F8 v2 therefore aggregates:

```text
CE                    weighted by balanced CE voxel count
Lovasz                weighted by full compact-pool voxel count
balanced false-edit   weighted by sampled KEEP count
pool false-edit       weighted by full compact KEEP count
```

The validation edit objective is reconstructed as:

```text
val_edit = mean_balanced_CE + beta * mean_full_pool_Lovasz
val_objective = mean_FM + lambda_edit * val_edit
```

`best.pt` is selected using this corrected `val_objective`. All scheduled step checkpoints are still saved, and final model selection should use deployment metrics rather than validation objective alone.

Important v2 statistics retained in checkpoint/training history and validation records include:

```text
balanced_false_edit_rate
pool_false_edit_rate
num_pool_predicted_edits
pool_predicted_edit_fraction
dynamic_keep_fraction_realized
num_lovasz_voxels
num_dynamic_keeps
num_background_keeps
num_pool_keeps
num_pool_dynamic_keeps
num_pool_background_keeps
```

`false_edit_rate` is kept as a backward-compatible alias for the deployment-like `pool_false_edit_rate`.

---

## 1. Build the 4096-window training edit sidecar

The expensive WM latent cache is reused. Only exact Strong-W2Det / GT edit labels are built.

```bash
cd /root/nas/occ/swfm

python tools/real_motion/build_p0_f8_edit_targets_fast.py \
  --source-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --output /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_4096.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --workers 16 \
  --prefetch-samples 64 \
  --checkpoint-every 512 \
  --easy-keep-limit 4096
```

If interrupted, append:

```text
--resume
```

This step does **not** rebuild VAE latents and does not retrain MSP.

Existing P0-F8 v1-format edit sidecars are compatible with v2 sampling/validation logic and do not need to be rebuilt.

## 2. Build the validation edit sidecar

```bash
python tools/real_motion/build_p0_f8_edit_targets_fast.py \
  --source-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --output /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --workers 8 \
  --prefetch-samples 32
```

## 3. Train P0-F8 v2

```bash
python tools/real_motion/train_p0_f8_anchor_relative_edit_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --val-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f8_anchor_relative_edit_4096 \
  --steps 1200 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 2e-5 \
  --edit-head-lr 1e-3 \
  --backbone-lr-scale 1.0 \
  --keep-ratio 1.0 \
  --keep-when-no-edit 64 \
  --keep-bias 2.0 \
  --lovasz-weight 0.5 \
  --edit-grad-ratio 0.5 \
  --min-train-windows 4000 \
  --val-every 200 \
  --sample-steps 10 \
  --collapse-check-step 200 \
  --amp
```

P0-F8 automatically saves:

```text
step_0200.pt
step_0400.pt
step_0600.pt
step_0800.pt
step_1000.pt
step_1200.pt
latest.pt
best.pt
last.pt
```

Do not resume a P0-F8 v1 checkpoint into v2. The protocol field is intentionally different.

Important standard console logs remain:

```text
edit_lambda_calibration ...
step=... fm=... edit=... ce=... lovasz=...
edit_acc=...
false_edit=...
pred_edit=...
E/K=.../...
```

With `--collapse-check-step 200`, every due validation records an
`all_keep_validation_gate_v1` block. If the complete compact validation pool
contains zero predicted non-KEEP actions, the trainer first saves `latest.pt`,
the scheduled `step_XXXX.pt`, and `best.pt` when applicable, then exits the loop
cleanly. `training_report.json` records
`EARLY_STOPPED_ALL_KEEP_COLLAPSE`; this prevents a known all-KEEP run from
consuming the remainder of the 1200-step budget. Set the option to `0` to
disable this guard.

At validation steps, the JSON record additionally exposes the v2 population statistics listed above. The same v2-only training statistics are persisted into the validation-step `training_history[*].train` record inside checkpoints and `training_report.json`.

The intended behavior is:

- `edit_accuracy` rises;
- `pool_predicted_edit_fraction` is nonzero by the step-200 gate;
- `pool_false_edit_rate` stays low;
- `dynamic_keep_fraction_realized` stays near 0.5 when both KEEP strata are sufficiently populated;
- `num_lovasz_voxels >= num_supervised_voxels` in the usual case;
- deployment metrics, not validation objective alone, choose the final checkpoint.

## 4. Evaluate 200–1200

```bash
OUT=/root/nas/occ/swfm/outputs/p0_f8_anchor_relative_edit_4096
VAL=/root/nas/occ/swfm/data/p0_f5_wm_val_top2
VAE=/root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt

for STEP in 200 400 600 800 1000 1200; do
  python tools/real_motion/eval_p0_f8_anchor_relative_edit_wm.py \
    --cache "$VAL" \
    --vae-ckpt "$VAE" \
    --sparse-ckpt "$OUT/step_$(printf '%04d' $STEP).pt" \
    --output "$OUT/eval_${STEP}.json" \
    --amp
done
```

### Diagnose edit calibration with a one-pass KEEP-margin sweep

If a checkpoint predicts nonzero edits but harms the Strong-W2Det anchor, sweep
an inference-only additive KEEP-logit margin before spending more training
steps. The causal WM sample, sparse VAE decode, and edit-head logits are computed
once per window and shared by every margin:

```bash
python tools/real_motion/eval_p0_f8_anchor_relative_edit_wm.py \
  --cache "$VAL" \
  --vae-ckpt "$VAE" \
  --sparse-ckpt "$OUT/step_0200.pt" \
  --output "$OUT/keep_margin_sweep_200.json" \
  --keep-logit-margins 0 0.25 0.5 0.75 1.0 1.5 2.0 \
  --min-delta-overall 0.0 \
  --min-delta-moving 0.0 \
  --min-delta-moving-1s -0.5 \
  --amp
```

Each `margin_results` entry contains deployment metrics, raw/effective action
statistics, and its point-estimate decision gate. `selection.selected_margin`
is populated only when at least one margin satisfies all predeclared checks.
Among passing margins, ranking is frozen to Moving delta, Overall delta, lower
effective false-edit rate, then lower margin. Do not treat this validation-cache
sweep as an unbiased final result: lock the selected margin before an independent
evaluation or paired scene bootstrap.

Each deployment report includes:

```text
Overall mIoU / delta Overall
Moving-mIoU / delta Moving
Overall @1s/2s/3s
Moving @1s/2s/3s
action histogram
raw and effective action histograms
raw edit and effective change fractions of causal support
no-op CLEAR/WRITE counts
effective false-edit rate on same-support target KEEP voxels
same-support GT oracle
```

`action_histogram` is retained as a backward-compatible alias for
`raw_action_histogram`.  Raw `CLEAR` on a non-dynamic anchor and raw `WRITE`
of the existing dynamic class are deployed no-ops, so use
`effective_action_histogram` and `effective_edit_fraction_of_support` when
interpreting how much the final occupancy actually changed.

Success criterion:

```text
Delta Overall >= 0
Delta Moving  > 0
```

The main question is whether explicit anchor-relative KEEP/CLEAR/WRITE increases repair gain without recreating the 1s anchor harm seen in P0-F6/F7.

---

## 5. Frozen causal-endpoint head probe

Run this diagnostic when the causal checkpoint predicts edits but every
predeclared KEEP-margin fails the deployment gate.  It distinguishes an
unusable causal representation from joint-training/edit-head optimization.

The original joint trainer supervises the head on the endpoint estimate from a
flow-matching interpolation state.  The deployed head instead receives the
deterministic multi-step rollout that starts from the Strong-W2Det anchor.  This
probe removes that input-distribution difference while keeping the causal WM
representation fixed:

```text
cached history + Strong-W2Det anchor
        -> frozen checkpoint transition
        -> exact deterministic NFE=10 deployment rollout
        -> frozen VAE sparse semantic evidence
        -> fresh trainable edit head only
```

The rollout never reads `repair_target_latent` or future GT.  GT-derived edit
targets remain ordinary training labels only.  The jointly trained edit head is
discarded before the probe starts.

Use the exact causal checkpoint that failed the margin sweep:

```bash
ROOT=/root/nas/occ/swfm
TRAIN="$ROOT/data/p0_f7_wm_train_top2_4096"
VAL="$ROOT/data/p0_f5_wm_val_top2"
TRAIN_EDIT="$ROOT/data/p0_f8_edit_train_4096.pt"
VAL_EDIT="$ROOT/data/p0_f8_edit_val.pt"
VAE=/root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt
CAUSAL_OUT="$ROOT/outputs/p0_f8_anchor_relative_edit_4096"
CAUSAL="$CAUSAL_OUT/step_0200.pt"
PROBE_OUT="$ROOT/outputs/p0_f8_frozen_causal_probe_step200"

python tools/real_motion/train_p0_f8_frozen_causal_endpoint_probe.py \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --train-edit-targets "$TRAIN_EDIT" \
  --val-edit-targets "$VAL_EDIT" \
  --causal-ckpt "$CAUSAL" \
  --vae-ckpt "$VAE" \
  --output-dir "$PROBE_OUT" \
  --steps 600 \
  --val-every 200 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --min-train-windows 4000 \
  --keep-ratio 1.0 \
  --keep-when-no-edit 64 \
  --keep-bias 2.0 \
  --lovasz-weight 0.5 \
  --collapse-check-step 200 \
  --seed 20260903 \
  --amp
```

The probe checkpoints contain only the edit head and are cryptographically
bound to the exact causal checkpoint and VAE.  To limit storage it writes only:

```text
best.pt
latest.pt
last.pt
training_report.json
```

There are no numbered probe checkpoints.  Evaluate `best.pt` with the same
predeclared one-pass margin sweep:

```bash
python tools/real_motion/eval_p0_f8_anchor_relative_edit_wm.py \
  --cache "$VAL" \
  --vae-ckpt "$VAE" \
  --sparse-ckpt "$CAUSAL" \
  --edit-head-probe-ckpt "$PROBE_OUT/best.pt" \
  --output "$PROBE_OUT/keep_margin_sweep_best.json" \
  --keep-logit-margins 0 0.25 0.5 0.75 1.0 1.5 2.0 \
  --min-delta-overall 0.0 \
  --min-delta-moving 0.0 \
  --min-delta-moving-1s -0.5 \
  --amp
```

Interpretation is frozen before the run:

- any predeclared margin passes: the frozen causal endpoint contains usable
  action information; the original joint training/input mismatch is the main
  suspect, and staged head training is justified;
- no margin passes: a fresh head cannot recover useful edits from the fixed
  causal endpoint, so change the representation/causal WM rather than tuning
  KEEP bias or extending the same head objective.
