# P0-F9 V17-RL C0: source-level control fidelity for scene supervision

## Why C0 exists

The first C experiment changed more than the intended scene loss. Its control
used 4-window scene batching and froze the historical encoder. The control
therefore fell well below the shared RL epoch-5 start, so C-S vs C-C measured
scene supervision under a changed continuation protocol rather than a clean
"original RL + scene loss" treatment.

C0 fixes that confound before interpreting scene supervision.

## Base contract

The base path is the historical V17-RL continuation contract:

- flatten only supervised V17 sources exactly as the original trainer;
- use the checkpoint batch size (historically 256 sources);
- restore the epoch-5 AdamW state;
- keep **all V17 parameters trainable**;
- use the original `L_pos + L_exist + 0.25 L_overlap` objective;
- use the same remaining 10-epoch cosine LR schedule;
- use a fixed source-level shuffle seed.

`C0-C` contains only this path.

`C0-S` keeps the same source batch and optimizer step, then adds an independent
scene gradient before clipping/`optimizer.step()`:

```text
C0-C: grad = grad(L_RL(source_batch))
C0-S: grad = grad(L_RL(source_batch))
             + alpha(t) * grad(L_scene(scene_batch))
```

The scene batch never replaces, filters, or rebatches the source-level base
batch. The scene renderer is the same sparse-compute/full-scene-normalized
18-way CE introduced in C. It uses no GT-moving source filter and no GT support
to define the query region.

## RNG/state isolation

The extra scene forward is wrapped so its RNG consumption cannot perturb the
next source-batch RNG sequence. The script also rejects BatchNorm-like running
statistics; V17 uses LayerNorm/attention, so the scene forward contributes only
the intended gradient treatment before the optimizer step.

## Alpha calibration

The old C alpha must **not** be reused. C0 changes both the parameter scope and
the base batch contract.

C0 calibrates once using:

- motion gradient from a real 256-source historical RL batch;
- scene gradient from an independent scene minibatch;
- all trainable residual-path parameters (existence head excluded from the
  gradient-scale comparison because scene CE has no existence-head path).

The default target remains 0.25 scene-gradient / motion-gradient by median
norm. No alpha sweep or dynamic balancing is used.

## Efficient experiment order

1. **Control fidelity first.** Run C0-C for three continuation epochs
   (epoch 5 -> epoch 8). Evaluate `C0-C epoch_0008` with the A1 evaluator and
   compare it to the already-run historical `B-C epoch_0008`. These should be
   nearly identical; a material mismatch means C0 is not yet a valid control.
2. During the same C0-C run, save `step_000300.pt` and `step_000600.pt`.
3. If control fidelity passes, calibrate C0 alpha.
4. Run only C0-S for 600 source optimizer steps. It uses the same first 600
   source batches as C0-C and saves the same step checkpoints.
5. Compare C0-S vs C0-C at step 300 and step 600, and also compare C0-S against
   the shared RL epoch-5+A1 start.

This avoids paying for scene rendering over three full epochs merely to prove
the control contract.

## Decision

The experiment has two gates.

**Fidelity gate:** `C0-C epoch8` must reproduce historical `B-C epoch8` under the
same A1 hard evaluator closely enough that any residual difference is clearly
smaller than the scene-supervision effect of interest.

**Treatment gate:** only after fidelity passes, inspect fixed-time
`C0-S - C0-C` and `C0-S - start` on occupancy IoU, semantic mIoU, and
Moving-mIoU. Do not select a checkpoint by ADE or scene CE.
