# Shared-History Execution Speed V2

本分支只做执行提速候选，不改变当前 Shared-History 科学模型。

## 冻结不变

- REMOVE 路径完整保留，不删除静态/动态 REMOVE。
- 六个 future 全部计算 CE + Lovasz。
- source motion / existence 六帧监督不变。
- ego 六帧监督不变。
- Shared Encoder、STWM、CCR/Surface、KTA、SE(2)、缓存协议不变。
- 原训练 checkpoint、Adam、RNG、cosine 和输出目录只读；测速不会发布 scientific update。

因此本轮不包含 2/6 horizon、periodic Lovasz、sparse CE、Encoder 缩减/冻结或 REMOVE-off renderer。

## 候选

一次只读 probe 比较六种执行：

    current_h2      当前正式 trainer：逐窗网络 + [01][23][45]
    sequential_h3   逐窗网络不变 + [012][345]
    sequential_h6   逐窗网络不变 + [012345]
    packed_h2       global B<=4 Shared Encoder/Motion/Repair 合批 + [01][23][45]
    packed_h3       global B<=4 Shared Encoder/Motion/Repair 合批 + [012][345]
    packed_h6       global B<=4 Shared Encoder/Motion/Repair 合批 + [012345]

Packed 版本只合并 trainable network。每个 scene 的 causal projection / renderer 仍保持独立，用于先隔离网络合批本身的真实收益。跨 scene renderer batching 暂未混入这一轮。

## Packed B4 的具体执行

原来 global batch=4 实际是四个窗口依次执行；候选改成：

    history[B,4] -> one SharedEncoder stem
                      |
    all source tensors concatenate
                      |
              one STWM / decoder
                      |
    all canonical point histories concatenate
                      |
              one Repair readout
                      |
    split back to four scene boundaries
                      |
             same per-scene renderer/loss
                      |
             one multi-root shared backward

不同 scene 的 source / canonical point 数量仍为 ragged；通过 source length、point length 和 actor offset 保持原 identity/order，不 pad 成固定人口。

## H6 renderer

WideGroupedScene 是测速目录内的独立实现。它复制当前 GroupedScene 的完整 REMOVE/source overlay/ADD/composition 顺序，只把一次 horizon 上限由 3 放宽到 6。

不修改正式 tools/joint_occ_experiments/batching/geometry.py，因此不会使正在运行的 Shared-History checkpoint implementation fingerprint 失效。

## 运行

当前两轮训练可以继续跑，不要为了测速修改它的目录。训练完成或任意安全 checkpoint 都可作为只读源。

    conda activate OccFM
    cd /root/nas/occ/swfm
    git fetch origin
    git checkout feature/shared-history-execution-speed-v2
    PROBE_STEPS=8 PASSES=2 bash tools/shared_history_experiments/speed/run.sh

脚本默认自动寻找最新 outputs/p0_f9_shared_history/shared_20pct_*ep_*/last.pt。也可以显式指定：

    SOURCE=/root/nas/occ/swfm/outputs/p0_f9_shared_history/实际目录/last.pt \
    PROBE_STEPS=8 PASSES=2 \
    bash tools/shared_history_experiments/speed/run.sh

结果写入新的 outputs/p0_f9_shared_history_speed/full6_exec_*/，重点看 summary.txt、speed.json、progress.jsonl。

## Gate

每个候选都从同一 source snapshot 重新加载 model weights、Adam、RNG、当前 cosine update 和同一批真实训练 windows。

warmup 不计入速度。正式 timing 至少 AB/BA 两遍，并在每个 candidate 前恢复 source 状态。

检查 loss diagnostics、gradient links、final model+Adam numerical parity、peak allocated/reserved memory 和 each-pass wall time。

数值容差沿用仓库现有 execution probe 的 BF16 数值容差，不要求不同算术顺序 bitwise 相同。

recommended_backend 只会从 numerical parity 通过、current baseline replay 通过、两次 pass 均更快、GPU 至少保留 10% memory headroom 的候选中选择真实 seconds/window 最小者。

不会自动修改正式训练器或 checkpoint。

## 下一步

先取得这一轮真实 L40S/H100 timing。如果 packed network 已明显提速，但 soft_render_full_ce_lovasz + occupancy_backward 仍占主要时间，再单独实现 scene×horizon renderer batching，并用同一套 probe gate 比较。

如果完整六帧 exact execution 仍不足够快，才进入会改变训练轨迹的候选：2/6 horizon、periodic Lovasz 或 sparse CE。
