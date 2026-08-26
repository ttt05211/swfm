# SWFM：Real-Motion Sparse Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的研究实现。核心原则：**transport what is physically deterministic; generate only what truly moves**。

本仓库把官方 OccFM 固定为 `upstream_occfm` submodule，不直接魔改上游源码。我们的实现集中在 `real_motion/` 与 `tools/real_motion/`。

## 0. 当前状态

这版已经把代码链路补到 **P0-ready + tiny-training-ready（前提是服务器 integration tests 通过）**：

```text
nuScenes / Occ3D labels
  -> ego compensation
  -> causal real-motion split
  -> SE(3) static transport
  -> causal occupancy KTA
  -> horizon-dependent motion tube
  -> frozen OccFM VAE cache (sharded)
  -> motion-window sparse CFM
  -> frozen decoder
  -> Static-Protected Motion Composition
  -> Moving-mIoU v2 / P0 / profiler
```

**仍然不能在没有服务器验证的情况下声称“训练已跑通”**。第一次上服务器必须先跑第 4 节的 integration/equivalence checks，再严格按 P0-A → B → C → D → E/F 的顺序走。

---

## 1. 安装

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git submodule update --init --recursive
```

环境优先沿用官方 OccFM。最低附加依赖：

```bash
pip install nuscenes_devkit numpy torch einops einops_exts easydict pyyaml pytest
```

检查 pinned upstream：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python -c "from forecast.models.worldmodels.occfm import OccFM; print('OccFM OK')"
```

官方 commit：

```text
64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

---

## 2. 方法 contract（不要边跑边改）

### 2.1 Real-motion 三种 occupied 状态 + free

历史 occupancy 先统一到参考 ego frame，再做 causal motion decomposition：

- `confident_static`
- `moving`
- `uncertain`
- `free`

**关键修复：free 绝不等于 confident-static。**

```python
confident_static = occupied & (state == STATIC)
```

`real_motion/motion.py::decompose_masks()` 已直接返回安全的显式 mask；cache builder 不允许再自行用 `state == STATIC` 推导保护区域。

同时，最终默认不是只看逐 voxel persistence。对当前帧每个同语义 8-connected component，会在 ego-compensated history 中做 causal backward association，并用历史质心速度做 hysteresis：

- `speed >= 0.5 m/s`：整个 component 记为 moving；
- `speed <= 0.2 m/s` 且 persistence 足够高：整个 component 记为 confident-static；
- 其余：uncertain。

这样避免一辆平移汽车因为连续帧仍有重叠体素而被错误拆成“静态内部 + 运动边缘”。新出现/跟踪不完整的 occupancy 默认进入 uncertain，而不是 hard-freeze。阈值属于 causal detector 配置，正式实验前用 train/calibration 固定。

### 2.2 Static branch

`confident_static` 只经过 benchmark 允许的 future ego SE(3) transport：

```text
real_motion/geometry.py
```

默认 OccFM/Occ3D grid：

```text
range: [-40,-40,-1, 40,40,5.4]
voxel: [0.4,0.4,0.4]
shape: [200,200,16]
```

### 2.3 KTA

`real_motion/kta.py` 提供一个 dependency-free **reference** causal occupancy KTA baseline：

1. ego compensated history；
2. 当前 moving/uncertain BEV component；
3. 与上一帧同语义 component 做最近邻匹配；
4. 估计 constant planar velocity；
5. 外推未来 occupancy。

它只作为 **motion prior + support builder**，不是最终预测。

它的用途是让仓库从 raw nuScenes 到 P0/cache/inference 可以独立闭环，并不是要替代你已经验证过的 KTA 实现。**正式论文跑分优先接入已验证的 KTA**；只要保持“仅历史输入 → future semantic prior + BEV support”的 causal contract，后端无需改动。

### 2.4 Motion tube

```python
M_gen[h] = Dilate(KTA_support[h], r_h)
```

默认六个 0.5s horizon 的 radius：

```text
[1,2,3,4,5,6]
```

正式值必须由 P0-B coverage–active-ratio 曲线确定并冻结。

---

## 3. 为什么 v1 用 motion-window sparse，而不是任意 packed tokens

官方 OccFM transition 是：

```text
Conv3d + temporal attention + U-Net down/up + DiT blocks
```

它不是纯 token Transformer。为了最大复用官方 checkpoint，v1 在原 `50×50` latent grid 上选少量固定 window（默认 `20×20`），每个 window 跑原结构。

因此论文措辞应该是：

> **motion-guided sparse windows / motion-block sparse world model**

不要写成“只处理任意 active tokens”。

### Window planner 最终 contract

- `generation_support`：**required future support**，唯一可以创建 window 的信号，同时也是 future flow-loss mask；
- `planning_support`：historical moving + future KTA tube 等 context，只用于候选 window 的 tie-break，不能创建 history-only window。

原因：v1 不同 window 之间没有跨窗口 attention，单独的 history-only window 对另一个 future window没有帮助。

`WindowPlanner` + crop/scatter 已改成 GPU-safe vectorized gather/scatter。Planner 对 50×50 mask 用 integral image **穷举全部合法 top-left**：先最大化仍未覆盖的 future required cells，再用 historical/KTA context 做 tie-break；规划只在 CPU 上做一次，避免 Python `.item()` 对 CUDA 反复同步。

---

## 4. 正式 P0 前先跑两个实现验收

### 4.1 单元测试

```bash
PYTHONPATH=$PWD pytest -q tests
```

本地当前：

```text
31 passed, 1 skipped
```

skip 的是需要官方 checkpoint + GPU 的 transition equivalence integration test。

### 4.2 最关键：Official transition equivalence

这项比普通 unit test 更重要。用：

```text
window = 50×50
prior = 0
origin = (0,0)
same input / timestep / trajectory / checkpoint
```

比较官方 transition 和修改版：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/check_transition_equivalence.py \
  --ckpt /path/to/official_occfm.ckpt \
  --device cuda \
  --output outputs/transition_equivalence.json
```

默认要求：

```text
RMS <= 2e-5
max_abs <= 2e-4
```

**失败时不要训练。** 先定位 checkpoint/position/AdaLN/rearrange 是否与官方不等价。

也可以：

```bash
OCCFM_TRANSITION_CKPT=/path/to/official_occfm.ckpt \
PYTHONPATH=$PWD/upstream_occfm:$PWD pytest -q -m integration
```

---

## 5. 第一步：生成 raw prepared shards

这一步把前半段真正闭环：

```text
raw labels
 -> ego compensation
 -> real-motion masks
 -> static SE3
 -> KTA
 -> causal motion tube
 -> GT target/metric branch（仅训练/评估）
```

命令：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/prepare_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_train_temporal_v3_scene.pkl \
  --output data/prepared_train \
  --shard-size 16
```

先 smoke：

```bash
... --max-windows 16
```

raw prepared cache 本身也是 sharded + lazy load，避免一个几十 GB 的 `.pt`。

### 因果边界

以下只用历史：

- real-motion split
- KTA
- generation support

future GT 只进入训练 target / 评估分支：

- `future_dynamic_target_occ`：**仅保留 causal `generation_support` 内的动态语义 GT**，用于 WM 训练；
- `future_moving_occ`：满足 Moving-mIoU v2 0.5 m/s instance contract 的 metric-only GT；
- `gt_moving_support`：dual-box metric support；
- Moving-mIoU / P0 audit。

特别注意：**WM 训练 target 不由未来 GT 的“是否移动”来定义。** `M_gen` 始终来自历史 + KTA；future GT 只告诉模型在这个 causal support 内应该生成什么动态语义。这样不会把一个处于 uncertain/KTA support 内、但未来恰好停着的车辆错误训练成 empty。

`prepare_nuscenes_window(..., include_gt=False)` 是在线 inference/profiler 使用的纯 causal 路径。

还有一个容易混淆的坐标细节：**ego compensation 的副本只用于历史运动判定/KTA**；送入 frozen OccFM VAE 的每一帧 history branch 仍保持该帧自己的 native ego occupancy，维持官方 OccFM 预训练的数据接口，不把 6 帧先强行 warp 到 t0 再编码。

---

## 6. 正式训练前 P0

### P0-A（第一个做）：Frozen Causal Real-Motion Decomposition

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/p0_true_motion_decomposition.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/occfm_vae.ckpt \
  --wm-ckpt /path/to/occfm.ckpt \
  --output outputs/p0_a.json \
  --max-windows 128
```

同一个 frozen OccFM 分别预测：

```text
Full
Causal-static-only
Moving/uncertain-only
```

输出重点：

- causal-static-only − full 的 true-static mIoU；
- moving-only − full 的 Moving-mIoU v2。

这就是之前 semantic split 实验的 real-motion 版本。**这一项失败，不进入 sparse moving-only training。**

### P0-B：Sparsity + KTA tube coverage

```bash
PYTHONPATH=$PWD \
python tools/real_motion/p0_support_stats.py \
  --prepared data/prepared_val \
  --radii 0,1,2,3,4,5,6 \
  --schedule 1,2,3,4,5,6 \
  --latent-extra-radius 1 \
  --output outputs/p0_b.json
```

它按每个 0.5s horizon 分别报告：

- BEV future-moving **arrival occupancy** coverage（不是 dual-box old+new metric support）；
- BEV active ratio；
- latent arrival coverage；
- latent active ratio；
- true-moving voxel / occupied ratio；
- true-moving latent / dense latent ratio；
- proposed window 的 future coverage / window 数 / slot compute ratio；
- **future windows with history ratio** 与 **future required cells covered by history-connected windows ratio**，专门检查“future window 虽然覆盖 target，但因为不同 window 不通信而看不到历史运动”的风险。

不要再只看所有 horizon 混在一起的一个平均数。

### P0-C：Causal blind spots

```bash
PYTHONPATH=$PWD \
python tools/real_motion/p0_causal_audit.py \
  --prepared data/prepared_val \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --output outputs/p0_c.json
```

分别统计 1s/2s/3s：

- historical stationary → future moving；
- t0 前没有可用于 causal motion estimation 的实例比例；
- metric endpoint birth/death exclusion 数量。

### P0-D：Oracle headroom

```bash
PYTHONPATH=$PWD \
python tools/real_motion/p0_oracle.py \
  --prepared data/prepared_val \
  --output outputs/p0_d.json
```

同时给两个 Oracle：

1. **Decomposition Oracle**：SE3 static + GT moving，受 confident-static protection；
2. **Causal-Support Oracle**：再额外受 causal KTA tube 限制。

二者差值可以直接分离：

```text
decomposition headroom
vs
support-coverage headroom
```

### P0-E：Frozen VAE sanity

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/p0_vae_sanity.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/occfm_vae.ckpt \
  --output outputs/p0_e.json \
  --max-windows 128
```

比较：

- true-moving full latent reconstruction；
- causal generation-support + `E(empty)` canvas；
- GT-support canvas（诊断上限）。

---

## 7. Frozen Moving-mIoU v2

冻结 contract：

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
report horizons: 1s / 2s / 3s
```

实现已经把两个坐标 contract 显式分开：

1. **world XY** 的 t0/th GT center 用于判断 Moving；
2. t0/th GT boxes 再转换到 **target future ego grid** rasterize。

```text
Support_h = GT Box(t0 -> h) UNION GT Box(th -> h)
```

只统计冻结的动态 semantic IDs：

```text
bicycle(2), bus(3), car(4), construction_vehicle(5),
motorcycle(6), pedestrian(7), trailer(9), truck(10)
```

free/static 不进入 Moving-mIoU。

**1s、2s、3s 必须先分别算 mIoU，再做算术平均。** `MovingMIoUV2MultiHorizon` 强制这个协议。

Future GT 用于训练监督和 post-hoc evaluator，但**不用于 real-motion detector、KTA、generation support 或在线 inference 决策**。

---

## 8. 第二步：生成 frozen VAE latent cache

P0 通过后才生成训练 cache：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/build_latent_cache.py \
  --prepared data/prepared_train \
  --vae-ckpt /path/to/occfm_vae.ckpt \
  --output data/latent_train \
  --empty-latent data/empty_latent.pt \
  --mode sample \
  --shard-size 256
```

每个 sample：

```python
moving_history_latent
future_dynamic_target_latent  # 编码 future_dynamic_target_occ；与 metric-only Moving GT 明确分离
static_future_latent
kta_future_latent
generation_support      # latent future mask
planning_support        # context only
trajectory
```

VAE 训练时完全不重复执行。Sharded cache 的训练 sampler 会“先 shuffle shard、再 shuffle shard 内样本”，避免全局随机索引导致一个大 shard 几乎每个 sample 都重新 `torch.load`。

### 为什么 cache 不再是一个大 `.pt`

`real_motion_v3` 使用：

```text
latent_train/
  index.json
  shard_00000.pt
  shard_00001.pt
  ...
```

Dataset 一次只 lazy-load 一个 shard。

### VAE 是 stochastic 的

官方 `VaeQuant` 是：

```text
z = mu + sigma * eps
```

adapter 支持：

- `mode=sample`：固定 seed，可复现并保持官方 sampled-latent 分布；
- `mode=mean`：诊断用 deterministic mean。

P0 paired branch 会用同一个 epsilon seed，避免“换 branch 顺便换了一份 VAE noise”。训练 cache 默认 `mode=sample`；正式在线 inference 会从 sparse checkpoint 的 `cache_metadata` 自动恢复同一个 latent mode，禁止训练用 sampled latent、推理偷偷改成 mean。

`empty_latent` 必须是真实 `E(empty occupancy)`，禁止数值 0 代替。训练时使用的 exact `empty_latent` 会同时写进 sparse checkpoint，在线 inference 直接复用这一个 tensor，避免 stochastic VAE 的 empty representation 再次漂移。

Cache metadata 还记录 VAE checkpoint 的 SHA256、latent mode 和 latent support radius。正式 inference/profiler 会自动读取训练 contract：window size 不一致直接报错，VAE fingerprint 不一致直接报错，support radius 若想改必须显式 `--allow-support-override`，避免“训练/测试偷偷换 representation”。

---

## 8.5 Static/KTA prior 的数值尺度

官方 OccFM 在 transition 前对 latent 统一乘 `RESCALE_FACTOR=10`。Static/KTA prior 来自同一个 frozen VAE，因此进入 zero-init prior adapter 前也使用相同缩放：

```text
future static/KTA latent
        × 10
        ↓
zero-init prior projection
        ↓
token-wise AdaLN condition
```

History 时段的 prior 显式补零；只接受 `future-only F` 或已经对齐的 `H+F` 两种长度，不再通过模糊的时间长度启发式猜测。训练与 sampling 完全一致。

## 9. Sparse CFM correctness 修复

### 9.1 support 外不再是未监督随机 latent

以前一个 20×20 window 只有 support 内有 loss，但整个 window 都生成/scatter，会让无监督 margin 产生 dynamic false positive。

现在是 **masked latent inpainting**：

```text
active support     -> CFM noise / ODE update
outside support    -> 每一步都 clamp 到 E(empty)
```

因此：

- support 外不会用随机状态影响 active token；
- scatter 前再做一次 safety clamp；
- decoder spillover 最终还会被 occupancy-space `write_support` 限制。

### 9.2 overlapping windows 使用同一 global noise canvas

先生成：

```text
[B,F,C,50,50] global z0
```

然后 crop 给各 window。重叠 cell 初始噪声严格一致，不再是两条独立 stochastic trajectory 最后生硬平均。

### 9.3 support 不是“整块 overwrite mask”

最终 composition 有三种权限：

- `generation_support`：决定哪里花 WM 计算；
- `write_support`：只允许**动态预测**在哪些位置写入，不会整块覆盖；
- `confident_static`：绝对保护。

```python
writable = dynamic_prediction & write_support & ~confident_static
```

这仍然满足：**support != wholesale overwrite mask**。

---

## 10. Tiny-set training

只有下面全部通过后才开始：

```text
transition equivalence PASS
P0-A acceptable
P0-B sparse/coverage tradeoff acceptable
P0-C blind spots acceptable
P0-D headroom sufficient
P0-E VAE representation acceptable
```

先 64/128 windows latent shard：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/train_sparse.py \
  --cache data/latent_tiny \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output logs/swfm/tiny.pt \
  --window 20 --max-windows 8 \
  --batch-size 2 --steps 2000 --lr 2e-5 --amp
```

trainer 默认要求：

```text
future window coverage >= 95%
```

否则 hard fail。只有诊断时才能 `--allow-low-coverage`。

第一版 loss 只有 masked flow MSE，不加 Router / ABE / repair / preservation / confidence loss。

---

## 11. Sampling 与 end-to-end inference

从 cache 采样 latent：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/sample_sparse_latent.py \
  --cache data/latent_val \
  --ckpt logs/swfm/model.pt \
  --output outputs/moving_latent.pt
```

新 checkpoint 已内置训练时的 exact `empty_latent`；只有兼容旧 checkpoint 时才需要额外传 `--empty-latent`。

真正 raw nuScenes 在线路径：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/infer_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --vae-ckpt /path/to/occfm_vae.ckpt \
  --sparse-ckpt logs/swfm/model.pt \
  --output outputs/predictions \
  --max-samples 100
```

在线 path 调用 `include_gt=False`，不触碰 future semantic GT / future instance annotations。`--vae-mode auto`（默认）会强制与训练 checkpoint 的 latent mode 一致。

评估保存的预测：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/evaluate_predictions.py \
  --prepared data/prepared_val \
  --pred-dir outputs/predictions \
  --output outputs/eval.json
```

会同时给出 SWFM overall / Dynamic / Moving-mIoU v2、同 composition 规则下的 KTA baseline，以及 Moving support 内 voxel harm/repair。

---

## 12. 效率：必须测 end-to-end，不拿 backend microbenchmark 冒充 FPS

完整 profiler：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/profile_pipeline.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --vae-ckpt /path/to/occfm_vae.ckpt \
  --sparse-ckpt logs/swfm/model.pt \
  --repeats 20 \
  --output outputs/profile.json
```

nuScenes label 文件读取、pickle/表查询会在计时前预加载，避免把磁盘速度伪装成模型速度。它分别计时真正在线方法部分：

- real-motion + SE3 + KTA preprocessing；
- online condition VAE encoding；
- support planning + vectorized crop；
- NFE sparse WM；
- vectorized scatter；
- frozen decoder；
- final composition；
- total latency / FPS。

论文至少报告：

```text
mIoU
Moving-mIoU v2
active window / latent ratio
GFLOPs
end-to-end latency / FPS
peak memory（可选）
```

---

## 13. 训练后 Harm / Repair，不提前塞 gate

`real_motion/metrics/diagnostics.py` 支持：

- voxel-level harm/repair；
- instance/tube-level macro harm/repair；
- oracle KTA-vs-WM selector。

只有 oracle selector headroom 明显时才研究 gate：

```text
single-pass lightweight gate > multi-sampling > CFG
```

CFG 最后考虑，因为它直接伤害效率。

---

## 14. 困难机动 / KTA-hard 分层

主 Moving-mIoU v2 不变，只做 subset analysis。

工具：

```text
real_motion/metrics/stratified.py
tools/real_motion/analyze_motion_subsets.py
```

支持：

- Uniform / Easy
- Accel / Decel
- Turning
- Turning + Speed Change
- KTA-Easy / Medium / Hard

评估时可直接导出 per-instance records：

```bash
PYTHONPATH=$PWD python tools/real_motion/evaluate_predictions.py \
  --prepared data/prepared_val \
  --pred-dir outputs/predictions \
  --output outputs/eval.json \
  --subset-records outputs/moving_records.jsonl

# calibration：只在 calibration side 拟合 KTA error 分位点
PYTHONPATH=$PWD python tools/real_motion/analyze_motion_subsets.py \
  --records outputs/moving_records_calib.jsonl \
  --fit-kta-cuts \
  --output outputs/subsets_calib.json

# test：把上一步输出的两个 cut 原样填入
PYTHONPATH=$PWD python tools/real_motion/analyze_motion_subsets.py \
  --records outputs/moving_records_test.jsonl \
  --kta-cuts 0.8,1.7 \
  --output outputs/subsets_test.json
```

records 包含 interval speed、历史 speed、speed change、**turn rate (rad/s)**、KTA center error 和实例 support IoU。工具强制 calibration 时 `--fit-kta-cuts`、test 时显式 `--kta-cuts c1,c2` 二选一，避免在 test 上重新拟合难度分层。speed-change / turn-rate 阈值同样应在 train/calibration 冻结后原样搬到 test。

---

## 15. 推荐执行顺序

```text
0. pytest
1. transition equivalence
2. prepare smoke (16 windows)
3. P0-A  ← 第一个真正的方法实验
4. P0-B（coverage 用 future moving arrival occupancy，不用 dual-box old+new metric mask）
5. P0-C
6. P0-D
7. P0-E + end-to-end profiler smoke
8. build tiny latent cache
9. 64/128 tiny overfit
10. small held-out
11. full latent cache + full training
12. harm/repair + maneuver/KTA-hard analysis
13. official evaluation + final efficiency report
```

任何一步失败，都先定位它属于：

```text
motion/support
geometry/KTA
VAE representation
WM optimization
composition/evaluation
```

不要直接堆 loss 或 Router。

---

## 16. 关键代码索引

```text
real_motion/motion.py                  causal real-motion masks
real_motion/geometry.py                ego compensation + SE3 warp
real_motion/kta.py                     causal occupancy KTA
real_motion/support.py                 horizon tube / latent support
real_motion/nuscenes_adapter.py        nuScenes GT/pose/metric adapters
real_motion/prepared.py                raw end-to-end preparation
real_motion/occfm_io.py                pinned official VAE/WM adapter
real_motion/cache.py                   sharded latent cache
real_motion/windows.py                 sparse window planner + vectorized IO
real_motion/models/cfm.py              masked CFM inpainting
tools/real_motion/evaluate_predictions.py final overall/Dynamic/Moving evaluation
real_motion/models/transition.py       OccFM-compatible prior-conditioned transition
real_motion/composition.py             static-protected composition
real_motion/metrics/moving_miou_v2.py  frozen Moving-mIoU v2
real_motion/metrics/diagnostics.py      harm/repair + oracle selector
```

---

## 17. 不提交 GitHub 的资产

```text
nuScenes / Occ3D data
official VAE checkpoint
official OccFM checkpoint
prepared raw shards
latent shards
trained sparse checkpoints
predictions
```

---

## 18. 上游

基于 `Orbis36/OccFM-NeurIPS2025`：*Towards foundational LiDAR world models with efficient latent flow matching*, NeurIPS 2025。
