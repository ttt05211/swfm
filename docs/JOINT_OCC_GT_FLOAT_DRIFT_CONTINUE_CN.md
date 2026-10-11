# 显式接受小幅浮点差异，继续GT三轮

用户已明确授权采用sequential_h2：实测0.741306→0.504552s/窗，1.469237倍，首窗renderer概率/loss/梯度和单步通过；八步候选maxabs0.000817965、原repeat0.000402516。它是训练浮点执行变更，不是无损优化；最终精度尚未知。

不改/重新生成原speed.json，不把原pass_gate或容差改通过。独立接受策略要求相同source SHA、相同已测内核fingerprint、单步/梯度/速度/显存通过，仅允许已测多步状态maxabs≤0.001；此上限是用户接受的小幅漂移边界，不是新的等价容差或精度保证。

新增入口在horizons/continuation子目录，原顶层训练/两帧内核和旧fingerprint一个文件都不改，因此既有报告可复用。原GT权重/Adam/RNG/LR/游标/数据顺序/batch4/source128/三轮总周期原样恢复；不是从头或额外三轮。没有128步对照/新测速/新bank/旧缓存写入。

## 直接启动

现有GT训练应已安全停止。自动取最近一次窄版probe的speed.json；实际内容还会按source SHA/内核严格核验，错误报告会拒绝，不靠文件名判断通过。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export GT_RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt3_fast_20261011_123726
export HORIZON_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt3_horizon_float_resume_$(date +%Y%m%d_%H%M%S)
unset RESUME
(
set -euo pipefail
git fetch origin feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
export HORIZON_PROOF="$(find /root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego \
  -maxdepth 2 -type f -path '*/gt_horizon_probe_*/speed.json' \
  -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
test -n "$HORIZON_PROOF" && test -f "$HORIZON_PROOF"
echo "复用报告：$HORIZON_PROOF"
echo "新训练目录：$HORIZON_OUT"
bash tools/joint_occ_experiments/horizons/continuation/run.sh
)
```

首次导入必须新目录；新输出存horizon_import.json、float_drift_acceptance.json、last.pt、progress.jsonl、training.json、summary.txt。原断点与原report完全不写。
控制台应显示JOINT_OCC_RESUME completed_update=149（实际以源last.pt为准），LR/RNG/Adam intact；下一步150。
每64完整更新及epoch边界存last.pt，Ctrl-C/SIGTERM完成当前更新立即保存。OOM/半步失败不保存污染状态，不擅自缩batch重试；kill -9只恢复最后健康checkpoint。

## 暂停后恢复

必须HORIZON_OUT为同一新训练目录；原终端变量保留，新终端请填写实际打印路径。恢复不需报告或新接受检查，但仍显式承认该执行策略。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export HORIZON_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/这里替换为实际gt3_horizon_float_resume目录
export RESUME="$HORIZON_OUT/last.pt"
unset HORIZON_PROOF
bash tools/joint_occ_experiments/horizons/continuation/run.sh
```

完成原三轮后自动按原方案一次固定dev512，报告Transport、ADD-only、ADD+REMOVE及IoU/mIoU/Moving、ego误差，不选dev-best/阈值。继续使用原快速硬推理；本次训练吞吐不代表正式FPS。
最后发 `cat "$HORIZON_OUT/summary.txt"` 即可。17项针对性CPU恢复/数值/错误拒绝测试通过；没有重测CUDA或代跑服务器训练，不宣称最终精度一致。
