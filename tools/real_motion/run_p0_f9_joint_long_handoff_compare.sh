#!/usr/bin/env bash
# One matched-population pass; all inference routes fixed BEFORE dev scoring.
set -euo pipefail
export LONG_HANDOFF_MODES=redetect,reconciled,transport_history
export LONG_COMPARE_E14=1
echo '冻结epoch19，一趟比较：原递推/预测身份轨迹接续/纯搬运历史反馈/同人口原生六历史E14。'
echo '共享第一段Joint及GT计数；不训练、不调阈值、不自动挑选路由。默认dev64。'
bash /root/nas/occ/swfm/tools/real_motion/run_p0_f9_joint_long_rollout.sh "$@"
