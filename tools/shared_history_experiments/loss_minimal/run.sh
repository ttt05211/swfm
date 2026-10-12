#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export EPOCHS=1 REFINER_MODULE=tools.shared_history_experiments.loss_minimal.run
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/loss500_minimal_$(date +%Y%m%d_%H%M%S)}"
echo '仅新增CE+Lovasz+ego 500步单路；旧full/no_motion/no_existence不重跑。不启动完整训练。'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" "$@"
