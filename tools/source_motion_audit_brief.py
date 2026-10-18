"""Read saved motion-audit results only; no Torch, GPU or new inference.

Kept OUTSIDE source_motion_audit so adding this reader does not change that
package's implementation fingerprint or prevent an existing audit resuming.
"""
import argparse
import json
import math
import os
from pathlib import Path

MODES = ('learned', 'KTA_future', 'XY_only', 'yaw_only')
READOUTS = ('transport', 'semantic_refine')


def number(value, *, signed=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value): return 'NA'
    return format(value, '+.4f' if signed else '.4f')


def delta(a, b):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)): return 'NA'
    return number(a-b, signed=True)


def triple(metric):
    return '/'.join(number(metric.get(k)) for k in ('mIoU', 'IoU', 'MovingMicro'))


def mean(stats, name): return number(stats.get(name, {}).get('mean'))


def brief(result, path):
    lines = [f'结果: {path}', f'状态: {result.get("status", "unknown")} | 窗口: {result.get("windows", 0)}/64']
    if result.get('status') != 'complete': lines.append('尚未完成：下面仅为已保存窗口，不是最终64窗结果。')
    for kind, title in (('occ_gt', 'OCC+GT'), ('stc_gt', 'STC+GT')):
        routes = result.get('metrics', {}).get(kind, {})
        if not routes:
            lines.append(f'{title}: 暂无已保存指标'); continue
        lines += [f'\n{title}                 Transport                    Refiner',
                  '路线                    mIoU/IoU/Moving                 mIoU/IoU/Moving']
        for mode in MODES:
            lines.append(f'{mode:12s}  {triple(routes.get(mode+"/transport", {})):>27s}  {triple(routes.get(mode+"/semantic_refine", {})):>27s}')
        values = []
        for readout in READOUTS:
            a, b = routes.get('learned/'+readout, {}), routes.get('KTA_future/'+readout, {})
            values.append('/'.join(delta(a.get(k), b.get(k)) for k in ('mIoU', 'IoU', 'MovingMicro')))
        lines.append('学习−KTA      '+ ' | '.join(values))
        motion = result.get('source_motion', {}).get(kind, {})
        all_sources = motion.get('all', {})
        history = result.get('history_yaw_abs_deg', {}).get(kind, {})
        near_zero = all_sources.get('XY_near_zero_fraction_1cm')
        zero = number(100*near_zero) if isinstance(near_zero, (int, float)) else 'NA'
        lines.append(f'修正均值: XY={mean(all_sources, "XY_correction_m")}m; 未来yaw={mean(all_sources, "future_yaw_abs_deg")}°; '
                     f'历史yaw={number(history.get("mean"))}°; XY<1cm={zero}%')
        for key, label in (('moving', '运动'), ('slow_or_stationary', '低速/静止')):
            s = motion.get('groups', {}).get(key, {})
            errors = s.get('learned_error_m', {})
            lines.append(f'{label}: 可归属误差样本n={errors.get("n", 0)}; XY误差 学习/KTA={mean(s, "learned_error_m")}/{mean(s, "KTA_error_m")}m; '
                         f'改善/损害={s.get("XY_helped", 0)}/{s.get("XY_harmed", 0)}')
        unresolved = motion.get('groups', {}).get('unresolved', {}).get('samples', 0)
        repeat = result.get('reference_repeat', {}).get(kind, {})
        drift = [f'{r}:{repeat.get(r, {}).get("failed_windows", "NA")}窗/{repeat.get(r, {}).get("max_changed_voxels", "NA")}体素'
                 for r in READOUTS]
        lines.append(f'框运动未归属样本={unresolved}; 原路径重复漂移(失败窗/单窗最大差异): '+ ' | '.join(drift))
    lines += ['\nKTA仅关闭未来XY/yaw修正，历史yaw与Q不变；框误差是条件统计，n为source×时距而非独立物体。',
              '只读取已有JSON，不重跑推理、不改变结果；不自动判优或部署。']
    return '\n'.join(lines)


def find_result(argument=None):
    explicit = argument or os.environ.get('MOTION_AUDIT_DIR')
    if explicit:
        path = Path(explicit).expanduser()
        return path/'results.json' if path.is_dir() else path
    root = Path(__file__).resolve().parents[1]/'outputs/p0_f9_shared_history'
    files = list(root.glob('stage2_motion_audit_*/results.json'))
    if not files: raise FileNotFoundError('未找到诊断结果，请显式传入诊断目录或results.json路径。')
    # Latest saved REPORT, not the highest-scoring checkpoint or route.
    return max(files, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', nargs='?', help='诊断目录/results.json；省略时用MOTION_AUDIT_DIR，否则最新保存的标准诊断报告')
    a = ap.parse_args(argv)
    try:
        path = find_result(a.path).resolve()
        result = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(result, dict) or not isinstance(result.get('metrics'), dict):
            raise ValueError('不是运动诊断 results.json（需要metrics字段），请勿传state.json或日志。')
        print(brief(result, path))
    except (OSError, ValueError) as exc:
        ap.exit(2, str(exc)+'\n')


if __name__ == '__main__': main()
