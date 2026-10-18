"""Compact saved old/new motion-gain comparison; standard library only."""
import argparse
import json
from pathlib import Path
from tools.source_motion_audit_brief import brief, number, delta


def comparison(old, stage2):
    if old.get('status') != 'complete' or old.get('windows') != 64:
        return '旧版诊断尚未完成，不与完整第二阶段作人口不匹配的比较。\n'
    if old['contract']['keys'] != stage2['contract']['keys']:
        raise ValueError('comparison keys/order differ')
    lines = ['===== SAME64 / OLD BASELINE vs STAGE2 =====',
             'input route              metric   OLD_KTA OLD_learned OLD_gain | NEW_KTA NEW_learned NEW_gain']
    for kind in ('occ_gt', 'stc_gt'):
        for route in ('transport', 'semantic_refine'):
            for metric in ('mIoU', 'IoU', 'MovingMicro'):
                a, b = old['metrics'][kind], stage2['metrics'][kind]
                ok, ol = a['KTA_future/'+route][metric], a['learned/'+route][metric]
                nk, nl = b['KTA_future/'+route][metric], b['learned/'+route][metric]
                lines.append(f'{kind:6s} {route:15s} {metric:11s} {number(ok):>8s} {number(ol):>10s} {delta(ol, ok):>8s} | '
                             f'{number(nk):>8s} {number(nl):>10s} {delta(nl, nk):>8s}')
    lines += ['Gain=同一模型learned−KTA；两版KTA的Transport分数也列出供检查底稿是否一致。',
              '旧版Q/ICP固定，新版Q/历史yaw固定；这是各自运动贡献与联合架构差异，不是KTA Query单变量消融。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('path'); a = p.parse_args(argv)
    path = Path(a.path); path = path/'results.json' if path.is_dir() else path
    result = json.loads(path.read_text(encoding='utf-8'))
    print(comparison(result, result['stage2_reference']))


if __name__ == '__main__': main()
