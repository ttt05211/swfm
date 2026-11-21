"""CPU-only compact status for terminal reconnects."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    print('GAST_WORK='+str(a.work.resolve()))
    for name in ('prepared.json','cpu_check.json','gpu_check.json','progress.json'):
        path=a.work/name
        if not path.exists():continue
        r=json.loads(path.read_text());print(name,
            {k:r[k] for k in ('status','windows','total','device','strict_weights') if k in r},
            'cases='+str(len(r['cases'])) if isinstance(r.get('cases'),dict) else '')
        if 'updated_utc' in r:print('progress_age_seconds=',round(time.time()-r['updated_utc'],1))
    for name in ('comparison_assets.zip','rendered/index.html','rendered/render_check.png'):
        path=a.work/name
        if path.exists():print(name,round(path.stat().st_size/2**20,2),'MiB')


if __name__=='__main__':main()
