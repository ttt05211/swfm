"""CPU weight strict-load check, deliberately WITHOUT CUDA availability checks."""
import argparse
from pathlib import Path
from .model import load_model
from .common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--gast-root',type=Path,required=True);a=p.parse_args()
    model,evidence=load_model(a.gast_root,a.work/'downloads/gast_gt.pth',a.work,'cpu')
    write_json(a.work/'cpu_check.json',dict(status='complete',device='cpu',strict_weights=True,official=evidence))
    print('GAST_CPU_CHECK_OK',evidence['parameters'],'parameters; full official state_dict loaded; NO GPU requested',flush=True)


if __name__=='__main__':main()
