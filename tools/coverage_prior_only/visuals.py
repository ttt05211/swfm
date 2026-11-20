"""Same eight windows: GT / original / projection ON / projection OFF."""
import argparse
import html
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from tools.coverage_projection.visuals import bev


def render(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    links=[]
    for kind in ('occ','stc'):
        for case in manifest['cases']:
            prefix=f'{kind}_{case["index"]:02d}';values={}
            for role in ('control','joint','candidate'):
                with np.load(root/f'{prefix}_{role}.npz',allow_pickle=False) as f:values[role]={k:f[k] for k in f.files}
            old,joint,new=(values[k] for k in ('control','joint','candidate'))
            for value in (joint,new):
                if any(not np.array_equal(old[k],value[k]) for k in ('gt','t0','future_poses')):
                    raise ValueError('paired visual input/GT/poses differ')
            rows=[('GT',[old['t0'],*old['gt']]),('Original',[old['t0'],*old['prediction']]),
                ('Projection ON',[joint['t0'],*joint['prediction']]),('Projection OFF',[new['t0'],*new['prediction']])]
            image=Image.new('RGB',(1400,1390),'white');draw=ImageDraw.Draw(image)
            for col,title in enumerate(('t0 input','1 s','2 s','3 s')):draw.text((116+col*320,24),title,fill='black')
            for r,(label,volumes) in enumerate(rows):
                draw.text((4,76+r*330),label,fill='black')
                for col,volume in enumerate(volumes):image.paste(bev(volume),(116+col*320,56+r*330))
            name=prefix+'_bev.png';image.save(root/name)
            links.append(f'<h3>{kind} {case["index"]:02d} {html.escape(case["scene"])}</h3><img width="1200" src="{name}">')
    (root/'index.html').write_text('<meta charset="utf-8"><title>Coverage / projection ablation</title>'
        '<p>Fixed same8 windows, OccWorld/COME palette. BEV preview; native 3D arrays included. '
        'Rows: GT, Original, Projection ON, Projection OFF.</p>'+'\n'.join(links),encoding='utf-8')
    target=root.parent/'visuals_and_metrics.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.iterdir()):
            if path.is_file():archive.write(path,'visuals/'+path.name)
        for name in ('contract.json','evaluation.json','summary.txt'):
            if (root.parent/name).is_file():archive.write(root.parent/name,name)
    print(f'PRIOR_ONLY_VISUALS {root/"index.html"}; download {target} ({target.stat().st_size/2**20:.1f} MiB)',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);render(p.parse_args().root)
