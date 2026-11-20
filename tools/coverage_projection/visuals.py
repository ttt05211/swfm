"""Bounded same-window BEV gallery; retain native volumes for COME 3D rendering."""
import argparse
import html
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw

COLORS = np.array([[132,144,156],[255,120,50],[255,192,203],[255,255,0],
    [0,150,245],[0,255,255],[255,127,0],[255,0,0],[255,240,150],[135,60,0],
    [160,32,240],[255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],
    [0,175,0],[255,255,255]], np.uint8)


def bev(volume):
    # Display policy follows COME, but scoring/projection KEEP class 0.
    occupied = (volume > 0) & (volume < 17)
    heights = np.where(occupied, np.arange(volume.shape[-1]), -1).max(-1)
    labels = np.take_along_axis(volume, heights.clip(0)[..., None], -1)[..., 0]
    labels = np.where(heights >= 0, labels, 17)
    return Image.fromarray(COLORS[labels][::-1].copy()).resize((320, 320), Image.Resampling.NEAREST)


def render(root):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    links = []
    for kind in ('occ', 'stc'):
        for case in manifest['cases']:
            prefix = f'{kind}_{case["index"]:02d}'
            with np.load(root/(prefix+'_control.npz'), allow_pickle=False) as z: old = {k: z[k] for k in z.files}
            with np.load(root/(prefix+'_candidate.npz'), allow_pickle=False) as z: new = {k: z[k] for k in z.files}
            if not np.array_equal(old['gt'], new['gt']): raise ValueError('visual targets differ')
            canvas = Image.new('RGB', (1400, 1060), 'white'); draw = ImageDraw.Draw(canvas)
            rows = [('GT', [old['t0'], *old['gt']]), ('Original', [old['t0'], *old['prediction']]),
                    ('Coverage+projection', [new['t0'], *new['prediction']])]
            for col, title in enumerate(('t0 input', '1 s', '2 s', '3 s')):
                draw.text((116+col*320, 24), title, fill='black')
            for r, (label, volumes) in enumerate(rows):
                draw.text((4, 76+r*330), label, fill='black')
                for col, volume in enumerate(volumes): canvas.paste(bev(volume), (116+col*320, 56+r*330))
            filename = prefix+'_bev.png'; canvas.save(root/filename)
            links.append(f'<h3>{kind} {case["index"]:02d} {html.escape(case["scene"])}</h3><img width="1200" src="{filename}">')
    (root/'index.html').write_text('<meta charset="utf-8"><title>Fixed same-window comparison</title>'
        '<p>OccWorld/COME palette. BEV preview; native 3D volumes included. All fixed first8, no ranking.</p>'
        + '\n'.join(links), encoding='utf-8')
    archive = root.parent/'visuals_and_metrics.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):
            if p.is_file(): z.write(p, 'visuals/'+p.name)
        for name in ('evaluation.json', 'summary.txt', 'contract.json'):
            if (root.parent/name).is_file(): z.write(root.parent/name, name)
    print(f'VISUALS {root / "index.html"}; download DATA/results only: {archive} ({archive.stat().st_size/2**20:.1f} MiB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True, type=Path)
    render(p.parse_args().root)
