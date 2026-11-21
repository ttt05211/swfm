"""CPU/offscreen: existing COME palette/camera/renderer for BOTH methods."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .common import FORMAT,sha,read_json,write_json


def rows_and_settings(key,arrays,poses):
    from tools.paper_assets.qualitative_render import fixed_reference_settings
    if key.startswith('short'):
        t0=arrays['t0']
        rows=[('GT',[t0,*arrays['gt']]),('GAST',[t0,*arrays['g']]),('CarryWorld',[t0,*arrays['e']])]
        return rows,['t0','1 s','2 s','3 s'],{}
    rows=[(label,list(arrays[name])) for label,name in (('GT','gt'),('GAST','g'),('CarryWorld','e'))]
    settings={}
    for _,panels in rows:settings.update(fixed_reference_settings(panels,poses))
    return rows,[f'{i} s' for i in range(1,7)],settings


def render(assets,out,*,camera='come_reference',cases=None,preview=True,renderer=None):
    from tools.paper_assets.qualitative_render import render_grid
    from tools.paper_assets.qualitative_gallery import contact_sheets
    renderer=renderer or render_grid;assets=Path(assets);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=dict(assets_sha256=sha(assets),camera=camera,preview=preview,panel_size=[640,360],
        renderer='existing paper_assets render_grid + vtk_render; no geometry/forecast edits')
    helpers=Path(__file__).resolve().parents[1]/'paper_assets'
    receipt['implementation']={p.name:sha(p) for p in [Path(__file__),*(helpers/n for n in
        ('qualitative_render.py','qualitative_gallery.py','vtk_render.py','come.py'))]}
    previous=out/'render_receipt.json'
    if previous.exists() and read_json(previous)!=receipt:raise ValueError('render options/source differ; NEW output directory required')
    write_json(previous,receipt);entries=[];groups={'short':[],'long':[]}
    with zipfile.ZipFile(assets) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate output archive members')
        manifest=json.loads(z.read('manifest.json'))
        if manifest['format']!=FORMAT or manifest['status']!='complete':raise ValueError('completed GAST comparison packet required')
        if cases and not set(cases)<=set(manifest['cases']):raise ValueError('unknown case ID')
        for key,case in manifest['cases'].items():
            if cases and key not in cases:continue
            payload=z.read('cases/'+key+'.npz')
            if hashlib.sha256(payload).hexdigest()!=case['sha256']:raise ValueError('prediction checksum differs: '+key)
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:values={k:a[k] for k in a.files}
            n=len(case['report_indices']);shape=tuple(manifest['grid']['shape_hwd'])
            for name,value in values.items():
                if value.dtype!=np.uint8 or np.any(value>17) or value.shape!=(shape if name=='t0' else (n,*shape)):
                    raise ValueError('native comparison arrays required')
            rows,columns,settings=rows_and_settings(key,values,case['future_ego_poses'])
            image=out/(key+'.png');audit_path=out/(key+'.json')
            if image.exists() and audit_path.exists():
                audit=read_json(audit_path)
                if sha(image)!=audit['image_sha256']:raise ValueError('existing image changed: '+key)
            else:
                picture,audit=renderer(rows,columns,manifest['grid'],(640,360),camera,preview=preview,panel_settings=settings)
                picture.save(image)
                write_json(audit_path,dict(audit,case=key,scene=case['scene'],image_sha256=sha(image),source=receipt))
            group='short' if key.startswith('short') else 'long'
            caption=key+' '+case['scene'];groups[group].append((image.name,caption))
            entries.append(f'<h3>{html.escape(caption)}</h3><a href="{image.name}"><img width="1400" src="{image.name}"></a>')
            print('GAST_RENDER',key,case['scene'],flush=True)
    for group,items in groups.items():
        if items:contact_sheets(out,group,items)
    (out/'index.html').write_text('<meta charset="utf-8"><title>GAST / CarryWorld same-window comparison</title>'
        '<p>GT / GAST / CarryWorld. Same native semantics, COME palette and camera. Long: GT-ego conditioned 2x3s rollout, '
        'displayed in the common 1s ego frame. No circles or smoothing.</p>'+'\n'.join(entries),encoding='utf-8')
    print('GAST_GALLERY',out/'index.html',flush=True)


def render_check(out):
    import os
    import ctypes.util
    if os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')=='vtkOSOpenGLRenderWindow' and not ctypes.util.find_library('OSMesa'):
        raise RuntimeError('OSMesa system library missing; run: bash tools/gast_baseline/step.sh render-libs')
    from tools.paper_assets.vtk_render import render as native
    from .common import GRID
    sem=np.full(GRID['shape_hwd'],17,np.uint8);sem[98:103,98:103,:3]=11
    try:picture,audit=native(sem,GRID,camera='come_reference',size=(640,360))
    except Exception as exc:raise RuntimeError('Offscreen renderer unavailable. Run: bash tools/gast_baseline/step.sh render-libs; '+str(exc)) from exc
    if not np.asarray(picture)[...,3].any():raise RuntimeError('CPU rendering produced no occupied pixels')
    if os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')=='vtkOSOpenGLRenderWindow' and audit.get('render_window')!='vtkOSOpenGLRenderWindow':
        raise RuntimeError('requested CPU OSMesa but VTK selected a different backend; inspect render-libs')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);picture.save(out/'render_check.png');write_json(out/'render_check.json',audit)
    print('GAST_RENDER_CPU_OK',audit.get('graphics'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check',action='store_true');p.add_argument('--cases',nargs='+')
    p.add_argument('--camera',default='come_reference');a=p.parse_args()
    if a.check:render_check(a.out)
    elif a.assets:render(a.assets,a.out,camera=a.camera,cases=a.cases)
    else:p.error('--assets or --check required')


if __name__=='__main__':main()
