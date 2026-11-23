"""Readable GAST comparison figures: shared margins and explicit display frame.

Frozen semantic arrays are never modified. Motion views use one fixed 1s ego
reference. Without a separately verified input pose, short motion views omit t0.
"""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from tools.gast_baseline.common import FORMAT, read_json, sha, write_json
from tools.gast_baseline.render import rows_and_settings
from .qualitative_render import render_grid


def export_input_poses(prepared, out):
    prepared, out = Path(prepared), Path(out)
    source = read_json(prepared)
    if source['format'] != 'GAST_CPU_prepared_v1':
        raise ValueError('completed GAST prepared.json required')
    cases = {}
    for key, case in source['cases'].items():
        frame = case['frames'][3]
        if frame['token'] != case['original']['t0_token']:
            raise ValueError('input pose token differs: ' + key)
        cases[key] = dict(t0_token=frame['token'], pose=frame['pose'])
    payload = dict(format='GAST_display_input_poses_v1', prepared_sha256=sha(prepared), cases=cases)
    if out.exists() and read_json(out) != payload:
        raise ValueError('existing input pose export differs; choose a new file')
    write_json(out, payload)
    print('GAST_DISPLAY_POSES', out, len(cases), 'cases;', out.stat().st_size, 'bytes; CPU only')


def checked_input_poses(path, manifest):
    if path is None:
        return None
    value = read_json(path)
    if (value['format'] != 'GAST_display_input_poses_v1'
            or value['prepared_sha256'] != manifest['receipt']['prepared_sha256']
            or set(value['cases']) != set(manifest['cases'])):
        raise ValueError('input poses must come from this exact prepared.json and case population')
    for key, row in value['cases'].items():
        pose = np.asarray(row['pose'], dtype=np.float64)
        if (row['t0_token'] != manifest['cases'][key]['t0_token'] or pose.shape != (4, 4)
                or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1])):
            raise ValueError('input pose or token differs: ' + key)
    return value['cases']


def comparison_rows(key, arrays, case, view, input_pose=None):
    if view == 'ego':
        short = key.startswith('short')
        rows = [(label, [arrays['t0'], *arrays[name]] if short else list(arrays[name]))
                for label, name in (('GT', 'gt'), ('GAST', 'g'), ('CarryWorld', 'e'))]
        columns = [f'{i} s' for i in range(1, len(case['report_indices'])+1)]
        if short:
            columns = ['t0', *columns]
        return rows, columns, {}, 'each future ego frame; camera follows ego'
    if view == 'native':
        rows, columns, settings = rows_and_settings(key, arrays, case['future_ego_poses'])
        frame = 'each future ego frame' if key.startswith('short') else 'fixed 1s ego reference'
        return rows, columns, settings, frame
    if view != 'motion':
        raise ValueError('unknown display view')
    poses = np.asarray(case['future_ego_poses'], dtype=np.float64)[case['report_indices']]
    transforms = np.linalg.inv(poses[0]) @ poses
    rows = [(label, list(arrays[name])) for label, name in (('GT', 'gt'), ('GAST', 'g'), ('CarryWorld', 'e'))]
    columns = [f'{i} s' for i in range(1, len(poses) + 1)]
    if key.startswith('short') and input_pose is not None:
        if input_pose['t0_token'] != case['t0_token']:
            raise ValueError('input pose token differs')
        transforms = np.concatenate(((np.linalg.inv(poses[0]) @ np.asarray(input_pose['pose']))[None], transforms))
        rows = [(label, [arrays['t0'], *panels]) for label, panels in rows]
        columns = ['t0', *columns]
    settings = {id(panel): dict(display_transform=transform)
                for _, panels in rows for panel, transform in zip(panels, transforms)}
    return rows, columns, settings, 'fixed 1s ego reference'


def render(assets, out, *, view='native', cases=None, input_poses=None):
    assets, out = Path(assets), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    helpers = Path(__file__).parent
    receipt = dict(assets_sha256=sha(assets), view=view,
                   input_poses_sha256=sha(input_poses) if input_poses else None,
                   cases=list(cases) if cases else None, camera='come_reference',
                   trim='union of transparent margins across ALL methods/horizons; common scale and aspect',
                   implementation={name: sha(helpers/name) for name in
                       ('gast_readable_gallery.py', 'qualitative_render.py', 'vtk_render.py', 'come.py')})
    receipt_path = out/'render_receipt.json'
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError('display settings changed; choose a new output directory')
    write_json(receipt_path, receipt)
    entries = []
    with zipfile.ZipFile(assets) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate archive members')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != FORMAT or manifest['status'] != 'complete':
            raise ValueError('completed GAST comparison packet required')
        poses = checked_input_poses(input_poses, manifest)
        if cases and not set(cases) <= set(manifest['cases']):
            raise ValueError('unknown case IDs')
        for key, case in manifest['cases'].items():
            if cases and key not in cases:
                continue
            payload = archive.read('cases/'+key+'.npz')
            if hashlib.sha256(payload).hexdigest() != case['sha256']:
                raise ValueError('prediction checksum differs: ' + key)
            with np.load(io.BytesIO(payload), allow_pickle=False) as packed:
                arrays = {name: packed[name] for name in packed.files}
            if set(arrays) != {'t0', 'gt', 'g', 'e'}:
                raise ValueError('prediction fields differ')
            for name, array in arrays.items():
                expected = tuple(manifest['grid']['shape_hwd'])
                if name != 't0':
                    expected = (len(case['report_indices']), *expected)
                if array.shape != expected or array.dtype != np.uint8 or np.any(array > 17):
                    raise ValueError('native semantics required')
            rows, columns, settings, frame = comparison_rows(key, arrays, case, view,
                                                            poses[key] if poses else None)
            picture_path, audit_path = out/(key+'.png'), out/(key+'.json')
            if picture_path.exists() and audit_path.exists():
                audit = read_json(audit_path)
                if sha(picture_path) != audit['image_sha256']:
                    raise ValueError('completed picture changed')
            else:
                picture, audit = render_grid(rows, columns, manifest['grid'], (840, 472), 'come_reference',
                    preview=True, panel_settings=settings, trim_margins=True, fit_content=True)
                temp = picture_path.with_suffix('.tmp.png')
                picture.save(temp); temp.replace(picture_path)
                transform_audit = [[settings.get(id(panel), {}).get('display_transform', np.eye(4)).tolist()
                                    for panel in panels] for _, panels in rows]
                write_json(audit_path, dict(audit, image_sha256=sha(picture_path), case=key,
                    scene=case['scene'], display_frame=frame, display_transforms=transform_audit, source=receipt))
            title = html.escape(key+' '+case['scene'])
            entries.append(f'<h2>{title}</h2><p>{frame}</p><a href="{key}.png">'
                           f'<img loading="lazy" style="width:100%;max-width:1800px" src="{key}.png"></a>')
            print('GAST_DISPLAY', view, key, case['scene'], frame, flush=True)
    notes = ('Ego views: ALL future frames in their respective ego coordinates; display camera follows ego. '
             'Short native views: t0/1/2/3s in their respective ego frames; legacy native long views use fixed 1s reference. '
             'Motion views: all displayed futures use the SAME 1s reference. '
             'Without the exact input pose, t0 is omitted from motion views. '
             'All methods/horizons use one shared transparent-margin crop and scale. '
             'COME camera direction, palette and native semantic arrays are unchanged.')
    (out/'index.html').write_text('<meta charset="utf-8"><title>GT / GAST / CarryWorld</title>'
        '<body style="font-family:system-ui;max-width:1850px;margin:24px auto">'
        f'<h1>GT / GAST / CarryWorld — {view}</h1><p>{notes}</p>'+''.join(entries), encoding='utf-8')
    print('GAST_DISPLAY_GALLERY', out/'index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    export = sub.add_parser('export-poses')
    export.add_argument('--prepared', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    draw = sub.add_parser('render')
    draw.add_argument('--assets', type=Path, required=True)
    draw.add_argument('--out', type=Path, required=True)
    draw.add_argument('--view', choices=('native', 'motion', 'ego'), default='native')
    draw.add_argument('--cases', nargs='+')
    draw.add_argument('--input-poses', type=Path)
    args = parser.parse_args()
    if args.action == 'export-poses':
        export_input_poses(args.prepared, args.out)
    else:
        render(args.assets, args.out, view=args.view, cases=args.cases, input_poses=args.input_poses)


if __name__ == '__main__':
    main()
