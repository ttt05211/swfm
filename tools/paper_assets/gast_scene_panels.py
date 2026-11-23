"""Export independent native-ego PNGs per selected short or long scene."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image
from tools.gast_baseline.common import FORMAT, sha, write_json
from . import vtk_render


def export(assets, out, scenes, *, horizon='short', models=('gt', 'carry', 'gast')):
    assets, out = Path(assets), Path(out)
    if horizon not in ('short', 'long'):
        raise ValueError('short or long horizon required')
    sources = dict(gt='gt', carry='e', gast='g')
    if not models or len(set(models)) != len(models) or not set(models) <= set(sources):
        raise ValueError('select unique gt/carry/gast models')
    short = horizon == 'short'
    times = [0, 1, 2, 3] if short else [1, 2, 3, 4, 5, 6]
    report_indices = [1, 3, 5] if short else [1, 3, 5, 7, 9, 11]
    names = []
    for scene in scenes:
        match = re.fullmatch(r'(?:scene-)?(\d{1,4})', scene)
        if not match:
            raise ValueError('scene ID expected: ' + scene)
        names.append('scene-'+match.group(1).zfill(4))
    if len(names) != len(set(names)):
        raise ValueError('duplicate selected scene')
    if out.exists() or out.with_suffix('.zip').exists():
        raise ValueError('choose a NEW output path; existing exports are preserved')
    receipt = dict(source=str(assets.resolve()), source_sha256=sha(assets), view='native ego at each time',
                   camera='come_reference', times_s=times, models=list(models), scenes={})
    with zipfile.ZipFile(assets) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate source archive members')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != FORMAT or manifest['status'] != 'complete':
            raise ValueError('completed GAST comparison packet required')
        selected = {}
        for scene in names:
            hits = [key for key, case in manifest['cases'].items()
                    if key.startswith('short' if short else 'turn') and case['scene'] == scene]
            if len(hits) != 1:
                raise ValueError('one unambiguous '+horizon+' case required: ' + scene)
            selected[scene] = hits[0]
        for scene, key in selected.items():
            case = manifest['cases'][key]; payload = archive.read('cases/'+key+'.npz')
            if case['report_indices'] != report_indices:
                raise ValueError('exact integer-second '+horizon+' readouts required')
            if hashlib.sha256(payload).hexdigest() != case['sha256']:
                raise ValueError('source prediction checksum differs')
            with np.load(io.BytesIO(payload), allow_pickle=False) as packed:
                arrays = {name: packed[name] for name in packed.files}
            shape = tuple(manifest['grid']['shape_hwd'])
            if set(arrays) != {'t0', 'gt', 'g', 'e'}:
                raise ValueError('native comparison arrays required')
            for name, array in arrays.items():
                if array.dtype != np.uint8 or array.shape != (shape if name == 't0' else (len(report_indices), *shape)) or np.any(array > 17):
                    raise ValueError(horizon+' native semantic array required')
            rows = [(name, [arrays['t0'], *arrays[sources[name]]] if short else list(arrays[sources[name]]))
                    for name in models]
            raw = {}
            for _, frames in rows:
                for sem in frames:
                    if id(sem) not in raw:
                        raw[id(sem)], _ = vtk_render.render(sem, manifest['grid'], camera='come_reference',
                            size=vtk_render.COME_REFERENCE_SIZE)
            trimmed, box = vtk_render.trim_sequence(list(raw.values()))
            images = dict(zip(raw, trimmed)); folder = out/scene; folder.mkdir(parents=True)
            hashes = {}
            for prefix, frames in rows:
                for time, sem in zip(times, frames):
                    image = images[id(sem)]; opaque = Image.new('RGB', image.size, 'white')
                    opaque.paste(image, mask=image.getchannel('A'))
                    target = folder/f'{prefix}_{time}.png'; opaque.save(target, dpi=(300, 300))
                    hashes[target.name] = sha(target)
            if short:
                assert len({hashes[f'{name}_0.png'] for name in models}) == 1
            receipt['scenes'][scene] = dict(case=key, crop=box, pixel_size=list(trimmed[0].size), files=hashes)
            print('GAST_SCENE_PANELS', scene, key, len(times)*len(models), 'PNGs', flush=True)
    target = out.with_suffix('.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as archive:
        for scene in names:
            for image in sorted((out/scene).glob('*.png')):
                archive.write(image, image.relative_to(out).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(times)*len(models)*len(names):
            raise ValueError('export ZIP did not validate')
    write_json(out/'export_receipt.json', receipt)
    print('GAST_SCENE_ZIP', target.resolve(), f'{target.stat().st_size/2**20:.2f} MiB', flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--horizon', choices=('short', 'long'), default='short')
    parser.add_argument('--models', nargs='+', choices=('gt', 'carry', 'gast'), default=['gt', 'carry', 'gast'])
    args = parser.parse_args()
    export(args.assets, args.out, args.scenes, horizon=args.horizon, models=args.models)
