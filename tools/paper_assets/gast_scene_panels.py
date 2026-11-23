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


def frame_sequence(images, *, tight_frame=False, zoom=1.):
    """One viewport for every method/time; zoom may clip peripheral geometry."""
    if not images or len({im.size for im in images}) != 1:
        raise ValueError('same-size rendered images required')
    if not np.isfinite(zoom) or zoom < 1:
        raise ValueError('zoom must be finite and at least one')
    if tight_frame:
        boxes = []
        for image in images:
            visible = Image.new('RGB', image.size, 'white')
            visible.paste(image, mask=image.getchannel('A'))
            # Nonzero SSAA alpha alone can retain visually white canvas edges.
            mask = Image.fromarray((np.asarray(visible).min(axis=2) < 245).astype(np.uint8)*255)
            if mask.getbbox() is not None:
                boxes.append(mask.getbbox())
        margin = max(1, round(.00625*max(images[0].size)))
        box = (max(0, min(b[0] for b in boxes)-margin), max(0, min(b[1] for b in boxes)-margin),
               min(images[0].width, max(b[2] for b in boxes)+margin),
               min(images[0].height, max(b[3] for b in boxes)+margin)) if boxes else (
                   0, 0, images[0].width, images[0].height)
    else:
        _, box = vtk_render.trim_sequence(images)
    visible_box = tuple(box)
    if zoom != 1:
        center = (np.asarray(box[:2])+box[2:])/2
        half_size = (np.asarray(box[2:])-box[:2])/(2*zoom)
        box = tuple(int(v) for v in np.rint(np.r_[center-half_size, center+half_size]))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('zoom leaves an empty viewport')
    return [im.crop(box) for im in images], tuple(box), visible_box


def export(assets, out, scenes, *, horizon='short', models=('gt', 'carry', 'gast'),
           tight_frame=False, zoom=1., render_scale=1.):
    assets, out = Path(assets), Path(out)
    if horizon not in ('short', 'long'):
        raise ValueError('short or long horizon required')
    sources = dict(gt='gt', carry='e', gast='g')
    if not models or len(set(models)) != len(models) or not set(models) <= set(sources):
        raise ValueError('select unique gt/carry/gast models')
    if not np.isfinite(zoom) or zoom < 1 or not np.isfinite(render_scale) or render_scale < 1:
        raise ValueError('zoom and render scale must be finite and at least one')
    render_size = tuple(round(v*render_scale) for v in vtk_render.COME_REFERENCE_SIZE)
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
                   camera='come_reference', times_s=times, models=list(models), scenes={},
                   framing=dict(tight_frame=tight_frame, zoom=zoom, render_scale=render_scale,
                                raw_pixel_size=list(render_size),
                                visible_threshold_rgb=245 if tight_frame else None,
                                shared_across_all_models_and_times=True,
                                peripheral_geometry_may_be_clipped=zoom > 1))
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
                            size=render_size)
                        print('GAST_SCENE_RENDER', scene, f'{len(raw)}/{len(times)*len(models)}', flush=True)
            trimmed, box, visible_box = frame_sequence(list(raw.values()), tight_frame=tight_frame, zoom=zoom)
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
            receipt['scenes'][scene] = dict(case=key, crop=box, pre_zoom_crop=visible_box,
                                          pixel_size=list(trimmed[0].size), files=hashes)
            print('GAST_SCENE_PANELS', scene, key, len(times)*len(models), 'PNGs', flush=True)
    if sha(assets) != receipt['source_sha256']:
        raise ValueError('source packet changed while rendering')
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
    parser.add_argument('--tight-frame', action='store_true', help='trim visually white margins with one shared crop')
    parser.add_argument('--zoom', type=float, default=1., help='shared viewport zoom; values above one clip outer content')
    parser.add_argument('--render-scale', type=float, default=1., help='render more pixels before cropping for crisp zoomed exports')
    args = parser.parse_args()
    export(args.assets, args.out, args.scenes, horizon=args.horizon, models=args.models,
           tight_frame=args.tight_frame, zoom=args.zoom, render_scale=args.render_scale)
