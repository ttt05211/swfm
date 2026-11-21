"""COME-style bright voxel assets, without Mayavi/Xvfb or model changes.

Palette and camera reference: synsin0/COME tools/vis_utils.py and
tools/visualize_nuscenes_occupancy.py (VIEW_1). The renderer below is our own
dependency-light exposed-cube-face renderer, not a pixel-identical Mayavi port.
Input arrays are read only; no fake ego voxels or geometry rescaling is applied.
"""
import argparse
import base64
import hashlib
import itertools
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Direct semantic IDs, not COME's 1-indexed LUT: prepend others, append free.
COLORS = np.array([
    [132, 144, 156], [255, 120, 50], [255, 192, 203], [255, 255, 0],
    [0, 150, 245], [0, 255, 255], [255, 127, 0], [255, 0, 0],
    [255, 240, 150], [135, 60, 0], [160, 32, 240], [255, 0, 255],
    [139, 137, 137], [75, 0, 75], [150, 240, 80], [230, 230, 250],
    [0, 175, 0], [255, 255, 255]], dtype=np.uint8)
CAMERA = np.array([-70., -70., 60.])
FOCAL = np.array([.008333206176757812, -.008333206176757812, -3.])


def project(points):
    """Perspective projection in native x/y/z, with COME's oblique camera."""
    forward = FOCAL-CAMERA; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points, dtype=np.float64)-CAMERA
    depth = delta @ forward
    if np.any(depth <= 0):
        raise ValueError('geometry behind the display camera')
    return np.stack((delta @ right / depth, -delta @ up / depth), axis=-1), depth


def cube_faces(sem, grid):
    """Only camera-facing, exposed FULL voxel faces; no top-height collapse.

    Removing internal faces is display occlusion culling, not downsampling.
    The camera lies on -x/-y/+z sides of the standard occupancy volume.
    """
    sem = np.asarray(sem)
    if sem.ndim != 3 or tuple(sem.shape) != tuple(grid['shape_hwd']):
        raise ValueError('semantic shape must match grid.shape_hwd')
    occupied = (sem > 0) & (sem < 17)  # COME display filter only.
    origin = np.array([grid[k] for k in ('x_min', 'y_min', 'z_min')])
    step = np.asarray(grid['voxel_size'])
    high = origin+np.asarray(sem.shape)*step
    if not (CAMERA[0] < origin[0] and CAMERA[1] < origin[1] and CAMERA[2] > high[2]):
        raise ValueError('fixed COME camera must be outside -x/-y/+z grid faces')
    vertices, rgb = [], []
    for axis, side, shade in ((0, -1, .83), (1, -1, .92), (2, 1, 1.)):
        neighbor = np.zeros_like(occupied)
        dst, src = [slice(None)]*3, [slice(None)]*3
        dst[axis], src[axis] = ((slice(1, None), slice(None, -1)) if side < 0
                               else (slice(None, -1), slice(1, None)))
        neighbor[tuple(dst)] = occupied[tuple(src)]
        indices = np.argwhere(occupied & ~neighbor)
        other = [a for a in range(3) if a != axis]
        offsets = np.zeros((4, 3))
        offsets[:, axis] = int(side > 0)
        offsets[:, other] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        vertices.append(origin+(indices[:, None, :]+offsets[None])*step)
        rgb.append(np.rint(COLORS[sem[tuple(indices.T)]]*shade).astype(np.uint8))
    return np.concatenate(vertices), np.concatenate(rgb)


def grid_bounds(grid):
    """One fixed display framing for all frames: no per-frame zoom/crop."""
    lo = np.array([grid[k] for k in ('x_min', 'y_min', 'z_min')])
    hi = lo+np.asarray(grid['shape_hwd'])*np.asarray(grid['voxel_size'])
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    uv, _ = project(corners)
    return np.stack((uv.min(0), uv.max(0)))


def sequence_bounds(frames, grid):
    """Tight shared framing, based on displayed geometry, never GT scores."""
    bounds = []
    for sem in frames:
        faces, _ = cube_faces(sem, grid)
        if len(faces):
            uv, _ = project(faces)
            bounds.append([uv.min((0, 1)), uv.max((0, 1))])
    if not bounds:
        return grid_bounds(grid)
    values = np.asarray(bounds)
    return np.stack((values[:, 0].min(0), values[:, 1].max(0)))


def render(sem, grid, *, size=(1280, 850), bounds=None, supersample=2):
    """Transparent RGBA; side lighting only, no outlines/ground plane/axes."""
    faces, colors = cube_faces(sem, grid)
    bounds = grid_bounds(grid) if bounds is None else np.asarray(bounds)
    width, height = size
    span = bounds[1]-bounds[0]
    scale = .94*min(width/span[0], height/span[1])
    uv, depth = project(faces)
    xy = ((uv-bounds.mean(0))*scale+np.array(size)/2)*supersample
    canvas = Image.new('RGBA', (width*supersample, height*supersample))
    draw = ImageDraw.Draw(canvas)
    # Far -> near. All cubes share an axis-aligned lattice, no intersecting mesh.
    order = np.argsort(-depth.mean(1), kind='stable')
    for i in order:
        draw.polygon([tuple(p) for p in xy[i]], fill=tuple(colors[i])+(255,))
    return canvas.resize(size, Image.Resampling.LANCZOS)


def font(size):
    from matplotlib.font_manager import findfont
    return ImageFont.truetype(findfont('DejaVu Sans'), size)


def save_image(image, stem):
    """White PNG, transparent PNG and SVG wrapper for figure composition."""
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    image.save(stem.parent/(stem.name+'_transparent.png'))
    white = Image.new('RGBA', image.size, 'white'); white.alpha_composite(image)
    white.convert('RGB').save(stem.parent/(stem.name+'.png'))
    payload = base64.b64encode((stem.parent/(stem.name+'_transparent.png')).read_bytes()).decode()
    (stem.parent/(stem.name+'.svg')).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" '
        f'viewBox="0 0 {image.width} {image.height}">'
        f'<image width="{image.width}" height="{image.height}" href="data:image/png;base64,{payload}"/></svg>',
        encoding='utf-8')


def sheet(rows, path, *, scene, note):
    """Minimal white contact sheet; image panels raster, SVG labels editable."""
    width, row_height = 2000, 320
    height = 90+len(rows)*row_height+75
    image = Image.new('RGB', (width, height), 'white'); draw = ImageDraw.Draw(image)
    draw.text((36, 20), f'{scene}  /  bright semantic voxels', font=font(28), fill='black')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="36" y="47" font-family="sans-serif" font-size="28">{escape(scene)} / bright semantic voxels</text>']
    for r, (title, panels) in enumerate(rows):
        y = 90+r*row_height
        draw.text((36, y), title, font=font(25), fill='black')
        svg.append(f'<text x="36" y="{y+25}" font-family="sans-serif" font-size="25">{escape(title)}</text>')
        cell = (width-48)//len(panels)
        for i, (label, panel) in enumerate(panels):
            box = panel.copy(); box.thumbnail((cell-8, row_height-55), Image.Resampling.LANCZOS)
            x, yy = 24+i*cell+(cell-box.width)//2, y+47+(row_height-90-box.height)//2
            image.paste(box, (x, yy), box)
            draw.text((24+i*cell+14, y+row_height-28), label, font=font(19), fill='#333333')
            import io
            buf = io.BytesIO(); box.save(buf, format='PNG')
            payload = base64.b64encode(buf.getvalue()).decode()
            svg.extend([f'<image x="{x}" y="{yy}" width="{box.width}" height="{box.height}" href="data:image/png;base64,{payload}"/>',
                        f'<text x="{24+i*cell+14}" y="{y+row_height-9}" font-family="sans-serif" font-size="19">{escape(label)}</text>'])
    draw.text((36, height-42), note, font=font(18), fill='#555555')
    svg.append(f'<text x="36" y="{height-24}" font-family="sans-serif" font-size="18" fill="#555">{escape(note)}</text></svg>')
    image.save(path/'overview.png')
    (path/'overview.svg').write_text(''.join(svg), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(assets, out, *, include_futures=True):
    """Re-render an existing asset export. No bundle/model/GT/cache reads."""
    assets, out = Path(assets).resolve(), Path(out).resolve()
    manifest_file, history_file = assets/'manifest.json', assets/'raw/history.npz'
    before = {str(p): digest(p) for p in (manifest_file, history_file)}
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'overview_assets_v1' or manifest.get('status') != 'complete':
        raise ValueError('expected a completed paper_assets.run export')
    grid = manifest['grid']
    with np.load(history_file, allow_pickle=False) as raw:
        semantics = raw['semantics']
    if len(semantics) != 4:
        raise ValueError('FOUR histories required')
    # Resolve all optional content before output creation, never substitute a model.
    future_path, future = None, None
    kind = manifest['future_content']
    if include_futures:
        stem = {'KTA_SCHEMATIC_NOT_MODEL_PREDICTION': 'kta_schematic',
                'MODEL_PREDICTION': 'prediction'}.get(kind)
        if stem is None:
            raise ValueError('unknown future content; refusing an ambiguous label')
        future_path = assets/'raw'/(stem+'.npz'); before[str(future_path)] = digest(future_path)
        with np.load(future_path, allow_pickle=False) as raw:
            future = raw['semantics']
        if len(future) != 6:
            raise ValueError('SIX future frames required')
    out.mkdir(parents=True, exist_ok=False)
    display_frames = list(semantics)+(list(future) if future is not None else [])
    bounds, histories, futures = sequence_bounds(display_frames, grid), [], []
    times = np.asarray(manifest['window']['history_times_s'])
    times = times-times[-1]
    for i, sem in enumerate(semantics):
        image = render(sem, grid, bounds=bounds)
        save_image(image, out/'history'/f'history_{i}')
        histories.append((f'{times[i]:+.1f}s' if i < 3 else 't = 0', image))
        print(f'COME_RENDER history {i+1}/4', flush=True)
    save_image(histories[-1][1], out/'hero_t0')
    rows = [('Observed history / real occupancy', histories)]
    if future is not None:
        for i, sem in enumerate(future):
            image = render(sem, grid, bounds=bounds)
            save_image(image, out/'future'/f'{stem}_{(i+1)*.5:.1f}s')
            futures.append((f'{(i+1)*.5:.1f}s', image))
            print(f'COME_RENDER future {i+1}/6 ({kind})', flush=True)
        # Four panels vs six panels remain same physical camera, but a denser row.
        rows.append(('Model prediction' if stem == 'prediction' else
                     'KTA illustration / NOT a model prediction', futures))
    note = 'COME palette + camera direction; native xyz, full voxels; no model or metric changes.'
    sheet(rows, out, scene=manifest['window']['scene_name'], note=note)
    if any(digest(p) != sha for p, sha in before.items()):
        raise RuntimeError('source assets changed during rendering')
    receipt = dict(protocol='come_style_voxel_assets_v1', status='complete',
        source_assets=str(assets), source_sha256=before, window=manifest['window'],
        future_content=kind if future is not None else None, grid=grid,
        reference='https://github.com/synsin0/COME/blob/main/tools/vis_utils.py',
        camera_reference='https://github.com/synsin0/COME/blob/main/tools/visualize_nuscenes_occupancy.py',
        palette_by_semantic_id=COLORS.tolist(), display_filter='0 < semantics < 17 (display only)',
        coordinate_order='native xyz; no transpose, z exaggeration, decimation or ego insertion',
        camera=dict(position=CAMERA.tolist(), focal_point=FOCAL.tolist(), up=[0, 0, 1],
                    projection='perspective, fixed grid framing across frames', shared_bounds=bounds.tolist()),
        renderer='our exposed-cube-face Pillow renderer; not pixel-identical Mayavi',
        source_writes=False, training_updates=0,
        files=sorted(str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()))
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f'COME_ASSETS_READY {out}/overview.png', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path, help='existing completed paper_assets export')
    parser.add_argument('--out', required=True, type=Path, help='NEW directory; never overwrite')
    parser.add_argument('--history-only', action='store_true', help='render only real observed occupancy')
    args = parser.parse_args()
    export(args.assets, args.out, include_futures=not args.history_only)


if __name__ == '__main__':
    main()
