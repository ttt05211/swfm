"""Render every short/long candidate for manual selection."""
import argparse
from html import escape
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from .qualitative_render import font, load_case, render_grid, short_rows, fixed_reference_settings
from .vtk_render import CAMERAS


def road_settings(panels, future_poses, grid):
    poses = np.asarray(future_poses)[1::2]
    transforms = np.linalg.inv(poses[0]) @ poses
    origin = np.array([grid['x_min'], grid['y_min']])
    step = np.array(grid['voxel_size'][:2])
    road_points = []
    for grid_frame, transform in zip(panels, transforms):
        road = np.argwhere(np.any(grid_frame == 11, axis=2))
        xy = (road + .5) * step + origin
        road_points.append(xy @ transform[:2, :2].T + transform[:2, 3])
    if sum(len(points) for points in road_points):
        lo, hi = np.quantile(np.concatenate(road_points), [.16, .84], axis=0)
        center = (lo + hi) / 2
        radius = np.maximum((hi - lo) * .52 + 3., [10., 9.])
        lo, hi = center - radius, center + radius
    else:
        lo = origin
        hi = origin + step * np.array(grid['shape_hwd'][:2])
    bottom = grid['z_min']
    top = bottom + grid['shape_hwd'][2] * grid['voxel_size'][2]
    bounds = (lo[0], hi[0], lo[1], hi[1], bottom, top)
    return {id(frame): dict(display_transform=transform, display_bounds=bounds)
            for frame, transform in zip(panels, transforms)}


def contact_sheets(out, group, entries):
    for offset in range(0, len(entries), 8):
        page = entries[offset:offset + 8]
        cols = 2 if group == 'short' else 1
        width, height = ((1240, 268) if cols == 2 else (1460, 204))
        cell = width // cols
        rows = (len(page) + cols - 1) // cols
        canvas = Image.new('RGB', (width, rows * height), 'white')
        draw = ImageDraw.Draw(canvas)
        face = font(17)
        for i, (name, caption) in enumerate(page):
            x = (i % cols) * cell
            y = (i // cols) * height
            with Image.open(out / name) as shot:
                shot.thumbnail((cell - 20, height - 32), Image.Resampling.LANCZOS)
                canvas.paste(shot, (x + (cell - shot.width) // 2, y + 28))
            draw.text((x + 10, y + 3), caption, font=face, fill='black')
        canvas.save(out / f'{group}_{offset // 8 + 1}.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', choices=('short', 'long'))
    parser.add_argument('--camera', choices=CAMERAS, default='come_reference')
    parser.add_argument('--cases', nargs='+', help='Optional explicit candidate IDs for a small preview')
    args = parser.parse_args(argv)
    entries = {'short': [], 'left': [], 'right': []}
    receipt = {}
    with zipfile.ZipFile(args.assets) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != 'carryworld_qualitative_v1' or manifest['preview']:
            raise ValueError('Expected a complete server qualitative archive')
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Archive has duplicate members')
        if args.cases and not set(args.cases) <= set(manifest['cases']):
            parser.error('Unknown candidate ID')
        args.out.mkdir(parents=True, exist_ok=False)
        grid = manifest['grid']
        for key, case in manifest['cases'].items():
            if args.cases and key not in args.cases:
                continue
            short = key.startswith('short')
            if args.only and args.only != ('short' if short else 'long'):
                continue
            arrays = load_case(archive, manifest, key)
            scene = case['scene']
            caption = f'{key}  {scene}'
            if short:
                group = 'short'
                counts = case['selection'].get('box_vehicle_counts_0_1_2_3s')
                if counts:
                    caption += '  |  GT vehicles ' + '/'.join(map(str, counts))
                image, config = render_grid(short_rows(arrays), ['Input (0 s)', '1 s', '2 s', '3 s'],
                                            grid, (512, 288), args.camera, preview=True)
            else:
                group = case['selection']['direction']
                caption += f'  |  {group} turn'
                panels = list(arrays['e'])
                settings = (fixed_reference_settings(panels, case['future_ego_poses'])
                            if args.camera == 'come_reference' else
                            road_settings(panels, case['future_ego_poses'], grid))
                image, config = render_grid([('CarryWorld', panels)], [f'{i} s' for i in range(1, 7)],
                                            grid, (448, 252), args.camera, preview=True,
                                            panel_settings=settings)
                config['frame'] = 'fixed 1 s ego frame'
                if 'display_bounds' in settings[id(panels[0])]:
                    config['road_view_bounds'] = list(settings[id(panels[0])]['display_bounds'])
            name = f'{key}_{scene}.png'
            image.save(args.out / name)
            entries[group].append((name, caption))
            receipt[key] = dict(case=case, file=name, rendering=config)
            print('QUALITATIVE_GALLERY', caption, flush=True)
    html = ['<!doctype html><meta charset="utf-8"><title>CarryWorld qualitative candidates</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%;height:auto;display:block}',
            'article{margin:16px 0 32px}h3{margin:0 0 8px}</style><h1>CarryWorld candidates</h1>']
    for group, cases in entries.items():
        if not cases:
            continue
        contact_sheets(args.out, group, cases)
        html.append(f'<h2>{group.title()} ({len(cases)})</h2>')
        for name, caption in cases:
            html.append(f'<article><h3>{escape(caption)}</h3><a href="{name}">'
                        f'<img loading="lazy" src="{name}" alt="{escape(caption)}"></a></article>')
    (args.out / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    (args.out / 'gallery.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
