"""Render qualitative CarryWorld panels from a compact semantic-grid archive."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import come, vtk_render

PALETTE = come.COLORS  # OccWorld / COME's shared 16-class LUT, indexed by Occ3D ID.


def load_case(archive, manifest, key):
    info = manifest['cases'][key]
    expected = dict(t0=())
    if key.startswith('short'):
        expected.update(gt=(3,), e=(3,))
        if not manifest['preview']:
            expected['c'] = (3,)
    else:
        expected = dict(e=(6,))
    if set(info['arrays']) != set(expected):
        raise ValueError(f'{key}: missing or extra render arrays')
    shape = tuple(manifest['grid']['shape_hwd'])
    arrays = {}
    for name, prefix in expected.items():
        values = np.load(io.BytesIO(archive.read(f'{key}/{name}.npy')), allow_pickle=False)
        if values.shape != prefix + shape or values.dtype != np.uint8 or np.any(values > 17):
            raise ValueError(f'{key}/{name}: expected native uint8 semantic volume')
        arrays[name] = values
    return arrays


def short_rows(arrays):
    t0 = arrays['t0']
    rows = [('GT', [t0, *arrays['gt']])]
    if 'c' in arrays:
        rows.append(('Transport only', [t0, *arrays['c']]))
    rows.append(('CarryWorld', [t0, *arrays['e']]))
    return rows


def font(size):
    from matplotlib.font_manager import findfont
    return ImageFont.truetype(findfont('DejaVu Serif'), size=size)


def fixed_reference_settings(panels, future_poses):
    """Physical XYZ poses -> one 1s ego frame, before COME's display axes."""
    poses = np.asarray(future_poses, dtype=np.float64)[1::2]
    if poses.shape != (len(panels), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('one full future ego pose per selected panel required')
    transforms = np.linalg.inv(poses[0]) @ poses
    return {id(frame): dict(display_transform=transform)
            for frame, transform in zip(panels, transforms)}


def render_grid(rows, columns, grid, size, camera, *, preview=False, panel_settings=None):
    # Reuse the same rendered t0 wherever all models share the observed input.
    width, height = size
    reference = camera == 'come_reference'
    render_size = vtk_render.COME_REFERENCE_SIZE if reference else (width * 2, height * 2)
    rendered = {}
    camera_audit = None
    panel_settings = panel_settings or {}
    for _, panels in rows:
        for sem in panels:
            if id(sem) not in rendered:
                options = panel_settings.get(id(sem), {})
                shot, audit = vtk_render.render(sem, grid, size=render_size,
                                                 camera=camera, palette=PALETTE, **options)
                rendered[id(sem)] = shot
                if camera_audit is None:
                    camera_audit = {key: audit[key] for key in
                                    ('fitted_camera', 'view_angle', 'camera_clipping_range',
                                     'automatic_camera_fit')}
    if reference:
        trimmed = list(rendered.values())
        crop = (0, 0, *render_size)
    else:
        trimmed, crop = vtk_render.trim_sequence(list(rendered.values()))
    crop_w, crop_h = trimmed[0].size
    scale = min((width - 12) / crop_w, (height - 12) / crop_h)
    display_size = (round(crop_w * scale), round(crop_h * scale))
    rendered = {key: panel.resize(display_size, Image.Resampling.LANCZOS) for key, panel in zip(rendered, trimmed)}

    side, top = (81, 59) if preview else (104, 96)
    gap, margin = 9, 6
    column_font = font(28 if preview else 64)
    label_font = font(23 if preview else 52)
    total_w = side + len(columns) * width + (len(columns) - 1) * gap + margin
    total_h = top + len(rows) * height + (len(rows) - 1) * gap + margin
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(columns):
        x = side + col * (width + gap)
        draw.text((x + width / 2, top / 2 + 1), label, font=column_font, fill='#151515', anchor='mm')
    for row, (label, panels) in enumerate(rows):
        y = top + row * (height + gap)
        left_label = Image.new('RGBA', (height, side), (255, 255, 255, 0))
        ImageDraw.Draw(left_label).text((height / 2, side / 2), label, font=label_font,
                                        fill='#151515', anchor='mm')
        left_label = left_label.transpose(Image.Transpose.ROTATE_90)
        canvas.paste(left_label, (0, y), left_label)
        for col, sem in enumerate(panels):
            x = side + col * (width + gap)
            panel = rendered[id(sem)]
            canvas.paste(panel, (x + (width - panel.width) // 2, y + (height - panel.height) // 2), panel)
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline='#353535', width=1 if preview else 2)
    return canvas, dict(camera=camera, native_geometry=True, panel_width=width, panel_height=height,
                        shared_pixel_crop=list(crop), palette=PALETTE.tolist(), columns=columns,
                        rows=[label for label, _ in rows], raw_render_size=list(render_size),
                        automatic_margin_crop=not reference,
                        camera_parameters=camera_audit,
                        backend='PyVista / VTK', mayavi_executed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--camera', choices=vtk_render.CAMERAS, default='come_reference')
    parser.add_argument('--short', help='Short candidate ID, e.g. short01')
    parser.add_argument('--left', help='Left-turn candidate ID')
    parser.add_argument('--right', help='Right-turn candidate ID')
    args = parser.parse_args(argv)
    if bool(args.left) != bool(args.right):
        parser.error('Pass both --left and --right to compose the final long figure')
    final = bool(args.short or args.left)

    with zipfile.ZipFile(args.assets) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != 'carryworld_qualitative_v1' or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Unsupported or ambiguous qualitative archive')
        if final and manifest['preview']:
            raise ValueError('Local preview assets contain only GT and the final short forecast')
        cases = manifest['cases']
        short_keys = [args.short] if args.short else ([] if final else [k for k in cases if k.startswith('short')])
        if any(k not in cases or not k.startswith('short') for k in short_keys):
            parser.error('Unknown short candidate')
        turns = [args.left, args.right] if args.left else ([] if final else [k for k in cases if k.startswith('turn')])
        if any(k not in cases or not k.startswith('turn') for k in turns):
            parser.error('Unknown turn candidate')
        if args.left and [cases[k]['selection']['direction'] for k in turns] != ['left', 'right']:
            parser.error('Select a left turn and a right turn in that order')
        args.out.mkdir(parents=True, exist_ok=False)
        outputs = {}

        def save(name, rows, columns, keys, size, panel_settings=None):
            image, info = render_grid(rows, columns, manifest['grid'], size, args.camera, preview=not final,
                                      panel_settings=panel_settings)
            image.save(args.out / (name + '.png'), dpi=(300, 300))
            if final:
                image.save(args.out / (name + '.pdf'), resolution=300, quality=97, subsampling=0)
            outputs[name] = dict(info, cases={k: cases[k] for k in keys})
            print('QUALITATIVE_RENDERED', name, *(cases[k]['scene'] for k in keys), flush=True)

        short_columns = ['Input (0 s)', '1 s', '2 s', '3 s']
        for key in short_keys:
            name = 'short_qualitative' if final else f'{key}_{cases[key]["scene"]}'
            save(name, short_rows(load_case(archive, manifest, key)), short_columns, [key],
                 (864, 486) if final else (560, 315))
        if turns:
            rows = []
            settings = {}
            for index, key in enumerate(turns):
                info = cases[key]
                selection = info['selection']
                direction = selection['direction'].capitalize()
                label = (f'({chr(97 + index)}) {direction} turn' if final else
                         f'{key}  {info["scene"]}  {direction} {abs(selection["yaw_deg"]):.0f}°')
                panels = list(load_case(archive, manifest, key)['e'])
                rows.append((label, panels))
                settings.update(fixed_reference_settings(panels, info['future_ego_poses']))
            save('long_qualitative' if final else 'turn_candidates', rows, [f'{i} s' for i in range(1, 7)],
                 turns, (640, 360) if final else (464, 261), settings)
        receipt = {k: v for k, v in manifest.items() if k != 'cases'}
        (args.out / 'figures.json').write_text(json.dumps(dict(receipt, figures=outputs), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
