"""Native PyVista/VTK cube-glyph rendering, following SparseWorld-TC vis_id.py.

COME palette, white background, true z buffering and native voxel dimensions.
No smoothing, denoising, interpolation, fake ego or input/prediction edits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from . import come

CAMERAS = {
    'come_reference': ([-70., -70., 60.], [.008333206176757812, -.008333206176757812, -3.]),
    'come': ([-70., -70., 60.], [.008333206176757812, -.008333206176757812, -3.]),
    'sparseworld': ([-80., 0., 50.], [0., 0., 0.]),
    'motion_side': ([-12., -88., 62.], [0., 0., 0.]),
    'road_high': ([-8., -24., 110.], [0., 0., 0.]),
}

# COME tools/visualize_nuscenes_occupancy.py, VIEW_1. This profile preserves
# the camera exactly, rather than using reset_camera to keep its direction.
COME_REFERENCE_SIZE = (2560, 1440)
COME_REFERENCE_CLIPPING = (10., 222.91192666552377)
COME_DISPLAY_AXES = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])


def backend():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError('PyVista backend unavailable. Install with the active interpreter: '
                           'python -m pip install -r tools/paper_assets/requirements-viz.txt; '
                           'no silent Pillow fallback.') from exc
    return pv


def voxel_mesh(semantics, grid, *, palette=come.COLORS):
    """PolyData centres -> actual VTK glyph cubes, native size and class RGB."""
    pv = backend()
    sem = np.asarray(semantics)
    if sem.ndim != 3 or tuple(sem.shape) != tuple(grid['shape_hwd']):
        raise ValueError('semantic shape does not match the declared grid')
    if not np.issubdtype(sem.dtype, np.integer):
        raise ValueError('discrete integer semantic labels required')
    at = np.argwhere((sem > 0) & (sem < 17))  # COME DISPLAY filter, no metric change.
    labels = sem[tuple(at.T)].astype(np.uint8)
    origin = np.array([grid[k] for k in ('x_min', 'y_min', 'z_min')], dtype=np.float64)
    step = np.array(grid['voxel_size'], dtype=np.float64)
    if step.shape != (3,) or not np.all(np.isfinite(step) & (step > 0)):
        raise ValueError('three positive finite native voxel dimensions required')
    points = origin+(at+.5)*step  # Explicit xyz; never swap x/y or collapse z.
    cloud = pv.PolyData(points)
    cube = pv.Cube(center=(0, 0, 0), x_length=step[0], y_length=step[1], z_length=step[2])
    if not len(at):
        return pv.PolyData(), dict(occupied_voxels=0, points_per_cube=cube.n_points,
                                  faces_per_cube=cube.n_cells)
    cloud['rgb'] = palette[labels]
    cloud['semantic_id'] = labels
    mesh = cloud.glyph(scale=False, orient=False, geom=cube)
    if mesh.n_points != len(at)*cube.n_points or mesh.n_cells != len(at)*cube.n_cells:
        raise RuntimeError('glyph count mismatch; refusing an altered voxel view')
    return mesh, dict(occupied_voxels=len(at), points_per_cube=cube.n_points,
                     faces_per_cube=cube.n_cells, glyph_points=mesh.n_points,
                     glyph_faces=mesh.n_cells, native_voxel_m=step.tolist())


def grid_bounds(grid):
    lo = [grid[k] for k in ('x_min', 'y_min', 'z_min')]
    hi = np.array(lo)+np.array(grid['shape_hwd'])*np.array(grid['voxel_size'])
    return tuple(v for pair in zip(lo, hi) for v in pair)


def trim_sequence(images):
    """Shared transparent-margin crop only, no scaling or geometry changes."""
    boxes = [im.getchannel('A').getbbox() for im in images]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return list(images), (0, 0, images[0].width, images[0].height)
    margin = int(.025*max(images[0].size))
    box = (max(0, min(b[0] for b in boxes)-margin), max(0, min(b[1] for b in boxes)-margin),
           min(images[0].width, max(b[2] for b in boxes)+margin),
           min(images[0].height, max(b[3] for b in boxes)+margin))
    return [im.crop(box) for im in images], box


def render(sem, grid, *, camera='come', size=(2560, 1600), palette=come.COLORS,
           display_transform=None, display_bounds=None, marker_xyz=None):
    pv = backend()
    if camera not in CAMERAS:
        raise ValueError('unknown camera')
    if len(size) != 2 or min(size) < 32:
        raise ValueError('image dimensions must be at least 32 pixels')
    mesh, audit = voxel_mesh(sem, grid, palette=palette)
    reference = camera == 'come_reference'
    if reference and display_bounds is not None:
        raise ValueError('COME reference camera has fixed framing; display_bounds is unsupported')
    if display_transform is not None and mesh.n_points:
        mesh.transform(np.asarray(display_transform), inplace=True)
    if reference and mesh.n_points:
        # COME meshgrid maps array[i,j,k] to display[j,i,k]. Apply AFTER a
        # physical XYZ ego transform, so fixed-reference rollout remains valid.
        mesh.transform(COME_DISPLAY_AXES, inplace=True)
    bounds = grid_bounds(grid) if display_bounds is None else display_bounds
    # Local theme, not pv.set_plot_theme(): leave other tools' global theme alone.
    plotter = pv.Plotter(off_screen=True, window_size=size, theme=pv.themes.DocumentTheme())
    try:
        plotter.set_background('white')
        plotter.enable_anti_aliasing('ssaa')
        if mesh.n_points:
            plotter.add_mesh(mesh, scalars='rgb', rgb=True, show_scalar_bar=False,
                             show_edges=False, smooth_shading=False,
                             ambient=.3, diffuse=.7, specular=0., opacity=1.)
        position, target = CAMERAS[camera]
        plotter.camera_position = [position, target, [0., 0., 1.]]
        plotter.camera.view_angle = 40. if camera in ('come', 'come_reference') else 30.
        # Fixed full-grid framing, shared by all temporal frames. Retain the
        # reference direction; VTK adjusts distance to fit the native volume.
        if reference:
            plotter.camera.clipping_range = COME_REFERENCE_CLIPPING
        else:
            plotter.reset_camera(bounds=bounds, render=False)
            plotter.reset_camera_clipping_range()
        pixels = plotter.screenshot(transparent_background=True, return_img=True)
        if pixels is None or pixels.shape != (size[1], size[0], 4):
            raise RuntimeError('VTK did not produce the declared RGBA image')
        capabilities = plotter.render_window.ReportCapabilities()
        audit.update(backend='PyVista / VTK', pyvista=pv.__version__,
                     vtk='.'.join(map(str, pv.vtk_version_info)),
                     render_window=plotter.render_window.GetClassName(),
                     graphics=[s.strip() for s in capabilities.splitlines()
                               if 'OpenGL vendor' in s or 'OpenGL renderer' in s or 'OpenGL version' in s],
                     camera_name=camera, reference_position=position, reference_target=target,
                     fitted_camera=[list(v) for v in plotter.camera_position],
                     view_angle=plotter.camera.view_angle, fixed_native_grid_bounds=list(grid_bounds(grid)),
                     antialiasing='VTK SSAA', smooth_shading=False, geometry_smoothing=False,
                     pixel_size=list(size), geometry='native cube glyphs; no resampling')
        audit.update(automatic_camera_fit=not reference,
                     camera_clipping_range=list(plotter.camera.clipping_range))
        if reference:
            audit.update(reference_source='synsin0/COME tools/visualize_nuscenes_occupancy.py VIEW_1',
                         display_axes='COME meshgrid XY order', display_axis_matrix=COME_DISPLAY_AXES.tolist())
        if display_transform is not None:
            audit.update(display_transform=np.asarray(display_transform).tolist(), display_bounds=list(bounds))
        result = Image.fromarray(pixels)
        if marker_xyz is not None:
            from PIL import ImageDraw
            plotter.renderer.SetWorldPoint(*marker_xyz, 1.)
            plotter.renderer.WorldToDisplay()
            x, y, _ = plotter.renderer.GetDisplayPoint()
            y = size[1] - y - 1
            radius = size[0] * .024
            box = (x - radius, y - radius, x + radius, y + radius)
            draw = ImageDraw.Draw(result)
            draw.ellipse(box, outline='white', width=8)
            draw.ellipse(box, outline='#181818', width=3)
        return result, audit
    finally:
        plotter.close()


def export(assets, out, *, include_futures=True, camera='come', size=(2560, 1600)):
    """Read-only rerender of a completed paper_assets export; no model load."""
    backend()  # Fail before creating outputs, never silently substitute renderer.
    assets, out = Path(assets).resolve(), Path(out).resolve()
    meta_path, history_path = assets/'manifest.json', assets/'raw/history.npz'
    before = {str(p): come.digest(p) for p in (meta_path, history_path)}
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('protocol') != 'overview_assets_v1' or meta.get('status') != 'complete':
        raise ValueError('expected completed paper_assets.run export')
    with np.load(history_path, allow_pickle=False) as raw:
        history = raw['semantics']
    if len(history) != 4:
        raise ValueError('FOUR histories required')
    future, stem = None, None
    kind = meta.get('future_content')
    if include_futures:
        stem = {'MODEL_PREDICTION': 'prediction',
                'KTA_SCHEMATIC_NOT_MODEL_PREDICTION': 'kta_schematic'}.get(kind)
        if stem is None:
            raise ValueError('unknown future type; no implicit model or GT substitution')
        future_path = assets/'raw'/(stem+'.npz'); before[str(future_path)] = come.digest(future_path)
        with np.load(future_path, allow_pickle=False) as raw:
            future = raw['semantics']
        if len(future) != 6:
            raise ValueError('SIX future frames required')
    if camera not in CAMERAS:
        raise ValueError('unknown camera')
    out.mkdir(parents=True, exist_ok=False)
    grid, histories, futures, audits = meta['grid'], [], [], []
    times = np.asarray(meta['window']['history_times_s']); times = times-times[-1]
    for i, sem in enumerate(history):
        image, audit = render(sem, grid, camera=camera, size=size)
        histories.append((f'{times[i]:+.1f}s' if i < 3 else 't = 0', image))
        audits.append(dict(frame=f'history_{i}', **audit))
        print(f'VTK_RENDER history {i+1}/4 glyphs={audit["occupied_voxels"]}', flush=True)
    histories[-1][1].save(out/'hero_t0_full_frame.png')
    other_camera = 'sparseworld' if camera == 'come' else 'come'
    alternate, audit = render(history[-1], grid, camera=other_camera, size=size)
    alternate_images, alternate_crop = trim_sequence([alternate])
    come.save_image(alternate_images[0], out/'views'/f'{other_camera}_t0')
    audits.append(dict(frame='alternate_t0', **audit))
    if future is not None:
        for i, sem in enumerate(future):
            image, audit = render(sem, grid, camera=camera, size=size)
            futures.append((f'{(i+1)*.5:.1f}s', image))
            audits.append(dict(frame=f'future_{i}', **audit))
            print(f'VTK_RENDER future {i+1}/6 ({kind})', flush=True)
    cropped, shared_crop = trim_sequence([im for _, im in histories+futures])
    histories = [(label, im) for (label, _), im in zip(histories, cropped[:4])]
    futures = [(label, im) for (label, _), im in zip(futures, cropped[4:])]
    for i, (_, im) in enumerate(histories):
        come.save_image(im, out/'history'/f'history_{i}')
    for i, (_, im) in enumerate(futures):
        come.save_image(im, out/'future'/f'{stem}_{(i+1)*.5:.1f}s')
    come.save_image(histories[-1][1], out/'hero_t0')
    rows = [('Observed history / native PyVista voxels', histories)]
    if futures:
        rows.append(('Model prediction' if stem == 'prediction' else
                     'KTA illustration / NOT a model prediction', futures))
    come.sheet(rows, out, scene=meta['window']['scene_name'],
               note='PyVista/VTK cube glyphs + COME colors. Native geometry; NO smoothing or denoising.')
    if any(come.digest(p) != sha for p, sha in before.items()):
        raise RuntimeError('source asset changed while rendering')
    receipt = dict(protocol='pyvista_native_voxel_assets_v1', status='complete',
        source_assets=str(assets), source_sha256=before, window=meta['window'], grid=grid,
        renderer='PyVista / VTK; no Pillow 3D fallback', camera=camera,
        display_crop=dict(shared_history_future_pixels=list(shared_crop),
                          alternate_view_pixels=list(alternate_crop), rescaling=False,
                          original_render_pixels=list(size)),
        future_content=kind if future is not None else None,
        display_filter='0 < semantic_id < 17 (display only)', palette_by_semantic_id=come.COLORS.tolist(),
        geometry=dict(voxel_dimensions='native', coordinate_order='xyz',
                      denoising=False, interpolation=False, smoothing=False,
                      downsampling=False, fake_ego=False),
        frames=audits, source_writes=False, training_updates=0,
        files=sorted(str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()),
        reference='https://github.com/MrPicklesGG/SparseWorld-TC/blob/main/vis_id.py')
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f'VTK_ASSETS_READY {out}/overview.png', flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path, help='NEW output directory')
    ap.add_argument('--camera', choices=tuple(CAMERAS), default='come')
    ap.add_argument('--width', type=int, default=2560)
    ap.add_argument('--height', type=int, default=1600)
    ap.add_argument('--history-only', action='store_true')
    args = ap.parse_args(argv)
    export(args.assets, args.out, include_futures=not args.history_only,
           camera=args.camera, size=(args.width, args.height))


if __name__ == '__main__':
    main()
