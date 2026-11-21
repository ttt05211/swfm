import numpy as np
import pytest

pytest.importorskip('pyvista')
from tools.paper_assets import vtk_render as renderer


def grid(shape=(3, 3, 3)):
    return dict(x_min=-.6, y_min=-.6, z_min=-.6, voxel_size=[.4]*3, shape_hwd=shape)


def test_real_vtk_glyphs_keep_native_dimensions_labels_and_input():
    sem = np.full((3, 3, 3), 17, dtype=np.uint8); sem[1, 1, 1] = 4
    before = sem.copy()
    mesh, audit = renderer.voxel_mesh(sem, grid())
    assert mesh.n_points == audit['points_per_cube'] and mesh.n_cells == 6
    assert np.allclose(mesh.bounds, [-.2, .2, -.2, .2, -.2, .2])
    assert np.all(mesh['rgb'] == [0, 150, 245])
    assert np.all(mesh['semantic_id'] == 4)
    assert audit['occupied_voxels'] == 1
    assert np.array_equal(sem, before)


def test_adjacent_cubes_color_mapping_no_interpolated_semantics():
    sem = np.full((3, 3, 3), 17, dtype=np.uint8)
    sem[0, 1, 1] = 11; sem[1, 1, 1] = 16; sem[2, 1, 1] = 255
    mesh, audit = renderer.voxel_mesh(sem, grid())
    assert audit['occupied_voxels'] == 2 and mesh.n_cells == 12
    assert {tuple(c) for c in mesh['rgb']} == {(255, 0, 255), (0, 175, 0)}
    sem[:] = 0
    assert renderer.voxel_mesh(sem, grid())[0].n_points == 0


def test_margin_trim_shared_for_sequence_not_individual_zoom():
    from PIL import Image
    a = np.zeros((100, 120, 4), np.uint8); b = a.copy()
    a[20:50, 30:60] = 255; b[40:80, 50:90] = 255
    images, box = renderer.trim_sequence([Image.fromarray(a), Image.fromarray(b)])
    assert images[0].size == images[1].size
    assert box[0] <= 30 and box[1] <= 20 and box[2] >= 90 and box[3] >= 80
    assert np.count_nonzero(np.asarray(images[0])[..., 3]) == 900
    assert np.count_nonzero(np.asarray(images[1])[..., 3]) == 1600


def test_actual_offscreen_rgba_two_cameras_close_windows():
    import pyvista as pv
    from pyvista.plotting.plotter import _ALL_PLOTTERS
    count = len(_ALL_PLOTTERS)
    sem = np.full((3, 3, 3), 17, dtype=np.uint8); sem[1, 1, 1] = 4
    before = pv.global_theme.background
    for camera in renderer.CAMERAS:
        image, audit = renderer.render(sem, grid(), camera=camera, size=(320, 200))
        pixels = np.asarray(image)
        assert pixels.shape == (200, 320, 4) and pixels[..., 3].any()
        assert (pixels[..., 3] == 0).any()
        assert 'VTK' in audit['backend'] and audit['graphics']
    assert len(_ALL_PLOTTERS) == count
    assert pv.global_theme.background == before


def test_export_source_reads_only_and_unknown_future_rejected(tmp_path):
    import json
    source = tmp_path/'assets'; (source/'raw').mkdir(parents=True)
    sem = np.full((4, 3, 3, 3), 17, dtype=np.uint8); sem[:, 1, 1, 1] = 4
    np.savez_compressed(source/'raw/history.npz', semantics=sem)
    meta = dict(protocol='overview_assets_v1', status='complete', grid=grid(),
                window=dict(scene_name='test', history_times_s=[0, .5, 1, 1.5]), future_content='UNKNOWN')
    (source/'manifest.json').write_text(json.dumps(meta))
    sha = renderer.come.digest(source/'raw/history.npz')
    with pytest.raises(ValueError, match='unknown future'):
        renderer.export(source, tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
    renderer.export(source, tmp_path/'result', include_futures=False, size=(320, 200))
    assert renderer.come.digest(source/'raw/history.npz') == sha
    result = json.loads((tmp_path/'result/manifest.json').read_text())
    assert result['geometry']['smoothing'] is False
    assert result['future_content'] is None and len(result['frames']) == 5
    assert all(f['pyvista'] and f['vtk'] and f['graphics'] for f in result['frames'])
    with pytest.raises(FileExistsError):
        renderer.export(source, tmp_path/'result', include_futures=False)
