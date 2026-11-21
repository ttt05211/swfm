import numpy as np
import pytest
from tools.paper_assets import come


def grid(shape=(2, 2, 2)):
    return dict(x_min=0., y_min=0., z_min=0., voxel_size=[.4]*3, shape_hwd=shape)


def test_come_semantic_ids_not_shifted():
    assert come.COLORS[11].tolist() == [255, 0, 255]
    assert come.COLORS[16].tolist() == [0, 175, 0]
    assert come.COLORS[4].tolist() == [0, 150, 245]
    assert come.COLORS[3].tolist() == [255, 255, 0]


def test_faces_full_height_filter_and_no_mutation():
    sem = np.full((2, 2, 2), 17, dtype=np.uint8)
    sem[0, 0, :] = 11
    before = sem.copy()
    faces, colors = come.cube_faces(sem, grid())
    assert faces.shape == (5, 4, 3)  # two side faces each + one top
    assert faces[..., 2].max() == .8
    assert len(colors) == 5
    assert np.array_equal(sem, before)
    sem[:] = 0
    assert not len(come.cube_faces(sem, grid())[0])
    sem[:] = 255
    assert not len(come.cube_faces(sem, grid())[0])


def test_projection_render_empty_and_fixed_framing(tmp_path):
    sem = np.full((2, 2, 2), 17, dtype=np.uint8)
    bounds = come.grid_bounds(grid())
    empty = come.render(sem, grid(), size=(180, 100), bounds=bounds)
    assert not np.asarray(empty)[..., 3].any()
    sem[0, 0, 0] = 4
    im = come.render(sem, grid(), size=(180, 100), bounds=bounds)
    assert np.asarray(im)[..., 3].any()
    come.save_image(im, tmp_path/'frame_1.5s')
    assert (tmp_path/'frame_1.5s.png').is_file()
    assert (tmp_path/'frame_1.5s_transparent.png').is_file()
    assert (tmp_path/'frame_1.5s.svg').is_file()


def test_shared_bounds_cover_every_frame_without_individual_zoom():
    a = np.full((2, 2, 2), 17, dtype=np.uint8)
    b = a.copy(); a[0, 0, 0] = 11; b[1, 1, 1] = 4
    bounds = come.sequence_bounds([a, b], grid())
    for sem in (a, b):
        xy, _ = come.project(come.cube_faces(sem, grid())[0])
        assert np.all(xy >= bounds[0]-1e-12)
        assert np.all(xy <= bounds[1]+1e-12)


def test_export_labels_future_type_preserves_raw(tmp_path):
    import json
    source = tmp_path/'source'; (source/'raw').mkdir(parents=True)
    row = dict(protocol='overview_assets_v1', status='complete', grid=grid(),
               window=dict(history_times_s=[0, .5, 1, 1.5], scene_name='test'),
               future_content='UNKNOWN')
    (source/'manifest.json').write_text(json.dumps(row))
    np.savez_compressed(source/'raw/history.npz', semantics=np.full((4, 2, 2, 2), 17, np.uint8))
    before = come.digest(source/'raw/history.npz')
    with pytest.raises(ValueError, match='unknown future'):
        come.export(source, tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
    come.export(source, tmp_path/'output', include_futures=False)
    assert come.digest(source/'raw/history.npz') == before
    with pytest.raises(FileExistsError):
        come.export(source, tmp_path/'output', include_futures=False)
