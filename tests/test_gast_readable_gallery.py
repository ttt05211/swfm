import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.paper_assets import gast_readable_gallery as gallery
from tools.paper_assets import qualitative_render as layout
from tools.gast_baseline.common import sha


def sample_case():
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = np.arange(1, 7) * 5.
    case = dict(t0_token='anchor', future_ego_poses=poses.tolist(), report_indices=[1, 3, 5])
    arrays = {name: np.full((3, 2, 2, 2), 17, np.uint8) for name in ('gt', 'g', 'e')}
    arrays['t0'] = np.full((2, 2, 2), 17, np.uint8)
    return case, arrays


def test_fixed_frame_exposes_vehicle_motion_and_keeps_static_world_point_stationary():
    case, arrays = sample_case()
    before = {k: v.copy() for k, v in arrays.items()}
    rows, columns, settings, frame = gallery.comparison_rows('short01', arrays, case, 'motion')
    assert columns == ['1 s', '2 s', '3 s'] and frame == 'fixed 1s ego reference'
    for _, panels in rows:
        transforms = np.array([settings[id(p)]['display_transform'] for p in panels])
        np.testing.assert_allclose(transforms[:, 0, 3], [0, 10, 20])
        # A world-static landmark has native x=20/10/0; it stays x=20.
        assert [float((t @ [x, 0, 0, 1])[0]) for t, x in zip(transforms, [20, 10, 0])] == [20, 20, 20]
        # A vehicle keeping x=20 relative to ego moves in the fixed frame.
        assert [float((t @ [20, 0, 0, 1])[0]) for t in transforms] == [20, 30, 40]
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])
    native = gallery.comparison_rows('short01', arrays, case, 'native')
    assert native[1] == ['t0', '1 s', '2 s', '3 s'] and native[2] == {}


def test_input_pose_required_to_include_t0_in_the_same_fixed_frame():
    case, arrays = sample_case()
    pose = dict(t0_token='anchor', pose=np.eye(4).tolist())
    rows, columns, settings, _ = gallery.comparison_rows('short01', arrays, case, 'motion', pose)
    assert columns == ['t0', '1 s', '2 s', '3 s']
    assert len({id(panels[0]) for _, panels in rows}) == 1
    for _, panels in rows:
        np.testing.assert_allclose(settings[id(panels[0])]['display_transform'][:3, 3], [-10, 0, 0])
    with pytest.raises(ValueError, match='token'):
        gallery.comparison_rows('short01', arrays, case, 'motion', dict(pose, t0_token='wrong'))


def test_pose_sidecar_checks_exact_preparation_and_does_not_rewrite_existing_export(tmp_path):
    case, _ = sample_case()
    frames = [dict(token='past', pose=np.eye(4).tolist()) for _ in range(4)]
    frames[3]['token'] = 'anchor'
    prepared = tmp_path/'prepared.json'
    prepared.write_text(json.dumps(dict(format='GAST_CPU_prepared_v1', cases={'short01': dict(original=case, frames=frames)})))
    out = tmp_path/'display_poses.json'
    gallery.export_input_poses(prepared, out)
    manifest = dict(receipt=dict(prepared_sha256=sha(prepared)), cases={'short01': case})
    assert gallery.checked_input_poses(out, manifest)['short01']['t0_token'] == 'anchor'
    before = sha(out)
    gallery.export_input_poses(prepared, out)
    assert sha(out) == before
    with pytest.raises(ValueError, match='exact prepared'):
        gallery.checked_input_poses(out, dict(manifest, receipt=dict(prepared_sha256='wrong')))
    bad = copy.deepcopy(manifest); bad['cases']['short01']['t0_token'] = 'wrong'
    with pytest.raises(ValueError, match='token'):
        gallery.checked_input_poses(out, bad)


def test_common_crop_retains_all_methods_extent_and_identical_shared_input(monkeypatch):
    monkeypatch.setattr(layout.vtk_render, 'COME_REFERENCE_SIZE', (240, 120))
    seen = []
    def render(sem, grid, **kwargs):
        marker = int(sem[0, 0, 0]); seen.append(marker)
        image = Image.new('RGBA', (240, 120), (0, 0, 0, 0))
        x = 40+marker*45
        ImageDraw.Draw(image).rectangle((x, 40, x+4, 60), fill=(30+marker*30, 80, 170, 255))
        audit = dict(fitted_camera=[], view_angle=40., camera_clipping_range=[10, 222.9], automatic_camera_fit=False)
        return image, audit
    monkeypatch.setattr(layout.vtk_render, 'render', render)
    shared = np.zeros((1, 1, 1), np.uint8)
    rows = [(name, [shared, np.full((1, 1, 1), i, np.uint8)])
            for i, name in enumerate(('GT', 'GAST', 'CarryWorld'), 1)]
    picture, audit = layout.render_grid(rows, ['t0', '1 s'], {}, (240, 135), 'come_reference',
                                      preview=True, trim_margins=True, fit_content=True)
    assert seen == [0, 1, 2, 3]
    left, top, right, bottom = audit['shared_pixel_crop']
    assert left <= 40 and right >= 179 and top <= 40 and bottom >= 61
    assert audit['automatic_margin_crop'] and audit['content_aspect_panels']
    h = audit['panel_height']; values = np.asarray(picture)
    panels = [values[59+i*(h+9):59+i*(h+9)+h, 81:321] for i in range(3)]
    assert all(np.array_equal(panels[0], p) for p in panels[1:])
