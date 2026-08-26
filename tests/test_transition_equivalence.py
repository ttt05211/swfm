"""Server integration test; skipped locally unless an official transition checkpoint is provided."""
import os, pathlib, subprocess, sys
import pytest

@pytest.mark.integration
def test_official_modified_transition_equivalence():
    ckpt=os.environ.get('OCCFM_TRANSITION_CKPT')
    root=pathlib.Path(__file__).parents[1]
    upstream=root/'upstream_occfm'
    if not ckpt or not upstream.exists():
        pytest.skip('set OCCFM_TRANSITION_CKPT and initialize upstream_occfm')
    cmd=[sys.executable,str(root/'tools/real_motion/check_transition_equivalence.py'),
         '--ckpt',ckpt,'--device','cuda']
    subprocess.run(cmd,check=True,env={**os.environ,'PYTHONPATH':f'{upstream}:{root}'})
