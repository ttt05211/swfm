"""Two independent processes on ONE physical GPU, fixed candidate integer merge."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import parallel as original
from .common import select_source,validate_source
from .run import PROTOCOL,select_jobs,summary
from .recipe import specification

PARALLEL_PROTOCOL='frozen_current_forward_CVstate_range_ALL4369_TWOprocess_ONE_GPU_integer_merge_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._stc_state_range_parent',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="'tools.source_scene_ego_all.run'"
if source.count(old)!=1:raise RuntimeError('original same-GPU child module hook changed')
source=source.replace(old,"'tools.causal_forward_stc_candidate.run'").replace('EGO_ALL','STC_CANDIDATE')
old="sorted(Path(__file__).parent.glob('*.py'))"
new="[*sorted(Path(__file__).parent.glob('*.py')), root/'tools/source_scene_ego_all/parallel.py', root/'tools/source_scene_ego_all/run.py', root/'tools/causal_forward_stc_candidate/run.sh']"
if source.count(old)!=1:raise RuntimeError('original coordinator provenance hook changed')
exec(compile(source.replace(old,new),original.__file__,'exec'),runner.__dict__)
runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL;runner.PARALLEL_PROTOCOL=PARALLEL_PROTOCOL
runner.select_source,runner.validate_source=select_source,validate_source
runner.select_jobs,runner.summary=select_jobs,summary
_merge_states=runner.merge_states
saved_progress=runner.saved_progress


def merge_states(states,keys,source_sha,voxel_count,modality):
    for state in states:
        method=state['contract']['base']['source_contract'].get('evaluation_method')
        if method!=specification():raise RuntimeError('shards do not share the SAME frozen candidate recipe')
    return _merge_states(states,keys,source_sha,voxel_count,modality)


runner.merge_states=merge_states


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
