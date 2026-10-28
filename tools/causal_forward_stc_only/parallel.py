"""Input-specific policies, TWO independent model processes on ONE GPU."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import parallel as original
from .common import select_source,validate_source,specification
from .run import PROTOCOL,select_jobs,summary

PARALLEL_PROTOCOL='frozen_current_OCCraw_STCadapt_ALL4369_TWOprocess_ONE_GPU_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._stc_only_parent',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="'tools.source_scene_ego_all.run'"
if source.count(old)!=1:raise RuntimeError('original same-GPU child module hook changed')
source=source.replace(old,"'tools.causal_forward_stc_only.run'").replace('EGO_ALL','STC_ONLY')
old="sorted(Path(__file__).parent.glob('*.py'))"
new="[*sorted(Path(__file__).parent.glob('*.py')), root/'tools/source_scene_ego_all/parallel.py', root/'tools/source_scene_ego_all/run.py', root/'tools/causal_forward_stc_only/run.sh', root/'tools/causal_forward_stc_candidate/common.py', root/'tools/causal_forward_stc_candidate/method.py', root/'tools/causal_forward_stc_candidate/recipe.py']"
if source.count(old)!=1:raise RuntimeError('original coordinator provenance hook changed')
exec(compile(source.replace(old,new),original.__file__,'exec'),runner.__dict__)
runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL;runner.PARALLEL_PROTOCOL=PARALLEL_PROTOCOL
runner.select_source,runner.validate_source=select_source,validate_source
runner.select_jobs,runner.summary=select_jobs,summary
_merge_states=runner.merge_states
saved_progress=runner.saved_progress


def merge_states(states,keys,source_sha,voxel_count,modality):
    for state in states:
        if state['contract']['base']['source_contract'].get('evaluation_method')!=specification():
            raise RuntimeError('shards do not share the SAME declared input-policy recipe')
    return _merge_states(states,keys,source_sha,voxel_count,modality)


runner.merge_states=merge_states


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
