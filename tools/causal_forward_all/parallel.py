"""TWO shards on ONE GPU, current forward epoch2, exact integer-count merge."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import parallel as original
from .common import select_source,validate_source
from .run import PROTOCOL,select_jobs,summary

PARALLEL_PROTOCOL='frozen_current_forward_AdaLN_epoch2_ALL4369_TWOprocess_ONE_GPU_integer_merge_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._forward_all_parallel',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="'tools.source_scene_ego_all.run'"
if source.count(old)!=1:raise RuntimeError('original same-GPU child module hook changed')
source=source.replace(old,"'tools.causal_forward_all.run'").replace('EGO_ALL','FORWARD_ALL')
old="sorted(Path(__file__).parent.glob('*.py'))"
new="[*sorted(Path(__file__).parent.glob('*.py')), root/'tools/source_scene_ego_all/parallel.py', root/'tools/source_scene_ego_all/run.py', root/'tools/causal_forward_all/run.sh']"
if source.count(old)!=1:raise RuntimeError('original coordinator provenance hook changed')
exec(compile(source.replace(old,new),original.__file__,'exec'),runner.__dict__)
runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL;runner.PARALLEL_PROTOCOL=PARALLEL_PROTOCOL
runner.select_source,runner.validate_source=select_source,validate_source
runner.select_jobs,runner.summary=select_jobs,summary
merge_states,saved_progress=runner.merge_states,runner.saved_progress


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
