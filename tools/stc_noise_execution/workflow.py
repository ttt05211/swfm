"""Resume the exact saved workflow arguments; finished clean arm is skipped."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
from tools.stc_noise_adaptation import workflow as original
from tools.shared_history_experiments.actions.validation.workflow import launch_child


def main(event=None, argv=None, launch=None):
    event = event or threading.Event()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv); root = Path(a.out_dir).resolve()
    saved = json.loads((root/'workflow.json').read_text(encoding='utf-8'))
    args = []
    for key, value in saved['args'].items():
        if key == 'override':
            for v in value or []: args += ['--override', v]
        elif isinstance(value, bool):
            if value: args += ['--'+key.replace('_', '-')]
        elif value is not None: args += ['--'+key.replace('_', '-'), str(value)]
    args += ['--out-dir', str(root), '--resume-workflow']
    if a.max_updates: args += ['--max-updates', str(a.max_updates)]
    def child(command, check):
        command = list(command)
        at = command.index('tools.stc_noise_adaptation.run')
        command[at] = 'tools.stc_noise_execution.run'
        return launch(command, check=check) if launch else launch_child(command, event)
    # Original input/implementation/STC/initializer comparisons remain intact.
    # New execution provenance is recorded separately, not forged as old code.
    return original.main(event, args, launch=child)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
