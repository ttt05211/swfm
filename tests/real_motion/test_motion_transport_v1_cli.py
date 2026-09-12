import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TOOLS=(
    'build_motion_transport_v1_manifest.py',
    'preflight_motion_transport_v1.py',
    'accept_motion_transport_v1_ddp.py',
    'profile_motion_transport_v1.py',
    'train_motion_transport_v1.py',
    'eval_motion_transport_v1.py',
)

def test_all_motion_transport_v1_cli_imports_and_help():
    for name in TOOLS:
        path=ROOT/'tools'/'real_motion'/name
        proc=subprocess.run([sys.executable,str(path),'--help'],cwd=ROOT,text=True,capture_output=True,timeout=30)
        assert proc.returncode==0, f'{name} failed import/help:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
        assert 'usage:' in proc.stdout.lower()
