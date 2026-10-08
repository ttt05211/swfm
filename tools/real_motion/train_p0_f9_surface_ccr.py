#!/usr/bin/env python3
"""Full-data validation of integrated surface CCR before clean joint training."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import signal
import threading
from tools.real_motion import surface_ccr_screen_common as backend
from tools.real_motion.train_p0_f9_height_shared_field import main

if __name__=='__main__':
    stopped=threading.Event()
    def request_stop(signum,frame):
        stopped.set()
        print('Stop requested: finish update and save optimizer/RNG/cursor. Do not kill -9.',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    sys.exit(main(stopped,backend=backend))
