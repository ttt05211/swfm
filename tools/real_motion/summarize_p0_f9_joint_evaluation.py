#!/usr/bin/env python3
"""Print ALL metrics and existing timing logs; no Torch/GPU/checkpoint load."""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.real_motion.joint_evaluation_reporting import read_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', help='completed evaluation directory or evaluation.json')
    args = parser.parse_args()
    print(read_report(args.evaluation), end='')


if __name__ == '__main__': main()
