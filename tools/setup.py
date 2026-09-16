#!/usr/bin/env python3
"""Build and configure the checkout using the current Python environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pythia8-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4, help='Parallel compiler processes')
    args = parser.parse_args()

    if not 1 <= args.jobs <= 64:
        parser.error('--jobs must be 1..64')
    prefix = args.pythia8_dir.resolve()

    if not (prefix / 'include/Pythia8/Pythia.h').is_file():
        parser.error('--pythia8-dir does not contain Pythia headers')
    subprocess.run([sys.executable, str(ROOT / 'tools/fetch_inputs.py')], check=True)
    build = ['make', '-C', str(ROOT / 'cpp'),
             f'PYTHIA8_DIR={prefix}', f'PYTHON={sys.executable}']
    subprocess.run([*build, f'-j{args.jobs}'], check=True)
    subprocess.run([*build, 'check'], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/configure.py')], check=True)


if __name__ == '__main__':
    main()
