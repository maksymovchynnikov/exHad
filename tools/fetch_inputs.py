#!/usr/bin/env python3
"""Fetch the pinned upstream form-factor implementations; never vendor their git history."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'deliver': ('https://github.com/preimitz/DeLiVeR.git', '7b2bbd79fbacfed9a01d24e903762b4759b9e4a2'),
}


def fetch():
    for name, (url, revision) in SOURCES.items():
        target = ROOT / 'external' / name
        if target.exists():
            actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            changed = subprocess.check_output(['git', '-C', str(target), 'diff', '--name-only', 'HEAD', '--', 'src'], text=True)
            if actual != revision or changed:
                raise ValueError(f'{target} differs from the pinned input; use a clean checkout')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', url, str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)
    print('Pinned phenomenology inputs are present.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch()
