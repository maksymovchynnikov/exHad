"""make check: the exhad binary links, reads Pythia 8.317 XML and accepts one event-worker event."""
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    xmldoc = Path(os.environ.get('EXHAD_PYTHIA8DATA', '').strip() or sys.exit('set EXHAD_PYTHIA8DATA')).resolve()
    declared = re.search(r'name=["\']Pythia:versionNumber["\'][^>]*default=["\']([^"\']+)',
                         (xmldoc / 'Version.xml').read_text())
    # Pythia refuses to construct when the linked library and the XML versions differ.
    result = subprocess.run([str(ROOT / 'cpp/exhad'), '2', '-1', '-1', '0', 'qq', '--component=rho',
                             '--parentSpin=carried', '--set=HadronLevel:mStringMin=1.0'],
                            input='REQUEST 0 1701 2\nQUIT\n',
                            env=dict(os.environ, PYTHIA8DATA=str(xmldoc)), capture_output=True, text=True)
    lines = result.stdout.splitlines()
    ready = lines[0].split() if lines else []
    if result.returncode or len(ready) != 5 or ready[0] != 'READY':
        sys.exit('exhad binary check failed (linkage or XML mismatch):\n' + result.stdout[-1200:] + result.stderr[-1200:])
    if not declared or float(ready[2]) != float(declared[1]):
        sys.exit(f'Pythia binary/XML mismatch: worker {ready[2]}, XML {declared and declared[1]}')
    if 'RESULT 0 EVENT 1701' not in lines:
        sys.exit('exhad check event was not accepted')
    print(f'exhad installation OK: {ROOT / "cpp/exhad"} (Pythia {declared[1]}), XML {xmldoc}')


if __name__ == '__main__':
    main()
