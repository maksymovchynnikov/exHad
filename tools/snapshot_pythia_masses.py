#!/usr/bin/env python3
"""Write data/common/pythia8317_m0.json: the Pythia 8.317 m0 of every |PDG| code exHad's Python side reads.

  python tools/snapshot_pythia_masses.py XMLDOC

The codes are those of every EventCalc decay table (data/*/eventcalc_branching_ratios*.json) and
of the Model-1 cards, the partons and leptons, and the hadrons exHad names (thresholds, open-charm poles).
The file is generated, never edited by hand; tests/test_physics.py checks it against the
live ParticleData.xml, and the unit tests read it without a Pythia build.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exhad import DATA  # noqa: E402
from exhad.core.event_worker import _PARTICLE_BLOCK, _XML_ATTRIBUTE  # noqa: E402

TARGET = DATA / 'common/pythia8317_m0.json'
NAMED = (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 111, 211, 321, 411, 413, 421, 433, 4122, 521)


def codes():
    found = set(NAMED)
    for path in sorted(DATA.glob('*/eventcalc_branching_ratios*.json')):
        found.update(abs(int(p)) for row in json.loads(path.read_text()) for p in row[1] if int(p) != -999)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'pdg_ids':
                    found.update(abs(int(p)) for p in value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    for path in DATA.glob('*/model1*.json'):
        walk(json.loads(path.read_text()))
    return sorted(found)


def snapshot(xmldoc):
    """The snapshot document built from XMLDOC/ParticleData.xml."""
    document = (Path(xmldoc) / 'ParticleData.xml').read_text(encoding='utf-8')
    masses = {}

    for header, _ in _PARTICLE_BLOCK.findall(document):
        attributes = dict(_XML_ATTRIBUTE.findall(header))
        masses[int(attributes['id'])] = float(attributes.get('m0', '0'))

    return {'source': 'Pythia 8.317 ParticleData.xml m0 [GeV], written by tools/snapshot_pythia_masses.py',
            'm0_gev': {str(code): masses[code] for code in codes()}}


def text(document):
    rows = ',\n'.join(f'  {json.dumps(code)}: {mass!r}' for code, mass in document['m0_gev'].items())
    return f'{{"source": {json.dumps(document["source"])},\n "m0_gev": {{\n{rows}\n}}}}\n'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    TARGET.write_text(text(snapshot(sys.argv[1])))
