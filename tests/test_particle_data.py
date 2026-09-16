"""Particle identity from scikit-hep particle and Pythia masses against the codes exHad ships and its C++ quantum-number tables."""
import json
from pathlib import Path
import re
import sys
import unittest
from particle import Particle, ParticleNotFound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exhad import DATA  # noqa: E402
from exhad.core import pdg  # noqa: E402


def shipped_codes():
    """Signed PDG codes of the EventCalc decay tables and the Model-1 cards."""
    codes = set()
    for path in sorted(DATA.glob('*/eventcalc_branching_ratios*.json')):
        codes.update(int(p) for row in json.loads(path.read_text()) for p in row[1] if int(p) != -999)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'pdg_ids':
                    codes.update(int(p) for p in value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    for path in DATA.glob('*/model1*.json'):
        walk(json.loads(path.read_text()))
    return sorted(codes)


class Identity(unittest.TestCase):
    def test_shipped_codes_have_table_identity(self):
        for code in shipped_codes():
            with self.subTest(pdg=code):
                self.assertEqual(pdg.charge(pdg.antiparticle(code)), -pdg.charge(code))
                self.assertEqual(pdg.antiparticle(pdg.antiparticle(code)), code)
                self.assertEqual(pdg.is_self_conjugate(code), pdg.antiparticle(code) == code)
                self.assertEqual(pdg.antiparticle(code), code if pdg.is_self_conjugate(code) else -code)
                self.assertEqual(pdg.is_lepton(code), 11 <= abs(code) <= 16)
                self.assertEqual(pdg.is_quark(code), 1 <= abs(code) <= 6)
                self.assertGreaterEqual(pdg.m0(code), 0.)

    def test_codes_outside_the_table_raise(self):
        for code in (51, 9000005, 4900022):
            with self.assertRaises(ParticleNotFound):
                pdg.charge(code)

    def test_symmetry_filter_tables_equal_particle(self):
        """G, C, J^P and isospin of cpp/src/symmetry_filter.cc; particle lists the D_s* parity as unknown."""
        source = re.sub(r'//[^\n]*', '', (ROOT / 'cpp/src/symmetry_filter.cc').read_text())

        def entries(name, pattern):
            body = re.search(name + r'\s*=\s*\{(.*?)\n\s*\};', source, re.S).group(1)
            return {int(k): tuple(map(int, v)) for k, *v in re.findall(pattern, body)}
        pairs = r'\{\s*(-?\d+)\s*,\s*([+-]?\d+)\s*\}'
        nested = r'\{\s*(-?\d+)\s*,\s*\{\s*(\d+)\s*,\s*([+-]?\d+)\s*\}\s*\}'
        parity_override = {433: -1, -433: -1}
        tables = {name: entries(name, pattern) for name, pattern in
                  (('G_TABLE', pairs), ('C_TABLE', pairs), ('JP_TABLE', nested), ('iso', nested))}
        self.assertEqual([len(t) for t in tables.values()], [10, 14, 106, 104])
        for code, (value,) in tables['G_TABLE'].items():
            self.assertEqual(int(Particle.from_pdgid(code).G), value, code)
        for code, (value,) in tables['C_TABLE'].items():
            self.assertEqual(int(Particle.from_pdgid(code).C), value, code)
        for code, value in tables['JP_TABLE'].items():
            entry = Particle.from_pdgid(code)
            self.assertEqual((int(2 * entry.J), parity_override.get(code, int(entry.P))), value, code)
        for code, value in tables['iso'].items():
            entry = Particle.from_pdgid(code)
            quarks = re.sub(r'sqrt\(\d\)', '', entry.quarks)
            hypercharge = ((1 if code > 0 else -1) if entry.pdgid.is_baryon else 0) + quarks.count('S') - quarks.count('s') \
                + quarks.count('c') - quarks.count('C')
            self.assertEqual((int(2 * entry.I), 2 * entry.three_charge // 3 - hypercharge), value, code)


if __name__ == '__main__':
    unittest.main()
