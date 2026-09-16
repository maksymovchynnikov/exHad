"""Particle identity from the PDG table of scikit-hep ``particle``; masses from Pythia 8.317.

Only identity data are taken from ``particle``: electric charge, antiparticle and
classification. Charges come from real table entries, never from decoding the
number, so codes absent from the table raise. Masses are Pythia's m0, the masses
Pythia generates and decays with, from the generated snapshot
data/common/pythia8317_m0.json (tools/snapshot_pythia_masses.py).
"""
from functools import lru_cache
import json
from particle import Particle
from particle import pdgid
from .. import DATA

_M0 = {int(code): mass for code, mass in
       json.loads((DATA / 'common/pythia8317_m0.json').read_text())['m0_gev'].items()}


@lru_cache(maxsize=None)
def _entry(pid):
    return Particle.from_pdgid(int(pid))


def m0(pid):
    """Pythia 8.317 nominal mass of |pid| in GeV."""
    return _M0[abs(int(pid))]


def charge(pid):
    """Electric charge in units of e."""
    return _entry(pid).three_charge / 3


def antiparticle(pid):
    return int(_entry(pid).invert().pdgid)


def is_self_conjugate(pid):
    return bool(_entry(pid).is_self_conjugate)


def is_lepton(pid):
    return bool(pdgid.is_lepton(int(pid)))


def is_quark(pid):
    return bool(pdgid.is_quark(int(pid)))
