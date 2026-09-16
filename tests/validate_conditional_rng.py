"""Same-seed full daughter-record replay with SIMD versus exact scalar RNG.

Called by tools/configure.py before the accelerator is recorded. Both
executables use the same conditional fragmentation and all the same inputs;
only their seed initializer differs. This is not a comparison of different
fragmentation proposals or of different channel probabilities.
"""
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate(config, simd_binary, scalar_binary, events=32):
    from exhad.accelerator.sampler import AlpFermionAccelerator
    from exhad.worker import _sample_reference
    os.environ['EXHAD_RUNTIME_BINARY']=str(ROOT/'cpp/exhad')
    os.environ['EXHAD_PYTHIA8DATA']=config['xmldoc']
    fast, reference=AlpFermionAccelerator(simd_binary), AlpFermionAccelerator(scalar_binary)
    cases=[(m,'central',seed) for m,seed in
           [(2.4001,731),(2.5,1),(3.,2147483647),(3.5,2**64-1),(4.4,5),(4.9,9)]]
    cases += [(3.,v,731) for v in ('p-low','p-high','saturation-off')]
    records=[]
    try:
        for mass, variation, seed in cases:
            request=dict(model='alp-fermion',mass=mass,events=events,variation=variation,seed=seed)
            actual=_sample_reference(config,request,active_pool=fast)
            expected=_sample_reference(config,request,active_pool=reference)
            if actual != expected:
                raise AssertionError(f'SIMD/scalar full-event mismatch at {mass}, {variation}, {seed}')
            for event in actual:
                np.testing.assert_allclose(np.asarray(event)[:,:4].sum(axis=0),
                    [0.,0.,0.,mass],rtol=0,atol=2e-7)
            records.append(dict(mass_gev=mass,variation=variation,seed=seed,events=len(actual)))
        if dict(fast.stats)!=dict(reference.stats):
            raise AssertionError('Different channel selections under RNG replay')
    finally:
        fast.batches.close();reference.batches.close()
    return dict(full_daughter_records_identical=True,momentum_conservation=True,
                cases=records,channel_counts=dict(fast.stats))
