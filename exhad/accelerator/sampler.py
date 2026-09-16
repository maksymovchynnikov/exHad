"""Active pools of the Model-1 sampler, with batched native transport (exhad-batch).

model1.sampler.sample_deployment draws the owners, external rows and conditioned
families. Its Pythia-pool events come here: the alp-fermion eta-pipi family and charge
channel, or the scalar four-pion family, are drawn from the deployed cards; every
other family (and every B-L vector-current event) keeps the ordinary matched
rejection over envelope E' realized by the join-decision string sampler. Native
code checks canonical ancestry, matching, terminal mass shells and conservation.
See README.md.
"""
import atexit
from collections import Counter
import math
from pathlib import Path
import os
import subprocess
import tempfile
import threading
import numpy as np
from ..core.event_worker import WorkerPool
from ..core.eventcalc import decay, decay_rng
from ..model1 import sampler as model1
from ..model1.families import classify_channel

CHANNELS = {'-211:1 211:1 221:1', '111:2 221:1', '-211:1 211:1 331:1', '111:2 331:1'}


class _Batch:
    """One exhad-batch process of one portal, configured at one mass point and variation."""

    def __init__(self, binary, deployment, point, excluded, portal):
        self.errors = tempfile.TemporaryFile(mode='w+t')

        with tempfile.TemporaryDirectory(prefix='exhad-conditional-batch-') as temp:  # read before READY
            settings = Path(temp) / 'settings.txt'
            settings.write_text('\n'.join(deployment.generator_pythia_settings) + '\n')
            self.process = subprocess.Popen([str(binary), str(point.mass_gev), str(settings), portal],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors, text=True,
                bufsize=1024 * 1024, env=dict(os.environ, PYTHIA8DATA=str(deployment.xmldoc_path)))
            ready = self.process.stdout.readline().strip()

        if ready != f'READY CONDITIONAL-TERMINALS {point.mass_gev:.17g}':
            raise RuntimeError('invalid batched native startup: ' + self.error())
        matching = deployment.conditional_matching
        conditional = {} if matching is None else matching.at(point.mass_gev)[0]
        budget = model1._point_rejection_attempt_budget(point, excluded)
        envelope = model1._joint_rejection_envelope(point, excluded, conditional, budget['envelope'])
        lines = [f'CONFIG {len(point.source_weights)}']

        for source, configuration in model1._proposal_filters(deployment, point, excluded, envelope, conditional).items():
            lines += [f'{source} {point.source_weights[source]:.17g} {len(configuration)}', *configuration]
        self.process.stdin.write('\n'.join(lines) + '\n')
        self.process.stdin.flush()

        if self.process.stdout.readline().strip() != 'CONFIGURED':
            raise RuntimeError('native configuration failed: ' + self.error())

    def error(self):
        self.errors.seek(0)
        return self.errors.read()[-3000:]

    def draw(self, requests):
        timer = threading.Timer(300., self.process.kill)
        timer.start()

        try:
            self.process.stdin.write(f'DRAW {len(requests)}\n' + '\n'.join(requests) + '\n')
            self.process.stdin.flush()
            result = []
            for index in range(len(requests)):
                header = self.process.stdout.readline().strip().split(maxsplit=4)
                if len(header) != 5 or header[:2] != ['EVENT', str(index)]:
                    raise RuntimeError('invalid native terminal header: ' + self.error())
                count, source, channel = int(header[2]), header[3], header[4]
                if not 1 <= count <= 10000:
                    raise RuntimeError('unbounded native terminal record')
                particles = []
                for _ in range(count):
                    values = list(map(float, self.process.stdout.readline().split()))
                    if len(values) != 5 or not all(math.isfinite(x) for x in values):
                        raise RuntimeError('invalid native particle record')
                    px, py, pz, e, pdg = values
                    m2 = e * e - px * px - py * py - pz * pz
                    if m2 < -1e-8:
                        raise RuntimeError('spacelike native terminal')
                    particles.append([px, py, pz, e, float(np.sqrt(max(0., m2))), pdg])
                result.append((particles, source, channel))
            if self.process.stdout.readline().strip() != f'END {len(requests)}':
                raise RuntimeError('invalid terminal batch terminator')
            return result
        finally:
            timer.cancel()

    def close(self):
        try:  # communicate closes both pipes and reaps the process
            self.process.communicate('QUIT\n' if self.process.returncode is None else None, timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.communicate()
        self.errors.close()


class _Pool:
    """Leased exhad-batch processes of one portal; ``stats`` counts the native draws."""

    portal = ''

    def __init__(self, binary):
        self.binary, self.stats, self.batches = binary, Counter(), WorkerPool(4)
        atexit.register(self.batches.close)

    def generate(self, deployment, point, variation, requests, excluded=frozenset()):
        with self.batches.use((self.portal, point.mass_gev, variation), lambda: _Batch(
                self.binary, deployment, point, excluded, self.portal)) as batch:
            return batch.draw(requests)

    @staticmethod
    def rows(master, seeds, generated):
        """Six-field records on one EventCalc stable-decay stream."""
        rng, rows = decay_rng(master), {}
        for index, (particles, _, _) in zip(seeds, generated):
            rows[index] = [value for particle in particles for daughter in decay(particle, rng) for value in daughter]
        return rows


class VectorAccelerator(_Pool):
    """The ``active_pool`` of sample_deployment for the unweighted B-L vector current."""

    portal = 'b-l'

    def __call__(self, deployment, point, variation, master, seeds):
        model = deployment.family_model

        if model.contract.classifier_id != 'b-l-resolved-families':
            raise ValueError('uncertified family basis')

        if model1._conditioned_families(model) or deployment.conditional_matching is not None:
            raise ValueError('uncertified boundary-conditioned deployment')
        requests = []

        for seed in seeds.values():
            self.stats['active'] += 1
            requests.append(f'{seed} 0')
        generated = self.generate(deployment, point, variation, requests)

        for _, source, channel in generated:
            self.stats[f'0 {source}'] += 1

        return self.rows(master, seeds, generated)


class AlpFermionAccelerator(_Pool):
    """The ``active_pool`` of sample_deployment for unweighted alp-fermion."""

    portal = 'alp-fermion'

    def __call__(self, deployment, point, variation, master, seeds):
        model, matching, mass = deployment.family_model, deployment.conditional_matching, point.mass_gev

        if deployment.generator_scenario_id != 'central':
            raise ValueError('uncertified conditional tune')

        if model.contract.classifier_id != 'alp-fermion-exact-families':
            raise ValueError('uncertified family basis')
        conditioned = model1._conditioned_families(model)
        eta = point.family_probabilities['eta-pipi'] / math.fsum(
            p for f, p in point.family_probabilities.items() if f not in conditioned)
        proposal, _ = matching._proposal(matching.families['eta-pipi'], mass)
        factors, _ = matching.at(mass)
        channels = {key: proposal[key] * factors['eta-pipi'][key] for key in proposal}

        if set(channels) != CHANNELS:
            raise ValueError('uncertified primary channel basis')

        if not math.isclose(sum(channels.values()), 1., abs_tol=2e-12):
            raise ValueError('invalid charge probabilities')
        requests, expected = [], []

        for seed in seeds.values():
            self.stats['active'] += 1
            if model1._unit(seed ^ 0x923e0a1b57c6d48f) >= eta:
                key, request = None, f'{seed} 0'
                self.stats['ordinary_other_families'] += 1
            else:  # native mode 1 forces the three primaries
                key = model1._draw(channels, seed ^ 0xca036e2d4178bf59)
                request = f'{model1.splitmix64(seed ^ 0x73d1f09a842b6ec5)} 1 ' + ' '.join(
                    map(str, model1._parse_removed_key(key)))
                self.stats[f'1 {key}'] += 1
            requests.append(request)
            expected.append(key)
        generated = self.generate(deployment, point, variation, requests, conditioned | {'eta-pipi'})

        for (_, source, channel), wanted in zip(generated, expected):
            if wanted is not None and (channel != wanted or source != 'glue'):
                raise ValueError(f'conditional channel/source mismatch: {source}, {channel}, expected {wanted}')

        return self.rows(master, seeds, generated)


class ScalarAccelerator(_Pool):
    """The ``active_pool`` for unweighted scalar portals: rate-first four-pion (gluon MENU script)."""

    portal = 'scalar'

    def __call__(self, deployment, point, variation, master, seeds):
        model = deployment.family_model

        if deployment.generator_scenario_id != 'central':
            raise ValueError('uncertified conditional tune')

        if model.contract.classifier_id != 'scalar-families' or deployment.conditional_matching is not None:
            raise ValueError('uncertified family basis')
        conditioned = model1._conditioned_families(model)
        four = point.family_probabilities['four-pion'] / math.fsum(
            p for f, p in point.family_probabilities.items() if f not in conditioned)
        requests, wanted = [], []

        for seed in seeds.values():
            self.stats['active'] += 1
            wanted.append(model1._unit(seed ^ 0x3c6ef372fe94f82b) < four)
            if wanted[-1]:  # native mode 1 forces a four-pion primary history
                requests.append(f'{model1.splitmix64(seed ^ 0xa54ff53a5f1d36f1)} 1')
                self.stats['1 four-pion'] += 1
            else:
                requests.append(f'{seed} 0')
                self.stats['ordinary_other_families'] += 1
        generated = self.generate(deployment, point, variation, requests, conditioned | {'four-pion'})

        for (_, source, channel), four_pion in zip(generated, wanted):
            if four_pion and (source != 'glue' or classify_channel(
                    'scalar-families', model1._parse_removed_key(channel)) != 'four-pion'):
                raise ValueError(f'conditional four-pion mismatch: {source}, {channel}')

        return self.rows(master, seeds, generated)
