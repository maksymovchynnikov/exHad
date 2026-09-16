"""Persistent worker processes keep Pythia and external-module state out of the host."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from . import ROOT
from .models import check_numbers, check_request, validate_model


def default_workers():
    """One worker per physical core available to this process, at most 8."""

    try:  # Linux: distinct cores among the CPUs this process may use
        cores = len({Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()
                     for cpu in os.sched_getaffinity(0)})
    except (AttributeError, OSError):
        try:  # macOS
            cores = int(subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], capture_output=True,
                                       text=True, check=True).stdout)
        except (OSError, ValueError, subprocess.CalledProcessError):
            cores = os.cpu_count() or 1

    return max(1, min(cores, 8))


class Generator:
    """Generate rest-frame decays of one model in persistent worker processes.

    Each particle has ``[px, py, pz, E, mass, PDG]`` in GeV. Events are
    variable-length lists of particles, with no padding and no detector cuts.

    A request is split into chunks of chunk_size events, generated concurrently
    by persistent worker processes: by default one per physical core (at most
    8); workers=1 uses a single process. A request of at most chunk_size events
    is one chunk with the request seed; a larger one gives each chunk a seed
    derived from the seed, the event count and the chunk index. A seed thus
    reproduces a batch at fixed model/mass/count/chunk_size/runtime, for
    every worker count. Weighted results retain raw weights: normalize only after
    merging the complete requested sample. Instances are context managers, not
    thread-safe shared services.
    """

    def __init__(self, model, *, workers=None, chunk_size=512, variation='central',
                 execution='auto', root=None, python=None):
        for name, value in (('workers', 1 if workers is None else workers), ('chunk_size', chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

        if workers is not None and workers > 64:
            raise ValueError('workers must not exceed 64')

        if execution not in {'auto', 'reference'}:
            raise ValueError('execution must be auto or reference')
        self.model, self.variation = validate_model(str(model)), str(variation)
        self.workers, self.chunk_size = workers or default_workers(), chunk_size
        self._options = dict(root=root, python=python, variation=self.variation, execution=execution)
        self._workers = []
        self._executor = None
        self._closed = False

    @staticmethod
    def _seed(seed, count, index):
        payload = f'exHad-parallel-v1:{seed}:{count}:{index}'.encode('ascii')
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')

    def generate(self, mass, events, *, seed=1, mixing=None):
        """The decays that contain hadrons: the hadronic decays of a boson portal, and the
        semileptonic decays of hnl, which requires mixing (a charged lepton or a neutrino
        together with hadrons: the rows of its complete decay with a hadron or a quark)."""
        return self._run('generate', mass, events, seed, **({} if mixing is None else {'mixing': mixing}))

    def generate_all(self, mass, events, *, seed=1, mixing=None,
                     weighted=False, weight_floor_fraction=0.):
        """Generate every decay channel, including the leptonic and invisible modes.

        Returns a dict with events and channel_labels. For HNL, mixing must
        contain the relative squared mixings [|U_e|^2, |U_mu|^2, |U_tau|^2].
        Weighted sampling is alp-fermion only and retains explicit weight metadata.
        """
        return self._run('generate_all', mass, events, seed, mixing=mixing,
                         weighted=weighted, weight_floor_fraction=weight_floor_fraction)

    def generate_rows(self, mass, rows, *, seed=1, terminal='pythia', mixing=None):
        """Complete decays of named explicit rows of the model's EventCalc-format decay table.

        rows maps row labels of model_info(model)['tables']['decay'] to event counts; the
        result maps each label to its events, in that order. The host owns the rates: a row
        is realized at any mass where its particles are open, and partonic (Jets) rows raise.
        A label's events depend only on seed, label and count. terminal='pythia' decays the
        unstable daughters with Pythia; 'matched' keeps pi0 and K_S stable there and applies
        the pi0 -> gamma gamma and K_S convention of matched events. Taus keep full Pythia
        decays under both. HNL requires mixing, which weights a row's flavour and
        charge-conjugate entries by their widths, or by the mixing alone where the
        table gives the row no width at this mass.
        """
        return self._run('generate_rows', mass, 0, seed, rows=rows, terminal=terminal, mixing=mixing)

    def generate_weighted(self, mass, events, *, seed=1, weight_floor_fraction=0.):
        """alp-fermion proposal events with explicit raw importance weights.

        Normalize the fragmentation group's weights over the full analysis
        sample; leave external-owner weights at one. Never analyze the raw
        event counts as if these were unweighted physical events.

        A floor fraction in [0,1] optionally rejects low-weight proposals
        with exact compensating weights. Zero retains all positive-weight
        proposals; one reproduces the unweighted selection.
        """
        return self._run('generate_weighted', mass, events, seed,
                         weight_floor_fraction=weight_floor_fraction)

    def hadronize_hnl(self, mass, pdgs, primary_events, *, seed=1):
        """Replace only the q-qbar subsystem; retain the sampled spectator.

        Each primary particle uses EventCalc's eight fields
        [px, py, pz, E, mass, PDG, charge, stability], in the supplied PDG order.
        Rows are flat lists of 24 values. HNL partial widths remain host-owned.
        """
        return self._run('hadronize_hnl', mass, len(primary_events), seed,
                         pdgs=pdgs, primary=primary_events)

    def _run(self, method, mass, count, seed, *, pdgs=None, primary=None, **options):
        if self._closed:
            raise RuntimeError('Generator is closed')
        check_numbers(mass, count, seed)  # chunked requests check these first
        check_request(self.model, method, mass, count, seed, **options)
        rows = options.pop('rows', None)
        # Even an empty request is validated by the selected model's worker.
        jobs = ([(i, start, min(self.chunk_size, count - start), None)
                 for i, start in enumerate(range(0, count, self.chunk_size))] if rows is None else
                [(f'{label}:{i}', start, min(self.chunk_size, n - start), label) for label, n in rows.items()
                 for i, start in enumerate(range(0, max(n, 1), self.chunk_size))]) or [(0, 0, 0, None)]
        width = min(self.workers, len(jobs))
        available = Queue()

        def run_chunk(job):
            index, start, n, label = job
            worker = available.get()

            try:
                # A row label's events depend on seed, label and count alone, never on
                # how many labels the request carries, so its chunks are always seeded.
                chunk_seed = (seed if len(jobs) == 1 and label is None
                              else self._seed(seed, count if label is None else rows[label], index))
                if method == 'generate_rows':
                    return worker.generate_rows(mass, {} if label is None else {label: n},
                                                seed=chunk_seed, **options).get(label, [])
                if method == 'hadronize_hnl':
                    return worker.hadronize_hnl(mass, pdgs, primary[start:start + n], seed=chunk_seed)
                return getattr(worker, method)(mass, n, seed=chunk_seed, **options)
            finally:
                available.put(worker)

        try:
            while len(self._workers) < width:
                self._workers.append(Worker(self.model, **self._options))
            for worker in self._workers[:width]:
                available.put(worker)
            if width == 1:
                results = list(map(run_chunk, jobs))
            else:
                # Dynamic scheduling keeps a slow rejection chunk from pinning later
                # chunks to the same worker; output stays in chunk order.
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self.workers)
                futures = [self._executor.submit(run_chunk, job) for job in jobs]
                wait(futures)  # every worker is idle again before a failed chunk raises
                results = [future.result() for future in futures]
        except BaseException as error:
            if not isinstance(error, Exception):  # an interrupt can leave a worker reply unread
                self.close()
            raise
        if method in ('generate', 'hadronize_hnl'):
            return [event for events in results for event in events]
        if method == 'generate_rows':
            merged = {label: [] for label in rows}
            for (*_, label), events in zip(jobs, results):
                if label is not None:  # the validating job of an empty request carries no label
                    merged[label].extend(events)
            return merged
        merged = dict(results[0])
        for key in ('events', 'channel_labels', 'raw_weights', 'normalization_groups', 'family_labels'):
            if key in merged:
                merged[key] = [item for values in results for item in values[key]]
        return merged

    def close(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for worker in self._workers:
            worker.close()
        self._workers.clear()

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Generator is closed')
        return self

    def __exit__(self, *_):
        self.close()


class Worker:
    """One worker process; its requests are chunks of a Generator request."""

    def __init__(self, model, *, root=None, python=None, variation='central', execution='auto'):
        self.model, self.variation, self.execution = model, variation, execution
        self.root = Path(root or ROOT).resolve()
        config = self.root / '.runtime/current.json'

        if not config.is_file() or 'conditional_acceleration' not in json.loads(config.read_text()):
            raise RuntimeError('Build exHad and run python tools/configure.py first')
        self._errors = tempfile.TemporaryFile(mode='w+t')
        self._process = subprocess.Popen(
            [python or sys.executable, str(self.root / 'exhad/worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errors,
            text=True, bufsize=1, cwd=self.root,
        )

    def generate(self, mass, events, *, seed, mixing=None):
        request = self._base(mass, events, seed)
        if mixing is not None:
            request.update(mixing=mixing)
        return self._request(request, events)

    def generate_all(self, mass, events, *, seed, mixing, weighted, weight_floor_fraction):
        request = self._base(mass, events, seed)
        if weighted:
            request.update(weighted=True, weight_floor_fraction=weight_floor_fraction)
        request.update(all_decays=True, mixing=mixing)
        return self._request(request, events, full_reply=weighted, all_decays=True)

    def generate_weighted(self, mass, events, *, seed, weight_floor_fraction):
        request = dict(self._base(mass, events, seed), weighted=True,
                       weight_floor_fraction=weight_floor_fraction)
        return self._request(request, events, full_reply=True)

    def generate_rows(self, mass, rows, *, seed, terminal, mixing):
        request = dict(self._base(mass, sum(rows.values()), seed), rows=dict(rows),
                       terminal=terminal, mixing=mixing)
        events = iter(self._request(request, request['events']))
        return {label: [next(events) for _ in range(count)] for label, count in rows.items()}

    def _base(self, mass, events, seed):
        return dict(model=self.model, variation=self.variation, mass=float(mass),
                    events=events, seed=seed, execution=self.execution)

    def hadronize_hnl(self, mass, pdgs, primary_events, *, seed):
        request = dict(model='hnl', variation=self.variation, mass=float(mass),
                       pdgs=list(map(int, pdgs)), primary_events=primary_events,
                       seed=seed)
        return self._request(request, len(primary_events))

    def _request(self, request, events, *, full_reply=False, all_decays=False):
        if self._process.poll() is not None:
            raise RuntimeError('exHad worker has exited; construct a new Generator')
        self._process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self._process.stdin.flush()
        line = self._process.stdout.readline()

        if not line:
            self._errors.seek(0)
            raise RuntimeError('exHad worker failed:\n' + self._errors.read()[-8000:])
        reply = json.loads(line)

        if 'error' in reply:
            raise RuntimeError(reply['error'])

        if len(reply['events']) != events:
            raise RuntimeError('exHad returned the wrong event count')

        if all_decays and (reply.get('decay_scope') != 'all'
                           or len(reply.get('channel_labels', [])) != events):
            raise RuntimeError('Full-decay metadata has the wrong scope or count')

        if full_reply and not (
                all(len(reply.get(key, [])) == events for key in ('raw_weights', 'normalization_groups', 'family_labels'))
                and all(math.isfinite(w) and w > 0 for w in reply['raw_weights'])
                and set(reply['normalization_groups']) <= {'fragmentation', 'external'}
                and all(w == 1. for w, g in zip(reply['raw_weights'], reply['normalization_groups']) if g == 'external')
                and reply.get('weight_convention') == 'self-normalized-within-fragmentation; external-unit-weight'):
            raise RuntimeError('invalid importance-weight metadata (finite positive weights, '
                               'fragmentation/external groups, unit external weights)')

        return reply if full_reply or all_decays else reply['events']

    def close(self):
        process = self._process

        try:  # EOF ends the worker; communicate closes both pipes and reaps the process
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        self._errors.close()
