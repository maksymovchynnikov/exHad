"""Persistent C++ secondary decayer for externally sampled primary momenta."""
import atexit
from functools import lru_cache
import hashlib
import math
import select
import subprocess
import tempfile
from . import ROOT


class Decayer:
    def __init__(self, binary, xml, *arguments):
        self.errors = tempfile.TemporaryFile(mode='w+t')
        self.process = subprocess.Popen([binary, xml, *arguments], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.errors, text=True, bufsize=1)
        atexit.register(self.close)

    def finish(self, rows, seed):
        output = []

        for index, particles in enumerate(rows):
            pseed = int.from_bytes(hashlib.sha256(f'{seed}:{index}'.encode()).digest()[:8], 'big') % 899900000 + 1
            fields = [str(pseed), str(len(particles))]
            for p in particles:
                fields += [str(int(p[5])), *(str(float(x)) for x in p[:5])]
            self.process.stdin.write(' '.join(fields) + '\n')
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], 60)[0]:
                self.close()
                raise RuntimeError('Secondary-decay worker timed out')
            line = self.process.stdout.readline()
            if not line or line.startswith('ERROR'):
                raise RuntimeError('Secondary-decay worker failed: ' + line)
            values = line.split()
            count = int(values[0])
            particles = [list(map(float, values[i:i + 6])) for i in range(1, len(values), 6)]
            if count < 1 or len(values) != 1 + 6 * count or not all(math.isfinite(x) for p in particles for x in p) \
                    or any(min(p[3], p[4]) < 0 for p in particles):
                raise RuntimeError('Malformed secondary-decay reply (count, finite fields, E >= 0, m >= 0)')
            output.append(particles)

        return output

    def close(self):
        try:  # EOF ends the worker; communicate closes both pipes and reaps the process
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
        self.errors.close()


@lru_cache(maxsize=2)
def decayer(xml, stable=()):
    """The worker of this Pythia XML directory; stable lists PDG codes kept undecayed beyond its defaults."""
    return Decayer(ROOT / '.runtime/exhad-decay', xml, *(['--stable=' + ','.join(map(str, stable))] if stable else []))
