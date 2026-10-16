# exHad

exHad generates decays of GeV-scale feebly coupled particles into complete
final states, including all hadrons. It follows the decay model of *Rethinking
search signatures: Hadronic decays of GeV-scale feebly coupled particles*,
[arXiv:2609.16104](https://arxiv.org/abs/2609.16104).

For each decay, exHad returns the identities and four-momenta of all final
particles in the rest frame of the decaying particle; exHad does not simulate
the production of these particles or their decay positions. A simulation
program boosts these particles to the laboratory frame, and programs that use
[Pythia8](https://pythia.org) ([source](https://gitlab.com/Pythia8/releases))
can call exHad as a Pythia decay handler.
[EventCalc-SHiP](https://github.com/maksymovchynnikov/EventCalc-SHiP) contains a
working integration. The physics of each particle and of its decays is described in
[`docs/PHYSICS.md`](docs/PHYSICS.md).

## Particle models and their mass ranges

The particle model determines the decaying particle and the calculations and
measurements that fix its decay rates. It is selected in Python, in the C++
interface and with the command-line option `--model`. These calculations and
measurements give two kinds of rate. The first is the width of a single
exclusive channel, a decay into a fixed set of hadrons such as
$`\pi^+\pi^-\pi^0`$. The second is an inclusive width: the width of a
decay into a quark-antiquark pair or a gluon pair, for `dark-photon` and `b-l`
also the width into charmed hadrons, and for `hnl` the width of a decay into a
charged lepton or a neutrino together with a quark-antiquark pair. An inclusive
width fixes how often such a decay happens and says nothing about which hadrons
come out. Above a mass that depends on the particle, both kinds contribute to
the hadronic width at once: exHad generates the hadrons of the inclusive part
and fixes their composition, following the decay model described in
[`docs/PHYSICS.md`](docs/PHYSICS.md).

| Particle model | Decaying particle | Calculations and measurements of the exclusive and inclusive decay rates | Masses at which exHad generates decays |
|---|---|---|---|
| `dark-photon` | Dark photon $`A'`$, a vector boson coupled to the electromagnetic current | Exclusive hadronic widths from the form factors of [arXiv:1801.04847](https://arxiv.org/abs/1801.04847) and [arXiv:2201.01788](https://arxiv.org/abs/2201.01788); the width into hadrons made of $`u`$, $`d`$ and $`s`$ quarks from perturbative QCD to fourth order in the strong coupling, as in arXiv:2201.01788; from 2 GeV the widths of seven multipion and $`K\bar K\pi\pi`$ final states from fits to BaBar and BESIII cross sections; the width into charmed hadrons from the hadronic $`e^+e^-`$ cross section measured by BES, divided among the final states measured by BaBar and BESIII | 0.003 to 5 GeV |
| `b-l` | Gauge boson $`Z'`$ coupled to the difference between baryon number and lepton number | Exclusive hadronic widths from the quark flavour decomposition of the form factors of [arXiv:1801.04847](https://arxiv.org/abs/1801.04847), [arXiv:1911.11147](https://arxiv.org/abs/1911.11147) and [arXiv:2201.01788](https://arxiv.org/abs/2201.01788); the width into hadrons made of $`u`$, $`d`$ and $`s`$ quarks from the QCD corrections to fourth order in the strong coupling of arXiv:2201.01788; the width into charmed hadrons from that of `dark-photon`, scaled by the squared ratio of the charm quark charges under the two currents | 0.001 to 5 GeV |
| `alp-fermion` | Axion-like particle $`a`$ coupled at the scale $`\Lambda = 1`$ TeV to the axial currents of all fermions with equal strength and with zero coupling to gluons | Exclusive hadronic widths and the widths into gluon, strange quark, charm quark and nucleon pairs of [arXiv:2501.04525](https://arxiv.org/abs/2501.04525) | 0.01 to 5 GeV |
| `scalar-central`, `scalar-lower`, `scalar-upper`, `scalar-1809` | Scalar $`S`$ that mixes with the Higgs boson | Four calculations of the hadronic widths of one particle. `scalar-central`, `scalar-lower` and `scalar-upper` take the dispersive calculation of the pion and kaon form factors of [arXiv:2407.13587](https://arxiv.org/abs/2407.13587), as the mean of its Monte Carlo sample of widths and one standard deviation below and above that mean; `scalar-1809` takes the calculation of [arXiv:1809.01876](https://arxiv.org/abs/1809.01876) with the corrections and additional channels of [arXiv:1904.10447](https://arxiv.org/abs/1904.10447). The four differ in the widths into $`\pi\pi`$ and $`K\bar K`$ ([`docs/PHYSICS.md`](docs/PHYSICS.md#hadronic-widths-of-the-four-input-calculations)) | 0.01 to 63 GeV |
| `hnl` | Heavy neutral lepton $`N`$, a Majorana fermion that mixes with the three Standard Model neutrinos | Decay widths and matrix elements of [arXiv:1805.08567](https://arxiv.org/abs/1805.08567), with the exclusive channels matched to the inclusive quark-level widths separately for each weak current, quark flavour combination and mixing flavour | 0.02 to 40 GeV |

*The scalar models* means `scalar-central`, `scalar-lower`, `scalar-upper` and
`scalar-1809`. The complete list of the calculations and measurements behind
these decay rates is in
[`docs/PHYSICS.md`](docs/PHYSICS.md#input-calculations-and-data).

The last column gives the masses at which exHad generates samples of all decay
channels (`--all-decays`). A sample of the decays that contain hadrons begins at
the threshold of that particle's lightest such channel: 0.135 GeV for
`dark-photon` and `b-l` ($`\pi^0\gamma`$), 0.270 GeV for the scalar models
($`\pi^0\pi^0`$) and 0.279 GeV for `alp-fermion` ($`\pi^+\pi^-\gamma`$). Every decay
of $`N`$ produces a charged lepton or a neutrino, so a sample of the decays of
`hnl` that contain hadrons consists of its semileptonic decays, a charged lepton
or a neutrino together with hadrons, from 0.135 GeV ($`\pi^0\nu`$).

A request at a mass outside these ranges stops with an error; exHad does not
extrapolate decay rates or final states.

Inelastic dark matter is available through the separate
[Python interface below](#inelastic-dark-matter). It describes the decay of an
excited dark fermion into a lighter dark fermion and visible particles through
an off-shell dark photon. Its range is specified by the mass splitting.

## Known bugs/problems

For `alp-fermion`, the input squared matrix elements from the calculation of
[arXiv:2501.04525](https://arxiv.org/abs/2501.04525) can be negative in physical
phase space for $`\pi^+\pi^-\pi^0`$, $`3\pi^0`$ and $`K^+K^-\pi^0`$.
exHad samples these three channels using the absolute value, `Abs(...)`, of
the input expression. This produces nonnegative sampling weights but leaves
the problem in the source squared matrix elements unresolved. The tabulated
lifetimes and branching fractions are unchanged.

## Installation

exHad needs Linux or macOS, Python 3.11 or later, a C++17 compiler, `make`,
`git`, and a built installation of Pythia8.317 with its headers, library and
XML data files. Download Pythia8.317 from [pythia.org](https://pythia.org)
([source repository](https://gitlab.com/Pythia8/releases)) and follow its build
instructions. exHad links the Pythia C++ library; it works without the Pythia
Python bindings.

Run the following commands from the exHad directory. Replace
`/absolute/path/to/pythia8317` with your Pythia installation directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
python tools/setup.py --pythia8-dir /absolute/path/to/pythia8317
```

The setup downloads the vector-current form-factor code
[DeLiVeR](https://github.com/preimitz/DeLiVeR), builds the Pythia generators and
the accelerated sampler, and checks their compatibility with Pythia8.317.
Event generation needs no network access. `--jobs N` sets the number of parallel
compiler processes (default 4). The individual build commands are documented in
[`tools/README.md`](tools/README.md).

If you move the exHad directory, create the Python environment again, since the
installation with `pip install -e` records the absolute path. If you move or
replace Pythia, run `tools/setup.py` again.

## Generating decays

The options of a request, the sampling variations, the seeds and the parallel
generation are described in
[`docs/INTERFACES.md`](docs/INTERFACES.md#parameters-of-a-request).

### Command line

```bash
python -m exhad --model alp-fermion --mass 2.0 --events 1000 --seed 12345 \
  --output events.json
```

This command writes 1,000 hadronic decays of `alp-fermion` with mass 2 GeV to
`events.json`. It stops with an error if the output file exists. The command
`exhad-generate` is equivalent to `python -m exhad`. `--format` chooses the
file format, JSON or HepMC3 or HepMC2 ASCII; see
[Output formats](#output-formats).

exHad generates the events in parallel: it divides the request into chunks of up
to 512 events (`--chunk-size`) and generates the chunks simultaneously in worker
processes, each of which runs its own Pythia instances and normally uses one CPU
core. `--workers N` sets the number of worker processes; by default exHad uses
one per physical core, at most 8. The events of a request and their order are
the same for every number of workers.

With `--all-decays`, exHad samples among all decay channels of the particle
according to their branching ratios; without it, exHad returns the decays that
contain hadrons: the hadronic decays of a boson portal, and the semileptonic
decays of `hnl`.
`hnl` requires the three squared mixings $`|U_e|^2`$, $`|U_\mu|^2`$, $`|U_\tau|^2`$,
whose relative values select the mixing flavour of the decay:

```bash
python -m exhad --model hnl --mass 2.0 --events 1000 --all-decays \
  --mixing 1 0 0 --output hnl.json
```

### Python

```python
from exhad import Generator

with Generator("alp-fermion", workers=8, chunk_size=512) as generator:
    events = generator.generate(mass=2.0, events=1000, seed=12345)
```

`events` is a list of decays in the format described in
[Output formats](#output-formats). Keep one generator open for successive
requests, since starting it initializes Pythia; leaving the `with` block stops
its worker processes. `generate_all` returns a sample of all decay channels:

```python
with Generator("hnl") as generator:
    sample = generator.generate_all(2.0, 1000, mixing=[1., 0., 0.], seed=12345)
```

The methods of `Generator`, the weighted samples of `alp-fermion`, the branching
ratios and lifetimes returned by `model_info`, and `hadronize_hnl`, which forms
the hadrons of an `hnl` decay whose lepton and quark-antiquark pair a simulation
program has already sampled, are described in
[`docs/INTERFACES.md`](docs/INTERFACES.md#python-interface).

### Inelastic dark matter

```python
from exhad.inelastic_dark_matter import generate, ctau

sample = generate(mediator=20.0, count=1000, ratio=3.0, delta=0.1,
                  alpha_D=0.1, seed=12345)
decay_length_m = ctau(20.0, epsilon2=1e-8, ratio=3.0, delta=0.1, alpha_D=0.1)
```

All masses are in GeV. These parameters give a dark-photon mediator mass of
20 GeV, a lighter dark-fermion mass of `20/3` GeV, and an excited mass 10%
higher. `alpha_D` is the dark fine-structure constant and `epsilon2` is the
squared kinetic mixing, defined through the interaction in
[Inelastic dark matter](docs/PHYSICS.md#inelastic-dark-matter).
`ctau` returns the excited fermion's proper decay length in metres.

The mass splitting must lie above the electron-pair threshold and below
1.7 GeV; the mediator must be heavier than the splitting. The calculation
assumes an unpolarized excited fermion. Its exclusive channels and momentum
distributions are described in the same physics section.

`sample["particles"]` contains the complete rest-frame decays, including the
invisible lighter fermion (PDG code 4900101), with fields
`[px, py, pz, E, mass, PDG]`. Arrays are padded; discard rows with PDG `-999`.
`sample["q"]` gives the visible-system invariant mass and `sample["channel"]`
its channel label. Request these decays through the functions in
`exhad.inelastic_dark_matter`; the interface is Python-only.

### C++ and Pythia8

The header files in `include/exhad` provide a C++17 interface. It starts exHad
as a separate Python process and exchanges events with it, so a C++ program
works without the Python C API, and the Pythia8.317 library of exHad stays
apart from the Pythia library of the simulation program. To let exHad decay a
particle inside a Pythia8 program, register exHad as a Pythia decay handler
(the Pythia interface through which an external program decays chosen particles)
before `pythia.init()`:

```cpp
#include "exhad/Pythia8DecayHandler.hpp"
auto generator = std::make_shared<exhad::Generator>(
    exhadRoot, pythonExecutable, "alp-fermion");
auto decayer = std::make_shared<exhad::Pythia8DecayHandler>(
    generator, motherPdg, exhad::DecayScope::All, 12345, 64);
decayer->prepareParticle(pythia);
pythia.setDecayPtr(decayer);
```

The number `64` is the batch size: exHad generates 64 decays at a time at a
fixed mass, and a change of mass discards the unused decays. To obtain events
without a Pythia program:

```cpp
#include "exhad/Generator.hpp"
exhad::Generator generator(exhadRoot, pythonExecutable, "hnl");
auto events = generator.generate(3.0, 1000, 12345, exhad::DecayScope::All, {1., 0., 0.});
// events[i][j]: px, py, pz, energy, mass (GeV), PDG code
```

`examples/pythia8_decays.cc` is a complete Pythia8 program in which exHad
decays a particle. The build with CMake, the settings that keep particles stable
for detector transport and the use with several threads are described in
[`docs/INTERFACES.md`](docs/INTERFACES.md#c-interface-and-pythia8).
`examples/plot_channel_fractions.py` draws the fractions of the decay channels
of each particle as functions of its mass
([`docs/INTERFACES.md`](docs/INTERFACES.md#example-programs)):

```bash
python examples/plot_channel_fractions.py --models dark-photon --points 20 --events 2000
```

## Output formats

**JSON** (default). The file contains the particle model, the mass (`mass_gev`), the
seed, the variation, `decay_scope` (`hadronic`, `semileptonic` for `hnl`, or
`all`), the PDG identifier written for the decaying particle (`mother_pdg`),
the names of the particle fields and the events. Each event is a list of the
final particles of one decay, and each particle is
`[px, py, pz, E, mass, PDG]`: momentum
components, energy and mass in GeV, and the particle identification number of
the Particle Data Group (PDG code). Different decays contain different numbers
of particles. The decaying particle is at rest, and the particles of each decay
add up to its four-momentum; the decaying particle itself is absent from the
JSON file. All unstable particles are decayed, including $`\pi^0`$ and $`K^0_S`$,
except muons, charged pions, charged kaons, $`K^0_L`$ mesons and neutrons, which
are left undecayed so that a detector simulation can transport them. Neutrinos
are included.

**HepMC3 and HepMC2 ASCII.** exHad writes these through
[pyhepmc](https://scikit-hep.org/pyhepmc/), which it
installs as a dependency:

```bash
python -m exhad --model alp-fermion --mass 2.0 --events 1000 \
  --format hepmc --mother-pdg 9000006 --output events.hepmc
```

Use `--format hepmc2` for HepMC2 readers. Each event contains the decaying
particle at rest with status 2, its decay vertex at the origin, and the final
particles with status 1, in units of GeV and mm. Intermediate resonances are
absent from the file.

**PDG codes.** The final particles have their standard PDG codes. The decaying
particle has no standard PDG code: exHad writes $`A'`$ as 4900022, $`Z'`$ as 32, $`S`$
as 35, $`a`$ as 36 and $`N`$ as 9900012, and `--mother-pdg` (Python `mother_pdg`)
replaces this identifier by the one the simulation uses, as `9000006` above.
The fields of each file and the labels of the decay channels are described in
[`docs/INTERFACES.md`](docs/INTERFACES.md#contents-of-json-and-hepmc-files).

## Tests of the installation

```bash
python -m unittest discover -s tests -v
```

The tests check finite, on-shell momenta, conservation laws and reproducibility
with a fixed seed. Their coverage and the sampler comparisons are recorded in
[`tests/README.md`](tests/README.md). Physical accuracy and limitations are
described in [`docs/PHYSICS.md`](docs/PHYSICS.md).

## Documentation

- [`docs/PHYSICS.md`](docs/PHYSICS.md): the decay model, the physics of each
  particle, its inputs, its accuracy and its limitations.
- [`docs/INTERFACES.md`](docs/INTERFACES.md): the parameters of a request, the
  command line, the Python and C++ interfaces, the output files and the use of
  exHad decays in a simulation program.
- [`docs/SAMPLING.md`](docs/SAMPLING.md): how events are drawn and the accelerated
  sampler.
- [`tests/README.md`](tests/README.md): test coverage and sampler validation.
- [`THIRD_PARTY.md`](THIRD_PARTY.md): the downloaded programs and the numerical
  inputs, with their licenses.

## Citation and license

Please cite *Rethinking search signatures: Hadronic decays of GeV-scale feebly
coupled particles*, CERN-TH-2026-223, arXiv:2609.16104, together with the rate
calculations listed in the table of particle models and Pythia8.3,
[arXiv:2203.11601](https://arxiv.org/abs/2203.11601). Software citation metadata
is in `CITATION.cff`.

exHad is distributed under the BSD 3-Clause license (`LICENSE`). Parts of the
code are distributed under a separate BSD 3-Clause notice (`LICENSE-EventCalc`),
and [`THIRD_PARTY.md`](THIRD_PARTY.md) describes the licenses of the downloaded
programs and of the numerical inputs.
