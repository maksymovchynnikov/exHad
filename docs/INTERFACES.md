# Requesting decays from exHad and using them in simulation programs

This document is the reference for requesting decays from exHad through the command line, Python
and C++, for the data that each request returns, and for the steps that a simulation program
performs with the returned decays. The model table, the installation, first examples and the
output formats (particle format, PDG codes, supported files) are in the [README](../README.md).
The physics of each model is in [PHYSICS.md](PHYSICS.md), and the algorithms that draw the
hadronic events are in [SAMPLING.md](SAMPLING.md). *The scalar models* denotes `scalar-central`,
`scalar-lower`, `scalar-upper` and `scalar-1809`.

- [Parameters of a request](#parameters-of-a-request)
- [Command line](#command-line)
- [Python interface](#python-interface)
- [C++ interface and Pythia8](#c-interface-and-pythia8)
- [Parallel generation](#parallel-generation)
- [Random seeds and reproducibility](#random-seeds-and-reproducibility)
- [Failed requests](#failed-requests)
- [Contents of JSON and HepMC files](#contents-of-json-and-hepmc-files)
- [Weighted `alp-fermion` samples](#weighted-alp-fermion-samples)
- [Using exHad decays in a simulation program](#using-exhad-decays-in-a-simulation-program)
- [Generation speed](#generation-speed)
- [Example programs](#example-programs)

## Parameters of a request

Every request names a model, a mass, a number of events and a random seed. It can also choose the
decay channels, a model variation, the sampler, the division of the work among processes and, for
`hnl`, the squared mixings. The names and the defaults differ between the three interfaces.

| Parameter | Python | Command line | C++ |
|---|---|---|---|
| Model: a name from the model table of the [README](../README.md) | first argument of `Generator` | `--model` | `model`, third argument of `exhad::Generator` |
| Mass of the decaying particle in GeV ($`m_N`$ for `hnl`) | `mass` | `--mass` | `mass` |
| Number of events | `events` | `--events` (1000) | `count` |
| Random seed, an unsigned 64-bit integer | `seed` (1) | `--seed` (1) | `seed` (1) |
| [Decay channels](#choice-of-decay-channels): the decays that contain hadrons, or all decay channels | method `generate` or `generate_all` | `--all-decays` absent or present | `DecayScope::Hadronic` (the decays that contain hadrons) or `DecayScope::All` |
| [Squared mixings](#squared-mixings-of-hnl) of `hnl` | `mixing` | `--mixing UE2 UMU2 UTAU2` | `mixingSquared` |
| [Model variation](#model-variations) | `variation` (`"central"`) | `--variation` (`central`) | `variation` (`"central"`) |
| [Sampler](#rejection-and-accelerated-samplers): `auto` or `reference` | `execution` (`"auto"`) | always `auto` | `execution` (`"auto"`) |
| Number of [worker processes](#parallel-generation) | `workers` (one per physical CPU core, at most 8) | `--workers` (the same) | `workers` (0, which selects the same) |
| Number of events per [chunk](#parallel-generation) | `chunk_size` (512) | `--chunk-size` (512) | `chunkSize` (512) |
| [Weighted sample](#weighted-alp-fermion-samples) of `alp-fermion` | method `generate_weighted`, or `weighted=True` in `generate_all` | `--weighted` | not available |
| [Floor fraction](#weighted-alp-fermion-samples) $`\lambda`$ of a weighted sample | `weight_floor_fraction` (0) | `--weight-floor-fraction` (0.1) | not available |

### Choice of decay channels

A request generates either the decays that contain hadrons (the default) or decays into all
channels of the particle, and the branching ratio of the first group enters a simulation exactly
once in both cases.

- *Decays that contain hadrons* (`generate`, the command line without `--all-decays`,
  `DecayScope::Hadronic`). For a boson portal every event is a hadronic decay: hadrons, in a few
  channels together with a photon. For `hnl` every event is a semileptonic decay: a charged
  lepton or a neutrino together with hadrons. All three interfaces generate this sample for every
  model, and each requires the squared mixings of `hnl`. The simulation program selects or
  weights these decays with the branching ratio of the same group: the hadronic branching ratio
  of a boson portal, the semileptonic branching ratio of `hnl`. The branching ratios that
  `model_info` returns for each model, and how they relate to exHad's decays, are described in
  [Branching ratios, lifetimes and mass intervals](#branching-ratios-lifetimes-and-mass-intervals-model_info).
- *All decay channels* (`generate_all`, `--all-decays`, `DecayScope::All`). exHad chooses each
  decay among the hadronic and nonhadronic channels of the particle, including decays into
  neutrinos only, with probabilities equal to their branching ratios at the requested mass, and
  then generates the chosen decay. The simulation program applies no further branching ratio. Each
  event is returned with the [label of its decay channel](#labels-of-decay-channels).

The decays that contain hadrons follow the same description of the hadronic final states, and the
same hadronic width, in both kinds of sample.
[PHYSICS.md](PHYSICS.md#nonhadronic-decays-and-samples-of-all-decay-channels) lists the
nonhadronic channels of each model and their widths.

For `hnl`, [`hadronize_hnl`](#hnl-decays-sampled-by-the-simulation-program-hadronize_hnl) forms
the hadrons in decays into a lepton and a quark–antiquark pair that the simulation program
samples itself.

### Squared mixings of `hnl`

The squared mixings fix the relative rates of the `hnl` decays through the three lepton flavours.
They are three nonnegative numbers proportional to $`\lvert U_e\rvert^2`$, $`\lvert U_\mu\rvert^2`$ and
$`\lvert U_\tau\rvert^2`$, where $`U_\alpha`$ is the mixing of the heavy neutral lepton $`N`$ with
the neutrino of flavour $`\alpha`$ ([PHYSICS.md](PHYSICS.md#heavy-neutral-lepton-hnl)). Their sum
must be positive. Their common scale cancels in the branching ratios, so `1 0 0` selects pure
electron mixing. Python and the command line refuse mixings for other models. The C++ interface requires
three finite nonnegative numbers for every model and uses their values only for `hnl`.

### Model variations

A variation selects the central model, `central`, or a version with changed parameters of its
hadronic final states: `p-low`, `p-high` and `saturation-off` for `alp-fermion`, the scalar
models, `b-l` and `hnl`, and `coefficient-eigenmode-NN-plus` and `coefficient-eigenmode-NN-minus`
with NN = 01–31 for `dark-photon`. [PHYSICS.md](PHYSICS.md#model-variations) states what each
variation changes and which quantities it leaves unchanged.

Below the mass at which string fragmentation begins (1.70 GeV for `dark-photon`, 1.911 GeV for
`alp-fermion`, 2.00 GeV for the scalar models and `b-l`) the hadronic decays are the tabulated
exclusive channels, of which no variant exists: a request there with any variation other than
`central` stops with an error that names the mass and the variation.

### Rejection and accelerated samplers

Two samplers generate the hadronic events of the
[constrained fragmentation model](PHYSICS.md#hadronic-final-states-from-constrained-string-fragmentation),
and the parameter `execution` chooses between them.

- The *rejection sampler* lets [Pythia](https://pythia.org) generate complete trial events by
  string fragmentation and accepts each with a probability set by its
  [channel group](PHYSICS.md#widths-and-channel-groups) and charge combination.
- The *accelerated sampler* samples the same event distribution with fewer complete
  fragmentations: it decides acceptance while the string fragments, and it generates the rarest
  channel groups directly.

With `auto`, exHad uses the accelerated sampler for unweighted samples of the models and masses
listed in [SAMPLING.md](SAMPLING.md#requests-that-use-the-accelerated-sampler) and the rejection
sampler elsewhere. With `reference`, it uses the rejection sampler at every mass. Weighted samples
always use the rejection sampler. The two samplers give the same channel probabilities within the
statistical precision of the tabulated Pythia probabilities, and identical distributions within
each charge combination
([derivation](SAMPLING.md#accelerated-generation-of-unweighted-events) and
[statistical comparisons](SAMPLING.md#statistical-comparisons-with-the-rejection-sampler)).

## Command line

The command `exhad-generate` writes one sample of decays to a new file. Its output options are
listed below; its other options are the [parameters of a request](#parameters-of-a-request).

| Option | Meaning |
|---|---|
| `--output PATH` | Output file (required) |
| `--format FORMAT` | `json` (default), `hepmc` for HepMC3 ASCII or `hepmc2` for HepMC2 ASCII ([Contents of JSON and HepMC files](#contents-of-json-and-hepmc-files)) |
| `--mother-pdg CODE` | PDG identifier written for the decaying particle in HepMC files, a nonzero signed 32-bit integer; without it exHad writes the identifier of the model ([Output formats](../README.md#output-formats)) |

The command stops with an error message and writes no file when a parameter of
[Parameters of a request](#parameters-of-a-request) lies outside its allowed values, when the
[request fails](#failed-requests), and in the following combinations:

- `--model hnl` without `--mixing`;
- `--mixing` for another model;
- `--weighted` for a model other than `alp-fermion`;
- `--format hepmc` or `--format hepmc2` where pyhepmc cannot be imported
  ([Output formats](../README.md#output-formats));
- `--output` naming a file that exists.

## Python interface

The package `exhad` defines the class `Generator`, which generates decays in worker processes, and
the function [`model_info`](#branching-ratios-lifetimes-and-mass-intervals-model_info), which
returns the paths of the rate tables of a model.

```python
Generator(model, *, workers=None, chunk_size=512, variation="central",
          execution="auto", root=None, python=None)
```

`root` is the exHad directory (by default the directory that contains the imported `exhad`
package), and `python` is the Python executable that runs the worker processes (by default the
running interpreter). The other arguments are described in
[Parameters of a request](#parameters-of-a-request). The worker processes start with the first
request and keep their Pythia instances initialized until the generator is closed, so successive
requests to one open generator avoid this initialization. `Generator` is a context manager, and
leaving its `with` block closes it. One `Generator` serves one thread.

| Method | Result |
|---|---|
| `generate(mass, events, *, seed=1, mixing=None)` | List of the decays that contain hadrons, each a list of particles in the format of [Output formats](../README.md#output-formats): the hadronic decays of a boson portal, and the semileptonic decays of `hnl` (a charged lepton or a neutrino together with hadrons), which requires `mixing`. |
| `generate_all(mass, events, *, seed=1, mixing=None, weighted=False, weight_floor_fraction=0.)` | Dictionary with the decays in `events`, the [label of the decay channel](#labels-of-decay-channels) of each decay in `channel_labels`, and `decay_scope` equal to `"all"`. `mixing` is required for `hnl`. With `weighted=True`, the dictionary also contains the [weight fields](#weighted-alp-fermion-samples). |
| `generate_weighted(mass, events, *, seed=1, weight_floor_fraction=0.)` | Dictionary with weighted hadronic decays of `alp-fermion` in `events` and their [weight fields](#weighted-alp-fermion-samples) |
| `hadronize_hnl(mass, pdgs, primary_events, *, seed=1)` | List of decays in which exHad has replaced the quark–antiquark pair by hadrons; only for `Generator("hnl")` ([`hadronize_hnl`](#hnl-decays-sampled-by-the-simulation-program-hadronize_hnl)) |
| `close()` | Stops the worker processes |

Arguments that the calling process checks raise `ValueError`: an unknown model name, a number of
events or a seed of the wrong type or range, a mass that is not finite and positive, invalid
mixings or floor fraction, an invalid number of worker processes or chunk size, an unknown sampler,
and a method that the model does not have. Everything that the worker processes check or that
fails during generation raises `RuntimeError` with the message of the worker process, including
masses outside the range of decay generation and unknown variations. An installation whose
configuration step has not been completed raises `RuntimeError` at the first request.
[Failed requests](#failed-requests) states when the generator remains usable.

## C++ interface and Pythia8

The header-only C++17 interface in `include/exhad` gives C++ programs exHad decays either as
batches (`exhad::Generator`) or as decays of particles inside an event of Pythia8
(`exhad::Pythia8DecayHandler`).

### Requirements of C++ programs

The constructor of `exhad::Generator` starts exHad in a separate Python process, which runs the
[worker processes](#parallel-generation) and exchanges requests and events with the C++ program
through a socket. Every machine that runs such a program therefore needs a configured exHad
directory and its Python environment
([Installation](../README.md#installation)), while the program
itself needs neither the Python C API nor a JSON library.

- The Python process and its worker processes use exHad's Pythia8.317 library, which stays
  separate from the Pythia library linked into the simulation program.
- The paths of the exHad directory and of the Python executable must be absolute.
- The interface runs on Linux and macOS.
- The interface returns unweighted events. Weighted samples and `hadronize_hnl` are available in
  Python.

### Building with CMake

A CMake project adds the exHad directory and links the interface target `exhad::client`, and any
other C++17 build adds the directory `include/` of exHad to its include path.

```cmake
add_subdirectory(/absolute/path/to/exHad exhad)
target_link_libraries(my_simulation PRIVATE exhad::client)
```

The CMake file of exHad requires CMake 3.18 or later. `exhad::Generator` needs no Pythia headers or
library; `exhad::Pythia8DecayHandler` needs those of the simulation program.

The example program `examples/pythia8_decays.cc` ([Example programs](#example-programs)) is built
and run from the exHad directory, here with a Pythia installation in `/absolute/path/to/pythia8`:

```bash
cmake -S . -B build-client -DEXHAD_BUILD_EXAMPLE=ON -DPYTHIA8_DIR=/absolute/path/to/pythia8
cmake --build build-client
build-client/pythia8_decays "$PWD" "$PWD/.venv/bin/python" \
  /absolute/path/to/pythia8/share/Pythia8/xmldoc
```

`PYTHIA8_DIR` must contain `include/Pythia8/Pythia.h` and the `pythia8` library in `lib/`. The three
arguments of the program are the exHad directory, the Python executable of the exHad environment and
the XML data directory of the Pythia installation.

### Batches of decays without a Pythia program: `exhad::Generator`

`exhad::Generator` returns a vector of decays for one mass, number of events, seed and choice of
decay channels. Its constructor and generating function are

```cpp
Generator(const std::string& root, const std::string& python, std::string model,
          std::string variation = "central", std::string execution = "auto",
          int workers = 0, std::size_t chunkSize = 512, int timeoutSeconds = 600);

std::vector<Event> generate(double mass, std::size_t count, std::uint64_t seed = 1,
                            DecayScope scope = DecayScope::Hadronic,
                            std::array<double, 3> mixingSquared = {0., 0., 0.});
```

- `exhad::Event` is `std::vector<exhad::Particle>`, and `exhad::Particle` has the members `px`,
  `py`, `pz`, `energy` and `mass` in GeV and `pdg`, the PDG code.
- One call generates at most 1,000,000 events.
- `workers` lies between 0 and 64, where 0 selects the default number of worker processes, and
  `chunkSize` between 1 and 1,000,000.
- Each wait for data from the Python process is limited to `timeoutSeconds`, between 1 and
  86,400 s. The events of a call arrive after all of them are generated, so `timeoutSeconds` must
  exceed the generation time of the largest call. The first call also starts the worker processes
  and initializes their Pythia instances, and `timeoutSeconds` must cover that as well.
- The Python process starts with the constructor. `close()` stops it, and the destructor calls
  `close()`.
- The constructor checks the two paths, the names of the model, the variation and the sampler,
  `workers`, `chunkSize` and `timeoutSeconds`; `generate` checks the mass, the number of events
  and the three squared mixings. These checks throw `std::invalid_argument` before a request is
  sent, and the generator remains usable. An unknown model or variation, a request for `hnl` with
  squared mixings that add up to zero, and a mass outside the range of decay generation are
  reported by the Python process as failed requests.
- A failed request throws `std::runtime_error` with a message that starts with `exHad:` and gives
  either the error of the Python process or the statement that this process exited or did not
  answer within `timeoutSeconds`. The generator is then closed, and further requests need a new
  `exhad::Generator`.
- Each thread uses its own `exhad::Generator`.

### Decays inside a Pythia8 event: `exhad::Pythia8DecayHandler`

`exhad::Pythia8DecayHandler` replaces every decay of one particle species in a Pythia8 program by
an exHad decay, and adds the decay products, boosted to the momentum of the decaying particle, to
the Pythia event. Its constructor is

```cpp
Pythia8DecayHandler(std::shared_ptr<Generator> generator, int motherPdg, DecayScope scope,
                    std::uint64_t seed = 1, std::size_t batchSize = 64,
                    std::array<double, 3> mixingSquared = {0., 0., 0.});
Pythia8DecayHandler(std::shared_ptr<Generator> generator, DecayScope scope,
                    std::uint64_t seed = 1, std::size_t batchSize = 64,
                    std::array<double, 3> mixingSquared = {0., 0., 0.});
```

The second constructor uses the PDG identifier of the model,
`exhad::defaultMotherPdg(model)` ([Output formats](../README.md#output-formats)).

- The simulation program defines the decaying particle with the PDG code `motherPdg` in the
  particle data of Pythia, calls `prepareParticle(pythia)` of the handler, registers the handler
  with `pythia.setDecayPtr` and then calls `pythia.init()`.
- `prepareParticle(pythia)` allows the particle to decay and, if Pythia has no decay channel for
  it, adds a channel that is switched off and has zero branching ratio, because Pythia8.317 passes
  a particle to an external decay handler only if the particle has a decay table. Pythia never
  selects this channel. Pythia8.317 treats the identifiers 32, 35, 36 and 9900012 as resonances,
  which it decays itself and whose width and lifetime it recalculates; `prepareParticle` removes
  this property and fixes the width at zero, leaving the mass and the lifetime that the simulation
  program has set.
- The handler decays both the particle whose PDG code is the absolute value of `motherPdg` and its
  antiparticle, and it gives them the same decay products, because the decaying particles of all
  models are their own antiparticles.
- `scope` chooses between the decays that contain hadrons and all decay channels
  ([Choice of decay channels](#choice-of-decay-channels)). For `hnl`, `scope` is `DecayScope::All`
  and `mixingSquared` contains the squared mixings.
- The handler requests `batchSize` decays at a time at the mass of the decaying particle, between
  1 and 1,000,000, and uses each decay once. Each batch is one request to the Python process. The
  first batch uses `seed`, and every further batch uses a seed larger by one. When a particle
  arrives whose mass differs from the mass of the current batch, the unused decays are discarded
  and a new batch is generated at the new mass, so particles whose masses vary continuously need
  `batchSize` 1.
- The simulation program determines the production and the lifetime of the particle in Pythia.
  Pythia places the decay products at the decay vertex of the particle and records their mother
  links.
- The handler hands Pythia the final state of the exHad decay and nothing else: Pythia adds no
  photon radiation to these products, whatever the PDG code of the decaying particle.
  `allowHostRadiation()` of the handler lets Pythia radiate off them, and
  `allowHostRadiation(false)` returns to the default.

  The choice is a physical one. exHad calculates the branching ratios and the momenta of the decay
  channels without radiation off the decay products, so the default keeps the decay as exHad
  calculates it, and the products of a sample have the rates and the energies of that calculation.
  Radiation of the simulation program moves energy from the products into photons that exHad's
  branching ratios and momenta leave out, and it is the radiation of Pythia's shower, which
  describes a decay to a lepton pair and no other channel; a program switches it on to study that
  radiation itself.

  Pythia radiates off the products of a two-body decay to a lepton pair when `HadronLevel:QED` is
  on, and, for the codes of its hidden sector such as the default 4900022 of `dark-photon`, when
  `ParticleDecays:FSRinDecays` is on, which it is by default. Neither choice of the handler changes
  a setting of the simulation program, and the decays that Pythia performs itself, among them the
  decays of the products of an exHad decay, follow the settings of the program in either case.
- Pythia keeps one external decay handler per Pythia instance, and `setDecayPtr` replaces a
  handler set earlier. A program that uses its own external decays for other particles calls the
  exHad handler from its own handler.
- Each Pythia instance that runs in its own thread uses its own `exhad::Generator` and handler.
- A failed request throws an exception out of the Pythia call that requested the decay.
- The handler uses the `DecayHandler` interface of Pythia and is tested with Pythia8.317. A
  program linked to another Pythia version checks the handler by building and running the example
  program with that version.

### Particles left for detector transport

A Pythia program keeps the long-lived particles of exHad decays undecayed for detector transport
with `Pythia8DecayHandler::keepTransportParticlesStable(pythia)`, which switches off the decays of
muons, charged pions, charged kaons, $`K^0_L`$ mesons and neutrons in that Pythia instance. The
setting applies to these particles in every event of the instance, including particles that exHad
did not produce. A simulation framework that sets its own list of stable particles omits this call.

## Parallel generation

exHad generates the events of one request in parallel. It divides the request into chunks of
consecutive events and gives the chunks to several worker processes, which return them in the
order of the chunks.

A worker process is a background process with its own Pythia instances, which generates one chunk at
a time and normally occupies one CPU core. A chunk is a group of at most `chunk_size` consecutive
events of one request.

- By default exHad uses one worker process per physical CPU core available to it, at most 8.
  `workers=1` in Python, `--workers 1` on the command line or `workers = 1` in C++ generates every
  request in a single process. At most 64 worker processes are allowed. Fewer worker processes leave
  CPU cores and memory free for other programs.
- A request starts at most as many worker processes as it has chunks, and started processes remain
  until the generator is closed.
- A request with at most `chunk_size` events forms a single chunk, which is generated with the seed
  of the request. A larger request gives each chunk a seed computed from the seed of the request,
  the total number of events and the position of the chunk.
- For a fixed chunk size, the events and their order are the same for every number of worker
  processes, including one. A different chunk size changes the random numbers, and therefore the
  individual events, of every request with more events than the smaller chunk size; the
  distribution stays the same.

## Random seeds and reproducibility

A request reproduces exactly the same events whenever its method (`generate`, `generate_all`,
`generate_weighted`, or `hadronize_hnl` with the same input decays; in C++ the choice of decay
channels), model, variation, sampler, mass, number of events, seed and chunk size are the same,
together with the squared mixings of `hnl` and the floor fraction of a weighted sample, and exHad
and its Pythia installation are unchanged.

- Earlier requests to the same generator do not change the events.
- The events depend on the number of events in the request. A sample divided into requests of other
  sizes consists of different individual events with the same distribution.
- For the same seed, the samplers `auto` and `reference` give different individual events at the
  masses where `auto` uses the accelerated sampler.
- Independent samples need different seeds.
- `exhad::Pythia8DecayHandler` reproduces its decays for the same generator settings, seed, batch
  size and sequence of decaying particles and masses.

## Failed requests

A request that exHad cannot generate fails with an error in every interface. exHad never replaces
the requested events by events of
[unmodified Pythia](PHYSICS.md#hadronic-final-states-from-constrained-string-fragmentation).

- Requests fail at masses outside the ranges of decay generation in the model table of the
  [README](../README.md), and in the narrow intervals of mass, or of the invariant mass $`W`$ of the
  quark–antiquark pair of an `hnl` decay, named in the sections of the models in
  [PHYSICS.md](PHYSICS.md).
- Python raises `ValueError` or `RuntimeError` ([Python interface](#python-interface)). After a
  `RuntimeError` that reports a failed or exited worker process, or after an interrupted request
  (for example with Ctrl+C), the program constructs a new `Generator`. After other errors the
  generator remains usable.
- The command line stops with an error message and writes no file.
- The C++ interface throws `std::runtime_error` and closes the generator
  ([Batches of decays without a Pythia program](#batches-of-decays-without-a-pythia-program-exhadgenerator)).

## Contents of JSON and HepMC files

The command line writes the events as JSON, HepMC3 ASCII or HepMC2 ASCII files.
[Output formats](../README.md#output-formats) gives the particle format, the PDG codes and the
structure of these files; the fields below are what exHad writes in addition.

### JSON files

A JSON file contains one JSON object with the events and the parameters of the request.

| Key | Content |
|---|---|
| `schema` | `exhad-rest-frame-events-v1`, or `exhad-weighted-rest-frame-events-v1` for weighted samples |
| `model` | Model name |
| `mass_gev` | Mass of the decaying particle in GeV |
| `seed` | Seed |
| `variation` | Model variation |
| `decay_scope` | `hadronic`, `semileptonic` for `hnl`, or `all` |
| `particle_fields` | `["px", "py", "pz", "E", "mass", "PDG"]` |
| `events` | List of decays, each a list of particles |
| `execution` | Object with the key `chunk_size`, the number of events per chunk |
| `channel_labels` | Samples of all decay channels: the [label of the decay channel](#labels-of-decay-channels) of each decay |
| `mixing_squared` | `hnl`: the three squared mixings |
| Weight fields | Weighted samples: the fields listed in [Weighted `alp-fermion` samples](#weighted-alp-fermion-samples) |

### HepMC3 and HepMC2 files

Each event of a HepMC3 or HepMC2 file has the structure described in
[Output formats](../README.md#output-formats) and, in addition, contains

- the generated mass of the decaying particle and of each final particle;
- one event weight, named `nominal`;
- the string attributes `exhad_model` and `exhad_decay_scope`;
- an event number, counted from 0.

In Python, `exhad.output.write_events(path, payload, format="hepmc", mother_pdg=...)` writes the
same files from a dictionary `payload` with the keys `model`, `mass_gev`, `events` and, if present,
`decay_scope` and `hadronization_weights`. With `format="json"` it writes `payload` as a JSON file.
Like the command line, the function refuses to write to an existing path, and it refuses HepMC
output of a payload that contains `raw_weights` without normalized `hadronization_weights`.

### Labels of decay channels

In a sample of all decay channels, `channel_labels` gives for each decay the label of its decay
channel, as listed in the table.

| Model | Label of hadronic decays | Labels of nonhadronic decays |
|---|---|---|
| `dark-photon` | `hadronic` | `e+e-` ($`e^+e^-`$), `mu+mu-` ($`\mu^+\mu^-`$), `tau+tau-` ($`\tau^+\tau^-`$) |
| `b-l` | `hadronic` | `e+e-`, `mu+mu-` and `tau+tau-` as for `dark-photon`; `nu12-antinu12` ($`\nu_e\bar\nu_e`$), `nu14-antinu14` ($`\nu_\mu\bar\nu_\mu`$), `nu16-antinu16` ($`\nu_\tau\bar\nu_\tau`$) |
| `alp-fermion` | `hadronic` | `ePeM` ($`e^+e^-`$), `muPmuM` ($`\mu^+\mu^-`$), `tauPtauM` ($`\tau^+\tau^-`$), `2gamma` ($`\gamma\gamma`$) |
| The scalar models | `hadronic` | `ePeM`, `muPmuM` and `tauPtauM` as for `alp-fermion` |

The label of an `hnl` decay is the name of the entry of `tables["decay"]` of `model_info("hnl")`
from which the decay was generated, and the PDG codes listed with that entry give the primary decay
products of the channel, so a program identifies the channel from those codes. Decays into three
neutrinos, which the table does not list, have the label `3nu`.

## Weighted `alp-fermion` samples

For `alp-fermion`, exHad can generate weighted samples: trial events that the rejection sampler
would discard are kept, each with a weight.
**Every analysis of a weighted sample must use the event weights.**

In the [rejection sampler](SAMPLING.md#event-generation-by-rejection-sampling) each trial event
from string fragmentation has the weight $`w \geq 0`$ defined there and is accepted with
probability $`w/M`$, where $`M`$ is the bound of those weights at the requested mass. A weighted sample
uses a floor fraction $`\lambda`$ between 0 and 1 and the threshold $`t = \lambda M`$. It keeps a trial
event with probability $`\min(1, w/t)`$ and gives it the raw weight $`\max(w, t)`$. The product of these
two factors equals $`w`$ for every trial event, so the kept events, weighted with their raw weights,
have the distribution of the events accepted by the rejection sampler.

- With $`\lambda = 0`$, every trial event with $`w > 0`$ is kept, with raw weight $`w`$.
- With $`\lambda = 1`$, a trial event is kept with probability $`w/M`$, and all kept events have the
  same weight. The events are then identical to those of `execution="reference"` with the same
  seed.
- The default is $`\lambda = 0.1`$ on the command line and $`\lambda = 0`$ in Python.

| Weight field | Content |
|---|---|
| `raw_weights` | $`\max(w, t)`$ for `fragmentation` events, and 1 for all other events |
| `normalization_groups` | `fragmentation` for events kept from weighted trial events of string fragmentation; `external` for all other events, for example decays into nucleon pairs, decays from charm quark pairs, the channel groups $`3\pi`$ and $`\pi^+\pi^-\gamma`$, events from showered Pythia above 4 GeV, and nonhadronic decays |
| `family_labels` | Label of the channel group of each `fragmentation` event; `pythia` for events from [showered Pythia](PHYSICS.md#decays-into-quarks-and-gluons-above-4-gev) above 4 GeV; `external` for the other hadronic events; the [label of the decay channel](#labels-of-decay-channels) for nonhadronic decays |
| `weight_floor_fraction` | Floor fraction $`\lambda`$ |
| `weight_convention` | Text that states the normalization rule below |
| `hadronization_weights` | Command line only: the normalized weights |
| `fragmentation_normalization` | Command line only: the factor that multiplies the raw weights of `fragmentation` events |
| `normalization_scope` | Command line only: text that states that the normalization covers the complete sample before any selection |

The raw weights of `fragmentation` events share an arbitrary common factor. For every event, exHad
first draws whether it comes from string fragmentation or from a separately generated channel, with
the probabilities of the model, and only then generates it. The events of the `fragmentation` group
therefore already occur with their correct total probability, and their normalized weights must add
up to their number. A program combines all requests at one mass and with the same settings,
including all chunks, and divides the raw weights of the `fragmentation` events by their mean over
all these events, while all other events keep weight 1. The normalization precedes any selection
and is never done separately for each chunk or request. The command line normalizes its complete
sample in this way and writes the result to `hadronization_weights`.

Unequal weights reduce the statistical precision. A selection of events with normalized weights
$`w_i`$ has the precision of $`N_{\rm eff} = (\sum_i w_i)^2 / \sum_i w_i^2`$ unweighted events, which
depends on the selection, so the number of generated events does not measure the precision of a
weighted sample.

## Using exHad decays in a simulation program

exHad returns the decay products of a particle at rest. The production of the particle, its
lifetime, its decay position and the detector response are computed by the simulation program
([PHYSICS.md](PHYSICS.md#scope-of-the-description)), which places the decays in the laboratory,
takes the branching ratios and lifetimes of the model from `model_info` and, for `hnl`, can let
exHad form the hadrons of decays that the program samples itself.

### Decays in the laboratory frame

The particles of every returned decay are given in the rest frame of the decaying particle, and the
simulation program boosts them to the laboratory frame with the velocity of that particle and places
them at its decay position. For HepMC files, it moves the vertex from the origin to the decay
position. `exhad::Pythia8DecayHandler` performs the boost inside Pythia
([Decays inside a Pythia8 event](#decays-inside-a-pythia8-event-exhadpythia8decayhandler)). The
returned particles are final: the simulation program does not hadronize or decay them again, and it
passes the undecayed particles listed in [Output formats](../README.md#output-formats) to detector
transport.

### Branching ratios, lifetimes and mass intervals: `model_info`

`exhad.model_info(model)` returns, without starting a worker process, a dictionary with the paths of
the tables of branching ratios and of decay lengths or total widths of a model, from which a
simulation program takes the rates with which it selects decays.

| Key | Content |
|---|---|
| `model` | Model name |
| `mother_pdg` | PDG identifier that exHad writes for the decaying particle unless the program gives its own ([Output formats](../README.md#output-formats)) |
| `support_gev` | Mass interval `[lower, upper]` in GeV, with a meaning that differs between the models and is given below the table: for a boson portal, the interval in which exHad forms the hadrons of a decay by fragmenting a quark–antiquark pair or a gluon pair; for `hnl`, the masses of $`N`$ whose decays the descriptions at a given invariant mass of the quark–antiquark pair cover |
| `generation_gev` | Mass intervals `[lower, upper]` in GeV of decay generation: `all` for samples of all decay channels, and for samples of the decays that contain hadrons `hadronic` (a boson portal) or `semileptonic` (`hnl`), as in the model table of the [README](../README.md) |
| `exclusive_gev` | Mass interval `[lower, upper]` in GeV, up to the lower end of `support_gev`, in which every hadronic decay is one of the separately calculated exclusive channels. `hnl`, whose exclusive channels are part of its tables, has no such key. |
| `exclusive_rows` | Those exclusive channels, each with its name in the table (`label`) and the sorted PDG codes of its final particles (`pdg_signature`) |
| `tables` | Absolute paths of the tables of the model: `decay` for the branching ratios (`dark-photon`, `alp-fermion`, the scalar models, `hnl`), `ctau` for the decay length $`c\tau`$ (`dark-photon`, the scalar models) and `widths` for the total widths (`hnl`). `b-l` has none of these tables. |

The ends of `support_gev` differ between the models.

- Its lower end is the mass at which exHad begins to form the hadrons by fragmentation: 1.70 GeV
  for `dark-photon`, 1.911 GeV for `alp-fermion` and 2.00 GeV for `b-l` and the scalar models. It
  is the upper end of `exclusive_gev`, and below it every hadronic decay is one of the exclusive
  channels of `exclusive_rows`
  ([PHYSICS.md](PHYSICS.md#exclusive-decays-below-the-masses-of-string-fragmentation)).
- Its upper end is the highest mass at which exHad generates decays of the model: 5.00 GeV for
  `dark-photon`, `b-l` and `alp-fermion`, and 63.0 GeV for the scalar models. It is the upper end
  of both intervals of `generation_gev`, and a request above it stops with an error.
- The description that forms the hadrons inside the interval depends on the model and on the mass.
  `dark-photon` and `b-l` use the constrained fragmentation model over the whole interval.
  `alp-fermion` and the scalar models use it up to 4 GeV, take each event from it or from the
  parton shower description between 4 and 5 GeV, and use the parton shower description from 5 GeV
  ([PHYSICS.md](PHYSICS.md#decays-into-quarks-and-gluons-above-4-gev)).
- For `hnl` the key gives 0.02–5.27 GeV: the masses of $`N`$ for which the invariant mass $`W`$ of
  the quark–antiquark pair lies, in every decay, within the descriptions that form the hadrons at a
  given $`W`$; those descriptions reach $`W`$ = 5.27 GeV. Each of them follows $`W`$, so the upper
  end bounds $`W`$ alone: exHad generates `hnl` decays at every mass of `generation_gev`, to 40 GeV,
  and its channels with a $`b`$ quark open at 5.42 GeV
  ([PHYSICS.md](PHYSICS.md#hadrons-from-the-quarkantiquark-pair-at-invariant-mass-w)).

Take the mass interval of a request from `generation_gev`, for every model.

The units of the `ctau` and `widths` tables and the couplings to which they refer are given with the
interaction of each model in PHYSICS.md: [`dark-photon`](PHYSICS.md#dark-photon-dark-photon),
[the scalar models](PHYSICS.md#scalars-mixing-with-the-higgs-boson-scalar-central-scalar-lower-scalar-upper-scalar-1809)
and [`hnl`](PHYSICS.md#heavy-neutral-lepton-hnl).

`tables["decay"]` is a JSON list with one entry per decay channel. An entry contains the name of the
channel, the PDG codes of its primary decay products (with `-999` in unused places), the branching
ratio as a list of `[mass, branching ratio]` nodes with masses in GeV, and the squared matrix element
that sets the momentum distribution of the primary decay products, either as a formula in their
energies and the mass of the decaying particle or as a constant. An entry of `hnl` contains three
lists of nodes and three matrix elements, one for mixing with each of the flavours $`e`$, $`\mu`$
and $`\tau`$. In an `hnl` entry whose neutrino has the flavour of the mixing, the table writes this
neutrino as $`\nu_e`$ for all three flavours, and the branching ratio includes the charge conjugate
decay. [PHYSICS.md](PHYSICS.md#numerical-inputs) describes how exHad evaluates the tables at the
requested mass.

For `alp-fermion` and the scalar models, the dictionary contains two further keys.

- `eventcalc_rows` lists the hadronic channels of `tables["decay"]` from which exHad starts a
  hadronic decay, each with its name in the table (`authority_row_id`) and the sorted PDG codes of
  its primary decay products (`pdg_signature`). For `alp-fermion` these are the decays into gluon
  pairs, $`s\bar s`$ pairs, $`c\bar c`$ pairs, $`p\bar p`$ and $`n\bar n`$; the scalar models add the
  decays into $`\pi^+\pi^-`$, $`\pi^0\pi^0`$, $`K^+K^-`$, $`K^0_LK^0_L`$, $`K^0_SK^0_S`$ and $`b\bar b`$ pairs.
- `owner_probabilities` gives the mass nodes in GeV (`masses`) and, for each of these channels
  (`probabilities`), the probability per decay with which exHad starts a hadronic decay from that
  channel, to be interpolated linearly in mass. At every mass these probabilities add up to the
  hadronic branching ratio of `tables["decay"]`.

The branching ratios of exHad's samples of all decay channels relate to these tables as follows.

- For `alp-fermion`, the scalar models and `hnl`, samples of all decay channels use the branching
  ratios of `tables["decay"]`, with the rate of each channel set to zero at and below its threshold
  ([PHYSICS.md](PHYSICS.md#numerical-inputs)). For `alp-fermion` and the scalar models, a simulation
  program that selects decays with this table combines the listed hadronic channels into one
  hadronic channel and generates its decays with `generate`.
- For the scalar models below 5 GeV, the probabilities of the individual hadronic channels differ
  from their branching ratios in the table, because exHad continues the decays into two mesons above
  2 GeV and subtracts their widths from those of gluon pairs and $`s\bar s`$ pairs
  ([PHYSICS.md](PHYSICS.md#hadronic-widths-of-the-four-input-calculations)); the sum over the
  channels is the same.
- For `dark-photon`, samples of all decay channels use exHad's own hadronic width together with the
  widths of the lepton pairs ([PHYSICS.md](PHYSICS.md#nonhadronic-channels-and-their-widths)), and
  the hadronic branching ratio of `tables["decay"]` differs from the one that exHad uses. Between 2
  and 3.5 GeV the table is 3 to 4% lower. Above 3.5 GeV, where the decays into charmed hadrons open
  and both descriptions vary rapidly with mass ([PHYSICS.md](PHYSICS.md#dark-photon-dark-photon)),
  the table is up to 20% lower and up to 3% higher. Between 1.70 and 2 GeV the two differ by at
  most 9% in either direction. A simulation program that selects hadronic decays with this table
  therefore obtains a hadronic fraction that differs from the one of exHad's samples of all decay
  channels.
- For `b-l`, `model_info` gives no branching ratios, and the branching ratios of exHad enter only its
  samples of all decay channels.

### `hnl` decays sampled by the simulation program: `hadronize_hnl`

`Generator("hnl").hadronize_hnl(mass, pdgs, primary_events, *, seed=1)` forms the hadrons from the
quark–antiquark pair of `hnl` decays whose primary decay products the simulation program has
sampled, and returns the charged lepton or neutrino of each decay unchanged.

This method serves simulation programs that compute the widths and the momenta of the primary decay
products themselves, for decays such as $`N \to \ell^- u\bar d`$ through the charged current and
$`N \to \nu q\bar q`$ through the neutral current. It needs no mixings, because `pdgs` fixes the current
and the lepton flavour.

- `mass` is $`m_N`$ in GeV.
- `pdgs` lists the signed PDG codes of the three primary decay products: a charged lepton or a
  neutrino, a quark and an antiquark. A decay through the charged current combines a charged lepton
  with $`u\bar d`$, $`u\bar s`$, $`c\bar d`$, $`c\bar s`$, $`u\bar b`$ or $`c\bar b`$, or with their charge
  conjugates, at total charge zero, for example `[11, 2, -1]` for $`N \to e^- u\bar d`$. A decay
  through the neutral current combines a neutrino with a quark and an antiquark of one flavour
  $`u`$, $`d`$, $`s`$, $`c`$ or $`b`$, for example `[12, 2, -2]` for $`N \to \nu_e u\bar u`$.
- `primary_events` contains one entry per decay: 24 numbers that form three records
  `[px, py, pz, E, mass, PDG, charge, stability]` in the order of `pdgs`, with momenta, energies and
  masses in GeV in the rest frame of $`N`$. exHad does not use the charge and stability entries.
- exHad checks that the PDG codes agree with `pdgs`, that energies and masses are nonnegative, that
  the four-momenta add up to $`(0, 0, 0, m_N)`$ and that the particles lie on their mass shells.
- The invariant mass $`W`$ of the quark–antiquark pair must not exceed $`m_N`$ minus the mass of the
  lepton. It must be at least the lowest $`W`$ at which exHad forms hadrons from the pair: the mass
  of the lightest pair of hadrons that the current can produce (for example $`\pi^+\pi^0`$ for
  $`u\bar d`$), or, for the neutral currents into $`c\bar c`$ and $`b\bar b`$ and the charged
  currents into $`u\bar b`$ and $`c\bar b`$, the values of $`W`$ from which
  [showered Pythia](PHYSICS.md#currents-with-a-bottom-quark-or-a-charm-quark-pair) generates their
  hadrons.
- Each returned event starts with the lepton or neutrino, whose six numbers
  `[px, py, pz, E, mass, PDG]` are those of the input, followed by the hadrons. The four-momenta of
  the hadrons add up to the four-momentum of the quark–antiquark pair, so their invariant mass is
  $`W`$. [PHYSICS.md](PHYSICS.md#hadrons-from-the-quarkantiquark-pair-at-invariant-mass-w) describes
  the hadrons of each current as functions of $`W`$.
- A $`\tau`$ lepton is returned undecayed, and the simulation program decays it.
- Decays into a lepton and one meson, such as $`N \to e^-\pi^+`$, and decays into a lepton and two
  pions are separate channels of the table of `model_info("hnl")`, which the simulation program
  generates itself. The branching ratios of the decays into a lepton and a quark–antiquark pair in
  that table do not include them
  ([PHYSICS.md](PHYSICS.md#decay-channels-and-matching-of-exclusive-and-inclusive-widths)).

## Generation speed

The time per event depends on the model, the mass, the sampler and the machine. Three settings
make large samples fast.

- Keep one generator open for successive requests. Each worker process initializes Pythia once, at
  its first request.
- Leave the sampler at `auto`. At the masses listed in
  [SAMPLING.md](SAMPLING.md#requests-that-use-the-accelerated-sampler) it then uses the accelerated
  sampler, which needs fewer complete Pythia fragmentations per generated event than `reference`.
- Leave the number of worker processes at its default on a machine that is otherwise free, and
  lower it to share the machine with other programs.

## Example programs

The directory `examples/` contains the C++ program `examples/pythia8_decays.cc`, which decays
particles with exHad inside Pythia8 and checks the result.

The program defines in Pythia a neutral spin-zero particle with mass 2 GeV and PDG code 9900035,
keeps the particles for detector transport undecayed
([Particles left for detector transport](#particles-left-for-detector-transport)), and attaches
`exhad::Pythia8DecayHandler` with `alp-fermion`, all decay channels, seed 12345 and batches of 8
decays. It decays 20 such particles, alternately at rest and with momentum 10 GeV along the $`z`$
axis, each produced at the vertex (1, 2, 3) mm with zero lifetime. It checks that the decay products
conserve four-momentum after the boost, have the decaying particle as their mother and start at its
vertex (the program compares the $`z`$ coordinate), and it prints `PASS` when all checks succeed.
[Building with CMake](#building-with-cmake) gives the commands to build and run it.

The same directory contains the Python program `examples/plot_channel_fractions.py`, which draws
the fractions of the decay channels of each model as functions of the mass of the decaying
particle, one figure per model:

```bash
python examples/plot_channel_fractions.py --models dark-photon b-l --points 20 --events 2000 \
    --output-dir figures
```

It takes the models and their mass ranges from `model_info`, generates the events of every mass
point with `Generator` in parallel worker processes, and sorts each event into a channel group by the
hadrons it contains ($`\pi`$, $`2\pi`$ to $`6\pi`$, $`K`$, $`K\bar{K}`$, $`K\bar{K}\pi`$,
$`K\bar{K}\pi\pi`$, $`\eta^{(\prime)}+X`$, $`\pi^0\gamma`$, $`\eta\gamma`$, $`N\bar{N}`$, Other). The
groups are hierarchical: an event with an $`\eta`$ and two kaons is $`\eta^{(\prime)}+X`$. The events contain the particles left for
detector transport, so the program first rebuilds $`\pi^0`$, $`\eta`$, $`K^0_S`$ and $`\eta'`$ from the
invariant masses of their decay products; events whose parents it cannot rebuild are counted as
Other. `--nonhadronic` generates the complete decays (`generate_all`), normalizes to the total width
and draws the fraction of each channel without hadrons as a dashed curve, under the
[label of that decay channel](#labels-of-decay-channels). The six channels with the largest
fraction at any mass of the scan get a curve each, and the remaining ones are summed into one
further curve.
`--workers` sets the number of worker processes, `--seed` the seed of the first mass point. Each
figure is written as PDF and PNG next to a JSON file with the plotted numbers. The plotted
fractions have the statistical uncertainty of the generated sample, which at 2000 events per point
is a few per cent of a fraction of 0.1 and larger for the rare channels.
