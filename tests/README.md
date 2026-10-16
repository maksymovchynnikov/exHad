# Tests and sampler validation

## Running the tests

Run from the repository root after installation:

```bash
python -m unittest discover -s tests -v
```

`tests/test_units.py` and `tests/test_particle_data.py` need no Pythia build.
With a configured installation, `tests/test_physics.py`, which takes about two
minutes, checks for every model that the final particles have finite momenta,
lie on their mass shell and conserve four-momentum. It also checks that a seed
reproduces the same events after other requests; that a request gives the same
events for any number of worker processes; that `hadronize_hnl` returns the
lepton unchanged, hadrons with the invariant mass of the quark-antiquark pair,
and mirrored events for charge conjugate decays; that the `alp-fermion` decays
into charmed hadrons obey the quantum number restrictions; that the branching
ratios of all decay channels add up to one; that the C++ interface gives the
same events as Python; and that a fixed set of requests reproduces stored event
checksums. These tests check the installation and the reproducibility of the
events. The accuracy of the decays is described in
[`docs/PHYSICS.md`](../docs/PHYSICS.md).

## Tests of the accelerated sampler at installation

`tools/configure.py` builds the accelerated sampler in every installation and, before enabling
it, tests the properties of the installed Pythia library on which the constructions above rely;
the configuration fails if any test fails.

The accelerated sampler calls the installed Pythia library through its public interfaces for
fragmentation models, flavour selection and user hooks (`FragmentationModel`, `StringFlav` and
`UserHooks`) and leaves the library unmodified. The Pythia settings listed in
[Conditions under which fragmentation attempts are independent](../docs/SAMPLING.md#conditions-under-which-fragmentation-attempts-are-independent)
are checked each time the sampler starts. The tests at installation are the following.

- **Pythia files.** SHA-256 checksums are compared with those of the Pythia8.317 files whose code
  the constructions above rely on: four headers present in every installation, which declare the
  interfaces for fragmentation models and user hooks and the classes for string fragmentation and
  for the selection of flavour, transverse momentum and momentum fraction; and, where the Pythia
  directory contains its source files, the five source files that implement string fragmentation,
  that selection, the fragmentation model interface, the hadron level and the main Pythia class. A
  missing header, or a file that differs from its checksum, stops the configuration and names the
  file. A Pythia installation without source files is tested by the following tests alone.
- **Mesons from string breaks.** With exHad's fragmentation parameters, the mesons that a string
  break can produce are exactly π, K, η, η′, ρ, $`K^\ast`$, ω and φ.
- **The group ηππ.** For each of ηπ⁺π⁻, ηπ⁰π⁰, η′π⁺π⁻ and η′π⁰π⁰, all sets of at most three such
  mesons whose Pythia decay channels can produce exactly this final state are enumerated. Only the
  set of the three mesons of the final state passes the quantum number test of `alp-fermion`, and
  no sequence of two breaks and a final join of an $`s\bar s`$ string produces it.
- **Four pions.** For π⁺π⁻π⁺π⁻, π⁺π⁻π⁰π⁰ and π⁰π⁰π⁰π⁰, all sets of at most four such mesons are
  enumerated. The sets of at least three mesons that pass the quantum number test of the scalar
  models are exactly the six sets listed in
  [Direct generation of four pions in decays of the scalar models](../docs/SAMPLING.md#direct-generation-of-four-pions-in-decays-of-the-scalar-models),
  and no string break or final join with an s quark produces a π or ρ meson.
- **Flavour probabilities.** For string ends of flavour u, d and s, $`2\times10^7`$ string breaks and
  final joins drawn by Pythia's own flavour selection agree within five standard deviations with
  the probabilities that the sampler computes: that a break yields a hadron, that a break produces
  each isosinglet meson, and that a final join produces each isosinglet meson; and, for u and d
  ends, that a break or a final join produces a π⁰, ρ⁰, charged π or charged ρ meson.
- **Unconstrained strings.** Without forced flavours and without decisions at the final join, the
  fragmentation of the accelerated sampler reproduces unmodified Pythia hadron by hadron for the
  same random numbers: 20,000 $`gg`$ and 20,000 $`s\bar s`$ strings at 3.6 GeV, and 20,000 $`u\bar u`$
  and 20,000 $`s\bar s`$ strings of `b-l` at 2.0 and 3.6 GeV. At 2.0 GeV the $`s\bar s`$ pair is
  fragmented as a low-mass system.
- **Random number initialization.** The accelerated sampler initializes Pythia's random number
  generator with a faster evaluation of Pythia's own seed procedure. For about 32,000 seeds it
  reproduces Pythia's generator state bit for bit, together with the next 128 random numbers. The
  faster evaluation exists in a version that uses the vector instructions of the processor and in
  a version without them; complete `alp-fermion` events generated with the two versions are
  identical at nine combinations of mass, variation and seed.

`.runtime/accelerator/build.json` records the checked files and the results of all tests. If a
test fails, the configuration stops and exHad generates no events until a configuration succeeds.

## Statistical comparisons with the rejection sampler

Two-sample tests of channel groups, charge combinations, multiplicities and momentum
distributions, with Holm correction at familywise level 0.01, found no significant difference
between the accelerated sampler and the rejection sampler at any tested mass and variation. The
group ηππ of `alp-fermion` was not compared in this way.

Each comparison generates independent samples with the two samplers at one mass and variation and
applies a set of two-sample tests to the same observables. Holm's step-down correction adjusts the
p values of the set so that the probability of rejecting equality in any test of the set, if the
two distributions are equal, is at most 1%. Below, $`z`$ denotes a difference in units of its
standard error.

**`b-l`.** Samples of 100,000 events from each sampler were compared at 2.0 GeV, where the light
quark pair uses the flavour chain, at 2.2 and 3.0 GeV in `central`, at 2.1 GeV in `p-high`, and at
4.5 GeV in all four variations. Each comparison used 21 tests: the numbers of particles, charged
particles, strange particles, photons and baryons; the composition of the final states and the
number of each particle species; the energy and transverse momentum of the leading particle; the
total energies of charged particles and of photons; and the invariant mass of the leading π⁺π⁻
pair. The smallest corrected p value was 0.31. At 2.0 GeV, samples of 200,000 events agree in the
mean numbers of particles, charged particles, photons and kaons within $`|z| < 1.3`$. Decaying every
completed string, including those that the sampler discards before decays, gave no event whose
weight exceeds its bounds from the primary hadrons or from the short-lived decays, in 4.1 million
strings between 2.0 and 5.0 GeV in all variations.

**Scalar models.** For `scalar-central`, four comparisons each used 11 tests of channel groups,
sources, compositions of the final states, multiplicities and momentum distributions. The group 4π
was compared with the rejection sampler in which only 4π has a nonzero weight, at 3.5 GeV (100,000
events of the accelerated sampler and 32,000 of the rejection sampler) and at 4.5 GeV (30,000 and
6,000 events). The other groups were compared with the rejection sampler in which the weight of 4π
is zero, at 3.5 and 4.5 GeV (100,000 events each). No test rejected equality. For 4π events at
3.0 GeV (60,000 and 16,000 events), the sets of primary hadrons agree (χ² p value 0.48), with a
fraction of sets of three hadrons of 0.4844 and 0.4854 ($`z = -0.21`$). At 4.0 GeV the fraction of 4π
among string fragmentation events agrees with $`P_{4\pi}`$ within $`|z| \le 1.4`$ and with the
rejection sampler within $`|z| \le 0.9`$ in all four variations.

**`alp-fermion`.** For the channel groups other than ηππ, 3π and ππγ, 27 tests of the same kinds
compared the accelerated sampler with the rejection sampler in which the weights of ηππ are zero,
at 3.0, 3.6, 4.0 (all four variations), 4.4 and 5.0 GeV, with 90,000 to 300,000 events per sample. At
5.0 GeV, above the
[transition to showered Pythia](../docs/PHYSICS.md#decays-into-quarks-and-gluons-above-4-gev), the comparison
used the string fragmentation events of both samplers. No test rejected equality. At 3.6 GeV, in
samples of 300,000 events, events in which all primary hadrons are isosinglet mesons, the histories
enhanced by forced isosinglet attempts, make up 0.554% and 0.561% of the samples ($`z = -0.35`$). A
sampler that uses the bound of gluon strings without forced isosinglet attempts gives 0.313%
($`z = -14.6`$), so the comparison detects an incorrect treatment of these histories. For the group
ηππ, the installation tests verify that the
[construction](../docs/SAMPLING.md#direct-generation-of-the-group-ηππ-in-alp-fermion-decays) includes every
fragmentation history of these final states.

The equivalence follows from the constructions of the accelerated sampler; the comparisons test
their implementation at the stated masses and variations.
