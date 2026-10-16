# Installation details

Run these commands from the repository root, after the Python installation
steps in the [README](../README.md#installation).

`tools/setup.py` performs four steps, which can also be run one by one; the
option `--jobs N` sets the number of parallel compiler processes (default 4).

1. `python tools/fetch_inputs.py` downloads, at a fixed version, the program
   [DeLiVeR](https://github.com/preimitz/DeLiVeR) of arXiv:2201.01788, which
   computes the hadronic form factors of vector currents, into `external/`. The
   tables of [ReD-DeLiVeR](https://github.com/anafoguel/ReD-DeLiVeR) that exHad
   uses are part of `data/` and are not downloaded
   ([`THIRD_PARTY.md`](../THIRD_PARTY.md)). Event generation needs no network
   access.
2. `make -C cpp PYTHIA8_DIR=/absolute/path/to/pythia8317` builds `cpp/exhad`,
   the program that generates Pythia events for exHad. The build stops for any
   Pythia version other than 8.317.
3. `make -C cpp check` generates one event to check that the program links to
   the Pythia library and reads the Pythia8.317 data files.
4. `python tools/configure.py` builds the programs `.runtime/exhad-decay`, which
   decays unstable particles, and `.runtime/exhad-shower`, which generates the
   parton shower description, and the accelerated sampler
   `.runtime/accelerator/exhad-batch` ([`docs/SAMPLING.md`](../docs/SAMPLING.md)).
   It tests them against the installed Pythia library and writes
   `.runtime/current.json`. This step takes about a minute. If a test fails, the
   step fails, and exHad generates no events.
