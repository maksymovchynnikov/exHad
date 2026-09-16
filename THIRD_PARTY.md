# Third-party software and numerical inputs

- Pythia8.317 is a separately installed dependency. Its source, library and
  XML payload are not distributed here. Retain its own license and citation
  requirements when redistributing a linked executable.
- DeLiVeR is fetched from `preimitz/DeLiVeR` at commit
  `7b2bbd79fbacfed9a01d24e903762b4759b9e4a2`. The checkout inspected here
  contains no top-level license. exHad does not relicense or vendor that code;
  check its upstream terms before redistributing it.
- The 29 dark-photon R tables in `data/dark-photon/primary_modes.json` are
  taken from ReD-DeLiVeR (`anafoguel/ReD-DeLiVeR`, commit
  `30dfb69042ecf2a5ae76c50fd5282ef1b33df531`, MIT license).
- The decay kinematics in `exhad/core/kinematics.py` and the terminal
  convention in `exhad/core/eventcalc.py` originated in EventCalc, as did the
  rate tables `data/*/eventcalc_*`. Their BSD-3-Clause notice is retained in
  `LICENSE-EventCalc`. They do not import or require EventCalc.
- Numerical phenomenology inputs and experimental tables retain their
  provenance and citations. Citation is not a claim of ownership or a grant
  to relicense upstream software. hipsofcobra is cited, not vendored.

See [PHYSICS.md](docs/PHYSICS.md) for the model-to-source mapping.
