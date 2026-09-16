// The worker's baryon guard on synthetic final states: an invariant violation
// only below the physical pair threshold 2 m_p, never at or just above it.
#define main exhad_worker_main
#include "../cpp/src/main.cc"
#undef main

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Pythia pythia(argv[1], false);
  const double threshold = 2.0 * pythia.particleData.m0(2212);
  Event event;
  event.init("(synthetic)", &pythia.particleData);
  int failures = 0;
  auto check = [&](std::vector<int> ids, double mass, bool expected, int status = 1) {
    event.reset();
    for (int id : ids) event.append(id, status, 0, 0, 0., 0., 0., 0., 0.);
    const bool below = baryonBelowPairThreshold(event, mass, pythia.particleData);
    std::cout << (below == expected ? "ok  " : "FAIL") << " M=" << std::setprecision(17) << mass
              << " n=" << ids.size() << " below=" << below << "\n";
    failures += below != expected;
  };
  check({2212, -2212}, 1.8770, false);
  check({2212, -2212}, 1.8799, false);                  // just above 2 m_p
  check({2112, -2212}, 1.8779, false);
  check({2212, -2212}, threshold, false);
  check({2212, -2212}, 1.8760, true);
  check({2212, -2212}, threshold * (1.0 - 1e-12), true);
  check({3122, -2212, 211}, 1.5, true);
  check({211, -211}, 1.0, false);
  check({10213, 111}, 1.5, false);                      // an L=1 meson carries no baryon number
  check({2212, -2212}, 1.0, false, -84);                // not final
  if (threshold != 2.0 * 0.93827) { std::cout << "FAIL Pythia 8.317 m0(2212)\n"; ++failures; }
  return failures ? 1 : 0;
}
