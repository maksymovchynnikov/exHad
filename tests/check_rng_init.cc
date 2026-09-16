// Standalone exact-state and next-draw checks against pinned Pythia 8.317.
#include "fast_pythia_seed.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::mt19937 generator(20260912);
  std::vector<int> seeds{1, 2, 900000000, 30081, 30082, 30083, 19780503};
  for (int i = 0; i < 30000; ++i) seeds.push_back(1 + generator() % 900000000);
  Pythia8::Rndm original(1), fast(1);
  for (int seed : seeds) {
    original.init(seed);
    exhad_rng::initialize(fast, seed);
    const auto a = original.getState(), b = fast.getState();
    if (!(a == b) || a.seed != b.seed || a.cd != b.cd || a.cm != b.cm) {
      std::cerr << "state mismatch for seed " << seed << '\n'; return 1;
    }
    for (int n = 0; n < 100; ++n)
      if (original.flat() != fast.flat()) {
        std::cerr << "draw mismatch for seed " << seed << '\n'; return 2;
      }
  }
  volatile double checksum = 0.;
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (int seed : seeds) { original.init(seed); checksum += original.flat(); }
    const auto middle = std::chrono::steady_clock::now();
    for (int seed : seeds) { exhad_rng::initialize(fast, seed); checksum += fast.flat(); }
    const auto end = std::chrono::steady_clock::now();
    const double reference = std::chrono::duration<double>(middle-start).count();
    const double optimized = std::chrono::duration<double>(end-middle).count();
    std::cout << "{\"seeds\":" << seeds.size() << ",\"draws_compared_per_seed\":100,"
              << "\"reference_seconds\":" << reference << ",\"optimized_seconds\":" << optimized
              << ",\"speedup\":" << reference/optimized << "}\n";
  }
  return checksum == 0.;
}
