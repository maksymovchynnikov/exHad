#pragma once

// Exact initialization of the pinned Pythia 8.317 Marsaglia-Zaman state.
// The multiplicative recurrence in F_179 is evaluated in logarithms; the
// 48 binary fractions are assembled as an integer. The state equals that of
// Pythia's Rndm::init(seed) bit for bit.
#include "Pythia8/Basics.h"
#include <array>
#include <cstdint>
#include <stdexcept>

namespace exhad_rng {
struct SeedTables {
  std::array<unsigned char, 179> logarithm{};
  std::array<unsigned char, 178> power{};
  std::array<unsigned char, 169> phase{};
  std::array<unsigned char, 97 * 48 + 169> linear{};
  std::array<std::array<unsigned, 3>, 3> jump48{};

  SeedTables() {
    // Find a primitive element of the nonzero residues modulo 179.
    bool found = false;
    for (unsigned base = 2; base < 179 && !found; ++base) {
      unsigned value = 1, period = 0;
      do {
        power[period] = static_cast<unsigned char>(value);
        logarithm[value] = static_cast<unsigned char>(period);
        value = value * base % 179;
        ++period;
      } while (value != 1 && period < 178);
      found = period == 178 && value == 1;
    }
    if (!found) throw std::logic_error("no primitive element modulo 179");
    unsigned value = 0;
    for (unsigned n = 0; n < linear.size(); ++n) {
      linear[n] = static_cast<unsigned char>(value);
      if (n < 169) phase[value] = static_cast<unsigned char>(n);
      value = (53 * value + 1) % 169;
      if (n == 168 && value != 0)
        throw std::logic_error("unexpected seed LCG period");
    }
    for (unsigned basis = 0; basis < 3; ++basis) {
      unsigned i = basis == 0, j = basis == 1, k = basis == 2;
      for (unsigned n = 0; n < 48; ++n) {
        const unsigned next = (i + j + k) % 178;
        i = j;
        j = k;
        k = next;
      }
      jump48[0][basis] = i;
      jump48[1][basis] = j;
      jump48[2][basis] = k;
    }
  }
};

inline Pythia8::RndmState exactState(int seed) {
  if (seed <= 0) throw std::invalid_argument("exactState requires a positive seed");
  static const SeedTables tables;
  const int ij = (seed / 30082) % 31329;
  const int kl = seed % 30082;
  unsigned i = tables.logarithm[(ij / 177) % 177 + 2];
  unsigned j = tables.logarithm[ij % 177 + 2];
  unsigned k = tables.logarithm[(kl / 169) % 178 + 1];
  const unsigned char* linear = tables.linear.data() + tables.phase[kl % 169] + 1;
  Pythia8::RndmState state;
  // Advance to the start of four successive 48-bit words and compute their
  // recurrences together (instruction-level parallelism on one core). The
  // resulting state equals Pythia's Rndm::init(seed) bit for bit.
  for (int index = 0; index < 97; index += 4) {
    unsigned a[4], b[4], c[4];
    std::uint64_t bits[4]{};
    const unsigned count = index == 96 ? 1 : 4;
    for (unsigned lane = 0; lane < count; ++lane) {
      a[lane] = i;
      b[lane] = j;
      c[lane] = k;
      const auto& jump = tables.jump48;
      const unsigned nextI = (jump[0][0]*i + jump[0][1]*j + jump[0][2]*k) % 178;
      const unsigned nextJ = (jump[1][0]*i + jump[1][1]*j + jump[1][2]*k) % 178;
      const unsigned nextK = (jump[2][0]*i + jump[2][1]*j + jump[2][2]*k) % 178;
      i = nextI;
      j = nextJ;
      k = nextK;
    }
    for (int bit = 0; bit < 48; ++bit) {
      for (unsigned lane = 0; lane < count; ++lane) {
        unsigned exponent = a[lane] + b[lane] + c[lane];
        if (exponent >= 178) exponent -= 178;
        if (exponent >= 178) exponent -= 178;
        a[lane] = b[lane];
        b[lane] = c[lane];
        c[lane] = exponent;
        bits[lane] = (bits[lane] << 1) |
            ((linear[lane*48+bit] * tables.power[exponent] >> 5) & 1U);
      }
    }
    for (unsigned lane = 0; lane < count; ++lane)
      state.u[index+lane] = static_cast<double>(bits[lane]) * 0x1p-48;
    linear += count*48;
  }
  state.c = 362436. * 0x1p-24;
  state.cd = 7654321. * 0x1p-24;
  state.cm = 16777213. * 0x1p-24;
  state.i97 = 96;
  state.j97 = 32;
  state.seed = seed;
  state.sequence = 0;
  return state;
}

inline void initialize(Pythia8::Rndm& random, int seed) {
  // setState does not set Pythia's private initRndm flag. The worker's
  // generators have already been initialized by Pythia::init; retain the
  // public initializer for other callers and its zero/time/negative policy.
  if (seed <= 0 || random.getState().seed <= 0) random.init(seed);
  else random.setState(exactState(seed));
}
} // namespace exhad_rng
