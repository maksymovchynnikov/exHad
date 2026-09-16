#pragma once

// Bit-exact SIMD initializer for the conditional accelerator.
// Sixteen independent 48-bit words are evaluated together; every state is computed from its seed on demand.
// EXHAD_RNG_FORCE_SCALAR provides the exact scalar path for replay validation.
#include "fast_pythia_seed.h"
#include <array>
#include <cstdint>
#if defined(__aarch64__) && !defined(EXHAD_RNG_FORCE_SCALAR)
#include <arm_neon.h>
#endif

namespace exhad_rng_simd {

#if defined(__aarch64__) && !defined(EXHAD_RNG_FORCE_SCALAR)
inline uint16x8_t reduce178(uint16x8_t value) {
  const auto modulus = vdupq_n_u16(178);
  value = vsubq_u16(value, vandq_u16(vcgeq_u16(value, modulus), modulus));
  return vsubq_u16(value, vandq_u16(vcgeq_u16(value, modulus), modulus));
}

inline uint16x8_t nextLinear(uint16x8_t value) {
  // x=53*l+1 is at most 8905. floor(x*193/32768) underestimates
  // floor(x/169) by at most one, so the final subtract is exact.
  const auto x = vmlaq_n_u16(vdupq_n_u16(1), value, 53);
  const auto quotient = vreinterpretq_u16_s16(vqdmulhq_n_s16(
      vreinterpretq_s16_u16(x), 193));
  const auto remainder = vmlsq_n_u16(x, quotient, 169);
  const auto modulus = vdupq_n_u16(169);
  return vsubq_u16(remainder,
      vandq_u16(vcgeq_u16(remainder, modulus), modulus));
}

inline Pythia8::RndmState vectorState(int seed) {
  if (seed <= 0) throw std::invalid_argument("vectorState requires a positive seed");
  static const exhad_rng::SeedTables tables;
  // Padding permits a three-table SIMD gather without out-of-bounds loads.
  static const auto paddedPower = [] {
    std::array<unsigned char, 192> result{};
    for (unsigned n = 0; n < tables.power.size(); ++n)
      result[n] = tables.power[n];
    return result;
  }();
  const uint8x16x4_t table0 = {{vld1q_u8(paddedPower.data()),
      vld1q_u8(paddedPower.data()+16), vld1q_u8(paddedPower.data()+32),
      vld1q_u8(paddedPower.data()+48)}};
  const uint8x16x4_t table1 = {{vld1q_u8(paddedPower.data()+64),
      vld1q_u8(paddedPower.data()+80), vld1q_u8(paddedPower.data()+96),
      vld1q_u8(paddedPower.data()+112)}};
  const uint8x16x4_t table2 = {{vld1q_u8(paddedPower.data()+128),
      vld1q_u8(paddedPower.data()+144), vld1q_u8(paddedPower.data()+160),
      vld1q_u8(paddedPower.data()+176)}};
  const int ij = (seed / 30082) % 31329;
  const int kl = seed % 30082;
  unsigned i = tables.logarithm[(ij / 177) % 177 + 2];
  unsigned j = tables.logarithm[ij % 177 + 2];
  unsigned k = tables.logarithm[(kl / 169) % 178 + 1];
  const auto* linear = tables.linear.data() + tables.phase[kl % 169];
  Pythia8::RndmState state;
  for (unsigned index = 0; index < 97; index += 16) {
    std::uint16_t aa[16]{}, bb[16]{}, cc[16]{}, ll[16]{};
    const unsigned count = std::min(16U, 97U-index);
    for (unsigned lane = 0; lane < count; ++lane) {
      aa[lane] = i;
      bb[lane] = j;
      cc[lane] = k;
      ll[lane] = linear[(index+lane)*48];
      const auto& jump = tables.jump48;
      const unsigned nextI = (jump[0][0]*i+jump[0][1]*j+jump[0][2]*k)%178;
      const unsigned nextJ = (jump[1][0]*i+jump[1][1]*j+jump[1][2]*k)%178;
      const unsigned nextK = (jump[2][0]*i+jump[2][1]*j+jump[2][2]*k)%178;
      i = nextI;
      j = nextJ;
      k = nextK;
    }
    auto a0 = vld1q_u16(aa), a1 = vld1q_u16(aa+8);
    auto b0 = vld1q_u16(bb), b1 = vld1q_u16(bb+8);
    auto c0 = vld1q_u16(cc), c1 = vld1q_u16(cc+8);
    auto l0 = vld1q_u16(ll), l1 = vld1q_u16(ll+8);
    std::uint32_t wordChunks[2][16];
    for (unsigned chunk = 0; chunk<2; ++chunk) {
      auto word0 = vdupq_n_u32(0), word1 = word0, word2 = word0, word3 = word0;
      for (unsigned bit = 0; bit<24; ++bit) {
        const auto next0 = reduce178(vaddq_u16(vaddq_u16(a0, b0), c0));
        const auto next1 = reduce178(vaddq_u16(vaddq_u16(a1, b1), c1));
        a0 = b0;
        b0 = c0;
        c0 = next0;
        a1 = b1;
        b1 = c1;
        c1 = next1;
        const auto exponent = vcombine_u8(vmovn_u16(next0), vmovn_u16(next1));
        const auto powers = vorrq_u8(vqtbl4q_u8(table0, exponent), vorrq_u8(
            vqtbl4q_u8(table1, vsubq_u8(exponent, vdupq_n_u8(64))),
            vqtbl4q_u8(table2, vsubq_u8(exponent, vdupq_n_u8(128)))));
        l0 = nextLinear(l0);
        l1 = nextLinear(l1);
        const auto mask = vdupq_n_u16(1);
        const auto bit0 = vandq_u16(vshrq_n_u16(vmulq_u16(
            l0, vmovl_u8(vget_low_u8(powers))), 5), mask);
        const auto bit1 = vandq_u16(vshrq_n_u16(vmulq_u16(
            l1, vmovl_u8(vget_high_u8(powers))), 5), mask);
        word0 = vorrq_u32(vshlq_n_u32(word0, 1), vmovl_u16(vget_low_u16(bit0)));
        word1 = vorrq_u32(vshlq_n_u32(word1, 1), vmovl_u16(vget_high_u16(bit0)));
        word2 = vorrq_u32(vshlq_n_u32(word2, 1), vmovl_u16(vget_low_u16(bit1)));
        word3 = vorrq_u32(vshlq_n_u32(word3, 1), vmovl_u16(vget_high_u16(bit1)));
      }
      vst1q_u32(wordChunks[chunk], word0);
      vst1q_u32(wordChunks[chunk]+4, word1);
      vst1q_u32(wordChunks[chunk]+8, word2);
      vst1q_u32(wordChunks[chunk]+12, word3);
    }
    for (unsigned lane = 0; lane<count; ++lane) {
      const std::uint64_t bits = (std::uint64_t(wordChunks[0][lane])<<24)
          | wordChunks[1][lane];
      state.u[index+lane] = static_cast<double>(bits)*0x1p-48;
    }
  }
  state.c = 362436.*0x1p-24;
  state.cd = 7654321.*0x1p-24;
  state.cm = 16777213.*0x1p-24;
  state.i97 = 96;
  state.j97 = 32;
  state.seed = seed;
  state.sequence = 0;
  return state;
}
#else
inline Pythia8::RndmState vectorState(int seed) {
  return exhad_rng::exactState(seed);
}
#endif

inline void initialize(Pythia8::Rndm& random, int seed) {
  // Identical first-initialization and zero/negative-seed policy to
  // exhad_rng::initialize: setState alone does not set Pythia's initRndm flag.
  if (seed <= 0 || random.getState().seed <= 0) random.init(seed);
  else random.setState(vectorState(seed));
}
}  // namespace exhad_rng_simd
