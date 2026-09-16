#include "fast_pythia_seed_simd.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static bool identical(const Pythia8::RndmState& a, const Pythia8::RndmState& b) {
  return a.i97==b.i97 && a.j97==b.j97 && a.seed==b.seed && a.sequence==b.sequence
      && std::memcmp(a.u,b.u,sizeof a.u)==0
      && std::memcmp(&a.c,&b.c,sizeof(double))==0
      && std::memcmp(&a.cd,&b.cd,sizeof(double))==0
      && std::memcmp(&a.cm,&b.cm,sizeof(double))==0;
}

int main(int argc, char** argv) {
  try {
    const int randomCount=argc>1 ? std::stoi(argv[1]) : 1000;
    if (randomCount<1) throw std::invalid_argument("positive random seed count required");
    std::vector<int> seeds{1,2,168,169,170,30081,30082,30083,19780503,
      899999999,900000000,900000001,std::numeric_limits<int>::max()};
    for (int seed=1;seed<=30082;++seed) seeds.push_back(seed);
    for (int ij=0;ij<31329;ij+=137) {
      const int seed=ij*30082;
      for (int delta:{-1,0,1,168,169,170})
        if (seed+delta>0) seeds.push_back(seed+delta);
    }
    std::mt19937 generator(20260913);
    for (int i=0;i<randomCount;++i)
      seeds.push_back(1+generator()%std::numeric_limits<int>::max());
    Pythia8::Rndm original(1), candidate(1);
    for (int seed:seeds) {
      original.init(seed);
      const auto expected=original.getState();
      exhad_rng_simd::initialize(candidate,seed);
      if (!identical(expected,exhad_rng::exactState(seed))
          || !identical(expected,exhad_rng_simd::vectorState(seed))
          || !identical(expected,candidate.getState()))
        throw std::runtime_error("RNG state mismatch for seed="+std::to_string(seed));
      for (int i=0;i<128;++i)
        if (original.flat()!=candidate.flat())
          throw std::runtime_error("RNG stream mismatch for seed="+std::to_string(seed));
      if (!identical(original.getState(),candidate.getState()))
        throw std::runtime_error("post-draw RNG state mismatch");
    }
    for (int seed:{1,19780503,900000001,std::numeric_limits<int>::max()}) {
      Pythia8::Rndm freshOriginal, freshCandidate;
      freshOriginal.init(seed);
      exhad_rng_simd::initialize(freshCandidate,seed);
      if (!identical(freshOriginal.getState(),freshCandidate.getState()))
        throw std::runtime_error("first-initialization mismatch");
    }
    std::cout<<"{\"verified_seeds\":"<<seeds.size()
      <<",\"draws_per_seed\":128,\"all_state_fields_bitwise_equal\":true"
      <<",\"first_initialization_equal\":true}\n";
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
