#pragma once

// Isolated finite kernel for stock Pythia 8.317 StringFlav::combine.
#include "Pythia8/FragmentationFlavZpT.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace exhad::accelerator {

struct MesonOutcome {
  int pdg = 0;  // Zero is the stock combine/finite-loop failure outcome.
  long double probability = 0;
};

class FiniteMesonLaw {
 public:
  explicit FiniteMesonLaw(std::vector<MesonOutcome> outcomes)
      : outcomes_(std::move(outcomes)) {
    long double total = 0;
    for (const auto& outcome : outcomes_) {
      if (!std::isfinite(outcome.probability) || outcome.probability < 0)
        throw std::runtime_error("invalid finite meson probability");
      total += outcome.probability;
    }
    if (outcomes_.empty() || std::abs(total - 1.L) > 2.e-14L)
      throw std::runtime_error("finite meson law is not normalized");
  }

  const std::vector<MesonOutcome>& outcomes() const {
    return outcomes_;
  }
  long double probability(int pdg) const {
    for (const auto& outcome : outcomes_)
      if (outcome.pdg == pdg) return outcome.probability;
    return 0;
  }

 private:
  std::vector<MesonOutcome> outcomes_;
};

// The derived object is initialized against the owning Pythia Info object.
// Protected initialized arrays are the authoritative numerical inputs;
// no independently copied mixing-angle convention or tune defaults enter.
// Rebuild a law after changing settings/reinitializing flavour parameters.
class FiniteMesonChoice : public Pythia8::StringFlav {
 public:
  void init() override {
    initialized_ = false;
    if (mode("Fragmentation:model") != 0)
      throw std::runtime_error("finite meson kernel requires ordinary Lund StringFlav");
    if (flag("ClosePacking:doClosePacking"))
      throw std::runtime_error("finite meson kernel does not support changing close-packing parameters");
    if (flag("Ropewalk:RopeHadronization") || flag("Ropewalk:doFlavour"))
      throw std::runtime_error("finite meson kernel does not support flavour-rope parameter changes");
    Pythia8::StringFlav::init();
    initialized_ = true;
  }

  void init(double, double, double) override {
    initialized_ = false;
    throw std::runtime_error("finite meson kernel does not support dynamic flavour reinitialization");
  }

  FiniteMesonLaw oneCall(const Pythia8::FlavContainer& first,
                        const Pythia8::FlavContainer& second) const {
    if (!initialized_) throw std::runtime_error("finite meson kernel is not initialized");
    const int firstAbs = std::abs(first.id), secondAbs = std::abs(second.id);
    if (firstAbs < 1 || firstAbs > 3 || secondAbs < 1 || secondAbs > 3
        || (first.id > 0) == (second.id > 0)
        || first.nPop || second.nPop || first.idPop || second.idPop
        || first.idVtx || second.idVtx)
      throw std::runtime_error("finite meson kernel supports ordinary light q-qbar combinations only");
    const int largest = std::max(firstAbs, secondAbs);
    const int smallest = std::min(firstAbs, secondAbs);
    const int flav = largest < 3 ? 0 : largest - 2;
    if (!(mesonRateSum[flav] > 0))
      throw std::runtime_error("nonpositive meson multiplet normalization");
    std::map<int, long double> probabilities{{0, 0}};
    for (int spin = 0; spin < 6; ++spin) {
      const long double spinProbability =
          static_cast<long double>(mesonRate[flav][spin]) / mesonRateSum[flav];
      if (!std::isfinite(spinProbability) || spinProbability < 0)
        throw std::runtime_error("invalid initialized meson multiplet weight");
      if (!spinProbability) continue;
      const int multiplet = mesonMultipletCode[spin];
      if (largest != smallest) {
        int sign = largest % 2 == 0 ? 1 : -1;
        if ((largest == firstAbs && first.id < 0)
            || (largest == secondAbs && second.id < 0)) sign = -sign;
        probabilities[sign * (100 * largest + 10 * smallest + multiplet)]
            += spinProbability;
      } else {
        const long double firstMix = mesonMix1[flav][spin];
        const long double secondMix = mesonMix2[flav][spin];
        if (!(firstMix >= 0 && firstMix <= secondMix && secondMix <= 1))
          throw std::runtime_error("invalid initialized neutral-meson mixing");
        const long double mixProbabilities[] =
            {firstMix, secondMix - firstMix, 1 - secondMix};
        for (int member = 0; member < 3; ++member) {
          const int pdg = 110 * (member + 1) + multiplet;
          const long double probability = spinProbability * mixProbabilities[member];
          // Stock compare is suppression < uniform, so out-of-unit-range
          // settings (if supplied directly) behave as clipped probabilities.
          const long double suppression = pdg == 221
              ? std::clamp(etaSup, 0., 1.) : pdg == 331
              ? std::clamp(etaPrimeSup, 0., 1.) : 1.;
          probabilities[pdg] += probability * suppression;
          probabilities[0] += probability * (1 - suppression);
        }
      }
    }
    std::vector<MesonOutcome> outcomes;
    for (const auto& entry : probabilities)
      if (entry.second > 0 || entry.first == 0)
        outcomes.push_back({entry.first, entry.second});
    return FiniteMesonLaw(std::move(outcomes));
  }

 private:
  bool initialized_ = false;
};

}  // namespace exhad::accelerator
