#pragma once

#include <string>
#include <vector>

namespace Pythia8 {
  class ParticleData;
  class Event;
  class Rndm;
}

// Expected source quantum numbers and the enforced selection rules.
struct SourceQN {
  int sourceJ2   = 0;   // 2*J
  int expectedP  = +1;
  int expectedC  = +1;
  int expectedG  = +1;

  bool enforceCDet     = true;  // deterministic C-eigenstate veto
  bool enforceGProxy   = false;
  bool enforceJPC2Body = true;  // generic 2-body JPC solver
  bool undetVeto       = false; // parity/isospin undetermined => veto

  std::vector<int> allowedI2 = {0}; // 2*I
  int expectedI3_2 = 0;             // 2*I3
};

// The decaying particle itself, as its quantum numbers: 2*J, P, C (cGiven false
// when the source is not a C eigenstate) and the electric charge of the source
// current.  Whether a spin-1 source carries the parent's spin is not one of
// them and travels separately.
struct DecayingParticle {
  int j2 = 0, parity = 0, c = 0, charge = 0;
  bool cGiven = false;

  // The four protocol tokens "<2J> <P> <C|none> <charge>".
  static DecayingParticle read(const std::string& j2, const std::string& parity,
                               const std::string& c, const std::string& charge);

  // The selection rules of this source, before any component projection.
  SourceQN baseline() const;
};

class DiscreteSymmetryFilter {
public:
  DiscreteSymmetryFilter(const Pythia8::ParticleData& pd, SourceQN cfg);

  bool veto(const Pythia8::Event& event, Pythia8::Rndm& rndm) const;

  // Deterministic support-only form of the same source projector.  Unlike
  // veto(), a state with any nonzero target-isospin projection is retained;
  // no random Clebsch--Gordan accept/reject draw is made.
  bool supports(const Pythia8::Event& event) const;

  // Probability that veto() keeps a record with these final hadrons.
  double acceptance(const std::vector<int>& hadronIds) const;

private:
  bool vetoImpl(const Pythia8::Event& event, Pythia8::Rndm* rndm,
                bool supportOnly) const;
  long double projection(const std::vector<int>& hadronIds, bool supportOnly,
                         bool& stochastic) const;
  const Pythia8::ParticleData* pd_;
  SourceQN cfg_;
};
