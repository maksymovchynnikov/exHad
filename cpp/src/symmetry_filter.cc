#include "symmetry_filter.h"
#include "isospin_cg.h"

#include "Pythia8/Pythia.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Pythia8;

// One whole integer token; "2abc" and an empty field are malformed.
static int wholeNumber(const std::string& text, const char* what) {
  std::istringstream in(text);
  int value = 0;
  std::string extra;
  if (!(in >> value) || (in >> extra))
    throw std::runtime_error(std::string("malformed ") + what + ": " + text);
  return value;
}

DecayingParticle DecayingParticle::read(const std::string& j2, const std::string& parity,
                                        const std::string& c, const std::string& charge) {
  DecayingParticle source;
  source.j2 = wholeNumber(j2, "2J");
  source.parity = wholeNumber(parity, "parity");
  source.cGiven = c != "none";
  source.c = source.cGiven ? wholeNumber(c, "C") : 0;
  source.charge = wholeNumber(charge, "source charge");
  if ((source.j2 != 0 && source.j2 != 2) || std::abs(source.parity) != 1
      || (source.cGiven && std::abs(source.c) != 1) || std::abs(source.charge) > 1
      || (source.j2 == 0 && source.charge != 0))
    throw std::runtime_error("source quantum numbers are 2J in {0, 2}, P and C in {-1, +1} with C "
                             "possibly none, and a charge in {-1, 0, +1} that vanishes for spin 0");
  return source;
}

// A spin-0 source is an isosinglet carrying the G proxy, with G = C (-1)^I = C;
// a spin-1 one is isovector when charged and isoscalar or isovector when
// neutral, with 2*I3 twice the charge.  C is enforced exactly when it is given.
SourceQN DecayingParticle::baseline() const {
  SourceQN qn;
  qn.sourceJ2 = j2;
  qn.expectedP = parity;
  qn.expectedC = c;
  qn.enforceCDet = cGiven;
  qn.enforceGProxy = j2 == 0;
  if (qn.enforceGProxy) qn.expectedG = c;
  qn.allowedI2 = j2 == 0 ? std::vector<int>{0}
      : charge != 0 ? std::vector<int>{2} : std::vector<int>{0, 2};
  qn.expectedI3_2 = 2 * charge;
  return qn;
}

// Baryon test by PDG id alone: 1000 <= |id| < 1000000000.
static bool isBaryonPdgId(int id) {
  const int a = std::abs(id);
  return a >= 1000 && a < 1000000000;
}

// G-parity table: integer-isospin MESONS only (pi, rho, eta, eta', omega, phi).
static const std::unordered_map<int, int> G_TABLE = {
  {  211, -1 }, { -211, -1 }, { 111, -1 }, {  213, +1 }, { -213, +1 }, { 113, +1 },
  {  221, +1 }, {  331, +1 }, {  223, -1 }, {  333, -1 }
};

// Intrinsic C of self-conjugate neutral mesons: pi0, eta, eta', rho0, omega,
// phi, eta_c(1S,2S), J/psi, psi(2S), chi_c0, chi_c1, chi_c2, h_c.
static const std::unordered_map<int, int> C_TABLE = {
  { 111, +1 }, { 221, +1 }, { 331, +1 }, { 113, -1 }, { 223, -1 }, { 333, -1 },
  { 441, +1 }, { 100441, +1 }, { 443, -1 }, { 100443, -1 },
  { 10441, +1 }, { 20443, +1 }, { 445, +1 }, { 10443, -1 }
};

// JP table (2*J, P).
static const std::unordered_map<int, std::pair<int, int>> JP_TABLE = {
  // pseudoscalar mesons 0^-
  {  211, {0, -1} }, { -211, {0, -1} }, { 111, {0, -1} }, {  221, {0, -1} }, {  331, {0, -1} },
  {  321, {0, -1} }, { -321, {0, -1} }, { 311, {0, -1} }, { -311, {0, -1} }, {  130, {0, -1} }, {  310, {0, -1} },
  {  411, {0, -1} }, { -411, {0, -1} }, { 421, {0, -1} }, { -421, {0, -1} }, {  431, {0, -1} }, { -431, {0, -1} },
  {  441, {0, -1} }, { 100441, {0, -1} },
  // vector mesons 1^-
  {  213, {2, -1} }, { -213, {2, -1} }, { 113, {2, -1} }, {  223, {2, -1} }, {  333, {2, -1} },
  {  313, {2, -1} }, { -313, {2, -1} }, { 323, {2, -1} }, { -323, {2, -1} },
  {  413, {2, -1} }, { -413, {2, -1} }, { 423, {2, -1} }, { -423, {2, -1} }, {  433, {2, -1} }, { -433, {2, -1} },
  {  443, {2, -1} }, { 100443, {2, -1} },
  // low positive-parity charmonia
  { 10441, {0, +1} }, { 10443, {2, +1} }, { 20443, {2, +1} }, { 445, {4, +1} },
  // N 1/2^+, Delta 3/2^+
  { 2212, {1, +1} }, { -2212, {1, -1} }, { 2112, {1, +1} }, { -2112, {1, -1} },
  { 2224, {3, +1} }, { -2224, {3, -1} }, { 2214, {3, +1} }, { -2214, {3, -1} },
  { 2114, {3, +1} }, { -2114, {3, -1} }, { 1114, {3, +1} }, { -1114, {3, -1} },
  // Lambda, Sigma 1/2^+; Sigma* 3/2^+
  {  3122, {1, +1} }, { -3122, {1, -1} },
  {  3222, {1, +1} }, { -3222, {1, -1} }, {  3212, {1, +1} }, { -3212, {1, -1} }, {  3112, {1, +1} }, { -3112, {1, -1} },
  {  3224, {3, +1} }, { -3224, {3, -1} }, {  3214, {3, +1} }, { -3214, {3, -1} }, {  3114, {3, +1} }, { -3114, {3, -1} },
  // Xi 1/2^+, Xi* 3/2^+, Omega 3/2^+
  {  3322, {1, +1} }, { -3322, {1, -1} }, {  3312, {1, +1} }, { -3312, {1, -1} },
  {  3324, {3, +1} }, { -3324, {3, -1} }, {  3314, {3, +1} }, { -3314, {3, -1} },
  {  3334, {3, +1} }, { -3334, {3, -1} },
  // ground charmed baryons
  { 4122, {1, +1} }, { -4122, {1, -1} }, { 4112, {1, +1} }, { -4112, {1, -1} },
  { 4212, {1, +1} }, { -4212, {1, -1} }, { 4222, {1, +1} }, { -4222, {1, -1} },
  { 4114, {3, +1} }, { -4114, {3, -1} }, { 4214, {3, +1} }, { -4214, {3, -1} },
  { 4224, {3, +1} }, { -4224, {3, -1} }, { 4132, {1, +1} }, { -4132, {1, -1} },
  { 4232, {1, +1} }, { -4232, {1, -1} }, { 4312, {1, +1} }, { -4312, {1, -1} },
  { 4322, {1, +1} }, { -4322, {1, -1} }, { 4314, {3, +1} }, { -4314, {3, -1} },
  { 4324, {3, +1} }, { -4324, {3, -1} }, { 4332, {1, +1} }, { -4332, {1, -1} },
  { 4334, {3, +1} }, { -4334, {3, -1} }
};

// Isospin map: (I2, I3_2).
static bool getIsospin(int id, int& I2, int& I3_2) {
  static const std::unordered_map<int, std::pair<int, int>> iso = {
    // mesons: pi, rho, eta/eta'/omega/phi, K, K0/K0bar, K*, K*0, D, D0, D*, D*0, Ds, Ds*, charmonia
    {  211, {2, +2} }, { 111, {2, 0} }, { -211, {2, -2} }, {  213, {2, +2} }, { 113, {2, 0} }, { -213, {2, -2} },
    {  221, {0, 0} }, { 331, {0, 0} }, { 223, {0, 0} }, { 333, {0, 0} },
    {  321, {1, +1} }, { -321, {1, -1} }, {  311, {1, -1} }, { -311, {1, +1} },
    {  323, {1, +1} }, { -323, {1, -1} }, {  313, {1, -1} }, { -313, {1, +1} },
    {  411, {1, +1} }, { -411, {1, -1} }, {  421, {1, -1} }, { -421, {1, +1} },
    {  413, {1, +1} }, { -413, {1, -1} }, {  423, {1, -1} }, { -423, {1, +1} },
    {  431, {0, 0} }, { -431, {0, 0} }, {  433, {0, 0} }, { -433, {0, 0} },
    { 441, {0, 0} }, { 100441, {0, 0} }, { 443, {0, 0} }, { 100443, {0, 0} },
    { 10441, {0, 0} }, { 10443, {0, 0} }, { 20443, {0, 0} }, { 445, {0, 0} },
    // baryons: N, Delta, Lambda, Sigma, Sigma*, Xi, Xi*, Omega, and the charmed ones
    { 2212, {1, +1} }, { 2112, {1, -1} }, { -2212, {1, -1} }, { -2112, {1, +1} },
    { 2224, {3, +3} }, { 2214, {3, +1} }, { 2114, {3, -1} }, { 1114, {3, -3} },
    { -2224, {3, -3} }, { -2214, {3, -1} }, { -2114, {3, +1} }, { -1114, {3, +3} },
    {  3122, {0, 0} }, { -3122, {0, 0} },
    {  3222, {2, +2} }, {  3212, {2, 0} }, {  3112, {2, -2} }, { -3222, {2, -2} }, { -3212, {2, 0} }, { -3112, {2, +2} },
    {  3224, {2, +2} }, {  3214, {2, 0} }, {  3114, {2, -2} }, { -3224, {2, -2} }, { -3214, {2, 0} }, { -3114, {2, +2} },
    {  3322, {1, +1} }, {  3312, {1, -1} }, { -3322, {1, -1} }, { -3312, {1, +1} },
    {  3324, {1, +1} }, {  3314, {1, -1} }, { -3324, {1, -1} }, { -3314, {1, +1} },
    {  3334, {0, 0} }, { -3334, {0, 0} },
    { 4122, {0, 0} }, { -4122, {0, 0} },
    { 4222, {2, +2} }, { 4212, {2, 0} }, { 4112, {2, -2} }, { -4222, {2, -2} }, { -4212, {2, 0} }, { -4112, {2, +2} },
    { 4224, {2, +2} }, { 4214, {2, 0} }, { 4114, {2, -2} }, { -4224, {2, -2} }, { -4214, {2, 0} }, { -4114, {2, +2} },
    { 4232, {1, +1} }, { 4132, {1, -1} }, { -4232, {1, -1} }, { -4132, {1, +1} },
    { 4322, {1, +1} }, { 4312, {1, -1} }, { -4322, {1, -1} }, { -4312, {1, +1} },
    { 4324, {1, +1} }, { 4314, {1, -1} }, { -4324, {1, -1} }, { -4314, {1, +1} },
    { 4332, {0, 0} }, { -4332, {0, 0} }, { 4334, {0, 0} }, { -4334, {0, 0} }
  };

  auto it = iso.find(id);
  if (it == iso.end()) return false;
  I2   = it->second.first;
  I3_2 = it->second.second;
  return true;
}

static inline int parityFromL(int L) {
  return (L % 2 == 0) ? +1 : -1;
}

static bool isSelfConjugateNeutral(const ParticleData& pd, int id) {
  if (pd.antiId(id) != id) return false;
  return pd.chargeType(id) == 0;
}

static bool intrinsicC(int id, int& C) {
  auto it = C_TABLE.find(id);
  if (it == C_TABLE.end()) return false;
  C = it->second;
  return true;
}

DiscreteSymmetryFilter::DiscreteSymmetryFilter(const ParticleData& pd, SourceQN cfg)
: pd_(&pd), cfg_(std::move(cfg)) {}

bool DiscreteSymmetryFilter::veto(const Event& event, Rndm& rndm) const {
  return vetoImpl(event, &rndm, false);
}

bool DiscreteSymmetryFilter::supports(const Event& event) const {
  return !vetoImpl(event, nullptr, true);
}

double DiscreteSymmetryFilter::acceptance(const std::vector<int>& hadIds) const {
  bool stochastic = false;
  return static_cast<double>(projection(hadIds, false, stochastic));
}

bool DiscreteSymmetryFilter::vetoImpl(
    const Event& event, Rndm* rndm, bool supportOnly) const {
  std::vector<int> hadIds;
  hadIds.reserve(64);
  for (int i = 0; i < event.size(); ++i)
    if (event[i].isFinal() && event[i].isHadron()) hadIds.push_back(event[i].id());
  bool stochastic = false;
  const long double pAcc = projection(hadIds, supportOnly, stochastic);
  if (pAcc <= 0.0L) return true;
  // Exactly one flat() is drawn for a stochastic isospin projection.
  if (stochastic && !supportOnly) {
    if (rndm == nullptr)
      throw std::runtime_error("stochastic source veto lacks an RNG");
    if ((long double) rndm->flat() > pAcc) return true;
  }
  return false;
}

// Acceptance probability of the hadron multiset (in record order): 0 for a
// deterministic veto, 1 without a random draw, else the allowed-isospin weight.
long double DiscreteSymmetryFilter::projection(
    const std::vector<int>& hadIds, bool supportOnly, bool& stochastic) const {
  std::unordered_map<int, int> n;
  n.reserve(128);
  bool hasBaryon = false;
  for (int id : hadIds) {
    ++n[id];
    if (isBaryonPdgId(id)) hasBaryon = true;
  }

  // (1) Deterministic C-eigenstate veto (applies only when C is well-defined from IDs).
  // Multi-hadron: only if ALL hadrons are self-conjugate neutral and intrinsic C is known.
  if (cfg_.enforceCDet) {
    bool cApplicable = true;
    bool cDet = true;
    int totalC = 1;

    for (const auto& kv : n) {
      const int id  = kv.first;
      const int cnt = kv.second;

      if (!isSelfConjugateNeutral(*pd_, id)) {
        cApplicable = false;
        break;
      }

      int Ci = 0;
      if (!intrinsicC(id, Ci)) {
        cDet = false;
        break;
      }

      if (cnt % 2 != 0) totalC *= Ci;
    }

    if (cApplicable && cDet) {
      if (totalC != cfg_.expectedC) return 0.0L;
    }
  }

  // (2) G-proxy: explicitly NOT applicable if any baryon is present. G is
  // multiplicative on any state of integer-isospin mesons with defined G.
  if (cfg_.enforceGProxy) {
    bool gApplicable = !hasBaryon;

    bool gDet = true;
    int totalI3_2 = 0;
    bool i3Det = true;

    if (gApplicable) {
      for (const auto& kv : n) {
        const int id  = kv.first;
        const int cnt = kv.second;

        int I2 = 0, I3_2 = 0;
        if (!getIsospin(id, I2, I3_2)) {
          i3Det = false;
        } else {
          // Half-integer isospin => G not defined for the full state
          if (I2 % 2 != 0) {
            gApplicable = false;
            break;
          }
          totalI3_2 += cnt * I3_2;
        }
      }

      // G is multiplicative within an isomultiplet for any I3, so the
      // proxy applies whenever the state carries the SOURCE I3 (0 for
      // the neutral models, -1 for the tau^- convention CC current);
      // other I3 values are vetoed by the isospin filter anyway.
      if (gApplicable && i3Det && totalI3_2 != cfg_.expectedI3_2)
        gApplicable = false;
    }

    if (gApplicable) {
      int totalG = 1;
      for (const auto& kv : n) {
        const int id  = kv.first;
        const int cnt = kv.second;
        auto it = G_TABLE.find(id);
        if (it == G_TABLE.end()) {
          gDet = false;
          break;
        }
        if (cnt % 2 != 0) totalG *= it->second;
      }

      if (gDet && totalG != cfg_.expectedG) return 0.0L;  // undetermined G is kept
    }
  }

  // (3) Generic 2-body JPC consistency check (works for vector and for J=0 sources).
  // Enforces existence of some (L,S) giving the source J and expected parity.
  // Enforces C for 2-body ONLY in cases where C is determinable:
  //   - particle–antiparticle pair: C = (-1)^(L+S)
  //   - both self-conjugate neutral with known intrinsic C: C = C1*C2
  if (cfg_.enforceJPC2Body && hadIds.size() == 2) {
    const int id1 = hadIds[0];
    const int id2 = hadIds[1];

    auto it1 = JP_TABLE.find(id1);
    auto it2 = JP_TABLE.find(id2);
    if (it1 == JP_TABLE.end() || it2 == JP_TABLE.end()) {
      if (cfg_.undetVeto) return 0.0L;
    } else {
      const int J1_2 = it1->second.first;
      const int P1   = it1->second.second;
      const int J2_2 = it2->second.first;
      const int P2   = it2->second.second;

      const int Smin_2 = std::abs(J1_2 - J2_2);
      const int Smax_2 = J1_2 + J2_2;

      const int Lmax = 6;

      bool possible = false;

      for (int L = 0; L <= Lmax && !possible; ++L) {
        const int Ptot = P1 * P2 * parityFromL(L);
        if (Ptot != cfg_.expectedP) continue;

        const int L_2 = 2 * L;

        for (int S_2 = Smin_2; S_2 <= Smax_2; S_2 += 2) {
          if (S_2 % 2 != 0) continue; // require integer S
          const int S = S_2 / 2;

          const int Jmin_2 = std::abs(L_2 - S_2);
          const int Jmax_2 = L_2 + S_2;

          bool Jok = false;
          for (int J_2 = Jmin_2; J_2 <= Jmax_2; J_2 += 2) {
            if (J_2 == cfg_.sourceJ2) {
              Jok = true;
              break;
            }
          }
          if (!Jok) continue;

          // Deterministic C enforcement for 2-body when applicable.
          if (cfg_.enforceCDet) {
            bool cApplicable = false;
            bool cDet = true;
            int Cpair = 0;

            // A) particle–antiparticle pair (not identical)
            if (pd_->antiId(id1) == id2 && pd_->antiId(id2) == id1 && id1 != id2) {
              cApplicable = true;
              Cpair = ((L + S) % 2 == 0) ? +1 : -1; // (-1)^(L+S)
            }
            // B) both self-conjugate neutral
            else if (isSelfConjugateNeutral(*pd_, id1) && isSelfConjugateNeutral(*pd_, id2)) {
              cApplicable = true;
              int C1 = 0, C2 = 0;
              if (!intrinsicC(id1, C1) || !intrinsicC(id2, C2)) cDet = false;
              else Cpair = C1 * C2;
            }

            if (cApplicable && cDet) {
              if (Cpair != cfg_.expectedC) continue;
            }
          }

          possible = true;
          break;
        }
      }

      if (!possible) return 0.0L;
    }
  }

  // (4) Isospin: exact I3 + probabilistic projection
  bool det = true;
  std::vector<std::pair<int, int>> isoList;
  isoList.reserve(128);

  int totalI3_2 = 0;
  for (const auto& kv : n) {
    const int id  = kv.first;
    const int cnt = kv.second;

    int I2 = 0, I3_2 = 0;
    if (!getIsospin(id, I2, I3_2)) {
      det = false;
      continue;
    }

    totalI3_2 += cnt * I3_2;
    for (int k = 0; k < cnt; ++k) isoList.emplace_back(I2, I3_2);
  }

  if (!det) {
    if (cfg_.undetVeto) return 0.0L;
  } else {
    if (totalI3_2 != cfg_.expectedI3_2) return 0.0L;

    // The stochastic filter and the support-only selection both use the
    // canonical diagonal trace (sorted input, normalized, independent of
    // record and hash order).  Intermediate-coupling paths are orthogonal,
    // so their probabilities add; amplitudes of distinct paths never
    // interfere.
    const auto weights = diagonalIsospinProbabilities2(isoList);
    long double tot = 0.0L, allow = 0.0L;
    for (const auto& kv : weights) tot += kv.second;
    for (int want : cfg_.allowedI2) {
      auto it = weights.find(want);
      if (it != weights.end()) allow += it->second;
    }

    const long double supportFloor = supportOnly ? 1.0e-15L : 0.0L;
    if (tot <= 0.0L || allow <= supportFloor) return 0.0L;

    // P(accept) = summed probability of the allowed isospins.
    stochastic = true;
    return std::min(allow, 1.0L);
  }

  return 1.0L;
}
