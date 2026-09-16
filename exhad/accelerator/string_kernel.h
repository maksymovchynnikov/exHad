#pragma once

// Stock Pythia 8.317 string fragmentation with a flavour script per attempt and a
// decision at the final join, through the public FragmentationModel, StringFlav and
// UserHooks interfaces. Scripts: STOCK; SINGLETS (with
// probability beta every break is an isosinglet meson with its stock relative law);
// ORDERED (a forced three-meson eta(')-pi-pi history); MENU (a forced light pion/rho
// history drawn with its stock flavour weight times the filter acceptance). See README.md.
#include "finite_meson_choice.h"
#include "Pythia8/Pythia.h"
#include "Pythia8/MiniStringFragmentation.h"
#include "Pythia8/StringFragmentation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <set>
#include <utility>

namespace string_kernel {
using namespace Pythia8;
inline bool singlet(int id) {
  return id == 221 || id == 331 || id == 223 || id == 333;
}

constexpr int JOIN_TRIES = 10;  // StringFragmentation::finalTwo NTRYFLAV

class Flav final : public exhad::accelerator::FiniteMesonChoice {
 public:
  enum class Script { STOCK, SINGLETS, ORDERED, MENU } script = Script::STOCK;
  bool forced = false, invalid = false;
  int q0 = 0;
  // Stock singlet laws of one completed break (pick/combine do-while) and of the join.
  struct Law {
    std::vector<std::pair<int, double>> cumulative;
    long double total = 0;
  };
  std::array<Law, 4> step, join;
  std::array<long double, 4> retry{};  // sum_{k<JOIN_TRIES} (diagonal combine failure)^k

  void init() override {
    FiniteMesonChoice::init();
    for (int q = 1; q <= 3; ++q) {
      const auto meson = oneCall(FlavContainer(q), FlavContainer(-q));
      const long double z = 1 - failure(q), diagonal = (q < 3 ? 1 : probQandS - 2) / (probQandQQ * probQandS);
      long double power = 1;
      retry[q] = 0;
      for (int k = 0; k < JOIN_TRIES; ++k, power *= meson.probability(0)) retry[q] += power;
      step[q] = join[q] = Law{};
      for (int id : {221, 331, 223, 333}) {
        step[q].cumulative.push_back({id, double(step[q].total += diagonal * meson.probability(id) / z)});
        join[q].cumulative.push_back({id, double(join[q].total += retry[q] * meson.probability(id))});
      }
      for (int id : {0, 111, 113, 221, 331, 223, 333})
        if (q == 2 && oneCall(FlavContainer(1), FlavContainer(-1)).probability(id) != meson.probability(id))
          throw std::runtime_error("unequal u/d neutral-meson laws");
    }
    // Charged light mesons: the four end/new-flavour cases of one sign convention.
    for (int id : {0, 211, 213}) {
      const long double law = oneCall(FlavContainer(2), FlavContainer(-1)).probability(id);
      if (oneCall(FlavContainer(-1), FlavContainer(2)).probability(id) != law
          || oneCall(FlavContainer(1), FlavContainer(-2)).probability(-id) != law
          || oneCall(FlavContainer(-2), FlavContainer(1)).probability(-id) != law)
        throw std::runtime_error("unequal charged light-meson laws");
    }
    if (failure(1) != failure(2) || oneCall(FlavContainer(3), FlavContainer(-3)).probability(111) != 0.)
      throw std::runtime_error("uncertified light-flavour symmetry");
  }
  // Stock probability of an all-singlet n-hadron history with end flavour q0.
  long double chance(int n, int q) const {
    return std::pow(step.at(std::abs(q)).total, n - 1) * join.at(std::abs(q)).total;
  }

  // MENU. A kind is 111, 113 or the positive id of a charged pion or rho (211, 213),
  // whose sign follows the string end. Along a light history the laws below do not
  // depend on the u/d end or on the side (certified in init), so one ordered history
  // of entry e has stock flavour probability joinLaw(join) prod breakLaw(breaks).
  struct Entry {
    std::vector<int> breaks;
    int join = 0;
    long double acceptance = 0, weight = 0;
  };
  std::vector<Entry> menu;
  // One completed break from a light end, and the ten-try join of two light ends.
  long double breakLaw(int kind) const {
    const bool charged = kind == 211 || kind == 213;
    return oneCall(FlavContainer(2), FlavContainer(charged ? -1 : -2)).probability(kind)
        / (probQandQQ * probQandS * (1 - failure(2)));
  }
  long double joinLaw(int kind) const {
    const auto law = oneCall(FlavContainer(2), FlavContainer(kind == 211 || kind == 213 ? -1 : -2));
    long double tries = 0, power = 1;
    for (int k = 0; k < JOIN_TRIES; ++k, power *= law.probability(0)) tries += power;
    return tries * law.probability(kind);
  }
  // Entry weights: stock flavour probability of one order, times the number of distinct
  // break orders (drawn uniformly), times the filter acceptance. Returns their sum W.
  long double configureMenu(std::vector<Entry> entries) {
    long double total = 0;
    for (Entry& entry : entries) {
      std::sort(entry.breaks.begin(), entry.breaks.end());
      long double orders = 1, weight = joinLaw(entry.join);
      for (unsigned i = 0; i < entry.breaks.size(); ++i) {
        orders *= i + 1;
        weight *= breakLaw(entry.breaks[i]);
        if (i && entry.breaks[i] == entry.breaks[i - 1]) {
          unsigned run = 1;
          while (run <= i && entry.breaks[i - run] == entry.breaks[i]) ++run;
          orders /= run;
        }
      }
      if (!(weight > 0 && entry.acceptance > 0)) throw std::runtime_error("empty menu entry");
      total += entry.weight = weight * orders * entry.acceptance;
    }
    long double running = 0;
    for (Entry& entry : entries) entry.weight = (running += entry.weight) / total;
    entries.back().weight = 1;
    menu = std::move(entries);
    return total;
  }

  void configure(std::vector<int> target) {  // ORDERED: a sorted three-meson primary multiset
    orders.clear();
    weights.clear();
    long double sum = 0;
    do {
      orders.push_back({target[0], target[1], target[2]});
         weights.push_back(double(sum += std::abs(target[2]) == 211 ? 1 : retry[2]));
    } while (std::next_permutation(target.begin(), target.end()));
    for (auto& w : weights) w = double(w / sum);
    weights.back() = 1.;
  }
  void begin(const StringEnd* p, const StringEnd* n, bool force) {  // once per attempt
    pos = p;
    neg = n;
    q0 = p->flavOld.id;
    pending = 0;
    index = 0;
    invalid = false;
    forced = force;
    charged = false;
    if (script == Script::ORDERED) {
      const double u = rndmPtr->flat();
      unsigned k = 0;
      while (k + 1 < weights.size() && u >= weights[k]) ++k;
      order = orders[k];
    }
    if (script == Script::MENU) {
      const double u = rndmPtr->flat();
      unsigned k = 0;
      while (k + 1 < menu.size() && u >= menu[k].weight) ++k;
      entry = &menu[k];
      breaks = entry->breaks;
      for (unsigned i = unsigned(breaks.size()); i > 1; --i)  // uniform over distinct orders
        std::swap(breaks[i - 1], breaks[std::min(i - 1, unsigned(rndmPtr->flat() * i))]);
    }
  }
  FlavContainer pick(FlavContainer& old, double pT, double kappa, bool allowPop) override {
    const bool end = pos && (&old == &pos->flavOld || &old == &neg->flavOld);
    if (script == Script::MENU) {
      if (!end) return StringFlav::pick(old, pT, kappa, allowPop);  // the closed-loop start stays stock
      const bool light = (std::abs(old.id) == 1 || std::abs(old.id) == 2) && !old.nPop && !old.idPop;
      if (!invalid && light && index < breaks.size()) {
        const int kind = breaks[index++];
        if (kind != 211 && kind != 213) {
          pending = kind;
          return FlavContainer(-old.id, old.rank + 1);
        }
        // u and dbar ends give the positive meson; the other light end is the isospin partner.
        const bool up = old.id == 2 || old.id == -1;
        pending = up ? kind : -kind;
        return FlavContainer(old.id > 0 ? -(3 - old.id) : 3 + old.id, old.rank + 1);
      }
      invalid = true;
      pending = 111;  // no completion: the break takes a b flavour, so the energy is used up
      return FlavContainer(old.id > 0 ? -5 : 5, old.rank + 1);
    }
    if (script == Script::ORDERED) {
      if (!end) return FlavContainer(rndmPtr->flat() < 0.5 ? 1 : 2, old.rank + 1);  // light start, u:d = 1:1
      // At the first charged break only one pion charge fits the end; both charge orders weigh the same.
      if (!invalid && index < 2 && std::abs(order[index]) == 211 && !std::exchange(charged, true)
          && !successor(old.id, order[index])) for (int& id : order) if (std::abs(id) == 211) id = -id;
      const int next = !invalid && index < 2 ? successor(old.id, order[index]) : 0;
      if (next) {
        pending = order[index++];
        return FlavContainer(next, old.rank + 1);
      }
      invalid = true;
      pending = 111;  // no completion: the break takes a b flavour, so the energy is used up
      return FlavContainer(old.id > 0 ? -5 : 5, old.rank + 1);
    }
    if (!forced || !end || invalid) return StringFlav::pick(old, pT, kappa, allowPop);
    const int q = std::abs(old.id);
    if (q > 3 || old.nPop || old.idPop) {
      invalid = true;
      return StringFlav::pick(old, pT, kappa, allowPop);
    }
    pending = draw(step[q]);
    return FlavContainer(-old.id, old.rank + 1);
  }
  int combine(FlavContainer& first, FlavContainer& second) override {
    if (pending) {
      const int id = pending;
      pending = 0;
      return id;
    }
    if (script == Script::ORDERED)  // the final join
      return !invalid && index == 2 && successor(first.id, order[2]) == second.id ? order[2] : 0;
    if (script == Script::MENU) {  // the final join of a light quark and antiquark
      if (invalid || index != breaks.size()) return 0;
      const int quark = std::max(first.id, second.id), anti = std::min(first.id, second.id);
      const int kind = entry->join;
      if (kind != 211 && kind != 213) return quark == -anti && quark <= 2 && quark >= 1 ? kind : 0;
      return quark == 2 && anti == -1 ? kind : quark == 1 && anti == -2 ? -kind : 0;
    }
    if (!forced || invalid) return StringFlav::combine(first, second);
    const int q = std::abs(first.id);
    if (q < 1 || q > 3 || first.id != -second.id || first.nPop || second.nPop) {
      invalid = true;
      return StringFlav::combine(first, second);
    }
    return draw(join[q]);
  }

 private:
  const StringEnd* pos = nullptr, *neg = nullptr;
  std::vector<std::array<int, 3>> orders;
  std::vector<double> weights;
  std::array<int, 3> order{};
  const Entry* entry = nullptr;
  std::vector<int> breaks;
  unsigned index = 0;
  int pending = 0;
  bool charged = false;

  static int successor(int old, int hadron) {  // light end flavour after a pseudoscalar break
    if (std::abs(old) > 2 || !old) return 0;
    if (hadron == 111 || hadron == 221 || hadron == 331) return -old;
    if (hadron == 211) return old == 2 ? -1 : old == -1 ? 2 : 0;
    if (hadron == -211) return old == 1 ? -2 : old == -2 ? 1 : 0;
    return 0;
  }
  int draw(const Law& law) const {
    const double u = rndmPtr->flat() * double(law.total);
    for (const auto& item : law.cumulative) if (u < item.second) return item.first;
    return law.cumulative.back().first;
  }

 public:
  // Probability that one stock pick+combine iteration from quark end q returns no hadron:
  // eta/eta' suppression of a diagonal meson, or SU(6) rejection of a new baryon.
  long double failure(int q) const {
    const long double meson = 1 / probQandQQ;
    long double fail = 0;
    for (int f = 1; f <= 3; ++f)
      fail += meson * (f < 3 ? 1 : probQandS - 2) / probQandS
          * oneCall(FlavContainer(q), FlavContainer(-f)).probability(0);
    for (int nPop = 0; nPop < 2; ++nPop) {
      const double* w = dWT[nPop];
      const long double sPop = w[0] * (nPop ? scbBM[0] * popcornSpair : 1.);
      for (int idPop = 1; idPop <= 3; ++idPop) for (int drawn = 1; drawn <= 3; ++drawn)
        for (int keep = 0; keep < 2; ++keep) for (int spin : {1, 3}) {
          const bool light = idPop < 3 && drawn < 3;
          if (!light && keep) continue;
          const int vertex = light ? (keep ? idPop : 3 - idPop) : drawn;
          if (vertex == idPop && spin == 1) continue;
          const long double sVtx = w[idPop < 3 ? 1 : 2], kept = std::clamp(w[3], 0., 1.);
          const long double sSpin = idPop >= 3 ? w[4] : vertex == 3 ? w[5] : w[6];
          long double p = (1 - meson) * (nPop ? popFrac : 1.) / (1 + popFrac)
              * (idPop < 3 ? 1 : sPop) / (2 + sPop) * (drawn < 3 ? 1 : sVtx) / (2 + sVtx)
              * (light ? (keep ? kept : 1 - kept) : 1)
              * (vertex == idPop ? 1 : spin == 1 ? 1 / (1 + sSpin) : sSpin / (1 + sSpin));
          int spinFlav = spin - 1;
          if (spinFlav == 2 && vertex != idPop) spinFlav = 4;
          if (q != vertex && q != idPop) ++spinFlav;
          fail += p * (spinFlav < 0 || spinFlav > 5 ? 1.L : baryonCGMax[spinFlav] > 0
              ? 1 - std::clamp(baryonCGSum[spinFlav] / baryonCGMax[spinFlav], 0., 1.)
              : baryonCGSum[spinFlav] < 0 ? 1.L : 0.L);
        }
    }
    return fail;
  }
};

// Zero-charge sign assignments of a light kind multiset (charged kinds as 211/213), sorted.
inline std::set<std::vector<int>> chargeStates(const std::vector<int>& kinds) {
  std::set<std::vector<int>> states;
  std::vector<unsigned> charged;
  for (unsigned i = 0; i < kinds.size(); ++i) if (kinds[i] == 211 || kinds[i] == 213) charged.push_back(i);
  for (unsigned mask = 0; mask < 1u << charged.size(); ++mask) {
    int positive = 0;
    for (unsigned k = 0; k < charged.size(); ++k) positive += mask >> k & 1;
    if (2 * positive != int(charged.size())) continue;
    std::vector<int> state = kinds;
    for (unsigned k = 0; k < charged.size(); ++k) if (mask >> k & 1) state[charged[k]] = -state[charged[k]];
    std::sort(state.begin(), state.end());
    states.insert(state);
  }
  return states;
}

// The four-pion menu of a closed gluon string: every multiset of at least three pions and
// rhos with four pions after rho -> pi pi and an allowed charge state, with each join kind.
// The build audit proves these are all primaries that reach exactly four canonical pions.
inline std::vector<Flav::Entry> fourPionMenu(const std::function<double(const std::vector<int>&)>& acceptance) {
  std::vector<Flav::Entry> entries;
  const std::array<int, 4> kinds{111, 113, 211, 213};
  std::vector<int> multiset;
  std::function<void(unsigned, int)> grow = [&](unsigned first, int pions) {
    if (pions == 4) {
      const auto states = chargeStates(multiset);
      if (multiset.size() < 3 || states.empty()) return;
      const double a = acceptance(*states.begin());
      for (const auto& state : states)
        if (acceptance(state) != a) throw std::runtime_error("charge-asymmetric menu acceptance");
      if (!(a > 0.)) return;
      for (int join : std::set<int>(multiset.begin(), multiset.end())) {
        Flav::Entry entry{multiset, join, a, 0};
        entry.breaks.erase(std::find(entry.breaks.begin(), entry.breaks.end(), join));
        entries.push_back(entry);
      }
      return;
    }
    for (unsigned i = first; i < kinds.size(); ++i) {
      const int taken = kinds[i] % 10 == 3 ? 2 : 1;
      if (pions + taken > 4) continue;
      multiset.push_back(kinds[i]);
      grow(i, pions + taken);
      multiset.pop_back();
    }
  };
  grow(0, 0);
  return entries;
}

// LundFragmentation's dispatch, with Flav (no ministring retry after a failed string).
// A source whose decision is taken on the completed record (`lowMass`) also accepts
// the stock low-mass dispatch; a join-decision source requires a string.
class Model final : public FragmentationModel {
 public:
  StringFragmentation strings;
  MiniStringFragmentation ministring;
  Flav flav;
  bool lowMass = false;
  bool init(StringFlav*, StringPT* pT, StringZ* z, FragModPtr modifier) override {
    registerSubObject(strings);
    registerSubObject(ministring);
    registerSubObject(flav);
    flav.init();
    strings.init(&flav, pT, z, modifier);
    ministring.init(&flav, pT, z, modifier);
    mStringMin = parm("HadronLevel:mStringMin");
    return true;
  }
  bool fragment(int iSub, ColConfig& config, Event& event, bool isDiff, bool) override {
    if (iSub == -1) return true;
    if (!(config[iSub].massExcess > mStringMin)) {
      if (!lowMass) throw std::runtime_error("mini-string outside the string kernel");
      return ministring.fragment(iSub, config, event, isDiff);
    }
    return strings.fragment(iSub, config, event);
  }
 private:
  double mStringMin = 0.;
};

// Marks attempts, records their primaries and asks `join` at the final join (true: veto).
class Hooks final : public UserHooks {
 public:
  Flav* flav = nullptr;
  double beta = 0.;  // SINGLETS forcing probability
  std::vector<int> primaries;
  std::function<bool(const std::vector<int>&)> join;
  unsigned long long attempts = 0, forced = 0;
  bool canVetoFragmentation() override {
    return true;
  }
  void setStringEnds(const StringEnd* pos, const StringEnd* neg, std::vector<int>) override {
    ++attempts;
    primaries.clear();
    const bool force = flav->script == Flav::Script::SINGLETS && beta > 0. && rndmPtr->flat() < beta;
    forced += force;
    flav->begin(pos, neg, force);
  }
  bool doVetoFragmentation(Particle hadron, const StringEnd*) override {
    primaries.push_back(hadron.id());
    return false;
  }
  bool doVetoFragmentation(Particle first, Particle second, const StringEnd*, const StringEnd*) override {
    primaries.push_back(first.id());
    primaries.push_back(second.id());
    return join && join(primaries);
  }
};

// Settings under which the attempts are identically distributed and the scripts exact.
// `closed` adds the two-gluon condition (an open two-parton string never joins extra
// partons); `ordered` adds the condition that ends a forced ORDERED script.
inline void certify(Pythia& p, double mass, bool closed = true, bool ordered = true) {
  for (const char* flag : {"StringFlav:suppressLeadingB", "ClosePacking:doClosePacking",
       "Ropewalk:RopeHadronization", "Ropewalk:doFlavour", "Fragmentation:setVertices", "HadronLevel:Rescatter",
       "VariationFrag:flav", "VariationFrag:z", "VariationFrag:pT", "MiniStringFragmentation:tryAfterFailedFrag",
       "StringFragmentation:doStrangeJunctions"})
    if (p.settings.flag(flag)) throw std::runtime_error(std::string("uncertified string-kernel setting: ") + flag);
  if (p.settings.mode("Fragmentation:model") != 0 || p.settings.parm("StringFragmentation:stopMass") != 0.)
    throw std::runtime_error("string kernel requires ordinary Lund fragmentation with zero stopping mass");
  if (closed && !(0.5 * mass > 4. * p.settings.parm("FragmentationSystems:mJoin")))
    throw std::runtime_error("closed-string retries can change topology at this mass");
  if (ordered && !(p.settings.parm("StringFragmentation:stopNewFlav") * p.particleData.constituentMass(5)
        * (1. - p.settings.parm("StringFragmentation:stopSmear")) > mass))
    throw std::runtime_error("a b-flavour break does not end the string at this mass");
  if (p.particleData.constituentMass(1) != p.particleData.constituentMass(2))
    throw std::runtime_error("unequal u/d constituent masses");
}

// A string with u, d or s ends cannot use up its energy at the first break
// (stopMass 0, largest quark wMin), so it has at least three hadrons.
inline void certifyThreeHadrons(Pythia& p, double mass) {
  const double wMin = (2. * p.particleData.constituentMass(3)
      + p.settings.parm("StringFragmentation:stopNewFlav") * p.particleData.constituentMass(3))
      * (1. + p.settings.parm("StringFragmentation:stopSmear"));
  if (!(wMin < mass) || p.particleData.constituentMass(3) < p.particleData.constituentMass(2))
    throw std::runtime_error("a two-hadron light string is possible at this mass");
}
}  // namespace string_kernel
