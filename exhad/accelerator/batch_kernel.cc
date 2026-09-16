// Batched active-pool kernel (exhad-batch) for one portal's two sources. Mode 1
// realizes a rate-first family: the alp-fermion eta(')-pi-pi channel with the ORDERED string
// script, the scalar four-pion family with the MENU script; mode 0 draws every other
// family with the join-decision sampler. Pythia chains, symmetry filter, matching,
// ownership cut and decayers are main.cc's. See README.md.
#define main ordinary_exhad_main
#include "../../cpp/src/main.cc"
#undef main
#include "fast_pythia_seed_simd.h"
#include "string_kernel.h"
#include <array>

namespace {
using string_kernel::Flav;
double unit(std::uint64_t word) {
  return double(portable_rejection::splitmix(word))/18446744073709551616.;
}

// A portal's two sources, in seed-stream order: the first keeps the event seed
// word.  The decaying particle is its quantum numbers {2J, P, C, charge, C given}.
struct SourceSpec {
  const char* name;
  DecayingParticle particle;
  DecayMode decay;
  VComponent comp;
};

struct PortalSpec {
  const char* portal;
  bool rateFirst;
  SourceSpec source[2];
};

constexpr PortalSpec PORTALS[] = {
  {"alp-fermion", true, {{"pseudoscalar-gluon", {0, -1, +1, 0, true}, DecayMode::GG, VComponent::GLUE},
               {"pseudoscalar-strange", {0, -1, +1, 0, true}, DecayMode::QQ, VComponent::STRANGE}}},
  {"scalar", true, {{"scalar-gluon", {0, +1, +1, 0, true}, DecayMode::GG, VComponent::GLUE},
                  {"scalar-strange", {0, +1, +1, 0, true}, DecayMode::QQ, VComponent::STRANGE}}},
  {"b-l", false, {{"isoscalar-light-vector", {2, -1, -1, 0, true}, DecayMode::QQ, VComponent::OMEGA},
                {"isoscalar-strange-vector", {2, -1, -1, 0, true}, DecayMode::QQ, VComponent::PHI}}}};

struct Source {
  RunConfig rc{};
  std::unique_ptr<Pythia> had, dec, chn;
  std::shared_ptr<string_kernel::Model> model = std::make_shared<string_kernel::Model>();
  std::shared_ptr<string_kernel::Hooks> hooks = std::make_shared<string_kernel::Hooks>();
  std::unique_ptr<DiscreteSymmetryFilter> symmetry;
  std::map<std::vector<int>, double> acceptances;
  portable_rejection::Filter matching;
  double weight = 0., envelope = 0.;  // source probability; join envelope K
  // Largest weight an all-isosinglet state of at least two hadrons can match.
  double singletReach = std::numeric_limits<double>::infinity();
  bool derived = false;            // second seed stream of the event
  bool light = false;              // u/d string end
  bool chain = false;              // light end kept across trials (flavour chain), else redrawn per trial
  int end = 0;                     // the chain's current light end (0: draw at the next trial)
  bool change = false;             // the chain's next trial takes the other light end
  bool record = false;             // decide on the completed record (no join decision)
  Source(const SourceSpec& spec, bool second, const std::vector<std::string>& settings):derived(second) {
    rc.particle = spec.particle;
    rc.carriesParentSpin = true;  // every accelerated portal decays a boson of its own spin
    rc.decay = spec.decay;
    rc.comp = spec.comp;
    rc.undetVeto = true;
    rc.stopMass = 0.;
    rc.pythiaSets = settings;
    rc.pythiaSets.push_back("HadronLevel:mStringMin=1.0");
    light = spec.comp == VComponent::OMEGA || spec.comp == VComponent::RHO;
    PythiaChain chain = makeChain(rc);
    had = std::move(chain.had);
    dec = std::move(chain.dec);
    chn = std::move(chain.chn);
    hooks->flav = &model->flav;
    had->setUserHooksPtr(hooks);
    had->setFragmentationPtr(model);
    if (!had->init()) throw std::runtime_error("string kernel initialization failed");
    symmetry = std::make_unique<DiscreteSymmetryFilter>(had->particleData, sourceQN(rc, rc.comp));
  }
  bool glue() const {
    return rc.comp == VComponent::GLUE;
  }
  // Mass excess of this source's string; at or below mStringMin Pythia fragments it
  // as a low-mass system instead, which has no final join.
  double excess(double mass) const {
    return glue()?mass:mass-2.*had->particleData.constituentMass(light?1:3);
  }
  double acceptance(std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    const auto found = acceptances.find(ids);
    return found != acceptances.end()?found->second:acceptances[ids] = symmetry->acceptance(ids);
  }
  void seed(std::uint64_t word) {  // once per event; offsets as in seedChain
    const int base = int(word%MAX_WORKER_BASE_SEED)+1;
    exhad_rng_simd::initialize(had->rndm, base);
    exhad_rng_simd::initialize(dec->rndm, base+DECAYER_SEED_OFFSET);
    exhad_rng_simd::initialize(chn->rndm, base+CHANNEL_DECAYER_SEED_OFFSET);
  }
  // One trial. A light end is drawn afresh as in the reference, or kept by the chain.
  bool hadronize(double mass) {
    int q = glue()?21:3;
    if (light) {
      if (!chain || !end) end = had->rndm.flat()<0.5?2:1;
      else if (change) end = 3-end;
      change = false;
      q = end;
    }
    buildPair(*had, mass, q, glue()?21:-q, polarMode(rc));
    return had->forceHadronLevel();
  }
  std::vector<int> primaries() const {  // hadrons of a completed record, as the filter reads them
    std::vector<int> ids;
    for (int i = 0; i<had->event.size(); ++i)
      if (had->event[i].isFinal() && had->event[i].isHadron()) ids.push_back(had->event[i].id());
    return ids;
  }
};

struct Tally {
  std::uint64_t events = 0, attempts = 0, forced = 0, strings = 0, decays = 0, switches = 0, retries = 0;
};

// Mode 0: the reference nested rejection over E'. The uniform u of an attempt at its join
// accepts if u < a v/(K D) and switches source if u < a (v + sigma_o (E' - v))/(K D), else
// continues (a: filter acceptance, v: matched weight, D: SINGLETS density ratio). Decays
// stop once u exceeds the threshold of the largest v still possible.
struct JoinSampler {
  std::vector<Source*> sources;
  double scale = 0., u = 0., vmax = 0., reach = 0., baryon = 0., fresh = 0.;
  bool decided = false, open = false, hyperons = false;
  // The per-attempt decision, from the primaries of the attempt.
  bool veto(Source& s, const std::vector<int>& ids) {
    const Flav& flav = s.model->flav;
    if (flav.forced && flav.invalid) return true;
    const bool singlets = std::all_of(ids.begin(), ids.end(), string_kernel::singlet);
    if (flav.forced && !singlets) throw std::runtime_error("singlet script produced a non-singlet");
    const double density = s.hooks->beta>0.?1.-s.hooks->beta
        +(singlets?s.hooks->beta/double(flav.chance(int(ids.size()), flav.q0)):0.):1.;
    scale = s.acceptance(ids)/(s.envelope*density);
    // A baryon-number state can only match the baryon-reachable families (none for alp-fermion).
    vmax = std::any_of(ids.begin(), ids.end(), [](int id) {return std::abs(id)>1000;})?std::min(reach, baryon)
        :singlets && ids.size() >= 2?std::min(reach, s.singletReach):reach;
    fresh = 0.5*reach/(s.envelope*density);
    const double bound = s.chain?changeBound(s, vmax):threshold(s, vmax);
    if (bound>1.+1e-12) throw std::runtime_error("join envelope exceeded");
    u = s.had->rndm.flat();
    open = u<bound;
    decided = true;
    if (s.chain && open && !(u<threshold(s, vmax)) && u<fresh) {
      open = false;
      s.change = true;
    }
    // A veto retries the same string ends within Pythia's attempt budget: a fresh identically
    // distributed attempt for a fixed end. A continued light attempt outside the chain
    // completes its string, and the next trial redraws the u/d end.
    return !open && !s.record && (!s.light || (s.chain && !s.change));
  }
  bool decide(Source& s, const std::vector<int>& ids) {
    if (s.model->flav.script == Flav::Script::ORDERED || s.model->flav.script == Flav::Script::MENU || s.record) return false;
    return veto(s, ids);
  }
  double threshold(const Source& s, double v) const {
    return scale*(v+(1.-s.weight)*(reach-v));
  }
  // Flavour chain: below this the continuation takes the other light end, which carries half
  // the reference's continue mass E' - a (v + sigma_o (E' - v)); fresh is its v-free part.
  double changeBound(const Source& s, double v) const {
    return fresh+0.5*threshold(s, v);
  }
  Source* draw(double mass, std::uint64_t seed, std::vector<int>& roots, Tally& tally) {
    Source* s = unit(seed^0xD1B54A32D192ED03ULL)<sources[0]->weight?sources[0]:sources[1];
    for (Source* x:sources) {
      x->end = 0;
      x->change = false;
    }
    std::array<bool, 2> seeded{};
    for (int calls = 0; calls<1<<20; ++calls) {
      if (!seeded[s->derived]) {
        s->seed(s->derived?portable_rejection::splitmix(seed^0x5851F42D4C957F2DULL):seed);
        seeded[s->derived] = true;
      }
      ++tally.strings;
      decided = false;
      open = false;
      if (!s->hadronize(mass)) {
        s->change = false;
        continue;
      }
      if (s->record) veto(*s, s->primaries());
      if (!decided) throw std::runtime_error("string completed without a join decision");
      if (!open) continue;
      roots = ancestryRoots(s->had->event, false);
      double v = 0.;
      if (!continueEvent(*s->chn, *s->had)) {
        s->change = s->chain && u<fresh;
        continue;
      }
      // A channel-level record whose ownership cut resolves bounds the decayed weight.
      // Where an unresolved one (a hyperon terminal) can still match a positive weight
      // after the full decay, only the primaries bound it.
      portable_rejection::State cut;
      if (!hyperons || matchedOwnershipCut(s->chn->event, roots, {}, cut))
        vmax = matchedWeight(s->chn->event, roots, s->matching, v)?std::min(vmax, v):0.;
      if (!(u<threshold(*s, vmax)) && (!s->chain || u<fresh || !(u<changeBound(*s, vmax)))) {
        s->change = s->chain && u<fresh;
        continue;
      }
      ++tally.decays;
      if (!continueEvent(*s->dec, *s->chn) || !matchedWeight(s->dec->event, roots, s->matching, v)) {
        s->change = s->chain && u<fresh;
        continue;
      }
      if (spacelikeTerminal(s->dec->event)) v = 0.;
      if (v>vmax*(1.+1e-12)) throw std::runtime_error("matched weight exceeds its channel-level bound");
      if (u<scale*v) return s;
      if (u<threshold(*s, v)) {
        s = s == sources[0]?sources[1]:sources[0];
        s->end = 0;
        ++tally.switches;
      }
      else s->change = s->chain && u<changeBound(*s, v);
    }
    throw std::runtime_error("join sampler exhausted its budget");
  }
};

// Mode 1: gluon strings with the ORDERED script until the primaries are the
// target; lost ownership and spacelike terminals retry, as the reference rejects them.
Source* eta(Source& g, double mass, std::uint64_t seed, const std::vector<int>& target, std::vector<int>& roots, Tally& tally) {
  g.seed(seed);
  for (int calls = 0; calls<1<<20; ++calls) {
    ++tally.strings;
    if (!g.hadronize(mass)) continue;
    roots = ancestryRoots(g.had->event, false);
    std::vector<int> ids;
    for (int root:roots) ids.push_back(g.had->event[root].id());
    std::sort(ids.begin(), ids.end());
    if (ids != target) throw std::runtime_error("primary mismatch");
    if (!continueEvent(*g.chn, *g.had) || !continueEvent(*g.dec, *g.chn)) continue;
    portable_rejection::State cut;
    if (matchedOwnershipCut(g.dec->event, roots, {}, cut) && !spacelikeTerminal(g.dec->event)) return &g;
    ++tally.retries;
  }
  throw std::runtime_error("conditional budget exhausted");
}

// Scalar mode 1: gluon strings with the MENU script until the decayed record is four
// pions; other decays, lost ownership and spacelike terminals retry, as the reference
// gives them no four-pion weight.
Source* fourPion(Source& g, double mass, std::uint64_t seed, const std::set<std::vector<int>>& states, std::vector<int>& roots, Tally& tally) {
  g.seed(seed);
  for (int calls = 0; calls<1<<20; ++calls) {
    ++tally.strings;
    if (!g.hadronize(mass)) continue;
    roots = ancestryRoots(g.had->event, false);
    std::vector<int> ids;
    for (int root:roots) ids.push_back(g.had->event[root].id());
    std::sort(ids.begin(), ids.end());
    if (!states.count(ids)) throw std::runtime_error("primary outside the four-pion menu");
    ++tally.decays;
    if (!continueEvent(*g.chn, *g.had) || !continueEvent(*g.dec, *g.chn)) continue;
    portable_rejection::State cut;
    if (matchedOwnershipCut(g.dec->event, roots, {}, cut) && g.matching.family(cut) == "four-pion"
       && !spacelikeTerminal(g.dec->event)) return &g;
    ++tally.retries;
  }
  throw std::runtime_error("conditional budget exhausted");
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("args: mass settings-file alp-fermion|scalar|b-l");
    const double mass = std::stod(argv[1]);
    std::ifstream settingsFile(argv[2]);
    if (!settingsFile) throw std::runtime_error("missing settings");
    std::vector<std::string> settings;
    std::string line;
    while (std::getline(settingsFile, line)) if (!line.empty()) settings.push_back(line);
    const std::string portal = argv[3];
    const PortalSpec* spec = nullptr;
    for (const auto& item:PORTALS) if (portal == item.portal) spec = &item;
    if (!spec) throw std::runtime_error("unsupported conditional portal "+portal);
    // Six Pythia instances: initialize the two source chains side by side.
    auto pending = std::async(std::launch::async, [&]{
      return std::make_unique<Source>(spec->source[1], true, settings);
    });
    Source first(spec->source[0], false, settings);
    const std::unique_ptr<Source> held = pending.get();
    Source& second = *held;
    std::map<std::string, Source*> available{{spec->source[0].name, &first}, {spec->source[1].name, &second}};
    for (Source* s:{&first, &second}) {
      s->record = !(s->excess(mass)>s->had->settings.parm("HadronLevel:mStringMin")+1e-9*mass);
      s->model->lowMass = s->record;
      string_kernel::certify(*s->had, mass, s->glue(), spec->rateFirst);
    }
    const bool scalar = portal == "scalar";
    // Scalar MENU: every allowed four-pion primary state (at least three hadrons).
    std::set<std::vector<int>> fourPionStates;
    if (scalar) {
      string_kernel::certifyThreeHadrons(*first.had, mass);
      auto entries = string_kernel::fourPionMenu([&first](const std::vector<int>& ids) {return first.acceptance(ids);});
      for (const auto& entry:entries) {
        std::vector<int> kinds = entry.breaks;
        kinds.push_back(entry.join);
        std::sort(kinds.begin(), kinds.end());
        for (const auto& state:string_kernel::chargeStates(kinds)) fourPionStates.insert(state);
      }
      first.model->flav.configureMenu(std::move(entries));
    }
    JoinSampler sampler;
    std::cout<<std::setprecision(17)<<"READY CONDITIONAL-TERMINALS "<<mass<<'\n'<<std::flush;
    std::string verb;
    int count = 0;
    if (!(std::cin>>verb>>count) || verb != "CONFIG" || count != 2) throw std::runtime_error("invalid configuration");
    double total = 0.;
    for (int i = 0; i<count; ++i) {
      std::string name;
      double weight;
      int lines;
      if (!(std::cin>>name>>weight>>lines) || !available.count(name) || !(std::isfinite(weight) && weight >= 0.) || lines<1 || lines>10000)
        throw std::runtime_error("invalid source filter");
      Source* source = available.at(name);
      available.erase(name);
      source->weight = weight;
      total += weight;
      sampler.sources.push_back(source);
      std::getline(std::cin, line);
      source->matching.read(std::cin, lines);
    }
    if (!(std::abs(total-1.) <= 2e-12)) throw std::runtime_error("source weights do not close");
    sampler.reach = first.matching.envelope;
    if (second.matching.envelope != sampler.reach) throw std::runtime_error("uncertified join-sampler configuration");
    if (spec->rateFirst) {
      // SINGLETS: beta/(1-beta) = 2 max_q chance(3,q) >= chance(n,q), so the all-singlet
      // density ratio is at least 2(1-beta) where the filter acceptance can reach 1;
      // every other state with a > 1/2 has a baryon (bound sigma_strange E', 2 sigma_strange <= 1).
      const Flav& flav = first.model->flav;
      const double chance = double(std::max({flav.chance(3, 1), flav.chance(3, 2), flav.chance(3, 3)}));
      first.hooks->beta = 2.*chance/(1.+2.*chance);
      first.envelope = sampler.reach/(2.*(1.-first.hooks->beta));
      second.envelope = sampler.reach;
      if (!(second.weight <= 0.5) || first.matching.classifier != (scalar?"scalar-families":"alp-fermion-exact-families")
         || !first.matching.removedFamilies.count("baryon-pair"))
        throw std::runtime_error("uncertified join-sampler configuration");
    } else {
      // Open q-qbar strings of the isoscalar vector current: every state with a
      // non-singlet hadron has a <= 1/2. Two or more isosinglet mesons never match
      // exact-3pi: each decays to at least two pions or to a non-pion terminal (eta
      // itself is terminal); an isosinglet baryon pair matches at most the remainder.
      // Each source takes the smaller of the stock envelope and its SINGLETS envelope: for
      // the strange end D >= 2(1-beta) (n >= 2 hadrons); for the light source the flavour
      // chain, since its u and d attempt streams have different normalizations, which a
      // plain veto or proposal mixture would weight differently from the reference.
      if (first.matching.classifier != "b-l-resolved-families"
         || first.matching.remainder != "remainder"
         || !first.matching.removedFamilies.count("exact-nucleon-pair"))
        throw std::runtime_error("uncertified join-sampler configuration");
      sampler.baryon = first.matching.weights.at(first.matching.remainder);
      sampler.hyperons = true;
      const bool baryons = mass >= 2.*first.had->particleData.m0(3122);
      for (Source* s:{&first, &second}) {
        s->singletReach = 0.;
        for (const auto& [family, weight]:s->matching.weights)
          if (family != "exact-3pi") s->singletReach = std::max(s->singletReach, weight);
        if (sampler.baryon>s->singletReach) throw std::runtime_error("uncertified join-sampler configuration");
        const double other = 1.-s->weight;
        if (s->record) {
          s->hooks->beta = 0.;
          s->envelope = sampler.reach;
          continue;
        }
        // Stock attempts bound every all-isosinglet state by the singlet reach.
        s->envelope = std::max(0.5*sampler.reach, s->singletReach+other*(sampler.reach-s->singletReach));
        const Flav& flav = s->model->flav;
        const double baryonBound = baryons?sampler.baryon+other*(sampler.reach-sampler.baryon):0.;
        double beta = 0., forced = 0.;
        if (s->light) {
          // Flavour chain: the change band needs [E' + a (v + sigma_o (E' - v))]/(2 K D) <= 1, so
          // K = 0.75 E'/(1 - beta) with beta/(1 - beta) = chance(2)/3 covers the singlet mesons.
          const double chance = double(std::max(flav.chance(2, 1), flav.chance(2, 2)));
          beta = chance/(3.+chance);
          forced = std::max(0.75*sampler.reach, 0.5*(sampler.reach+baryonBound))/(1.-beta);
        } else {
          const double chance = double(flav.chance(2, 3));
          beta = chance/(1.+chance);
          forced = std::max(0.5*sampler.reach, baryonBound)/(1.-beta);
        }
        if (!(forced<s->envelope)) continue;
        s->chain = s->light;
        s->hooks->beta = beta;
        s->envelope = forced;
        s->model->flav.script = Flav::Script::SINGLETS;
      }
    }
    for (Source* s:{&first, &second}) s->hooks->join = [&sampler, s](const std::vector<int>& ids) {
      return sampler.decide(*s, ids);
    };
    std::cout<<"CONFIGURED\n"<<std::flush;
    std::map<std::string, Tally> tallies;
    while (std::cin>>verb) {
      if (verb == "QUIT") break;
      int events = 0;
      if (verb != "DRAW" || !(std::cin>>events) || events<1 || events>1000000)
        throw std::runtime_error("invalid draw batch");
      // Read the complete request before writing to avoid a pipe deadlock.
      struct Request{
        std::uint64_t seed;
        int mode;
        std::vector<int> primary;
      };
      std::vector<Request> requests;
      requests.reserve(events);
      for (int i = 0; i<events; ++i) {
        Request r{};
        if (!(std::cin>>r.seed>>r.mode) || r.mode<0 || r.mode>(spec->rateFirst?1:0))
          throw std::runtime_error("invalid conditional event request");
        if (r.mode && !scalar) for (int k = 0; k<3; ++k) {
          int id;
          if (!(std::cin>>id)) throw std::runtime_error("truncated primary");
          r.primary.push_back(id);
        }
        std::sort(r.primary.begin(), r.primary.end());
        requests.push_back(r);
      }
      for (int i = 0; i<events; ++i) {
        const auto& request = requests[i];
        std::vector<int> roots;
        Source* selected = nullptr;
        Tally& tally = tallies[request.mode?scalar?"four-pion":"eta-pipi":"other families"];
        first.hooks->attempts = second.hooks->attempts = first.hooks->forced = 0;
        if (request.mode && scalar) {
          first.model->flav.script = Flav::Script::MENU;
          selected = fourPion(first, mass, request.seed, fourPionStates, roots, tally);
        } else if (request.mode) {
          double threshold = 0.;
          for (int pdg:request.primary) threshold += first.had->particleData.m0(pdg);
          if (threshold >= mass || !(first.acceptance(request.primary)>0.)) throw std::runtime_error("unsupported conditional primary");
          first.model->flav.script = Flav::Script::ORDERED;
          first.model->flav.configure(request.primary);
          selected = eta(first, mass, request.seed, request.primary, roots, tally);
        } else {
          if (spec->rateFirst) first.model->flav.script = Flav::Script::SINGLETS;
          selected = sampler.draw(mass, request.seed, roots, tally);
        }
        portable_rejection::State cut;
        if (!matchedOwnershipCut(selected->dec->event, roots, {}, cut)) throw std::runtime_error("lost ownership");
        int terminals = 0;
        Vec4 sum;
        for (int j = 0; j<selected->dec->event.size(); ++j) {
          const auto& q = selected->dec->event[j];
          if (!q.isFinal() || !q.id()) continue;
          ++terminals;
          sum += q.p();
        }
        if (std::max({std::abs(sum.px()), std::abs(sum.py()), std::abs(sum.pz()), std::abs(sum.e()-mass)})>2e-6*mass)
          throw std::runtime_error("conditional conservation failure");
        ++tally.events;
        tally.attempts += first.hooks->attempts+second.hooks->attempts;
        tally.forced += first.hooks->forced;
        std::cout<<"EVENT "<<i<<' '<<terminals<<' '<<componentName(selected->rc.comp)<<' '<<portable_rejection::key(cut)<<'\n';
        for (int j = 0; j<selected->dec->event.size(); ++j) {
          const auto& q = selected->dec->event[j];
          if (!q.isFinal() || !q.id()) continue;
          std::cout<<q.px()<<' '<<q.py()<<' '<<q.pz()<<' '<<q.e()<<' '<<q.id()<<'\n';
        }
      }
      std::cout<<"END "<<events<<'\n'<<std::flush;
    }
    for (const auto& [family, t]:tallies)
      std::cerr<<"BATCH_COUNTERS "<<family<<" events "<<t.events<<" attempts "<<t.attempts<<" forced "<<t.forced
               <<" strings "<<t.strings<<" decays "<<t.decays<<" switches "<<t.switches<<" retries "<<t.retries<<'\n';
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
