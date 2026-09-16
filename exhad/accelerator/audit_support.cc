// Build-time audit of the string kernel for one initialized tune: exhaustive
// eta(')-pi-pi (alp-fermion) or four-pion (scalar) primary support; for every portal the
// enumerated one-step laws against stock StringFlav draws and bit-identical pass-through
// of Model and Hooks (for the vector current, of its open q-qbar sources).
#define main ordinary_exhad_main
#include "../../cpp/src/main.cc"
#undef main
#include "string_kernel.h"
#include <functional>

using Signature = std::vector<int>;
using Alternatives = std::set<Signature>;
bool subset(const Signature& small, const Signature& large) {
  return std::includes(large.begin(), large.end(), small.begin(), small.end());
}

// One-step laws of the SINGLETS stream (and, with `menu`, of the scalar MENU script):
// 2e7 stock break iterations and final joins per end flavour, |z| < 5.
void auditLaws(string_kernel::Flav& draw, bool menu) {
  auto probability = [](const string_kernel::Flav::Law& law, unsigned h) {
    return law.cumulative[h].second-(h?law.cumulative[h-1].second:0.);
  };
  for (int q = 1; q <= 3; ++q) {
    std::map<int, double> steps, joins, breaks, charged;
    double iterations = 0., completed = 0.;
    const double n = 2e7;
    for (double k = 0; k<n; ++k) {
      FlavContainer old(q, 1);
      int id = 0;
      do {
        FlavContainer next = draw.pick(old, -1., -1., true);
        id = draw.combine(old, next);
        ++iterations;
        if (id && next.id == -q) steps[id] += 1.;
      }while (!id);
      ++completed;
      breaks[id] += 1.;
      FlavContainer a(q), b(-q);
      id = 0;
      for (int t = 0; t<string_kernel::JOIN_TRIES && !id; ++t) id = draw.combine(a, b);
      joins[id] += 1.;
      if (menu && q<3) {  // the join of a light end with its isospin partner
        FlavContainer c(q), d(q-3);
        id = 0;
        for (int t = 0; t<string_kernel::JOIN_TRIES && !id; ++t) id = draw.combine(c, d);
        charged[id] += 1.;
      }
    }
    auto check = [](double count, double trials, long double p, const char* what) {
      const double z = (count-trials*double(p))/std::sqrt(trials*double(p*(1-p)));
      if (!(std::abs(z)<5.)) throw std::runtime_error(std::string("one-step law disagrees with stock draws: ")+what);
    };
    check(completed, iterations, 1-draw.failure(q), "iteration success");
    for (unsigned h = 0; h<draw.step[q].cumulative.size(); ++h) {
      const int id = draw.step[q].cumulative[h].first;
      check(steps[id], completed, probability(draw.step[q], h), "break");
      check(joins[id], n, probability(draw.join[q], h), "join");
    }
    // MENU laws from a u (d) end: pi+ rho+ (pi- rho-) with its partner, pi0 rho0 diagonally.
    if (menu && q<3) for (int kind:{111, 113, 211, 213}) {
      const bool isCharged = kind == 211 || kind == 213;
      const int id = isCharged && q == 1?-kind:kind;
      check(breaks[id], completed, draw.breakLaw(kind), "menu break");
      check((isCharged?charged:joins)[id], n, draw.joinLaw(kind), "menu join");
    }
  }
  std::cout<<std::setprecision(6)<<"PASS singlet break and join laws, 2e7 stock draws per end flavour:";
  for (int q = 1; q <= 3; ++q) std::cout<<" q="<<q<<" Z="<<double(1-draw.failure(q))<<" T_s="<<double(draw.step[q].total)
                               <<" J_s="<<double(draw.join[q].total);
  std::cout<<'\n';
  if (menu) {
    std::cout<<"PASS menu break and join laws, 2e7 stock draws per light end:";
    for (int kind:{111, 113, 211, 213}) std::cout<<" "<<kind<<": B="<<double(draw.breakLaw(kind))<<" J="<<double(draw.joinLaw(kind));
    std::cout<<'\n';
  }
}

// Stock-identity gate of the open q-qbar sources of the isoscalar vector current,
// including the low-mass dispatch the strange source takes at the 2 GeV endpoint.
int auditVector(const std::vector<std::string>& settings) {
  for (double mass:{2.0, 3.6}) for (int quark:{2, 3}) {
    RunConfig rc{};
    rc.particle = {2, -1, -1, 0, true};  // 1^-- of the isoscalar vector current
    rc.carriesParentSpin = true;
    rc.decay = DecayMode::QQ;
    rc.comp = quark == 3?VComponent::PHI:VComponent::OMEGA;
    rc.undetVeto = true;
    rc.stopMass = 0.;
    rc.pythiaSets = settings;
    rc.pythiaSets.push_back("HadronLevel:mStringMin=1.0");
    auto stock = makePythia(Stage::HADRONIZER, rc), kernel = makePythia(Stage::HADRONIZER, rc);
    auto model = std::make_shared<string_kernel::Model>();
    auto hooks = std::make_shared<string_kernel::Hooks>();
    hooks->flav = &model->flav;
    kernel->setUserHooksPtr(hooks);
    kernel->setFragmentationPtr(model);
    model->lowMass = true;
    if (!kernel->init()) throw std::runtime_error("string kernel initialization failed");
    string_kernel::certify(*kernel, mass, false, false);
    for (int i = 1; i <= 20000; ++i) for (Pythia* p:{stock.get(), kernel.get()}) {
      exhad_rng::initialize(p->rndm, i);
      buildPair(*p, mass, quark, -quark, polarMode(rc));
      p->forceHadronLevel();
      if (p == kernel.get()) for (int j = 0; j<std::max(p->event.size(), stock->event.size()); ++j)
        if (j >= p->event.size() || j >= stock->event.size() || p->event[j].id() != stock->event[j].id()
           || p->event[j].px() != stock->event[j].px() || p->event[j].pz() != stock->event[j].pz() || p->event[j].e() != stock->event[j].e())
          throw std::runtime_error("string kernel pass-through differs from stock Pythia");
    }
    std::cout<<"PASS pass-through identity, 20000 "<<(quark == 3?"s sbar":"u ubar")<<" strings at "<<mass<<" GeV\n";
    if (mass != 3.6 || quark != 3) continue;
    auditLaws(model->flav, false);
    std::cout<<std::setprecision(6)<<"PASS two-hadron all-singlet chances:";
    for (int q = 1; q <= 3; ++q) std::cout<<" q="<<q<<" chance(2)="<<double(model->flav.chance(2, q));
    std::cout<<'\n';
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("args: settings-file alp-fermion|scalar|b-l");
    std::vector<std::string> settings;
    std::ifstream input(argv[1]);
    std::string line;
    while (std::getline(input, line)) if (!line.empty()) settings.push_back(line);
    const std::string portal = argv[2];
    if (portal == "b-l") return auditVector(settings);
    if (portal != "alp-fermion" && portal != "scalar") throw std::runtime_error("unsupported audited portal");
    const bool scalar = portal == "scalar";
    RunConfig rc{};
    rc.particle = {0, scalar?+1:-1, +1, 0, true};  // 0^++ or 0^-+
    rc.carriesParentSpin = true;
    rc.decay = DecayMode::GG;
    rc.comp = VComponent::GLUE;
    rc.undetVeto = true;
    rc.stopMass = 0.;
    rc.pythiaSets = settings;
    const PythiaChain chain = makeChain(rc);
    const auto& had = chain.had;
    const auto& chn = chain.chn;
    exhad::accelerator::FiniteMesonChoice flav;
    flav.initInfoPtr(const_cast<Info&>(had->info));
    flav.init();
    std::set<int> possible;
    for (int q = 1; q <= 3; ++q) for (int qb = 1; qb <= 3; ++qb) {
      const auto law = flav.oneCall(FlavContainer(q), FlavContainer(-qb));
      for (const auto& outcome:law.outcomes())
        if (outcome.pdg && outcome.probability>0) possible.insert(outcome.pdg);
    }
    const std::set<int> expected{-323, -321, -313, -311, -213, -211, 111, 113, 211, 213, 221, 223, 311, 313, 321, 323, 331, 333};
    if (possible != expected) {
      std::cerr<<"ACTUAL PRIMARY SUPPORT";
      for (int id:possible) std::cerr<<' '<<id;
      std::cerr<<'\n';
      throw std::runtime_error("primary meson support differs from certified PS/V basis");
    }
    DiscreteSymmetryFilter symmetry(had->particleData, sourceQN(rc, rc.comp));
    // Primary multisets of at most maxSize PS/V mesons whose prompt decays can end in
    // exactly the canonical target, and those the source filter allows.
    auto support = [&](const Signature& target, unsigned maxSize, Alternatives& candidates, Alternatives& survivors) {
      std::map<int, Alternatives> memo;
      std::set<int> active;
      std::function<Alternatives(int)> cuts = [&](int id)->Alternatives {
        if (memo.count(id)) return memo.at(id);
        if (portable_rejection::canonical(id)) return subset({id}, target)?Alternatives{{id}}:Alternatives{};
        if (!chn->particleData.isParticle(id) || !chn->particleData.mayDecay(id)) return {};
        if (active.count(id)) throw std::runtime_error("cyclic prompt decay support");
        active.insert(id);
        Alternatives result;
        auto particle = chn->particleData.particleDataEntryPtr(id);
        for (int k = 0; k<particle->sizeChannels(); ++k) {
          const auto& channel = particle->channel(k);
          if (!channel.onMode() || channel.bRatio() <= 0.) continue;
          Alternatives partial{{}};
          for (int j = 0; j<channel.multiplicity(); ++j) {
            int daughter = channel.product(j);
            if (id<0 && chn->particleData.hasAnti(daughter)) daughter = -daughter;
            const auto choices = cuts(daughter);
            Alternatives next;
            for (const auto& prior:partial) for (const auto& choice:choices) {
              Signature combined = prior;
              combined.insert(combined.end(), choice.begin(), choice.end());
              std::sort(combined.begin(), combined.end());
              if (subset(combined, target)) next.insert(combined);
            }
            partial = std::move(next);
            if (partial.empty()) break;
          }
          result.insert(partial.begin(), partial.end());
        }
        active.erase(id);
        memo[id] = result;
        return result;
      };
      const Signature ids(possible.begin(), possible.end());
      std::function<void(Signature, unsigned)> enumerate = [&](Signature primary, unsigned first) {
        if (!primary.empty()) {
          Alternatives partial{{}};
          for (int id:primary) {
            Alternatives next;
            for (const auto& prior:partial) for (const auto& choice:cuts(id)) {
              Signature combined = prior;
              combined.insert(combined.end(), choice.begin(), choice.end());
              std::sort(combined.begin(), combined.end());
              if (subset(combined, target)) next.insert(combined);
            }
            partial = std::move(next);
          }
          if (partial.count(target)) {
            candidates.insert(primary);
            if (symmetry.acceptance(primary)>0.) survivors.insert(primary);
          }
        }
        if (primary.size() == maxSize) return;
        for (unsigned i = first; i<ids.size(); ++i) {
          auto next = primary;
          next.push_back(ids[i]);
          enumerate(next, i);
        }
      };
      enumerate({}, 0);
    };
    if (scalar) {
      // Every primary has at least one canonical terminal, so four pions need at most four
      // primaries; strings with u, d or s ends have at least three (certifyThreeHadrons).
      Alternatives menu, allowed;
      for (const auto& entry:string_kernel::fourPionMenu([&](const std::vector<int>& ids) {return symmetry.acceptance(ids);})) {
        Signature kinds = entry.breaks;
        kinds.push_back(entry.join);
        std::sort(kinds.begin(), kinds.end());
        for (const auto& state:string_kernel::chargeStates(kinds)) menu.insert(state);
      }
      for (Signature target:std::vector<Signature>{{-211, -211, 211, 211}, {-211, 111, 111, 211}, {111, 111, 111, 111}}) {
        Alternatives candidates, survivors;
        int twoHadron = 0;
        support(target, 4, candidates, survivors);
        for (const auto& state:survivors) if (state.size() >= 3) allowed.insert(state);
        else ++twoHadron;
        std::cout<<"TARGET";
        for (int id:target) std::cout<<' '<<id;
        std::cout<<" CANDIDATES "<<candidates.size()<<" ALLOWED "<<survivors.size()<<" TWO_HADRON "<<twoHadron<<'\n';
      }
      if (allowed != menu) throw std::runtime_error("four-pion menu differs from the allowed primary support");
      // No end or join with an s quark gives a pion or rho: s-sbar strings and closed
      // loops with an s start have no four-pion history.
      for (int q = 1; q <= 3; ++q) for (int f = 1; f <= 3; ++f) for (int sign:{1, -1}) {
        if (q<3 && f<3) continue;
        const auto law = flav.oneCall(FlavContainer(sign*q), FlavContainer(-sign*f));
        for (int id:{111, 113, 211, 213, -211, -213})
          if (law.probability(id)>0) throw std::runtime_error("strange flavour path to a menu meson");
      }
      std::cout<<"PASS exact four-pion support: "<<menu.size()<<" primary states of at least three pions and rhos, no strange path\n";
    } else for (Signature target:std::vector<Signature>{{111, 111, 221}, {-211, 211, 221}, {111, 111, 331}, {-211, 211, 331}}) {
      Alternatives survivors, candidates;
      support(target, 3, candidates, survivors);
      if (survivors != Alternatives{target}) throw std::runtime_error("canonical channel has another allowed primary ancestor");
      int strangePaths = 0;
      Signature order = target;
      do {
        for (bool firstPos:{false, true}) for (bool secondPos:{false, true})
          for (int firstQ = 1; firstQ <= 3; ++firstQ) for (int secondQ = 1; secondQ <= 3; ++secondQ) {
            int pos = 3, neg = -3;
            int old = firstPos?pos:neg;
            int created = (old>0?-1:1)*firstQ;
            const auto first = flav.oneCall(FlavContainer(old), FlavContainer(created));
            if (first.probability(order[0]) == 0) continue;
            (firstPos?pos:neg) = -created;
            old = secondPos?pos:neg;
            created = (old>0?-1:1)*secondQ;
            const auto second = flav.oneCall(FlavContainer(old), FlavContainer(created));
            if (second.probability(order[1]) == 0) continue;
            const auto last = flav.oneCall(FlavContainer(secondPos?-created:pos),
                                          FlavContainer(secondPos?neg:-created));
            if (last.probability(order[2])>0)++strangePaths;
          }
      } while (std::next_permutation(order.begin(), order.end()));
      if (strangePaths) throw std::runtime_error("strange source has a three-primary flavour path");
      std::cout<<"TARGET";
      for (int id:target) std::cout<<' '<<id;
      std::cout<<" CANDIDATES "<<candidates.size()<<" ALLOWED "<<survivors.size()
               <<" STRANGE_PATHS "<<strangePaths<<'\n';
    }
    if (!scalar) std::cout<<"PASS exact three-primary support for all four charge channels\n";
    // Pass-through: Model and Hooks with the STOCK script reproduce stock strings bit for bit.
    for (bool glue:{true, false}) {
      RunConfig source = rc;
      source.decay = glue?DecayMode::GG:DecayMode::QQ;
      source.comp = glue?VComponent::GLUE:VComponent::STRANGE;
      auto stock = makePythia(Stage::HADRONIZER, source), kernel = makePythia(Stage::HADRONIZER, source);
      auto model = std::make_shared<string_kernel::Model>();
      auto hooks = std::make_shared<string_kernel::Hooks>();
      hooks->flav = &model->flav;
      kernel->setUserHooksPtr(hooks);
      kernel->setFragmentationPtr(model);
      if (!kernel->init()) throw std::runtime_error("string kernel initialization failed");
      for (int i = 1; i <= 20000; ++i) for (Pythia* p:{stock.get(), kernel.get()}) {
        exhad_rng::initialize(p->rndm, i);
        buildPair(*p, 3.6, glue?21:3, glue?21:-3, PolarMode::ISOTROPIC);
        p->forceHadronLevel();
        if (p == kernel.get()) for (int j = 0; j<std::max(p->event.size(), stock->event.size()); ++j)
          if (j >= p->event.size() || j >= stock->event.size() || p->event[j].id() != stock->event[j].id()
             || p->event[j].px() != stock->event[j].px() || p->event[j].pz() != stock->event[j].pz() || p->event[j].e() != stock->event[j].e())
            throw std::runtime_error("string kernel pass-through differs from stock Pythia");
      }
      std::cout<<"PASS pass-through identity, 20000 "<<(glue?"gg":"s sbar")<<" strings at 3.6 GeV\n";
      if (!glue) continue;
      auditLaws(model->flav, scalar);
    }
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
