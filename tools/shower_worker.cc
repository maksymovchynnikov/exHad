// Colour-singlet two-parton subsystem: preserve its supplied four-momentum,
// shower once, fragment, then decay. Rates and spectator momenta are host-owned.
// Decays run at full hadron level in a second instance, so partonic decay
// products fragment; each reply is checked for charge and baryon-number conservation.
#include "Pythia8/Pythia.h"
#include "symmetry_filter.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  auto* protocol = std::cout.rdbuf();
  std::cout.rdbuf(std::cerr.rdbuf());
  try {
    // "<2J> <P> <C|none> <charge>": only a spin-0 source projects its hadrons.
    const DecayingParticle source = DecayingParticle::read(argv[2], argv[3], argv[4], argv[5]);
    const bool spinZero = source.j2 == 0;
    Pythia8::Pythia p(argv[1], false);
    p.readString("ProcessLevel:all = off");
    p.readString("PartonLevel:FSRinResonances = off");
    p.readString("Print:quiet = on");
    p.readString("Random:setSeed = on");
    p.readString("Random:seed = 1");
    for (int id : {13, 211, 321, 130, 2112})
      p.readString(std::to_string(id) + ":mayDecay = off");
    Pythia8::Pythia d(p.settings, p.particleData, false);  // same settings, decays on
    p.readString("HadronLevel:Decay = off");  // the spin-0 filter sees primary hadrons
    if (!p.init() || !d.init()) throw std::runtime_error("Shower worker initialization failed");
    DiscreteSymmetryFilter filter(p.particleData, source.baseline());
    std::ostream out(protocol);
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::istringstream in(line);
        int seed, n;
        if (!(in >> seed >> n) || seed<1 || seed>900000000 || n != 2)
          throw std::runtime_error("Expected two partons and a valid seed");
        struct Parton {
          int id;
          double x, y, z, e, m;
        };
        Parton a[2];
        Pythia8::Vec4 total;
        for (auto& q : a) {
          if (!(in >> q.id >> q.x >> q.y >> q.z >> q.e >> q.m)
              || !std::isfinite(q.x+q.y+q.z+q.e+q.m) || q.m<0 || q.e<q.m)
            throw std::runtime_error("Malformed parton");
          total += Pythia8::Vec4(q.x, q.y, q.z, q.e);
        }
        std::string extra;
        if (in>>extra) throw std::runtime_error("Trailing fields");
        bool gg = a[0].id == 21 && a[1].id == 21;
        if (!gg && !(a[0].id*a[1].id<0 && std::abs(a[0].id) <= 5 && std::abs(a[1].id) <= 5))
          throw std::runtime_error("Partons are not a colour-singlet pair");
        if (total.pAbs()>1e-7*total.e() || total.mCalc() <= 0)
          throw std::runtime_error("Parton pair must be in its rest frame");
        const double scale = total.mCalc()/2.;
        p.rndm.init(seed);
        bool ok = false;
        for (int attempt = 0; attempt<10000 && !ok; ++attempt) {
          p.event.reset();
          for (int i = 0; i<2; ++i) {
            auto& q = a[i];
            int col = gg ? 101+i : (q.id>0 ? 101:0);
            int acol = gg ? 102-i : (q.id<0 ? 101:0);
            int k = p.event.append(q.id, 23, col, acol, q.x, q.y, q.z, q.e, q.m);
            p.event[k].scale(scale);
          }
          // Official Pythia standalone-shower interface (example main234).
          p.forceTimeShower(1, 2, scale);
          if (!p.next()) continue;
          if (spinZero && !filter.supports(p.event)) continue;
          // Continue like cpp/src/main.cc continueEvent (rebind particle data)
          // on the same random stream.
          d.event = p.event;
          d.event.init("(copied event)", &d.particleData);
          for (int i = 0; i<d.event.size(); ++i) d.event[i].setPDEPtr(d.particleData.findParticle(d.event[i].id()));
          d.rndm.setState(p.rndm.getState());
          ok = d.forceHadronLevel();
          p.rndm.setState(d.rndm.getState());
        }
        if (!ok) throw std::runtime_error("Shower/fragmentation exhausted at fixed channel and momenta");
        Pythia8::Vec4 final;
        int count = 0, charge = d.particleData.chargeType(a[0].id)+d.particleData.chargeType(a[1].id), baryon = 0;
        for (int i = 1; i<d.event.size(); ++i) if (d.event[i].isFinal()) {
          auto& q = d.event[i];
          if (q.isParton()) throw std::runtime_error("Undecayed parton in output");
          final += q.p();
          ++count;
          charge -= q.chargeType();
          if (q.particleDataEntry().isBaryon()) baryon += q.id()>0 ? 1:-1;
        }
        auto residual = final-total;
        if (residual.pAbs()>2e-7*total.e() || std::abs(residual.e())>2e-7*total.e())
          throw std::runtime_error("Shower output violates four-momentum conservation");
        if (charge != 0 || baryon != 0)
          throw std::runtime_error("Shower output violates charge or baryon-number conservation");
        out<<count<<std::setprecision(17);
        for (int i = 1; i<d.event.size(); ++i) if (d.event[i].isFinal()) {
          auto& q = d.event[i];
          out<<' '<<q.px()<<' '<<q.py()<<' '<<q.pz()<<' '<<q.e()<<' '<<q.m()<<' '<<q.id();
        }
        out<<'\n'<<std::flush;
      } catch (const std::exception& e) {
        out<<"ERROR "<<e.what()<<'\n'<<std::flush;
      }
    }
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    std::cout.rdbuf(protocol);
    return 1;
  }
  std::cout.rdbuf(protocol);
}
