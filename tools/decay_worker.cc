// Finish already sampled, colourless primary decays. Never resample their
// identities or momenta. Decays run at full hadron level, so partonic decay
// products fragment; each reply is checked for charge and baryon-number conservation.
#include "Pythia8/Pythia.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    // Optional --stable=ID[,ID...]: further particles kept undecayed (all others decay as before).
    const std::string option = argc == 3 ? argv[2] : "";
    if (argc < 2 || argc > 3 || (argc == 3 && option.rfind("--stable=", 0) != 0))
      throw std::runtime_error("Expected Pythia XML directory and optional --stable=ID[,ID...]");
    auto* protocol = std::cout.rdbuf();
    std::cout.rdbuf(std::cerr.rdbuf());
    Pythia8::Pythia p(argv[1], false);
    p.readString("ProcessLevel:all = off");
    p.readString("PartonLevel:all = off");
    p.readString("HadronLevel:Decay = on");
    p.readString("Print:quiet = on");
    p.readString("Random:setSeed = on");
    p.readString("Random:seed = 1");
    std::vector<int> stable{13, 211, 321, 130, 2112};
    if (argc == 3) {
      std::istringstream list(option.substr(9));
      for (std::string field; std::getline(list, field, ',');) {
        char* end = nullptr;
        long id = std::strtol(field.c_str(), &end, 10);
        if (field.empty() || *end || id < -99999999 || id > 99999999 || !p.particleData.isParticle(id))
          throw std::runtime_error("Invalid stable particle " + field);
        stable.push_back(id);
      }
    }
    for (int id : stable)
      p.readString(std::to_string(id) + ":mayDecay = off");
    if (!p.init()) throw std::runtime_error("Decayer initialization failed");
    std::ostream out(protocol);
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::istringstream input(line);
        int seed, count;
        if (!(input >> seed >> count) || seed < 1 || seed > 900000000 || count < 1 || count > 1000)
          throw std::runtime_error("Invalid decay request");
        struct Primary {
          int id;
          double x, y, z, e, m;
        };
        std::vector<Primary> primaries(count);
        Pythia8::Vec4 total;
        int charge = 0, baryon = 0;
        for (auto& a : primaries) {
          if (!(input >> a.id >> a.x >> a.y >> a.z >> a.e >> a.m)
              || !p.particleData.isParticle(a.id) || p.particleData.colType(a.id) != 0
              || !std::isfinite(a.e+a.x+a.y+a.z+a.m) || a.e < 0 || a.m < 0)
            throw std::runtime_error("Invalid colourless primary");
          total += Pythia8::Vec4(a.x, a.y, a.z, a.e);
          charge += p.particleData.chargeType(a.id);
          if (p.particleData.isBaryon(a.id)) baryon += a.id > 0 ? 1 : -1;
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("Trailing request fields");
        p.rndm.init(seed);
        bool ok = false;
        for (int attempt = 0; attempt<100 && !ok; ++attempt) {
          p.event.reset();
          p.event[0].p(total);  // the event check compares finals with the system
          p.event[0].m(total.mCalc());
          for (const auto& a : primaries)
            p.event.append(a.id, 91, 0, 0, a.x, a.y, a.z, a.e, a.m);
          ok = p.forceHadronLevel();
        }
        if (!ok) throw std::runtime_error("Secondary decay failed; selected channel retained");
        int n = 0;
        for (int i = 1; i<p.event.size(); ++i) if (p.event[i].isFinal()) {
          const auto& a = p.event[i];
          ++n;
          charge -= a.chargeType();
          if (a.particleDataEntry().isBaryon()) baryon -= a.id() > 0 ? 1 : -1;
        }
        if (charge != 0 || baryon != 0)
          throw std::runtime_error("Secondary decay violates charge or baryon-number conservation");
        out << n << std::setprecision(17);
        for (int i = 1; i<p.event.size(); ++i) if (p.event[i].isFinal()) {
          const auto& a = p.event[i];
          out << ' ' << a.px() << ' ' << a.py() << ' ' << a.pz()
              << ' ' << a.e() << ' ' << a.m() << ' ' << a.id();
        }
        out << '\n' << std::flush;
      } catch (const std::exception& e) {
        out << "ERROR " << e.what() << '\n' << std::flush;
      }
    }
    // Restore cout before the saved stream buffer leaves scope.
    std::cout.rdbuf(protocol);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
