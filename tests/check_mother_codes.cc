// Pythia8DecayHandler on a model's default mother entry: 20 of 20 decays, momentum conserved,
// host tau0 and zero width kept (Pythia's entries 32, 35, 36 and 9900012 are resonances).
#include "exhad/Pythia8DecayHandler.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 5) return 2;
  try {
    const std::string model = argv[4];
    auto generator = std::make_shared<exhad::Generator>(argv[1], argv[2], model);
    Pythia8::Pythia pythia(argv[3], false);
    const int mother = exhad::defaultMotherPdg(model);
    pythia.particleData.m0(mother, 2.0);
    pythia.particleData.tau0(mother, 1000.);
    for (const char* setting : {"ProcessLevel:all = off", "PartonLevel:all = off",
                                "HadronLevel:Hadronize = off", "Print:quiet = on"})
      pythia.readString(setting);
    exhad::Pythia8DecayHandler::keepTransportParticlesStable(pythia);
    auto handler = std::make_shared<exhad::Pythia8DecayHandler>(
        generator, exhad::DecayScope::All, 12345, 8, std::array<double, 3>{model == "hnl" ? 1. : 0., 0., 0.});
    handler->prepareParticle(pythia);
    pythia.setDecayPtr(handler);
    if (!pythia.init()) return 3;
    int decayed = 0;
    for (int event = 0; event < 20; ++event) {
      pythia.event.reset();
      Pythia8::Vec4 parent(0., 0., 10., std::sqrt(104.));
      int index = pythia.event.append(mother, 1, 0, 0, parent, 2.);
      if (!pythia.moreDecays() || pythia.event[index].isFinal()) continue;
      Pythia8::Vec4 total;
      for (int i = index + 1; i < pythia.event.size(); ++i)
        if (pythia.event[i].isFinal()) total += pythia.event[i].p();
      total -= parent;
      decayed += std::abs(total.e()) + total.pAbs() < 1e-7 * parent.e();
    }
    std::cout << mother << ' ' << decayed << ' ' << pythia.particleData.tau0(mother) << ' '
              << pythia.particleData.mWidth(mother) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
