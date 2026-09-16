// Build/run instructions: docs/INTERFACES.md (C++ interface and Pythia 8). This is also a live integration test.
#include "exhad/Pythia8DecayHandler.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: pythia8_decays EXHAD_ROOT PYTHON PYTHIA_XMLDOC\n"; return 2;
  }
  try {
    auto generator = std::make_shared<exhad::Generator>(argv[1], argv[2], "alp-fermion");
    Pythia8::Pythia pythia(argv[3], false);
    // The ALP default mother A0 (36), a Pythia resonance. Mass and lifetime belong to the host.
    const int mother = exhad::defaultMotherPdg(generator->model());
    pythia.particleData.m0(mother, 2.0);
    pythia.particleData.tau0(mother, 1000.);
    pythia.readString("ProcessLevel:all = off");
    pythia.readString("PartonLevel:all = off");
    pythia.readString("HadronLevel:Hadronize = off");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("Print:quiet = on");
    exhad::Pythia8DecayHandler::keepTransportParticlesStable(pythia);
    auto handler = std::make_shared<exhad::Pythia8DecayHandler>(
        generator, exhad::DecayScope::All, 12345, 8);
    handler->prepareParticle(pythia);
    pythia.setDecayPtr(handler);
    if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");
    for (int event = 0; event < 20; ++event) {
      pythia.event.reset();
      double pz = event % 2 ? 10. : 0.;
      Pythia8::Vec4 parent(0., 0., pz, std::sqrt(4. + pz * pz));
      int index = pythia.event.append(mother, 1, 0, 0, parent, 2.);
      pythia.event[index].vProd(1., 2., 3., 0.);
      if (!pythia.moreDecays() || pythia.event[index].isFinal())
        throw std::runtime_error("Mother did not decay");
      Pythia8::Vec4 total;
      for (int i = index + 1; i < pythia.event.size(); ++i) if (pythia.event[i].isFinal()) {
        const auto& daughter = pythia.event[i];
        total += daughter.p();
        if (daughter.mother1() != index || std::abs(daughter.zProd() - 3.) > 1e-9)
          throw std::runtime_error("Wrong daughter parent or vertex");
      }
      auto difference = total - parent;
      if (std::abs(difference.e()) + difference.pAbs() > 1e-7 * parent.e())
        throw std::runtime_error("Four-momentum conservation failed");
    }
    if (pythia.particleData.tau0(mother) != 1000. || pythia.particleData.mWidth(mother) != 0.)
      throw std::runtime_error("Pythia replaced the host's mother lifetime or width");
    std::cout << "PASS: 20 full alp-fermion decays; rest/boosted momenta, mother links, vertices and host lifetime\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
