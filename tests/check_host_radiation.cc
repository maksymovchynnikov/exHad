// The host's Pythia adds no radiation to the products of an exHad decay, for every mother code:
// 200 decays per code, counting the final state Pythia shows against the one exHad delivered.
// With allowHostRadiation() the host radiates off the lepton pairs again.
#include "exhad/Pythia8DecayHandler.hpp"
#include <iostream>

namespace {
struct Counts {
  int twoLepton = 0, shower = 0, extra = 0, photons = 0;
};

// Decays `events` dark photons of 2 GeV through the mother code `mother`, with the host allowed
// to radiate off exHad's products or not. Counts the entries Pythia's shower added (status 51 and
// 52), and, for the decays in which exHad delivered an electron or muon pair, the final-state
// particles beyond that pair and the photons among them.
Counts scan(const std::shared_ptr<exhad::Generator>& generator, const char* xmldoc, int mother,
            bool allowRadiation, int events) {
  Pythia8::Pythia pythia(xmldoc, false);
  pythia.particleData.m0(mother, 2.0);
  pythia.particleData.tau0(mother, 1000.);
  // HadronLevel:QED is the host's photon radiation in two-body decays; the codes outside Pythia's
  // hidden-sector range radiate through it alone.
  for (const char* setting : {"ProcessLevel:all = off", "PartonLevel:all = off",
                              "HadronLevel:Hadronize = off", "HadronLevel:QED = on",
                              "Print:quiet = on"})
    pythia.readString(setting);
  exhad::Pythia8DecayHandler::keepTransportParticlesStable(pythia);
  auto handler = std::make_shared<exhad::Pythia8DecayHandler>(
      generator, mother, exhad::DecayScope::All, 12345, 100);
  if (allowRadiation) handler->allowHostRadiation();
  handler->prepareParticle(pythia);
  pythia.setDecayPtr(handler);
  if (!pythia.init()) throw std::runtime_error("Pythia did not initialize");
  Counts counts;
  for (int event = 0; event < events; ++event) {
    pythia.event.reset();
    const int index = pythia.event.append(mother, 1, 0, 0, Pythia8::Vec4(0., 0., 10., std::sqrt(104.)), 2.);
    if (!pythia.moreDecays() || pythia.event[index].isFinal())
      throw std::runtime_error("The mother did not decay");
    for (int i = index + 1; i < pythia.event.size(); ++i)
      counts.shower += pythia.event[i].statusAbs() == 51 || pythia.event[i].statusAbs() == 52;
    // exHad's own output: the daughters of the mother, which the shower leaves in place.
    const int first = pythia.event[index].daughter1(), last = pythia.event[index].daughter2();
    const bool leptonPair = last - first == 1
        && (pythia.event[first].idAbs() == 11 || pythia.event[first].idAbs() == 13)
        && pythia.event[last].idAbs() == pythia.event[first].idAbs();
    if (!leptonPair) continue;
    ++counts.twoLepton;
    for (int i = index + 1; i < pythia.event.size(); ++i)
      if (pythia.event[i].isFinal()) {
        ++counts.extra;
        counts.photons += pythia.event[i].id() == 22;
      }
    counts.extra -= 2;  // The pair exHad delivered.
  }
  return counts;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  try {
    auto generator = std::make_shared<exhad::Generator>(argv[1], argv[2], "dark-photon");
    for (int mother : {4900022, 32, 35, 36, 9900012}) {
      const Counts off = scan(generator, argv[3], mother, false, 200);
      const Counts on = scan(generator, argv[3], mother, true, 200);
      std::cout << mother << ' ' << off.twoLepton << ' ' << off.shower << ' ' << off.extra << ' '
                << off.photons << ' ' << on.twoLepton << ' ' << on.photons << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
