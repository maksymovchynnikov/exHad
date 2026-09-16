#pragma once
#include "Generator.hpp"
#include "Pythia8/Pythia.h"
#include <memory>

namespace exhad {
// Register before Pythia::init(). Pythia keeps production, lifetime and vertex
// handling; exHad supplies unweighted daughters and never reuses an event.
class Pythia8DecayHandler : public Pythia8::DecayHandler {
 public:
  // The mother is the model's defaultMotherPdg.
  Pythia8DecayHandler(std::shared_ptr<Generator> generator, DecayScope scope,
      std::uint64_t seed = 1, std::size_t batchSize = 64,
      std::array<double, 3> mixingSquared = {0., 0., 0.})
      : Pythia8DecayHandler(generator, generator ? defaultMotherPdg(generator->model()) : 0,
                            scope, seed, batchSize, mixingSquared) {}
  Pythia8DecayHandler(std::shared_ptr<Generator> generator, int motherPdg,
      DecayScope scope, std::uint64_t seed = 1, std::size_t batchSize = 64,
      std::array<double, 3> mixingSquared = {0., 0., 0.})
      : generator_(std::move(generator)), mother_(std::abs(motherPdg)), scope_(scope),
        seed_(seed), batchSize_(batchSize), mixing_(mixingSquared) {
    if (!generator_ || !mother_ || !batchSize_ || batchSize_ > 1000000)
      throw std::invalid_argument("Invalid exHad decay-handler configuration");
  }
  std::vector<int> handledParticles() override {
    return {mother_};
  }
  // exHad calculates its channel rates and momenta without radiation off the decay products, so
  // the host's Pythia must not add any to them. Switch this on to let it radiate off them anyway.
  void allowHostRadiation(bool allow = true) {
    hostRadiation_ = allow;
  }
  // Pythia 8.317's canDecay() checks for a table entry even for an external
  // decayer. An inactive, zero-BR slot only enables that dispatch; it is
  // never sampled and supplies no physical branching fraction. Pythia marks
  // 32, 35, 36 and 9900012 as resonances, which it decays itself and whose
  // width and lifetime it recalculates: the mother becomes an ordinary decaying
  // particle with the host's m0 and tau0 and zero width.
  void prepareParticle(Pythia8::Pythia& pythia) const {
    auto entry = pythia.particleData.findParticle(mother_);
    if (!entry) throw std::invalid_argument("Define the mother in Pythia before attaching exHad");
    if (!entry->canDecay()) entry->addChannel(0, 0., 0, 22, 22);
    entry->setMayDecay(true);
    entry->setIsResonance(false);
    entry->setTauCalc(false);
    entry->setMWidth(0.);
    entry->setDoForceWidth(true);
  }
  bool decay(std::vector<int>& ids, std::vector<double>& masses,
             std::vector<Pythia8::Vec4>& momenta, int, const Pythia8::Event&) override {
    if (ids.size() != 1 || masses.size() != 1 || momenta.size() != 1
        || std::abs(ids[0]) != mother_)
      throw std::runtime_error("exHad received an unexpected Pythia mother");
    if (cursor_ == cache_.size() || mass_ != masses[0]) {
      mass_ = masses[0];
      cache_ = generator_->generate(mass_, batchSize_, seed_++, scope_, mixing_);
      cursor_ = 0;
    }
    const auto& daughters = cache_.at(cursor_++);
    const auto parent = momenta[0];
    for (const auto& p : daughters) {
      Pythia8::Vec4 lab(p.px, p.py, p.pz, p.energy);
      lab.bst(parent);
      ids.push_back(p.pdg);
      masses.push_back(p.mass);
      momenta.push_back(lab);
    }
    // Pythia 8.317 may shower the products of an external decay: off a lepton pair from a mother
    // in its hidden-sector range 4900001-4999999 (ParticleDecays:FSRinDecays, on by default), and
    // off any two-body external decay with HadronLevel:QED. Both evolve down from the mother mass
    // in this array, and the hidden-sector branch also takes it as the starting scale of the
    // products. The handler hands back the final state exHad produced, so it returns a zero
    // starting scale: no emission is possible off these products, whatever the mother code, and
    // the decays the host performs itself keep their own settings. Pythia uses this element for
    // nothing else once an external decay handler has taken the decay.
    if (!hostRadiation_) masses[0] = 0.;
    return true;  // Failures throw; never silently substitute raw Pythia.
  }

  // Explicit opt-in: leave these long-lived daughters for detector transport.
  // This changes the host particle-data settings globally, not just exHad events.
  static void keepTransportParticlesStable(Pythia8::Pythia& pythia) {
    for (int id : {13, 211, 321, 130, 2112}) pythia.particleData.mayDecay(id, false);
  }
 private:
  std::shared_ptr<Generator> generator_;
  int mother_;
  DecayScope scope_;
  std::uint64_t seed_;
  std::size_t batchSize_, cursor_ = 0;
  bool hostRadiation_ = false;
  std::array<double, 3> mixing_;
  double mass_ = -1.;
  std::vector<Event> cache_;
};
}  // namespace exhad
