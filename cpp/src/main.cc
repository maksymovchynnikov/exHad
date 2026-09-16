// exhad: one colour-singlet source current -> Pythia 8.317 hadronizer ->
// symmetry filter -> prompt-decay (channel) instance -> full decayer.
// Two stdio front ends share that chain: the fixed-configuration event worker
// (exhad <kind> <qq|gg> ...), which reads the mass with every request, and the
// dedicated HNL run server (exhad --runs), which reads each run's mass,
// weights and tables from stdin. The conditional accelerator of the alp-fermion, scalar
// and B-L portals (exhad-batch) includes it.
#include "Pythia8/Pythia.h"

#include "symmetry_filter.h"
#include "portable_rejection.h"
#include "fast_pythia_seed.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Pythia8;

enum class DecayMode    { QQ, GG };
enum class PolarMode    { ISOTROPIC, TRANSVERSE_VECTOR };
enum class InjectedMassMode { POLE, PYTHIA_WIDTH };

// Source-current components.  String ends: RHO/OMEGA/NCAX
// u ubar or d dbar (50:50); PHI/STRANGE/NCSAX s sbar; CHARM c cbar; GLUE gg.
// Charged currents: CCVEC (1^-, I^G=1^+) and CCAX (1^+, 1^-) use d ubar
// (tau^- convention, hadronic charge -1); CCCD c dbar and CCCS c sbar use the
// positive reference convention.  A neutral current without C reuses
// RHO/OMEGA/PHI and adds the axial NCAX (a1_0-like) and NCSAX (f1(1420)-like).
// NONE is the injection or charm-continuum worker.
enum class VComponent { NONE, RHO, OMEGA, PHI,
                        STRANGE, CHARM, GLUE, CCVEC, CCAX, CCCD, CCCS,
                        NCAX, NCSAX };

struct RunConfig {
  DecayingParticle particle;              // quantum numbers, never a kind name
  bool carriesParentSpin = false;         // the spin-1 source is the parent's spin, not an average
  DecayMode decay = DecayMode::QQ;
  double stopMass = -1.0;                 // <0: Pythia default
  std::vector<std::string> pythiaSets;    // applied after stopMass
  VComponent comp = VComponent::NONE;
  int qid = 1;                            // --flavor, component none only
  bool undetVeto = false, noFilter = false;
  InjectedMassMode injectedMassMode = InjectedMassMode::POLE;
  std::vector<int> injectIds;             // --inject: one unit-probability mode
  std::vector<std::string> vetoRows;      // --veto: exclusive-owner keys
};

static constexpr const char* EVENT_WORKER_SCHEMA =
    "exhad-persistent-event-worker";
static constexpr int PYTHIA_MAX_SEED = 900000000;
static constexpr int DECAYER_SEED_OFFSET = 12345;
static constexpr int CHANNEL_DECAYER_SEED_OFFSET = 54321;
static constexpr int MAX_WORKER_BASE_SEED =
    PYTHIA_MAX_SEED - CHANNEL_DECAYER_SEED_OFFSET;

static constexpr std::pair<const char*, VComponent> COMPONENT_NAMES[] = {
    {"none", VComponent::NONE},
    {"rho", VComponent::RHO}, {"omega", VComponent::OMEGA},
    {"phi", VComponent::PHI}, {"strange", VComponent::STRANGE},
    {"charm", VComponent::CHARM}, {"glue", VComponent::GLUE},
    {"ccvec", VComponent::CCVEC}, {"ccax", VComponent::CCAX},
    {"cccd", VComponent::CCCD}, {"cccs", VComponent::CCCS},
    {"ncax", VComponent::NCAX}, {"ncsax", VComponent::NCSAX}};

static constexpr std::pair<const char*, DecayMode> DECAY_NAMES[] = {{"qq", DecayMode::QQ}, {"gg", DecayMode::GG}};
static constexpr std::pair<const char*, bool> PARENT_SPIN_NAMES[] = {{"carried", true}, {"averaged", false}};
static constexpr std::pair<const char*, int> FLAVOR_NAMES[] = {{"c", 4}};
static constexpr std::pair<const char*, InjectedMassMode> MASS_MODE_NAMES[] = {
    {"pole", InjectedMassMode::POLE}, {"width", InjectedMassMode::PYTHIA_WIDTH}};

// Protocol string (READY record and conditional event header).
static const char* componentName(VComponent component) {
  for (const auto& item : COMPONENT_NAMES)
    if (item.second == component) return item.first;
  return "unknown";
}

static bool chargedCurrent(VComponent c) {
  return c == VComponent::CCVEC || c == VComponent::CCAX
      || c == VComponent::CCCD || c == VComponent::CCCS;
}

static PolarMode polarMode(const RunConfig& rc) {
  return rc.particle.j2 == 2 && rc.carriesParentSpin && rc.decay == DecayMode::QQ
      ? PolarMode::TRANSVERSE_VECTOR : PolarMode::ISOTROPIC;
}

// The injection and charm-continuum workers realize the C-odd neutral spin-1
// current, the one source whose components the projector never restricts.
static bool cOddNeutralVector(const DecayingParticle& source) {
  return source.j2 == 2 && source.parity == -1 && source.cGiven && source.c == -1 && source.charge == 0;
}

// ---------------------------------------------------------------- command line

static std::string lowerAscii(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

template <class Value, std::size_t N>
static Value named(const std::pair<const char*, Value> (&table)[N], const std::string& name, const char* what) {
  const std::string key = lowerAscii(name);
  for (const auto& item : table)
    if (key == item.first) return item.second;
  throw std::runtime_error(std::string("unknown ") + what + " " + name);
}

// A strict protocol number: one whole finite token ("2abc", "nan" and an ID "211.5" are malformed).
template <class T>
static T number(const std::string& text) {
  std::istringstream in(text);
  T value;
  std::string extra;
  if (!(in >> value) || (in >> extra) || !std::isfinite(double(value))) throw std::runtime_error("malformed number " + text);
  return value;
}

static double positiveMass(const std::string& text) {
  const double mass = number<double>(text);
  if (!(mass > 0.0)) throw std::runtime_error("mass must be positive: " + text);
  return mass;
}

// Event-worker command line; the worker always vetoes undetermined states.
static RunConfig parseArgs(int argc, char** argv) {
  if (argc < 6)
    throw std::runtime_error(
        "Usage: ./exhad --runs | ./exhad <2J> <P> <C|none> <charge> <qq|gg> "
        "--parentSpin=carried|averaged "
        "[--component= --stopMass= --set=Key=value --flavor= (unfiltered) --inject=id,id,... "
        "--injectMasses=pole|width --veto=id:count ...]");
  RunConfig rc;
  rc.particle = DecayingParticle::read(argv[1], argv[2], argv[3], argv[4]);
  rc.decay = named(DECAY_NAMES, argv[5], "decay mode");
  rc.undetVeto = true;
  bool parentSpin = false;
  for (int i = 6; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto eq = arg.find('=');
    if (eq == std::string::npos) throw std::runtime_error("Unrecognized argument: " + arg);
    const std::string key = lowerAscii(arg.substr(0, eq)), value = arg.substr(eq + 1);
    if (key == "--stopmass") {
      rc.stopMass = number<double>(value);
      if (!(rc.stopMass >= 0.0 && rc.stopMass <= 2.0))
        throw std::runtime_error("--stopMass must lie in [0, 2] GeV (or omit the flag for default).");
    } else if (key == "--set") rc.pythiaSets.push_back(value);
    else if (key == "--parentspin") {
      rc.carriesParentSpin = named(PARENT_SPIN_NAMES, value, "--parentSpin");
      parentSpin = true;
    } else if (key == "--component") rc.comp = named(COMPONENT_NAMES, value, "--component");
    else if (key == "--flavor") {
      rc.qid = named(FLAVOR_NAMES, value, "flavor");
      rc.noFilter = true;
    } else if (key == "--injectmasses") rc.injectedMassMode = named(MASS_MODE_NAMES, value, "--injectMasses");
    else if (key == "--veto") rc.vetoRows.push_back(value);
    else if (key == "--inject") {
      std::stringstream ids(value);
      for (std::string id; std::getline(ids, id, ',');) rc.injectIds.push_back(number<int>(id));
    } else throw std::runtime_error("Unrecognized argument: " + arg);
  }
  if (!parentSpin)
    throw std::runtime_error("--parentSpin=carried|averaged is required: whether the spin-1 source "
                             "is the parent's spin or an average over its polarizations is not a "
                             "quantum number, so it is never inferred from one");
  return rc;
}

// ------------------------------------------------------- source quantum numbers

// Fixed-component projections with a G proxy, keyed on the component alone: a
// component that several sources realize carries the same projection in each.
// Every other component keeps the source baseline (all spin-0 components, and
// the heavy-light charged currents below).
struct ComponentProjection {
  VComponent comp;
  int i2, p, g;
};

static constexpr ComponentProjection G_PROJECTIONS[] = {
    {VComponent::RHO, 2, -1, +1},
    {VComponent::OMEGA, 0, -1, -1},
    {VComponent::PHI, 0, -1, -1},
    {VComponent::CCVEC, 2, -1, +1},
    {VComponent::CCAX, 2, +1, -1},
    {VComponent::NCAX, 2, +1, -1},
    {VComponent::NCSAX, 0, +1, +1}};

static SourceQN sourceQN(const RunConfig& rc, VComponent comp) {
  if (rc.particle.j2 == 2 && rc.decay == DecayMode::GG)
    throw std::runtime_error("spin 1 -> gg forbidden for an on-shell spin-1 source into two massless vectors (Landau–Yang).");
  SourceQN qn = rc.particle.baseline();
  qn.undetVeto = rc.undetVeto;
  for (const auto& row : G_PROJECTIONS) {
    if (row.comp != comp) continue;
    qn.allowedI2 = {row.i2};
    qn.expectedP = row.p;
    qn.enforceGProxy = true;
    qn.expectedG = row.g;
  }
  if (comp == VComponent::CCCD || comp == VComponent::CCCS) {
    // The heavy-light V-A current is neither a C nor a G eigenstate and has
    // both parities; isospin (c dbar I=1/2, I3=+1/2; c sbar I=0) stays exact.
    qn.enforceJPC2Body = false;
    qn.allowedI2 = {comp == VComponent::CCCD ? 1 : 0};
    qn.expectedI3_2 = comp == VComponent::CCCD ? +1 : 0;
  }
  return qn;
}

// The fixed source components each set of quantum numbers and decay mode realizes.
static bool supportedComponent(const RunConfig& rc, VComponent c) {
  const bool qq = rc.decay == DecayMode::QQ;
  // A spin-0 source fragments a gluon pair or one open-flavour string.
  if (rc.particle.j2 == 0)
    return c == VComponent::GLUE ? !qq : qq && (c == VComponent::STRANGE || c == VComponent::CHARM);
  if (rc.particle.charge != 0) return qq && chargedCurrent(c);
  // Neutral spin-1: the vector-meson components always, and the axial ones only
  // for a source that is not a C eigenstate (they are not C-odd).
  return qq && (c == VComponent::RHO || c == VComponent::OMEGA || c == VComponent::PHI
                || (!rc.particle.cGiven && (c == VComponent::NCAX || c == VComponent::NCSAX)));
}

// ------------------------------------------------------------- Pythia instances

enum class Stage { HADRONIZER, DECAYER, CHANNEL_DECAYER };

static void setStable(Pythia& p, Stage stage) {
  // Full decayer keeps pi+-, K+-, K_L, n, mu+-, pi0, K_S.  The channel
  // decayer decays only prompt strong/EM resonances, keeping the particles
  // that label exclusive channels (pi, K, eta, eta', N, mu, ground hyperons).
  static const std::vector<int> decayer = {211, -211, 321, -321, 130, 2112, -2112, 13, -13, 111, 310};
  static const std::vector<int> channel = {
      211, -211, 111, 321, -321, 311, -311, 130, 310, 221, 331,
      2212, -2212, 2112, -2112, 13, -13,
      3122, -3122, 3112, -3112, 3212, -3212, 3222, -3222,
      3312, -3312, 3322, -3322, 3334, -3334};
  for (int pid : stage == Stage::DECAYER ? decayer : channel)
    p.readString(std::to_string(pid) + ":mayDecay = off");
}

// Every event reseeds all three generators, so no Random:seed is set here.
// Like Pythia's own PythiaParallel, each instance copies one XML-read
// database and may be initialized on its own thread.
static std::unique_ptr<Pythia> makePythia(Stage stage, const RunConfig& rc) {
  static Pythia xml("../share/Pythia8/xmldoc", false);
  auto p = std::make_unique<Pythia>(xml.settings, xml.particleData, false);
  p->readString("ProcessLevel:all = off");
  p->readString("PartonLevel:all  = off");
  p->readString("HadronLevel:all  = on");
  p->readString("HadronLevel:mStringMin = 0.5");
  const bool hadronizer = stage == Stage::HADRONIZER;
  p->readString(hadronizer ? "HadronLevel:Hadronize = on" : "HadronLevel:Hadronize = off");
  p->readString(hadronizer ? "HadronLevel:Decay     = off" : "HadronLevel:Decay     = on");
  if (!hadronizer) setStable(*p, stage);
  if (rc.stopMass >= 0.0)
    p->readString("StringFragmentation:stopMass = " + std::to_string(rc.stopMass));
  for (const std::string& s : rc.pythiaSets)
    if (!p->readString(s)) throw std::runtime_error("Pythia rejected setting: " + s);
  // Match the complete-event validation tolerance; failures retry, never repair.
  p->readString("Check:event = on");
  p->readString("Check:epTolErr = 2e-6");
  for (const char* s : {"Print:quiet = on", "Init:showAllSettings = off",
                        "Init:showChangedSettings = off", "Init:showChangedParticleData = off",
                        "Init:showAllParticleData = off", "Next:numberShowInfo = 0",
                        "Next:numberShowProcess = 0", "Next:numberShowEvent = 0"})
    p->readString(s);
  if (!p->init()) throw std::runtime_error("Pythia initialization failed");
  if (!hadronizer) setStable(*p, stage);
  return p;
}

struct PythiaChain {
  std::unique_ptr<Pythia> had, dec, chn;
};

static PythiaChain makeChain(const RunConfig& rc) {
  auto dec = std::async(std::launch::async, makePythia, Stage::DECAYER, std::cref(rc));
  auto chn = std::async(std::launch::async, makePythia, Stage::CHANNEL_DECAYER, std::cref(rc));
  PythiaChain chain{makePythia(Stage::HADRONIZER, rc), dec.get(), chn.get()};
  // A 0^- source resolves K+- K_S pi-+ and K+- K_L pi-+ separately, so the
  // channel decayer evolves K0/K0bar (after init, which re-stabilized them);
  // the full decayer continues that exact event.
  if (rc.particle.j2 == 0 && rc.particle.parity == -1
      && (!chain.chn->readString("311:mayDecay = on") || !chain.chn->readString("-311:mayDecay = on")
          || !chain.chn->particleData.mayDecay(311) || !chain.chn->particleData.mayDecay(-311)))
    throw std::runtime_error("failed to enable the 0^- neutral-kaon resolution in channel classifier");
  return chain;
}

// Continue an event in the next stage.  The copy re-points BOTH each
// particle's ParticleDataEntry AND the event's ParticleData pointer to the
// DESTINATION instance; otherwise its per-instance mayDecay settings are
// silently ignored.
static bool continueEvent(Pythia& dst, const Pythia& src) {
  dst.event = src.event;
  dst.event.init("(copied event)", &dst.particleData);
  for (int i = 0; i < dst.event.size(); ++i) {
    dst.event[i].setPDEPtr(dst.particleData.findParticle(dst.event[i].id()));
  }
  return dst.forceHadronLevel();
}

static void seedChain(Pythia& had, Pythia& dec, Pythia& chn, int seed) {
  exhad_rng::initialize(had.rndm, seed);
  exhad_rng::initialize(dec.rndm, seed + DECAYER_SEED_OFFSET);
  exhad_rng::initialize(chn.rndm, seed + CHANNEL_DECAYER_SEED_OFFSET);
  had.event.reset();
  dec.event.reset();
  chn.event.reset();
}

// ------------------------------------------------------------- parton sources

constexpr long double PI = 3.141592653589793238462643383279502884L;

static long double sampleCosTheta(Pythia& pythia, PolarMode mode) {
  if (mode == PolarMode::ISOTROPIC) {
    return 2.0L * (long double) pythia.rndm.flat() - 1.0L;
  }
  // transverse vector: dGamma/dcos(theta) ~ (1 + cos^2(theta))
  while (true) {
    long double c = 2.0L * (long double) pythia.rndm.flat() - 1.0L;
    long double w = 1.0L + c*c;
    if (2.0L * (long double) pythia.rndm.flat() < w) return c;
  }
}

static void resetSystem(Pythia& pythia, double M) {
  pythia.event.reset();
  pythia.event[0].p(0.0, 0.0, 0.0, M);
  pythia.event[0].m(M);
  pythia.event[0].status(-11);
}

// Outgoing colour-singlet pair with total energy M at rest: g g (isotropic),
// a flavour-diagonal q qbar sharing E = M/2, or a charged-string q qbar'.
static void buildPair(Pythia& pythia, double M, int idQ, int idQbar, PolarMode polMode) {
  resetSystem(pythia, M);
  const bool glue = idQ == 21, diagonal = glue || idQbar == -idQ;
  const double m1 = glue ? 0.0 : pythia.particleData.m0(idQ);
  const double m2 = glue ? 0.0 : pythia.particleData.m0(std::abs(idQbar));
  if (m1 + m2 > M) throw std::runtime_error("Kinematically forbidden: M < m_q + m_q'.");
  const double E1 = diagonal ? 0.5 * M : (M * M + m1 * m1 - m2 * m2) / (2.0 * M);
  const double E2 = diagonal ? E1 : M - E1;
  const double p  = glue ? E1 : std::sqrt(std::max(0.0, E1 * E1 - m1 * m1));

  const long double c = sampleCosTheta(pythia, glue ? PolarMode::ISOTROPIC : polMode);
  const long double sn = std::sqrt(std::max((long double) 0.0, 1.0L - c * c));
  const long double phi = 2.0L * PI * (long double) pythia.rndm.flat();
  const double px = (double)(p * sn * std::cos((double) phi));
  const double py = (double)(p * sn * std::sin((double) phi));
  const double pz = (double)(p * c);

  const int i1 = pythia.event.append(idQ,    23, 0, 0, 0, 0, 501, glue ? 502 : 0,  px,  py,  pz, E1, m1);
  const int i2 = pythia.event.append(idQbar, 23, 0, 0, 0, 0, glue ? 502 : 0, 501, -px, -py, -pz, E2, m2);
  pythia.event[0].daughters(i1, i2);
}

// String-end quark; the u/d coin is drawn for every light two-ended source.
static int sourceQuark(VComponent comp, int qid, Pythia& had) {
  if (comp == VComponent::CHARM) return 4;
  if (comp == VComponent::PHI || comp == VComponent::STRANGE || comp == VComponent::NCSAX) return 3;
  if (comp == VComponent::NONE) return qid;
  return had.rndm.flat() < 0.5 ? 2 : 1;
}

// --------------------------------------------------------- exclusive injection

// Draw Pythia line shapes conditioned on joint kinematic closure: each range
// is clipped only by the other daughters' minimum masses; sum(m_i) >= M is
// rejected, with no pole-mass fallback.
static std::vector<double> sampleInjectedMasses(Pythia& p, double M, const std::vector<int>& ids) {
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                         * std::max(1.0, std::abs(M));
  std::vector<double> supportMin(ids.size()), upper(ids.size());
  double minimumSum = 0.0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (!p.particleData.findParticle(ids[i]))
      throw std::runtime_error("unknown injection daughter ID " + std::to_string(ids[i]));
    supportMin[i] = p.particleData.m0Min(ids[i]);
    upper[i] = p.particleData.m0Max(ids[i]);
    minimumSum += supportMin[i];
  }
  if (!(M > minimumSum + tolerance))
    throw std::runtime_error("finite-width injection is at/below joint threshold");
  for (std::size_t i = 0; i < ids.size(); ++i)
    upper[i] = std::min(upper[i], M - (minimumSum - supportMin[i]) - tolerance);

  for (int attempt = 0; attempt < 100000; ++attempt) {
    std::vector<double> masses(ids.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      // m0Min == m0Max for stable/narrow particles: no RNG is consumed.
      masses[i] = upper[i] > supportMin[i] + tolerance
          ? p.particleData.mSelInRange(ids[i], supportMin[i], upper[i]) : supportMin[i];
      sum += masses[i];
    }
    if (M - sum > tolerance) return masses;
  }
  throw std::runtime_error(
      "failed to draw a jointly kinematic injected resonance-mass set after 100000 attempts");
}

static double twoBodyP(double Mm, double ma, double mb) {
  const double s = Mm * Mm;
  const double a = (ma + mb) * (ma + mb), b = (ma - mb) * (ma - mb);
  return std::sqrt(std::max((s - a) * (s - b), 0.0)) / (2.0 * Mm);
}

struct InjectedFourVector {
  double px, py, pz, e;
};

// Boost q out of the rest frame of `frame`; below `rest` in beta^2 q is kept.
static InjectedFourVector boostInjected(const InjectedFourVector& q,
                                        const InjectedFourVector& frame, double rest = 1e-28) {
  const double bx = frame.px / frame.e;
  const double by = frame.py / frame.e;
  const double bz = frame.pz / frame.e;
  const double beta2 = bx * bx + by * by + bz * bz;
  if (beta2 < rest) return q;
  const double gamma = 1.0 / std::sqrt(std::max(1.0 - beta2, 1e-30));
  const double bq = bx * q.px + by * q.py + bz * q.pz;
  const double fac = (gamma - 1.0) * bq / beta2 + gamma * q.e;
  return {q.px + fac * bx, q.py + fac * by, q.pz + fac * bz,
          gamma * (q.e + bq)};
}

// Isotropic 2-body state.
static std::vector<InjectedFourVector> injectedTwoBody(Pythia& p, double M, double m1, double m2) {
  const double E1 = (M * M + m1 * m1 - m2 * m2) / (2.0 * M);
  const double E2 = M - E1;
  const double pa2 = std::max(E1 * E1 - m1 * m1, 0.0);
  const double pa = std::sqrt(pa2);

  const double c = 2.0 * p.rndm.flat() - 1.0;
  const double s = std::sqrt(std::max(1.0 - c * c, 0.0));
  const double phi = 2.0 * M_PI * p.rndm.flat();
  const double px = pa * s * std::cos(phi);
  const double py = pa * s * std::sin(phi);
  const double pz = pa * c;
  return {{px, py, pz, E1}, {-px, -py, -pz, E2}};
}

// Flat 3-body phase space: M -> id1 + X(m23), X -> id2 + id3, with m23 drawn
// by accept-reject on p*(M;m1,m23) p*(m23;m2,m3).
static std::vector<InjectedFourVector> injectedThreeBody(Pythia& p, double M,
                                                         double m1, double m2, double m3) {
  const double m23lo = m2 + m3, m23hi = M - m1;
  double wmax = 0.0;
  for (int k = 0; k <= 24; ++k) {
    const double mm = m23lo + (m23hi - m23lo) * (k / 24.0);
    const double w = twoBodyP(M, m1, mm) * twoBodyP(mm, m2, m3);
    if (w > wmax) wmax = w;
  }
  wmax *= 1.05;
  double m23 = m23lo;
  for (int tries = 0; tries < 1000; ++tries) {
    m23 = m23lo + (m23hi - m23lo) * p.rndm.flat();
    const double w = twoBodyP(M, m1, m23) * twoBodyP(m23, m2, m3);
    if (p.rndm.flat() * wmax <= w) break;
  }

  // step 1: M (at rest) -> id1 + X(m23), isotropic
  const double pa = twoBodyP(M, m1, m23);
  const double E1 = std::sqrt(pa * pa + m1 * m1);
  const double EX = M - E1;
  double c = 2.0 * p.rndm.flat() - 1.0, s = std::sqrt(std::max(1.0 - c * c, 0.0));
  double phi = 2.0 * M_PI * p.rndm.flat();
  const double ax = s * std::cos(phi), ay = s * std::sin(phi), az = c;
  const double p1x = pa * ax, p1y = pa * ay, p1z = pa * az;   // id1
  const InjectedFourVector recoil{-p1x, -p1y, -p1z, EX};       // X

  // step 2: X -> id2 + id3 in X rest frame, isotropic, then boost by X
  const double pb = twoBodyP(m23, m2, m3);
  const double e2 = std::sqrt(pb * pb + m2 * m2);
  const double e3 = std::sqrt(pb * pb + m3 * m3);
  c = 2.0 * p.rndm.flat() - 1.0;
  s = std::sqrt(std::max(1.0 - c * c, 0.0));
  phi = 2.0 * M_PI * p.rndm.flat();
  const double bx = pb * s * std::cos(phi), by = pb * s * std::sin(phi), bz = pb * c;
  return {{p1x, p1y, p1z, E1}, boostInjected({bx, by, bz, e2}, recoil, 1e-16),
          boostInjected({-bx, -by, -bz, e3}, recoil, 1e-16)};
}

// Raubold--Lynch N-body phase space (port of EventCalc NBodyDecay.py):
// 256 weighted candidates of intermediate masses, one resampled, then
// sequential isotropic splits.
static std::vector<InjectedFourVector> injectedNBody(Pythia& p, double M,
                                                     const std::vector<double>& masses) {
  const std::size_t n = masses.size();
  std::vector<double> cumulative(n);
  double threshold = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    threshold += masses[i];
    cumulative[i] = threshold;
  }
  if (M <= threshold + 1e-14)
    throw std::runtime_error("N-body injection is at/below threshold");
  const double kinetic = M - threshold;

  constexpr int nCandidates = 256;
  std::vector<std::vector<double>> candidates;
  std::vector<double> weights;
  candidates.reserve(nCandidates);
  weights.reserve(nCandidates);
  double weightSum = 0.0;
  for (int candidate = 0; candidate < nCandidates; ++candidate) {
    std::vector<double> random(n - 2);
    for (double& value : random) value = p.rndm.flat();
    std::sort(random.begin(), random.end());
    std::vector<double> intermediate(n);
    intermediate[0] = masses[0];
    for (std::size_t i = 1; i + 1 < n; ++i)
      intermediate[i] = cumulative[i] + kinetic * random[i - 1];
    intermediate[n - 1] = M;
    double weight = 1.0;
    for (std::size_t i = 1; i < n; ++i)
      weight *= twoBodyP(intermediate[i], masses[i], intermediate[i - 1]);
    if (!std::isfinite(weight) || weight <= 0.0) continue;
    candidates.push_back(std::move(intermediate));
    weights.push_back(weight);
    weightSum += weight;
  }
  if (candidates.empty() || !(weightSum > 0.0))
    throw std::runtime_error("failed to sample positive N-body phase space");
  double choice = p.rndm.flat() * weightSum;
  std::size_t selected = 0;
  while (selected + 1 < weights.size() && choice >= weights[selected]) {
    choice -= weights[selected];
    ++selected;
  }
  const std::vector<double>& intermediate = candidates[selected];

  std::vector<InjectedFourVector> momenta(n);
  InjectedFourVector subsystem{0.0, 0.0, 0.0, M};
  for (std::size_t index = n - 1; index > 0; --index) {
    const double parentMass = intermediate[index];
    const double residualMass = intermediate[index - 1];
    const double momentum = twoBodyP(parentMass, masses[index], residualMass);
    const double cosine = 2.0 * p.rndm.flat() - 1.0;
    const double sine = std::sqrt(std::max(1.0 - cosine * cosine, 0.0));
    const double phi = 2.0 * M_PI * p.rndm.flat();
    const double px = momentum * sine * std::cos(phi);
    const double py = momentum * sine * std::sin(phi);
    const double pz = momentum * cosine;
    const InjectedFourVector daughterRest{
        px, py, pz,
        std::sqrt(momentum * momentum + masses[index] * masses[index])};
    const InjectedFourVector residualRest{
        -px, -py, -pz,
        std::sqrt(momentum * momentum + residualMass * residualMass)};
    momenta[index] = boostInjected(daughterRest, subsystem);
    subsystem = boostInjected(residualRest, subsystem);
  }
  momenta[0] = subsystem;
  return momenta;
}

// Build the explicit daughter list (status 91) at rest.  Four-momentum
// closure <= 1e-9 is required for N >= 4 and for line shapes.
static void buildInjected(Pythia& p, double M, const std::vector<int>& ids, InjectedMassMode massMode) {
  std::vector<double> masses;
  if (massMode == InjectedMassMode::PYTHIA_WIDTH) masses = sampleInjectedMasses(p, M, ids);
  else for (int id : ids) masses.push_back(p.particleData.m0(id));
  const std::vector<InjectedFourVector> momenta = ids.size() == 2
      ? injectedTwoBody(p, M, masses[0], masses[1])
      : ids.size() == 3 ? injectedThreeBody(p, M, masses[0], masses[1], masses[2])
      : injectedNBody(p, M, masses);
  resetSystem(p, M);
  int first = -1, last = -1;
  double px = 0.0, py = 0.0, pz = 0.0, energy = 0.0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto& q = momenta[i];
    last = p.event.append(ids[i], 91, 0, 0, q.px, q.py, q.pz, q.e, masses[i]);
    if (first < 0) first = last;
    px += q.px;
    py += q.py;
    pz += q.pz;
    energy += q.e;
  }
  p.event[0].daughters(first, last);
  const double closure = std::max({std::abs(px), std::abs(py), std::abs(pz), std::abs(energy - M)});
  if ((ids.size() > 3 || massMode == InjectedMassMode::PYTHIA_WIDTH) && closure > 1e-9)
    throw std::runtime_error("injection violates four-momentum closure");
}

// ------------------------------------------------------------ event ancestry

// Ancestry roots: every final hadron of a string event; every explicit
// injection daughter, including a direct photon.
static std::vector<int> ancestryRoots(const Event& event, bool injected) {
  std::vector<int> indices;
  indices.reserve(64);
  for (int i = injected ? 1 : 0; i < event.size(); ++i) {
    const Particle& p = event[i];
    if (p.isFinal() && (injected ? p.id() != 0 : p.isHadron())) indices.push_back(i);
  }
  return indices;
}

// Exact Python ownership-cut key from the channel-decayer record, whose finals
// are the first canonical antichain.  A noncanonical terminal (lepton,
// hyperon) is unrepresented and never collides with an exclusive owner.
static bool ownershipRejected(const std::unordered_set<std::string>& vetoKeys, const Event& event) {
  if (vetoKeys.empty()) return false;
  portable_rejection::State counts;
  for (int i = 0; i < event.size(); ++i) {
    const Particle& particle = event[i];
    if (!particle.isFinal() || particle.id() == 0) continue;
    if (!portable_rejection::canonical(particle.id())) return false;
    ++counts[particle.id()];
  }
  return !counts.empty() && vetoKeys.count(portable_rejection::key(counts));
}

// Same mother-link forest and first-canonical cut as the Python graph parser.
// An expanded K0 is followed through its actual decay, never relabelled.
static bool matchedOwnershipCut(
    const Event& event, const std::vector<int>& roots,
    const std::set<int>& expand, portable_rejection::State& state) {
  std::vector<bool> included(event.size(), false), root(event.size(), false);
  std::vector<std::vector<int>> children(event.size());
  for (int id : roots) included.at(id) = root.at(id) = true;
  for (int i = 0; i < event.size(); ++i) {
    if (root[i]) continue;
    int parent = -1;
    for (int candidate : {event[i].mother1(), event[i].mother2()}) {
      if (candidate <= 0 || candidate >= event.size() || !included[candidate]) continue;
      if (parent >= 0 && parent != candidate) throw std::runtime_error("multiple hadronic parents");
      parent = candidate;
    }
    if (parent >= 0) {
      included[i] = true;
      children[parent].push_back(i);
    }
  }
  for (int i = 0; i < event.size(); ++i)
    if (event[i].isFinal() && event[i].id() && !included[i])
      throw std::runtime_error("terminal outside matched hadronic forest");
  std::vector<int> pending = roots;
  int visits = 0;
  while (!pending.empty()) {
    const int index = pending.back();
    pending.pop_back();
    if (++visits > event.size()) throw std::runtime_error("cyclic matched forest");
    const int id = event[index].id();
    if (portable_rejection::canonical(id) && !expand.count(id)) ++state[id];
    else {
      if (children[index].empty()) return false;
      pending.insert(pending.end(), children[index].begin(), children[index].end());
    }
  }
  return !state.empty();
}

// Lossless graph record: raw Pythia indices and mother links, with the
// post-hadronization primaries marked for the Python ancestry reduction.
static void writeEventGraph(const Event& event, const std::vector<int>& primaryIndices, double mass) {
  const std::unordered_set<int> primaries(primaryIndices.begin(), primaryIndices.end());
  std::cout << std::setprecision(17) << "E 1 " << mass << "\n";
  for (int i = 0; i < event.size(); ++i) {
    const Particle& p = event[i];
    std::cout << "N " << i << " " << p.id() << " " << p.status() << " "
              << p.mother1() << " " << p.mother2() << " "
              << p.daughter1() << " " << p.daughter2() << " "
              << (p.isFinal() ? 1 : 0) << " "
              << (primaries.count(i) ? 1 : 0) << " "
              << p.px() << " " << p.py() << " " << p.pz() << " "
              << p.e() << " " << p.m() << "\n";
  }
  std::cout << "X\n";
}

// ------------------------------------------------------------ event pipeline

enum class Trial { ACCEPTED, HADRONIZATION_FAILED, CHARM_OWNER, VETOED, CHANNEL_FAILED };

// A final baryon below the lightest baryon pair, 2 m_p (less 64 ulp), is an
// invariant violation (a wrong effective energy); above it the source filter
// and ownership decide.
static bool baryonBelowPairThreshold(const Event& event, double mass, const ParticleData& particleData) {
  if (!(mass < 2.0 * particleData.m0(2212) * (1.0 - 64.0 * std::numeric_limits<double>::epsilon()))) return false;
  for (int i = 0; i < event.size(); ++i)
    if (event[i].isFinal() && event[i].isHadron() && particleData.isBaryon(event[i].id())) return true;
  return false;
}

// One string trial: partons -> hadronizer -> ownership/source filters ->
// prompt decays continued in the channel decayer.
static Trial stringTrial(const RunConfig& rc, VComponent comp, int qid, double mass,
    PolarMode polar, Pythia& had, Pythia& chn, const DiscreteSymmetryFilter& filter,
    std::vector<int>& roots) {
  if (comp == VComponent::CCVEC || comp == VComponent::CCAX) buildPair(had, mass, 1, -2, polar);
  else if (comp == VComponent::CCCD) buildPair(had, mass, 4, -1, polar);
  else if (comp == VComponent::CCCS) buildPair(had, mass, 4, -3, polar);
  else if (comp == VComponent::GLUE || rc.decay == DecayMode::GG) buildPair(had, mass, 21, 21, polar);
  else buildPair(had, mass, qid, -qid, polar);
  if (!had.forceHadronLevel()) return Trial::HADRONIZATION_FAILED;
  if (baryonBelowPairThreshold(had.event, mass, had.particleData))
    throw std::runtime_error("worker produced a baryon below the pair threshold");
  roots = ancestryRoots(had.event, false);
  // EventCalc owns HNL -> l D and l Ds explicitly: a single D+- (cccd) or
  // Ds+- (cccs) root is vetoed before either decay stage.
  if (roots.size() == 1) {
    const int absoluteId = std::abs(had.event[roots.front()].id());
    if ((comp == VComponent::CCCD && absoluteId == 411) || (comp == VComponent::CCCS && absoluteId == 431))
      return Trial::CHARM_OWNER;
  }
  if (!rc.noFilter && filter.veto(had.event, had.rndm)) return Trial::VETOED;
  return continueEvent(chn, had) ? Trial::ACCEPTED : Trial::CHANNEL_FAILED;
}

enum class WorkerEventStatus { EVENT, EXHAUSTED, OWNERSHIP_VETO };

// Exactly one accepted graph for one independently seeded logical request.
static WorkerEventStatus generateWorkerEvent(
    const RunConfig& rc, double mass, PolarMode polar,
    Pythia& had, Pythia& dec, Pythia& chn,
    const DiscreteSymmetryFilter& componentFilter,
    const std::unordered_set<std::string>& vetoKeys,
    int baseSeed, std::vector<int>& acceptedPrimaryIndices) {
  seedChain(had, dec, chn, baseSeed);
  constexpr int maximumAttempts = 30;
  for (int attempt = 0; attempt < maximumAttempts; ++attempt) {
    const int qid = sourceQuark(rc.comp, rc.qid, had);
    // A partonic seed below its constituent threshold is a rejected attempt.
    if (!chargedCurrent(rc.comp) && rc.decay == DecayMode::QQ
        && 2.0 * had.particleData.m0(qid) > mass) continue;
    const Trial trial = stringTrial(rc, rc.comp, qid, mass, polar, had, chn,
                                    componentFilter, acceptedPrimaryIndices);
    if (trial == Trial::CHARM_OWNER) return WorkerEventStatus::OWNERSHIP_VETO;
    if (trial != Trial::ACCEPTED) continue;
    // Reported only after full-decayer success, so a failed decay advances
    // the internal attempt exactly as for an accepted event.
    if (!continueEvent(dec, chn)) continue;
    return ownershipRejected(vetoKeys, chn.event) ? WorkerEventStatus::OWNERSHIP_VETO : WorkerEventStatus::EVENT;
  }
  return WorkerEventStatus::EXHAUSTED;
}

// Matched weight w_F c_F,key of a decayed record: 0 after an ownership-cut failure
// or for a removed family; false (no weight) for a removed external final-state key.
static bool matchedWeight(const Event& event, const std::vector<int>& roots,
    const portable_rejection::Filter& matching, double& weight) {
  weight = 0.;
  portable_rejection::State state;
  if (!matchedOwnershipCut(event, roots, {}, state)) return true;
  // Exact external keys are provider vetoes with subattempt-local seeds;
  // family exclusions are outer matching rejections instead.
  if (matching.removedKeys.count(portable_rejection::key(state))) return false;
  const std::string family = matching.family(state);
  if (family.empty()) return true;
  portable_rejection::State resolved = state;
  if (matching.conditional.count(family) && !matching.expand.empty()) {
    resolved.clear();
    if (!matchedOwnershipCut(event, roots, matching.expand, resolved))
      throw std::runtime_error("unresolved expanded matched ownership");
  }
  weight = matching.weight(family, resolved);
  return true;
}

static bool spacelikeTerminal(const Event& event) {
  for (int i = 0; i < event.size(); ++i) {
    const Particle& p = event[i];
    if (p.isFinal() && p.id() && p.e()*p.e() - p.px()*p.px() - p.py()*p.py() - p.pz()*p.pz() < -1.e-8)
      return true;
  }
  return false;
}

// Compiled matched rejection (portable Model 1): sub-attempt seeds, removed
// keys retried, family weight times conditional factor over the envelope.
static bool generateMatchedProposal(
    const RunConfig& rc, double mass, PolarMode polar,
    Pythia& had, Pythia& dec, Pythia& chn,
    const DiscreteSymmetryFilter& componentFilter,
    const std::unordered_set<std::string>& vetoKeys,
    const portable_rejection::Filter& matching,
    std::uint64_t logical, double uniform, int attempts,
    std::vector<int>& primaryIndices) {
  for (int attempt = 0; attempt < attempts; ++attempt) {
    const std::uint64_t local = attempt == 0 ? logical : portable_rejection::splitmix(
        logical ^ portable_rejection::splitmix(attempt));
    const int seed = static_cast<int>(local % MAX_WORKER_BASE_SEED) + 1;
    if (generateWorkerEvent(rc, mass, polar, had, dec, chn, componentFilter,
          vetoKeys, seed, primaryIndices) != WorkerEventStatus::EVENT) continue;
    double weight = 0.;
    if (!matchedWeight(dec.event, primaryIndices, matching, weight)) continue;
    return weight > 0. && uniform < weight / matching.envelope && !spacelikeTerminal(dec.event);
  }
  throw std::runtime_error("matched provider exhausted conditioning budget");
}

// ---------------------------------------------------------- persistent worker

static int runPersistentEventWorker(const RunConfig& rc) {
  const bool injectionOnly = !rc.injectIds.empty();
  const bool charmContinuum = !injectionOnly && cOddNeutralVector(rc.particle)
      && rc.decay == DecayMode::QQ && rc.comp == VComponent::NONE && rc.qid == 4 && rc.noFilter;
  const bool projectedReference = supportedComponent(rc, rc.comp);
  if (injectionOnly ? !cOddNeutralVector(rc.particle) || rc.decay != DecayMode::QQ
                          || rc.comp != VComponent::NONE || rc.noFilter || rc.injectIds.size() < 2
                    : !charmContinuum && !projectedReference)
    throw std::runtime_error("event worker requires one fixed supported source component, a 1^-- qq "
                             "component=none injection of two or more daughters, or the charm continuum");
  if (!rc.vetoRows.empty() && !projectedReference)
    throw std::runtime_error("ownership veto is supported only for a projected reference worker");
  // Strict: a malformed, noncanonical or duplicate key would silently admit
  // exclusive-owner states twice.
  std::unordered_set<std::string> vetoKeys;
  for (const std::string& row : rc.vetoRows) {
    std::istringstream fields(row);
    portable_rejection::State counts;
    for (std::string token; fields >> token;) {
      const auto colon = token.find(':');
      if (colon == std::string::npos) throw std::runtime_error("ownership veto contains a malformed token");
      const int id = std::stoi(token.substr(0, colon)), count = std::stoi(token.substr(colon + 1));
      if (!portable_rejection::canonical(id) || count <= 0 || !counts.emplace(id, count).second)
        throw std::runtime_error("ownership veto contains a noncanonical state");
    }
    if (counts.empty() || portable_rejection::key(counts) != row || !vetoKeys.insert(row).second)
      throw std::runtime_error("ownership veto contains a duplicate or noncanonical row");
  }
  const PythiaChain chain = makeChain(rc);
  Pythia& had = *chain.had;
  Pythia& dec = *chain.dec;
  Pythia& chn = *chain.chn;
  const PolarMode polar = polarMode(rc);
  const double version = had.settings.parm("Pythia:versionNumber");
  const double stringMin = had.settings.parm("HadronLevel:mStringMin");
  if (std::abs(version - 8.317) > 1e-12 || std::abs(stringMin - 1.0) > 1e-12)
    throw std::runtime_error("event worker requires Pythia 8.317 and HadronLevel:mStringMin=1.0");
  const DiscreteSymmetryFilter componentFilter(had.particleData, sourceQN(rc, rc.comp));

  std::cout << std::setprecision(17)
            << "READY " << EVENT_WORKER_SCHEMA << " " << version << " "
            << componentName(rc.comp) << " " << stringMin
            << "\n" << std::flush;

  std::string line;
  portable_rejection::Filter matching;
  while (std::getline(std::cin, line)) {
    if (line == "QUIT") {
      std::cout << "BYE\n" << std::flush;
      return 0;
    }
    std::istringstream request(line);
    std::string verb, extra;
    request >> verb;
    if (verb == "FILTER") {
      int count = 0;
      if (!(request >> count) || count < 1 || count > 10000 || (request >> extra)
          || injectionOnly || !projectedReference || rc.noFilter)
        throw std::runtime_error("invalid matching filter request");
      matching.read(std::cin, count);
      std::cout << "FILTER-READY\n" << std::flush;
      continue;
    }
    if (verb == "BATCH") {
      std::uint64_t serial = 0;
      int count = 0, attempts = 0;
      std::string massText;
      if (!(request >> serial >> count >> attempts >> massText) || (request >> extra)
          || count < 1 || count > 4096 || attempts < 1 || attempts > 1024
          || matching.weights.empty()) throw std::runtime_error("invalid matched batch");
      const double mass = positiveMass(massText);
      std::vector<std::pair<std::uint64_t, double>> proposals;
      for (int i = 0; i < count; ++i) {
        std::string item;
        if (!std::getline(std::cin, item)) throw std::runtime_error("truncated matched batch");
        std::istringstream row(item);
        std::uint64_t logical;
        double uniform;
        if (!(row >> logical >> uniform) || (row >> extra) || !std::isfinite(uniform)
            || uniform < 0. || uniform > 1.) throw std::runtime_error("invalid batch proposal");
        proposals.emplace_back(logical, uniform);
      }
      int selected = -1;
      std::string failure;
      std::vector<int> roots;
      for (int i = 0; i < count; ++i) {
        try {
          if (generateMatchedProposal(rc, mass, polar, had, dec, chn, componentFilter,
                vetoKeys, matching, proposals[i].first, proposals[i].second,
                attempts, roots)) { selected = i; break; }
        } catch (const std::exception& error) {
          // Another source may accept an earlier logical attempt. Defer this
          // error until Python merges the source-local first results in order.
          selected = i;
          failure = error.what();
          std::replace(failure.begin(), failure.end(), '\n', ' ');
          break;
        }
      }
      std::cout << (failure.empty() ? "BATCHRESULT " : "BATCHERROR ") << serial << ' ' << selected << ' '
                << (selected < 0 ? 0 : proposals[selected].first) << '\n';
      if (!failure.empty()) std::cout << failure << '\n';
      else if (selected >= 0) writeEventGraph(dec.event, roots, mass);
      std::cout << "END " << serial << '\n' << std::flush;
      continue;
    }
    std::uint64_t serial = 0;
    long long seed = 0;
    std::string massText;
    if (verb != "REQUEST" || !(request >> serial >> seed >> massText) || (request >> extra)) {
      std::cout << "FATAL malformed-request\n" << std::flush;
      return 2;
    }
    const double mass = positiveMass(massText);
    if (seed < 1 || seed > MAX_WORKER_BASE_SEED) {
      std::cout << "RESULT " << serial << " ERROR " << seed << " invalid-seed\n" << std::flush;
      continue;
    }
    std::vector<int> roots;
    WorkerEventStatus status = WorkerEventStatus::EXHAUSTED;
    if (injectionOnly) {
      // One flat() draw precedes the injected kinematics (part of the seeded stream); a failed decay is EXHAUSTED.
      seedChain(had, dec, chn, static_cast<int>(seed));
      had.rndm.flat();
      buildInjected(had, mass, rc.injectIds, rc.injectedMassMode);
      roots = ancestryRoots(had.event, true);
      if (continueEvent(chn, had) && continueEvent(dec, chn)) status = WorkerEventStatus::EVENT;
    } else {
      status = generateWorkerEvent(rc, mass, polar, had, dec, chn, componentFilter,
                                   vetoKeys, static_cast<int>(seed), roots);
    }
    const bool event = status == WorkerEventStatus::EVENT;
    std::cout << "RESULT " << serial << (event ? " EVENT " : " EXHAUSTED ") << seed << "\n";
    if (event) writeEventGraph(dec.event, roots, mass);
    std::cout << "END " << serial << "\n" << std::flush;
  }
  return 0;
}

// --------------------------------------------------------- dedicated HNL runs

// One dedicated run's tables, each row "<value> <entries>...": injection
// "<probability> <id> <id> ..." (the rest of unit probability is the string);
// morph "<weight> <id>:<count> ...", also keyed by the charge conjugate, where a
// bare "<weight>" row covers every other state (default 1).  Every value is
// finite and nonnegative, every ID and count a whole integer token.
struct RunTables {
  struct Mode {
    std::vector<int> ids;
    double prob;
  };
  std::vector<Mode> modes;
  double pTot = 0.0, other = 1.0, wMax = 1.0;
  std::unordered_map<std::string, double> weights;
  bool morph;

  RunTables(std::istream& in, std::size_t nModes, std::size_t nMorph) : morph(nMorph > 0) {
    std::string line;
    for (std::size_t i = 0; i < nModes + nMorph && std::getline(in, line); ++i) {
      std::istringstream row(line);
      double value = 0.0;
      if (!(row >> value) || !(std::isfinite(value) && value >= 0.0))
        throw std::runtime_error("run table value must be finite and nonnegative: " + line);
      if (i < nModes) {
        Mode mode{{}, value};
        for (std::string token; row >> token;) mode.ids.push_back(number<int>(token));
        if (mode.ids.size() < 2) throw std::runtime_error("injection mode needs two or more daughters: " + line);
        if (value == 0.0) continue;
        modes.push_back(std::move(mode));
        pTot += value;
        continue;
      }
      portable_rejection::State ids, conjugate;
      for (std::string token; row >> token;) {
        const auto colon = token.find(':');
        const int id = number<int>(token.substr(0, colon)), count = number<int>(token.substr(colon + 1));
        if (colon == std::string::npos || count < 1) throw std::runtime_error("morph state needs <id>:<count> tokens: " + line);
        const bool selfConjugate = id == 22 || id == 111 || id == 221 || id == 331 || id == 113
            || id == 223 || id == 333 || id == 130 || id == 310 || id == 9010221 || id == 9000111;
        ids[id] += count;
        conjugate[selfConjugate ? id : -id] += count;
      }
      wMax = std::max(wMax, value);
      if (ids.empty()) other = value;
      else weights[portable_rejection::key(ids)] = weights[portable_rejection::key(conjugate)] = value;
    }
    if (pTot > 1.0 + 1e-9) throw std::runtime_error("inject probabilities sum to > 1");
  }

  // A mode, or the string branch (nullptr) when the flat lands beyond pTot.
  const Mode* draw(Rndm& rndm) const {
    double x = rndm.flat();
    if (x >= pTot) return nullptr;
    for (const auto& m : modes) {
      if (x < m.prob) return &m;
      x -= m.prob;
    }
    return &modes.back();
  }

  double weight(const Event& event) const {
    portable_rejection::State counts;
    for (int i = 0; i < event.size(); ++i)
      if (event[i].isFinal()) ++counts[event[i].id()];
    const auto found = weights.find(portable_rejection::key(counts));
    return found == weights.end() ? other : found->second;
  }
};

// N accepted events, streamed as "E" plus "id px py pz e" final-state rows,
// then "END <accepted>".  Injection is decided once per accepted slot; a filter
// veto or hadronization failure retries the same component, a morph rejection,
// decay failure or charm-owner veto redraws it (one flat over the weights).
static void dedicatedRun(const RunConfig& rc, double M, Pythia& had, Pythia& dec, Pythia& chn, int nAccepted,
    const std::vector<double>& weights, const RunTables& tables) {
  const PolarMode polar = polarMode(rc);
  const std::vector<VComponent> components = rc.particle.charge != 0
      ? std::vector<VComponent>{VComponent::CCVEC, VComponent::CCAX}
      : std::vector<VComponent>{VComponent::RHO, VComponent::OMEGA, VComponent::PHI, VComponent::NCAX, VComponent::NCSAX};
  if (weights.size() != components.size()) throw std::runtime_error("one weight per source component");
  double total = 0.0;
  for (double weight : weights) total += weight;
  if (!(total > 0.0)) throw std::runtime_error("component weights sum to zero");
  std::map<VComponent, DiscreteSymmetryFilter> filters;
  for (VComponent c : components) filters.emplace(c, DiscreteSymmetryFilter(had.particleData, sourceQN(rc, c)));

  VComponent comp = VComponent::NONE;
  bool redrawComp = true, freshEvent = true;
  int nAcc = 0;
  long long nAtt = 0, nHadFail = 0;
  const long long maxAttempts = (tables.morph ? 3000LL : 30LL) * nAccepted;
  std::cout << std::setprecision(17);
  while (nAcc < nAccepted && nAtt < maxAttempts) {
    ++nAtt;
    const RunTables::Mode* mode = nullptr;
    if (freshEvent && !tables.modes.empty()) {
      mode = tables.draw(had.rndm);
      if (mode == nullptr) freshEvent = false;  // this slot is a string event
    }
    if (mode != nullptr) {
      buildInjected(had, M, mode->ids, InjectedMassMode::POLE);
      if (!continueEvent(chn, had)) continue;
    } else {
      if (redrawComp) {
        double x = had.rndm.flat() * total;
        comp = components.back();
        for (std::size_t k = 0; k + 1 < components.size(); ++k)
          if ((x -= weights[k]) < 0.0) {
            comp = components[k];
            break;
          }
      }
      redrawComp = false;
      std::vector<int> roots;
      const int qid = chargedCurrent(comp) ? 1 : sourceQuark(comp, rc.qid, had);
      const Trial trial = stringTrial(rc, comp, qid, M, polar, had, chn, filters.at(comp), roots);
      if (trial == Trial::HADRONIZATION_FAILED) {
        // A pathological parameter point stops early with what was accepted.
        if (++nHadFail > 100 && nHadFail > nAtt / 2) break;
        continue;
      }
      if (trial == Trial::VETOED) continue;
      redrawComp = trial != Trial::ACCEPTED;
      if (trial == Trial::CHANNEL_FAILED) ++nHadFail;
      if (redrawComp) continue;
      if (tables.morph && had.rndm.flat() * tables.wMax > tables.weight(chn.event)) {
        redrawComp = true;
        continue;
      }
    }
    // Continue the exact prompt-decay realization that was classified.
    if (!continueEvent(dec, chn)) {
      if (mode == nullptr) {
        ++nHadFail;
        redrawComp = true;
      }
      continue;
    }
    if (mode == nullptr) {
      redrawComp = true;
      freshEvent = true;
    }
    std::cout << "E\n";
    for (int i = 0; i < dec.event.size(); ++i) {
      const Particle& p = dec.event[i];
      if (p.isFinal()) std::cout << p.id() << " " << p.px() << " " << p.py() << " " << p.pz() << " " << p.e() << "\n";
    }
    ++nAcc;
  }
  std::cout << "END " << nAcc << "\n" << std::flush;
}

// Run settings may set only string-fragmentation parameters (StringFlav, StringZ,
// StringPT, StringFragmentation namespaces).  A decay-only Pythia 8.317 instance
// reads them solely when a particle decays through a channel with partons among
// its products (hadron flavours from StringFlav) or through onium -> gamma g g
// (stopMass).  String hadrons of these light currents never reach such channels
// and injection daughters that could are rejected, so both decayers are built
// once without the run settings, and only hadronizers are kept per settings list.
static bool fragmentationSetting(const std::string& setting) {
  const std::string key = lowerAscii(setting.substr(0, setting.find('=')));
  for (const char* prefix : {"stringflav:", "stringz:", "stringpt:", "stringfragmentation:"})
    if (key.rfind(prefix, 0) == 0) return true;
  return false;
}

// Absolute IDs whose decay chains can reach such a channel (all channels, on or off).
static std::set<int> partonicDecaySpecies(ParticleData& particleData) {
  std::set<int> species;
  for (std::size_t previous = std::size_t(-1); previous != species.size();) {
    previous = species.size();
    for (const auto& entry : particleData)
      for (int i = 0; i < entry.second->sizeChannels(); ++i) {
        const DecayChannel& channel = entry.second->channel(i);
        bool partonic = channel.meMode() == 92;
        for (int k = 0; k < channel.multiplicity(); ++k) {
          const int id = std::abs(channel.product(k));
          partonic = partonic || id < 10 || id == 21 || (id > 1000 && id < 10000 && (id / 10) % 10 == 0)
              || (id > 80 && id < 84) || species.count(id);
        }
        if (partonic) species.insert(entry.first);
      }
  }
  return species;
}

// "RUN <2J> <P> <C|none> <charge> <parentSpin> <N> <seed> <mass> <w1,w2,...>
// <settings> <modes> <morph>", then that many Pythia-setting, injection and
// morph rows.  The 64 most recent hadronizers are kept, and every instance is
// reseeded for every run.
static int serveDedicatedRuns() {
  const RunConfig decayerConfig;  // the decayers initialize alongside the first hadronizer
  auto decayer = std::async(std::launch::async, makePythia, Stage::DECAYER, std::cref(decayerConfig));
  auto channel = std::async(std::launch::async, makePythia, Stage::CHANNEL_DECAYER, std::cref(decayerConfig));
  std::unique_ptr<Pythia> dec, chn;
  std::set<int> partonic;
  std::list<std::pair<std::vector<std::string>, std::unique_ptr<Pythia>>> hadronizers;  // most recent first
  for (std::string line; std::getline(std::cin, line) && line != "QUIT";) {
    std::istringstream header(line);
    std::string verb, j2, parity, cParity, charge, parentSpin, massText, weightText, extra;
    int nAccepted = 0, seed = 0;
    std::size_t nSets = 0, nModes = 0, nMorph = 0;
    if (!(header >> verb >> j2 >> parity >> cParity >> charge >> parentSpin >> nAccepted >> seed
                 >> massText >> weightText >> nSets >> nModes >> nMorph)
        || verb != "RUN" || (header >> extra) || nAccepted < 1 || seed < 1 || seed > MAX_WORKER_BASE_SEED)
      throw std::runtime_error("malformed dedicated run request");
    RunConfig rc;
    rc.particle = DecayingParticle::read(j2, parity, cParity, charge);
    rc.carriesParentSpin = named(PARENT_SPIN_NAMES, parentSpin, "parent spin");
    const double mass = positiveMass(massText);
    if (rc.particle.j2 != 2 || rc.particle.parity != -1 || rc.particle.cGiven || rc.carriesParentSpin)
      throw std::runtime_error("dedicated runs require a 1^- current that is not a C eigenstate and "
                               "averages the parent polarization");
    for (std::size_t i = 0; i < nSets && std::getline(std::cin, line); ++i) {
      if (!fragmentationSetting(line)) throw std::runtime_error("dedicated runs set only string fragmentation: " + line);
      rc.pythiaSets.push_back(line);
    }
    const RunTables tables(std::cin, nModes, nMorph);
    if (!std::cin) throw std::runtime_error("truncated dedicated run request");
    std::vector<double> weights;
    std::stringstream weightList(weightText);
    for (std::string token; std::getline(weightList, token, ',');) {
      weights.push_back(number<double>(token));
      if (!(weights.back() >= 0.0)) throw std::runtime_error("component weights must be nonnegative: " + weightText);
    }

    auto found = std::find_if(hadronizers.begin(), hadronizers.end(),
                              [&](const auto& entry) {
                                return entry.first == rc.pythiaSets;
                              });
    if (found == hadronizers.end()) {
      hadronizers.emplace_front(rc.pythiaSets, makePythia(Stage::HADRONIZER, rc));
      if (hadronizers.size() > 64) hadronizers.pop_back();
    } else hadronizers.splice(hadronizers.begin(), hadronizers, found);
    if (!dec) {
      dec = decayer.get();
      chn = channel.get();
      partonic = partonicDecaySpecies(dec->particleData);
    }
    Pythia& had = *hadronizers.front().second;
    for (const auto& mode : tables.modes)
      for (int id : mode.ids)
        if (!had.particleData.isParticle(id) || partonic.count(std::abs(id)))
          throw std::runtime_error("unknown or partonically decaying injection ID " + std::to_string(id));
    seedChain(had, *dec, *chn, seed);
    dedicatedRun(rc, mass, had, *dec, *chn, nAccepted, weights, tables);
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--runs") return serveDedicatedRuns();
    return runPersistentEventWorker(parseArgs(argc, argv));
  } catch (const std::exception& e) {
    std::cout << "Fatal: " << e.what() << "\n" << std::flush;
    return 2;
  }
}
