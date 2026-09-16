// The stochastic isospin filter accepts a primary state with the exact total-
// isospin projection P(I | state) (tests/isospin_projector_fixture.tsv, exact
// fractions), independently of the event-record order.  Isospin-only source:
// the C, G and two-body JPC stages are off.  A fixed-value random engine
// reads the acceptance boundary of veto() directly.
#include "Pythia8/Pythia.h"
#include "symmetry_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace Pythia8;

namespace {
struct Fixed : RndmEngine {
  double value = 0.;
  long calls = 0;
  double flat() override { ++calls; return value; }
};

double bitsToDouble(std::uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
std::uint64_t doubleToBits(double d) { std::uint64_t b; std::memcpy(&b, &d, 8); return b; }
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Pythia pythia(argv[1], false);
  pythia.readString("ProcessLevel:all = off");
  pythia.readString("Print:quiet = on");
  if (!pythia.init()) return 2;
  auto engine = std::make_shared<Fixed>();
  Rndm rndm(1);
  rndm.rndmEnginePtr(engine);
  const std::map<std::string, int> name = {
    {"pi+", 211}, {"pi0", 111}, {"pi-", -211}, {"rho+", 213}, {"rho0", 113}, {"rho-", -213},
    {"K+", 321}, {"K0", 311}, {"K-", -321}, {"K0bar", -311}, {"eta", 221}, {"eta'", 331},
    {"omega", 223}, {"p", 2212}, {"n", 2112}, {"pbar", -2212}, {"nbar", -2112},
    {"Delta++", 2224}, {"Delta-", 1114}};

  Event event;
  event.init("(synthetic)", &pythia.particleData);
  auto fill = [&](const std::vector<int>& ids) {
    event.reset();
    for (int id : ids) {
      const double m = pythia.particleData.m0(id);
      event.append(id, 1, 0, 0, 0., 0., 0., m, m);
    }
  };
  auto accepts = [&](const DiscreteSymmetryFilter& filter, double u) {
    engine->value = u;
    return !filter.veto(event, rndm);
  };
  // Largest double u that veto() accepts (it accepts iff u <= pAcc).
  auto boundary = [&](const DiscreteSymmetryFilter& filter) {
    if (accepts(filter, 1.0)) return 1.0;
    std::uint64_t lo = 0, hi = doubleToBits(1.0);
    while (hi - lo > 1) {
      const std::uint64_t mid = lo + (hi - lo) / 2;
      (accepts(filter, bitsToDouble(mid)) ? lo : hi) = mid;
    }
    return bitsToDouble(lo);
  };

  std::ifstream in(argv[2]);
  std::string line;
  int rows = 0, failures = 0;
  long orders = 0;
  std::map<std::string, double> spot;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> column;
    std::stringstream cells(line);
    for (std::string cell; std::getline(cells, cell, '\t');) column.push_back(cell);
    std::vector<int> ids;
    std::stringstream names(column.at(0));
    for (std::string item; names >> item;) ids.push_back(name.at(item));
    const int i2 = std::stoi(column.at(1)), i3 = std::stoi(column.at(2));
    const auto slash = column.at(3).find('/');
    const long double exact = std::stold(column[3].substr(0, slash)) / std::stold(column[3].substr(slash + 1));

    SourceQN q;
    q.enforceCDet = false; q.enforceGProxy = false; q.enforceJPC2Body = false; q.undetVeto = true;
    q.allowedI2 = {i2}; q.expectedI3_2 = i3;
    const DiscreteSymmetryFilter filter(pythia.particleData, q);

    std::sort(ids.begin(), ids.end());
    double first = -1.;
    bool ok = true;
    long n = 0;
    do {  // every distinct record order
      fill(ids);
      engine->calls = 0;
      const bool acceptsAtZero = accepts(filter, 0.0);
      const long draws = engine->calls;
      const bool supported = filter.supports(event);
      ok = ok && acceptsAtZero == supported && supported == (exact > 0);
      ok = ok && draws == (exact > 0 ? 1 : 0);   // exact zeros are vetoed without a draw
      if (exact > 0) {
        const double a = boundary(filter);
        if (n == 0) first = a;
        ok = ok && a == first && std::fabs((long double)a - exact) <= 1e-12L;
      }
      ++n;
    } while (std::next_permutation(ids.begin(), ids.end()));
    orders += n;
    ++rows;
    if (!ok) {
      ++failures;
      std::cout << "FAIL " << column[0] << " 2I=" << i2 << " exact=" << column[3] << " boundary=" << first << '\n';
    }
    spot[column[0] + "|" + column[1]] = first;
  }
  // Exact isospin spot values (fixture rows are written in their table order).
  const std::vector<std::pair<std::string, double>> expected = {
    {"pi- pi0 pi0|2", 0.4}, {"pi- pi0 pi0 pi+|0", 2. / 15.}, {"pi0 pi0 pi0 pi0|0", 0.2}};
  for (const auto& [key, value] : expected) {
    const auto it = spot.find(key);
    if (it == spot.end() || std::fabs(it->second - value) > 1e-12) {
      ++failures;
      std::cout << "FAIL spot " << key << '\n';
    }
  }
  std::cout << rows << " projector rows, " << orders << " record orders, " << failures << " failures\n";
  return failures || rows != 427 ? 1 : 0;
}
