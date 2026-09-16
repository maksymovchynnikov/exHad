#pragma once

// Execution-only mirror of exhad/model1/families.py classify_channel on canonical
// ownership PDGs. Rates, source support and charge weights arrive from Python's
// prepared query point; this header owns no model probabilities.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace portable_rejection {
using State = std::map<int, int>;
inline bool pion(int id) {
  return id == 111 || std::abs(id) == 211;
}

inline bool kaon(int id) {
  return std::abs(id) == 321 || std::abs(id) == 311 || id == 130 || id == 310;
}

inline bool canonical(int id) {
  return id == 22 || pion(id) || kaon(id) || id == 221 || id == 331
      || std::abs(id) == 2112 || std::abs(id) == 2212;
}

inline std::string key(const State& state) {
  std::ostringstream out;
  for (const auto& item : state) {
    if (out.tellp() > 0) out << ' ';
    out << item.first << ':' << item.second;
  }
  return out.str();
}

inline std::string classify(const std::string& classifier, const State& state) {
  int n = 0, np = 0, nk = 0, ne = 0, n221 = 0, ng = 0, nbp = 0, nbm = 0;
  for (const auto& item : state) {
    const int id = item.first, count = item.second;
    if (!canonical(id) || count < 1) throw std::runtime_error("noncanonical matched state");
    n += count;
    if (pion(id)) np += count;
    if (kaon(id)) nk += count;
    if (id == 221 || id == 331) ne += count;
    if (id == 221) n221 += count;
    if (id == 22) ng += count;
    if (id >= 1000) nbp += count;
    if (id <= -1000) nbm += count;
  }
  if (classifier == "b-l-resolved-families") {
    if (n == 5 && np == 5) return "exact-5pi";
    if (n221 == 1 && n == 4 && np == 3) return "eta-three-pion";
    if (n221 == 1 && n == 3 && nk == 2) return "eta-kaon-pair";
    if (state == State{{22, 1}, {111, 1}}) return "exact-pi0-gamma";
    if (n && n == np) {
      if (n == 2 || n == 3 || n == 4 || n == 6) return "exact-" + std::to_string(n) + "pi";
      return "remainder";
    }
    if (n == 2 && nk == 2) return "exact-kk";
    if (n == 3 && nk == 2 && np == 1) return "exact-kkpi";
    if (n == 4 && nk == 2 && np == 2) return "exact-kkpipi";
    if (state == State{{-2112, 1}, {2112, 1}} || state == State{{-2212, 1}, {2212, 1}})
      return "exact-nucleon-pair";
    return "remainder";
  }
  if (classifier == "alp-fermion-exact-families") {
    if (nbp || nbm) return "baryon-pair";
    if (nk) {
      if (n == 3 && nk == 2 && np == 1) return "kkpi";
      return np >= 2 ? "kkpipi" : "remainder";
    }
    if (ne == 1 && np == 2 && n == 3) return "eta-pipi";
    if (ne) return "remainder";
    if (np == 3 && n == 3) return "three-pion";
    if (state == State{{-211, 1}, {22, 1}, {211, 1}}) return "pipi-gamma";
    if (np == 3 || (np == 2 && ng >= 1)) return "remainder";
    if (np == 4) return "four-pion";
    if (np >= 5) return "fiveplus-pion";
    return "remainder";
  }
  if (classifier == "scalar-families") {
    if (n == 2 && np == 2) return "pipi";
    if (n == 2 && nk == 2) return "kk";
    if (n == 4 && np == 4) return "four-pion";
    if (nbp && nbm) return "baryon-pair";
    if (ne) return "eta-multihadron";
    if (nk) return "kaon-multihadron";
    if (n >= 5 && np == n) return "higher-multipion";
    return "remainder";
  }
  if (classifier == "hnl-fixed-w-families") {
    // Preserve the Python classifier's literal pair-membership condition.
    if (nbp && nbm) return "nucleon-pair";
    if (n && np == n) {
      if (n == 2) return "two-pion";
      if (n == 3) return "three-pion";
      if (n == 4) return "four-pion";
      return n == 5 ? "five-plus" : "remaining";
    }
    if (n == 2 && nk == 2) return "kaon-pair";
    if (n == 3 && nk == 2 && np == 1) return "kaon-pair-pion";
    if (n == 3 && n221 == 1 && np == 2) return "eta-pion-pion";
    return "remaining";
  }
  throw std::runtime_error("unsupported compiled portable classifier: " + classifier);
}

inline std::uint64_t splitmix(std::uint64_t value) {
  value += UINT64_C(0x9E3779B97F4A7C15);
  value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31);
}

struct Filter {
  std::string classifier, remainder;
  double envelope = 0.;
  std::map<std::string, double> weights;
  std::set<std::string> removedFamilies, removedKeys;
  std::map<std::string, std::map<std::string, double>> conditional;
  std::set<int> expand;

  void read(std::istream& input, int count) {
    *this = Filter{};
    for (int i = 0; i < count; ++i) {
      std::string line, verb, name, rest;
      if (!std::getline(input, line)) throw std::runtime_error("truncated matching configuration");
      std::istringstream row(line);
      row >> verb;
      double value = 0.;
      if (verb == "MODEL") row >> classifier >> remainder;
      else if (verb == "ENVELOPE") row >> envelope;
      else if (verb == "F") {
        row >> name >> value;
        if (!weights.emplace(name, value).second) throw std::runtime_error("duplicate matching family");
      } else if (verb == "R") {
        row >> name;
        removedFamilies.insert(name);
      }
      else if (verb == "V") {
        std::getline(row >> std::ws, rest);
        removedKeys.insert(rest);
      }
      else if (verb == "C") {
        row >> name >> value;
        std::getline(row >> std::ws, rest);
        if (!conditional[name].emplace(rest, value).second) throw std::runtime_error("duplicate conditional state");
      } else if (verb == "X") {
        int id = 0;
        row >> id;
        if (std::abs(id) != 311) throw std::runtime_error("unsupported matching expansion");
        expand.insert(id);
      } else throw std::runtime_error("unknown matching configuration row");
      if (row.fail() || !std::isfinite(value) || value < 0.) throw std::runtime_error("malformed matching row");
    }
    if (classifier.empty() || remainder.empty() || weights.empty()
        || !weights.count(remainder) || !std::isfinite(envelope) || envelope <= 0.)
      throw std::runtime_error("incomplete matching configuration");
    classify(classifier, State{{111, 1}}); // reject unsupported classifiers even for empty batches
  }
  std::string family(const State& state) const {
    const std::string raw = classify(classifier, state);
    if (removedFamilies.count(raw) || removedKeys.count(key(state))) return {};
    return weights.count(raw) ? raw : remainder;
  }
  // Family weight times the charge-channel factor, bounded by the envelope.
  double weight(const std::string& family, const State& resolved) const {
    double factor = weights.at(family);
    const auto found = conditional.find(family);
    if (found != conditional.end()) {
      const auto channel = found->second.find(key(resolved));
      if (channel == found->second.end()) throw std::runtime_error("unresolved channel in matched family");
      factor *= channel->second;
    }
    if (factor > envelope * (1. + 2.e-12)) throw std::runtime_error("matching weight exceeds envelope");
    return factor;
  }
};
} // namespace portable_rejection
