#include "isospin_cg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

static long double lnFact(int n) {
  // ln(n!) = lgamma(n + 1), tabulated once for 0 <= n < 128; other n evaluate the same expression directly.
  static const std::array<long double, 128> table = [] {
    std::array<long double, 128> values{};
    for (unsigned i = 0; i < values.size(); ++i)
      values[i] = std::lgamma((long double) i + 1.0L);
    return values;
  }();
  if (n >= 0 && n < int(table.size())) return table[n];
  return std::lgamma((long double) n + 1.0L);
}

namespace {
using CGKey = std::array<int, 6>;
struct CGHash {
  std::size_t operator()(const CGKey& key) const {
    std::size_t value = 0;
    for (int item : key) value = value * 131 + static_cast<unsigned>(item);
    return value;
  }
};
} // namespace

static bool halfToIntSafe(int x, int& out) {
  if (x % 2 != 0) return false;
  out = x / 2;
  return true;
}

static long double wigner3j(int j1_2, int j2_2, int j3_2,
                            int m1_2, int m2_2, int m3_2) {
  if (m1_2 + m2_2 + m3_2 != 0) return 0.0L;
  if (std::abs(m1_2) > j1_2 || std::abs(m2_2) > j2_2 || std::abs(m3_2) > j3_2) return 0.0L;
  if ((j1_2 + j2_2 + j3_2) % 2 != 0) return 0.0L;
  if (j3_2 < std::abs(j1_2 - j2_2) || j3_2 > j1_2 + j2_2) return 0.0L;

  int a, b, c, d, t1, t2, t3, t4, t5, t6;
  if (!halfToIntSafe(j1_2 + j2_2 - j3_2, a)) return 0.0L;
  if (!halfToIntSafe(j1_2 - j2_2 + j3_2, b)) return 0.0L;
  if (!halfToIntSafe(-j1_2 + j2_2 + j3_2, c)) return 0.0L;
  if (!halfToIntSafe(j1_2 + j2_2 + j3_2, d)) return 0.0L;
  d += 1;

  if (!halfToIntSafe(j1_2 + m1_2, t1)) return 0.0L;
  if (!halfToIntSafe(j1_2 - m1_2, t2)) return 0.0L;
  if (!halfToIntSafe(j2_2 + m2_2, t3)) return 0.0L;
  if (!halfToIntSafe(j2_2 - m2_2, t4)) return 0.0L;
  if (!halfToIntSafe(j3_2 + m3_2, t5)) return 0.0L;
  if (!halfToIntSafe(j3_2 - m3_2, t6)) return 0.0L;

  if (a < 0 || b < 0 || c < 0 || d <= 0) return 0.0L;
  if (t1 < 0 || t2 < 0 || t3 < 0 || t4 < 0 || t5 < 0 || t6 < 0) return 0.0L;

  int phaseExp;
  if (!halfToIntSafe(j1_2 - j2_2 - m3_2, phaseExp)) return 0.0L;
  const long double phase = (phaseExp % 2 == 0) ? 1.0L : -1.0L;

  const long double lnDelta = lnFact(a) + lnFact(b) + lnFact(c) - lnFact(d);
  const long double lnProd  = lnFact(t1) + lnFact(t2) + lnFact(t3) + lnFact(t4) + lnFact(t5) + lnFact(t6);
  const long double pref = phase * std::exp(0.5L * (lnDelta + lnProd));

  int e1, e2;
  if (!halfToIntSafe(j3_2 - j2_2 + m1_2, e1)) return 0.0L;
  if (!halfToIntSafe(j3_2 - j1_2 - m2_2, e2)) return 0.0L;

  int kmin = 0;
  kmin = std::max(kmin, -e1);
  kmin = std::max(kmin, -e2);

  int kmax = a;
  kmax = std::min(kmax, t2);
  kmax = std::min(kmax, t3);

  if (kmin > kmax) return 0.0L;

  long double sum = 0.0L;
  for (int k = kmin; k <= kmax; ++k) {
    const int A = k;
    const int B = a - k;
    const int C = t2 - k;
    const int D = t3 - k;
    const int E = e1 + k;
    const int F = e2 + k;
    if (B < 0 || C < 0 || D < 0 || E < 0 || F < 0) continue;

    const long double lnTerm = - (lnFact(A) + lnFact(B) + lnFact(C) + lnFact(D) + lnFact(E) + lnFact(F));
    long double term = std::exp(lnTerm);
    if (k % 2) term = -term;
    sum += term;
  }

  return pref * sum;
}

// Clebsch-Gordan coefficient <j1 m1; j2 m2 | J M>; all arguments doubled.
static long double cg(int j1_2, int m1_2, int j2_2, int m2_2, int J_2, int M_2) {
  if (m1_2 + m2_2 != M_2) return 0.0L;
  if (std::abs(m1_2) > j1_2 || std::abs(m2_2) > j2_2 || std::abs(M_2) > J_2) return 0.0L;
  if ((j1_2 + j2_2 + J_2) % 2 != 0) return 0.0L;
  if (J_2 < std::abs(j1_2 - j2_2) || J_2 > j1_2 + j2_2) return 0.0L;

  // These coefficients depend only on the six doubled quantum numbers,
  // never on a mass, tune, source, event or random state. Keep a bounded,
  // thread-local exact memoization rather than redoing Wigner sums for
  // every rejected hadronization proposal.
  static thread_local std::unordered_map<CGKey, long double, CGHash> cache;
  const CGKey key{{j1_2, m1_2, j2_2, m2_2, J_2, M_2}};
  const auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  const int m3_2 = -M_2;
  const long double threej = wigner3j(j1_2, j2_2, J_2, m1_2, m2_2, m3_2);
  if (threej == 0.0L) {
    if (cache.size() < 4096) cache.emplace(key, 0.0L);
    return 0.0L;
  }

  const int phaseExp = (j1_2 - j2_2 + M_2) / 2;
  const long double phase = (phaseExp % 2 == 0) ? 1.0L : -1.0L;
  const long double sqrt2J1 = std::sqrt((long double)(J_2 + 1));
  const long double result = phase * sqrt2J1 * threej;
  if (cache.size() < 4096) cache.emplace(key, result);
  return result;
}

std::unordered_map<int, long double>
diagonalIsospinProbabilities2(
    const std::vector<std::pair<int, int>>& inputParts) {
  std::unordered_map<int, long double> out;
  if (inputParts.empty()) return out;

  // Sorting the input makes the projection independent of event-record and hash order.
  std::vector<std::pair<int, int>> parts = inputParts;
  std::sort(parts.begin(), parts.end());

  int mcur_2 = parts[0].second;
  std::map<int, long double> weights;
  weights[parts[0].first] = 1.0L;

  for (std::size_t idx = 1; idx < parts.size(); ++idx) {
    const int i2 = parts[idx].first;
    const int mi2 = parts[idx].second;
    const int mnew_2 = mcur_2 + mi2;

    std::map<int, long double> updated;
    for (const auto& current : weights) {
      const int currentI2 = current.first;
      const int minimumI2 = std::abs(currentI2 - i2);
      const int maximumI2 = currentI2 + i2;
      for (int nextI2 = minimumI2; nextI2 <= maximumI2; nextI2 += 2) {
        const long double coefficient = cg(
            currentI2, mcur_2, i2, mi2, nextI2, mnew_2);
        const long double probability = coefficient * coefficient;
        if (probability > 1.0e-28L)
          updated[nextI2] += current.second * probability;
      }
    }
    weights.swap(updated);
    mcur_2 = mnew_2;
    if (weights.empty()) break;
  }

  long double total = 0.0L;
  for (const auto& entry : weights) total += entry.second;
  if (!(total > 0.0L) || !std::isfinite(total)) return out;

  out.reserve(weights.size());
  for (const auto& entry : weights) out[entry.first] = entry.second / total;
  return out;
}

