#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

// Arguments are doubled isospins (2*I, 2*I3), one pair per hadron.

// Total-isospin projector of a product state: the canonical diagonal trace
// over mutually orthogonal intermediate-coupling paths, P(I) = sum over paths
// of the product of squared Clebsch-Gordan coefficients, normalized to one.
// It is the acceptance of the stochastic source filter and the support check
// of the showered alp-fermion and scalar events (tools/shower_worker.cc).  The input is sorted internally so the result cannot
// depend on event-record or unordered-map iteration order.
std::unordered_map<int, long double>
diagonalIsospinProbabilities2(
    const std::vector<std::pair<int, int>>& parts);
