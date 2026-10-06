#pragma once

#include "core/tree.h"

#include <vector>

namespace dt {

// CART minimal cost-complexity pruning (Breiman et al. 1984, ch. 3).
// Cost of a subtree T: R(T) + alpha * |leaves(T)|, where R(T) is the fraction
// of the training rows (the root's count) that T misclassifies.

// Prune to T(alpha): the smallest subtree with minimal cost.
void cartCostComplexityPrune(Tree &tree, double alpha);

struct PruningSequence {
  // alpha_1 = 0 < alpha_2 < ...: alpha_k turns T_{k-1} into T_k.
  std::vector<double> alphas;
  // Per node: the alpha at which weakest-link pruning turns it into a leaf
  // (+infinity for leaves).
  std::vector<double> collapseAlpha;
};

PruningSequence cartPruningSequence(const Tree &tree);

} // namespace dt
