#pragma once

#include "core/tree.h"

namespace dt {

// CART minimal cost-complexity pruning (Breiman et al. 1984, ch. 3).
// Cost of a subtree T: R(T) + alpha * |leaves(T)|, where R(T) is the fraction
// of the training rows (the root's count) that T misclassifies.

// Prune to T(alpha): the smallest subtree with minimal cost.
void cartCostComplexityPrune(Tree &tree, double alpha);

} // namespace dt
