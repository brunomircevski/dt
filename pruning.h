#pragma once

#include "dataset.h"
#include "thread_pool.h"
#include "tree.h"

#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// C4.5 (Release 8) post-processing, in the order c4.5 applies it.
// ---------------------------------------------------------------------------

// build.c, end of FormTree: a subtree that makes as many training errors as a
// single leaf would is collapsed into that leaf.
void c45CollapseUselessSplits(Node *root);

// contin.c, ContinTest: a threshold is the largest training value of the
// feature (over the whole training set) that is <= the cut's midpoint.
// The training partition does not change; only unseen values in the gap do.
// `distinctValues[f]` are feature f's distinct training values, increasing;
// if empty they are computed here.
void c45UseTrainingValueThresholds(Node *root, const Dataset &train,
                                   std::vector<std::vector<float>> &distinctValues);

// prune.c: error-based pruning. Each node's error is estimated with the upper
// limit of a binomial confidence interval (confidence factor CF). A subtree is
// replaced by a leaf, or by its most-used branch ("subtree raising"), when that
// is not estimated to be worse.
void c45PessimisticPrune(std::unique_ptr<Node> &root, const Dataset &train,
                         double confidenceFactor, bool subtreeRaising, ThreadPool *pool);

// ---------------------------------------------------------------------------
// CART minimal cost-complexity pruning (Breiman et al. 1984, ch. 3)
// ---------------------------------------------------------------------------
// Cost of a subtree T: R(T) + alpha * |leaves(T)|, where R(T) is the fraction
// of the `totalRows` training rows that T misclassifies.

// Prune to T(alpha): the smallest subtree with minimal cost.
void cartCostComplexityPrune(Node *root, double alpha, std::size_t totalRows);

// For every node, the alpha at which weakest-link pruning turns it into a leaf
// (+infinity for leaves... and nodes that are never pruned before the root).
// Stored in Node::errors. Returns the distinct alphas of the pruning sequence
// alpha_1 = 0 < alpha_2 < ... (alpha_k turns T_{k-1} into T_k).
std::vector<double> cartPruningSequence(Node *root, std::size_t totalRows);
