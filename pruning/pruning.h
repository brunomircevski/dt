#pragma once

#include "tree_base.h"

// Post-training pruning (C4.5 pessimistic error or CART cost-complexity).
// Works on any TreeBase — same tree for Serial, Paraller, and (future) CUDA.
void pruneTree(TreeBase &tree, const TrainingOptions &options);
