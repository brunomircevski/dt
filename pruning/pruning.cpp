#include "pruning/pruning.h"

void pruneTree(TreeBase &tree, const TrainingOptions &options) {
  tree.prune(options);
}
