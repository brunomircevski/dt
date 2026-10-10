#include "app/trainer.h"

#include "algo/c45_pruning.h"
#include "algo/cart_pruning.h"
#include "algo/split_rules.h"
#include "build/grower.h"
#include "core/timing.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>

namespace dt {

namespace {

// Subtree k of a pruning sequence is the tree pruned at any alpha in
// [alpha_k, alpha_k+1); errors are measured at the geometric midpoint, as
// Breiman et al. do. The last subtree (the root alone) gets a finite point
// above every alpha so it stays inside the half-open intervals below.
std::vector<double> evaluationPoints(const std::vector<double> &alphas) {
  std::vector<double> betas(alphas.size());
  for (std::size_t k = 0; k < alphas.size(); ++k) {
    betas[k] = k + 1 < alphas.size() ? std::sqrt(alphas[k] * alphas[k + 1])
                                     : std::numeric_limits<double>::max();
  }
  return betas;
}

// Add to errors[k] how many of `rows` of `data` the tree pruned at betas[k]
// misclassifies. The pruned tree predicts with the first node on the row's
// path whose collapse alpha is <= beta; along the path the running minimum of
// collapse alphas only decreases, so each path node owns one interval of beta
// and one walk per row covers every k. Chunks of rows run in parallel.
void addPrunedErrors(const Tree &tree, const std::vector<double> &collapse,
                     const Dataset &data, std::span<const std::uint32_t> rows,
                     const std::vector<double> &betas, std::vector<double> &errors,
                     ThreadPool *pool) {
  const std::size_t chunks = pool ? 4 * (pool->workerCount() + 1) : 1;
  std::vector<std::vector<double>> chunkErrors(chunks); // difference arrays over k
  parallelFor(pool, chunks, [&](std::size_t chunk) {
    std::vector<double> &local = chunkErrors[chunk];
    local.assign(betas.size() + 1, 0.0);
    const std::size_t end = rows.size() * (chunk + 1) / chunks;
    for (std::size_t index = rows.size() * chunk / chunks; index < end; ++index) {
      const std::uint32_t row = rows[index];
      const std::uint16_t truth = data.labels[row];
      double upper = INFINITY;
      for (std::uint32_t id = 0;;) {
        const Node &node = tree.nodes[id];
        const double lower = node.isLeaf() ? -INFINITY : std::min(upper, collapse[id]);
        if (lower < upper && node.label != truth) {
          // betas in [lower, upper) misclassify this row.
          const auto first = std::lower_bound(betas.begin(), betas.end(), lower);
          const auto last = std::lower_bound(betas.begin(), betas.end(), upper);
          local[static_cast<std::size_t>(first - betas.begin())] += 1;
          local[static_cast<std::size_t>(last - betas.begin())] -= 1;
        }
        if (node.isLeaf()) {
          break;
        }
        upper = lower;
        id = data.value(row, static_cast<std::size_t>(node.feature)) <= node.threshold
                 ? node.left
                 : node.right;
      }
    }
  });
  for (const std::vector<double> &local : chunkErrors) {
    double running = 0.0;
    for (std::size_t k = 0; k < betas.size(); ++k) {
      running += local[k];
      errors[k] += running;
    }
  }
}

// Breiman's 1-SE rule: the error estimate of every subtree comes from n rows;
// take the smallest subtree whose error is within one standard error of the
// lowest. Returns its alpha.
double oneStandardErrorAlpha(const std::vector<double> &alphas, const std::vector<double> &errors,
                             double n, bool verbose) {
  std::vector<double> risk(alphas.size());
  std::size_t bestIndex = 0;
  for (std::size_t k = 0; k < alphas.size(); ++k) {
    risk[k] = errors[k] / n;
    if (risk[k] < risk[bestIndex]) {
      bestIndex = k;
    }
  }
  const double standardError = std::sqrt(risk[bestIndex] * (1.0 - risk[bestIndex]) / n);
  std::size_t chosen = bestIndex;
  for (std::size_t k = bestIndex; k < alphas.size(); ++k) {
    if (risk[k] <= risk[bestIndex] + standardError) {
      chosen = k;
    }
  }
  if (verbose) {
    std::cout << "  CV over " << alphas.size() << " subtrees: best alpha "
              << alphas[bestIndex] << " (error " << risk[bestIndex] << "), 1-SE choice "
              << alphas[chosen] << " (error " << risk[chosen] << ")\n";
  }
  return alphas[chosen];
}

// K-fold cross-validation: grow a tree on each fold's complement and measure
// it on the fold, along the main tree's pruning sequence.
double crossValidateAlpha(const Dataset &train, const Tree &mainTree, const Options &options,
                          const SplitRules &rules, Grower &grower, ThreadPool *pool,
                          bool verbose) {
  const std::vector<double> alphas = cartPruningSequence(mainTree).alphas;
  const std::vector<double> betas = evaluationPoints(alphas);

  const int folds = options.cart.folds;
  std::vector<std::uint32_t> order(train.rowCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 random = seededRandom(options.seed, RandomStream::CrossValidation);
  std::shuffle(order.begin(), order.end(), random);
  std::vector<int> foldOf(train.rowCount);
  for (std::size_t index = 0; index < order.size(); ++index) {
    foldOf[order[index]] = static_cast<int>(index % folds);
  }

  std::vector<double> errors(betas.size(), 0.0);
  for (int fold = 0; fold < folds; ++fold) {
    std::vector<std::uint32_t> fitRows; // increasing, as grow() wants them
    std::vector<std::uint32_t> testRows;
    for (std::uint32_t row = 0; row < train.rowCount; ++row) {
      (foldOf[row] == fold ? testRows : fitRows).push_back(row);
    }
    GrowTimings ignored;
    const Tree foldTree = grower.grow(rules, fitRows, ignored);
    addPrunedErrors(foldTree, cartPruningSequence(foldTree).collapseAlpha, train, testRows,
                    betas, errors, pool);
  }
  const double alpha =
      oneStandardErrorAlpha(alphas, errors, static_cast<double>(train.rowCount), verbose);
  if (verbose) {
    std::cout << "  same tree without CV: --alpha " << std::setprecision(17) << alpha
              << std::setprecision(6) << "\n";
  }
  return alpha;
}

} // namespace

Tree trainTree(const Dataset &train, const Options &options, ThreadPool *pool,
               TrainTimings &timings, bool verbose) {
  validateOptions(options);
  if (train.classCount() == 0 || train.rowCount == 0) {
    throw std::runtime_error("Cannot train on an empty dataset.");
  }
  ThreadPool *buildPool = options.backend == Backend::Serial ? nullptr : pool;
  const bool isCart = options.algorithm == Algorithm::Cart;
  const bool crossValidate = isCart && options.cart.pruning == CartPruning::CrossValidation;

  std::unique_ptr<Grower> grower;
  if (options.backend == Backend::Cuda) {
    if (!pool) {
      throw std::logic_error("The Cuda backend needs a thread pool");
    }
    grower = makeGpuGrower(train, options, *pool, crossValidate, timings.gpuSetupSeconds);
  } else {
    grower = makeCpuGrower(train, options, buildPool, crossValidate);
  }

  const SplitRules rules(options, train.classCount());
  GrowTimings growTimings;
  std::vector<float> sortedValues; // C4.5's thresholds need them
  Tree tree = grower->grow(rules, {}, growTimings, isCart ? nullptr : &sortedValues);
  timings.presortSeconds += growTimings.prepareSeconds;
  timings.buildSeconds += growTimings.buildSeconds;

  if (!isCart) {
    ScopedTimer timer(timings.pruneSeconds);
    // Part of growing in c4.5 (done with or without pruning).
    c45CollapseUselessSplits(tree);
    c45UseTrainingValueThresholds(tree, sortedValues, train.rowCount, buildPool);
    if (options.c45.prune) {
      if (verbose) {
        std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
      }
      c45PessimisticPrune(tree, train, options.c45.confidence, buildPool, grower->workspace());
    }
    return tree;
  }
  if (verbose && options.cart.pruning != CartPruning::None) {
    std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
  }

  double alpha = options.cart.alpha;
  if (crossValidate) {
    ScopedTimer timer(timings.cvSeconds);
    alpha = crossValidateAlpha(train, tree, options, rules, *grower, buildPool, verbose);
  }
  if (options.cart.pruning != CartPruning::None) {
    ScopedTimer timer(timings.pruneSeconds);
    cartCostComplexityPrune(tree, alpha);
  }
  return tree;
}

} // namespace dt
