#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dt {

// Which learning algorithm to run. Each one has its own split rule, stopping
// rules and pruning method; the option groups below belong to one of them.
enum class Algorithm {
  Cart, // Breiman et al. 1984: binary splits, max impurity decrease, CCP pruning
  C45   // Quinlan 1993 (Release 8): gain ratio, MDL threshold cost, EBP pruning
};

// Where the tree is grown.
enum class Backend {
  Serial,   // one thread; the reference implementation
  Parallel, // CPU thread pool (feature- and node-level parallelism)
  Cuda      // GPU for large nodes (breadth-first), CPU pool for small subtrees
};

// How CART gets the cost-complexity parameter alpha. A fixed alpha is the
// default, as in scikit-learn (ccp_alpha) and rpart (cp): one tree, grown on
// all rows, like C4.5.
enum class CartPruning {
  None,            // keep the maximal tree
  Alpha,           // fixed alpha
  CrossValidation  // K-fold CV over the pruning sequence (1-SE rule): K + 1 trees
};

// How the GPU scores cuts.
enum class GpuSweep {
  Auto,    // OnePass on GPUs with fast double precision (data-center parts)
  OnePass, // every cut in double precision
  TwoPass  // single-precision filter, double precision only for candidates
};

// CART always uses the Gini index, as in Breiman et al.
struct CartOptions {
  // Minimal cost-complexity pruning: the smallest subtree minimising
  // R(T) + alpha * |leaves(T)|, R(T) = training misclassification rate.
  CartPruning pruning = CartPruning::Alpha;
  double alpha = 0.00001; // CartPruning::Alpha
  int folds = 10;        // CartPruning::CrossValidation
};

// Defaults match the original c4.5 program (-c 25, subtree raising).
struct C45Options {
  bool prune = true;        // error-based pruning with subtree raising
  double confidence = 0.25; // -c 25
};

struct ParallelOptions {
  int threads = 0; // 0 = std::thread::hardware_concurrency()
  // Nodes with at least this many rows build their left child as a pool task.
  std::size_t nodeTaskRows = 4096;
  // Nodes with at least this many rows scan / partition features in parallel.
  std::size_t featureParallelRows = 4096;
};

struct GpuOptions {
  // Nodes with fewer rows are copied back and finished by the CPU pool while
  // the GPU keeps working on the large nodes of the next level.
  std::size_t minRows = 512;
  GpuSweep sweep = GpuSweep::Auto;
  // Print per-level and per-kernel GPU times. Synchronises after every
  // kernel, so it slows the build down.
  bool profile = false;
};

struct Options {
  std::string datasetPath = "datasets/covertype.csv";
  Backend backend = Backend::Parallel;
  Algorithm algorithm = Algorithm::Cart;

  // Fraction of rows held out as a test set (0 = train and evaluate on all
  // rows). The split is a deterministic shuffle so runs are comparable.
  double holdout = 0.0;
  // Seeds the random choices: the holdout split and CART's CV folds. Same seed = same choices on every machine and backend.
  std::uint64_t seed = 1;
  // Stress tests: duplicate the loaded rows in memory (slightly rescaled) to
  // get a bigger dataset without a bigger CSV. 1 = no duplication.
  std::size_t multiplier = 1;

  bool printTree = false; // print the tree as text to stdout
  std::string dumpPath;   // write the tree as text to a file

  // Pre-pruning (stopping rules), the same for both algorithms.
  int maxDepth = -1; // -1 = unlimited
  // Every child of a split keeps at least this many rows. 0 = the
  // algorithm's own default: 1 for CART, 2 for C4.5 (its -m).
  std::size_t minLeaf = 0;

  CartOptions cart;
  C45Options c45;
  ParallelOptions parallel;
  GpuOptions gpu;
};

const char *backendName(Backend backend);
const char *algorithmName(Algorithm algorithm);

// Threads to use (options.parallel.threads, or all cores).
unsigned threadCount(const Options &options);

// Parse argv on top of `options`. Throws std::runtime_error on bad input.
// Returns false if --help was requested.
bool applyCommandLine(int argc, char *argv[], Options &options);
void printUsage(const char *program);

// Reject settings that are not a valid CART or C4.5 configuration.
void validateOptions(const Options &options);

} // namespace dt
