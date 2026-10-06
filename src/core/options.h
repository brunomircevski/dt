#pragma once

#include <cstddef>
#include <string>

namespace dt {

// Which learning algorithm to run. Each one has its own split rule, stopping
// rules and pruning method; the option groups below belong to one of them.
enum class Algorithm {
  Cart, // Breiman et al. 1984: binary splits, max impurity decrease, CCP pruning
  C45   // Quinlan 1993 (Release 8): gain ratio, MDL threshold cost, EBP pruning
};

// Impurity used by CART. C4.5 always uses entropy (information gain).
enum class Criterion : int { Gini, Entropy };

// Where the tree is grown.
enum class Backend {
  Serial,   // one thread; the reference implementation
  Parallel, // CPU thread pool (feature- and node-level parallelism)
  Cuda      // GPU for large nodes (breadth-first), CPU pool for small subtrees
};

// How CART chooses the cost-complexity parameter alpha.
enum class CartPruning {
  None,           // keep the maximal tree
  Alpha,          // fixed alpha
  CrossValidation // Breiman: K-fold CV over the pruning sequence, 1-SE rule
};

// How the GPU scores cuts.
enum class GpuSweep {
  Auto,    // OnePass on GPUs with fast double precision (data-center parts)
  OnePass, // every cut in double precision
  TwoPass  // single-precision filter, double precision only for candidates
};

struct CartOptions {
  Criterion criterion = Criterion::Gini;
  std::size_t minSplit = 2; // nodes with fewer rows become leaves
  std::size_t minLeaf = 1;  // every child must keep at least this many rows
  // A split must satisfy (rows / totalRows) * impurityDecrease >= this
  // (scikit-learn's min_impurity_decrease).
  double minDecrease = 0.0;

  // Minimal cost-complexity pruning: the smallest subtree minimising
  // R(T) + alpha * |leaves(T)|, R(T) = training misclassification rate.
  CartPruning pruning = CartPruning::CrossValidation;
  double alpha = 0.0; // CartPruning::Alpha
  int folds = 10;     // CartPruning::CrossValidation
};

// Defaults match the original c4.5 program (-m 2 -c 25, subtree raising).
struct C45Options {
  std::size_t minObjects = 2; // -m: min cases in at least two branches
  bool prune = true;          // error-based pruning with subtree raising
  double confidence = 0.25;   // -c 25
};

struct ParallelOptions {
  int threads = 0; // 0 = std::thread::hardware_concurrency()
  // Nodes with at least this many rows build their left child as a pool task.
  std::size_t nodeTaskRows = 4096;
  // Nodes with at least this many rows scan / partition features in parallel.
  std::size_t featureParallelRows = 65536;
};

struct GpuOptions {
  // Nodes with fewer rows are copied back and finished by the CPU pool while
  // the GPU keeps working on the large nodes of the next level.
  std::size_t minRows = 512;
  GpuSweep sweep = GpuSweep::Auto;
};

struct Options {
  std::string datasetPath = "datasets/covertype.csv";
  Backend backend = Backend::Parallel;
  Algorithm algorithm = Algorithm::Cart;

  // Fraction of rows held out as a test set (0 = train and evaluate on all
  // rows). The split is a deterministic shuffle so runs are comparable.
  double holdout = 0.0;
  // Stress tests: duplicate the loaded rows in memory (slightly rescaled) to
  // get a bigger dataset without a bigger CSV. 1 = no duplication.
  std::size_t multiplier = 1;

  bool printTree = false; // print the tree as text to stdout
  std::string dumpPath;   // write the tree as text to a file

  int maxDepth = -1; // both algorithms; -1 = unlimited

  CartOptions cart;
  C45Options c45;
  ParallelOptions parallel;
  GpuOptions gpu;
};

const char *backendName(Backend backend);
const char *algorithmName(Algorithm algorithm);
const char *criterionName(Criterion criterion);

// Threads to use (options.parallel.threads, or all cores).
unsigned threadCount(const Options &options);

// Parse argv on top of `options`. Throws std::runtime_error on bad input.
// Returns false if --help was requested.
bool applyCommandLine(int argc, char *argv[], Options &options);
void printUsage(const char *program);

// Reject settings that are not a valid CART or C4.5 configuration.
void validateOptions(const Options &options);

} // namespace dt
