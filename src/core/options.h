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
  Simple,   // one thread, textbook algorithm without optimizations (reference)
  Serial,   // one thread; the optimized algorithm
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
  int threads = 0; // 0 = every CPU the process may run on (threadCount())
  // Nodes with at least this many rows build their left child as a pool task.
  static constexpr std::size_t nodeTaskRows = 4096;
  // Nodes with at least this many rows scan / partition features in parallel.
  // Below it, waking the other threads costs more than they save: at 4096,
  // a 48-core server spent a third of the build in the kernel (futex). 16K-64K
  // were best on 10.5M and 115M rows.
  static constexpr std::size_t featureParallelRows = 32768;
};

struct GpuOptions {
  // Nodes with fewer rows are copied back and finished by the CPU pool while
  // the GPU keeps working on the large nodes of the next level. 0 = auto:
  // the training rows / kAutoMinRowsShare, between kAutoMinRows and
  // kAutoMaxMinRows (CART: rows / kAutoMinRowsShareCart, between kAutoMinRows
  // and kAutoMaxMinRowsCart).
  // A GPU level costs about the same for a thousand mid-size nodes as for a
  // few big ones, so on bigger datasets the CPU takes over somewhat earlier,
  // but not much: measured on an H100 (32 CPU cores) and an A100 (48), C4.5
  // was best at about 2.5K rows on 10.5M rows and at 4K on 115M; CART, whose
  // CPU subtrees are few, at 1-2.5K and 2K (16-64K before the CPU subtrees
  // stopped waiting on the node store).
  std::size_t minRows = 0;
  static constexpr std::size_t kAutoMinRows = 1024;
  static constexpr std::size_t kAutoMaxMinRows = 4096;
  static constexpr std::size_t kAutoMinRowsShare = 4096;
  static constexpr std::size_t kAutoMaxMinRowsCart = 2048;
  static constexpr std::size_t kAutoMinRowsShareCart = 8192;
  // Grow every node on the GPU: no hand-offs to the CPU, and no host arena
  // for them (C4.5's pruning allocates it when it needs it). With --threads 1
  // the CPU side is one thread (the staged copies too).
  bool only = false;
  GpuSweep sweep = GpuSweep::Auto;
};

struct Options {
  std::string datasetPath = "datasets/covertype.csv";
  Backend backend = Backend::Parallel;
  Algorithm algorithm = Algorithm::Cart;

  // Fraction of rows held out as a test set (0 = train and evaluate on all
  // rows). The split is a deterministic shuffle so runs are comparable.
  double holdout = 0.0;
  // The last N rows of the file are the test set (0 = off), e.g. HIGGS's and
  // SUSY's published split (their last 500,000 rows). Not with holdout.
  std::size_t testLast = 0;
  // Seeds the random choices: the holdout split and CART's CV folds. Same seed = same choices on every machine and backend.
  std::uint64_t seed = 1;
  // Energy benchmarks: train this many trees on the loaded data, one after
  // another, and print each one's training window as Unix times (0 = one
  // tree, no windows printed). Every tree but the last is trained in a forked
  // child process, so each one starts as a fresh process's only tree would
  // (bench/power); the last one, trained by ./tree itself, is the one
  // evaluated and dumped.
  std::size_t repeat = 0;

  std::string dumpPath;   // write the tree as text to a file
  // Write where the training time goes, as JSON (src/core/trace.h).
  std::string profilePath;

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

// Threads to use (options.parallel.threads, or every CPU the process may run on).
unsigned threadCount(const Options &options);

// Parse argv on top of `options`. Throws std::runtime_error on bad input.
// Returns false if --help was requested.
bool applyCommandLine(int argc, char *argv[], Options &options);
void printUsage(const char *program);

// Reject settings that are not a valid CART or C4.5 configuration.
void validateOptions(const Options &options);

} // namespace dt
