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

// CART always uses the Gini index, as in Breiman et al.
struct CartOptions {
  // Minimal cost-complexity pruning with a fixed alpha, as scikit-learn
  // (ccp_alpha) and rpart (cp) do: the smallest subtree minimising
  // R(T) + alpha * |leaves(T)|, R(T) = training misclassification rate.
  bool prune = true;
  double alpha = 0.00001;
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
  // the GPU keeps working on the large nodes of the next level. 0 = auto
  // (gpuMinRows()): the training rows / kAutoMinRowsShare, between
  // kAutoMinRows and kAutoMaxMinRows (CART: rows / kAutoMinRowsShareCart,
  // between kAutoMinRows and kAutoMaxMinRowsCart).
  // A GPU level costs about the same for a thousand mid-size nodes as for a
  // few big ones, so on bigger datasets the CPU takes over somewhat earlier,
  // but not much: measured on an H100 (32 CPU cores) and an A100 (48), C4.5
  // was best at about 2.5K rows on 10.5M rows and at 4K on 115M; CART, whose
  // CPU subtrees are few, at 1-2.5K and 2K.
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
};

struct Options {
  std::string datasetPath = "datasets/covertype.csv";
  // --rows N: only the training file's first N rows (0 = all), e.g. sorel_4m
  // from SOREL's CSV without a copy of its first 4M rows. --test-last and
  // --holdout then split these rows.
  std::size_t rows = 0;
  Backend backend = Backend::Parallel;
  Algorithm algorithm = Algorithm::Cart;

  // The test set, at most one of these three (none: no test accuracy):
  // a separate CSV with the same header as the training file,
  std::string testPath;
  // the last N rows of the training file (e.g. HIGGS's and SUSY's published
  // split: their last 500,000 rows), or
  std::size_t testLast = 0;
  // a fraction of the rows, shuffled with `seed` (the same split on every run).
  double holdout = 0.0;
  // Seeds the holdout split: the same seed gives the same split on every
  // machine and backend.
  std::uint64_t seed = 1;
  // Energy benchmarks: train this many trees on the loaded data, one after
  // another, and print each one's training window as Unix times (0 = one
  // tree, no windows printed). With N > 1 every tree is trained in a forked
  // child process, so each one starts as a fresh process's only tree would
  // (bench/power); the last one is the one evaluated and dumped.
  std::size_t repeat = 0;
  // --repeat-warmup: seconds that each --repeat tree's child keeps its threads
  // busy before its timer starts, as a single run's CSV load does before its
  // tree. Without it, a GPU tree that follows an idle CPU (the previous
  // tree's GPU-only build) set up 10% (H100 server) to 40% (A100 server) slower:
  // the socket's idle power state slows its host-memory copies.
  double repeatWarmup = 0.0;

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

// --cuda: nodes with fewer rows go to the CPU (options.gpu.minRows, or the
// automatic value for this many training rows; 1 with --gpu-only).
std::size_t gpuMinRows(const Options &options, std::size_t trainingRows);

// Parse argv on top of `options`. Throws std::runtime_error on bad input.
// Returns false if --help was requested.
bool applyCommandLine(int argc, char *argv[], Options &options);
void printUsage(const char *program);

// Reject settings that are not a valid CART or C4.5 configuration.
void validateOptions(const Options &options);

} // namespace dt
