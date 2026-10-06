#pragma once

#include <cstddef>
#include <string>

// Which learning algorithm to run. Each one has its own split rule, stopping
// rules and pruning method; the options below say which knobs belong to which.
enum class Algorithm {
  Cart, // Breiman et al. 1984: binary splits, max impurity decrease, CCP pruning
  C45   // Quinlan 1993 (Release 8): gain ratio, MDL threshold cost, EBP pruning
};

// Impurity used by CART. C4.5 always uses entropy (information gain).
enum class Criterion { Gini, Entropy };

// Where the tree is grown.
enum class Backend {
  Serial,   // one thread; the reference implementation
  Parallel, // CPU thread pool (feature- and node-level parallelism)
  Cuda      // GPU for large nodes (breadth-first), CPU pool for small subtrees
};

struct Options {
  // ---------------------------------------------------------------------------
  // Run
  // ---------------------------------------------------------------------------
  std::string datasetPath = "datasets/covertype.csv";
  Backend backend = Backend::Parallel;
  Algorithm algorithm = Algorithm::Cart;

  // Fraction of rows held out as a test set (0 = train and evaluate on all
  // rows). The split is a deterministic shuffle so runs are comparable.
  double holdoutFraction = 0.0;

  // Demo only: duplicate the loaded rows in memory (slightly rescaled) to
  // stress the builders without a bigger CSV. 1 = no duplication.
  std::size_t datasetMultiplier = 1;

  bool printTree = false;       // print the tree as text to stdout
  std::string svgPath;          // render the tree to SVG (needs python3)
  std::string dumpPath;         // write the tree as text to a file
  // Instead of growing a tree, read one written by --dump and only run the
  // algorithm's post-processing / pruning on it.
  std::string loadTreePath;

  // ---------------------------------------------------------------------------
  // Growth limits (both algorithms)
  // ---------------------------------------------------------------------------
  int maxDepth = -1; // -1 = unlimited

  // ---------------------------------------------------------------------------
  // CART
  // ---------------------------------------------------------------------------
  Criterion criterion = Criterion::Gini;
  std::size_t minSamplesSplit = 2; // nodes with fewer rows become leaves
  std::size_t minSamplesLeaf = 1;  // every child must keep at least this many
  // A split must satisfy (rows/totalRows) * impurityDecrease >= this
  // (same definition as scikit-learn's min_impurity_decrease).
  double minImpurityDecrease = 0.0;

  // Minimal cost-complexity pruning: keep the smallest subtree minimising
  // R(T) + alpha * |leaves(T)|, where R(T) is the training misclassification
  // rate. Used only when cartPrune is true.
  bool cartPrune = false;
  double ccpAlpha = 0.0;
  // If > 1, ignore ccpAlpha and choose alpha by K-fold cross-validation over
  // the weakest-link pruning sequence using the 1-SE rule (Breiman's method).
  int ccpFolds = 0;

  // ---------------------------------------------------------------------------
  // C4.5 (defaults match the original c4.5 program)
  // ---------------------------------------------------------------------------
  std::size_t c45MinObjects = 2;      // -m: min cases in at least two branches
  bool c45Prune = true;               // error-based (pessimistic) pruning
  double c45ConfidenceFactor = 0.25;  // -c 25
  bool c45SubtreeRaising = true;      // replace a node by its largest branch

  // ---------------------------------------------------------------------------
  // Parallel CPU (Parallel backend, and the CPU side of the Cuda backend)
  // ---------------------------------------------------------------------------
  int threads = 0; // 0 = std::thread::hardware_concurrency()
  // Nodes with at least this many rows spawn their children as separate tasks.
  std::size_t minRowsForNodeTask = 4096;
  // Nodes with at least this many rows scan/partition features in parallel.
  std::size_t minRowsForFeatureParallel = 65536;

  // ---------------------------------------------------------------------------
  // Cuda backend
  // ---------------------------------------------------------------------------
  // Nodes with fewer rows are copied back and finished by the CPU pool while
  // the GPU keeps working on the large nodes of the next level.
  std::size_t gpuMinRows = 512;
};

const char *backendName(Backend backend);
const char *algorithmName(Algorithm algorithm);
const char *criterionName(Criterion criterion);

// Parse argv on top of `options`. Throws std::runtime_error on bad input.
// Returns false if --help was requested.
bool applyCommandLine(int argc, char *argv[], Options &options);
void printUsage(const char *program);

// Reject combinations that are not a valid CART or C4.5 configuration.
void validateOptions(const Options &options);
