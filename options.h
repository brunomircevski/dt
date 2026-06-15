#pragma once

#include <cstddef>
#include <memory>
#include <string>

enum class Backend { Serial, Parallel, Cuda };

enum class ImpurityMeasure {
  Entropy,
  Gini
};

enum class SplitSelectionMode {
  MeanGainFiltered,
  MaxGain
};

enum class PruningMode {
  None,
  PessimisticError,
  CostComplexity
};

struct Options {
  // ---------------------------------------------------------------------------
  // All backends (TreeSerial, TreeParallel, TreeCuda)
  // ---------------------------------------------------------------------------
  Backend backend = Backend::Parallel; // CLI: --serial | --parallel | --cuda
  std::string datasetPath = "datasets/covertype.csv"; // CLI: <path> (positional)

  int maxDepth = -1; // CLI: -d <N>
  std::size_t minSamplesToSplit = 2;
  std::size_t minSamplesPerLeaf = 1;

  PruningMode pruningMode = PruningMode::None;
  SplitSelectionMode splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
  double epsilon = 1e-9;
  double pruningConfidenceFactor = 0.25;
  double ccpAlpha = 0.5;
  ImpurityMeasure impurityMeasure = ImpurityMeasure::Entropy;

  // ---------------------------------------------------------------------------
  // TreeParallel + TreeCuda
  //
  // Minimum rows in a node before left/right subtrees may be built on different
  // CPU threads. Below this threshold both children stay on the current thread.
  // ---------------------------------------------------------------------------
  std::size_t minRowsToParallelize = 32;

  // ---------------------------------------------------------------------------
  // TreeParallel only (ignored by TreeSerial and TreeCuda)
  // ---------------------------------------------------------------------------
  // Threads that score features in parallel inside one node.
  int parallelMaxFeatureThreadCount = 4;
  // Threads that build sibling subtrees in parallel.
  int parallelMaxNodeThreadCount = 4;
  // Minimum feature count before feature search is parallelized.
  std::size_t parallelMinFeaturesToParallelize = 4;

  // ---------------------------------------------------------------------------
  // TreeCuda only (ignored by TreeSerial and TreeParallel)
  // ---------------------------------------------------------------------------
  // CPU threads that walk the tree and build subtrees in parallel. Each thread
  // may check out a GPU worker for large nodes; at most cudaGpuWorkerCount
  // threads can hold a worker at once. Must be >= cudaGpuWorkerCount.
  int cudaCpuThreadCount = 8;
  // Concurrent GPU workers: each owns VRAM scratch + one CUDA stream. Worker i
  // is sized for max(1, N >> i) rows so total scratch stays ~2× root, not T×.
  int cudaGpuWorkerCount = 4;
  // Nodes with fewer rows use the CPU split path inside TreeCuda. 0 = always GPU.
  std::size_t cudaMinRowsForGpu = 2048;

  // Large-node GPU scan: split each feature into tiles so more blocks run.
  std::size_t cudaRowsPerTile = 32768;
  // Max tiles per feature; also sizes GPU buffers allocated at fit() time.
  int cudaMaxTilesPerFeature = 128;
  // CUDA kernel launch parameters (threads per block).
  int cudaScoreThreadsPerBlock = 256;
  int cudaGatherBlockSize = 256;

  // Demo only: duplicate loaded rows in memory to stress GPU without re-reading
  // CSV. 1 = no duplication; 2 = double the dataset, etc.
  std::size_t demoDatasetMultiplier = 1; // CLI: -m <N>
};

class TreeBase;

const char *backendName(Backend backend);
std::unique_ptr<TreeBase> createTree(Backend backend);
void applyCommandLine(int argc, char *argv[], Options &options);
