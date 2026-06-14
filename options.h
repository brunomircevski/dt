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
  Backend backend = Backend::Parallel;
  std::string datasetPath = "datasets/covertype.csv";

  int maxDepth = -1;
  std::size_t minSamplesToSplit = 2;
  std::size_t minSamplesPerLeaf = 1;

  PruningMode pruningMode = PruningMode::None;
  SplitSelectionMode splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
  double epsilon = 1e-9;
  double pruningConfidenceFactor = 0.25;
  double ccpAlpha = 0.5;
  ImpurityMeasure impurityMeasure = ImpurityMeasure::Entropy;

  int maxFeatureThreadCount = 4;
  int maxNodeThreadCount = 4;
  std::size_t minFeaturesToParallelize = 4;
  std::size_t minRowsToParallelize = 32;

  // TreeCuda only (ignored by TreeSerial / TreeParallel).
  //
  // Large-node GPU scan: split each feature's sorted rows into tiles so more
  // blocks can run in parallel. Smaller values = more tiles.
  std::size_t cudaRowsPerTile = 32768;
  // Max tiles per feature; also sizes GPU buffers allocated at fit() time.
  int cudaMaxTilesPerFeature = 128;
  // Threads per block in the split-scoring kernels (32–1024).
  int cudaScoreThreadsPerBlock = 256;
  // Threads per block in the gather kernel (32–1024).
  int cudaGatherBlockSize = 256;
  // Concurrent GPU workers (scratch + stream each). Set in main, not CLI.
  // 1 = one stream; higher T adds VRAM (~geometric series) and allows more
  // overlapping large-node GPU jobs. Does not size the CPU tree-walk pool.
  int cudaGpuWorkerCount = 4;
  // Nodes with fewer rows use the exact CPU split path (same as TreeSerial).
  // 0 = always GPU. Values above dataset size = always CPU (matches serial).
  std::size_t cudaMinRowsForGpu = 2048;
};

class TreeBase;

const char *backendName(Backend backend);
std::unique_ptr<TreeBase> createTree(Backend backend);
void applyCommandLine(int argc, char *argv[], Options &options);
