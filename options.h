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
  // On large nodes, each feature's sorted rows are split into tiles so more GPU
  // blocks can work in parallel. Smaller values = more tiles, more parallelism.
  std::size_t cudaRowsPerTile = 32768;
  // Cap on tiles per feature. Also sizes GPU buffers allocated at fit() time.
  // Higher = more parallelism on big nodes, but more VRAM.
  int cudaMaxTilesPerFeature = 128;
  // Threads per block when scanning sorted values to find the best split.
  // Must be between 32 and 1024 (CUDA warp size and max threads per block).
  int cudaScoreThreadsPerBlock = 256;
  // Threads per block when copying this node's rows into GPU scratch memory.
  int cudaGatherBlockSize = 256;
};

class TreeBase;

const char *backendName(Backend backend);
std::unique_ptr<TreeBase> createTree(Backend backend);
void applyCommandLine(int argc, char *argv[], Options &options);
