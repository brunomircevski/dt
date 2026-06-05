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

  // TreeCuda launch tuning (ignored by TreeSerial / TreeParallel).
  std::size_t cudaRowsPerTile = 32768;
  int cudaMaxTilesPerFeature = 128;
  int cudaScoreThreadsPerBlock = 256;
  int cudaGatherBlockSize = 256;
};

class TreeBase;

const char *backendName(Backend backend);
std::unique_ptr<TreeBase> createTree(Backend backend);
void applyCommandLine(int argc, char *argv[], Options &options);
