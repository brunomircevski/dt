#pragma once

#include "dataset.h"
#include "timing.h"
#include "tree_base.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

inline std::string shellEscapeSingleQuoted(const std::string &value) {
  std::string escaped = "'";
  for (char character : value) {
    if (character == '\'') {
      escaped += "'\\''";
    } else {
      escaped += character;
    }
  }
  escaped += "'";
  return escaped;
}

inline std::string impurityMeasureToString(ImpurityMeasure measure) {
  switch (measure) {
    case ImpurityMeasure::Entropy: return "Entropy";
    case ImpurityMeasure::Gini: return "Gini";
  }
  return "Unknown";
}

inline std::string splitSelectionModeToString(SplitSelectionMode mode) {
  switch (mode) {
    case SplitSelectionMode::MeanGainFiltered: return "MeanGainFiltered";
    case SplitSelectionMode::MaxGain: return "MaxGain";
  }
  return "Unknown";
}

inline std::string pruningModeToString(PruningMode mode) {
  switch (mode) {
    case PruningMode::None: return "None";
    case PruningMode::PessimisticError: return "PessimisticError";
    case PruningMode::CostComplexity: return "CostComplexity";
  }
  return "Unknown";
}

} // namespace

inline void printTimingLine(const char *label, double seconds) {
  std::cout << "  " << label << " = " << std::fixed << std::setprecision(4)
            << seconds * 1000.0 << " ms\n";
}

inline void generateTreeSvg(const TreeBase &tree, const std::string &svgPath,
                            const Options &options,
                            const Dataset &dataset) {
  std::ostringstream printedTree;

  std::size_t correctPredictions = 0;
  for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
    const Sample &sample = dataset.samples[index];
    const std::string prediction = tree.predict(sample);
    if (prediction == sample.label) {
      ++correctPredictions;
    }
  }

  const double accuracy = static_cast<double>(correctPredictions) /
                          static_cast<double>(dataset.samples.size()) * 100.0;

  printedTree << "META: checked samples = " << dataset.samples.size() << "\n";
  printedTree << "META: correct predictions = " << correctPredictions << "\n";
  printedTree << "META: tree depth = " << tree.treeDepth() << "\n";
  printedTree << "META: node count = " << tree.nodeCount() << "\n";

  std::ostringstream accStr;
  accStr << std::fixed << std::setprecision(4) << accuracy;
  printedTree << "META: accuracy = " << accStr.str() << "%\n";

  printedTree << "OPTION: impurityMeasure = "
              << impurityMeasureToString(options.impurityMeasure) << "\n";
  printedTree << "OPTION: splitSelectionMode = "
              << splitSelectionModeToString(options.splitSelectionMode) << "\n";
  printedTree << "OPTION: pruningMode = "
              << pruningModeToString(options.pruningMode) << "\n";
  printedTree << "OPTION: maxDepth = " << options.maxDepth << "\n";
  printedTree << "OPTION: minSamplesToSplit = " << options.minSamplesToSplit
              << "\n";
  printedTree << "OPTION: minSamplesPerLeaf = " << options.minSamplesPerLeaf
              << "\n";
  printedTree << "OPTION: parallelMaxFeatureThreadCount = "
              << options.parallelMaxFeatureThreadCount << "\n";
  printedTree << "OPTION: parallelMaxNodeThreadCount = "
              << options.parallelMaxNodeThreadCount << "\n";
  printedTree << "OPTION: parallelMinFeaturesToParallelize = "
              << options.parallelMinFeaturesToParallelize << "\n";
  printedTree << "OPTION: minRowsToParallelize = "
              << options.minRowsToParallelize << "\n";
  printedTree << "OPTION: cudaCpuThreadCount = "
              << options.cudaCpuThreadCount << "\n";
  printedTree << "OPTION: cudaRowsPerTile = " << options.cudaRowsPerTile << "\n";
  printedTree << "OPTION: cudaMaxTilesPerFeature = "
              << options.cudaMaxTilesPerFeature << "\n";
  printedTree << "OPTION: cudaScoreThreadsPerBlock = "
              << options.cudaScoreThreadsPerBlock << "\n";
  printedTree << "OPTION: cudaGatherBlockSize = " << options.cudaGatherBlockSize
              << "\n";
  printedTree << "OPTION: cudaGpuWorkerCount = "
              << options.cudaGpuWorkerCount << "\n";
  printedTree << "OPTION: cudaMinRowsForGpu = "
              << options.cudaMinRowsForGpu << "\n";

  std::ostringstream factorStr;
  factorStr << std::fixed << std::setprecision(4)
            << options.pruningConfidenceFactor;
  printedTree << "OPTION: pruningConfidenceFactor = " << factorStr.str() << "\n";

  std::ostringstream alphaStr;
  alphaStr << std::fixed << std::setprecision(4) << options.ccpAlpha;
  printedTree << "OPTION: ccpAlpha = " << alphaStr.str() << "\n";

  tree.print(printedTree);

  const std::string treeTextPath = "tree_visualization.txt";
  std::ofstream output(treeTextPath);
  if (!output) {
    throw std::runtime_error("Could not write visualization input file.");
  }

  output << printedTree.str();
  output.close();

  const std::string command =
      "python3 python/render_tree_svg.py " +
      shellEscapeSingleQuoted(treeTextPath) + " " +
      shellEscapeSingleQuoted(svgPath);

  const int status = std::system(command.c_str());
  std::remove(treeTextPath.c_str());
  if (status != 0) {
    throw std::runtime_error("Could not generate tree SVG.");
  }
}

inline void printSummary(const TreeBase &tree, const Dataset &dataset) {
  std::cout << "Summary:\n";

  std::size_t correctPredictions = 0;
  for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
    const Sample &sample = dataset.samples[index];
    const std::string prediction = tree.predict(sample);
    if (prediction == sample.label) {
      ++correctPredictions;
    }
  }

  const double accuracy = static_cast<double>(correctPredictions) /
                          static_cast<double>(dataset.samples.size());

  std::cout << "  checked samples = " << dataset.samples.size() << '\n';
  std::cout << "  correct predictions = " << correctPredictions << '\n';
  std::cout << "  tree depth = " << tree.treeDepth() << '\n';
  std::cout << "  node count = " << tree.nodeCount() << '\n';
  std::cout << "  accuracy = " << std::fixed << std::setprecision(4)
            << accuracy * 100.0 << "%\n";

  const FitTimings &timings = tree.fitTimings();
  std::cout << "Timings:\n";
  printTimingLine("load time", timings.loadSeconds);
  printTimingLine("gpu alloc time", timings.gpuAllocSeconds);
  printTimingLine("gpu upload time", timings.gpuUploadSeconds);
  printTimingLine("build time", timings.buildSeconds);
  printTimingLine("prune time", timings.pruneSeconds);
  printTimingLine("end-to-end total", timings.endToEndSeconds());
}

inline void printDatasetSummary(const Dataset &dataset) {
  std::cout << "Loaded samples: " << dataset.samples.size() << '\n';
  std::cout << "Features:";
  for (const std::string &featureName : dataset.featureNames) {
    std::cout << ' ' << featureName;
  }
  std::cout << "\n\n";

  if (!dataset.samples.empty()) {
    const Sample &firstSample = dataset.samples.front();
    std::cout << "First sample: ";
    for (std::size_t index = 0; index < firstSample.features.size(); ++index) {
      std::cout << dataset.featureNames[index] << '='
                << firstSample.features[index] << ' ';
    }
    std::cout << "label=" << firstSample.label << "\n\n";
  }
}
