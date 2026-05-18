#pragma once

#include "c45_tree.h"
#include "dataset.h"

#include <string>

void generateTreeSvg(const C45Tree &tree, const std::string &svgPath,
                     const TrainingOptions &options,
                     const Dataset &dataset);
void printDatasetSummary(const Dataset &dataset);
