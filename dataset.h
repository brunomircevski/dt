#pragma once

#include <string>
#include <vector>

struct Sample {
    // All numeric measurements of one flower.
    // For Iris this means:
    // sepal length, sepal width, petal length, petal width
    std::vector<double> features;

    // Correct class/species of this flower.
    std::string label;
};

struct Dataset {
    // Names of the columns in the same order as Sample::features.
    std::vector<std::string> featureNames;

    // All rows from the CSV file.
    std::vector<Sample> samples;
};

Dataset loadDataset(const std::string& filePath);
