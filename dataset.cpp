#include "dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Very small CSV splitter used for this learning project.
// It is intentionally simple because the Iris dataset does not contain
// quoted commas or other advanced CSV features.
std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string cell;

    while (std::getline(stream, cell, ',')) {
        parts.push_back(cell);
    }

    return parts;
}

}  // namespace

Dataset loadDataset(const std::string& filePath) {
    std::ifstream input(filePath);
    if (!input) {
        throw std::runtime_error("Could not open dataset file: " + filePath);
    }

    Dataset dataset;
    std::string line;

    if (!std::getline(input, line)) {
        throw std::runtime_error("Dataset file is empty: " + filePath);
    }

    const std::vector<std::string> header = splitCsvLine(line);
    if (header.size() < 3) {
        throw std::runtime_error("Unexpected CSV header format in: " + filePath);
    }

    // Header layout is:
    // Id, feature1, feature2, feature3, feature4, label
    // The first column is only a row id, so we skip it.
    for (std::size_t index = 1; index + 1 < header.size(); ++index) {
        dataset.featureNames.push_back(header[index]);
    }

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> cells = splitCsvLine(line);
        if (cells.size() != header.size()) {
            throw std::runtime_error("Malformed CSV row: " + line);
        }

        Sample sample;
        // Each flower measurement is parsed as a numeric feature.
        // This makes it possible for C4.5 to search for thresholds like
        // "PetalLengthCm <= 2.45".
        for (std::size_t index = 1; index + 1 < cells.size(); ++index) {
            sample.features.push_back(std::stod(cells[index]));
        }

        // The last column is the species we want the model to predict.
        sample.label = cells.back();
        dataset.samples.push_back(sample);
    }

    return dataset;
}
