#include "c45_tree.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
std::string indent(int depth) {
    return std::string(static_cast<std::size_t>(depth) * 2, ' ');
}

}  // namespace

void C45Tree::fit(const Dataset& dataset, const TrainingOptions& options) {
    if (dataset.samples.empty()) {
        throw std::runtime_error("Cannot train on an empty dataset.");
    }

    // Save where the training data is.
    // We use a pointer so we do not copy the whole dataset.
    dataset_ = &dataset;
    options_ = options;

    // At the start, the root sees all rows.
    std::vector<std::size_t> rowIndices(dataset.samples.size());
    for (std::size_t i = 0; i < rowIndices.size(); ++i) {
        rowIndices[i] = i;
    }

    // Build the whole tree recursively.
    root_ = buildNode(rowIndices, 0);
}

std::string C45Tree::predict(const Sample& sample) const {
    if (!root_) {
        throw std::runtime_error("Cannot predict before training.");
    }

    // Start at the root and keep answering the node's question until
    // we reach a leaf.
    const Node* current = root_.get();

    while (!current->isLeaf) {
        const double value = sample.features[current->featureIndex];

        if (value <= current->threshold) {
            current = current->leftChild.get();
        } else {
            current = current->rightChild.get();
        }
    }

    return current->leafLabel;
}

void C45Tree::print(std::ostream& output) const {
    if (!root_) {
        output << "Tree is empty.\n";
        return;
    }

    printNode(root_.get(), output, 0, "ROOT");
}

double C45Tree::entropy(const std::vector<std::size_t>& rowIndices) const {
    // THEORY:
    // Entropy tells us how mixed the classes are.
    //
    // If all flowers in this group are the same species:
    // entropy = 0
    //
    // If the species are mixed:
    // entropy is bigger
    //
    // So a "good" split is one that creates children with lower entropy.

    std::map<std::string, int> counts;
    for (std::size_t rowIndex : rowIndices) {
        counts[dataset_->samples[rowIndex].label]++;
    }

    double result = 0.0;
    const double total = static_cast<double>(rowIndices.size());

    for (const auto& entry : counts) {
        const double probability = static_cast<double>(entry.second) / total;

        // log2 is the base-2 logarithm.
        // The formula is:
        // H(S) = -sum(p * log2(p))
        if (probability > 0.0) {
            result -= probability * std::log2(probability);
        }
    }

    return result;
}

double C45Tree::giniIndex(const std::vector<std::size_t>& rowIndices) const {
    // THEORY:
    // Gini index is another impurity measure used by decision trees.
    //
    // If one class completely dominates the node:
    // gini = 0
    //
    // If classes are mixed:
    // gini becomes larger
    //
    // Formula:
    // Gini(S) = 1 - sum(p^2)
    //
    // Intuition:
    // If we pick one sample at random, p^2 measures how likely it is
    // to belong to a specific class twice in a row.
    // Summing those probabilities tells us how "pure" the node already is.

    std::map<std::string, int> counts;
    for (std::size_t rowIndex : rowIndices) {
        counts[dataset_->samples[rowIndex].label]++;
    }

    const double total = static_cast<double>(rowIndices.size());
    double sumOfSquaredProbabilities = 0.0;

    for (const auto& entry : counts) {
        const double probability = static_cast<double>(entry.second) / total;
        sumOfSquaredProbabilities += probability * probability;
    }

    return 1.0 - sumOfSquaredProbabilities;
}

double C45Tree::impurity(const std::vector<std::size_t>& rowIndices) const {
    // This wrapper lets the rest of the code ask for "the configured impurity"
    // without caring whether training was set to entropy or Gini.
    if (options_.impurityMeasure == ImpurityMeasure::Gini) {
        return giniIndex(rowIndices);
    }

    return entropy(rowIndices);
}

double C45Tree::informationGain(
    const std::vector<std::size_t>& rowIndices,
    const std::vector<std::vector<std::size_t>>& partitions
) const {
    // THEORY:
    // Information gain = impurity before split - impurity after split
    //
    // If the split makes the data much cleaner, gain is large.
    //
    // In classic C4.5 the impurity is entropy.
    // Here we make that part configurable so the same tree code can also
    // work with Gini index.

    const double beforeSplit = impurity(rowIndices);
    const double total = static_cast<double>(rowIndices.size());

    double afterSplit = 0.0;

    for (const std::vector<std::size_t>& part : partitions) {
        if (part.empty()) {
            continue;
        }

        const double weight = static_cast<double>(part.size()) / total;
        afterSplit += weight * impurity(part);
    }

    return beforeSplit - afterSplit;
}

double C45Tree::splitInformation(
    const std::vector<std::size_t>& rowIndices,
    const std::vector<std::vector<std::size_t>>& partitions
) const {

    const double total = static_cast<double>(rowIndices.size());
    double result = 0.0;

    for (const std::vector<std::size_t>& part : partitions) {
        if (part.empty()) {
            continue;
        }

        const double probability = static_cast<double>(part.size()) / total;
        result -= probability * std::log2(probability);
    }

    return result;
}

SplitResult C45Tree::findBestSplit(const std::vector<std::size_t>& rowIndices) const {
    // THEORY:
    // For the Iris dataset, every feature is numeric.
    //
    // That means every question looks like:
    // "Is this feature <= some threshold?"
    //
    // We try many candidate thresholds and keep the one with the best gain ratio.

    SplitResult best;

    // For each feature
    for (std::size_t featureIndex = 0; featureIndex < dataset_->featureNames.size(); ++featureIndex) {
        // Each pair stores:
        //   first  = feature value
        //   second = class label
        //
        // C++ note:
        // std::pair is just a small object that holds two values together.
        std::vector<std::pair<double, std::string>> values;

        for (std::size_t rowIndex : rowIndices) {
            values.push_back({
                dataset_->samples[rowIndex].features[featureIndex],
                dataset_->samples[rowIndex].label
            });
        }

        // Sort by feature value so neighboring values can be inspected.
        std::sort(values.begin(), values.end());

        // For each sample
        for (std::size_t i = 1; i < values.size(); ++i) {
            const double leftValue = values[i - 1].first;
            const double rightValue = values[i].first;
            const std::string& leftLabel = values[i - 1].second;
            const std::string& rightLabel = values[i].second;

            // If values are equal, there is no room for a new threshold.
            if (std::fabs(leftValue - rightValue) < options_.epsilon) {
                continue;
            }

            // A useful candidate usually appears where the class changes.
            if (leftLabel == rightLabel) {
                continue;
            }

            // Example:
            // if values are 1.4 and 1.5, try threshold 1.45
            const double threshold = (leftValue + rightValue) / 2.0;

            std::vector<std::size_t> leftRows;
            std::vector<std::size_t> rightRows;

            for (std::size_t rowIndex : rowIndices) {
                const double value = dataset_->samples[rowIndex].features[featureIndex];
                if (value <= threshold) {
                    leftRows.push_back(rowIndex);
                } else {
                    rightRows.push_back(rowIndex);
                }
            }

            if (leftRows.empty() || rightRows.empty()) {
                continue;
            }

            const std::vector<std::vector<std::size_t>> partitions = {leftRows, rightRows};
            const double gain = informationGain(rowIndices, partitions);
            const double splitInfo = splitInformation(rowIndices, partitions);

            if (splitInfo <= options_.epsilon) {
                continue;
            }

            const double ratio = gain / splitInfo;

            if (!best.valid || ratio > best.gainRatio) {
                best.valid = true;
                best.featureIndex = featureIndex;
                best.featureName = dataset_->featureNames[featureIndex];
                best.threshold = threshold;
                best.informationGain = gain;
                best.splitInformation = splitInfo;
                best.gainRatio = ratio;
            }
        }
    }

    return best;
}

std::unique_ptr<Node> C45Tree::buildNode(const std::vector<std::size_t>& rowIndices, int depth) const {
    // THEORY:
    // Building the tree is recursive.
    //
    // That means the function calls itself on smaller and smaller groups.
    // Each call builds one node.

    // If all rows already belong to one class, we are done.
    if (allSameLabel(rowIndices)) {
        return Node::createLeaf(dataset_->samples[rowIndices.front()].label);
    }

    // Simple stopping rules to keep the example manageable.
    if (depth >= options_.maxDepth || rowIndices.size() < options_.minSamplesToSplit) {
        return Node::createLeaf(getMajorityLabel(rowIndices));
    }

    const SplitResult split = findBestSplit(rowIndices);
    if (!split.valid || split.gainRatio <= options_.epsilon) {
        return Node::createLeaf(getMajorityLabel(rowIndices));
    }

    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;

    for (std::size_t rowIndex : rowIndices) {
        const double value = dataset_->samples[rowIndex].features[split.featureIndex];
        if (value <= split.threshold) {
            leftRows.push_back(rowIndex);
        } else {
            rightRows.push_back(rowIndex);
        }
    }

    // Stop if split created empty group - no progress made
    if (leftRows.empty() || rightRows.empty()) {
        return Node::createLeaf(getMajorityLabel(rowIndices));
    }

    // Create a decision node:
    // "Is featureName <= threshold?"
    std::unique_ptr<Node> node = Node::createDecision(
        split.featureName,
        split.featureIndex,
        split.threshold
    );

    // Recursively build the two children.
    node->leftChild = buildNode(leftRows, depth + 1);
    node->rightChild = buildNode(rightRows, depth + 1);

    return node;
}

bool C45Tree::allSameLabel(const std::vector<std::size_t>& rowIndices) const {
    const std::string& firstLabel = dataset_->samples[rowIndices.front()].label;

    for (std::size_t rowIndex : rowIndices) {
        if (dataset_->samples[rowIndex].label != firstLabel) {
            return false;
        }
    }

    return true;
}

std::string C45Tree::getMajorityLabel(const std::vector<std::size_t>& rowIndices) const {
    std::map<std::string, int> counts;
    for (std::size_t rowIndex : rowIndices) {
        counts[dataset_->samples[rowIndex].label]++;
    }

    std::string bestLabel;
    int maxCount = -1;

    for (const auto& [label, count] : counts) {
        if (count > maxCount) {
            maxCount = count;
            bestLabel = label;
        }
    }
    return bestLabel;
}

void C45Tree::printNode(
    const Node* node,
    std::ostream& output,
    int depth,
    const std::string& edgeText
) const {
    output << indent(depth) << edgeText << ": ";

    if (node->isLeaf) {
        output << "Leaf -> " << node->leafLabel << '\n';
        return;
    }

    output << "if " << node->featureName << " <= "
           << std::fixed << std::setprecision(3) << node->threshold << '\n';

    if (node->leftChild) {
        printNode(node->leftChild.get(), output, depth + 1, "yes");
    }

    if (node->rightChild) {
        printNode(node->rightChild.get(), output, depth + 1, "no");
    }
}
