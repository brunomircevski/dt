#include "c45_tree.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace
{
	std::string indent(int depth)
	{
		return std::string(static_cast<std::size_t>(depth) * 2, ' ');
	}

	bool areEffectivelyEqual(double left, double right, double epsilon)
	{
		return std::fabs(left - right) <= epsilon;
	}

} // namespace

void C45Tree::fit(const Dataset &dataset, const TrainingOptions &options)
{
	if (dataset.samples.empty())
	{
		throw std::runtime_error("Cannot train on an empty dataset.");
	}

	// Save where the training data is.
	// We use a pointer so we do not copy the whole dataset.
	dataset_ = &dataset;
	options_ = options;

	// At the start, the root sees all rows.
	std::vector<std::size_t> rowIndices(dataset.samples.size());
	for (std::size_t i = 0; i < rowIndices.size(); ++i)
	{
		rowIndices[i] = i;
	}

	// Build the whole tree recursively.
	root_ = buildNode(rowIndices, 0);

	// Post-pruning is a second phase:
	// first let the tree grow, then remove branches that do not improve
	// classification on the samples that reached them.
	if (options_.pruningMode != PruningMode::None)
	{
		applySelectedPruning(rowIndices);
	}
}

std::string C45Tree::predict(const Sample &sample) const
{
	if (!root_)
	{
		throw std::runtime_error("Cannot predict before training.");
	}

	// Start at the root and keep answering the node's question until
	// we reach a leaf.
	const Node *current = root_.get();

	while (!current->isLeaf)
	{
		const double value = sample.features[current->featureIndex];

		if (value <= current->threshold)
		{
			current = current->leftChild.get();
		}
		else
		{
			current = current->rightChild.get();
		}
	}

	return current->leafLabel;
}

void C45Tree::print(std::ostream &output) const
{
	if (!root_)
	{
		output << "Tree is empty.\n";
		return;
	}

	printNode(root_.get(), output, 0, "ROOT");
}

int C45Tree::treeDepth() const { return computeTreeDepth(root_.get()); }

std::size_t C45Tree::nodeCount() const { return countNodes(root_.get()); }

std::map<std::string, int>
C45Tree::computeClassCounts(const std::vector<std::size_t> &rowIndices) const
{
	// THEORY:
	// Many later calculations need to know how many training rows of each
	// class reached the current node. We compute those counts once here so
	// helpers like entropy() and getMajorityLabel() can reuse the same idea.
	std::map<std::string, int> counts;
	for (std::size_t rowIndex : rowIndices)
	{
		counts[dataset_->samples[rowIndex].label]++;
	}
	return counts;
}

double C45Tree::entropy(const std::vector<std::size_t> &rowIndices) const
{
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

	const std::map<std::string, int> counts = computeClassCounts(rowIndices);

	double result = 0.0;
	const double total = static_cast<double>(rowIndices.size());

	for (const auto &entry : counts)
	{
		const double probability = static_cast<double>(entry.second) / total;

		// log2 is the base-2 logarithm.
		// The formula is:
		// H(S) = -sum(p * log2(p))
		if (probability > 0.0)
		{
			result -= probability * std::log2(probability);
		}
	}

	return result;
}

double C45Tree::giniIndex(const std::vector<std::size_t> &rowIndices) const
{
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

	const std::map<std::string, int> counts = computeClassCounts(rowIndices);

	const double total = static_cast<double>(rowIndices.size());
	double sumOfSquaredProbabilities = 0.0;

	for (const auto &entry : counts)
	{
		const double probability = static_cast<double>(entry.second) / total;
		sumOfSquaredProbabilities += probability * probability;
	}

	return 1.0 - sumOfSquaredProbabilities;
}

double C45Tree::impurity(const std::vector<std::size_t> &rowIndices) const
{
	// This wrapper lets the rest of the code ask for "the configured impurity"
	// without caring whether training was set to entropy or Gini.
	if (options_.impurityMeasure == ImpurityMeasure::Gini)
	{
		return giniIndex(rowIndices);
	}

	return entropy(rowIndices);
}

double C45Tree::informationGain(
	const std::vector<std::size_t> &rowIndices,
	const std::vector<std::vector<std::size_t>> &partitions) const
{
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

	for (const std::vector<std::size_t> &part : partitions)
	{
		if (part.empty())
		{
			continue;
		}

		const double weight = static_cast<double>(part.size()) / total;
		afterSplit += weight * impurity(part);
	}

	return beforeSplit - afterSplit;
}

double C45Tree::splitInformation(
	const std::vector<std::size_t> &rowIndices,
	const std::vector<std::vector<std::size_t>> &partitions) const
{

	const double total = static_cast<double>(rowIndices.size());
	double result = 0.0;

	for (const std::vector<std::size_t> &part : partitions)
	{
		if (part.empty())
		{
			continue;
		}

		const double probability = static_cast<double>(part.size()) / total;
		result -= probability * std::log2(probability);
	}

	return result;
}

C45Tree::PartitionedRows
C45Tree::partitionRows(const std::vector<std::size_t> &rowIndices,
					   std::size_t featureIndex, double threshold) const
{
	// THEORY:
	// A numeric binary tree node asks exactly one question:
	// "Is feature <= threshold?"
	// This helper executes that question for every row that reached the node
	// and returns the two child groups.
	PartitionedRows partitions;
	for (std::size_t rowIndex : rowIndices)
	{
		const double value = dataset_->samples[rowIndex].features[featureIndex];
		if (value <= threshold)
		{
			partitions.leftRows.push_back(rowIndex);
		}
		else
		{
			partitions.rightRows.push_back(rowIndex);
		}
	}
	return partitions;
}

std::vector<double> C45Tree::collectNumericThresholdCandidates(
	const std::vector<std::size_t> &rowIndices,
	std::size_t featureIndex) const
{
	// THEORY:
	// For numeric C4.5 we sort the rows by one feature and only test
	// thresholds between adjacent distinct values where the class label
	// changes. That keeps the search small without missing useful binary
	// thresholds.
	std::vector<std::pair<double, std::string>> values;
	values.reserve(rowIndices.size());

	for (std::size_t rowIndex : rowIndices)
	{
		values.push_back({dataset_->samples[rowIndex].features[featureIndex],
						  dataset_->samples[rowIndex].label});
	}

	std::sort(values.begin(), values.end());

	std::vector<double> thresholds;
	for (std::size_t index = 1; index < values.size(); ++index)
	{
		const double leftValue = values[index - 1].first;
		const double rightValue = values[index].first;
		const std::string &leftLabel = values[index - 1].second;
		const std::string &rightLabel = values[index].second;

		if (std::fabs(leftValue - rightValue) < options_.epsilon)
		{
			continue;
		}

		if (leftLabel == rightLabel)
		{
			continue;
		}

		thresholds.push_back((leftValue + rightValue) / 2.0);
	}

	return thresholds;
}

SplitResult C45Tree::scoreSplit(const std::vector<std::size_t> &rowIndices,
								std::size_t featureIndex,
								double threshold) const
{
	// THEORY:
	// After proposing one threshold, we score it exactly the way C4.5 does
	// for numeric binary tests:
	//   information gain = entropy(parent) - weighted entropy(children)
	//   split information = entropy of the split sizes
	//   gain ratio = information gain / split information
	SplitResult result;
	result.featureIndex = featureIndex;
	result.featureName = dataset_->featureNames[featureIndex];
	result.threshold = threshold;

	const PartitionedRows partitions =
		partitionRows(rowIndices, featureIndex, threshold);
	if (partitions.leftRows.empty() || partitions.rightRows.empty())
	{
		return result;
	}

	if (options_.minSamplesPerLeaf != 0)
	{
		if (partitions.leftRows.size() < options_.minSamplesPerLeaf ||
			partitions.rightRows.size() < options_.minSamplesPerLeaf)
		{
			return result;
		}
	}

	const std::vector<std::vector<std::size_t>> groups = {partitions.leftRows,
														  partitions.rightRows};
	result.informationGain = informationGain(rowIndices, groups);

	if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
		// For pure gain mode (CART), we don't need split information.
		result.splitInformation = 0.0;
		result.gainRatio = 0.0;
		result.valid = (result.informationGain > options_.epsilon);
		return result;
	}

	result.splitInformation = splitInformation(rowIndices, groups);

	if (result.informationGain <= options_.epsilon ||
		result.splitInformation <= options_.epsilon)
	{
		return result;
	}

	result.gainRatio = result.informationGain / result.splitInformation;
	result.valid = true;
	return result;
}

SplitResult
C45Tree::chooseBestSplit(const std::vector<SplitResult> &candidates) const
{
	if (candidates.empty())
	{
		return SplitResult{};
	}

	std::vector<const SplitResult *> selectableCandidates;
	selectableCandidates.reserve(candidates.size());

	if (options_.splitSelectionMode == SplitSelectionMode::MeanGainFiltered)
	{
		// compare gain ratios only among candidates with at least average gain.
		double gainSum = 0.0;
		for (const SplitResult &candidate : candidates)
		{
			gainSum += candidate.informationGain;
		}
		const double averageGain = gainSum / static_cast<double>(candidates.size());

		for (const SplitResult &candidate : candidates)
		{
			if (candidate.informationGain + options_.epsilon >= averageGain)
			{
				selectableCandidates.push_back(&candidate);
			}
		}
	}
	else if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
		// Pure maximization of gain (Gini or Information Gain).
		const SplitResult *bestCandidate = &candidates.front();
		for (std::size_t index = 1; index < candidates.size(); ++index)
		{
			const SplitResult *candidate = &candidates[index];
			if (candidate->informationGain >
				bestCandidate->informationGain + options_.epsilon)
			{
				bestCandidate = candidate;
			}
			//Jeśli są takie same wybierz mniejszy index lub niższy threshold
			else if (areEffectivelyEqual(candidate->informationGain,
										 bestCandidate->informationGain,
										 options_.epsilon))
			{
				if (candidate->featureIndex < bestCandidate->featureIndex)
				{
					bestCandidate = candidate;
				}
				else if (candidate->featureIndex == bestCandidate->featureIndex &&
						 candidate->threshold <
							 bestCandidate->threshold - options_.epsilon)
				{
					bestCandidate = candidate;
				}
			}
		}
		return *bestCandidate;
	}
	else if (options_.splitSelectionMode == SplitSelectionMode::ClassicC45)
	{
		// Classic C4.5-style simplified rule for this project:
		// every candidate that has real positive information gain is eligible,
		// and the final choice is made by gain ratio.
		for (const SplitResult &candidate : candidates)
		{
			selectableCandidates.push_back(&candidate);
		}
	}

	if (selectableCandidates.empty())
	{
		return SplitResult{};
	}

	const SplitResult *bestCandidate = selectableCandidates.front();
	for (std::size_t index = 1; index < selectableCandidates.size(); ++index)
	{
		const SplitResult *candidate = selectableCandidates[index];

		if (candidate->gainRatio > bestCandidate->gainRatio + options_.epsilon)
		{
			bestCandidate = candidate;
			continue;
		}

		if (!areEffectivelyEqual(candidate->gainRatio, bestCandidate->gainRatio,
								 options_.epsilon))
		{
			continue;
		}

		if (candidate->informationGain >
			bestCandidate->informationGain + options_.epsilon)
		{
			bestCandidate = candidate;
			continue;
		}

		if (!areEffectivelyEqual(candidate->informationGain,
								 bestCandidate->informationGain,
								 options_.epsilon))
		{
			continue;
		}

		if (candidate->featureIndex < bestCandidate->featureIndex)
		{
			bestCandidate = candidate;
			continue;
		}

		if (candidate->featureIndex == bestCandidate->featureIndex &&
			candidate->threshold < bestCandidate->threshold - options_.epsilon)
		{
			bestCandidate = candidate;
		}
	}

	return *bestCandidate;
}

SplitResult
C45Tree::findBestSplit(const std::vector<std::size_t> &rowIndices) const
{
	// THEORY:
	// For numeric-only datasets, every internal node asks one binary question:
	// "Is feature <= threshold?"
	// We generate candidate thresholds, score them, then choose the best one
	// using the configured split-selection policy.
	std::vector<SplitResult> scoredCandidates;

	for (std::size_t featureIndex = 0;
		 featureIndex < dataset_->featureNames.size(); ++featureIndex)
	{
		const std::vector<double> thresholds =
			collectNumericThresholdCandidates(rowIndices, featureIndex);

		for (double threshold : thresholds)
		{
			SplitResult candidate = scoreSplit(rowIndices, featureIndex, threshold);
			if (candidate.valid)
			{
				scoredCandidates.push_back(candidate);
			}
		}
	}

	return chooseBestSplit(scoredCandidates);
}

bool C45Tree::shouldStopGrowing(const std::vector<std::size_t> &rowIndices,
								int depth) const
{
	// THEORY:
	// A decision tree stops growing when extra splits are no longer useful.
	// For this numeric-only learning version we use simple, explicit rules:
	// pure node, too few samples, or optional depth limit reached.
	if (allSameLabel(rowIndices))
	{
		return true;
	}

	if (rowIndices.size() < options_.minSamplesToSplit)
	{
		return true;
	}

	if (options_.maxDepth >= 0 && depth >= options_.maxDepth)
	{
		return true;
	}

	return false;
}

std::unique_ptr<Node>
C45Tree::buildNode(const std::vector<std::size_t> &rowIndices,
				   int depth) const
{
	// THEORY:
	// Building the tree is recursive.
	//
	// That means the function calls itself on smaller and smaller groups.
	// Each call builds one node.

	// If all rows already belong to one class, we are done.
	if (allSameLabel(rowIndices))
	{
		return Node::createLeaf(dataset_->samples[rowIndices.front()].label,
								rowIndices.size());
	}

	if (shouldStopGrowing(rowIndices, depth))
	{
		return Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size());
	}

	const SplitResult split = findBestSplit(rowIndices);
	if (!split.valid || split.gainRatio <= options_.epsilon)
	{
		return Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size());
	}

	const PartitionedRows partitions =
		partitionRows(rowIndices, split.featureIndex, split.threshold);

	// Stop if split created empty group - no progress made
	if (partitions.leftRows.empty() || partitions.rightRows.empty())
	{
		return Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size());
	}

	// Create a decision node:
	// "Is featureName <= threshold?"
	std::unique_ptr<Node> node =
		Node::createDecision(split.featureName, split.featureIndex,
							 split.threshold, rowIndices.size());

	// Recursively build the two children.
	node->leftChild = buildNode(partitions.leftRows, depth + 1);
	node->rightChild = buildNode(partitions.rightRows, depth + 1);

	return node;
}

std::size_t C45Tree::countCorrectPredictions(
	const Node *node, const std::vector<std::size_t> &rowIndices) const
{
	std::size_t correct = 0;

	for (std::size_t rowIndex : rowIndices)
	{
		const Sample &sample = dataset_->samples[rowIndex];
		const Node *current = node;

		while (!current->isLeaf)
		{
			if (sample.features[current->featureIndex] <= current->threshold)
			{
				current = current->leftChild.get();
			}
			else
			{
				current = current->rightChild.get();
			}
		}

		if (current->leafLabel == sample.label)
		{
			++correct;
		}
	}

	return correct;
}

std::size_t C45Tree::countLeafNodes(const Node *node) const
{
	if (!node)
	{
		return 0;
	}

	if (node->isLeaf)
	{
		return 1;
	}

	return countLeafNodes(node->leftChild.get()) +
		   countLeafNodes(node->rightChild.get());
}

std::size_t C45Tree::countNodes(const Node *node) const
{
	if (!node)
	{
		return 0;
	}

	return 1 + countNodes(node->leftChild.get()) +
		   countNodes(node->rightChild.get());
}

int C45Tree::computeTreeDepth(const Node *node) const
{
	// A leaf has depth 0. A decision node has depth 1 plus the deeper child.
	if (!node)
	{
		return -1;
	}

	if (node->isLeaf)
	{
		return 0;
	}

	const int leftDepth = computeTreeDepth(node->leftChild.get());
	const int rightDepth = computeTreeDepth(node->rightChild.get());
	return 1 + std::max(leftDepth, rightDepth);
}

bool C45Tree::allSameLabel(const std::vector<std::size_t> &rowIndices) const
{
	const std::string &firstLabel = dataset_->samples[rowIndices.front()].label;

	for (std::size_t rowIndex : rowIndices)
	{
		if (dataset_->samples[rowIndex].label != firstLabel)
		{
			return false;
		}
	}

	return true;
}

std::string
C45Tree::getMajorityLabel(const std::vector<std::size_t> &rowIndices) const
{
	const std::map<std::string, int> counts = computeClassCounts(rowIndices);

	std::string bestLabel;
	int maxCount = -1;

	for (const auto &[label, count] : counts)
	{
		if (count > maxCount)
		{
			maxCount = count;
			bestLabel = label;
		}
	}
	return bestLabel;
}

void C45Tree::printNode(const Node *node, std::ostream &output, int depth,
						const std::string &edgeText) const
{
	output << indent(depth) << edgeText << " [n=" << node->sampleCount << "]: ";

	if (node->isLeaf)
	{
		output << "Leaf -> " << node->leafLabel << '\n';
		return;
	}

	output << "if " << node->featureName << " <= " << std::fixed
		   << std::setprecision(3) << node->threshold << '\n';

	if (node->leftChild)
	{
		printNode(node->leftChild.get(), output, depth + 1, "yes");
	}

	if (node->rightChild)
	{
		printNode(node->rightChild.get(), output, depth + 1, "no");
	}
}
