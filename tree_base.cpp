// Shared decision-tree logic (impurity, splits, serial build, predict, print).

#include "tree_base.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
	std::string indent(int depth)
	{
		return std::string(static_cast<std::size_t>(depth) * 2, ' ');
	}
} // namespace

bool TreeBase::areEffectivelyEqual(double left, double right, double epsilon) {
  // Floating point numbers are not exact, so compare them with a small margin.
  return std::fabs(left - right) <= epsilon;
}

bool TreeBase::isBetterC45(const SplitResult &lhs, const SplitResult &rhs,
                           double epsilon) {
  // C4.5 mainly chooses the largest gain ratio.
  // The extra checks make ties stable and repeatable.
  if (lhs.gainRatio > rhs.gainRatio + epsilon) return true;
  if (rhs.gainRatio > lhs.gainRatio + epsilon) return false;
  if (lhs.informationGain > rhs.informationGain + epsilon) return true;
  if (rhs.informationGain > lhs.informationGain + epsilon) return false;
  if (lhs.featureIndex != rhs.featureIndex) {
    return lhs.featureIndex < rhs.featureIndex;
  }
  return lhs.threshold < rhs.threshold - epsilon;
}

bool TreeBase::isBetterMaxGain(const SplitResult &lhs, const SplitResult &rhs,
                               double epsilon) {
  // CART-style selection: choose the split that reduces impurity the most.
  // If scores are equal, prefer the earlier feature and lower threshold.
  if (lhs.informationGain > rhs.informationGain + epsilon) return true;
  if (rhs.informationGain > lhs.informationGain + epsilon) return false;
  if (lhs.featureIndex != rhs.featureIndex) {
    return lhs.featureIndex < rhs.featureIndex;
  }
  return lhs.threshold < rhs.threshold - epsilon;
}

void TreeBase::prepareFit(const Dataset &dataset, const TrainingOptions &options) {
  // Common setup for all backends. Serial, Parallel, and future CUDA all need
  // the same data pointer, options, and label mapping before training starts.
  if (dataset.samples.empty()) {
    throw std::runtime_error("Cannot train on an empty dataset.");
  }

  dataset_ = &dataset;
  options_ = options;
  buildClassMapping(dataset);
  buildTimeSeconds_ = 0.0;
  pruneTimeSeconds_ = 0.0;
}

TreeBase::BuildTimePoint TreeBase::startBuildTimer() const {
  return std::chrono::steady_clock::now();
}

void TreeBase::finishBuildTimer(BuildTimePoint start) {
  buildTimeSeconds_ = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start)
                          .count();
}

void TreeBase::finalizeFit(const TrainingOptions &options) { prune(options); }

std::vector<std::size_t> TreeBase::makeRootRowIndices() const {
  // The tree stores row numbers, not copies of samples. This keeps splits cheap:
  // children only pass around lists of indices into dataset_->samples.
  std::vector<std::size_t> rowIndices(dataset_->samples.size());
  for (std::size_t i = 0; i < rowIndices.size(); ++i) {
    rowIndices[i] = i;
  }
  return rowIndices;
}

std::string TreeBase::predict(const Sample &sample) const
{
	if (!root_)
	{
		throw std::runtime_error("Cannot predict before training.");
	}

	// Prediction is just walking the tree: at each decision node, go left or right
	// until we hit a leaf; the leaf's label is our answer. No threading needed here.
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

void TreeBase::print(std::ostream &output) const
{
	if (!root_)
	{
		output << "Tree is empty.\n";
		return;
	}

	printNode(root_.get(), output, 0, "ROOT");
}

int TreeBase::treeDepth() const { return computeTreeDepth(root_.get()); }

std::size_t TreeBase::nodeCount() const { return countNodes(root_.get()); }

void TreeBase::buildClassMapping(const Dataset &dataset)
{
	// Convert string labels ("setosa", "versicolor", ...) into small integer IDs.
	// Integer class IDs make counting and entropy math faster than comparing strings.
	classLabels_.clear();
	labelToClassId_.clear();
	numClasses_ = 0;

	std::vector<std::string> uniqueLabels;
	uniqueLabels.reserve(4);
	
	for (const Sample &sample : dataset.samples)
	{
		if (std::find(uniqueLabels.begin(), uniqueLabels.end(), sample.label) ==
			uniqueLabels.end())
		{
			uniqueLabels.push_back(sample.label);
		}
	}

	std::sort(uniqueLabels.begin(), uniqueLabels.end());
	classLabels_ = uniqueLabels;
	numClasses_ = static_cast<std::uint16_t>(classLabels_.size());
	for (std::uint16_t classIndex = 0; classIndex < numClasses_; ++classIndex)
	{
		labelToClassId_[classLabels_[classIndex]] = classIndex;
	}
}

std::uint16_t TreeBase::classIdForRow(std::size_t rowIndex) const
{
	// Fast lookup: row index → numeric class id (built once in buildClassMapping).
	return labelToClassId_.at(dataset_->samples[rowIndex].label);
}

std::vector<std::uint32_t> TreeBase::computeClassCountsArray(
	const std::vector<std::size_t> &rowIndices) const
{
	// How many rows of each class reach this node? Used by entropy, Gini, majority vote.
	std::vector<std::uint32_t> counts(numClasses_, 0);
	for (std::size_t rowIndex : rowIndices)
	{
		++counts[classIdForRow(rowIndex)];
	}
	return counts;
}

double TreeBase::entropy(const std::vector<std::size_t> &rowIndices) const
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

	const std::vector<std::uint32_t> counts = computeClassCountsArray(rowIndices);

	double result = 0.0;
	const double total = static_cast<double>(rowIndices.size());

	for (std::uint32_t count : counts)
	{
		if (count == 0)
		{
			continue;
		}

		const double probability = static_cast<double>(count) / total;
		result -= probability * std::log2(probability);
	}

	return result;
}

double TreeBase::giniIndex(const std::vector<std::size_t> &rowIndices) const
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

	const std::vector<std::uint32_t> counts = computeClassCountsArray(rowIndices);

	const double total = static_cast<double>(rowIndices.size());
	double sumOfSquaredProbabilities = 0.0;

	for (std::uint32_t count : counts)
	{
		if (count == 0)
		{
			continue;
		}

		const double probability = static_cast<double>(count) / total;
		sumOfSquaredProbabilities += probability * probability;
	}

	return 1.0 - sumOfSquaredProbabilities;
}

double TreeBase::impurity(const std::vector<std::size_t> &rowIndices) const
{
	// This wrapper lets the rest of the code ask for "the configured impurity"
	// without caring whether training was set to entropy or Gini.
	if (options_.impurityMeasure == ImpurityMeasure::Gini)
	{
		return giniIndex(rowIndices);
	}

	return entropy(rowIndices);
}

double TreeBase::informationGain(
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

double TreeBase::splitInformation(
	const std::vector<std::size_t> &rowIndices,
	const std::vector<std::vector<std::size_t>> &partitions) const
{
	// C4.5 only: split information measures how balanced the child sizes are.
	//
	// Example: splitting 100 rows into 50|50 has high split info (balanced).
	// Splitting 99|1 has low split info — one child is almost empty.
	//
	// Gain ratio = information gain / split information.
	// This penalizes splits that are informative only because one side is tiny
	// (e.g. "petal length <= 0.1" might isolate one outlier but is not useful).

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

TreeBase::PartitionedRows
TreeBase::partitionRows(const std::vector<std::size_t> &rowIndices,
					   std::size_t featureIndex, double threshold) const
{
	// Fallback partitioner: scan every row and compare feature value to threshold.
	// Used when we do not have a sorted view cached from split search.
	// Question at each decision node: "Is feature <= threshold?"
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

double TreeBase::impurityFromCounts(const std::vector<std::uint32_t> &counts,
								   std::size_t total) const
{
	// Same math as entropy()/giniIndex(), but from pre-computed class counts.
	// During the threshold sweep we update counts incrementally instead of
	// rescanning all rows for every candidate threshold — much faster.
	if (total == 0)
	{
		return 0.0;
	}

	const double totalDouble = static_cast<double>(total);

	if (options_.impurityMeasure == ImpurityMeasure::Gini)
	{
		double sumOfSquaredProbabilities = 0.0;
		for (std::uint32_t count : counts)
		{
			if (count == 0)
			{
				continue;
			}

			const double probability = static_cast<double>(count) / totalDouble;
			sumOfSquaredProbabilities += probability * probability;
		}
		return 1.0 - sumOfSquaredProbabilities;
	}

	double result = 0.0;
	for (std::uint32_t count : counts)
	{
		if (count == 0)
		{
			continue;
		}

		const double probability = static_cast<double>(count) / totalDouble;
		result -= probability * std::log2(probability);
	}
	return result;
}

// Build a reusable sorted view for one feature at the current node.
//
// THEORY — why sort?
// For numeric features, the best threshold always lies between two sorted values.
// After sorting rows by feature value we can try every gap in one left-to-right sweep,
// updating class counts in O(1) per step instead of repartitioning all rows each time.
//
// Threshold rule: only consider gaps where adjacent rows have different labels.
// If two neighbors share the same class, moving the split between them cannot
// reduce impurity (both sides would still contain that class mix).
TreeBase::SortedFeatureView TreeBase::buildSortedFeatureView(
	const std::vector<std::size_t> &rowIndices, std::size_t featureIndex) const
{
	SortedFeatureView view;
	view.featureIndex = featureIndex;
	view.totalRows = rowIndices.size();
	view.rows.reserve(rowIndices.size());

	for (std::size_t rowIndex : rowIndices)
	{
		const Sample &sample = dataset_->samples[rowIndex];
		view.rows.push_back(
			{rowIndex, sample.features[featureIndex], classIdForRow(rowIndex)});
	}

	std::sort(view.rows.begin(), view.rows.end(),
			  [](const SortedFeatureRow &left, const SortedFeatureRow &right) {
				  if (left.value < right.value)
				  {
					  return true;
				  }
				  if (right.value < left.value)
				  {
					  return false;
				  }
				  return left.classId < right.classId;
			  });

	return view;
}

TreeBase::SplitResult TreeBase::scoreCandidateFromCounts(
	const SortedFeatureView &view, const std::vector<std::uint32_t> &totalCounts,
	double parentImpurity, const std::vector<std::uint32_t> &leftCounts,
	const std::vector<std::uint32_t> &rightCounts, std::size_t leftSize,
	double threshold) const
{
	(void)totalCounts;
	// Score one possible question:
	// "Is feature[featureIndex] <= threshold?"
	// Counts tell us how mixed the left and right children would be.
	SplitResult result;
	result.featureIndex = view.featureIndex;
	result.featureName = dataset_->featureNames[view.featureIndex];
	result.threshold = threshold;
	result.leftRowCount = leftSize;

	const std::size_t rightSize = view.totalRows - leftSize;
	// A split must actually divide the data — both sides need at least one row.
	if (leftSize == 0 || rightSize == 0)
	{
		return result;
	}

	if (options_.minSamplesPerLeaf != 0)
	{
		if (leftSize < options_.minSamplesPerLeaf ||
			rightSize < options_.minSamplesPerLeaf)
		{
			return result;
		}
	}

	const double leftImpurity = impurityFromCounts(leftCounts, leftSize);
	const double rightImpurity = impurityFromCounts(rightCounts, rightSize);
	const double totalDouble = static_cast<double>(view.totalRows);
	// Weighted average impurity of the two children (size-proportional).
	const double afterSplit =
		(static_cast<double>(leftSize) / totalDouble) * leftImpurity +
		(static_cast<double>(rightSize) / totalDouble) * rightImpurity;

	result.informationGain = parentImpurity - afterSplit;

	if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
		// CART: information gain alone decides; no gain ratio.
		result.splitInformation = 0.0;
		result.gainRatio = 0.0;
		result.valid = (result.informationGain > options_.epsilon);
		return result;
	}

	const double leftProbability = static_cast<double>(leftSize) / totalDouble;
	const double rightProbability = static_cast<double>(rightSize) / totalDouble;
	result.splitInformation =
		-leftProbability * std::log2(leftProbability) -
		rightProbability * std::log2(rightProbability);

	if (result.informationGain <= options_.epsilon ||
		result.splitInformation <= options_.epsilon)
	{
		return result;
	}

	result.gainRatio = result.informationGain / result.splitInformation;
	result.valid = true;
	return result;
}

TreeBase::PartitionedRows
TreeBase::partitionFromSortedView(const SortedFeatureView &view,
								 std::size_t leftSize) const
{
	// Because rows are already sorted by feature value, the first leftSize rows
	// are exactly the "left" partition — no need to re-compare thresholds.
	PartitionedRows partitions;
	partitions.leftRows.reserve(leftSize);
	partitions.rightRows.reserve(view.rows.size() - leftSize);

	for (std::size_t index = 0; index < leftSize; ++index)
	{
		partitions.leftRows.push_back(view.rows[index].rowIndex);
	}

	for (std::size_t index = leftSize; index < view.rows.size(); ++index)
	{
		partitions.rightRows.push_back(view.rows[index].rowIndex);
	}

	return partitions;
}

TreeBase::SplitResult
TreeBase::chooseBestSplit(const std::vector<SplitResult> &candidates) const
{
	// Input: one best candidate per feature.
	// Output: the single split we will actually use at this node.
	if (candidates.empty())
	{
		return SplitResult{};
	}

	// CART: pick the feature with the highest information gain (or Gini gain).
	if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
		const SplitResult *bestCandidate = &candidates.front();
		for (std::size_t index = 1; index < candidates.size(); ++index)
		{
			if (TreeBase::isBetterMaxGain(candidates[index], *bestCandidate, options_.epsilon))
			{
				bestCandidate = &candidates[index];
			}
		}
		return *bestCandidate;
	}

	// C4.5 pre-pruning at split time (not the same as post-pruning after the tree is built):
	// discard features whose gain is below the average gain at this node, then pick
	// the highest gain ratio among the survivors.
	double gainSum = 0.0;
	for (const SplitResult &candidate : candidates)
	{
		gainSum += candidate.informationGain;
	}
	const double averageGain = gainSum / static_cast<double>(candidates.size());

	const SplitResult *bestCandidate = nullptr;
	for (const SplitResult &candidate : candidates)
	{
		if (candidate.informationGain + options_.epsilon < averageGain)
		{
			continue;
		}

		if (!bestCandidate || TreeBase::isBetterC45(candidate, *bestCandidate, options_.epsilon))
		{
			bestCandidate = &candidate;
		}
	}

	return bestCandidate ? *bestCandidate : SplitResult{};
}


TreeBase::SplitResult TreeBase::scoreAllThresholdsForFeature(const SortedFeatureView &view) const
{
	// One left-to-right sweep over sorted rows.
	// leftCounts grows as we move the split point; rightCounts = total - left.
	// This avoids building two child row lists for every possible threshold.

	std::vector<std::uint32_t> totalCounts(numClasses_, 0);
	for (const SortedFeatureRow &row : view.rows)
	{
		++totalCounts[row.classId];
	}

	const double parentImpurity = impurityFromCounts(totalCounts, view.totalRows);
	std::vector<std::uint32_t> leftCounts(numClasses_, 0);
	std::vector<std::uint32_t> rightCounts(numClasses_, 0);

	SplitResult bestForFeature;
	bool hasBest = false;

	for (std::size_t index = 1; index < view.rows.size(); ++index)
	{
		const SortedFeatureRow &previousRow = view.rows[index - 1];
		const SortedFeatureRow &currentRow = view.rows[index];
		// Row previousRow now belongs to the left child.
		leftCounts[previousRow.classId]++;

		// Skip duplicate feature values — threshold would not change the partition.
		if (std::fabs(previousRow.value - currentRow.value) < options_.epsilon)
		{
			continue;
		}

		// Skip if both neighbors have the same class — split cannot help purity.
		if (previousRow.classId == currentRow.classId)
		{
			continue;
		}

		for (std::uint16_t classIndex = 0; classIndex < numClasses_; ++classIndex)
		{
			rightCounts[classIndex] =
				totalCounts[classIndex] - leftCounts[classIndex];
		}

		// Threshold halfway between the two distinct values.
		const double threshold = (previousRow.value + currentRow.value) / 2.0;
		const SplitResult candidate = scoreCandidateFromCounts(
			view, totalCounts, parentImpurity, leftCounts, rightCounts, index,
			threshold);
		if (!candidate.valid)
		{
			continue;
		}

		if (!hasBest)
		{
			bestForFeature = candidate;
			hasBest = true;
			continue;
		}

		if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
		{
			if (TreeBase::isBetterMaxGain(candidate, bestForFeature, options_.epsilon))
			{
				bestForFeature = candidate;
			}
		}
		else if (TreeBase::isBetterC45(candidate, bestForFeature, options_.epsilon))
		{
			bestForFeature = candidate;
		}
	}

	return bestForFeature;
}

TreeBase::SplitSearchResult TreeBase::evaluateFeatureSplit(
	const std::vector<std::size_t> &rowIndices, std::size_t featureIndex) const
{
	// Shared by Serial and Parallel:
	// build the sorted view for one feature, then find its best threshold.
	SplitSearchResult result;
	result.view = buildSortedFeatureView(rowIndices, featureIndex);
	result.split = scoreAllThresholdsForFeature(result.view);
	result.hasView = result.split.valid;
	return result;
}

TreeBase::SplitSearchResult TreeBase::reduceBestSplitSearch(
	std::vector<SplitSearchResult> featureResults,
	const std::vector<std::size_t> &rowIndices) const
{
	// Each feature search returns its own best split. Now choose the best
	// feature overall, then keep the sorted view for that winning feature.
	std::vector<SplitResult> featureBestCandidates;
	featureBestCandidates.reserve(featureResults.size());
	for (const SplitSearchResult &result : featureResults)
	{
		if (result.split.valid)
		{
			featureBestCandidates.push_back(result.split);
		}
	}

	SplitSearchResult bestSearch;
	bestSearch.split = chooseBestSplit(featureBestCandidates);
	if (!bestSearch.split.valid)
	{
		return bestSearch;
	}

	// Reuse the sorted view from the winning feature so expandOneNode can split
	// rows without scanning them again.
	for (SplitSearchResult &result : featureResults)
	{
		if (result.split.valid &&
			result.split.featureIndex == bestSearch.split.featureIndex &&
			TreeBase::areEffectivelyEqual(result.split.threshold, bestSearch.split.threshold,
								options_.epsilon))
		{
			bestSearch.view = std::move(result.view);
			bestSearch.hasView = result.hasView;
			return bestSearch;
		}
	}

	bestSearch.view =
		buildSortedFeatureView(rowIndices, bestSearch.split.featureIndex);
	bestSearch.hasView = true;
	return bestSearch;
}

bool TreeBase::shouldStopGrowing(const std::vector<std::size_t> &rowIndices,
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

bool TreeBase::isSplitRejected(const SplitResult &split) const
{
	// C4.5 rejects splits with negligible gain ratio; CART only checks validity.
	return !split.valid ||
		   (options_.splitSelectionMode != SplitSelectionMode::MaxGain &&
			split.gainRatio <= options_.epsilon);
}

// Grow a single node: either return a leaf, or a decision node + row partitions.
TreeBase::NodeExpandResult
TreeBase::expandOneNode(const std::vector<std::size_t> &rowIndices,
					   int depth) const
{
	// This function decides what one node should become.
	// It does not build children itself; it only returns the node and the row
	// lists that children should receive.

	// Perfect purity → leaf with that label (no point splitting further).
	if (allSameLabel(rowIndices))
	{
		return {Node::createLeaf(dataset_->samples[rowIndices.front()].label,
								 rowIndices.size()),
				{}};
	}

	if (shouldStopGrowing(rowIndices, depth))
	{
		// Mixed classes but stopping rules say "do not split" → majority-vote leaf.
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	const SplitSearchResult search = findBestSplitAtNode(rowIndices);
	const SplitResult &split = search.split;
	if (isSplitRejected(split))
	{
		// No useful split found → majority-vote leaf.
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	PartitionedRows partitions;
	if (search.hasView && split.leftRowCount > 0 &&
		split.leftRowCount < rowIndices.size())
	{
		// Fast path: partition using the sorted view we already built during search.
		partitions = partitionFromSortedView(search.view, split.leftRowCount);
	}
	else
	{
		partitions =
			partitionRows(rowIndices, split.featureIndex, split.threshold);
	}

	if (partitions.leftRows.empty() || partitions.rightRows.empty())
	{
		// Safety net: degenerate split → leaf.
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	// Decision node: stores the question "feature <= threshold?" for prediction.
	return {Node::createDecision(split.featureName, split.featureIndex, split.threshold,
							   rowIndices.size()),
			std::move(partitions)};
}

std::unique_ptr<Node>
TreeBase::buildNode(const std::vector<std::size_t> &rowIndices,
				   int depth) const
{
	// Serial recursive tree builder (used by TreeSerial and small parallel subtrees).
	// The parallel backend can still reuse this when a subtree is too small to
	// be worth sending to another thread.
	NodeExpandResult expanded = expandOneNode(rowIndices, depth);
	if (expanded.node->isLeaf)
	{
		return std::move(expanded.node);
	}

	// Depth-first: fully build left subtree, then right subtree.
	expanded.node->leftChild =
		buildNode(expanded.partitions.leftRows, depth + 1);
	expanded.node->rightChild =
		buildNode(expanded.partitions.rightRows, depth + 1);
	return std::move(expanded.node);
}



std::size_t TreeBase::countCorrectPredictions(
	const Node *node, const std::vector<std::size_t> &rowIndices) const
{
	// Used by CART cost-complexity pruning: how many rows does this subtree classify correctly?
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

std::size_t TreeBase::countLeafNodes(const Node *node) const
{
	// Leaf count matters for cost-complexity pruning (each leaf adds a penalty term).
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

std::size_t TreeBase::countNodes(const Node *node) const
{
	// Total nodes (decision + leaf) — useful for comparing pruned vs unpruned trees.
	if (!node)
	{
		return 0;
	}

	return 1 + countNodes(node->leftChild.get()) +
		   countNodes(node->rightChild.get());
}

int TreeBase::computeTreeDepth(const Node *node) const
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

bool TreeBase::allSameLabel(const std::vector<std::size_t> &rowIndices) const
{
	// Quick purity check: if every row shares one label, entropy = 0 and we stop.
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
TreeBase::getMajorityLabel(const std::vector<std::size_t> &rowIndices) const
{
	// When we cannot split further, predict the most common class at this node.
	const std::vector<std::uint32_t> counts = computeClassCountsArray(rowIndices);

	std::uint16_t bestClassId = 0;
	std::uint32_t maxCount = 0;

	for (std::uint16_t classIndex = 0; classIndex < numClasses_; ++classIndex)
	{
		if (counts[classIndex] > maxCount)
		{
			maxCount = counts[classIndex];
			bestClassId = classIndex;
		}
	}
	return classLabels_[bestClassId];
}

void TreeBase::printNode(const Node *node, std::ostream &output, int depth,
						const std::string &edgeText) const
{
	// Human-readable tree dump for debugging and learning.
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

namespace {

double clampConfidenceFactor(double value) {
  // C4.5 pessimistic pruning expects a confidence factor in [0, 0.5].
  return std::max(0.0, std::min(0.5, value));
}

double inverseStandardNormalCdf(double probability) {
  // Helper for C4.5 pessimistic pruning. It estimates a normal-distribution
  // z-score without depending on an external statistics library.
  const double a1 = -39.6968302866538;
  const double a2 = 220.946098424521;
  const double a3 = -275.928510446969;
  const double a4 = 138.357751867269;
  const double a5 = -30.6647980661472;
  const double a6 = 2.50662827745924;

  const double b1 = -54.4760987982241;
  const double b2 = 161.585836858041;
  const double b3 = -155.698979859887;
  const double b4 = 66.8013118877197;
  const double b5 = -13.2806815528857;

  const double c1 = -0.00778489400243029;
  const double c2 = -0.322396458041136;
  const double c3 = -2.40075827716184;
  const double c4 = -2.54973253934373;
  const double c5 = 4.37466414146497;
  const double c6 = 2.93816398269878;

  const double d1 = 0.00778469570904146;
  const double d2 = 0.32246712907004;
  const double d3 = 2.445134137143;
  const double d4 = 3.75440866190742;

  const double pLow = 0.02425;
  const double pHigh = 1.0 - pLow;

  if (probability <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (probability >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  if (probability < pLow) {
    const double q = std::sqrt(-2.0 * std::log(probability));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  if (probability <= pHigh) {
    const double q = probability - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

double addErrs(double sampleCount, double observedErrors, double confidenceFactor) {
  if (confidenceFactor > 0.5) {
    return 0.0;
  }

  if (observedErrors < 1.0) {
    const double base =
        sampleCount * (1.0 - std::pow(confidenceFactor, 1.0 / sampleCount));
    if (observedErrors == 0.0) {
      return base;
    }
    return base + observedErrors * (addErrs(sampleCount, 1.0, confidenceFactor) - base);
  }

  if (observedErrors + 0.5 >= sampleCount) {
    return std::max(sampleCount - observedErrors, 0.0);
  }

  const double z = inverseStandardNormalCdf(1.0 - confidenceFactor);
  const double errorRate = (observedErrors + 0.5) / sampleCount;
  const double zSquared = z * z;
  const double upperBoundRate =
      (errorRate + zSquared / (2.0 * sampleCount) +
       z * std::sqrt((errorRate / sampleCount) -
                     (errorRate * errorRate / sampleCount) +
                     (zSquared / (4.0 * sampleCount * sampleCount)))) /
      (1.0 + zSquared / sampleCount);

  return std::max(upperBoundRate * sampleCount - observedErrors, 0.0);
}

double estimatePessimisticLeafErrorCount(std::size_t observedErrors,
                                         std::size_t sampleCount,
                                         double confidenceFactor) {
  // C4.5 does not trust the training error exactly. It adds a small penalty so
  // a tiny leaf with few samples is not treated as perfectly reliable.
  if (sampleCount == 0) {
    return 0.0;
  }

  const double clampedFactor = clampConfidenceFactor(confidenceFactor);
  const double n = static_cast<double>(sampleCount);
  const double errors = static_cast<double>(observedErrors);
  return errors + addErrs(n, errors, clampedFactor);
}

} // namespace

void TreeBase::runPruning(const TrainingOptions &options) {
  // Pruning runs after the full tree is built. It replaces some subtrees with
  // leaves when the simpler leaf looks good enough.
  pruneTimeSeconds_ = 0.0;

  if (!root_ || options.pruningMode == PruningMode::None) {
    return;
  }

  if (options.pruningMode == PruningMode::PessimisticError) {
    // C4.5-style pruning: compare the estimated error of the subtree with the
    // estimated error of one majority-label leaf.
    const auto pruneStart = std::chrono::steady_clock::now();
    const double confidenceFactor = options.pruningConfidenceFactor;

    std::vector<std::size_t> rowIndices(dataset_->samples.size());
    for (std::size_t index = 0; index < rowIndices.size(); ++index) {
      rowIndices[index] = index;
    }

    std::function<double(const Node *, const std::vector<std::size_t> &)>
        estimateSubtreeErrors;
    estimateSubtreeErrors = [this, confidenceFactor, &estimateSubtreeErrors](
                                const Node *subtree,
                                const std::vector<std::size_t> &indices) -> double {
      // Walk the subtree and add up estimated leaf errors.
      if (!subtree) {
        return 0.0;
      }

      if (subtree->isLeaf) {
        std::size_t observedErrors = 0;
        for (std::size_t rowIndex : indices) {
          if (dataset_->samples[rowIndex].label != subtree->leafLabel) {
            ++observedErrors;
          }
        }
        return estimatePessimisticLeafErrorCount(observedErrors, indices.size(),
                                                 confidenceFactor);
      }

      const PartitionedRows parts =
          partitionRows(indices, subtree->featureIndex, subtree->threshold);

      return estimateSubtreeErrors(subtree->leftChild.get(), parts.leftRows) +
             estimateSubtreeErrors(subtree->rightChild.get(), parts.rightRows);
    };

    std::function<void(std::unique_ptr<Node> &, const std::vector<std::size_t> &)>
        pruneNode = [&](std::unique_ptr<Node> &node,
                        const std::vector<std::size_t> &indices) {
          if (!node || node->isLeaf) {
            return;
          }

          const PartitionedRows partitions =
              partitionRows(indices, node->featureIndex, node->threshold);

          pruneNode(node->leftChild, partitions.leftRows);
          pruneNode(node->rightChild, partitions.rightRows);

          const std::string majorityLabel = getMajorityLabel(indices);
          std::size_t leafCorrect = 0;
          for (std::size_t rowIndex : indices) {
            if (dataset_->samples[rowIndex].label == majorityLabel) {
              ++leafCorrect;
            }
          }

          const std::size_t observedLeafErrors = indices.size() - leafCorrect;
          const double estimatedLeafErrors = estimatePessimisticLeafErrorCount(
              observedLeafErrors, indices.size(), confidenceFactor);
          const double estimatedSubtreeErrors =
              estimateSubtreeErrors(node.get(), indices);

          if (estimatedLeafErrors <= estimatedSubtreeErrors + options_.epsilon) {
            node = Node::createLeaf(majorityLabel, indices.size());
          }
        };

    pruneNode(root_, rowIndices);

    pruneTimeSeconds_ = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pruneStart)
                            .count();
    return;
  }

  if (options.pruningMode == PruningMode::CostComplexity) {
    // CART-style pruning: a subtree pays for its mistakes plus a penalty for
    // every leaf. Bigger alpha means stronger preference for smaller trees.
    const auto pruneStart = std::chrono::steady_clock::now();
    const double ccpAlpha = options.ccpAlpha;

    std::vector<std::size_t> rowIndices(dataset_->samples.size());
    for (std::size_t index = 0; index < rowIndices.size(); ++index) {
      rowIndices[index] = index;
    }

    std::function<void(std::unique_ptr<Node> &, const std::vector<std::size_t> &)>
        pruneNode = [&](std::unique_ptr<Node> &node,
                        const std::vector<std::size_t> &indices) {
          if (!node || node->isLeaf) {
            return;
          }

          const PartitionedRows partitions =
              partitionRows(indices, node->featureIndex, node->threshold);

          pruneNode(node->leftChild, partitions.leftRows);
          pruneNode(node->rightChild, partitions.rightRows);

          if (node->isLeaf) {
            return;
          }

          const std::string majorityLabel = getMajorityLabel(indices);
          std::size_t errorsAsLeaf = 0;
          for (std::size_t rowIndex : indices) {
            if (dataset_->samples[rowIndex].label != majorityLabel) {
              ++errorsAsLeaf;
            }
          }

          const std::size_t errorsInSubtree =
              indices.size() - countCorrectPredictions(node.get(), indices);
          const std::size_t leavesInSubtree = countLeafNodes(node.get());

          const double costAsLeaf =
              static_cast<double>(errorsAsLeaf) + ccpAlpha;
          const double costAsSubtree =
              static_cast<double>(errorsInSubtree) +
              ccpAlpha * static_cast<double>(leavesInSubtree);

          // If the leaf is no worse than the subtree, keep the simpler model.
          if (costAsLeaf <= costAsSubtree + options_.epsilon) {
            node = Node::createLeaf(majorityLabel, indices.size());
          }
        };

    pruneNode(root_, rowIndices);

    pruneTimeSeconds_ = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pruneStart)
                            .count();
  }
}

void TreeBase::prune(const TrainingOptions &options) { runPruning(options); }

double TreeBase::buildTimeSeconds() const {
  return buildTimeSeconds_;
}

double TreeBase::pruneTimeSeconds() const {
  return pruneTimeSeconds_;
}
