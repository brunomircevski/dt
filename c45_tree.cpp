#include "c45_tree.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
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

	// When two splits look equally good on paper, we need fixed tie-breakers so the
	// tree is always the same (important for debugging and for parallel vs serial).
	// C4.5 order: gain ratio → information gain → earlier feature column → lower threshold.
	bool isBetterC45(const SplitResult &lhs, const SplitResult &rhs, double epsilon)
	{
		// 1. Primary check: Prefer the split with a significantly higher Gain Ratio.
		if (lhs.gainRatio > rhs.gainRatio + epsilon) return true;
		if (rhs.gainRatio > lhs.gainRatio + epsilon) return false;

		// 2. First tie-breaker: Prefer the split with a significantly higher Information Gain.
		if (lhs.informationGain > rhs.informationGain + epsilon) return true;
		if (rhs.informationGain > lhs.informationGain + epsilon) return false;

		// 3. Second tie-breaker: Prefer the feature that appears earlier in the dataset.
		if (lhs.featureIndex != rhs.featureIndex)
		{
			return lhs.featureIndex < rhs.featureIndex;
		}

		// 4. Third tie-breaker: Prefer the lower threshold value to keep splits conservative.
		return lhs.threshold < rhs.threshold - epsilon;
	}

	// CART order: information gain (or Gini gain) → earlier feature → lower threshold.
	bool isBetterMaxGain(const SplitResult &lhs, const SplitResult &rhs, double epsilon)
	{
		// 1. Primary check: Prefer the split with a significantly higher Information/Gini Gain.
		if (lhs.informationGain > rhs.informationGain + epsilon) return true;
		if (rhs.informationGain > lhs.informationGain + epsilon) return false;

		// 2. First tie-breaker: Prefer the feature that appears earlier in the dataset.
		if (lhs.featureIndex != rhs.featureIndex)
		{
			return lhs.featureIndex < rhs.featureIndex;
		}

		// 3. Second tie-breaker: Prefer the lower threshold value to keep splits conservative.
		return lhs.threshold < rhs.threshold - epsilon;
	}

} // namespace

C45Tree::C45Tree() = default;

C45Tree::~C45Tree() = default;

void C45Tree::fit(const Dataset &dataset, const TrainingOptions &options)
{
	if (dataset.samples.empty())
	{
		throw std::runtime_error("Cannot train on an empty dataset.");
	}

	// Remember the dataset and options for the whole training run.
	// We store a pointer so we never copy thousands of rows into the tree object.
	dataset_ = &dataset;
	options_ = options;
	buildClassMapping(dataset);

	fitContext_ = std::make_unique<FitContext>();
	const std::size_t threadCount =
		static_cast<std::size_t>(std::max(options_.maxThreadCount, 1));
	const bool needsVd = options_.gleamsMode == GleamsMode::VDa ||
						 options_.gleamsMode == GleamsMode::VDTa;
	const bool needsTa = options_.gleamsMode == GleamsMode::Ta ||
						 options_.gleamsMode == GleamsMode::VDTa;
	if (needsVd)
	{
		fitContext_->vdExecutor = std::make_unique<TaskExecutor>(threadCount);
	}
	if (needsTa)
	{
		fitContext_->taExecutor = std::make_unique<TaskExecutor>(threadCount);
	}

	// rowIndices lists which training rows "belong" to the current node.
	// At the root, that is every row: 0, 1, 2, ..., n-1.
	std::vector<std::size_t> rowIndices(dataset.samples.size());
	for (std::size_t i = 0; i < rowIndices.size(); ++i)
	{
		rowIndices[i] = i;
	}

	const auto buildStart = std::chrono::steady_clock::now();
	if (usesAsyncTreeBuilding())
	{
		auto rootPromise = std::make_shared<std::promise<std::unique_ptr<Node>>>();
		std::future<std::unique_ptr<Node>> rootFuture = rootPromise->get_future();
		auto onComplete = wrapNodeOnComplete(
			[rootPromise](std::unique_ptr<Node> node) mutable {
				rootPromise->set_value(std::move(node));
			});

		// Copy — rowIndices must stay valid for applySelectedPruning after the build.
		const RowIndexList rootRows =
			std::make_shared<std::vector<std::size_t>>(rowIndices);
		fitContext_->pendingNodeTasks.fetch_add(1, std::memory_order_relaxed);
		fitContext_->taExecutor->submit([this, rootRows,
										 onComplete = std::move(onComplete)]() mutable {
			expandNodeAsync({rootRows, 0, std::move(onComplete)});
		});

		waitForAllNodeTasks();
		root_ = rootFuture.get();
	}
	else
	{
		root_ = buildNode(rowIndices, 0);
	}
	buildTimeSeconds_ = std::chrono::duration<double>(
							std::chrono::steady_clock::now() - buildStart)
							.count();

	// Post-pruning is a second phase:
	// first let the tree grow, then remove branches that do not improve
	// classification on the samples that reached them.
	pruneTimeSeconds_ = 0.0;
	if (options_.pruningMode != PruningMode::None)
	{
		const auto pruneStart = std::chrono::steady_clock::now();
		applySelectedPruning(rowIndices);
		pruneTimeSeconds_ = std::chrono::duration<double>(
								std::chrono::steady_clock::now() - pruneStart)
								.count();
	}

	fitContext_.reset();
}

std::string C45Tree::predict(const Sample &sample) const
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

double C45Tree::buildTimeSeconds() const { return buildTimeSeconds_; }

double C45Tree::pruneTimeSeconds() const { return pruneTimeSeconds_; }

void C45Tree::buildClassMapping(const Dataset &dataset)
{
	classLabels_.clear();
	labelToClassId_.clear();
	numClasses_ = 0;

	std::vector<std::string> uniqueLabels;
	uniqueLabels.reserve(16);
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

std::uint16_t C45Tree::classIdForRow(std::size_t rowIndex) const
{
	return labelToClassId_.at(dataset_->samples[rowIndex].label);
}

std::vector<std::uint32_t> C45Tree::computeClassCountsArray(
	const std::vector<std::size_t> &rowIndices) const
{
	std::vector<std::uint32_t> counts(numClasses_, 0);
	for (std::size_t rowIndex : rowIndices)
	{
		++counts[classIdForRow(rowIndex)];
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
	// C4.5 only: measures how "balanced" the split sizes are (entropy of child sizes).
	// Gain ratio = information gain / split information, so we prefer splits that are
	// both informative AND not wildly uneven (e.g. 99% of rows on one side).

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
	// Used only after the winning split is chosen (buildNode), not while scoring candidates.
	// Question: "Is feature <= threshold?"
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

double C45Tree::impurityFromCounts(const std::vector<std::uint32_t> &counts,
								   std::size_t total) const
{
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
// Thresholds are only placed between adjacent values with different labels.
C45Tree::SortedFeatureView C45Tree::buildSortedFeatureView(
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

SplitResult C45Tree::scoreCandidateFromCounts(
	const SortedFeatureView &view, const std::vector<std::uint32_t> &totalCounts,
	double parentImpurity, const std::vector<std::uint32_t> &leftCounts,
	const std::vector<std::uint32_t> &rightCounts, std::size_t leftSize,
	double threshold) const
{
	(void)totalCounts;
	SplitResult result;
	result.featureIndex = view.featureIndex;
	result.featureName = dataset_->featureNames[view.featureIndex];
	result.threshold = threshold;
	result.leftRowCount = leftSize;

	const std::size_t rightSize = view.totalRows - leftSize;
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
	const double afterSplit =
		(static_cast<double>(leftSize) / totalDouble) * leftImpurity +
		(static_cast<double>(rightSize) / totalDouble) * rightImpurity;

	result.informationGain = parentImpurity - afterSplit;

	if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
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

C45Tree::PartitionedRows
C45Tree::partitionFromSortedView(const SortedFeatureView &view,
								 std::size_t leftSize) const
{
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

SplitResult
C45Tree::chooseBestSplit(const std::vector<SplitResult> &candidates) const
{
	// Input: at most one best candidate per feature (already filtered by reduceBestPerFeature).
	// Output: the single split we will actually use at this node.
	if (candidates.empty())
	{
		return SplitResult{};
	}

	// CART: pick the feature winner with highest information gain (or Gini gain).
	if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
	{
		const SplitResult *bestCandidate = &candidates.front();
		for (std::size_t index = 1; index < candidates.size(); ++index)
		{
			if (isBetterMaxGain(candidates[index], *bestCandidate, options_.epsilon))
			{
				bestCandidate = &candidates[index];
			}
		}
		return *bestCandidate;
	}

	// C4.5: ignore features whose gain is below average, then pick highest gain ratio.
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

		if (!bestCandidate || isBetterC45(candidate, *bestCandidate, options_.epsilon))
		{
			bestCandidate = &candidate;
		}
	}

	return bestCandidate ? *bestCandidate : SplitResult{};
}

bool C45Tree::usesVerticalParallelism() const
{
	return fitContext_ != nullptr && fitContext_->vdExecutor != nullptr;
}

bool C45Tree::usesAsyncTreeBuilding() const
{
	return fitContext_ != nullptr && fitContext_->taExecutor != nullptr;
}

bool C45Tree::shouldParallelizeVD(std::size_t featureCount) const
{
	return usesVerticalParallelism() &&
		   featureCount >= options_.minFeaturesToParallelize;
}

bool C45Tree::shouldParallelizeTa(std::size_t rowCount) const
{
	return usesAsyncTreeBuilding() &&
		   rowCount >= options_.minRowsToParallelize;
}

void C45Tree::notifyNodeTaskFinished() const
{
	if (fitContext_->pendingNodeTasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
	{
		std::lock_guard<std::mutex> lock(fitContext_->completionMutex);
		fitContext_->completionCv.notify_all();
	}
}

void C45Tree::waitForAllNodeTasks() const
{
	std::unique_lock<std::mutex> lock(fitContext_->completionMutex);
	fitContext_->completionCv.wait(lock, [this]() {
		return fitContext_->pendingNodeTasks.load(std::memory_order_acquire) == 0;
	});
}

std::function<void(std::unique_ptr<Node>)> C45Tree::wrapNodeOnComplete(
	std::function<void(std::unique_ptr<Node>)> userCallback) const
{
	return [this, userCallback = std::move(userCallback)](std::unique_ptr<Node> node) mutable {
		userCallback(std::move(node));
		notifyNodeTaskFinished();
	};
}

SplitResult C45Tree::scoreAllThresholdsForFeature(const SortedFeatureView &view) const
{
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
		leftCounts[previousRow.classId]++;

		if (std::fabs(previousRow.value - currentRow.value) < options_.epsilon)
		{
			continue;
		}

		if (previousRow.classId == currentRow.classId)
		{
			continue;
		}

		for (std::uint16_t classIndex = 0; classIndex < numClasses_; ++classIndex)
		{
			rightCounts[classIndex] =
				totalCounts[classIndex] - leftCounts[classIndex];
		}

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
			if (isBetterMaxGain(candidate, bestForFeature, options_.epsilon))
			{
				bestForFeature = candidate;
			}
		}
		else if (isBetterC45(candidate, bestForFeature, options_.epsilon))
		{
			bestForFeature = candidate;
		}
	}

	return bestForFeature;
}

SplitResult C45Tree::reduceBestPerFeature(
	const std::vector<SplitResult> &scoredCandidates) const
{
	// We scored every (feature, threshold) pair. C4.5/CART first pick the best threshold
	// per feature, then chooseBestSplit picks the best feature among those.
	const std::size_t featureCount = dataset_->featureNames.size();
	std::vector<SplitResult> bestPerFeature(featureCount);
	std::vector<bool> hasBestPerFeature(featureCount, false);

	for (const SplitResult &candidate : scoredCandidates)
	{
		if (!candidate.valid)
		{
			continue;
		}

		const std::size_t featureIndex = candidate.featureIndex;
		if (!hasBestPerFeature[featureIndex])
		{
			bestPerFeature[featureIndex] = candidate;
			hasBestPerFeature[featureIndex] = true;
			continue;
		}

		SplitResult &bestForFeature = bestPerFeature[featureIndex];
		if (options_.splitSelectionMode == SplitSelectionMode::MaxGain)
		{
			if (isBetterMaxGain(candidate, bestForFeature, options_.epsilon))
			{
				bestForFeature = candidate;
			}
		}
		else if (isBetterC45(candidate, bestForFeature, options_.epsilon))
		{
			bestForFeature = candidate;
		}
	}

	std::vector<SplitResult> featureBestCandidates;
	featureBestCandidates.reserve(featureCount);
	for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex)
	{
		if (hasBestPerFeature[featureIndex])
		{
			featureBestCandidates.push_back(bestPerFeature[featureIndex]);
		}
	}

	return chooseBestSplit(featureBestCandidates);
}

C45Tree::SplitSearchResult
C45Tree::findBestSplitAtNode(const std::vector<std::size_t> &rowIndices) const
{
	const std::size_t featureCount = dataset_->featureNames.size();
	if (featureCount == 0)
	{
		return {};
	}

	std::vector<SplitSearchResult> featureResults;
	featureResults.reserve(featureCount);

	if (shouldParallelizeVD(featureCount))
	{
		const RowIndexList rows =
			std::make_shared<std::vector<std::size_t>>(rowIndices);
		std::vector<std::future<SplitSearchResult>> futures(featureCount);
		for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex)
		{
			futures[featureIndex] = fitContext_->vdExecutor->submit(
				[this, rows, featureIndex]() {
					SplitSearchResult result;
					result.view = buildSortedFeatureView(*rows, featureIndex);
					result.split = scoreAllThresholdsForFeature(result.view);
					result.hasView = result.split.valid;
					return result;
				});
		}

		for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex)
		{
			featureResults.push_back(futures[featureIndex].get());
		}
	}
	else
	{
		for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex)
		{
			SplitSearchResult result;
			result.view = buildSortedFeatureView(rowIndices, featureIndex);
			result.split = scoreAllThresholdsForFeature(result.view);
			result.hasView = result.split.valid;
			featureResults.push_back(std::move(result));
		}
	}

	std::vector<SplitResult> featureBestCandidates;
	featureBestCandidates.reserve(featureCount);
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

	for (SplitSearchResult &result : featureResults)
	{
		if (result.split.valid &&
			result.split.featureIndex == bestSearch.split.featureIndex &&
			areEffectivelyEqual(result.split.threshold, bestSearch.split.threshold,
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

SplitResult
C45Tree::findBestSplit(const std::vector<std::size_t> &rowIndices) const
{
	return findBestSplitAtNode(rowIndices).split;
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

bool C45Tree::isSplitRejected(const SplitResult &split) const
{
	return !split.valid ||
		   (options_.splitSelectionMode != SplitSelectionMode::MaxGain &&
			split.gainRatio <= options_.epsilon);
}

C45Tree::NodeExpandResult
C45Tree::expandOneNode(const std::vector<std::size_t> &rowIndices,
					   int depth) const
{
	if (allSameLabel(rowIndices))
	{
		return {Node::createLeaf(dataset_->samples[rowIndices.front()].label,
								 rowIndices.size()),
				{}};
	}

	if (shouldStopGrowing(rowIndices, depth))
	{
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	const SplitSearchResult search = findBestSplitAtNode(rowIndices);
	const SplitResult &split = search.split;
	if (isSplitRejected(split))
	{
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	PartitionedRows partitions;
	if (search.hasView && split.leftRowCount > 0 &&
		split.leftRowCount < rowIndices.size())
	{
		partitions = partitionFromSortedView(search.view, split.leftRowCount);
	}
	else
	{
		partitions =
			partitionRows(rowIndices, split.featureIndex, split.threshold);
	}

	if (partitions.leftRows.empty() || partitions.rightRows.empty())
	{
		return {Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size()),
				{}};
	}

	return {Node::createDecision(split.featureName, split.featureIndex, split.threshold,
							   rowIndices.size()),
			std::move(partitions)};
}

std::unique_ptr<Node>
C45Tree::buildNode(const std::vector<std::size_t> &rowIndices,
				   int depth) const
{
	NodeExpandResult expanded = expandOneNode(rowIndices, depth);
	if (expanded.node->isLeaf)
	{
		return std::move(expanded.node);
	}

	expanded.node->leftChild =
		buildNode(expanded.partitions.leftRows, depth + 1);
	expanded.node->rightChild =
		buildNode(expanded.partitions.rightRows, depth + 1);
	return std::move(expanded.node);
}

void C45Tree::scheduleAsyncChildren(
	std::unique_ptr<Node> parent, PartitionedRows partitions, int depth,
	std::function<void(std::unique_ptr<Node>)> onComplete) const
{
	auto parentHolder =
		std::make_shared<std::unique_ptr<Node>>(std::move(parent));
	auto pendingChildren = std::make_shared<std::atomic<int>>(2);
	auto childMutex = std::make_shared<std::mutex>();

	const auto attachChild =
		[parentHolder, pendingChildren, childMutex,
		 onComplete](bool isLeft) {
			return [parentHolder, pendingChildren, childMutex, onComplete,
					isLeft](std::unique_ptr<Node> child) {
				{
					std::lock_guard<std::mutex> lock(*childMutex);
					if (isLeft)
					{
						(*parentHolder)->leftChild = std::move(child);
					}
					else
					{
						(*parentHolder)->rightChild = std::move(child);
					}
				}

				if (pendingChildren->fetch_sub(1, std::memory_order_acq_rel) == 1)
				{
					onComplete(std::move(*parentHolder));
				}
			};
		};

	const auto submitChild = [&](std::vector<std::size_t> childRows, bool isLeft) {
		const RowIndexList childRowsShared =
			std::make_shared<std::vector<std::size_t>>(std::move(childRows));
		fitContext_->pendingNodeTasks.fetch_add(1, std::memory_order_relaxed);
		fitContext_->taExecutor->submit(
			[this, childRowsShared, depth, isLeft,
			 childOnComplete = wrapNodeOnComplete(attachChild(isLeft))]() mutable {
				expandNodeAsync(
					{childRowsShared, depth + 1, std::move(childOnComplete)});
			});
	};

	submitChild(std::move(partitions.leftRows), true);
	submitChild(std::move(partitions.rightRows), false);
}

void C45Tree::expandNodeAsync(NodeBuildJob job) const
{
	const RowIndexList &rowIndices = job.rowIndices;
	const int depth = job.depth;
	std::function<void(std::unique_ptr<Node>)> onComplete = std::move(job.onComplete);

	if (!shouldParallelizeTa(rowIndices->size()))
	{
		onComplete(buildNode(*rowIndices, depth));
		return;
	}

	NodeExpandResult expanded = expandOneNode(*rowIndices, depth);
	if (expanded.node->isLeaf)
	{
		onComplete(std::move(expanded.node));
		return;
	}

	scheduleAsyncChildren(std::move(expanded.node), std::move(expanded.partitions),
						  depth, std::move(onComplete));
}

std::size_t C45Tree::countCorrectPredictions(
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

void C45Tree::applySelectedPruning(const std::vector<std::size_t>& rowIndices)
{
	// Pruning runs after the tree is fully grown.
	// Implementation lives in pruning/*.cpp — we only dispatch here.
	if (!root_)
	{
		return;
	}

	if (options_.pruningMode == PruningMode::PessimisticError)
	{
		prunePessimisticError(root_, rowIndices);
		return;
	}

	if (options_.pruningMode == PruningMode::CostComplexity)
	{
		pruneCostComplexity(root_, rowIndices);
	}
}
