#include "c45_tree.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
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

	// One possible question at a node: "Is feature X <= threshold?"
	// We collect many of these, score them, then pick the winner.
	struct SplitCandidateSpec
	{
		std::size_t featureIndex = 0;
		double threshold = 0.0;
	};

} // namespace

// Thread pool reused for the whole fit() call.
// Why a pool? Creating OS threads inside every findBestSplit() would be very slow.
// Workers sit in a loop, take tasks from a queue, and run them.
class C45Tree::SplitThreadPool
{
public:
	explicit SplitThreadPool(std::size_t threadCount)
	{
		if (threadCount == 0)
		{
			threadCount = 1;
		}

		workers_.reserve(threadCount);
		for (std::size_t index = 0; index < threadCount; ++index)
		{
			workers_.emplace_back([this]() { workerLoop(); });
		}
	}

	~SplitThreadPool()
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stop_ = true;
		}
		taskCv_.notify_all();
		for (std::thread &worker : workers_)
		{
			if (worker.joinable())
			{
				worker.join();
			}
		}
	}

	SplitThreadPool(const SplitThreadPool &) = delete;
	SplitThreadPool &operator=(const SplitThreadPool &) = delete;

	// Run work(0), work(1), ... work(count-1) on several threads.
	// Each worker grabs the next index from a shared counter until all are done.
	// The caller blocks until every worker finishes (so findBestSplit can safely
	// read scoredCandidates after this returns).
	void parallel_for(std::size_t count,
					  const std::function<void(std::size_t)> &work)
	{
		if (count == 0)
		{
			return;
		}

		if (workers_.empty())
		{
			for (std::size_t index = 0; index < count; ++index)
			{
				work(index);
			}
			return;
		}

		std::atomic<std::size_t> nextIndex{0};
		const std::size_t workersToUse = std::min(workers_.size(), count);
		std::atomic<std::size_t> activeWorkers{workersToUse};

		for (std::size_t workerIndex = 0; workerIndex < workersToUse; ++workerIndex)
		{
			enqueue([&]() {
				while (true)
				{
					const std::size_t index =
						nextIndex.fetch_add(1, std::memory_order_relaxed);
					if (index >= count)
					{
						break;
					}
					work(index);
				}

				if (activeWorkers.fetch_sub(1, std::memory_order_acq_rel) == 1)
				{
					std::lock_guard<std::mutex> lock(doneMutex_);
					doneCv_.notify_one();
				}
			});
		}

		std::unique_lock<std::mutex> lock(doneMutex_);
		doneCv_.wait(lock, [&]() { return activeWorkers.load() == 0; });
	}

private:
	// Each OS thread runs this forever until the pool is destroyed.
	void workerLoop()
	{
		while (true)
		{
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(mutex_);
				taskCv_.wait(lock, [&]() { return stop_ || !tasks_.empty(); });
				if (stop_ && tasks_.empty())
				{
					return;
				}
				task = std::move(tasks_.front());
				tasks_.pop();
			}
			task();
		}
	}

	void enqueue(std::function<void()> task)
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			tasks_.push(std::move(task));
		}
		taskCv_.notify_one();
	}

	std::vector<std::thread> workers_;
	std::queue<std::function<void()>> tasks_;
	std::mutex mutex_;
	std::condition_variable taskCv_;
	bool stop_ = false;
	std::mutex doneMutex_;
	std::condition_variable doneCv_;
};

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

	// Start worker threads only if the user asked for parallelism.
	// They are torn down at the end of fit() so we do not keep threads alive after training.
	splitThreadPool_.reset();
	if (options_.maxThreadCount > 1)
	{
		splitThreadPool_ = std::make_unique<SplitThreadPool>(
			static_cast<std::size_t>(options_.maxThreadCount));
	}

	// rowIndices lists which training rows "belong" to the current node.
	// At the root, that is every row: 0, 1, 2, ..., n-1.
	std::vector<std::size_t> rowIndices(dataset.samples.size());
	for (std::size_t i = 0; i < rowIndices.size(); ++i)
	{
		rowIndices[i] = i;
	}

	// Build the whole tree recursively.
	const auto buildStart = std::chrono::steady_clock::now();
	root_ = buildNode(rowIndices, 0);
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

	splitThreadPool_.reset();
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
		// CART only cares about impurity drop (Gini or entropy, depending on options).
		result.splitInformation = 0.0;
		result.gainRatio = 0.0;
		result.valid = (result.informationGain > options_.epsilon);
		return result;
	}

	// C4.5: also compute gain ratio; invalid if gain or split info is effectively zero.
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

bool C45Tree::shouldParallelizeSplitSearch(std::size_t candidateCount) const
{
	// On tiny nodes there may be only a few thresholds to try — threading costs more than it saves.
	return splitThreadPool_ != nullptr &&
		   candidateCount >= options_.minCandidatesToParallelize;
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

SplitResult
C45Tree::findBestSplit(const std::vector<std::size_t> &rowIndices) const
{
	// This function answers: "What is the best split for the rows at this node?"
	//
	// Steps:
	//   1. List every (feature, threshold) we want to try.
	//   2. Score each one (partition rows + impurity) — parallel if enough candidates.
	//   3. Keep the best threshold per feature.
	//   4. chooseBestSplit picks the overall winner (C4.5 or CART rules).
	std::vector<SplitCandidateSpec> candidates;
	const std::size_t featureCount = dataset_->featureNames.size();
	candidates.reserve(featureCount * 8);

	for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex)
	{
		const std::vector<double> thresholds =
			collectNumericThresholdCandidates(rowIndices, featureIndex);
		for (double threshold : thresholds)
		{
			candidates.push_back({featureIndex, threshold});
		}
	}

	if (candidates.empty())
	{
		return SplitResult{};
	}

	// Define job to score a candidate in lambda function
	std::vector<SplitResult> scoredCandidates(candidates.size());
	const auto evaluateCandidate = [&](std::size_t candidateIndex) {
		const SplitCandidateSpec &candidate = candidates[candidateIndex];
		scoredCandidates[candidateIndex] =
			scoreSplit(rowIndices, candidate.featureIndex, candidate.threshold);
	};

	// Parallelize the scoring of the candidates
	if (shouldParallelizeSplitSearch(candidates.size()))
	{
		splitThreadPool_->parallel_for(candidates.size(), evaluateCandidate);
	}
	else
	{
		// Same math as parallel path; easier to debug with maxThreadCount = 1.
		for (std::size_t candidateIndex = 0; candidateIndex < candidates.size();
			 ++candidateIndex)
		{
			evaluateCandidate(candidateIndex);
		}
	}

	return reduceBestPerFeature(scoredCandidates);
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

	// Ask every feature/threshold combination which split is best here.
	const SplitResult split = findBestSplit(rowIndices);
	if (!split.valid ||
		(options_.splitSelectionMode != SplitSelectionMode::MaxGain && split.gainRatio <= options_.epsilon))
	{
		// No split beats a simple majority-vote leaf — stop growing this branch.
		return Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size());
	}

	// We only partition once, using the winning split (not during every candidate test).
	const PartitionedRows partitions =
		partitionRows(rowIndices, split.featureIndex, split.threshold);

	if (partitions.leftRows.empty() || partitions.rightRows.empty())
	{
		return Node::createLeaf(getMajorityLabel(rowIndices), rowIndices.size());
	}

	// Internal node: walk left if feature <= threshold, else right.
	std::unique_ptr<Node> node =
		Node::createDecision(split.featureName, split.featureIndex,
							 split.threshold, rowIndices.size());

	// Same buildNode logic on smaller row sets — classic divide and conquer.
	node->leftChild = buildNode(partitions.leftRows, depth + 1);
	node->rightChild = buildNode(partitions.rightRows, depth + 1);

	return node;
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
