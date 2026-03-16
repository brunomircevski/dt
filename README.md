# C4.5 Decision Tree Classifier (dt)

A high-performance C# implementation of the C4.5 decision tree algorithm, featuring both a sequential and a highly optimized multithreaded version.

## Overview

The project provides two primary classifiers:
- **`C45Classifier`**: The standard, sequential implementation. Good for smaller datasets and basic CPU footprints.
- **`C45ClassifierMulti`**: A heavily optimized, parallelized drop-in replacement that scales incredibly well on multi-core hardware (16+ cores).

Both implement the same logic: building trees using information gain ratio for both continuous and categorical variables.

## Multithreading Architecture (C45ClassifierMulti)

The multi-threaded variant parallelizes work at multiple levels to maintain a high CPU saturation without incurring massive thread orchestration overhead:

1. **Parallel Attribute Evaluation**: The hottest loop in the algorithm (evaluating the gain ratio of each feature at each node) is processed simultaneously using `Parallel.For`.
2. **O(N log N) Continuous Splits**: Threshold scanning for continuous data is done in a single O(N) sweep after an initial sort, instead of the naïve O(N²).
3. **Parallel Subtree Construction**: Branches are built concurrently (`Parallel.Invoke` for binary splits, `Parallel.ForEach` for N-way splits). 

## Hardware Tuning

`C45ClassifierMulti` exposes two key mechanisms that should be tuned based on your hardware. These can be adjusted inside `C45ClassifierMulti.cs`.

### 1. Thread Pool Pre-Population (`SetMinThreads`)
By default, the .NET thread pool spins up new threads very slowly (~1 every 500ms). On massively parallel CPUs (like an Intel Core i7-14700KF with 28 threads), this causes the first 10+ seconds of the build to be artificially bottlenecked.
- **High-End CPUs (16+ cores)**: Leave the `ThreadPool.SetMinThreads(Environment.ProcessorCount * 2, ...)` call intact.
- **Low-End/Standard CPUs (4 to 8 cores)**: The default thread pool grows fast enough. You can change it to `Environment.ProcessorCount` or remove the call completely to reduce idle thread waste.

### 2. `ParallelChildThreshold`
Subtrees are only handed off to new threads if their estimated computational cost (`rows × remaining_attributes`) exceeds this threshold. If nodes are too small, parallelizing them adds more scheduling overhead than it saves.
- **High-End CPUs (20+ cores)**: Keep it low (e.g., `200`) so more tasks are dispatched and all 20+ threads are fed.
- **Mid-Range (8 cores)**: Tune up to `~500`.
- **Low-End (4 cores)**: Tune up to `~1500 - 2000` to avoid task-switching overhead overpowering the actual work.

## Usage

Both implementations share the same API geometry. Building against a dataset parsed by `CsvReader`:

```csharp
// Load data
var trainData = CsvReader.Read("data/dataset-train.csv");
var testData = CsvReader.Read("data/dataset-test.csv");

// Build (Swap C45Classifier with C45ClassifierMulti for speed)
var tree = C45ClassifierMulti.BuildTree(trainData);

// Print stats
C45ClassifierMulti.PrintStatistics(tree);

// Evaluate accuracy
C45ClassifierMulti.Evaluate(tree, testData);
```
