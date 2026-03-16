using System.Collections.Concurrent;
using System.Globalization;

namespace dt;

/// <summary>
/// Multithreaded C4.5 decision tree classifier.
/// Drop-in replacement for <see cref="C45Classifier"/>.
///
/// Parallelism strategy (v2)
/// ─────────────────────────
/// 1. Thread pool primed  – MinThreads is raised to ProcessorCount×2 so the
///    pool doesn't throttle thread injection for deeply nested parallel work.
///
/// 2. Attribute evaluation – Parallel.For over all attributes at each node.
///    Completely lock-free: results go into pre-allocated arrays.
///
/// 3. Child building
///    • Binary (continuous) splits  → Parallel.Invoke so the calling thread
///      handles one branch in-place (no wasted blocked thread).
///    • N-way (categorical) splits  → Parallel.ForEach with work-stealing.
///    • Below ParallelChildThreshold both fall back to sequential to avoid
///      task-overhead dominating tiny leaf work.
///
/// 4. GainRatioContinuous is now O(N log N) instead of O(N²)
///    via an incremental left-sweep that accumulates class counts in a single
///    pass after the initial sort, eliminating per-threshold full rescans.
/// </summary>
public static class C45ClassifierMulti
{
    private const int MinSamplesPerNode = 2;

    /// <summary>
    /// Fork parallel child tasks only when max(childRows) × childAttrs exceeds
    /// this value. Lower = more tasks, higher CPU usage, higher overhead.
    /// 200 is a good default for 20+ core machines and medium-to-large datasets.
    /// </summary>
    private const int ParallelChildThreshold = 200;

    public static long LastBuildTimeMs { get; private set; }

    // ─── Public API ────────────────────────────────────────────────────────

    /// <summary>Builds a C4.5 decision tree from the given dataset.</summary>
    public static Node BuildTree(Dataset ds)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();

        // Prime the thread pool: default slow-start (1 thread / 500 ms) would
        // starve a 20+ core machine for the first several seconds of the build.
        int desired = Environment.ProcessorCount * 2;
        ThreadPool.GetMinThreads(out int curWorkers, out int curIO);
        if (curWorkers < desired)
            ThreadPool.SetMinThreads(desired, curIO);

        var allRows  = Enumerable.Range(0, ds.Rows.Count).ToArray();
        var allAttrs = Enumerable.Range(0, ds.ColumnNames.Length)
                                 .Where(i => i != ds.ClassIndex)
                                 .ToArray();

        var tree = BuildNode(ds, allRows, allAttrs);

        watch.Stop();
        LastBuildTimeMs = watch.ElapsedMilliseconds;
        return tree;
    }

    /// <summary>Predicts the class for a given data row using the provided tree.</summary>
    public static string Predict(Node node, string[] columnNames, string[] row)
    {
        if (node.IsLeaf) return node.Label!;

        int attrIndex = Array.IndexOf(columnNames, node.Attribute);
        string value = row[attrIndex];

        if (node.Threshold.HasValue)
        {
            double dv = double.Parse(value, CultureInfo.InvariantCulture);
            string branch = dv <= node.Threshold.Value ? "<=" : ">";
            if (!node.Children.ContainsKey(branch)) return "Unknown";
            return Predict(node.Children[branch], columnNames, row);
        }

        return node.Children.TryGetValue(value, out var child)
            ? Predict(child, columnNames, row)
            : "Unknown";
    }

    /// <summary>Prints various statistics about the generated tree.</summary>
    public static void PrintStatistics(Node tree)
    {
        Console.WriteLine("Tree Statistics:");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine($"Total Nodes:    {tree.CountNodes()}");
        Console.WriteLine($"Leaf Nodes:     {tree.CountLeaves()}");
        Console.WriteLine($"Max Depth:      {tree.GetMaxDepth()}");
        Console.WriteLine($"Build Time:     {LastBuildTimeMs} ms");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine();
    }

    /// <summary>Evaluates the tree against a test dataset and prints the results.</summary>
    public static void Evaluate(Node tree, Dataset testData)
    {
        // Prediction is read-only → PLINQ with no contention.
        int correct = testData.Rows
            .AsParallel()
            .Count(row => Predict(tree, testData.ColumnNames, row) == row[testData.ClassIndex]);

        double accuracy = (double)correct / testData.Rows.Count * 100;
        Console.WriteLine("Evaluation Results:");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine($"Total Samples:  {testData.Rows.Count}");
        Console.WriteLine($"Correct:        {correct}");
        Console.WriteLine($"Accuracy:       {accuracy:F2}%");
        Console.WriteLine(new string('─', 40));
    }

    // ─── Core recursive builder ────────────────────────────────────────────

    private static Node BuildNode(Dataset ds, int[] rows, int[] attrs)
    {
        // ── Base cases ───────────────────────────────────────────────────
        if (rows.Length == 0)
            return new Node { Label = "Unknown" };

        var classes = rows.Select(r => ds.Rows[r][ds.ClassIndex]).Distinct().ToList();
        if (classes.Count == 1)
            return new Node { Label = classes[0] };

        if (attrs.Length == 0 || rows.Length < MinSamplesPerNode)
            return new Node { Label = MajorityClass(ds, rows) };

        // ── Parallel attribute evaluation (lock-free) ─────────────────────
        var gainRatios  = new double[attrs.Length];
        var thresholds  = new double?[attrs.Length];

        Parallel.For(0, attrs.Length, i =>
            gainRatios[i] = CalculateGainRatio(ds, rows, attrs[i], out thresholds[i]));

        // ── Pick best attribute ───────────────────────────────────────────
        int    bestIdx       = -1;
        double bestGainRatio = -1;
        for (int i = 0; i < attrs.Length; i++)
        {
            if (gainRatios[i] > bestGainRatio)
            { bestGainRatio = gainRatios[i]; bestIdx = i; }
        }

        if (bestIdx == -1 || bestGainRatio <= 0)
            return new Node { Label = MajorityClass(ds, rows) };

        int    bestAttr      = attrs[bestIdx];
        double? bestThreshold = thresholds[bestIdx];

        var node = new Node
        {
            Attribute = ds.ColumnNames[bestAttr],
            Threshold = bestThreshold,
            MajorityLabel = MajorityClass(ds, rows)
        };

        // ── Build children ────────────────────────────────────────────────
        if (bestThreshold.HasValue)
        {
            // Continuous: binary split → Parallel.Invoke reuses calling thread
            var (left, right) = SplitContinuous(ds, rows, bestAttr, bestThreshold.Value);

            bool fork = (long)Math.Max(left.Length, right.Length) * attrs.Length
                        > ParallelChildThreshold;

            if (fork)
            {
                Node leftNode = null!, rightNode = null!;
                // Parallel.Invoke: one branch on calling thread, one on pool thread.
                // Neither thread is blocked doing nothing — both do real work.
                Parallel.Invoke(
                    () => leftNode  = BuildNode(ds, left,  attrs),
                    () => rightNode = BuildNode(ds, right, attrs));

                node.Children["<="] = leftNode;
                node.Children[">"]  = rightNode;
            }
            else
            {
                node.Children["<="] = BuildNode(ds, left,  attrs);
                node.Children[">"]  = BuildNode(ds, right, attrs);
            }
        }
        else
        {
            // Categorical: N-way split → Parallel.ForEach with work-stealing
            var splits         = SplitCategorical(ds, rows, bestAttr);
            var remainingAttrs = attrs.Where(a => a != bestAttr).ToArray();

            bool fork = splits.Values.Any(s =>
                (long)s.Length * remainingAttrs.Length > ParallelChildThreshold);

            if (fork)
            {
                var results = new ConcurrentDictionary<string, Node>(
                    concurrencyLevel: Environment.ProcessorCount,
                    capacity: splits.Count);

                Parallel.ForEach(splits, kvp =>
                    results[kvp.Key] = BuildNode(ds, kvp.Value, remainingAttrs));

                foreach (var kvp in results)
                    node.Children[kvp.Key] = kvp.Value;
            }
            else
            {
                foreach (var kvp in splits)
                    node.Children[kvp.Key] = BuildNode(ds, kvp.Value, remainingAttrs);
            }
        }

        return node;
    }

    /// <summary>
    /// Prunes the tree using Reduced Error Pruning (REP) against a validation dataset.
    /// Returns the new (possibly pruned) root.
    /// </summary>
    public static Node Prune(Node tree, Dataset valData)
    {
        var valRows = Enumerable.Range(0, valData.Rows.Count).ToArray();
        return PruneNode(tree, valData, valRows);
    }

    private static Node PruneNode(Node node, Dataset valData, int[] valRows)
    {
        if (node.IsLeaf) return node;

        // Route validation rows to children
        var childRows = new Dictionary<string, List<int>>();
        foreach (var kvp in node.Children)
            childRows[kvp.Key] = new List<int>();

        int attrIndex = Array.IndexOf(valData.ColumnNames, node.Attribute);

        foreach (var r in valRows)
        {
            string value = valData.Rows[r][attrIndex];
            string branch;
            if (node.Threshold.HasValue)
            {
                double dValue = double.Parse(value, CultureInfo.InvariantCulture);
                branch = dValue <= node.Threshold.Value ? "<=" : ">";
            }
            else
            {
                branch = value;
            }

            if (childRows.ContainsKey(branch))
                childRows[branch].Add(r);
        }

        // Evaluate whether to prune children in parallel
        bool fork = valRows.Length > ParallelChildThreshold;
        var newChildren = new ConcurrentDictionary<string, Node>();

        if (fork && node.Children.Count > 1)
        {
            if (node.Threshold.HasValue)
            {
                // Binary split
                Node leftNode = null!, rightNode = null!;
                var leftRows = childRows["<="].ToArray();
                var rightRows = childRows[">"].ToArray();
                var leftChild = node.Children["<="];
                var rightChild = node.Children[">"];

                Parallel.Invoke(
                    () => leftNode = PruneNode(leftChild, valData, leftRows),
                    () => rightNode = PruneNode(rightChild, valData, rightRows)
                );

                newChildren["<="] = leftNode;
                newChildren[">"] = rightNode;
            }
            else
            {
                // N-way split
                Parallel.ForEach(node.Children, kvp =>
                {
                    var cRows = childRows[kvp.Key].ToArray();
                    newChildren[kvp.Key] = PruneNode(kvp.Value, valData, cRows);
                });
            }
        }
        else
        {
            // Sequential
            foreach (var kvp in node.Children)
            {
                var cRows = childRows[kvp.Key].ToArray();
                newChildren[kvp.Key] = PruneNode(kvp.Value, valData, cRows);
            }
        }

        node.Children = new Dictionary<string, Node>(newChildren);

        // Evaluate pruning this node
        if (valRows.Length == 0) return node;

        int correctWithSubtree = 0;
        int correctPruned = 0;

        foreach (var r in valRows)
        {
            var row = valData.Rows[r];
            var actual = row[valData.ClassIndex];
            var predicted = Predict(node, valData.ColumnNames, row);
            
            if (predicted == actual) correctWithSubtree++;
            if (node.MajorityLabel == actual) correctPruned++;
        }

        if (correctPruned >= correctWithSubtree)
        {
            // Prune: turn into leaf
            return new Node { Label = node.MajorityLabel };
        }

        return node;
    }

    /// <summary>
    /// Prints statistics comparing the original (pre-counted) and pruned trees.
    /// </summary>
    public static void PrintPruneStatistics(int origNodes, int origLeaves, int origDepth, Node pruned)
    {
        int prunNodes = pruned.CountNodes();
        int prunLeaves = pruned.CountLeaves();
        int prunDepth = pruned.GetMaxDepth();

        double reduction = origNodes == 0 ? 0 : (1.0 - (double)prunNodes / origNodes) * 100;

        Console.WriteLine("Pruning Results:");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine($"Original Nodes: {origNodes}");
        Console.WriteLine($"Pruned Nodes:   {prunNodes} (-{reduction:F2}%)");
        Console.WriteLine($"Original Leaves: {origLeaves}");
        Console.WriteLine($"Pruned Leaves:   {prunLeaves}");
        Console.WriteLine($"Original Depth: {origDepth}");
        Console.WriteLine($"Pruned Depth:   {prunDepth}");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine();
    }

    // ─── Gain ratio ────────────────────────────────────────────────────────

    private static double CalculateGainRatio(Dataset ds, int[] rows, int attr, out double? bestThreshold)
    {
        bestThreshold = null;
        return IsContinuous(ds, rows, attr)
            ? GainRatioContinuous(ds, rows, attr, out bestThreshold)
            : GainRatioCategorical(ds, rows, attr);
    }

    private static bool IsContinuous(Dataset ds, int[] rows, int attr)
    {
        foreach (var r in rows)
        {
            return double.TryParse(
                ds.Rows[r][attr], NumberStyles.Any, CultureInfo.InvariantCulture, out _);
        }
        return false;
    }

    // ─── Entropy helpers (counts-based, no allocations) ────────────────────

    /// <summary>Entropy from a pre-built count array.</summary>
    private static double EntropyFromCounts(int[] counts, int total)
    {
        if (total == 0) return 0;
        double h = 0, n = total;
        foreach (int c in counts)
        {
            if (c <= 0) continue;
            double p = c / n;
            h -= p * Math.Log2(p);
        }
        return h;
    }

    /// <summary>
    /// Entropy for the RIGHT partition = total[i] − left[i], without allocating
    /// a new array. This is the key enabler for the O(N) threshold sweep.
    /// </summary>
    private static double EntropyFromCountsDiff(int[] total, int[] left, int rightN)
    {
        if (rightN == 0) return 0;
        double h = 0, n = rightN;
        for (int i = 0; i < total.Length; i++)
        {
            int c = total[i] - left[i];
            if (c <= 0) continue;
            double p = c / n;
            h -= p * Math.Log2(p);
        }
        return h;
    }

    // ─── Categorical ────────────────────────────────────────────────────────

    private static double GainRatioCategorical(Dataset ds, int[] rows, int attr)
    {
        double baseEntropy = Entropy(ds, rows);
        var    splits      = SplitCategorical(ds, rows, attr);
        double total       = rows.Length;

        double weightedEntropy = 0, splitInfo = 0;
        foreach (var subset in splits.Values)
        {
            double w = subset.Length / total;
            weightedEntropy += w * Entropy(ds, subset);
            splitInfo       -= w * Math.Log2(w);
        }

        double infoGain = baseEntropy - weightedEntropy;
        return splitInfo == 0 ? 0 : infoGain / splitInfo;
    }

    private static Dictionary<string, int[]> SplitCategorical(Dataset ds, int[] rows, int attr)
    {
        var groups = new Dictionary<string, List<int>>();
        foreach (var r in rows)
        {
            string val = ds.Rows[r][attr];
            if (!groups.ContainsKey(val)) groups[val] = [];
            groups[val].Add(r);
        }
        return groups.ToDictionary(g => g.Key, g => g.Value.ToArray());
    }

    // ─── Continuous (O(N log N) incremental sweep) ──────────────────────────

    /// <summary>
    /// Evaluates all possible split thresholds in a SINGLE left-to-right pass
    /// after sorting, maintaining incremental class-count arrays.
    ///
    /// Complexity: O(N log N) for the sort + O(N·K) for the sweep,
    /// where K = number of distinct class labels (usually very small).
    /// The original implementation was O(N²·K) — for large nodes this is the
    /// dominant source of the speedup.
    /// </summary>
    private static double GainRatioContinuous(Dataset ds, int[] rows, int attr, out double? bestThreshold)
    {
        bestThreshold = null;

        // Sort rows by attribute value.
        var points = rows
            .Select(r => (Row: r, Val: double.Parse(ds.Rows[r][attr], CultureInfo.InvariantCulture)))
            .OrderBy(p => p.Val)
            .ToArray();

        if (points.Length < 2) return -1;

        // Build class → compact index mapping + total count array.
        var classToIdx = new Dictionary<string, int>(8);
        foreach (var r in rows)
        {
            var cls = ds.Rows[r][ds.ClassIndex];
            if (!classToIdx.ContainsKey(cls))
                classToIdx[cls] = classToIdx.Count;
        }

        int K = classToIdx.Count;
        int N = rows.Length;

        var totalCounts = new int[K];
        foreach (var r in rows)
            totalCounts[classToIdx[ds.Rows[r][ds.ClassIndex]]]++;

        double baseEntropy = EntropyFromCounts(totalCounts, N);

        // Single left-to-right sweep: accumulate leftCounts one row at a time.
        // Right counts are derived as totalCounts − leftCounts (no allocation).
        var leftCounts = new int[K];
        double bestRatio = -1;

        for (int i = 0; i < points.Length - 1; i++)
        {
            leftCounts[classToIdx[ds.Rows[points[i].Row][ds.ClassIndex]]]++;

            // Skip if next point has the same value (not a valid split boundary).
            if (points[i].Val == points[i + 1].Val) continue;

            int    leftN  = i + 1;
            int    rightN = N - leftN;
            double leftW  = (double)leftN / N;
            double rightW = (double)rightN / N;

            double leftEntropy  = EntropyFromCounts(leftCounts, leftN);
            double rightEntropy = EntropyFromCountsDiff(totalCounts, leftCounts, rightN);

            double infoGain  = baseEntropy - (leftW * leftEntropy + rightW * rightEntropy);
            double splitInfo = -(leftW * Math.Log2(leftW) + rightW * Math.Log2(rightW));

            if (splitInfo == 0) continue;
            double ratio = infoGain / splitInfo;

            if (ratio > bestRatio)
            {
                bestRatio     = ratio;
                bestThreshold = (points[i].Val + points[i + 1].Val) / 2.0;
            }
        }

        return bestRatio;
    }

    private static (int[] left, int[] right) SplitContinuous(Dataset ds, int[] rows, int attr, double threshold)
    {
        var left  = new List<int>(rows.Length / 2);
        var right = new List<int>(rows.Length / 2);
        foreach (var r in rows)
        {
            if (double.Parse(ds.Rows[r][attr], CultureInfo.InvariantCulture) <= threshold)
                left.Add(r);
            else
                right.Add(r);
        }
        return (left.ToArray(), right.ToArray());
    }

    // ─── Shared helpers ─────────────────────────────────────────────────────

    private static double Entropy(Dataset ds, int[] rows)
    {
        if (rows.Length == 0) return 0;
        var counts = ClassCounts(ds, rows);
        double total = rows.Length, h = 0;
        foreach (var c in counts.Values)
        {
            double p = c / total;
            h -= p * Math.Log2(p);
        }
        return h;
    }

    private static Dictionary<string, int> ClassCounts(Dataset ds, int[] rows)
    {
        var counts = new Dictionary<string, int>();
        foreach (var r in rows)
        {
            var label = ds.Rows[r][ds.ClassIndex];
            counts.TryAdd(label, 0);
            counts[label]++;
        }
        return counts;
    }

    private static string MajorityClass(Dataset ds, int[] rows)
        => ClassCounts(ds, rows).OrderByDescending(kv => kv.Value).First().Key;
}
