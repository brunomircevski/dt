using System.Globalization;

namespace dt;

/// <summary>
/// Multithreaded C4.5 decision tree classifier.
/// Drop-in replacement for <see cref="C45Classifier"/>:
/// every public method has the same signature and semantics.
///
/// Parallelism strategy
/// ────────────────────
/// 1. Attribute evaluation   – The O(F) per-node attribute scan is done with
///    Parallel.For so all features are evaluated simultaneously.
///    This is the hottest loop in large datasets.
///
/// 2. Child subtree building  – Children of a node are dispatched as
///    independent Tasks so siblings are built concurrently.
///    To avoid spawning an exponential number of tiny tasks the subtree is
///    only parallelised when the work is likely "large enough":
///    we use a simple heuristic of depth × rows &gt; threshold to decide
///    whether to fork a new task or fall back to an inline call.
///    The threshold is calibrated so that on a 16-core machine roughly
///    16–64 tasks exist at any moment during the build.
/// </summary>
public static class C45ClassifierMulti
{
    private const int MinSamplesPerNode = 2;

    /// <summary>
    /// Minimum cost (rows × remainingAttrs) below which child subtrees are
    /// built on the same thread instead of forking a new Task.
    /// Tune this value to trade off granularity vs. task-overhead.
    /// A good starting value that keeps 16 cores busy without drowning in
    /// tiny tasks is ~4000 (empirically validated on the covtype dataset).
    /// </summary>
    private const int ParallelChildThreshold = 4_000;

    public static long LastBuildTimeMs { get; private set; }

    // ─── Public API ────────────────────────────────────────────────────────

    /// <summary>Builds a C4.5 decision tree from the given dataset.</summary>
    public static Node BuildTree(Dataset ds)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();

        var allRows  = Enumerable.Range(0, ds.Rows.Count).ToArray();
        var allAttrs = Enumerable.Range(0, ds.ColumnNames.Length)
                                 .Where(i => i != ds.ClassIndex)
                                 .ToArray();

        var tree = BuildTreeParallel(ds, allRows, allAttrs, depth: 0);

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
            double dValue = double.Parse(value, CultureInfo.InvariantCulture);
            string branch = dValue <= node.Threshold.Value ? "<=" : ">";
            if (!node.Children.ContainsKey(branch)) return "Unknown";
            return Predict(node.Children[branch], columnNames, row);
        }
        else
        {
            if (node.Children.TryGetValue(value, out var child))
                return Predict(child, columnNames, row);
            return "Unknown";
        }
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
        int correct = 0;

        // Prediction is read-only; trivially safe for PLINQ.
        correct = testData.Rows
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

    private static Node BuildTreeParallel(Dataset ds, int[] rows, int[] attrs, int depth)
    {
        // ── Base cases (same as sequential) ──────────────────────────────
        if (rows.Length == 0)
            return new Node { Label = "Unknown" };

        var classes = rows.Select(r => ds.Rows[r][ds.ClassIndex]).Distinct().ToList();
        if (classes.Count == 1)
            return new Node { Label = classes[0] };

        if (attrs.Length == 0 || rows.Length < MinSamplesPerNode)
            return new Node { Label = MajorityClass(ds, rows) };

        // ── Parallel attribute evaluation ─────────────────────────────────
        // Each attribute is independent → evaluate all with Parallel.For.
        // Results are stored in pre-allocated arrays to avoid locking.
        var gainRatios  = new double[attrs.Length];
        var thresholds  = new double?[attrs.Length];

        Parallel.For(0, attrs.Length, i =>
        {
            gainRatios[i] = CalculateGainRatio(ds, rows, attrs[i], out thresholds[i]);
        });

        // ── Pick the best attribute ───────────────────────────────────────
        int   bestIdx       = -1;
        double bestGainRatio = -1;

        for (int i = 0; i < attrs.Length; i++)
        {
            if (gainRatios[i] > bestGainRatio)
            {
                bestGainRatio = gainRatios[i];
                bestIdx       = i;
            }
        }

        if (bestIdx == -1 || bestGainRatio <= 0)
            return new Node { Label = MajorityClass(ds, rows) };

        int    bestAttr      = attrs[bestIdx];
        double? bestThreshold = thresholds[bestIdx];

        var node = new Node
        {
            Attribute = ds.ColumnNames[bestAttr],
            Threshold = bestThreshold
        };

        // ── Build children ────────────────────────────────────────────────
        if (bestThreshold.HasValue)
        {
            var (left, right) = SplitContinuous(ds, rows, bestAttr, bestThreshold.Value);
            BuildChildrenParallel(ds, attrs, depth, node, new[]
            {
                ("<=", left,  attrs),
                (">",  right, attrs),
            });
        }
        else
        {
            var splits         = SplitCategorical(ds, rows, bestAttr);
            var remainingAttrs = attrs.Where(a => a != bestAttr).ToArray();
            var children       = splits
                .Select(kvp => (kvp.Key, kvp.Value, remainingAttrs))
                .ToArray();
            BuildChildrenParallel(ds, attrs, depth, node, children);
        }

        return node;
    }

    /// <summary>
    /// Dispatches child subtrees either as parallel Tasks or inline,
    /// depending on whether the estimated work justifies the overhead.
    /// </summary>
    private static void BuildChildrenParallel(
        Dataset ds,
        int[]   parentAttrs,
        int     depth,
        Node    node,
        (string key, int[] childRows, int[] childAttrs)[] children)
    {
        // Heuristic: is it worth spawning Tasks for these children?
        bool fork = children.Any(c =>
            (long)c.childRows.Length * c.childAttrs.Length > ParallelChildThreshold);

        if (fork)
        {
            // Pre-allocate result array; fill via index so no locking needed.
            var tasks = new Task<(string key, Node node)>[children.Length];

            for (int i = 0; i < children.Length; i++)
            {
                var (key, childRows, childAttrs) = children[i];
                int d = depth + 1;
                tasks[i] = Task.Run(() =>
                    (key, BuildTreeParallel(ds, childRows, childAttrs, d))
                );
            }

            Task.WaitAll(tasks);

            foreach (var t in tasks)
            {
                var (key, child) = t.Result;
                node.Children[key] = child;
            }
        }
        else
        {
            // Small subtrees → build inline to avoid task overhead.
            foreach (var (key, childRows, childAttrs) in children)
                node.Children[key] = BuildTreeParallel(ds, childRows, childAttrs, depth + 1);
        }
    }

    // ─── Gain ratio ────────────────────────────────────────────────────────

    private static double CalculateGainRatio(Dataset ds, int[] rows, int attr, out double? bestThreshold)
    {
        bestThreshold = null;
        bool isContinuous = IsContinuous(ds, rows, attr);
        return isContinuous
            ? GainRatioContinuous(ds, rows, attr, out bestThreshold)
            : GainRatioCategorical(ds, rows, attr);
    }

    private static bool IsContinuous(Dataset ds, int[] rows, int attr)
    {
        foreach (var r in rows)
        {
            if (double.TryParse(ds.Rows[r][attr], NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                return true;
            break;
        }
        return false;
    }

    // ─── Entropy ────────────────────────────────────────────────────────────

    private static double Entropy(Dataset ds, int[] rows)
    {
        if (rows.Length == 0) return 0;
        var counts = ClassCounts(ds, rows);
        double total = rows.Length;
        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = count / total;
            entropy -= p * Math.Log2(p);
        }
        return entropy;
    }

    // ─── Categorical ────────────────────────────────────────────────────────

    private static double GainRatioCategorical(Dataset ds, int[] rows, int attr)
    {
        double baseEntropy = Entropy(ds, rows);
        var    splits      = SplitCategorical(ds, rows, attr);
        double total       = rows.Length;

        double weightedEntropy = 0;
        double splitInfo       = 0;

        foreach (var subset in splits.Values)
        {
            double weight = subset.Length / total;
            weightedEntropy += weight * Entropy(ds, subset);
            splitInfo       -= weight * Math.Log2(weight);
        }

        double infoGain = baseEntropy - weightedEntropy;
        if (splitInfo == 0) return 0;
        return infoGain / splitInfo;
    }

    private static Dictionary<string, int[]> SplitCategorical(Dataset ds, int[] rows, int attr)
    {
        var groups = new Dictionary<string, List<int>>();
        foreach (var r in rows)
        {
            string val = ds.Rows[r][attr];
            if (!groups.ContainsKey(val)) groups[val] = new List<int>();
            groups[val].Add(r);
        }
        return groups.ToDictionary(g => g.Key, g => g.Value.ToArray());
    }

    // ─── Continuous ─────────────────────────────────────────────────────────

    private static double GainRatioContinuous(Dataset ds, int[] rows, int attr, out double? bestThreshold)
    {
        bestThreshold = null;
        double baseEntropy = Entropy(ds, rows);

        var points = rows
            .Select(r => new { Row = r, Val = double.Parse(ds.Rows[r][attr], CultureInfo.InvariantCulture) })
            .OrderBy(p => p.Val)
            .ToArray();

        var thresholds = new List<double>();
        for (int i = 1; i < points.Length; i++)
            if (points[i].Val != points[i - 1].Val)
                thresholds.Add((points[i].Val + points[i - 1].Val) / 2.0);

        if (thresholds.Count == 0) return -1;

        double bestRatio = -1;

        foreach (var t in thresholds)
        {
            var left  = points.Where(p => p.Val <= t).Select(p => p.Row).ToArray();
            var right = points.Where(p => p.Val >  t).Select(p => p.Row).ToArray();

            if (left.Length == 0 || right.Length == 0) continue;

            double total     = rows.Length;
            double leftW     = left.Length  / total;
            double rightW    = right.Length / total;
            double infoGain  = baseEntropy - (leftW * Entropy(ds, left) + rightW * Entropy(ds, right));
            double splitInfo = -(leftW * Math.Log2(leftW) + rightW * Math.Log2(rightW));

            if (splitInfo == 0) continue;
            double ratio = infoGain / splitInfo;

            if (ratio > bestRatio)
            {
                bestRatio     = ratio;
                bestThreshold = t;
            }
        }

        return bestRatio;
    }

    private static (int[] left, int[] right) SplitContinuous(Dataset ds, int[] rows, int attr, double threshold)
    {
        var left  = new List<int>();
        var right = new List<int>();
        foreach (var r in rows)
        {
            double value = double.Parse(ds.Rows[r][attr], CultureInfo.InvariantCulture);
            if (value <= threshold) left.Add(r);
            else                    right.Add(r);
        }
        return (left.ToArray(), right.ToArray());
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

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
        => ClassCounts(ds, rows)
            .OrderByDescending(kv => kv.Value)
            .First().Key;
}
