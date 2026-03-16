using System.Globalization;

namespace dt;

public static class C45Classifier
{
    private const int MinSamplesPerNode = 2;

    public static long LastBuildTimeMs { get; private set; }

    /// <summary>
    /// Builds a C4.5 decision tree from the given dataset.
    /// </summary>
    public static Node BuildTree(Dataset ds)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        
        var allRows = Enumerable.Range(0, ds.Rows.Count).ToArray();
        // Features are all attributes except the class index.
        var allAttrs = Enumerable.Range(0, ds.ColumnNames.Length)
            .Where(i => i != ds.ClassIndex)
            .ToArray();
            
        var tree = BuildTree(ds, allRows, allAttrs);
        
        watch.Stop();
        LastBuildTimeMs = watch.ElapsedMilliseconds;
        
        return tree;
    }

    /// <summary>
    /// Predicts the class for a given data row using the provided tree.
    /// </summary>
    public static string Predict(Node node, string[] columnNames, string[] row)
    {
        if (node.IsLeaf) return node.Label!;

        int attrIndex = Array.IndexOf(columnNames, node.Attribute);
        string value = row[attrIndex];

        if (node.Threshold.HasValue)
        {
            // Continuous split
            double dValue = double.Parse(value, CultureInfo.InvariantCulture);
            string branch = dValue <= node.Threshold.Value ? "<=" : ">";
            if (!node.Children.ContainsKey(branch)) return "Unknown"; // Should not happen with training data
            return Predict(node.Children[branch], columnNames, row);
        }
        else
        {
            // Categorical split
            if (node.Children.TryGetValue(value, out var child))
            {
                return Predict(child, columnNames, row);
            }
            // Fallback: if value unseen in training, we'd ideally return majority class of parent, 
            // but for now let's just return a placeholder or the most common child.
            return "Unknown"; 
        }
    }

    /// <summary>
    /// Prints various statistics about the generated tree.
    /// </summary>
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

    /// <summary>
    /// Evaluates the tree against a test dataset and prints the results.
    /// </summary>
    public static void Evaluate(Node tree, Dataset testData)
    {
        int correct = 0;
        foreach (var row in testData.Rows)
        {
            string predicted = Predict(tree, testData.ColumnNames, row);
            string actual = row[testData.ClassIndex];
            if (predicted == actual) correct++;
        }

        double accuracy = (double)correct / testData.Rows.Count * 100;
        Console.WriteLine($"Evaluation Results:");
        Console.WriteLine(new string('─', 40));
        Console.WriteLine($"Total Samples:  {testData.Rows.Count}");
        Console.WriteLine($"Correct:        {correct}");
        Console.WriteLine($"Accuracy:       {accuracy:F2}%");
        Console.WriteLine(new string('─', 40));
    }

    private static Node BuildTree(Dataset ds, int[] rows, int[] attrs)
    {
        if (rows.Length == 0) return new Node { Label = "Unknown" };

        var classes = rows.Select(r => ds.Rows[r][ds.ClassIndex]).Distinct().ToList();
        if (classes.Count == 1)
            return new Node { Label = classes[0] };

        if (attrs.Length == 0 || rows.Length < MinSamplesPerNode)
            return new Node { Label = MajorityClass(ds, rows) };

        int bestAttr = -1;
        double bestGainRatio = -1;
        double? bestThreshold = null;

        foreach (var attr in attrs)
        {
            double gainRatio = CalculateGainRatio(ds, rows, attr, out double? threshold);
            if (gainRatio > bestGainRatio)
            {
                bestGainRatio = gainRatio;
                bestAttr = attr;
                bestThreshold = threshold;
            }
        }

        if (bestAttr == -1 || bestGainRatio <= 0)
            return new Node { Label = MajorityClass(ds, rows) };

        var node = new Node
        {
            Attribute = ds.ColumnNames[bestAttr],
            Threshold = bestThreshold,
            MajorityLabel = MajorityClass(ds, rows)
        };

        if (bestThreshold.HasValue)
        {
            // Continuous
            var (left, right) = SplitContinuous(ds, rows, bestAttr, bestThreshold.Value);
            node.Children["<="] = BuildTree(ds, left, attrs);
            node.Children[">"] = BuildTree(ds, right, attrs);
        }
        else
        {
            // Categorical
            var splits = SplitCategorical(ds, rows, bestAttr);
            // After a categorical split, this attribute shouldn't be reused in subtrees.
            var remainingAttrs = attrs.Where(a => a != bestAttr).ToArray();
            foreach (var kvp in splits)
            {
                node.Children[kvp.Key] = BuildTree(ds, kvp.Value, remainingAttrs);
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

        // Recursively prune children
        var newChildren = new Dictionary<string, Node>();
        foreach (var kvp in node.Children)
        {
            var cRows = childRows[kvp.Key].ToArray();
            newChildren[kvp.Key] = PruneNode(kvp.Value, valData, cRows);
        }
        node.Children = newChildren;

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

    private static double CalculateGainRatio(Dataset ds, int[] rows, int attr, out double? bestThreshold)
    {
        bestThreshold = null;
        bool isContinuous = IsContinuous(ds, rows, attr);

        if (isContinuous)
        {
            return GainRatioContinuous(ds, rows, attr, out bestThreshold);
        }
        else
        {
            return GainRatioCategorical(ds, rows, attr);
        }
    }

    private static bool IsContinuous(Dataset ds, int[] rows, int attr)
    {
        // Check a sample to see if it's numeric.
        foreach(var r in rows)
        {
            if (double.TryParse(ds.Rows[r][attr], NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                return true;
            break;
        }
        return false;
    }

    // ─── Entropy ────────────────────────────────────────────────────────

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

    // ─── Categorical Logic ─────────────────────────────────────────────

    private static double GainRatioCategorical(Dataset ds, int[] rows, int attr)
    {
        double baseEntropy = Entropy(ds, rows);
        var splits = SplitCategorical(ds, rows, attr);
        double total = rows.Length;
        
        double weightedEntropy = 0;
        double splitInfo = 0;

        foreach(var subset in splits.Values)
        {
            double weight = subset.Length / total;
            weightedEntropy += weight * Entropy(ds, subset);
            splitInfo -= weight * Math.Log2(weight);
        }

        double infoGain = baseEntropy - weightedEntropy;
        if (splitInfo == 0) return 0;

        return infoGain / splitInfo;
    }

    private static Dictionary<string, int[]> SplitCategorical(Dataset ds, int[] rows, int attr)
    {
        var groups = new Dictionary<string, List<int>>();
        foreach(var r in rows)
        {
            string val = ds.Rows[r][attr];
            if (!groups.ContainsKey(val)) groups[val] = new List<int>();
            groups[val].Add(r);
        }
        return groups.ToDictionary(g => g.Key, g => g.Value.ToArray());
    }

    // ─── Continuous Logic ──────────────────────────────────────────────

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
        {
            if (points[i].Val != points[i - 1].Val)
                thresholds.Add((points[i].Val + points[i - 1].Val) / 2.0);
        }

        if (thresholds.Count == 0) return -1;

        double bestRatio = -1;

        foreach (var t in thresholds)
        {
            var left = points.Where(p => p.Val <= t).Select(p => p.Row).ToArray();
            var right = points.Where(p => p.Val > t).Select(p => p.Row).ToArray();

            if (left.Length == 0 || right.Length == 0) continue;

            double total = rows.Length;
            double leftW = left.Length / total;
            double rightW = right.Length / total;

            double infoGain = baseEntropy - (leftW * Entropy(ds, left) + rightW * Entropy(ds, right));
            double splitInfo = -(leftW * Math.Log2(leftW) + rightW * Math.Log2(rightW));

            if (splitInfo == 0) continue;
            double ratio = infoGain / splitInfo;

            if (ratio > bestRatio)
            {
                bestRatio = ratio;
                bestThreshold = t;
            }
        }

        return bestRatio;
    }

    private static (int[] left, int[] right) SplitContinuous(Dataset ds, int[] rows, int attr, double threshold)
    {
        var left = new List<int>();
        var right = new List<int>();

        foreach (var r in rows)
        {
            double value = double.Parse(ds.Rows[r][attr], CultureInfo.InvariantCulture);
            if (value <= threshold)
                left.Add(r);
            else
                right.Add(r);
        }

        return (left.ToArray(), right.ToArray());
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
    {
        return ClassCounts(ds, rows)
            .OrderByDescending(kv => kv.Value)
            .First()
            .Key;
    }
}
