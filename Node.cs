namespace dt;

public class Node
{
    /// <summary>Name of the attribute used for splitting (null for leaf nodes).</summary>
    public string? Attribute { get; set; }

    /// <summary>Threshold value for continuous attribute splits.</summary>
    public double? Threshold { get; set; }

    /// <summary>Predicted class label (set only for leaf nodes).</summary>
    public string? Label { get; set; }

    /// <summary>Child branches keyed by "&lt;=" or "&gt;".</summary>
    public Dictionary<string, Node> Children { get; set; } = new();

    public bool IsLeaf => Label is not null;

    public int CountNodes() => 1 + Children.Values.Sum(c => c.CountNodes());

    public int GetMaxDepth()
    {
        if (IsLeaf || Children.Count == 0) return 1;
        return 1 + Children.Values.Max(c => c.GetMaxDepth());
    }

    public int CountLeaves()
    {
        if (IsLeaf) return 1;
        return Children.Values.Sum(c => c.CountLeaves());
    }
}
