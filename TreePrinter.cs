namespace dt;

public static class TreePrinter
{
    /// <summary>
    /// Prints the decision tree to the console in an indented text format.
    /// </summary>
    public static void Print(Node node, string indent = "")
    {
        if (node.IsLeaf)
        {
            Console.WriteLine($"{indent}→ {node.Label}");
            return;
        }

        foreach (var (branch, child) in node.Children)
        {
            Console.WriteLine($"{indent}{node.Attribute} {branch} {node.Threshold}");
            Print(child, indent + "  ");
        }
    }
}
