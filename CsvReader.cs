namespace dt;

/// <summary>
/// Holds the parsed dataset: column names and rows of string values.
/// The last column is assumed to be the class label.
/// </summary>
public class Dataset(string[] columnNames, List<string[]> rows)
{
    public string[] ColumnNames { get; } = columnNames;
    public List<string[]> Rows { get; } = rows;

    public int FeatureCount => ColumnNames.Length - 1;
    public int ClassIndex { get; set; } = columnNames.Length - 1; // Default to last
}

public static class CsvReader
{
    /// <summary>
    /// Reads a CSV file and returns a <see cref="Dataset"/>.
    /// Skips the first column if it is named "Id".
    /// </summary>
    public static Dataset Read(string path)
    {
        var lines = File.ReadAllLines(path)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToList();

        if (lines.Count == 0)
            throw new InvalidOperationException("CSV file is empty.");

        var header = lines[0].Split(',');
        var skipFirst = header[0].Equals("Id", StringComparison.OrdinalIgnoreCase);
        var startIndex = skipFirst ? 1 : 0;

        var columnNames = header[startIndex..];

        var rows = new List<string[]>();
        for (int i = 1; i < lines.Count; i++)
        {
            var parts = lines[i].Split(',');
            rows.Add(parts[startIndex..]);
        }

        return new Dataset(columnNames, rows);
    }
}
