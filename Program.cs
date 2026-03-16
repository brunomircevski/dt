using dt;

const string TrainPath = "data/covtype-train-tiny.csv";
const string TestPath = "data/covtype-test-tiny.csv";

var trainData = CsvReader.Read(TrainPath);
var testData = CsvReader.Read(TestPath);

Console.WriteLine($"Loaded {trainData.Rows.Count} training rows, {testData.Rows.Count} test rows.");
Console.WriteLine($"Features: {trainData.FeatureCount}");
Console.WriteLine();

Console.WriteLine("Building tree...");
var tree = C45Classifier.BuildTree(trainData);

C45Classifier.PrintStatistics(tree);

Console.WriteLine("Evaluating tree...");
C45Classifier.Evaluate(tree, testData);
