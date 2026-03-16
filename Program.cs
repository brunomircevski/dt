using dt;

const string TrainPath = "data/covtype-small-train.csv";
const string ValPath = "data/covtype-small-val.csv";
const string TestPath = "data/covtype-small-test.csv";

var trainData = CsvReader.Read(TrainPath);
var valData = CsvReader.Read(ValPath);
var testData = CsvReader.Read(TestPath);

Console.WriteLine($"Loaded {trainData.Rows.Count} training rows, {valData.Rows.Count} validation rows, {testData.Rows.Count} test rows.");
Console.WriteLine($"Features: {trainData.FeatureCount}\n");

Console.WriteLine("Building tree...");

var tree = C45ClassifierMulti.BuildTree(trainData);
C45ClassifierMulti.PrintStatistics(tree);

Console.WriteLine("Evaluating Original Tree...");
C45ClassifierMulti.Evaluate(tree, testData);

Console.WriteLine("Pruning tree...");
int origNodes = tree.CountNodes();
int origLeaves = tree.CountLeaves();
int origDepth = tree.GetMaxDepth();

var prunedTree = C45ClassifierMulti.Prune(tree, valData);
C45ClassifierMulti.PrintPruneStatistics(origNodes, origLeaves, origDepth, prunedTree);

Console.WriteLine("Evaluating Pruned Tree...");
C45ClassifierMulti.Evaluate(prunedTree, testData);
