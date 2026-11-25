// Weka J48 for the pruned-accuracy benchmark: one fit gives both trees.
// J48Fit's phased build (bench/tools/adapters/J48Fit.java: its depth limit,
// J48's own C45ModelSelection and C45PruneableClassifierTree) is run step by
// step: grow, collapse -> the unpruned tree (what J48 -U builds: -U changes
// only whether prune() runs), evaluated; then prune() (error-based, CF,
// subtree raising) -> the pruned tree, evaluated.
//
//   java -cp weka.jar:bounce.jar:classes J48Acc train=STEM test=STEM n_train=N n_test=M \
//       n_features=F n_classes=K options="-C 0.25 -M 2" max_depth=D pred=PREFIX
//
// Prints one JSON array (unpruned, pruned); PREFIX-unpruned.i16 and
// PREFIX-pruned.i16: the test predictions, int16 little-endian. The test rows
// are streamed from the file one at a time (never held: SOREL's would take
// 80 GB as Weka doubles).
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Locale;
import java.util.Map;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class J48Acc {
    static Map<String, String> args = J48Fit.args;

    static String evaluate(C45PruneableClassifierTree root, Instances train, Instances header, int features,
                           boolean pruned, double seconds) throws Exception {
        String text = root.toString();
        int depth = 0;
        for (String line : text.split("\n")) {
            if (line.contains(" <= ") || line.contains(" > ")) {
                depth = Math.max(depth, line.split("\\|", -1).length);
            }
        }
        int correct = 0;
        for (Instance row : train) if (root.classifyInstance(row) == row.classValue()) correct++;
        double trainAcc = (double) correct / train.numInstances();
        int testRows = Integer.parseInt(args.get("n_test"));
        String pred = args.get("pred") + (pruned ? "-pruned.i16" : "-unpruned.i16");
        correct = 0;
        try (J48Fit.Reader x = new J48Fit.Reader(args.get("test") + ".f32");
             J48Fit.Reader y = new J48Fit.Reader(args.get("test") + ".y.i32");
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
                     new FileOutputStream(pred), 1 << 20))) {
            double[] values = new double[features + 1];
            for (int i = 0; i < testRows; i++) {
                for (int j = 0; j < features; j++) values[j] = x.getFloat();
                int label = y.getInt();
                values[features] = label;
                Instance row = new DenseInstance(1.0, values.clone());
                row.setDataset(header);
                int p = (int) root.classifyInstance(row);
                if (p == label) correct++;
                out.writeShort(Short.reverseBytes((short) p));
            }
        }
        return String.format(Locale.ROOT,
                "{\"pruned\": %s, \"fit_seconds\": %.3f, \"nodes\": %d, \"leaves\": %d, \"depth\": %d, "
                        + "\"train_accuracy\": %.9f, \"test_accuracy\": %.9f, \"pred\": \"%s\"}",
                pruned, seconds, root.numNodes(), root.numLeaves(), depth, trainAcc,
                (double) correct / testRows, pred);
    }

    public static void main(String[] argv) throws Exception {
        for (String arg : argv) {
            int eq = arg.indexOf('=');
            args.put(arg.substring(0, eq), arg.substring(eq + 1));
        }
        int features = Integer.parseInt(args.get("n_features"));
        int classes = Integer.parseInt(args.get("n_classes"));
        Instances train = J48Fit.load("train", Integer.parseInt(args.get("n_train")), features, classes);
        Instances header = new Instances(train, 0);
        J48 j48 = J48Fit.options();
        if (j48.getUnpruned() || j48.getReducedErrorPruning() || j48.getBinarySplits()
                || !j48.getCollapseTree()) {
            throw new IllegalArgumentException("J48Acc: J48's default pruned tree only (-C CF -M N)");
        }
        // J48Fit.build's steps, the unpruned tree evaluated before prune().
        long start = System.nanoTime();
        C45ModelSelection selection = new C45ModelSelection(j48.getMinNumObj(), train,
                j48.getUseMDLcorrection(), j48.getDoNotMakeSplitPointActualValue());
        boolean cleanup = !j48.getSaveInstanceData();
        C45PruneableClassifierTree root = new J48Fit.DepthLimitedTree(selection, true, j48.getConfidenceFactor(),
                j48.getSubtreeRaising(), cleanup, true, 0, Integer.parseInt(args.get("max_depth")));
        Instances data = new Instances(train);
        data.deleteWithMissingClass();
        root.buildTree(data, j48.getSubtreeRaising() || !cleanup);
        root.collapse();
        double grown = (System.nanoTime() - start) / 1e9;
        String unpruned = evaluate(root, train, header, features, false, grown);
        start = System.nanoTime();
        root.prune();
        if (cleanup) root.cleanup(new Instances(data, 0));
        selection.cleanup();
        double pruning = (System.nanoTime() - start) / 1e9;
        String pruned = evaluate(root, train, header, features, true, grown + pruning);
        System.out.println("[" + unpruned + ", " + pruned + "]");
    }
}
