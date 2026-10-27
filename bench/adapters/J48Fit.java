// Weka J48 adapter: fit one protocol on a prepared dataset, print one JSON line.
//
//   java -cp weka.jar:bounce.jar:classes J48Fit data=DIR n_train=N n_test=M \
//       n_features=F n_classes=K warmup=ROWS eval=0|1 options="-C 0.25 -M 2" \
//       [max_depth=D]
//
// The data are read from the binary files straight into Weka Instances (no
// ARFF/CSV parser, whose temporary objects would dominate peak memory), in
// 16 MB blocks (files over 2 GB: airline, ECBDL, chess 1000M).
// max_depth=D: J48 has no depth limit, so the adapter grows the tree with a
// subclass of J48's own tree that makes every node at depth D (the root is at
// depth 0) a leaf, as ./tree -d D and scikit-learn's max_depth do; everything
// else is J48's code. It then always takes the phased path below, with or
// without -U (unpruned: growing, then only the collapse).
// warmup=ROWS: one untimed build on the first ROWS training rows first, so the
// JIT has compiled the hot code (0 = none). Timed: J48.buildClassifier's work,
// in two parts: growing (build_seconds) and C4.5's collapse, pruning and
// cleanup (prune_seconds); train_seconds is their sum. The two parts are the
// steps of C45PruneableClassifierTree.buildClassifier, called one by one
// (pruned trees only: -R, -B and -U go through J48 untouched). check=1: also
// build with plain J48 and fail unless the trees print the same. Loading and
// prediction are not timed; with eval=1 the training and test accuracy are
// measured after the timed build. The peak resident memory (VmHWM) is read
// right after the build, before evaluation.
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class J48Fit {
    static Map<String, String> args = new HashMap<>();

    // A binary file read in order, one 16 MB block at a time.
    static class Reader implements AutoCloseable {
        final FileChannel channel;
        final ByteBuffer block = ByteBuffer.allocate(1 << 24).order(ByteOrder.LITTLE_ENDIAN);

        Reader(String file) throws IOException {
            channel = FileChannel.open(Path.of(args.get("data"), file));
            block.flip();
        }

        void need(int bytes) throws IOException {
            if (block.remaining() >= bytes) return;
            block.compact();
            while (block.position() < bytes) {
                if (channel.read(block) < 0) throw new IOException("unexpected end of file");
            }
            block.flip();
        }

        float getFloat() throws IOException { need(4); return block.getFloat(); }
        int getInt() throws IOException { need(4); return block.getInt(); }
        public void close() throws IOException { channel.close(); }
    }

    static Instances load(String part, int rows, int features, int classes) throws IOException {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int j = 0; j < features; j++) attributes.add(new Attribute("f" + j));
        ArrayList<String> labels = new ArrayList<>();
        for (int k = 0; k < classes; k++) labels.add("c" + k);
        attributes.add(new Attribute("class", labels));
        Instances data = new Instances(part, attributes, rows);
        data.setClassIndex(features);
        try (Reader x = new Reader(part + ".f32"); Reader y = new Reader(part + ".y.i32")) {
            for (int i = 0; i < rows; i++) {
                double[] values = new double[features + 1];
                for (int j = 0; j < features; j++) values[j] = x.getFloat();
                values[features] = y.getInt();
                data.add(new DenseInstance(1.0, values));
            }
        }
        return data;
    }

    static J48 options() throws Exception {
        J48 tree = new J48();
        tree.setOptions(Utils.splitOptions(args.get("options")));
        return tree;
    }

    // Whether build() can time growing and pruning apart: J48's default
    // pruned tree, the path of J48.buildClassifier copied in build(); with
    // max_depth also the unpruned tree.
    static boolean phased(J48 j48) {
        return (!j48.getUnpruned() || args.containsKey("max_depth"))
                && !j48.getReducedErrorPruning() && !j48.getBinarySplits();
    }

    // Splits nothing: every node it selects for is a leaf.
    static class NoSplitSelection extends ModelSelection {
        public ClassifierSplitModel selectModel(Instances data) throws Exception {
            return new NoSplit(new Distribution(data));
        }

        public ClassifierSplitModel selectModel(Instances train, Instances test) throws Exception {
            return selectModel(train);
        }

        public String getRevision() { return "J48Fit"; }
    }

    // J48's tree, with every node at depth maxDepth a leaf. The children are
    // made by getNewTree, as in C45PruneableClassifierTree, one level deeper.
    static class DepthLimitedTree extends C45PruneableClassifierTree {
        final ModelSelection selection;
        final boolean pruneTree, raiseTree, cleanup, collapseTree;
        final float cf;
        final int depth, maxDepth;

        DepthLimitedTree(ModelSelection selection, boolean pruneTree, float cf, boolean raiseTree,
                         boolean cleanup, boolean collapseTree, int depth, int maxDepth) throws Exception {
            super(depth < maxDepth ? selection : new NoSplitSelection(), pruneTree, cf, raiseTree,
                  cleanup, collapseTree);
            this.selection = selection;
            this.pruneTree = pruneTree;
            this.cf = cf;
            this.raiseTree = raiseTree;
            this.cleanup = cleanup;
            this.collapseTree = collapseTree;
            this.depth = depth;
            this.maxDepth = maxDepth;
        }

        @Override
        protected ClassifierTree getNewTree(Instances data) throws Exception {
            DepthLimitedTree tree = new DepthLimitedTree(selection, pruneTree, cf, raiseTree, cleanup,
                    collapseTree, depth + 1, maxDepth);
            tree.buildTree(data, raiseTree || !cleanup);
            return tree;
        }
    }

    // The tree and the seconds growing it and pruning it (the latter 0 when not phased).
    interface Predictor { double classify(Instance row) throws Exception; }

    // A class, not a record: the nodes have Java 11.
    static class Built {
        final Predictor model;
        final J48 j48;
        final ClassifierTree root;
        final double build, prune;

        Built(Predictor model, J48 j48, ClassifierTree root, double build, double prune) {
            this.model = model;
            this.j48 = j48;
            this.root = root;
            this.build = build;
            this.prune = prune;
        }
    }

    static Built build(Instances train) throws Exception {
        J48 j48 = options();
        if (!phased(j48)) {
            long start = System.nanoTime();
            j48.buildClassifier(train);
            return new Built(j48::classifyInstance, j48, null, (System.nanoTime() - start) / 1e9, 0);
        }
        // J48.buildClassifier, then C45PruneableClassifierTree.buildClassifier.
        long start = System.nanoTime();
        j48.getCapabilities().testWithFail(train);
        C45ModelSelection selection = new C45ModelSelection(j48.getMinNumObj(), train,
                j48.getUseMDLcorrection(), j48.getDoNotMakeSplitPointActualValue());
        boolean cleanup = !j48.getSaveInstanceData();
        boolean prune = !j48.getUnpruned();
        C45PruneableClassifierTree root = args.containsKey("max_depth")
                ? new DepthLimitedTree(selection, prune, j48.getConfidenceFactor(), j48.getSubtreeRaising(),
                        cleanup, j48.getCollapseTree(), 0, Integer.parseInt(args.get("max_depth")))
                : new C45PruneableClassifierTree(selection, true, j48.getConfidenceFactor(),
                        j48.getSubtreeRaising(), cleanup, j48.getCollapseTree());
        Instances data = new Instances(train);
        data.deleteWithMissingClass();
        root.buildTree(data, j48.getSubtreeRaising() || !cleanup);
        long grown = System.nanoTime();
        if (j48.getCollapseTree()) root.collapse();
        if (prune) root.prune();
        if (cleanup) root.cleanup(new Instances(data, 0));
        selection.cleanup();
        long end = System.nanoTime();
        return new Built(root::classifyInstance, null, root, (grown - start) / 1e9, (end - grown) / 1e9);
    }

    static double accuracy(Predictor model, Instances data) throws Exception {
        int correct = 0;
        for (Instance row : data) {
            if (model.classify(row) == row.classValue()) correct++;
        }
        return (double) correct / data.numInstances();
    }

    public static void main(String[] argv) throws Exception {
        for (String arg : argv) {
            int eq = arg.indexOf('=');
            args.put(arg.substring(0, eq), arg.substring(eq + 1));
        }
        int features = Integer.parseInt(args.get("n_features"));
        int classes = Integer.parseInt(args.get("n_classes"));
        Instances train = load("train", Integer.parseInt(args.get("n_train")), features, classes);

        int warmup = Math.min(Integer.parseInt(args.get("warmup")), train.numInstances());
        if (warmup > 0) build(new Instances(train, 0, warmup));
        Built built = build(train);
        Predictor model = built.model;
        J48 plain = built.root != null ? null : built.j48;
        String text = plain == null ? built.root.toString() : plain.toString();
        int nodes = plain == null ? built.root.numNodes() : (int) plain.measureTreeSize();
        int leaves = plain == null ? built.root.numLeaves() : (int) plain.measureNumLeaves();
        long peakKiB = -1;
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith("VmHWM:")) peakKiB = Long.parseLong(line.replaceAll("\\D+", ""));
        }

        // Depth: a node at depth d is printed after d-1 "|   " prefixes, and the
        // deepest test line carries a leaf at depth (prefixes + 1).
        int depth = 0;
        for (String line : text.split("\n")) {
            if (line.contains(" <= ") || line.contains(" > ")) {
                depth = Math.max(depth, line.split("\\|", -1).length);
            }
        }
        StringBuilder out = new StringBuilder(String.format(Locale.ROOT,
                "{\"train_seconds\": %.9f, %s\"nodes\": %d, \"leaves\": %d, \"depth\": %d, "
                        + "\"n_train_loaded\": %d, \"n_features_loaded\": %d, \"peak_rss_train_bytes\": %d",
                built.build + built.prune, built.root == null ? "" : String.format(Locale.ROOT,
                        "\"build_seconds\": %.9f, \"prune_seconds\": %.9f, ", built.build, built.prune),
                nodes, leaves, depth,
                train.numInstances(), train.numAttributes() - 1, peakKiB * 1024));
        if ("1".equals(args.get("check")) && built.root != null) {
            J48 reference = options();
            reference.buildClassifier(train);
            if (!reference.toString().contains(text)) {
                throw new IllegalStateException("the phased build grew another tree than J48");
            }
            out.append(", \"check\": \"same tree as J48\"");
        }
        if (args.get("eval").equals("1")) {
            Instances test = load("test", Integer.parseInt(args.get("n_test")), features, classes);
            out.append(String.format(Locale.ROOT, ", \"train_accuracy\": %.9f, \"test_accuracy\": %.9f",
                    accuracy(model, train), accuracy(model, test)));
        }
        System.out.println(out.append("}"));
    }
}
