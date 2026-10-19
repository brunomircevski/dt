// Weka J48 (C4.5, unpruned: -U -M 2) on the prepared chess split: one untimed
// warm-up build on 50,000 rows, then REPS timed builds; writes the last tree
// and its test predictions for fit.py.
//
//   java -cp weka.jar:bounce.jar:<this dir's classes> J48Export DATA REPS OUT_PREFIX
//
// Output: OUT_PREFIX.times (seconds per build), OUT_PREFIX.tree (preorder, one
// node per line: "split <attribute> <threshold> <rows A> <rows B>" with the
// "<=" child first, or "leaf <class> <rows A> <rows B>"), OUT_PREFIX.pred (test
// classes, 0 = A, 1 = B). J48 has no depth limit: fit.py cuts the tree at the
// limit afterwards (a node at the limit becomes a leaf of its majority class).
// The tree is read through reflection (J48.m_root), with the exact double
// thresholds; J48's graph() and toString() round them.
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class J48Export {
    static String data;

    static int rows(String part) throws Exception {
        String meta = Files.readString(Path.of(data, "meta.json"));
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"n_" + part + "\": *(\\d+)").matcher(meta);
        m.find();
        return Integer.parseInt(m.group(1));
    }

    static Instances load(String part) throws Exception {
        int n = rows(part);
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("x"));
        attributes.add(new Attribute("y"));
        attributes.add(new Attribute("class", new ArrayList<>(List.of("A", "B"))));
        Instances out = new Instances(part, attributes, n);
        out.setClassIndex(2);
        ByteBuffer x = ByteBuffer.wrap(Files.readAllBytes(Path.of(data, part + ".f32")))
                .order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer y = ByteBuffer.wrap(Files.readAllBytes(Path.of(data, part + ".y.i32")))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            double a = x.getFloat(), b = x.getFloat();
            out.add(new DenseInstance(1.0, new double[] {a, b, y.getInt()}));
        }
        return out;
    }

    static J48 build(Instances train) throws Exception {
        J48 tree = new J48();
        tree.setOptions(Utils.splitOptions("-U -M 2"));
        tree.buildClassifier(train);
        return tree;
    }

    static void write(ClassifierTree node, Instances header, StringBuilder out) throws Exception {
        Distribution d = node.getLocalModel().distribution();
        String counts = String.format(Locale.ROOT, " %.17g %.17g", d.perClass(0), d.perClass(1));
        if (node.isLeaf()) {
            out.append("leaf ").append(header.classAttribute().value(d.maxClass())).append(counts).append('\n');
            return;
        }
        C45Split split = (C45Split) node.getLocalModel();
        if (node.getSons().length != 2) throw new IllegalStateException("not a binary split");
        out.append(String.format(Locale.ROOT, "split %s %.17g", header.attribute(split.attIndex()).name(),
                split.splitPoint())).append(counts).append('\n');
        for (ClassifierTree son : node.getSons()) write(son, header, out);
    }

    public static void main(String[] argv) throws Exception {
        data = argv[0];
        int reps = Integer.parseInt(argv[1]);
        String out = argv[2];
        Instances train = load("train");
        build(new Instances(train, 0, 50000));
        StringBuilder times = new StringBuilder();
        J48 model = null;
        for (int r = 0; r < reps; r++) {
            long start = System.nanoTime();
            model = build(train);
            times.append(String.format(Locale.ROOT, "%.9f%n", (System.nanoTime() - start) / 1e9));
        }
        Files.writeString(Path.of(out + ".times"), times);
        Field root = J48.class.getDeclaredField("m_root");
        root.setAccessible(true);
        StringBuilder tree = new StringBuilder();
        write((ClassifierTree) root.get(model), train, tree);
        Files.writeString(Path.of(out + ".tree"), tree);
        Instances test = load("test");
        StringBuilder pred = new StringBuilder();
        for (Instance row : test) pred.append((int) model.classifyInstance(row)).append('\n');
        Files.writeString(Path.of(out + ".pred"), pred);
        Files.writeString(Path.of(out + ".version"), "Weka " + weka.core.Version.VERSION
                + ", Java " + System.getProperty("java.version") + "\n");
    }
}
