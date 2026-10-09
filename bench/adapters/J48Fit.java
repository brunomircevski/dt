// Weka J48 adapter: fit one protocol on a prepared dataset, print one JSON line.
//
//   java -cp weka.jar:bounce.jar:classes J48Fit data=DIR n_train=N n_test=M \
//       n_features=F n_classes=K warmup=ROWS eval=0|1 options="-C 0.25 -M 2"
//
// The data are read from the binary files straight into Weka Instances (no
// ARFF/CSV parser, whose temporary objects would dominate peak memory).
// warmup=ROWS: one untimed build on the first ROWS training rows first, so the
// JIT has compiled the hot code (0 = none). Timed: buildClassifier. Loading and
// prediction are not timed; with eval=1 the training and test accuracy are
// measured after the timed build. The peak resident memory (VmHWM) is read
// right after the build, before evaluation.
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class J48Fit {
    static Map<String, String> args = new HashMap<>();

    static ByteBuffer read(String file) throws IOException {
        return ByteBuffer.wrap(Files.readAllBytes(Path.of(args.get("data"), file)))
                .order(ByteOrder.LITTLE_ENDIAN);
    }

    static Instances load(String part, int rows, int features, int classes) throws IOException {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int j = 0; j < features; j++) attributes.add(new Attribute("f" + j));
        ArrayList<String> labels = new ArrayList<>();
        for (int k = 0; k < classes; k++) labels.add("c" + k);
        attributes.add(new Attribute("class", labels));
        Instances data = new Instances(part, attributes, rows);
        data.setClassIndex(features);
        ByteBuffer x = read(part + ".f32");
        ByteBuffer y = read(part + ".y.i32");
        for (int i = 0; i < rows; i++) {
            double[] values = new double[features + 1];
            for (int j = 0; j < features; j++) values[j] = x.getFloat();
            values[features] = y.getInt();
            data.add(new DenseInstance(1.0, values));
        }
        return data;
    }

    static J48 build(Instances train) throws Exception {
        J48 tree = new J48();
        tree.setOptions(Utils.splitOptions(args.get("options")));
        tree.buildClassifier(train);
        return tree;
    }

    static double accuracy(J48 model, Instances data) throws Exception {
        int correct = 0;
        for (Instance row : data) {
            if (model.classifyInstance(row) == row.classValue()) correct++;
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
        long start = System.nanoTime();
        J48 model = build(train);
        double seconds = (System.nanoTime() - start) / 1e9;
        long peakKiB = -1;
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith("VmHWM:")) peakKiB = Long.parseLong(line.replaceAll("\\D+", ""));
        }

        // Depth: a node at depth d is printed after d-1 "|   " prefixes, and the
        // deepest test line carries a leaf at depth (prefixes + 1).
        int depth = 0;
        for (String line : model.toString().split("\n")) {
            if (line.contains(" <= ") || line.contains(" > ")) {
                depth = Math.max(depth, line.split("\\|", -1).length);
            }
        }
        StringBuilder out = new StringBuilder(String.format(Locale.ROOT,
                "{\"train_seconds\": %.9f, \"nodes\": %d, \"leaves\": %d, \"depth\": %d, "
                        + "\"n_train_loaded\": %d, \"n_features_loaded\": %d, \"peak_rss_train_bytes\": %d",
                seconds, (int) model.measureTreeSize(), (int) model.measureNumLeaves(), depth,
                train.numInstances(), train.numAttributes() - 1, peakKiB * 1024));
        if (args.get("eval").equals("1")) {
            Instances test = load("test", Integer.parseInt(args.get("n_test")), features, classes);
            out.append(String.format(Locale.ROOT, ", \"train_accuracy\": %.9f, \"test_accuracy\": %.9f",
                    accuracy(model, train), accuracy(model, test)));
        }
        System.out.println(out.append("}"));
    }
}
