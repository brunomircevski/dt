#include "options.h"

#include <iostream>
#include <stdexcept>
#include <string>

const char *backendName(Backend backend) {
  switch (backend) {
  case Backend::Serial: return "serial";
  case Backend::Parallel: return "parallel";
  case Backend::Cuda: return "cuda";
  }
  return "unknown";
}

const char *algorithmName(Algorithm algorithm) {
  switch (algorithm) {
  case Algorithm::Cart: return "CART";
  case Algorithm::C45: return "C4.5";
  }
  return "unknown";
}

const char *criterionName(Criterion criterion) {
  switch (criterion) {
  case Criterion::Gini: return "gini";
  case Criterion::Entropy: return "entropy";
  }
  return "unknown";
}

void printUsage(const char *program) {
  std::cout
      << "Usage: " << program << " [options] [dataset.csv]\n"
      << "\n"
      << "Backend:   --serial | --parallel | --cuda\n"
      << "Algorithm: --cart | --c45\n"
      << "\n"
      << "Common:\n"
      << "  -d, --max-depth N        depth limit (-1 = unlimited)\n"
      << "  --holdout F              hold out fraction F of rows as test set\n"
      << "  -m N                     duplicate rows N times in memory (demo)\n"
      << "  --threads N              CPU threads (0 = all cores)\n"
      << "  --print                  print the tree\n"
      << "  --dump FILE              write the tree as text\n"
      << "  --svg FILE               render the tree to SVG\n"
      << "  --load-tree FILE         prune a tree read from FILE instead of growing\n"
      << "\n"
      << "CART:\n"
      << "  --criterion gini|entropy impurity (default gini)\n"
      << "  --min-split N            min rows to split a node (default 2)\n"
      << "  --min-leaf N             min rows per child (default 1)\n"
      << "  --min-decrease X         min weighted impurity decrease\n"
      << "  --alpha X                cost-complexity prune with alpha X\n"
      << "  --cv K                   cost-complexity prune, alpha by K-fold CV\n"
      << "\n"
      << "C4.5:\n"
      << "  --min-objs N             C4.5 -m (default 2)\n"
      << "  --cf X                   pruning confidence factor (default 0.25)\n"
      << "  --no-prune               build the unpruned tree\n"
      << "  --no-raising             disable subtree raising\n"
      << "\n"
      << "Cuda:\n"
      << "  --gpu-min-rows N         smaller nodes go to the CPU pool (default 512)\n";
}

namespace {

// Small cursor over argv that turns "missing value" into a clear error.
class ArgReader {
public:
  ArgReader(int argc, char *argv[]) : argc_(argc), argv_(argv) {}

  bool done() const { return index_ >= argc_; }
  std::string next() { return argv_[index_++]; }

  std::string value(const std::string &flag) {
    if (done()) {
      throw std::runtime_error("Option " + flag + " requires a value");
    }
    return next();
  }

private:
  int argc_;
  char **argv_;
  int index_ = 1;
};

std::size_t parseCount(const std::string &flag, const std::string &text) {
  try {
    std::size_t used = 0;
    const long long value = std::stoll(text, &used);
    if (used != text.size() || value < 0) {
      throw std::invalid_argument(text);
    }
    return static_cast<std::size_t>(value);
  } catch (const std::logic_error &) {
    throw std::runtime_error("Option " + flag + " expects a non-negative integer, got '" +
                             text + "'");
  }
}

double parseReal(const std::string &flag, const std::string &text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size()) {
      throw std::invalid_argument(text);
    }
    return value;
  } catch (const std::logic_error &) {
    throw std::runtime_error("Option " + flag + " expects a number, got '" + text + "'");
  }
}

} // namespace

bool applyCommandLine(int argc, char *argv[], Options &options) {
  ArgReader args(argc, argv);
  while (!args.done()) {
    const std::string arg = args.next();

    if (arg == "-h" || arg == "--help") {
      return false;
    } else if (arg == "--serial") {
      options.backend = Backend::Serial;
    } else if (arg == "--parallel") {
      options.backend = Backend::Parallel;
    } else if (arg == "--cuda") {
      options.backend = Backend::Cuda;
    } else if (arg == "--cart") {
      options.algorithm = Algorithm::Cart;
    } else if (arg == "--c45") {
      options.algorithm = Algorithm::C45;
    } else if (arg == "-d" || arg == "--max-depth") {
      options.maxDepth = std::stoi(args.value(arg));
    } else if (arg.rfind("-d", 0) == 0 && arg.size() > 2 && arg[2] != '-') {
      options.maxDepth = std::stoi(arg.substr(2));
    } else if (arg == "-m") {
      options.datasetMultiplier = parseCount(arg, args.value(arg));
    } else if (arg == "--holdout") {
      options.holdoutFraction = parseReal(arg, args.value(arg));
    } else if (arg == "--threads") {
      options.threads = static_cast<int>(parseCount(arg, args.value(arg)));
    } else if (arg == "--print") {
      options.printTree = true;
    } else if (arg == "--dump") {
      options.dumpPath = args.value(arg);
    } else if (arg == "--load-tree") {
      options.loadTreePath = args.value(arg);
    } else if (arg == "--svg") {
      options.svgPath = args.value(arg);
    } else if (arg == "--criterion") {
      const std::string value = args.value(arg);
      if (value == "gini") {
        options.criterion = Criterion::Gini;
      } else if (value == "entropy") {
        options.criterion = Criterion::Entropy;
      } else {
        throw std::runtime_error("Unknown criterion: " + value);
      }
    } else if (arg == "--min-split") {
      options.minSamplesSplit = parseCount(arg, args.value(arg));
    } else if (arg == "--min-leaf") {
      options.minSamplesLeaf = parseCount(arg, args.value(arg));
    } else if (arg == "--min-decrease") {
      options.minImpurityDecrease = parseReal(arg, args.value(arg));
    } else if (arg == "--alpha") {
      options.cartPrune = true;
      options.ccpAlpha = parseReal(arg, args.value(arg));
      options.ccpFolds = 0;
    } else if (arg == "--cv") {
      options.cartPrune = true;
      options.ccpFolds = static_cast<int>(parseCount(arg, args.value(arg)));
    } else if (arg == "--min-objs") {
      options.c45MinObjects = parseCount(arg, args.value(arg));
    } else if (arg == "--cf") {
      options.c45Prune = true;
      options.c45ConfidenceFactor = parseReal(arg, args.value(arg));
    } else if (arg == "--prune") {
      options.c45Prune = true;
      options.cartPrune = true;
    } else if (arg == "--no-prune") {
      options.c45Prune = false;
      options.cartPrune = false;
    } else if (arg == "--no-raising") {
      options.c45SubtreeRaising = false;
    } else if (arg == "--gpu-min-rows") {
      options.gpuMinRows = parseCount(arg, args.value(arg));
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    } else {
      options.datasetPath = arg;
    }
  }
  return true;
}

void validateOptions(const Options &options) {
  if (options.datasetMultiplier < 1) {
    throw std::runtime_error("-m must be at least 1");
  }
  if (options.holdoutFraction < 0.0 || options.holdoutFraction >= 1.0) {
    throw std::runtime_error("--holdout must be in [0, 1)");
  }
  if (options.algorithm == Algorithm::Cart) {
    if (options.minSamplesSplit < 2) {
      throw std::runtime_error("CART: --min-split must be at least 2");
    }
    if (options.minSamplesLeaf < 1) {
      throw std::runtime_error("CART: --min-leaf must be at least 1");
    }
    if (options.ccpAlpha < 0.0) {
      throw std::runtime_error("CART: --alpha must be >= 0");
    }
    if (options.cartPrune && options.ccpFolds == 1) {
      throw std::runtime_error("CART: --cv needs at least 2 folds");
    }
  } else {
    if (options.c45MinObjects < 1) {
      throw std::runtime_error("C4.5: --min-objs must be at least 1");
    }
    if (options.c45ConfidenceFactor <= 0.0 || options.c45ConfidenceFactor >= 1.0) {
      throw std::runtime_error("C4.5: --cf must be in (0, 1)");
    }
  }
}
