#include "core/options.h"

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace dt {

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

unsigned threadCount(const Options &options) {
  if (options.parallel.threads > 0) {
    return static_cast<unsigned>(options.parallel.threads);
  }
  return std::max(1u, std::thread::hardware_concurrency());
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
      << "  --min-leaf N             every child keeps at least N rows (default: 1 for\n"
      << "                           CART, 2 for C4.5)\n"
      << "  --no-prune               keep the unpruned tree (pruning is on by default)\n"
      << "  --holdout F              hold out fraction F of rows as test set\n"
      << "  --seed N                 random seed for the holdout split and CART's CV\n"
      << "                           folds (default 1)\n"
      << "  -m N                     duplicate rows N times in memory (stress tests)\n"
      << "  --threads N              CPU threads (0 = all cores)\n"
      << "  --print                  print the tree\n"
      << "  --dump FILE              write the tree as text\n"
      << "\n"
      << "CART (default: cost-complexity pruning with a fixed alpha, one tree):\n"
      << "  --alpha X                cost-complexity parameter (default 0.0001)\n"
      << "  --cv K                   choose alpha by K-fold cross-validation instead\n"
      << "                           (K + 1 trees)\n"
      << "\n"
      << "C4.5 (default: error-based pruning with CF 0.25 and subtree raising):\n"
      << "  --cf X                   pruning confidence factor\n"
      << "\n"
      << "Parallel and Cuda (CPU side):\n"
      << "  --task-rows N            nodes with >= N rows grow a child as a pool task\n"
      << "                           (default 4096)\n"
      << "  --feature-parallel-rows N  nodes with >= N rows scan features in parallel\n"
      << "                           (default 4096)\n"
      << "\n"
      << "Cuda:\n"
      << "  --gpu-min-rows N         smaller nodes go to the CPU pool (default 512)\n"
      << "  --gpu-sweep MODE         auto | one-pass | two-pass (default auto: one-pass\n"
      << "                           on GPUs with fast double precision)\n"
      << "  --gpu-profile            print per-level / per-kernel GPU times (slower)\n";
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

long long parseInteger(const std::string &flag, const std::string &text, long long minimum) {
  try {
    std::size_t used = 0;
    const long long value = std::stoll(text, &used);
    if (used != text.size() || value < minimum) {
      throw std::invalid_argument(text);
    }
    return value;
  } catch (const std::logic_error &) {
    throw std::runtime_error("Option " + flag + " expects an integer >= " +
                             std::to_string(minimum) + ", got '" + text + "'");
  }
}

std::size_t parseCount(const std::string &flag, const std::string &text) {
  return static_cast<std::size_t>(parseInteger(flag, text, 0));
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

// The flags of `group` that were given, in the group's order.
std::vector<std::string> givenOf(const std::set<std::string> &given,
                                 std::initializer_list<const char *> group) {
  std::vector<std::string> found;
  for (const char *flag : group) {
    if (given.count(flag) > 0) {
      found.push_back(flag);
    }
  }
  return found;
}

// A flag the chosen algorithm or backend would ignore, or two flags that
// contradict each other, is an error rather than silently dropped. Checked
// once every flag is read, so their order does not matter.
void checkFlagCombination(const Options &options, const std::set<std::string> &given) {
  for (const auto &group : {givenOf(given, {"--serial", "--parallel", "--cuda"}),
                            givenOf(given, {"--cart", "--c45"}),
                            givenOf(given, {"--no-prune", "--cv", "--alpha"}),
                            givenOf(given, {"--no-prune", "--cf"})}) {
    if (group.size() > 1) {
      throw std::runtime_error("Options " + group[0] + " and " + group[1] +
                               " contradict each other; give only one");
    }
  }
  // `flags` must not be given in this configuration.
  auto refuse = [&](std::initializer_list<const char *> flags, const std::string &reason) {
    const std::vector<std::string> found = givenOf(given, flags);
    if (!found.empty()) {
      throw std::runtime_error("Option " + found[0] + " " + reason);
    }
  };
  if (options.algorithm == Algorithm::C45) {
    refuse({"--cv", "--alpha"},
           "is for CART (it chooses CART's alpha); C4.5 prunes with --cf");
  } else {
    refuse({"--cf"}, "is for C4.5; CART prunes with --alpha or --cv");
  }
  if (options.backend == Backend::Serial) {
    refuse({"--task-rows", "--feature-parallel-rows"}, "is for --parallel and --cuda, not --serial");
  }
  if (options.backend != Backend::Cuda) {
    refuse({"--gpu-min-rows", "--gpu-sweep", "--gpu-profile"}, "is for --cuda only");
  }
}

} // namespace

bool applyCommandLine(int argc, char *argv[], Options &options) {
  ArgReader args(argc, argv);
  std::set<std::string> given; // flags only; their values are read separately
  while (!args.done()) {
    const std::string arg = args.next();
    if (!arg.empty() && arg[0] == '-') {
      given.insert(arg);
    }

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
      options.maxDepth = static_cast<int>(parseInteger(arg, args.value(arg), -1));
    } else if (arg.rfind("-d", 0) == 0 && arg.size() > 2 && arg[2] != '-') {
      options.maxDepth = static_cast<int>(parseInteger("-d", arg.substr(2), -1));
    } else if (arg == "--min-leaf") {
      options.minLeaf = static_cast<std::size_t>(parseInteger(arg, args.value(arg), 1));
    } else if (arg == "--no-prune") {
      options.cart.pruning = CartPruning::None;
      options.c45.prune = false;
    } else if (arg == "-m") {
      options.multiplier = parseCount(arg, args.value(arg));
    } else if (arg == "--holdout") {
      options.holdout = parseReal(arg, args.value(arg));
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint64_t>(parseCount(arg, args.value(arg)));
    } else if (arg == "--threads") {
      options.parallel.threads = static_cast<int>(parseCount(arg, args.value(arg)));
    } else if (arg == "--print") {
      options.printTree = true;
    } else if (arg == "--dump") {
      options.dumpPath = args.value(arg);
    } else if (arg == "--alpha") {
      options.cart.pruning = CartPruning::Alpha;
      options.cart.alpha = parseReal(arg, args.value(arg));
    } else if (arg == "--cv") {
      options.cart.pruning = CartPruning::CrossValidation;
      options.cart.folds = static_cast<int>(parseCount(arg, args.value(arg)));
    } else if (arg == "--task-rows") {
      options.parallel.nodeTaskRows = parseCount(arg, args.value(arg));
    } else if (arg == "--feature-parallel-rows") {
      options.parallel.featureParallelRows = parseCount(arg, args.value(arg));
    } else if (arg == "--gpu-profile") {
      options.gpu.profile = true;
    } else if (arg == "--cf") {
      options.c45.prune = true;
      options.c45.confidence = parseReal(arg, args.value(arg));
    } else if (arg == "--gpu-min-rows") {
      options.gpu.minRows = parseCount(arg, args.value(arg));
    } else if (arg == "--gpu-sweep") {
      const std::string value = args.value(arg);
      if (value == "auto") {
        options.gpu.sweep = GpuSweep::Auto;
      } else if (value == "one-pass") {
        options.gpu.sweep = GpuSweep::OnePass;
      } else if (value == "two-pass") {
        options.gpu.sweep = GpuSweep::TwoPass;
      } else {
        throw std::runtime_error("Unknown --gpu-sweep mode: " + value);
      }
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    } else {
      options.datasetPath = arg;
    }
  }
  checkFlagCombination(options, given);
  return true;
}

void validateOptions(const Options &options) {
  if (options.multiplier < 1) {
    throw std::runtime_error("-m must be at least 1");
  }
  if (options.holdout < 0.0 || options.holdout >= 1.0) {
    throw std::runtime_error("--holdout must be in [0, 1)");
  }
  if (options.algorithm == Algorithm::Cart) {
    const CartOptions &cart = options.cart;
    if (cart.pruning == CartPruning::Alpha && cart.alpha < 0.0) {
      throw std::runtime_error("CART: --alpha must be >= 0");
    }
    if (cart.pruning == CartPruning::CrossValidation && cart.folds < 2) {
      throw std::runtime_error("CART: --cv needs at least 2 folds");
    }
  } else {
    if (options.c45.confidence <= 0.0 || options.c45.confidence >= 1.0) {
      throw std::runtime_error("C4.5: --cf must be in (0, 1)");
    }
  }
}

} // namespace dt
