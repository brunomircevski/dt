#include "core/options.h"

#include <algorithm>
#include <sched.h>
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
  case Backend::Simple: return "simple";
  case Backend::Serial: return "serial";
  case Backend::Multicore: return "multicore";
  case Backend::Gpu: return "gpu";
  case Backend::Hybrid: return "hybrid";
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
  // The CPUs this process may run on (taskset, numactl --cpunodebind, a
  // SLURM allocation), not every CPU of the machine.
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof allowed, &allowed) == 0 && CPU_COUNT(&allowed) > 0) {
    return static_cast<unsigned>(CPU_COUNT(&allowed));
  }
  return std::max(1u, std::thread::hardware_concurrency());
}

std::size_t handoffRows(const Options &options, std::size_t trainingRows) {
  const GpuOptions &gpu = options.gpu;
  if (options.backend == Backend::Gpu) {
    return 1;
  }
  if (gpu.handoffRows > 0) {
    return gpu.handoffRows;
  }
  return options.algorithm == Algorithm::Cart
             ? std::clamp(trainingRows / GpuOptions::kAutoMinRowsShareCart,
                          GpuOptions::kAutoMinRows, GpuOptions::kAutoMaxMinRowsCart)
             : std::clamp(trainingRows / GpuOptions::kAutoMinRowsShare, GpuOptions::kAutoMinRows,
                          GpuOptions::kAutoMaxMinRows);
}

void printUsage(const char *program) {
  std::cout
      << "Usage: " << program << " [options] dataset.csv\n"
      << "\n"
      << "Backend (default --multicore), the paper's five implementations:\n"
      << "  --simple                 Serial Simple: one thread, the textbook algorithm\n"
      << "  --serial                 Serial Optimized: one thread, presorted columns\n"
      << "  --multicore              Multicore CPU: every CPU thread (see --threads)\n"
      << "  --gpu                    GPU-Based: every node on the GPU (the CPU threads\n"
      << "                           feed the GPU and prune)\n"
      << "  --hybrid                 Hybrid: large nodes on the GPU, small subtrees on\n"
      << "                           the CPU threads\n"
      << "\n"
      << "Algorithm (default --cart):\n"
      << "  --cart                   CART (Gini index)\n"
      << "  --c45                    C4.5 (gain ratio)\n"
      << "\n"
      << "Tree:\n"
      << "  -d, --max-depth N        depth limit (-1 = unlimited, the default)\n"
      << "  --min-leaf N             every child keeps at least N rows (default: 1 for\n"
      << "                           CART, 2 for C4.5)\n"
      << "  --prune X                prune the grown tree (default: no pruning).\n"
      << "                           CART: cost-complexity pruning, X = alpha (>= 0,\n"
      << "                           e.g. 0.00001). C4.5: error-based pruning with\n"
      << "                           subtree raising, X = the confidence factor (in\n"
      << "                           (0, 1); the c4.5 program's default is 0.25).\n"
      << "                           Unpruned C4.5 still collapses splits that do not\n"
      << "                           reduce the training errors.\n"
      << "\n"
      << "Data:\n"
      << "  --rows N                 use only the first N rows of dataset.csv\n"
      << "  --test FILE              test set: a CSV with the same header\n"
      << "  --test-last N            test set: the last N rows of the training file\n"
      << "  --holdout F              test set: a random fraction F of the rows\n"
      << "  --seed N                 random seed for the holdout split (default 1)\n"
      << "\n"
      << "Run:\n"
      << "  --threads N              CPU threads for loading, evaluation and (multicore,\n"
      << "                           gpu, hybrid) training (0 = every CPU it may run on)\n"
      << "  --handoff-rows N         --hybrid: smaller nodes go to the CPU threads\n"
      << "                           (default: training rows / " << GpuOptions::kAutoMinRowsShare
      << ", between " << GpuOptions::kAutoMinRows << "\n"
      << "                           and " << GpuOptions::kAutoMaxMinRows << "; CART: rows / "
      << GpuOptions::kAutoMinRowsShareCart << ", between " << GpuOptions::kAutoMinRows
      << " and " << GpuOptions::kAutoMaxMinRowsCart << ")\n"
      << "\n"
      << "Output:\n"
      << "  --dump FILE              write the tree as text\n"
      << "  --profile FILE           write where the training time goes as JSON: phases,\n"
      << "                           thread time per kind of work, GPU kernels and copies\n"
      << "\n"
      << "Benchmarking (energy runs):\n"
      << "  --repeat N               train N trees in a row on the loaded data, each one\n"
      << "                           as a fresh process's only tree, and print each one's\n"
      << "                           time and Unix start/end\n"
      << "  --repeat-warmup S        with --repeat: keep the threads busy for S seconds\n"
      << "                           before each tree (as the CSV load does)\n";
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
  for (const auto &group :
       {givenOf(given, {"--simple", "--serial", "--multicore", "--gpu", "--hybrid"}),
        givenOf(given, {"--cart", "--c45"})}) {
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
  if (options.backend != Backend::Hybrid) {
    refuse({"--handoff-rows"}, "is for --hybrid only");
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
    } else if (arg == "--simple") {
      options.backend = Backend::Simple;
    } else if (arg == "--serial") {
      options.backend = Backend::Serial;
    } else if (arg == "--multicore") {
      options.backend = Backend::Multicore;
    } else if (arg == "--gpu") {
      options.backend = Backend::Gpu;
    } else if (arg == "--hybrid") {
      options.backend = Backend::Hybrid;
    } else if (arg == "--cart") {
      options.algorithm = Algorithm::Cart;
    } else if (arg == "--c45") {
      options.algorithm = Algorithm::C45;
    } else if (arg == "-d" || arg == "--max-depth") {
      options.maxDepth = static_cast<int>(parseInteger(arg, args.value(arg), -1));
    } else if (arg == "--min-leaf") {
      options.minLeaf = static_cast<std::size_t>(parseInteger(arg, args.value(arg), 1));
    } else if (arg == "--prune") {
      // One value for both algorithms; the chosen one reads its own.
      const double value = parseReal(arg, args.value(arg));
      options.cart.prune = options.c45.prune = true;
      options.cart.alpha = options.c45.confidence = value;
    } else if (arg == "--repeat") {
      options.repeat = parseCount(arg, args.value(arg));
    } else if (arg == "--repeat-warmup") {
      options.repeatWarmup = parseReal(arg, args.value(arg));
    } else if (arg == "--holdout") {
      options.holdout = parseReal(arg, args.value(arg));
    } else if (arg == "--test") {
      options.testPath = args.value(arg);
    } else if (arg == "--rows") {
      options.rows = static_cast<std::size_t>(parseInteger(arg, args.value(arg), 1));
    } else if (arg == "--test-last") {
      options.testLast = parseCount(arg, args.value(arg));
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint64_t>(parseCount(arg, args.value(arg)));
    } else if (arg == "--threads") {
      options.parallel.threads = static_cast<int>(parseCount(arg, args.value(arg)));
    } else if (arg == "--dump") {
      options.dumpPath = args.value(arg);
    } else if (arg == "--profile") {
      options.profilePath = args.value(arg);
    } else if (arg == "--handoff-rows") {
      options.gpu.handoffRows = parseCount(arg, args.value(arg));
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    } else if (!options.datasetPath.empty()) {
      throw std::runtime_error("Two datasets given: " + options.datasetPath + " and " + arg);
    } else {
      options.datasetPath = arg;
    }
  }
  if (options.datasetPath.empty()) {
    throw std::runtime_error("No dataset given (" + std::string(argv[0]) +
                             " [options] dataset.csv; --help lists the options)");
  }
  checkFlagCombination(options, given);
  return true;
}

void validateOptions(const Options &options) {
  if (options.holdout < 0.0 || options.holdout >= 1.0) {
    throw std::runtime_error("--holdout must be in [0, 1)");
  }
  if ((!options.testPath.empty()) + (options.testLast > 0) + (options.holdout > 0.0) > 1) {
    throw std::runtime_error("--test, --test-last and --holdout exclude each other");
  }
  if (options.repeatWarmup < 0.0) {
    throw std::runtime_error("--repeat-warmup must be >= 0");
  }
  if (options.repeat > 1 && !options.profilePath.empty()) {
    throw std::runtime_error("--profile traces one tree: use it without --repeat");
  }
  if (options.algorithm == Algorithm::Cart) {
    if (options.cart.prune && options.cart.alpha < 0.0) {
      throw std::runtime_error("CART: --prune (alpha) must be >= 0");
    }
  } else {
    if (options.c45.prune && (options.c45.confidence <= 0.0 || options.c45.confidence >= 1.0)) {
      throw std::runtime_error("C4.5: --prune (confidence factor) must be in (0, 1)");
    }
  }
}

} // namespace dt
