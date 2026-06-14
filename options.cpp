#include "options.h"

#include "tree_base.h"
#include "tree_cuda.h"
#include "tree_parallel.h"
#include "tree_serial.h"

#include <stdexcept>
#include <string>

namespace {

struct BackendInfo {
  const char *name;
  std::unique_ptr<TreeBase> (*create)();
};

const BackendInfo kBackends[] = {
    {"serial",
     +[]() -> std::unique_ptr<TreeBase> { return std::make_unique<TreeSerial>(); }},
    {"parallel",
     +[]() -> std::unique_ptr<TreeBase> { return std::make_unique<TreeParallel>(); }},
    {"cuda",
     +[]() -> std::unique_ptr<TreeBase> { return std::make_unique<TreeCuda>(); }},
};

const BackendInfo &lookupBackend(Backend backend) {
  const auto index = static_cast<std::size_t>(backend);
  if (index >= sizeof(kBackends) / sizeof(kBackends[0])) {
    throw std::runtime_error("Unknown backend");
  }
  return kBackends[index];
}

} // namespace

const char *backendName(Backend backend) { return lookupBackend(backend).name; }

std::unique_ptr<TreeBase> createTree(Backend backend) {
  return lookupBackend(backend).create();
}

void applyCommandLine(int argc, char *argv[], Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cuda") {
      options.backend = Backend::Cuda;
    } else if (arg == "--parallel") {
      options.backend = Backend::Parallel;
    } else if (arg == "--serial") {
      options.backend = Backend::Serial;
    } else if (arg.rfind("-d", 0) == 0) {
      const std::string value = arg.substr(2);
      if (value.empty()) {
        if (i + 1 >= argc) {
          throw std::runtime_error("Option -d requires a value");
        }
        options.maxDepth = std::stoi(argv[++i]);
      } else {
        options.maxDepth = std::stoi(value);
      }
    } else if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("Unknown option: " + arg);
    } else if (arg.rfind("-", 0) == 0) {
      throw std::runtime_error("Unknown option: " + arg);
    } else {
      options.datasetPath = arg;
    }
  }
}
