// Linked into the CPU-only build (make cpu) instead of gpu_builder.cu.

#include "gpu_builder.h"

#include <stdexcept>

void gpuPrepare() {}

std::unique_ptr<Node> gpuGrowTree(const Dataset &, const SplitRules &, const dt::EntryCodec &,
                                  const Options &, ThreadPool &, GpuTimings &,
                                  std::vector<std::vector<float>> *) {
  throw std::runtime_error("This binary was built without CUDA (use `make` instead of "
                           "`make cpu`), or pick --parallel / --serial.");
}
