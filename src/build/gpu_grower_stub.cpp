// Linked into the CPU-only build (make cpu) instead of gpu_grower.cu.

#include "build/grower.h"

#include <stdexcept>

namespace dt {

void gpuPrepare() {}

std::unique_ptr<Grower> makeGpuGrower(const Dataset &, const Options &, ThreadPool &, bool,
                                      double &) {
  throw std::runtime_error("This binary was built without CUDA (use `make` instead of "
                           "`make cpu`), or pick --parallel / --serial.");
}

} // namespace dt
