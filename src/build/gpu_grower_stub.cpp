// Linked into the CPU-only build (make cpu) instead of gpu_grower.cu.

#include "build/grower.h"

#include <stdexcept>

namespace dt {

void gpuPrepare(unsigned) {}

std::unique_ptr<Grower> makeGpuGrower(const Dataset &, const Options &, ThreadPool &, double &) {
  throw std::runtime_error("This binary was built without CUDA (use `make` instead of "
                           "`make cpu`), or pick --multicore, --serial or --simple.");
}

} // namespace dt
