#pragma once

// Named ranges for Nsight Systems timelines (nsys profile --trace=cuda,nvtx).
// Compiled in only with `make NVTX=1` (-DDT_NVTX); otherwise they cost nothing.
#ifdef DT_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace dt {

// A null name marks no range.
class NvtxRange {
public:
  explicit NvtxRange([[maybe_unused]] const char *name) {
#ifdef DT_NVTX
    if (name) {
      nvtxRangePushA(name);
      pushed_ = true;
    }
#endif
  }
  ~NvtxRange() {
#ifdef DT_NVTX
    if (pushed_) {
      nvtxRangePop();
    }
#endif
  }
  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;

#ifdef DT_NVTX
private:
  bool pushed_ = false;
#endif
};

} // namespace dt
