#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/mman.h>
int main() {
  const size_t bytes = size_t{1200} << 20;
  int ro = 0, dev = 0; cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, dev);
  int pageable = 0; cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
  printf("readOnlySupported %d pageableMemoryAccess %d\n", ro, pageable);
  void *d; cudaMalloc(&d, bytes);
  for (int huge = 0; huge < 2; ++huge) {
    char *h = (char*)aligned_alloc(2 << 20, bytes);
    if (huge) madvise(h, bytes, MADV_HUGEPAGE);
    for (size_t i = 0; i < bytes; i += 4096) h[i] = 1;
    const unsigned flagList[] = {cudaHostRegisterReadOnly, cudaHostRegisterDefault};
    for (unsigned flags : flagList) {
      auto t0 = std::chrono::steady_clock::now();
      cudaError_t e = cudaHostRegister(h, bytes, flags);
      auto t1 = std::chrono::steady_clock::now();
      cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
      auto t2 = std::chrono::steady_clock::now();
      if (e == cudaSuccess) cudaHostUnregister(h); else cudaGetLastError();
      auto t3 = std::chrono::steady_clock::now();
      auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      printf("huge %d flags %u: register %s %.1f ms, copy %.1f ms (%.1f GB/s), unregister %.1f ms\n", huge, flags,
             cudaGetErrorString(e), ms(t0, t1), ms(t1, t2), bytes / ms(t1, t2) / 1e6, ms(t2, t3));
    }
    free(h);
  }
}
