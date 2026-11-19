// STREAM-style memory bandwidth: copy and triad over 3 x 4 GiB arrays, all threads.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N (512L * 1024 * 1024)
int main(void) {
  double *a = malloc(N * 8), *b = malloc(N * 8), *c = malloc(N * 8);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) a[i] = 1, b[i] = 2, c[i] = 0;  // first touch: pages on each thread's socket
  double best_copy = 0, best_triad = 0;
  for (int r = 0; r < 10; r++) {
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) c[i] = a[i];
    t = omp_get_wtime() - t;
    if (2.0 * N * 8 / t > best_copy) best_copy = 2.0 * N * 8 / t;
    t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) a[i] = b[i] + 3.0 * c[i];
    t = omp_get_wtime() - t;
    if (3.0 * N * 8 / t > best_triad) best_triad = 3.0 * N * 8 / t;
  }
  printf("threads %d  copy %.1f GB/s  triad %.1f GB/s  (check %.1f)\n", omp_get_max_threads(),
         best_copy / 1e9, best_triad / 1e9, a[N / 2]);
  return 0;
}
