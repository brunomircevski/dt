// rusage: run a command and write its resource usage, as measured by the
// kernel, to a file.
//
//   rusage OUT command [args...]
//
// Why a separate launcher: on Linux a process's peak RSS (ru_maxrss) starts at
// the peak RSS of the process it was forked from, and exec keeps it. Started
// straight from the Python harness, every tool would report at least the
// harness's own peak (hundreds of MiB after it has checked the data). This
// launcher is tiny, so the command's peak RSS is its own, to within ~1 MiB.
//
// OUT gets one line: "maxrss_kib user_seconds system_seconds". The exit status
// is the command's (128 + signal number if a signal ended it).
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: rusage OUT command [args...]\n");
    return 2;
  }
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 2;
  }
  if (pid == 0) {
    execvp(argv[2], argv + 2);
    perror(argv[2]);
    _exit(127);
  }
  int status;
  struct rusage usage;
  if (wait4(pid, &status, 0, &usage) < 0) {
    perror("wait4");
    return 2;
  }
  FILE *out = fopen(argv[1], "w");
  if (!out) {
    perror(argv[1]);
    return 2;
  }
  fprintf(out, "%ld %.6f %.6f\n", usage.ru_maxrss,
          usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6,
          usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6);
  fclose(out);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
