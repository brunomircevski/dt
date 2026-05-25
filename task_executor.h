#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

// Lightweight thread pool: submit work and receive std::future results.
// Intended contract for a future CUDA backend (per-feature / per-node tasks).
class TaskExecutor {
public:
  explicit TaskExecutor(std::size_t threadCount);
  ~TaskExecutor();

  TaskExecutor(const TaskExecutor &) = delete;
  TaskExecutor &operator=(const TaskExecutor &) = delete;

  template <typename F>
  auto submit(F &&f) -> std::future<std::invoke_result_t<F>>;

  void waitAll();

private:
  void workerLoop();
  void enqueue(std::function<void()> task);
  void onTaskFinished();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable taskCv_;
  bool stop_ = false;
  std::atomic<std::size_t> activeTasks_{0};
  std::mutex waitMutex_;
  std::condition_variable waitCv_;
};

template <typename F>
auto TaskExecutor::submit(F &&f) -> std::future<std::invoke_result_t<F>> {
  using Result = std::invoke_result_t<F>;
  auto task = std::make_shared<std::packaged_task<Result()>>(
      std::bind(std::forward<F>(f)));
  std::future<Result> result = task->get_future();
  activeTasks_.fetch_add(1, std::memory_order_relaxed);
  enqueue([task, this]() {
    try {
      (*task)();
    } catch (...) {
      onTaskFinished();
      throw;
    }
    onTaskFinished();
  });
  return result;
}
