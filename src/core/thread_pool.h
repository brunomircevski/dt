#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dt {

// A small work-queue thread pool with two ways to use it:
//
//  * submit(task) + waitIdle(): fire-and-forget tasks (the tree builders spawn
//    one task per large subtree; the task writes its result into the tree).
//  * parallelFor(n, body): fork-join over n independent items. The calling
//    thread works too, and it only ever waits for items that are already
//    running on other threads, so nested parallelFor calls (from inside pool
//    tasks) can never deadlock the pool.
class ThreadPool {
public:
  explicit ThreadPool(unsigned workerCount);
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  unsigned workerCount() const { return static_cast<unsigned>(workers_.size()); }

  void submit(std::function<void()> task);

  // Run queued tasks on the calling thread until no task is queued or running.
  // Rethrows the first exception thrown by a submitted task.
  void waitIdle();

  template <class Body> void parallelFor(std::size_t count, Body &&body);

private:
  void push(std::function<void()> task, bool front);
  bool runOne();
  void workerLoop();
  void finishTask();

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable workCv_;
  std::condition_variable idleCv_;
  std::size_t pending_ = 0; // queued + running submit() tasks (guarded by mutex_)
  bool stop_ = false;
  std::exception_ptr firstError_;
};

template <class Body> void ThreadPool::parallelFor(std::size_t count, Body &&body) {
  if (count == 0) {
    return;
  }
  if (count == 1 || workers_.empty()) {
    for (std::size_t index = 0; index < count; ++index) {
      body(index);
    }
    return;
  }

  struct State {
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::size_t count = 0;
    std::mutex errorMutex;
    std::exception_ptr error;
  };
  auto state = std::make_shared<State>();
  state->count = count;
  auto *bodyPtr = &body; // stays alive until every claimed item is done

  // Claim items until none are left. A helper that starts late finds
  // next >= count and returns without touching `body`.
  auto work = [state, bodyPtr]() {
    for (std::size_t index; (index = state->next.fetch_add(1)) < state->count;) {
      try {
        (*bodyPtr)(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(state->errorMutex);
        if (!state->error) {
          state->error = std::current_exception();
        }
      }
      if (state->done.fetch_add(1) + 1 == state->count) {
        state->done.notify_all();
      }
    }
  };

  const std::size_t helpers = std::min<std::size_t>(count - 1, workers_.size());
  for (std::size_t helper = 0; helper < helpers; ++helper) {
    push(work, /*front=*/true);
  }
  work();
  for (std::size_t done; (done = state->done.load()) < count;) {
    state->done.wait(done);
  }
  if (state->error) {
    std::rethrow_exception(state->error);
  }
}

// pool->parallelFor, or a plain loop when there is no pool (serial backend).
template <class Body> void parallelFor(ThreadPool *pool, std::size_t count, Body &&body) {
  if (pool) {
    pool->parallelFor(count, body);
  } else {
    for (std::size_t index = 0; index < count; ++index) {
      body(index);
    }
  }
}

} // namespace dt
