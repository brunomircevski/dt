#include "core/thread_pool.h"

namespace dt {

ThreadPool::ThreadPool(unsigned workerCount) {
  workers_.reserve(workerCount);
  for (unsigned index = 0; index < workerCount; ++index) {
    workers_.emplace_back([this]() { workerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  workCv_.notify_all();
  for (std::thread &worker : workers_) {
    worker.join();
  }
}

void ThreadPool::push(std::function<void()> task, bool front) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (front) {
      queue_.push_front(std::move(task));
    } else {
      queue_.push_back(std::move(task));
    }
  }
  workCv_.notify_one();
  idleCv_.notify_one(); // a thread in waitIdle() can help
}

void ThreadPool::submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_;
  }
  push(
      [this, task = std::move(task)]() {
        try {
          task();
        } catch (...) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!firstError_) {
            firstError_ = std::current_exception();
          }
        }
        finishTask();
      },
      /*front=*/false);
}

void ThreadPool::finishTask() {
  bool idle;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    idle = --pending_ == 0;
  }
  if (idle) {
    idleCv_.notify_all();
  }
}

bool ThreadPool::runOne() {
  std::function<void()> task;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    task = std::move(queue_.front());
    queue_.pop_front();
  }
  task();
  return true;
}

void ThreadPool::waitIdle() {
  while (true) {
    if (runOne()) {
      continue;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    idleCv_.wait(lock, [this]() { return pending_ == 0 || !queue_.empty(); });
    if (pending_ == 0) {
      if (firstError_) {
        std::exception_ptr error = firstError_;
        firstError_ = nullptr;
        std::rethrow_exception(error);
      }
      return;
    }
  }
}

void ThreadPool::workerLoop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      workCv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
      if (queue_.empty()) {
        return; // stop_ and nothing left to do
      }
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    task();
  }
}

} // namespace dt
