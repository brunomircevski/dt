#include "task_executor.h"

// Start `threadCount` worker threads (minimum one); each runs workerLoop().
TaskExecutor::TaskExecutor(std::size_t threadCount) {
  if (threadCount == 0) {
    threadCount = 1;
  }

  workers_.reserve(threadCount);
  for (std::size_t index = 0; index < threadCount; ++index) {
    workers_.emplace_back([this]() { workerLoop(); });
  }
}

// Signal shutdown, wake all workers so they can exit, then join every thread.
TaskExecutor::~TaskExecutor() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  taskCv_.notify_all();
  for (std::thread &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

// Block until every submitted task has finished (activeTasks_ reaches zero).
void TaskExecutor::waitAll() {
  std::unique_lock<std::mutex> lock(waitMutex_);
  waitCv_.wait(lock, [this]() { return activeTasks_.load() == 0; });
}

// Called when a task completes; wake waitAll() only when the last task ends.
void TaskExecutor::onTaskFinished() {
  if (activeTasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lock(waitMutex_);
    waitCv_.notify_all();
  }
}

// Pull tasks from the queue until stop_ is set and the queue is empty.
void TaskExecutor::workerLoop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // Sleep until there is work to do or the pool is shutting down.
      taskCv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

// Add a task to the shared queue and wake one idle worker.
void TaskExecutor::enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
  }
  taskCv_.notify_one();
}
