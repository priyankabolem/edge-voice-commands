#include "kws/threads.h"

// The prebuilt ExecuTorch SDK in the pip wheel ships the threadpool library but not
// its header (which needs pthreadpool.h). These declarations match
// extension/threadpool/threadpool.h at ExecuTorch v1.5.1; only the functions used
// here are declared, and ThreadPool is only handled through a pointer.
namespace executorch::extension::threadpool {
class ThreadPool {
 public:
  size_t get_thread_count() const;
  bool _unsafe_reset_threadpool(uint32_t num_threads);
};
ThreadPool* get_threadpool();
}  // namespace executorch::extension::threadpool

namespace kws {

bool SetRuntimeThreads(uint32_t num_threads) {
  return executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(num_threads);
}

size_t RuntimeThreads() {
  return executorch::extension::threadpool::get_threadpool()->get_thread_count();
}

}  // namespace kws
