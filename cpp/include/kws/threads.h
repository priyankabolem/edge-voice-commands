#ifndef KWS_THREADS_H_
#define KWS_THREADS_H_

#include <cstddef>
#include <cstdint>

namespace kws {

// Sets the size of the ExecuTorch threadpool that the XNNPACK delegate runs on.
// Must be called before any KwsModel is created: resetting the pool destroys the
// previous one, which an already-loaded model may still reference.
bool SetRuntimeThreads(uint32_t num_threads);

size_t RuntimeThreads();

}  // namespace kws

#endif  // KWS_THREADS_H_
