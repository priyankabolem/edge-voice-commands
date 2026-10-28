#include "alloc_counter.h"

#include <atomic>
#include <cstdlib>

namespace {

std::atomic<bool> g_counting{false};
std::atomic<long> g_count{0};

void Count() {
  if (g_counting.load(std::memory_order_relaxed)) g_count.fetch_add(1, std::memory_order_relaxed);
}

// Inside this library, malloc() refers to the real implementation.
void* CountedMalloc(size_t size) {
  Count();
  return malloc(size);
}
void* CountedCalloc(size_t count, size_t size) {
  Count();
  return calloc(count, size);
}
void* CountedRealloc(void* ptr, size_t size) {
  Count();
  return realloc(ptr, size);
}

struct Interpose {
  const void* replacement;
  const void* original;
};
__attribute__((used, section("__DATA,__interpose"))) const Interpose kInterposers[] = {
    {reinterpret_cast<const void*>(&CountedMalloc), reinterpret_cast<const void*>(&malloc)},
    {reinterpret_cast<const void*>(&CountedCalloc), reinterpret_cast<const void*>(&calloc)},
    {reinterpret_cast<const void*>(&CountedRealloc), reinterpret_cast<const void*>(&realloc)},
};

}  // namespace

extern "C" void AllocCounterStart() {
  g_count = 0;
  g_counting = true;
}

extern "C" long AllocCounterStop() {
  g_counting = false;
  return g_count.load();
}
