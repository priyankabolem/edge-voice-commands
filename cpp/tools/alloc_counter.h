#ifndef KWS_TOOLS_ALLOC_COUNTER_H_
#define KWS_TOOLS_ALLOC_COUNTER_H_

// Counts malloc/calloc/realloc calls made anywhere in the process while counting is
// enabled. The interposers live in a shared library because dyld applies interposing
// from libraries loaded at launch, not from the main executable.
extern "C" {
void AllocCounterStart();
long AllocCounterStop();  // Returns the number of allocations since Start().
}

#endif  // KWS_TOOLS_ALLOC_COUNTER_H_
