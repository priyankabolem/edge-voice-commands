// Records per-operator timings of the XNNPACK delegate with ETDump.
// Runs the model `iters` times with an ETDumpGen event tracer attached and writes the
// trace for kws/profile_ops.py to summarize (it reads the file with executorch.devtools).
//
// Usage: profile_ops <model.pte> <out.etdp> [--threads N] [--iters N]

#include <executorch/devtools/etdump/etdump_flatcc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>

#include "kws/kws_model.h"
#include "kws/threads.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.pte> <out.etdp> [--threads N] [--iters N]\n", argv[0]);
    return 2;
  }
  int threads = 1;
  int iters = 100;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--threads") == 0) {
      threads = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      iters = std::atoi(argv[i + 1]);
    }
  }
  try {
    if (!kws::SetRuntimeThreads(static_cast<uint32_t>(threads))) {
      throw std::runtime_error("failed to set threads");
    }
    executorch::etdump::ETDumpGen etdump;
    kws::KwsModel model(argv[1], &etdump);
    for (int i = 0; i < model.input_size(); ++i) model.input()[i] = 0.01f * (i % 97) - 0.5f;
    for (int i = 0; i < iters; ++i) {
      if (model.Run() == nullptr) throw std::runtime_error("inference failed");
    }

    const executorch::etdump::ETDumpResult result = etdump.get_etdump_data();
    if (result.buf == nullptr || result.size == 0) throw std::runtime_error("empty ETDump");
    std::ofstream out(argv[2], std::ios::binary);
    out.write(static_cast<const char*>(result.buf), static_cast<std::streamsize>(result.size));
    std::free(result.buf);
    return out ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
