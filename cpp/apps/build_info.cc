// Prints the compiler, flags and ExecuTorch runtime this build uses, as JSON.

#include "kws/build_info.h"

#include <cstdio>

int main() {
  std::printf(
      "{\"compiler\": \"%s\", \"build_type\": \"%s\", \"cxx_flags\": \"%s\", "
      "\"executorch_version\": \"%s\", \"executorch_runtime\": \"%s\"}\n",
      kws::kCompiler, kws::kBuildType, kws::kCxxFlags, kws::kExecuTorchVersion,
      kws::kExecuTorchSource);
  return 0;
}
