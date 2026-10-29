#!/usr/bin/env bash
# Build the ExecuTorch C++ runtime from source (Release, XNNPACK backend) and install it
# into third_party/executorch-install. Follows docs/source/using-executorch-building-from-source.md
# at the pinned tag, with only the options this project uses:
#   XNNPACK backend, data loader extension, devtools + event tracer (ETDump profiling).
# The event tracer is enabled to match the prebuilt runtime in the executorch pip wheel.
#
# Usage: scripts/build_executorch.sh   (run from the repository root)
set -euo pipefail

TAG="v1.5.1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/executorch"
BUILD="$ROOT/third_party/executorch-build"
PREFIX="$ROOT/third_party/executorch-install"
VENV_BIN="$ROOT/.venv/bin"
export PATH="$VENV_BIN:$PATH"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/pytorch/executorch.git "$SRC"
fi
git -C "$SRC" submodule update --init --recursive --depth 1 \
  third-party/flatbuffers \
  third-party/flatcc \
  third-party/gflags \
  third-party/json \
  backends/xnnpack/third-party/XNNPACK \
  backends/xnnpack/third-party/FP16 \
  backends/xnnpack/third-party/FXdiv \
  backends/xnnpack/third-party/cpuinfo \
  backends/xnnpack/third-party/pthreadpool

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DPYTHON_EXECUTABLE="$VENV_BIN/python" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON

cmake --build "$BUILD" -j "$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) + 1 ))"
cmake --install "$BUILD"
echo "$TAG ($(git -C "$SRC" rev-parse --short HEAD))" > "$PREFIX/VERSION"
echo "ExecuTorch $TAG installed to $PREFIX"
