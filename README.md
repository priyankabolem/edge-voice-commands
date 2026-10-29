# edge-voice-commands

Real-time voice command recognition that runs fully on-device on a CPU. A small keyword-spotting model is trained in PyTorch, quantized to INT8, exported with ExecuTorch (PyTorch's on-device runtime), and run from C++ with a hand-written audio front end and a streaming detector.

## Overview

The model recognizes ten commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`) plus `_unknown_` and `_silence_` from 1 s windows of 16 kHz audio (Google Speech Commands v0.02).

I built the project in stages and measured each one before moving on:

1. Data splits and a NumPy reference log-mel front end
2. A depthwise-separable CNN (DS-CNN) trained under a 50K-parameter budget
3. INT8 post-training quantization (PT2E flow, XNNPACK quantizer)
4. Export to ExecuTorch `.pte` programs with the XNNPACK backend, checked against PyTorch
5. A C++ pipeline (WAV reader, FFT, log-mel, ExecuTorch runner) checked against Python
6. Latency benchmarks and per-operator profiling
7. A streaming detector that processes audio in 10 ms hops

Every number in this README is generated from the JSON files in `results/` by `scripts/render_results.py`. Those files are written by the scripts in this repository.

## Pipeline

```
                    Python (training and export)
  Speech Commands -> kws/data.py -> kws/features.py (NumPy log-mel, reference)
                  -> kws/train.py (DS-CNN) -> kws/quantize.py (PT2E INT8)
                  -> kws/export.py (ExecuTorch .pte, XNNPACK delegate)
                  -> kws/export_assets.py (window, mel filterbank, norm stats, labels)

                    C++ (inference)
  16 kHz PCM, 10 ms hops
    -> wav_reader -> fft + log_mel (one 25 ms frame per hop, 40 mel bins)
    -> ring_buffer (last 98 frames, about 1 s)
    -> kws_model (ExecuTorch Program/Method, XNNPACK) every 2 hops
    -> posterior (softmax, smoothing, threshold, refractory period) -> detection
```

## Results

Measured on the machine listed under [Environment](#environment).

### Model

<!-- results:model -->
| Parameters | Budget | MACs per inference | Best epoch (val) | Training time |
|---|---|---|---|---|
| 23,180 | 50,000 | 20,823,808 | 26 (95.12%) | 15.8 min (mps) |
<!-- /results:model -->

### Accuracy across runtimes

<!-- results:accuracy -->
| Precision | Runtime | Test accuracy |
|---|---|---|
| FP32 | PyTorch (eager) | 94.56% |
| INT8 | PyTorch (PT2E reference) | 94.36% |
| FP32 | ExecuTorch `.pte`, Python runtime | 94.56% |
| INT8 | ExecuTorch `.pte`, Python runtime | 94.40% |
| FP32 | C++ features + ExecuTorch | 94.56% |
| INT8 | C++ features + ExecuTorch | 94.40% |

Test set: 4,979 clips, 12 classes.
<!-- /results:accuracy -->

### Size

<!-- results:sizes -->
| Artifact | Size |
|---|---|
| FP32 PyTorch checkpoint (`.pt`) | 114.3 KB (117,049 B) |
| FP32 ExecuTorch program (`.pte`) | 97.7 KB (100,016 B) |
| INT8 ExecuTorch program (`.pte`) | 43.4 KB (44,464 B) |
<!-- /results:sizes -->

### Quantization experiments

<!-- results:quantization -->
| Calibration clips | Weights | Test accuracy | Drop (points) | Top-1 agreement with FP32 | Logit SQNR (dB) |
|---|---|---|---|---|---|
| 100 | per-channel | 94.44% | 0.12 | 98.84% | 27.77 |
| 100 | per-tensor | 94.28% | 0.28 | 98.53% | 25.15 |
| 500 | per-channel | 94.36% | 0.20 | 99.10% | 27.02 |
| 500 | per-tensor | 94.38% | 0.18 | 98.47% | 25.41 |
| 2000 | per-channel | 94.50% | 0.06 | 99.14% | 26.94 |
| 2000 | per-tensor | 94.32% | 0.24 | 98.39% | 24.59 |
<!-- /results:quantization -->

### Parity checks

<!-- results:parity -->
| Comparison (full test set) | Max abs logit diff | Top-1 agreement |
|---|---|---|
| PyTorch FP32 vs `.pte` FP32 | 6.20e-06 | 100.00% |
| PyTorch INT8 vs `.pte` INT8 | 0.315 (6 output steps) | 99.60% |
| C++ FP32 vs Python `.pte` FP32 | 2.11e-05 | 100.00% |
| C++ INT8 vs Python `.pte` INT8 | 0.158 | 100.00% |

C++ vs NumPy log-mel features on 200 test WAV files: max abs difference 1.06e-04 (target < 1e-03).
Heap allocations over 1000 iterations of the per-frame, per-clip and per-inference paths: 0.
<!-- /results:parity -->

The INT8 differences are whole steps of the output quantization scale, and they only change the top-1 class when the top two classes are nearly tied. This is expected when integer kernels are compared with a float simulation of quantization (`results/parity_python.json` records the margins).

### Latency (C++)

<!-- results:latency -->
| Config | Inference median | p95 | p99 | max | Feature frame median | Feature clip median |
|---|---|---|---|---|---|---|
| FP32, 1 thread | 582.5 | 597.8 | 615.3 | 655.9 | 2.5 | 237.4 |
| FP32, 4 threads | 217.1 | 238.5 | 295.0 | 379.5 | 2.5 | 237.3 |
| INT8, 1 thread | 301.5 | 309.1 | 319.5 | 335.4 | 2.5 | 237.5 |
| INT8, 4 threads | 137.0 | 152.6 | 176.2 | 262.2 | 2.5 | 237.7 |

All values in microseconds. Method: 50 warm-up + 1000 timed iterations per stage, steady_clock, 3 process repeats; summary = median across repeats.
<!-- /results:latency -->

### Where the time goes

<!-- results:profile -->
| Model | XNNPACK operator | Median (us) | Share of inference |
|---|---|---|---|
| FP32 | Convolution (NHWC, F32) IGEMM #1 | 102.0 | 17.80% |
| FP32 | Fully Connected (NC, F32) GEMM #1 | 86.0 | 15.01% |
| FP32 | Fully Connected (NC, F32) GEMM #2 | 86.0 | 15.01% |
| INT8 | Convolution (NHWC, QC8) IGEMM #1 | 108.0 | 36.36% |
| INT8 | Fully Connected (NC, QS8, QC8W) GEMM #1 | 29.0 | 9.76% |
| INT8 | Fully Connected (NC, QS8, QC8W) GEMM #2 | 29.0 | 9.76% |

Single thread, 200 iterations, recorded with ETDump.
<!-- /results:profile -->

The first convolution has a single input channel, so XNNPACK cannot vectorize it across channels. As a result it costs far more time per multiply-accumulate than the pointwise layers, and INT8 does not speed it up (`results/op_profile.json` has every operator and the MAC count of each layer).

### Streaming

<!-- results:streaming -->
| Metric (test stream) | Value |
|---|---|
| Stream length / keywords | 120.8 s / 30 |
| Threshold (chosen on validation stream) | 0.6 |
| Detection rate | 76.67% (23/30) |
| False alarms | 6 (178.8 per hour) |
| Median detection delay after speech onset | 0.217 s |
| Per-hop processing p99 / max | 317.6 / 370.5 us |
| Hop budget / overruns | 10000 us / 0 |
| Real-time factor | 0.0153 |
| Heap allocations in streaming loop | 0 |

Threshold sweep:

| Threshold | Validation: detected / false alarms | Test: detected / false alarms |
|---|---|---|
| 0.3 | 24 / 27 | 25 / 30 |
| 0.4 | 27 / 14 | 26 / 15 |
| 0.5 | 25 / 7 | 26 / 7 |
| 0.6 | 24 / 5 | 23 / 6 |
| 0.7 | 21 / 5 | 21 / 3 |
| 0.8 | 16 / 5 | 19 / 1 |
| 0.9 | 13 / 3 | 15 / 1 |
<!-- /results:streaming -->

Each false alarm and miss is listed in `results/streaming.json` together with the nearest ground-truth event.

### Streaming-aware training and consecutive detections

This section is an experiment. The baseline model and detector (one inference above the threshold) remain the defaults used everywhere else in this README, in `make all`, and by the C++ apps. The experimental model is only built with `make extra2` and writes to its own files (`models/kws_stream_*`, `results/streaming_model/`).

To reduce the streaming errors above, I tried two changes, with their settings fixed before looking at any results:

- **Streaming-aware model** (`--variant streaming_model` in `configs/config.yaml`). Training uses time shifts of up to 300 ms instead of 100 ms, as long as most of the keyword stays inside the window. In addition, 20% of keyword clips are shifted so that only the end of the word is left at the start of the window and relabelled `_unknown_` (word-tail negatives), which targets errors like the end of "stop" being read as "up".
- **Consecutive detections**: the detector fires only after the same keyword stays above the threshold for 3 inferences in a row, instead of 1.

All four combinations use the Stage 7 method: the threshold is chosen on the validation stream, and the same test stream is scored.

<!-- results:streaming_comparison -->
| Model | Consecutive | Clip accuracy (INT8 `.pte`) | Threshold (from validation) | Validation: detected / FA | Test: detected / FA | Test FA per hour | Test median delay |
|---|---|---|---|---|---|---|---|
| Baseline | 1 | 94.40% | 0.6 | 24 / 5 | 23 / 6 | 178.8 | 0.217 s |
| Baseline | 3 | 94.40% | 0.6 | 24 / 5 | 22 / 5 | 149.0 | 0.262 s |
| Streaming-aware | 1 | 92.05% | 0.7 | 22 / 4 | 21 / 3 | 89.4 | 0.247 s |
| Streaming-aware | 3 | 92.05% | 0.6 | 22 / 5 | 22 / 3 | 89.4 | 0.265 s |
<!-- /results:streaming_comparison -->

The streaming-aware model has fewer false alarms on the test stream, but not on the validation stream, and it loses clip accuracy. With 30 keywords per stream, the differences between these rows are one to three events, so I do not consider any of them a reliable improvement. One effect is consistent: word-tail negatives lower the model's confidence on word endings, so the "stop" → "up" false alarm no longer fires with three consecutive detections. A larger evaluation set is needed before choosing between these configurations, so I kept the baseline as the default. `results/streaming_comparison.json` lists every false alarm and miss for each run.

### Prebuilt vs source-built ExecuTorch runtime

The results above use the prebuilt C++ runtime that ships in the `executorch` pip wheel. To check whether that choice costs anything, I also built ExecuTorch v1.5.1 from source (`scripts/build_executorch.sh`: Release, XNNPACK, data loader, devtools and event tracer) and linked the same C++ code against it statically.

<!-- results:runtimes -->
|  | Prebuilt (pip wheel) | Built from source |
|---|---|---|
| ExecuTorch | 1.5.1 | v1.5.1 (3b60683) |
| Linking | shared libraries | static |
| `stream` binary | 104.0 KB (106,504 B) | 1836.9 KB (1,880,952 B) |
| Binary + ExecuTorch shared libraries | 2693.1 KB (2,757,768 B) | 1836.9 KB (1,880,952 B) |
| INT8, 1 thread: inference median / p99 (us) | 301.0 / 319.4 | 301.2 / 317.3 |
| INT8, 4 threads: inference median / p99 (us) | 137.0 / 168.0 | 136.2 / 176.0 |
| FP32, 1 thread: inference median / p99 (us) | 582.6 / 607.7 | 582.7 / 607.2 |
| FP32, 4 threads: inference median / p99 (us) | 217.8 / 316.5 | 217.4 / 329.5 |
| C++ FP32 test accuracy (agreement with Python `.pte`) | 94.56% (100.00%) | 94.56% (100.00%) |
| C++ INT8 test accuracy (agreement with Python `.pte`) | 94.40% (100.00%) | 94.40% (100.00%) |
| Hot-path heap allocations | 0 | 0 |

Identical streaming detections on the test stream: yes. Latency method: benchmark config; builds interleaved per repeat; median across repeats.
<!-- /results:runtimes -->

The two runtimes produce the same outputs and the same latency within run-to-run noise, so for this model the prebuilt runtime gives up no performance. The source build is still useful for what the wheel cannot do: one static binary with a smaller total size, control over build options (the source Release build compiles logging out), and the starting point for cross-compiling to another device.

## How to reproduce

Requirements: macOS (Apple Silicon or Intel) or Linux, Python 3.11, and [uv](https://github.com/astral-sh/uv). CMake, Ninja and clang-format are installed into the virtual environment from `requirements.txt`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
make all        # downloads the dataset (2.3 GB) and regenerates every file in results/
make readme     # re-renders the README tables from results/
make test lint
```

To build ExecuTorch from source and compare it with the prebuilt runtime:

```bash
make extra1     # scripts/build_executorch.sh, cpp/build-source, results/runtime_comparison.json
make readme
```

`scripts/build_executorch.sh` clones tag v1.5.1 into `third_party/` with only the submodules this configuration needs, then configures and installs it:

```bash
cmake -S third_party/executorch -B third_party/executorch-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=third_party/executorch-install \
  -DPYTHON_EXECUTABLE=.venv/bin/python -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
cmake --build third_party/executorch-build -j "$(( $(sysctl -n hw.ncpu) + 1 ))"
cmake --install third_party/executorch-build
```

| Target | What it does | Writes |
|---|---|---|
| `make stage0` | Environment record and ExecuTorch smoke test | `results/environment.json`, `results/smoke_test.json` |
| `make stage1` | Download data, build splits, precompute features | `results/data_summary.json` |
| `make stage2` | Train and evaluate the FP32 model | `results/train_log.json`, `fp32_pytorch.json`, `model_summary.json` |
| `make stage3` | INT8 quantization and experiments | `results/int8_pytorch.json`, `quant_experiments.json` |
| `make stage4` | Export `.pte` files and check them in Python | `results/delegation.txt`, `pte_eval.json`, `parity_python.json` |
| `make stage5` | Export C++ assets, build C++, C++ parity | `results/parity_cpp.json` |
| `make stage6` | Latency benchmark and per-operator profile | `results/benchmark.json`, `op_profile.json` |
| `make stage7` | Build streams, streaming evaluation | `results/streaming.json` |

The C++ apps in `cpp/build/` can also be used directly:

```bash
cpp/build/classify_wav models/kws_int8.pte assets clip.wav
cpp/build/stream models/kws_int8.pte assets long_recording.wav --threshold 0.6
cpp/build/bench models/kws_int8.pte assets --threads 1
```

## Repository layout

```
kws/            Python package: data, features, model, train, evaluate, quantize, export,
                export_assets, parity_cpp, benchmark, profile_ops, evaluate_stream
scripts/        dataset download, feature precompute, stream generation, environment record,
                README rendering, ExecuTorch source build
cpp/            CMake project: include/kws, src (wav_reader, fft, log_mel, ring_buffer,
                posterior, kws_model, threads, assets), apps, tools/alloc_counter
tests/          pytest suite; tests that need the dataset or the C++ build skip without them
configs/        config.yaml with every hyperparameter and path
results/        measurement outputs (committed)
```

## Design decisions

- **One reference for the features.** `kws/features.py` writes out every step: periodic Hann window, zero padding to 512, power spectrum, HTK mel filterbank, `log(mel + 1e-6)`, per-bin normalization. The C++ code reads the window, filterbank and normalization statistics from exported files instead of re-deriving them, so the two implementations cannot drift apart. Training also computes its features through the same NumPy code after waveform augmentation.
- **No leakage between splits.** I used the official speaker-disjoint split lists. Each background-noise file is cut by time into separate train, validation and test regions, which are used both for `_silence_` examples and for augmentation noise. Normalization statistics and quantization calibration use training data only. The streaming threshold is chosen on a stream built from validation clips and reported on a stream built from test clips.
- **Per-channel INT8 weights.** All calibration settings give similar test accuracy. Per-channel weights consistently track the FP32 model more closely (higher logit SQNR, higher top-1 agreement), and the gain is largest for depthwise layers once BatchNorm is folded in.
- **ExecuTorch `Program`/`Method` API instead of `Module`.** `Module::forward()` returns a `std::vector`, which allocates on every call. I set up the method allocator, the memory-planned buffers and the input tensor once in the constructor, so each inference is a copy-in, `execute()`, and a pointer to the output.
- **Measured, not assumed, zero allocation.** `cpp/tools/alloc_counter` interposes `malloc`/`calloc`/`realloc` for the whole process, including calls from inside ExecuTorch, XNNPACK and libc++. A self-test makes sure the counter is actually active before any result is trusted. The streaming loop is counted end to end; the model's one-time lazy setup is moved into a warm-up inference at startup.
- **Small, owned DSP code.** The 512-point radix-2 FFT is about 50 lines of float32 code with preallocated scratch buffers. The mel filterbank is stored sparsely, so each filter only touches its non-zero bins.
- **Incremental streaming features.** Each 10 ms hop completes exactly one new 25 ms frame. The stream computes only that frame and keeps the last 98 in a ring buffer. A test checks that the ring contents are bit-identical to clip-mode features for the same second of audio.
- **Single-threaded INT8 as the default.** Four threads lower median latency but widen the tail, and one thread already fits the 10 ms hop budget with a large margin.

## Limitations

- **Hardware.** Everything runs on a laptop CPU through XNNPACK, not on a DSP or NPU, and I did not measure power or energy.
- **Streaming test data.** The streaming test is a synthetic stream of concatenated clips mixed with background noise, not a live microphone or natural continuous speech. Each stream is about two minutes with 30 keywords, so one keyword or one false alarm moves the rates noticeably.
- **Dataset.** Speech Commands is mostly clean, isolated words, recorded close to the microphone.
- **Streaming accuracy.** Streaming detection is clearly weaker than clip accuracy. The model was trained on whole, roughly centred words, so partial words at the edge of the sliding window can trigger false alarms, and very short words can stay below the threshold. Training with larger time shifts and word-tail negatives, and requiring consecutive detections, did not give a reliable improvement on the two 2-minute streams (see above); a larger streaming evaluation set is the next step.
- **Ground-truth onsets.** Speech onsets in the streaming ground truth are estimated with an energy threshold. Clips that start with background noise get an onset that is too early, which can turn a correct detection into a miss plus a false alarm.
- **Runtime builds.** The main C++ results use the prebuilt ExecuTorch runtime from the pip wheel. I compared it with a v1.5.1 source build on this one machine and model only (see above). The v1.5.1 source install does not ship the ETDump headers, so per-operator profiling (`profile_ops`) is built only against the prebuilt runtime.
- **Thread control.** The XNNPACK thread count is set through ExecuTorch's experimental `_unsafe_reset_threadpool` API (the header is not part of the prebuilt SDK), and it must be called before a model is loaded.
- **Measurement noise.** All measurements come from one machine. macOS does not allow pinning threads to cores, so tail latencies include OS scheduling effects.
- **Reproducibility of training.** Training on the Apple GPU (MPS) is not bit-for-bit reproducible, so retraining can change the last digits of the accuracy numbers.
- **Allocation check platform.** The allocation counter uses dyld interposing and is only built on macOS.

## Environment

<!-- results:environment -->
|  |  |
|---|---|
| OS | macOS 26.6.2 (arm64) |
| CPU | Apple M3, 4 performance + 4 efficiency cores |
| Python | 3.11.15 |
| PyTorch / torchao / ExecuTorch | 2.15.0 / 0.18.0 / 1.5.1 |
| C++ compiler / flags | AppleClang 21.0.0.21000334 / `-O3 -DNDEBUG` (Release) |
| ExecuTorch C++ runtime | prebuilt (executorch pip wheel) |
| CMake | cmake version 4.4.4 |
<!-- /results:environment -->

## Continuous integration

GitHub Actions runs `ruff` (lint and format check), `clang-format`, the README freshness check (`scripts/render_results.py --check`), and the Python test suite on Linux. CI does not download the dataset or build the C++ code; tests that need either one are skipped there. The model, quantization and export tests run on an untrained model.
