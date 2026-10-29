#ifndef KWS_FFT_H_
#define KWS_FFT_H_

#include <vector>

namespace kws {

// Iterative radix-2 FFT of a real signal. Twiddles, the bit-reversal table and
// scratch buffers are allocated once in the constructor; PowerSpectrum() does not
// allocate.
//
// On ARM builds with KWS_ENABLE_NEON, butterfly stages with at least 4 butterflies per
// block and the power computation use NEON; otherwise (or with use_simd = false) the
// scalar path runs. Both paths perform the same fused multiply-adds in the same order,
// so their outputs are bit-identical.
class RealFft {
 public:
  explicit RealFft(int n, bool use_simd = true);

  // True when this build contains the NEON path.
  static bool SimdAvailable();

  int size() const { return n_; }
  int num_bins() const { return n_ / 2 + 1; }
  bool using_simd() const { return use_simd_; }

  // input: n real samples. power: n/2 + 1 values of |X[k]|^2.
  void PowerSpectrum(const float* input, float* power);

 private:
  void Butterflies();
  void ButterfliesNeon();
  void Power(float* power) const;
  void PowerNeon(float* power) const;

  int n_;
  bool use_simd_;
  std::vector<int> bit_reverse_;
  // Twiddles for each stage stored contiguously: the stage with half-size h uses
  // entries [h - 1, 2h - 1), i.e. e^(-2 pi i j / 2h) for j = 0..h-1.
  std::vector<float> stage_cos_;
  std::vector<float> stage_sin_;
  std::vector<float> re_;
  std::vector<float> im_;
};

}  // namespace kws

#endif  // KWS_FFT_H_
