#ifndef KWS_FFT_H_
#define KWS_FFT_H_

#include <vector>

namespace kws {

// Iterative radix-2 FFT of a real signal. Twiddles, the bit-reversal table and
// scratch buffers are allocated once in the constructor; PowerSpectrum() does not
// allocate.
class RealFft {
 public:
  explicit RealFft(int n);  // n must be a power of two.

  int size() const { return n_; }
  int num_bins() const { return n_ / 2 + 1; }

  // input: n real samples. power: n/2 + 1 values of |X[k]|^2.
  void PowerSpectrum(const float* input, float* power);

 private:
  int n_;
  std::vector<int> bit_reverse_;
  std::vector<float> cos_;
  std::vector<float> sin_;
  std::vector<float> re_;
  std::vector<float> im_;
};

}  // namespace kws

#endif  // KWS_FFT_H_
