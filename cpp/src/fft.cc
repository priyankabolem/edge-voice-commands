#include "kws/fft.h"

#include <cmath>
#include <stdexcept>

namespace kws {

RealFft::RealFft(int n) : n_(n), bit_reverse_(n), cos_(n / 2), sin_(n / 2), re_(n), im_(n) {
  if (n < 2 || (n & (n - 1)) != 0) throw std::invalid_argument("FFT size must be a power of two");

  int bits = 0;
  while ((1 << bits) < n) ++bits;
  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
    bit_reverse_[i] = r;
  }
  // Twiddles computed in double, then rounded once, to keep float error minimal.
  for (int k = 0; k < n / 2; ++k) {
    const double angle = -2.0 * M_PI * k / n;
    cos_[k] = static_cast<float>(std::cos(angle));
    sin_[k] = static_cast<float>(std::sin(angle));
  }
}

void RealFft::PowerSpectrum(const float* input, float* power) {
  for (int i = 0; i < n_; ++i) {
    re_[bit_reverse_[i]] = input[i];
    im_[bit_reverse_[i]] = 0.0f;
  }
  for (int len = 2; len <= n_; len <<= 1) {
    const int half = len / 2;
    const int stride = n_ / len;
    for (int start = 0; start < n_; start += len) {
      for (int j = 0; j < half; ++j) {
        const float wr = cos_[j * stride];
        const float wi = sin_[j * stride];
        const int a = start + j;
        const int b = a + half;
        const float tr = wr * re_[b] - wi * im_[b];
        const float ti = wr * im_[b] + wi * re_[b];
        re_[b] = re_[a] - tr;
        im_[b] = im_[a] - ti;
        re_[a] += tr;
        im_[a] += ti;
      }
    }
  }
  for (int k = 0; k <= n_ / 2; ++k) power[k] = re_[k] * re_[k] + im_[k] * im_[k];
}

}  // namespace kws
