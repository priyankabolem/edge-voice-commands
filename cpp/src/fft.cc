#include "kws/fft.h"

#include <cmath>
#include <stdexcept>

#if defined(KWS_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define KWS_FFT_HAVE_NEON 1
#endif

// This file is compiled with -ffp-contract=off: every fused multiply-add is written
// explicitly (std::fma / vfmaq_f32), so the scalar and NEON paths round identically.

namespace kws {

RealFft::RealFft(int n, bool use_simd)
    : n_(n),
      use_simd_(use_simd && SimdAvailable()),
      bit_reverse_(n),
      stage_cos_(n > 1 ? n - 1 : 0),
      stage_sin_(n > 1 ? n - 1 : 0),
      re_(n),
      im_(n) {
  if (n < 2 || (n & (n - 1)) != 0) throw std::invalid_argument("FFT size must be a power of two");

  int bits = 0;
  while ((1 << bits) < n) ++bits;
  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
    bit_reverse_[i] = r;
  }
  // Twiddles are computed in double and rounded once. Each stage's table is copied
  // from the full-size table (index j * stride), as the original strided loop used.
  std::vector<float> cos_full(n / 2), sin_full(n / 2);
  for (int k = 0; k < n / 2; ++k) {
    const double angle = -2.0 * M_PI * k / n;
    cos_full[k] = static_cast<float>(std::cos(angle));
    sin_full[k] = static_cast<float>(std::sin(angle));
  }
  for (int half = 1; half < n; half <<= 1) {
    const int stride = n / (2 * half);
    for (int j = 0; j < half; ++j) {
      stage_cos_[half - 1 + j] = cos_full[j * stride];
      stage_sin_[half - 1 + j] = sin_full[j * stride];
    }
  }
}

bool RealFft::SimdAvailable() {
#ifdef KWS_FFT_HAVE_NEON
  return true;
#else
  return false;
#endif
}

void RealFft::PowerSpectrum(const float* input, float* power) {
  for (int i = 0; i < n_; ++i) {
    re_[bit_reverse_[i]] = input[i];
    im_[bit_reverse_[i]] = 0.0f;
  }
  if (use_simd_) {
    ButterfliesNeon();
    PowerNeon(power);
  } else {
    Butterflies();
    Power(power);
  }
}

namespace {

// One stage of radix-2 butterflies over blocks of 2 * half points, scalar.
// t = w * b (complex); b = a - t; a = a + t.
void ScalarStage(int n, int half, const float* wcos, const float* wsin, float* re, float* im) {
  for (int start = 0; start < n; start += 2 * half) {
    for (int j = 0; j < half; ++j) {
      const int a = start + j;
      const int b = a + half;
      const float tr = std::fma(wcos[j], re[b], -(wsin[j] * im[b]));
      const float ti = std::fma(wcos[j], im[b], wsin[j] * re[b]);
      re[b] = re[a] - tr;
      im[b] = im[a] - ti;
      re[a] = re[a] + tr;
      im[a] = im[a] + ti;
    }
  }
}

}  // namespace

void RealFft::Butterflies() {
  for (int half = 1; half < n_; half <<= 1) {
    ScalarStage(n_, half, &stage_cos_[half - 1], &stage_sin_[half - 1], re_.data(), im_.data());
  }
}

void RealFft::Power(float* power) const {
  for (int k = 0; k <= n_ / 2; ++k) power[k] = std::fma(re_[k], re_[k], im_[k] * im_[k]);
}

#ifdef KWS_FFT_HAVE_NEON

void RealFft::ButterfliesNeon() {
  float* re = re_.data();
  float* im = im_.data();
  for (int half = 1; half < n_; half <<= 1) {
    const float* wcos = &stage_cos_[half - 1];
    const float* wsin = &stage_sin_[half - 1];
    if (half < 4) {  // Fewer than 4 butterflies per block: no full vector.
      ScalarStage(n_, half, wcos, wsin, re, im);
      continue;
    }
    for (int start = 0; start < n_; start += 2 * half) {
      for (int j = 0; j < half; j += 4) {
        float* ar = re + start + j;
        float* ai = im + start + j;
        float* br = ar + half;
        float* bi = ai + half;
        const float32x4_t wr = vld1q_f32(wcos + j);
        const float32x4_t wi = vld1q_f32(wsin + j);
        const float32x4_t vbr = vld1q_f32(br);
        const float32x4_t vbi = vld1q_f32(bi);
        const float32x4_t var = vld1q_f32(ar);
        const float32x4_t vai = vld1q_f32(ai);
        // Same operations as ScalarStage: fma(wr, br, -(wi * bi)), fma(wr, bi, wi * br).
        const float32x4_t tr = vfmaq_f32(vnegq_f32(vmulq_f32(wi, vbi)), wr, vbr);
        const float32x4_t ti = vfmaq_f32(vmulq_f32(wi, vbr), wr, vbi);
        vst1q_f32(br, vsubq_f32(var, tr));
        vst1q_f32(bi, vsubq_f32(vai, ti));
        vst1q_f32(ar, vaddq_f32(var, tr));
        vst1q_f32(ai, vaddq_f32(vai, ti));
      }
    }
  }
}

void RealFft::PowerNeon(float* power) const {
  const int bins = n_ / 2 + 1;
  int k = 0;
  for (; k + 4 <= bins; k += 4) {
    const float32x4_t r = vld1q_f32(&re_[k]);
    const float32x4_t i = vld1q_f32(&im_[k]);
    vst1q_f32(power + k, vfmaq_f32(vmulq_f32(i, i), r, r));
  }
  for (; k < bins; ++k) power[k] = std::fma(re_[k], re_[k], im_[k] * im_[k]);
}

#else

void RealFft::ButterfliesNeon() { Butterflies(); }
void RealFft::PowerNeon(float* power) const { Power(power); }

#endif

}  // namespace kws
