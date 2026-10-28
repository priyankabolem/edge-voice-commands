#include "kws/posterior.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace kws {

PosteriorDetector::PosteriorDetector(std::vector<bool> is_keyword, int window, float threshold,
                                     int refractory_steps)
    : num_classes_(static_cast<int>(is_keyword.size())),
      is_keyword_(std::move(is_keyword)),
      window_(window),
      threshold_(threshold),
      refractory_steps_(refractory_steps),
      history_(static_cast<size_t>(window) * num_classes_, 0.0f),
      smoothed_(num_classes_, 0.0f),
      steps_since_detection_(refractory_steps) {}

PosteriorDetector::Detection PosteriorDetector::Update(const float* logits) {
  float* slot = &history_[static_cast<size_t>(next_) * num_classes_];
  const float max_logit = *std::max_element(logits, logits + num_classes_);
  float denom = 0.0f;
  for (int c = 0; c < num_classes_; ++c) denom += std::exp(logits[c] - max_logit);
  for (int c = 0; c < num_classes_; ++c) slot[c] = std::exp(logits[c] - max_logit) / denom;
  next_ = (next_ + 1) % window_;
  count_ = std::min(count_ + 1, window_);

  // Re-summing the small window each step avoids drift from a running float sum.
  std::fill(smoothed_.begin(), smoothed_.end(), 0.0f);
  for (int i = 0; i < count_; ++i) {
    const float* p = &history_[static_cast<size_t>(i) * num_classes_];
    for (int c = 0; c < num_classes_; ++c) smoothed_[c] += p[c];
  }
  for (int c = 0; c < num_classes_; ++c) smoothed_[c] /= static_cast<float>(count_);

  Detection best;
  for (int c = 0; c < num_classes_; ++c) {
    if (is_keyword_[c] && smoothed_[c] > best.score) best = {c, smoothed_[c]};
  }
  ++steps_since_detection_;
  if (best.score > threshold_ && steps_since_detection_ > refractory_steps_) {
    steps_since_detection_ = 0;
    return best;
  }
  return {};
}

}  // namespace kws
