#include "kws/ring_buffer.h"

#include <algorithm>

namespace kws {

FeatureRingBuffer::FeatureRingBuffer(int capacity_rows, int row_size)
    : capacity_(capacity_rows),
      row_size_(row_size),
      data_(static_cast<size_t>(capacity_rows) * row_size) {}

void FeatureRingBuffer::Push(const float* row) {
  std::copy(row, row + row_size_, data_.begin() + static_cast<size_t>(next_) * row_size_);
  next_ = (next_ + 1) % capacity_;
  if (count_ < capacity_) ++count_;
}

void FeatureRingBuffer::CopyOrdered(float* out) const {
  // Oldest rows are [next_, capacity_), then the newest are [0, next_).
  const auto split = data_.begin() + static_cast<size_t>(next_) * row_size_;
  out = std::copy(split, data_.end(), out);
  std::copy(data_.begin(), split, out);
}

}  // namespace kws
