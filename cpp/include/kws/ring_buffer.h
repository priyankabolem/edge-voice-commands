#ifndef KWS_RING_BUFFER_H_
#define KWS_RING_BUFFER_H_

#include <vector>

namespace kws {

// Fixed-capacity ring of feature rows (e.g. the last 98 frames x 40 mels).
// Storage is allocated once; Push() and CopyOrdered() do not allocate.
class FeatureRingBuffer {
 public:
  FeatureRingBuffer(int capacity_rows, int row_size);

  // Copies `row` (row_size values) in, overwriting the oldest row when full.
  void Push(const float* row);

  // Writes all rows, oldest first, as capacity_rows x row_size contiguous values.
  // Only meaningful once full().
  void CopyOrdered(float* out) const;

  bool full() const { return count_ == capacity_; }
  int size() const { return count_; }

 private:
  int capacity_;
  int row_size_;
  int next_ = 0;  // Row index the next Push() writes to (also the oldest row when full).
  int count_ = 0;
  std::vector<float> data_;
};

}  // namespace kws

#endif  // KWS_RING_BUFFER_H_
