#ifndef KWS_POSTERIOR_H_
#define KWS_POSTERIOR_H_

#include <vector>

namespace kws {

// Turns a stream of logits into keyword detections:
//   softmax -> average over the last `window` inferences -> best keyword class
//   -> fire if its smoothed score exceeds `threshold` and we are not within
//      `refractory_steps` inference steps of the previous detection.
// Buffers are allocated in the constructor; Update() does not allocate.
class PosteriorDetector {
 public:
  struct Detection {
    int label = -1;  // -1 when nothing fired.
    float score = 0.0f;
  };

  // is_keyword[c] says whether class c may trigger a detection (false for
  // _unknown_ and _silence_).
  PosteriorDetector(std::vector<bool> is_keyword, int window, float threshold,
                    int refractory_steps);

  Detection Update(const float* logits);

  // Smoothed probabilities after the last Update().
  const std::vector<float>& smoothed() const { return smoothed_; }

 private:
  int num_classes_;
  std::vector<bool> is_keyword_;
  int window_;
  float threshold_;
  int refractory_steps_;
  std::vector<float> history_;  // window x num_classes ring of probabilities.
  std::vector<float> smoothed_;
  int next_ = 0;
  int count_ = 0;
  int steps_since_detection_;
};

}  // namespace kws

#endif  // KWS_POSTERIOR_H_
