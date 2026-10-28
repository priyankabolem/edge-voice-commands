#ifndef KWS_KWS_MODEL_H_
#define KWS_KWS_MODEL_H_

#include <memory>
#include <string>

namespace kws {

// Runs a keyword-spotting .pte program with the ExecuTorch runtime.
//
// All runtime memory (method allocator, memory-planned buffers, input tensor) is
// allocated in the constructor. Run() copies input() into the planned input buffer
// and executes without heap allocation. Load errors throw std::runtime_error.
class KwsModel {
 public:
  explicit KwsModel(const std::string& pte_path);
  ~KwsModel();
  KwsModel(const KwsModel&) = delete;
  KwsModel& operator=(const KwsModel&) = delete;

  // Input buffer of input_size() floats (1 x 1 x frames x mels); fill before Run().
  float* input();
  int input_size() const;
  int num_classes() const;

  // Returns num_classes() logits, valid until the next Run(); nullptr on runtime error.
  const float* Run();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kws

#endif  // KWS_KWS_MODEL_H_
