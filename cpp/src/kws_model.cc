#include "kws/kws_model.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kws {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Span;

constexpr uint32_t kMethodPoolBytes = 1 << 20;
constexpr uint32_t kTempPoolBytes = 512 << 10;
constexpr int kMaxDims = 8;

void Check(Error error, const std::string& what) {
  if (error != Error::Ok) {
    throw std::runtime_error(what + " failed with ExecuTorch error " +
                             std::to_string(static_cast<int>(error)));
  }
}

}  // namespace

struct KwsModel::Impl {
  // Members are destroyed in reverse declaration order, so the Method goes first,
  // before the memory and program it refers to.
  std::unique_ptr<FileDataLoader> loader;
  std::unique_ptr<Program> program;
  std::vector<uint8_t> method_pool;
  std::vector<uint8_t> temp_pool;
  std::unique_ptr<MemoryAllocator> method_allocator;
  std::unique_ptr<MemoryAllocator> temp_allocator;
  std::vector<std::vector<uint8_t>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  std::unique_ptr<HierarchicalAllocator> planned_memory;
  std::unique_ptr<MemoryManager> memory_manager;
  std::unique_ptr<Method> method;

  TensorImpl::SizesType sizes[kMaxDims] = {};
  TensorImpl::DimOrderType dim_order[kMaxDims] = {};
  std::vector<float> input_data;
  std::unique_ptr<TensorImpl> input_impl;
  EValue input_value;
  int num_classes = 0;
};

KwsModel::KwsModel(const std::string& pte_path, executorch::runtime::EventTracer* event_tracer)
    : impl_(std::make_unique<Impl>()) {
  executorch::runtime::runtime_init();
  Impl& m = *impl_;

  auto loader = FileDataLoader::from(pte_path.c_str());
  Check(loader.error(), "opening " + pte_path);
  m.loader = std::make_unique<FileDataLoader>(std::move(loader.get()));

  auto program = Program::load(m.loader.get());
  Check(program.error(), "loading program " + pte_path);
  m.program = std::make_unique<Program>(std::move(program.get()));

  auto meta = m.program->method_meta("forward");
  Check(meta.error(), "reading method meta");
  if (meta->num_inputs() != 1 || meta->num_outputs() != 1) {
    throw std::runtime_error("expected a program with one input and one output");
  }

  for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
    m.planned_buffers.emplace_back(static_cast<size_t>(meta->memory_planned_buffer_size(i).get()));
  }
  for (auto& buffer : m.planned_buffers) m.planned_spans.emplace_back(buffer.data(), buffer.size());

  m.method_pool.resize(kMethodPoolBytes);
  m.temp_pool.resize(kTempPoolBytes);
  m.method_allocator = std::make_unique<MemoryAllocator>(kMethodPoolBytes, m.method_pool.data());
  m.temp_allocator = std::make_unique<MemoryAllocator>(kTempPoolBytes, m.temp_pool.data());
  m.planned_memory = std::make_unique<HierarchicalAllocator>(
      Span<Span<uint8_t>>(m.planned_spans.data(), m.planned_spans.size()));
  m.memory_manager = std::make_unique<MemoryManager>(
      m.method_allocator.get(), m.planned_memory.get(), m.temp_allocator.get());

  auto method = m.program->load_method("forward", m.memory_manager.get(), event_tracer);
  Check(method.error(), "loading method forward");
  m.method = std::make_unique<Method>(std::move(method.get()));

  auto input_meta = meta->input_tensor_meta(0);
  Check(input_meta.error(), "reading input tensor meta");
  if (input_meta->scalar_type() != ScalarType::Float)
    throw std::runtime_error("input must be float32");
  const auto dims = input_meta->sizes();
  if (dims.size() > kMaxDims) throw std::runtime_error("input has too many dimensions");
  size_t numel = 1;
  for (size_t d = 0; d < dims.size(); ++d) {
    m.sizes[d] = dims[d];
    m.dim_order[d] = static_cast<TensorImpl::DimOrderType>(d);
    numel *= static_cast<size_t>(dims[d]);
  }
  m.input_data.assign(numel, 0.0f);
  m.input_impl = std::make_unique<TensorImpl>(ScalarType::Float, static_cast<ssize_t>(dims.size()),
                                              m.sizes, m.input_data.data(), m.dim_order);
  m.input_value = EValue(Tensor(m.input_impl.get()));

  auto output_meta = meta->output_tensor_meta(0);
  Check(output_meta.error(), "reading output tensor meta");
  m.num_classes = static_cast<int>(output_meta->nbytes() / sizeof(float));
}

KwsModel::~KwsModel() = default;

float* KwsModel::input() { return impl_->input_data.data(); }

int KwsModel::input_size() const { return static_cast<int>(impl_->input_data.size()); }

int KwsModel::num_classes() const { return impl_->num_classes; }

const float* KwsModel::Run() {
  if (impl_->method->set_input(impl_->input_value, 0) != Error::Ok) return nullptr;
  if (impl_->method->execute() != Error::Ok) return nullptr;
  return impl_->method->get_output(0).toTensor().const_data_ptr<float>();
}

}  // namespace kws
