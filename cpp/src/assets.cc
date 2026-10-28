#include "kws/assets.h"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace kws {
namespace {

std::vector<float> ReadF32(const std::string& path, size_t expected_count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(path + ": cannot open file");
  const auto bytes = static_cast<size_t>(file.tellg());
  if (bytes != expected_count * sizeof(float)) {
    throw std::runtime_error(path + ": expected " + std::to_string(expected_count) +
                             " float32 values, file has " + std::to_string(bytes) + " bytes");
  }
  std::vector<float> values(expected_count);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  return values;
}

std::map<std::string, std::string> ReadKeyValues(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error(path + ": cannot open file");
  std::map<std::string, std::string> values;
  std::string key, value;
  while (file >> key >> value) values[key] = value;
  return values;
}

}  // namespace

Assets Assets::Load(const std::string& dir) {
  const auto kv = ReadKeyValues(dir + "/config.txt");
  const auto get = [&](const std::string& key) {
    const auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error(dir + "/config.txt: missing key " + key);
    return it->second;
  };
  if (get("format_version") != "1") throw std::runtime_error("unsupported assets format_version");

  Assets assets;
  FeatureConfig& c = assets.config;
  c.sample_rate = std::stoi(get("sample_rate"));
  c.frame_length = std::stoi(get("frame_length"));
  c.hop_length = std::stoi(get("hop_length"));
  c.n_fft = std::stoi(get("n_fft"));
  c.n_mels = std::stoi(get("n_mels"));
  c.num_fft_bins = std::stoi(get("num_fft_bins"));
  c.log_offset = std::stof(get("log_offset"));
  if (c.num_fft_bins != c.n_fft / 2 + 1 || c.frame_length > c.n_fft) {
    throw std::runtime_error("inconsistent feature config in " + dir + "/config.txt");
  }

  assets.window = ReadF32(dir + "/window.f32", c.frame_length);
  assets.mel_fbank =
      ReadF32(dir + "/mel_fbank.f32", static_cast<size_t>(c.n_mels) * c.num_fft_bins);
  assets.norm_mean = ReadF32(dir + "/norm_mean.f32", c.n_mels);
  assets.norm_std = ReadF32(dir + "/norm_std.f32", c.n_mels);

  std::ifstream labels(dir + "/labels.txt");
  for (std::string line; std::getline(labels, line);) {
    if (!line.empty()) assets.labels.push_back(line);
  }
  if (static_cast<int>(assets.labels.size()) != std::stoi(get("num_labels"))) {
    throw std::runtime_error(dir + "/labels.txt: label count does not match config.txt");
  }
  return assets;
}

}  // namespace kws
