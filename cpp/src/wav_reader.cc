#include "kws/wav_reader.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

namespace kws {
namespace {

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatExtensible = 0xFFFE;

// WAV is little-endian; assembling values byte by byte keeps this host-independent.
uint16_t ReadU16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

uint32_t ReadU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

bool TagIs(const uint8_t* p, const char* tag) {
  return p[0] == tag[0] && p[1] == tag[1] && p[2] == tag[2] && p[3] == tag[3];
}

}  // namespace

std::vector<int16_t> ReadWavPcm16Mono(const std::string& path, int expected_sample_rate) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path + ": cannot open file");
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
  const auto fail = [&](const std::string& msg) { throw std::runtime_error(path + ": " + msg); };

  if (bytes.size() < 12 || !TagIs(&bytes[0], "RIFF") || !TagIs(&bytes[8], "WAVE")) {
    fail("not a RIFF/WAVE file");
  }

  bool have_format = false;
  size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    const uint8_t* chunk = &bytes[pos];
    const uint32_t chunk_size = ReadU32(chunk + 4);
    const size_t body = pos + 8;
    if (body + chunk_size > bytes.size()) fail("truncated chunk");

    if (TagIs(chunk, "fmt ")) {
      if (chunk_size < 16) fail("fmt chunk too small");
      const uint16_t format = ReadU16(&bytes[body]);
      const uint16_t channels = ReadU16(&bytes[body + 2]);
      const uint32_t sample_rate = ReadU32(&bytes[body + 4]);
      const uint16_t bits = ReadU16(&bytes[body + 14]);
      if (format != kFormatPcm && format != kFormatExtensible) {
        fail("unsupported audio format " + std::to_string(format) + " (expected PCM)");
      }
      if (channels != 1) fail(std::to_string(channels) + " channels (expected mono)");
      if (bits != 16) fail(std::to_string(bits) + "-bit samples (expected 16-bit)");
      if (static_cast<int>(sample_rate) != expected_sample_rate) {
        fail(std::to_string(sample_rate) + " Hz (expected " + std::to_string(expected_sample_rate) +
             " Hz)");
      }
      have_format = true;
    } else if (TagIs(chunk, "data")) {
      if (!have_format) fail("data chunk before fmt chunk");
      std::vector<int16_t> samples(chunk_size / 2);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(ReadU16(&bytes[body + 2 * i]));
      }
      return samples;
    }
    pos = body + chunk_size + (chunk_size & 1);  // Chunks are padded to even sizes.
  }
  fail("no data chunk");
  return {};
}

void PadOrTrim(std::vector<int16_t>* samples, size_t length) { samples->resize(length, 0); }

}  // namespace kws
