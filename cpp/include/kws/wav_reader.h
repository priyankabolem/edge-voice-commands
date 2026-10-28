#ifndef KWS_WAV_READER_H_
#define KWS_WAV_READER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace kws {

// Reads a RIFF/WAVE file containing 16-bit PCM, mono, at `expected_sample_rate`.
// Any other format is rejected with std::runtime_error describing what was found.
std::vector<int16_t> ReadWavPcm16Mono(const std::string& path, int expected_sample_rate = 16000);

// Zero-pads at the end or truncates to exactly `length` samples (matches kws.data.pad_or_trim).
void PadOrTrim(std::vector<int16_t>* samples, size_t length);

}  // namespace kws

#endif  // KWS_WAV_READER_H_
