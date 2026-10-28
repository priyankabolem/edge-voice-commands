"""Write everything the C++ feature pipeline needs into assets/.

Format (all binary files are raw little-endian float32, row-major, no header):
    window.f32       Hann window, frame_length values
    mel_fbank.f32    mel filterbank, n_mels x (n_fft / 2 + 1)
    norm_mean.f32    per-mel-bin mean, n_mels values (training split only)
    norm_std.f32     per-mel-bin std, n_mels values
    labels.txt       one label per line, in model output order
    config.txt       "key value" lines: feature parameters and array sizes
"""

import numpy as np

from kws.data import ROOT, label_names, load_config
from kws.features import FeatureConfig, LogMelExtractor

ASSETS_DIR = ROOT / "assets"


def write_f32(name: str, array: np.ndarray):
    data = np.ascontiguousarray(array, dtype="<f4")
    (ASSETS_DIR / name).write_bytes(data.tobytes())


def main():
    cfg = load_config()
    feature_cfg = FeatureConfig.from_dict(cfg["features"], cfg["data"]["sample_rate"])
    extractor = LogMelExtractor(feature_cfg)
    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    labels = label_names(cfg)

    ASSETS_DIR.mkdir(exist_ok=True)
    write_f32("window.f32", extractor.window)
    write_f32("mel_fbank.f32", extractor.fbank)
    write_f32("norm_mean.f32", stats["mean"])
    write_f32("norm_std.f32", stats["std"])
    (ASSETS_DIR / "labels.txt").write_text("\n".join(labels) + "\n")

    config = {
        "format_version": 1,
        "sample_rate": feature_cfg.sample_rate,
        "frame_length": feature_cfg.frame_length,
        "hop_length": feature_cfg.hop_length,
        "n_fft": feature_cfg.n_fft,
        "n_mels": feature_cfg.n_mels,
        "num_fft_bins": feature_cfg.n_fft // 2 + 1,
        "log_offset": repr(feature_cfg.log_offset),
        "num_labels": len(labels),
    }
    (ASSETS_DIR / "config.txt").write_text("".join(f"{k} {v}\n" for k, v in config.items()))
    print(f"Wrote assets to {ASSETS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
