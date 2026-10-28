"""Speech Commands v0.02: labels, official splits, _unknown_/_silence_ construction,
and training-time waveform augmentation."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("training", "validation", "testing")
NOISE_DIR = "_background_noise_"


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def label_names(cfg: dict) -> list[str]:
    data = cfg["data"]
    return list(data["keywords"]) + [data["unknown_label"], data["silence_label"]]


def read_wav_int16(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if sr != sample_rate or audio.ndim != 1:
        raise ValueError(
            f"{path}: expected mono {sample_rate} Hz, got {sr} Hz, shape {audio.shape}"
        )
    return audio


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad at the end or truncate to exactly `length` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768.0


def speaker_id(relpath: str) -> str:
    return Path(relpath).name.split("_nohash_")[0]


def official_split_files(dataset_dir: Path) -> dict[str, list[str]]:
    """Return {split: [relative wav path, ...]} using the official list files."""
    root = dataset_dir.parent.parent
    files = {}
    for split in SPLITS:
        ds = SPEECHCOMMANDS(str(root), url="speech_commands_v0.02", subset=split)
        files[split] = sorted(
            str(Path(ds.get_metadata(i)[0]).relative_to(dataset_dir.name)) for i in range(len(ds))
        )
    return files


@dataclass
class NoiseRegion:
    """A time range [start, end) of one background-noise recording."""

    name: str
    audio: np.ndarray  # int16, full recording
    start: int
    end: int

    def random_crop(self, rng: random.Random, length: int) -> np.ndarray:
        offset = rng.randint(self.start, self.end - length)
        return self.audio[offset : offset + length]


def load_noise_regions(dataset_dir: Path, cfg: dict) -> dict[str, list[NoiseRegion]]:
    """Cut every noise file by time into disjoint per-split regions."""
    fractions = cfg["data"]["noise_split"]
    regions = {split: [] for split in SPLITS}
    for path in sorted((dataset_dir / NOISE_DIR).glob("*.wav")):
        audio = read_wav_int16(path, cfg["data"]["sample_rate"])
        start = 0
        for split in SPLITS:
            end = start + int(round(fractions[split] * len(audio)))
            regions[split].append(NoiseRegion(path.name, audio, start, min(end, len(audio))))
            start = end
    return regions


def build_split_index(dataset_dir: Path, cfg: dict, seed: int) -> dict[str, list[dict]]:
    """Build the list of examples for each split.

    Keyword clips are used as-is. _unknown_ is sampled (without replacement) from all
    non-keyword words of the same split, sized to the mean keyword-class count.
    _silence_ examples are 1 s noise crops from the split's own noise region, sized so
    they make up `silence_fraction` of the split.
    """
    data_cfg = cfg["data"]
    keywords = set(data_cfg["keywords"])
    clip_samples = data_cfg["clip_samples"]
    split_files = official_split_files(dataset_dir)
    noise_regions = load_noise_regions(dataset_dir, cfg)
    rng = random.Random(seed)

    index = {}
    for split in SPLITS:
        keyword_items, unknown_pool = [], []
        for relpath in split_files[split]:
            word = Path(relpath).parent.name
            item = {
                "path": relpath,
                "label": word if word in keywords else data_cfg["unknown_label"],
            }
            (keyword_items if word in keywords else unknown_pool).append(item)

        n_unknown = int(round(len(keyword_items) / len(keywords)))
        unknown_items = rng.sample(unknown_pool, min(n_unknown, len(unknown_pool)))

        n_speech = len(keyword_items) + len(unknown_items)
        frac = data_cfg["silence_fraction"]
        n_silence = int(round(frac / (1.0 - frac) * n_speech))
        silence_items = []
        for _ in range(n_silence):
            region = rng.choice(noise_regions[split])
            offset = rng.randint(region.start, region.end - clip_samples)
            gain = rng.uniform(0.0, data_cfg["silence_max_gain"])
            silence_items.append(
                {
                    "path": None,
                    "label": data_cfg["silence_label"],
                    "noise": region.name,
                    "offset": offset,
                    "gain": gain,
                }
            )

        index[split] = keyword_items + unknown_items + silence_items
    return index


def load_example(
    item: dict, dataset_dir: Path, noise_audio: dict[str, np.ndarray], clip_samples: int
) -> np.ndarray:
    """Return the int16 waveform of one index entry, exactly `clip_samples` long."""
    if item["path"] is not None:
        return pad_or_trim(read_wav_int16(dataset_dir / item["path"]), clip_samples)
    crop = noise_audio[item["noise"]][item["offset"] : item["offset"] + clip_samples]
    scaled = np.round(crop.astype(np.float64) * item["gain"])
    return np.clip(scaled, -32768, 32767).astype(np.int16)


class Augmenter:
    """Training-only waveform augmentation: random time shift, then background-noise mixing."""

    def __init__(self, train_noise: list[NoiseRegion], cfg: dict):
        aug = cfg["augmentation"]
        self.max_shift = int(aug["time_shift_ms"] * cfg["data"]["sample_rate"] / 1000)
        self.noise_prob = aug["noise_prob"]
        self.noise_max_gain = aug["noise_max_gain"]
        self.noise = train_noise

    def time_shift(self, audio: np.ndarray, shift: int) -> np.ndarray:
        """Shift right (positive) or left (negative), filling with zeros."""
        out = np.zeros_like(audio)
        if shift > 0:
            out[shift:] = audio[:-shift]
        elif shift < 0:
            out[:shift] = audio[-shift:]
        else:
            out[:] = audio
        return out

    def __call__(self, audio: np.ndarray, rng: random.Random) -> np.ndarray:
        """Float waveform in [-1, 1] -> augmented float waveform in [-1, 1]."""
        out = self.time_shift(audio, rng.randint(-self.max_shift, self.max_shift))
        if rng.random() < self.noise_prob:
            region = rng.choice(self.noise)
            noise = int16_to_float(region.random_crop(rng, len(audio)))
            out = out + rng.uniform(0.0, self.noise_max_gain) * noise
        return np.clip(out, -1.0, 1.0).astype(np.float32)
