"""Build a ~2-minute streaming test file from one split's clips.

Keyword clips, _unknown_ words and pure-noise gaps are placed one after another on a
continuous background-noise bed. Clips and noise come only from the chosen split (noise
from that split's region of each background file), so the test stream never contains
training or validation audio.

Writes data/stream/<split>_stream.wav and <split>_stream.json (ground truth: label,
clip start, estimated speech onset and offset, in seconds).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kws.data import (  # noqa: E402
    ROOT,
    SPLITS,
    load_config,
    load_noise_regions,
    pad_or_trim,
    read_wav_int16,
    speech_bounds,
)

STREAM_DIR = ROOT / "data" / "stream"


def build_stream(cfg: dict, split: str) -> tuple[np.ndarray, list[dict]]:
    scfg = cfg["stream"]
    sr = cfg["data"]["sample_rate"]
    clip_samples = cfg["data"]["clip_samples"]
    keywords = cfg["data"]["keywords"]
    unknown = cfg["data"]["unknown_label"]
    rng = random.Random(f"{cfg['seed']}-{split}")
    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    index = json.loads((ROOT / cfg["paths"]["features_dir"] / f"{split}_index.json").read_text())

    by_label = {}
    for item in index:
        if item["path"] is not None:
            by_label.setdefault(item["label"], []).append(item)
    events = [
        it for kw in keywords for it in rng.sample(by_label[kw], scfg["clips_per_keyword"])
    ] + rng.sample(by_label[unknown], scfg["num_unknown"])
    events += [{"path": None, "label": "_gap_"}] * scfg["num_noise_gaps"]
    rng.shuffle(events)

    pieces, truth, cursor = [], [], 0
    for event in events:
        gap = int(rng.uniform(*scfg["gap_s"]) * sr)
        pieces.append(np.zeros(gap, np.float64))
        cursor += gap
        if event["path"] is None:
            pieces.append(np.zeros(clip_samples, np.float64))  # Noise bed only.
        else:
            clip = pad_or_trim(read_wav_int16(dataset_dir / event["path"]), clip_samples)
            onset, offset = speech_bounds(clip, sr, scfg["onset_rel_threshold"])
            truth.append(
                {
                    "label": event["label"],
                    "path": event["path"],
                    "clip_start_s": round(cursor / sr, 4),
                    "onset_s": round(cursor / sr + onset, 4),
                    "offset_s": round(cursor / sr + offset, 4),
                }
            )
            pieces.append(clip.astype(np.float64))
        cursor += clip_samples
    tail = int(scfg["gap_s"][1] * sr)
    pieces.append(np.zeros(tail, np.float64))
    speech = np.concatenate(pieces)

    # Continuous noise bed from this split's noise region, crossfaded crop by crop.
    regions = load_noise_regions(dataset_dir, cfg)[split]
    noise = np.zeros_like(speech)
    pos = 0
    while pos < len(noise):
        region = rng.choice(regions)
        length = min(region.end - region.start, len(noise) - pos)
        offset = rng.randint(region.start, region.end - length)
        noise[pos : pos + length] = region.audio[offset : offset + length]
        pos += length
    mixed = speech + scfg["noise_gain"] * noise
    return np.clip(np.round(mixed), -32768, 32767).astype(np.int16), truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS[1:], default="testing")
    args = parser.parse_args()
    cfg = load_config()
    audio, truth = build_stream(cfg, args.split)
    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = STREAM_DIR / f"{args.split}_stream.wav"
    sf.write(wav_path, audio, cfg["data"]["sample_rate"], subtype="PCM_16")
    meta = {
        "split": args.split,
        "duration_s": len(audio) / cfg["data"]["sample_rate"],
        "num_keywords": sum(t["label"] in cfg["data"]["keywords"] for t in truth),
        "num_unknown": sum(t["label"] == cfg["data"]["unknown_label"] for t in truth),
        "events": truth,
    }
    (STREAM_DIR / f"{args.split}_stream.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{wav_path.relative_to(ROOT)}: {meta['duration_s']:.1f} s, "
          f"{meta['num_keywords']} keywords, {meta['num_unknown']} unknown")  # fmt: skip


if __name__ == "__main__":
    main()
