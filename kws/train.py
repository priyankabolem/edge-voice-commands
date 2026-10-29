"""Train the DS-CNN with on-the-fly waveform augmentation.

Training clips are augmented as waveforms, then passed through the NumPy reference
feature extractor, so the model never sees features computed any other way.
Writes the best checkpoint (by validation accuracy) and results/train_log.json.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from kws.data import (
    ROOT,
    Augmenter,
    int16_to_float,
    keyword_shift,
    label_names,
    load_config,
    load_noise_regions,
    speech_bounds,
)
from kws.features import FeatureConfig, LogMelExtractor, normalize
from kws.model import DSCNN, count_params


class EpochShuffleSampler(Sampler):
    """Shuffles each epoch with a fixed seed and yields (epoch, index) pairs.

    Passing the epoch through the sampler lets persistent DataLoader workers seed
    augmentation per epoch without being restarted.
    """

    def __init__(self, num_examples: int, seed: int):
        self.num_examples = num_examples
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_examples

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        for i in torch.randperm(self.num_examples, generator=gen).tolist():
            yield self.epoch, i


class AugmentedTrainSet(Dataset):
    """Training waveforms -> augmented, normalized log-mel features.

    Each example's randomness is seeded from (seed, epoch, index), so results do not
    depend on the number of DataLoader workers.
    """

    def __init__(self, cfg: dict, mean: np.ndarray, std: np.ndarray):
        self.cfg = cfg
        self.features_dir = ROOT / cfg["paths"]["features_dir"]
        self.labels = np.load(self.features_dir / "training_labels.npy")
        self.mean, self.std = mean, std
        self.seed = cfg["seed"]
        self.extractor = LogMelExtractor(
            FeatureConfig.from_dict(cfg["features"], cfg["data"]["sample_rate"])
        )
        # Opened lazily in each worker so large arrays are not pickled to subprocesses.
        self.waves = None
        self.augmenter = None

        # Streaming-aware training (config variant): per-clip speech bounds for keyword
        # clips, used for constrained time shifts and word-tail negatives.
        self.aug = cfg["augmentation"]
        self.bounds = None
        if self.aug.get("tail_negative_prob", 0) > 0:
            names = label_names(cfg)
            self.is_keyword = [n in cfg["data"]["keywords"] for n in names]
            self.unknown_id = names.index(cfg["data"]["unknown_label"])
            self.bounds = self._keyword_bounds(cfg)

    def __len__(self) -> int:
        return len(self.labels)

    def _keyword_bounds(self, cfg: dict) -> np.ndarray:
        """(n, 2) speech onset/offset in samples for keyword clips; (0, 0) otherwise."""
        sr = cfg["data"]["sample_rate"]
        waves = np.load(self.features_dir / "training_waves.npy", mmap_mode="r")
        bounds = np.zeros((len(self.labels), 2), dtype=np.int64)
        for i, label in enumerate(self.labels):
            if self.is_keyword[label]:
                b = speech_bounds(waves[i], sr, self.aug["onset_rel_threshold"])
                if b is not None:
                    bounds[i] = round(b[0] * sr), round(b[1] * sr)
        return bounds

    def __getitem__(self, key: tuple[int, int]):
        epoch, i = key
        if self.waves is None:
            self.waves = np.load(self.features_dir / "training_waves.npy", mmap_mode="r")
            noise = load_noise_regions(ROOT / self.cfg["paths"]["dataset_dir"], self.cfg)
            self.augmenter = Augmenter(noise["training"], self.cfg)
        rng = random.Random(f"{self.seed}-{epoch}-{i}")
        label, shift = int(self.labels[i]), None
        if self.bounds is not None and self.is_keyword[label]:
            onset, offset = self.bounds[i]
            if offset > onset:
                shift, is_tail = keyword_shift(
                    int(onset), int(offset), rng, self.aug,
                    self.cfg["data"]["sample_rate"], len(self.waves[i]),
                )  # fmt: skip
                if is_tail:
                    label = self.unknown_id
        audio = self.augmenter(int16_to_float(self.waves[i]), rng, shift)
        feats = normalize(self.extractor(audio), self.mean, self.std)
        return torch.from_numpy(feats).unsqueeze(0), label


def load_eval_split(cfg: dict, split: str, mean: np.ndarray, std: np.ndarray):
    """Clean precomputed features for validation/testing as (N, 1, 98, 40) tensors."""
    features_dir = ROOT / cfg["paths"]["features_dir"]
    feats = normalize(np.load(features_dir / f"{split}_logmel.npy"), mean, std)
    labels = np.load(features_dir / f"{split}_labels.npy")
    return torch.from_numpy(feats).unsqueeze(1), torch.from_numpy(labels)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, feats: torch.Tensor, device, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    logits = [
        model(feats[i : i + batch_size].to(device)).cpu() for i in range(0, len(feats), batch_size)
    ]
    return torch.cat(logits)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="override train.epochs from the config")
    parser.add_argument("--variant", help="config variant, e.g. streaming_model")
    args = parser.parse_args()

    cfg = load_config(variant=args.variant)
    tcfg = cfg["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    seed_everything(cfg["seed"])
    device = pick_device()
    labels = label_names(cfg)

    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    mean, std = stats["mean"], stats["std"]
    train_set = AugmentedTrainSet(cfg, mean, std)
    val_x, val_y = load_eval_split(cfg, "validation", mean, std)

    model = DSCNN.from_config(cfg, len(labels)).to(device)
    num_params = count_params(model)
    if num_params > cfg["model"]["max_params"]:
        raise ValueError(f"{num_params} parameters exceeds budget {cfg['model']['max_params']}")
    print(f"DS-CNN parameters: {num_params}, device: {device}")

    sampler = EpochShuffleSampler(len(train_set), cfg["seed"])
    loader = DataLoader(
        train_set,
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        num_workers=tcfg["num_workers"],
        persistent_workers=tcfg["num_workers"] > 0,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"] * len(loader)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])

    checkpoint_path = ROOT / tcfg["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_val_acc, start = [], -1.0, time.perf_counter()
    for epoch in range(tcfg["epochs"]):
        sampler.epoch = epoch
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_acc = (predict(model, val_x, device).argmax(1) == val_y).float().mean().item()
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / seen,
            "train_acc": correct / seen,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(time.perf_counter() - start, 1),
        }
        history.append(record)
        print(
            f"epoch {record['epoch']:2d}  loss {record['train_loss']:.4f}  "
            f"train {record['train_acc']:.4f}  val {val_acc:.4f}  ({record['elapsed_s']} s)"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "labels": labels,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

    log = {
        "device": str(device),
        "num_params": num_params,
        "best_epoch": max(history, key=lambda r: r["val_acc"])["epoch"],
        "best_val_acc": best_val_acc,
        "train_seconds": round(time.perf_counter() - start, 1),
        "history": history,
    }
    (ROOT / cfg["paths"]["results_dir"] / "train_log.json").write_text(
        json.dumps(log, indent=2) + "\n"
    )
    print(f"Best validation accuracy {best_val_acc:.4f}; saved {checkpoint_path.name}")


if __name__ == "__main__":
    main()
