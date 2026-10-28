"""Download and extract Google Speech Commands v0.02 into data/ (checksum-verified)."""

from pathlib import Path

from torchaudio.datasets import SPEECHCOMMANDS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = SPEECHCOMMANDS(str(DATA_DIR), url="speech_commands_v0.02", download=True)
    print(f"Speech Commands v0.02 ready at {dataset._path} ({len(dataset)} utterances)")


if __name__ == "__main__":
    main()
