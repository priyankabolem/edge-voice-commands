"""Stage 0: record the machine and software environment into results/environment.json."""

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "environment.json"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model():
    if sys.platform == "darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def core_counts():
    counts = {"logical": os.cpu_count()}
    if sys.platform == "darwin":
        for key, name in [
            ("hw.physicalcpu", "physical"),
            ("hw.perflevel0.physicalcpu", "performance"),
            ("hw.perflevel1.physicalcpu", "efficiency"),
        ]:
            value = run(["sysctl", "-n", key])
            counts[name] = int(value) if value else None
    return counts


def first_line(cmd):
    out = run(cmd)
    return out.splitlines()[0] if out else None


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    env = {
        "os": platform.platform(),
        "os_version": run(["sw_vers", "-productVersion"])
        if sys.platform == "darwin"
        else platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_cores": core_counts(),
        "python": platform.python_version(),
        "packages": {
            name: pkg_version(name)
            for name in [
                "torch",
                "torchaudio",
                "torchao",
                "executorch",
                "numpy",
                "pyyaml",
                "pytest",
                "soundfile",
            ]
        },
        "c_compiler": first_line(["cc", "--version"]),
        "cxx_compiler": first_line(["c++", "--version"]),
        "cmake": first_line(["cmake", "--version"]) if shutil.which("cmake") else None,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
