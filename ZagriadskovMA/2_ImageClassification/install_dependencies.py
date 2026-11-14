"""Install CUDA PyTorch first, falling back to CPU if pip fails."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(pip + ["-r", str(root / "requirements-base.txt")], check=True)
    result = subprocess.run(pip + ["--upgrade", "torch==2.12.0+cu132", "torchvision==0.27.0+cu132",
                                   "--index-url", "https://download.pytorch.org/whl/cu132"])
    if result.returncode:
        print("CUDA installation failed; installing CPU builds", flush=True)
        subprocess.run(pip + ["--upgrade", "torch==2.12.0+cpu", "torchvision==0.27.0+cpu",
                              "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)


if __name__ == "__main__":
    main()
