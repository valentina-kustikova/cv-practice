"""Скачивает веса ResNet18 и MobileNetV2 для работы с фотографиями."""
import argparse
from pathlib import Path
from torch.hub import download_url_to_file
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, default=Path('weights'))
    args = parser.parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    for weights in (ResNet18_Weights.IMAGENET1K_V1, MobileNet_V2_Weights.IMAGENET1K_V1):
        name = Path(weights.url).name
        target = args.weights_dir / name
        if target.exists():
            print(f'Already exists: {target}')
            continue
        download_url_to_file(weights.url, str(target), hash_prefix=name.rsplit('-', 1)[1].split('.')[0])
        print(target)
