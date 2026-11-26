from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """One image and its class label, represented by the class directory name."""

    path: Path
    label: str
    source: str


class DataLoader(ABC):
    """Interface for dataset-specific image loading."""

    @abstractmethod
    def load(self, split: str) -> list[ImageRecord]:
        """Return records for a dataset split."""


class LandmarkDataLoader(DataLoader):
    """Load the architecture-landmark dataset described in the assignment.

    Expected layout::

        dataset_root/
            trainval/
                NNSUDataset/<class_name>/*.{jpg,png,...}
                ExtDataset/<class_name>/*.{jpg,png,...}
            test/
                NNSUDataset/<class_name>/*.{jpg,png,...}
                ExtDataset/<class_name>/*.{jpg,png,...}

    Both data sources are included by default. ``00_Other`` is kept as a
    separate class unless ``include_other=False`` is specified.
    """

    SOURCES = ("NNSUDataset", "ExtDataset")

    def __init__(
        self,
        root: str | Path,
        include_external: bool = True,
        include_other: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.include_external = include_external
        self.include_other = include_other

    def load(self, split: str) -> list[ImageRecord]:
        split_aliases = {"train": "trainval", "training": "trainval", "val": "trainval", "validation": "trainval"}
        split_dir_name = split_aliases.get(split.lower(), split.lower())
        if split_dir_name not in {"trainval", "test"}:
            raise ValueError("split must be one of: trainval, test")

        split_dir = self.root / split_dir_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory was not found: {split_dir}\n"
                "Pass --dataset pointing to the directory that contains 'trainval' and 'test'."
            )

        sources = ["NNSUDataset"]
        if self.include_external:
            sources.append("ExtDataset")

        records: list[ImageRecord] = []
        found_source_dir = False
        for source in sources:
            source_dir = split_dir / source
            if not source_dir.is_dir():
                continue
            found_source_dir = True
            for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
                if not self.include_other and class_dir.name == "00_Other":
                    continue
                for image_path in sorted(class_dir.rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        records.append(ImageRecord(image_path, class_dir.name, source))

        # A small fallback makes the loader useful for a split arranged directly
        # as split/<class_name>/<image>, without changing the required layout.
        if not found_source_dir:
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                if not self.include_other and class_dir.name == "00_Other":
                    continue
                for image_path in sorted(class_dir.rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        records.append(ImageRecord(image_path, class_dir.name, "root"))

        if not records:
            raise FileNotFoundError(
                f"No supported images found in '{split_dir}'. Checked sources: {', '.join(sources)}."
            )
        return records
