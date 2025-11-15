import os
from pathlib import Path
from typing import List, Tuple


def parse_class_from_path(path: str) -> int:
    parts = path.replace("/", "\\").split("\\")
    for part in parts:
        if part.startswith("01_"):
            return 0  # Кремль
        elif part.startswith("04_"):
            return 1  # Архангельский собор
        elif part.startswith("08_"):
            return 2  # Дворец труда


def load_split_lists(train_file: str, data_root: str) -> Tuple[List[str], List[str], List[int], List[int]]:
    with open(train_file, 'r', encoding='utf-8') as f:
        train_rel_paths = [line.strip() for line in f if line.strip()]

    train_abs_paths, train_labels = [], []
    for rel_path in train_rel_paths:
        abs_path = os.path.join(data_root, rel_path)
        train_abs_paths.append(abs_path)
        train_labels.append(parse_class_from_path(rel_path))

    all_images = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                abs_path = os.path.join(root, f)
                all_images.append(abs_path)

    train_set = set(train_abs_paths)
    test_abs_paths = [p for p in all_images if p not in train_set]
    test_labels = [parse_class_from_path(os.path.relpath(p, data_root)) for p in test_abs_paths]

    print(f"Загружено: {len(train_abs_paths)} train, {len(test_abs_paths)} test изображений")
    class_names = ["NizhnyNovgorodKremlin", "ArkhangelskCathedral", "PalaceOfLabor"]
    print("Распределение по классам (train):")
    for cls_id, name in enumerate(class_names):
        count = train_labels.count(cls_id)
        print(f"  {name}: {count}")

    return train_abs_paths, test_abs_paths, train_labels, test_labels