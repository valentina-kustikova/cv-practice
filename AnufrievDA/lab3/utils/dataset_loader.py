import os
import random
from typing import List, Tuple

def parse_class_from_path(path: str) -> int:
    # Приводим слеши к системному виду
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith("01_"):
            return 0
        elif part.startswith("04_"):
            return 1
        elif part.startswith("08_"):
            return 2
    return -1

def load_split_by_ratio(train_list_file: str, data_root: str, train_ratio: float = 0.8) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Читает список файлов, ПЕРЕМЕШИВАЕТ ЕГО, определяет классы и делит на train/test.
    """
    if not os.path.exists(train_list_file):
        raise FileNotFoundError(f"Файл списка не найден: {train_list_file}")

    with open(train_list_file, 'r', encoding='utf-8') as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    abs_paths = []
    labels = []
    
    print("Парсинг классов...")
    for rel_path in rel_paths:
        abs_path = os.path.join(data_root, rel_path)
        if os.path.exists(abs_path):
            lbl = parse_class_from_path(rel_path)
            if lbl != -1:
                abs_paths.append(abs_path)
                labels.append(lbl)
        else:
            # print(f"Warning: File not found: {abs_path}")
            pass

    n_total = len(abs_paths)
    if n_total == 0:
        raise ValueError("Не найдено ни одного изображения! Проверьте пути в train.txt и data_dir.")

    # --- ВАЖНОЕ ИЗМЕНЕНИЕ: ПЕРЕМЕШИВАЕМ ДАННЫЕ ---
    # Используем фиксированный seed, чтобы при каждом запуске разбиение было одинаковым
    # (чтобы можно было сравнивать разные алгоритмы на одних и тех же данных)
    combined = list(zip(abs_paths, labels))
    random.seed(42) 
    random.shuffle(combined)
    abs_paths[:], labels[:] = zip(*combined)
    # ---------------------------------------------

    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_train, n_total - 1))

    train_paths = abs_paths[:n_train]
    train_labels = labels[:n_train]
    test_paths = abs_paths[n_train:]
    test_labels = labels[n_train:]

    print(f"Всего валидных изображений: {n_total}")
    print(f"Train: {len(train_paths)} ({len(train_paths)/n_total:.1%})")
    print(f"Test:  {len(test_paths)}  ({len(test_paths)/n_total:.1%})")

    class_names = ["NizhnyNovgorodKremlin", "ArkhangelskCathedral", "PalaceOfLabor"]
    print("Распределение по классам (train):")
    for cls_id, name in enumerate(class_names):
        count = train_labels.count(cls_id)
        print(f"  {name}: {count}")
        
    print("Распределение по классам (test):")
    for cls_id, name in enumerate(class_names):
        count = test_labels.count(cls_id)
        print(f"  {name}: {count}")

    return train_paths, test_paths, train_labels, test_labels