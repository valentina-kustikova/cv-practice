# create_test_split.py
import os
from pathlib import Path

def create_test_split(data_dir="data"):
    data_path = Path(data_dir)
    
    # 1. Читаем train.txt
    train_file = data_path / "train.txt"
    if not train_file.exists():
        print("train.txt не найден!")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = f.read().strip().split('\n')
    
    train_paths = set()
    for line in train_lines:
        if line.strip():
            parts = line.strip().split()
            img_path = parts[0]  # только путь, без метки
            train_paths.add(img_path.replace('\\', '/'))  # на всякий случай
    
    print(f"В train.txt: {len(train_paths)} изображений")

    # 2. Собираем ВСЕ изображения из папок
    all_images = []
    class_map = {
        '01_NizhnyNovgorodKremlin': 'kremlin',
        '04_ArkhangelskCathedral': 'sobor', 
        '08_PalaceOfLabor': 'dvorec'
    }

    for class_folder in class_map.keys():
        for root in [data_path / "NNSUDataset" / class_folder, 
                     data_path / "ExtDataset" / class_folder]:
            if not root.exists():
                continue
            for file in root.rglob("*.*"):
                if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                    rel_path = file.relative_to(data_path).as_posix()
                    rel_path = rel_path.replace('\\', '/')
                    all_images.append((rel_path, class_map[class_folder]))

    print(f"Всего найдено изображений: {len(all_images)}")

    # 3. Формируем test.txt
    test_lines = []
    for img_path, label in all_images:
        if img_path not in train_paths:
            test_lines.append(f"{img_path} {label}")

    # 4. Сохраняем test.txt
    test_file = data_path / "test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    
    print(f"Создан test.txt: {len(test_lines)} изображений")
    print(f"   → {test_file}")

    # Статистика
    print("\nРаспределение по классам в test:")
    from collections import Counter
    labels = [line.split()[-1] for line in test_lines]
    print(Counter(labels))

if __name__ == "__main__":
    create_test_split()