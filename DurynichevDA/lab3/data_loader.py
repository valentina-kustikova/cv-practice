from pathlib import Path

def load_dataset(data_dir: str, train_txt: str):
    data_dir = Path(data_dir)
    train_paths = set()

    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().replace('\\', '/')
            if p:
                train_paths.add(p)

    class_map = {
        '01_NizhnyNovgorodKremlin': 0,
        '04_ArkhangelskCathedral': 1,
        '08_PalaceOfLabor': 2
    }
    class_names = ['Нижегородский Кремль', 'Архангельский собор', 'Дворец труда']

    train_data = []
    test_data = []

    for class_name, label in class_map.items():
        for source in ['ExtDataset', 'NNSUDataset']:
            folder = data_dir / source / class_name
            if folder.exists():
                # ИСПРАВЛЕНИЕ: list() для генераторов
                for img_path in list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png')):
                    rel_path = img_path.relative_to(data_dir).as_posix()
                    if rel_path in train_paths:
                        train_data.append((str(img_path), label))
                    else:
                        test_data.append((str(img_path), label))

    print(f"Загружено: {len(train_data)} train | {len(test_data)} test")
    return train_data, test_data, class_names