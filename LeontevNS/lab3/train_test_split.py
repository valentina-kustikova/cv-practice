import os
import random
import argparse
from pathlib import Path

def create_train_test_split(data_dir, output_dir, train_ratio=0.8):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / 'train.txt'
    test_file = output_dir / 'test.txt'
    
    all_images = []
    
    for dataset in ['NNSUDataset', 'ExtDataset']:
        dataset_path = data_dir / dataset
        if dataset_path.exists():
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.glob('*.jpg'):
                        rel_path = str(Path(dataset) / class_name / img_file.name)
                        all_images.append((rel_path, class_name))
    
    random.shuffle(all_images)
    
    class_images = {}
    for img_path, class_name in all_images:
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(img_path)
    
    train_paths = []
    test_paths = []
    
    for class_name, images in class_images.items():
        split_idx = int(len(images) * train_ratio)
        train_paths.extend(images[:split_idx])
        test_paths.extend(images[split_idx:])
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for path in train_paths:
            f.write(path + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for path in test_paths:
            f.write(path + '\n')
    
    print(f"Создано разделение:")
    print(f"  Обучающих изображений: {len(train_paths)}")
    print(f"  Тестовых изображений: {len(test_paths)}")
    print(f"  Всего изображений: {len(all_images)}")
    
    print("\nСтатистика по классам:")
    for class_name in class_images.keys():
        train_count = sum(1 for path in train_paths if class_name in path)
        test_count = sum(1 for path in test_paths if class_name in path)
        total = train_count + test_count
        print(f"  {class_name}: {total} изображений ({train_count} train, {test_count} test)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Создание разделения данных')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Путь к директории с данными')
    parser.add_argument('--output_dir', type=str, default='splits',
                       help='Директория для сохранения файлов разделения')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Доля обучающих данных')
    
    args = parser.parse_args()
    
    create_train_test_split(args.data_dir, args.output_dir, args.train_ratio)