import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class DataLoader:
    
    def __init__(self, data_dir: str, train_file: str = None):
        self.data_dir = Path(data_dir)
        self.train_file = train_file
        self.class_names = {
            '01_NizhnyNovgorodKremlin': 'Нижегородский Кремль',
            '04_ArkhangelskCathedral': 'Архангельский собор',
            '08_PalaceOfLabor': 'Дворец труда'
        }
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(self.class_names.keys()))}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
    
    def load_train_images(self, max_size: Tuple[int, int] = (800, 800)) -> Tuple[List[np.ndarray], List[int]]:
        if not self.train_file or not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Файл {self.train_file} не найден")
        
        images = []
        labels = []
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_paths = [line.strip() for line in f.readlines()]
        
        print(f"Загрузка {len(train_paths)} изображений...")
        
        for i, img_path in enumerate(train_paths):
            normalized_path = img_path.replace('\\', '/')
            full_path = self.data_dir / normalized_path
            if not full_path.exists():
                full_path = self.data_dir / img_path
                if not full_path.exists():
                    continue
            
            class_name = None
            for key in self.class_names.keys():
                if key in str(full_path):
                    class_name = key
                    break
            
            if class_name is None:
                continue
            
            try:
                img = cv2.imread(str(full_path))
                if img is not None and img.size > 0:
                    h, w = img.shape[:2]
                    if h > max_size[1] or w > max_size[0]:
                        scale = min(max_size[0] / w, max_size[1] / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    images.append(img)
                    labels.append(self.class_to_idx[class_name])
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Загружено {i + 1}/{len(train_paths)} изображений...")
            except Exception as e:
                print(f"Ошибка при загрузке изображения {full_path}: {e}")
                continue
        
        return images, labels
    
    def load_test_images(self, max_size: Tuple[int, int] = (800, 800)) -> Tuple[List[np.ndarray], List[int], List[str]]:
        if not self.train_file or not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Файл {self.train_file} не найден")
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_paths_raw = [line.strip() for line in f.readlines()]
            train_paths = set()
            for path in train_paths_raw:
                normalized = path.replace('\\', '/')
                train_paths.add(normalized)
        
        images = []
        labels = []
        paths = []
        
        for class_name in self.class_names.keys():
            class_dir = None
            for subdir in ['ExtDataset', 'NNSUDataset']:
                potential_dir = self.data_dir / subdir / class_name
                if potential_dir.exists():
                    class_dir = potential_dir
                    break
            
            if class_dir is None:
                continue
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                rel_path = str(img_file.relative_to(self.data_dir))
                rel_path = rel_path.replace('\\', '/')
                
                if rel_path in train_paths:
                    continue
                
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None and img.size > 0:
                        h, w = img.shape[:2]
                        if h > max_size[1] or w > max_size[0]:
                            scale = min(max_size[0] / w, max_size[1] / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        images.append(img)
                        labels.append(self.class_to_idx[class_name])
                        paths.append(str(img_file))
                except Exception as e:
                    print(f"Ошибка при загрузке изображения {img_file}: {e}")
                    continue
        
        return images, labels, paths
    
    def get_class_name(self, idx: int) -> str:
        return self.class_names.get(self.idx_to_class.get(idx, ''), 'Unknown')
    
    def get_num_classes(self) -> int:
        return len(self.class_names)
