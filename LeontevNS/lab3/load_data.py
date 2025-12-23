import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def load_images_from_split(file_path, mode=None, img_size=(224, 224), max_images=None):
    images = []
    labels = []
    label_to_name = {0: "01_NizhnyNovgorodKremlin", 1: "04_ArkhangelskCathedral", 2: "08_PalaceOfLabor"}
    name_to_label = {"01_NizhnyNovgorodKremlin": 0, "04_ArkhangelskCathedral": 1, "08_PalaceOfLabor": 2}
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_images:
        lines = lines[:max_images]
    
    print(f"Загрузка изображений из {file_path}...")
    
    for line in tqdm(lines, desc="Обработка путей"):
        path = line.strip()
        if not path:
            continue
        
        if not path.startswith('data'):
            path = os.path.join('data', path)
        
        class_name = os.path.basename(os.path.dirname(path))
        
        if class_name not in name_to_label:
            label = len(name_to_label)
            name_to_label[class_name] = label
            label_to_name[label] = class_name
        
        img = cv2.imread(path)
        if img is None:
            print(f"Предупреждение: не удалось загрузить изображение {path}")
            continue
        
        if mode == "NN":
            img = cv2.resize(img, img_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img)
        labels.append(name_to_label[class_name])
    
    print(f"Загружено {len(images)} изображений, {len(name_to_label)} классов")
    for label, name in label_to_name.items():
        count = labels.count(label)
        print(f"  Класс {name} (метка {label}): {count} изображений")
    
    return images, np.array(labels), label_to_name
