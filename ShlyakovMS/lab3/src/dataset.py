# src/dataset.py
import os
import cv2
import numpy as np
from tqdm import tqdm

def load_split_file(split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def load_images_from_list(image_list, data_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    # 0=kremlin, 1=dvorec, 2=sobor
    folder_to_label = {
        '01_NizhnyNovgorodKremlin': 0,
        '08_PalaceOfLabor': 1,
        '04_ArkhangelskCathedral': 2
    }
    
    data_path = os.path.abspath(data_dir)
    
    for item in tqdm(image_list, desc="Loading images"):
        line = item.strip()
        if not line:
            continue
            
        parts = line.split()
        img_path = parts[0].replace('\\', '/')
        
        # Определяем метку по папке
        label = None
        for folder, idx in folder_to_label.items():
            if folder in img_path:
                label = idx
                break
                
        if label is None:
            print(f"Неизвестный класс: {img_path}")
            continue
            
        full_path = os.path.join(data_path, img_path)
        if not os.path.exists(full_path):
            print(f"Нет файла: {full_path}")
            continue
            
        img = cv2.imread(full_path)
        if img is None:
            print(f"Не читается: {full_path}")
            continue
            
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img)
        labels.append(label)
        
    print(f"Загружено: {len(images)} изображений")
    return np.array(images), np.array(labels)