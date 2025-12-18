import os
import cv2
import numpy as np
from typing import List, Tuple

def parse_class_from_path(path: str) -> int:
    """
    Определяет метку класса по имени файла/папки.
    01_ -> 0 (Кремль)
    04_ -> 1 (Собор)
    08_ -> 2 (Дворец)
    """
    # Нормализуем путь и разбиваем на части
    parts = os.path.normpath(path).split(os.sep)
    
    # Проверяем имя файла и родительские папки
    for part in parts:
        if part.startswith("01_"):
            return 0
        if part.startswith("04_"):
            return 1
        if part.startswith("08_"):
            return 2
            
    return -1

def load_split_lists(train_file: str, data_root: str) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Загружает пути к файлам и метки.
    train_file: путь к txt файлу со списком тренировочных фото (относительные пути)
    data_root: корневая папка с датасетом
    """
    train_paths = []
    train_labels = []
    
    # 1. Читаем список тренировочных файлов
    if train_file and os.path.exists(train_file):
        with open(train_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                rel_path = line.strip()
                if not rel_path: continue
                
                full_path = os.path.join(data_root, rel_path)
                # Проверяем, существует ли файл
                if os.path.exists(full_path):
                    lbl = parse_class_from_path(rel_path)
                    if lbl != -1:
                        train_paths.append(full_path)
                        train_labels.append(lbl)
                else:
                    # Попробуем найти файл, если в train.txt пути с другими слешами
                    # или если путь указан не совсем точно
                    pass 
    
    # 2. Сканируем ВСЕ файлы в data_root
    all_paths = []
    all_labels = []
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                lbl = parse_class_from_path(file) 
                if lbl == -1:
                    lbl = parse_class_from_path(root)
                
                if lbl != -1:
                    all_paths.append(full_path)
                    all_labels.append(lbl)

    # 3. Формируем тестовую выборку (Все - Тренировочные)
    # Используем set для быстрого поиска (нормализуем пути)
    train_set = set(os.path.normpath(p) for p in train_paths)
    
    test_paths = []
    test_labels = []
    
    for p, l in zip(all_paths, all_labels):
        if os.path.normpath(p) not in train_set:
            test_paths.append(p)
            test_labels.append(l)
            
    print(f"Data loaded. Train: {len(train_paths)}, Test: {len(test_paths)}")
    return train_paths, test_paths, train_labels, test_labels