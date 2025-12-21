import os
import copy
import time
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from abstract import ClassificationStrategy


# ============================================================================
# Реализация стратегии на основе нейронной сети (Transfer Learning).
# Используется предобученная модель ResNet18.
#
# Ключевые моменты для защиты:
# 1. Transfer Learning (Перенос обучения): Мы берем веса модели, обученной на
#    огромном датасете ImageNet, и дообучаем только последние слои под нашу задачу.
#    Это позволяет получить высокую точность на маленьком наборе данных.
# 2. Аугментация данных: Искусственное расширение обучающей выборки (повороты,
#    отражения) для борьбы с переобучением.
# 3. Тензоры: Основная структура данных в PyTorch (многомерные матрицы),
#    которые могут обрабатываться на GPU.
# ============================================================================

class LandmarksDataset(Dataset):
    """Кастомный класс датасета для PyTorch"""

    def __init__(self, image_paths: List[str], labels: List[str], class_to_idx: Dict[str, int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]

        # Конвертируем строковую метку в число (0, 1, 2)
        label = self.class_to_idx[label_str]

        # Загружаем изображение и конвертируем в RGB (на случай если попался PNG с альфа-каналом)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка чтения {img_path}: {e}")
            # Возвращаем черную картинку в случае ошибки, чтобы не крашить обучение
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


class NeuralNetworkStrategy(ClassificationStrategy):
    """Стратегия классификации на основе нейронной сети ResNet18"""

    def __init__(self, model_architecture: str = 'resnet', epochs: int = 10, batch_size: int = 16):
        """
        Args:
            model_architecture: Архитектура (по умолчанию resnet18)
            epochs: Количество проходов по всему датасету
            batch_size: Размер пакета данных, обрабатываемого за раз
        """
        self.model_architecture = model_architecture
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство для вычислений: {self.device}")

        self.model = None
        self.class_names = []  # Список имен классов ['cathedral', 'kremlin', ...]
        self.class_to_idx = {}  # Словарь {'cathedral': 0, ...}

        # Трансформации для подготовки изображений под формат ResNet
        # ResNet обучался на картинках 224x224 с определенной нормализацией
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),  # Случайная обрезка (аугментация)
                transforms.RandomHorizontalFlip(),  # Случайное отражение (аугментация)
                transforms.ToTensor(),  # Перевод пикселей (0-255) в тензор (0.0-1.0)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # Нормализация по статистике ImageNet
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def _initialize_model(self, num_classes: int):
        """Инициализация предобученной модели"""
        print("Загрузка предобученной модели ResNet18...")

        # Загружаем веса, обученные на ImageNet. 
        # weights='DEFAULT' автоматически выберет лучшие доступные веса.
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            # Для старых версий torchvison
            self.model = models.resnet18(pretrained=True)

        # ЗАМОРОЗКА ВЕСОВ (Feature Extraction):
        # Мы говорим "не обучай эти слои", чтобы сохранить способность сети
        # выделять базовые признаки (границы, текстуры), полученные на ImageNet.
        # Это ускоряет обучение и требует меньше данных.
        for param in self.model.parameters():
            param.requires_grad = False

        # Заменяем последний полносвязный слой (fc).
        # Исходный слой классифицировал на 1000 классов ImageNet.
        # Наш новый слой будет иметь выходы только для наших 3 классов.
        # У нового слоя requires_grad=True по умолчанию.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model = self.model.to(self.device)

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучение нейросети"""
        # 1. Подготовка меток классов
        self.class_names = sorted(list(set(train_labels)))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        print(f"Классы для обучения: {self.class_to_idx}")

        # 2. Инициализация модели
        self._initialize_model(len(self.class_names))

        # 3. Создание датасета и загрузчика данных
        dataset = LandmarksDataset(train_data, train_labels, self.class_to_idx, self.data_transforms['train'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # 4. Определение функции потерь и оптимизатора
        # CrossEntropyLoss - стандарт для классификации.
        criterion = nn.CrossEntropyLoss()

        # SGD (Стохастический градиентный спуск) будет обновлять только веса последнего слоя (model.fc),
        # так как остальные заморожены.
        optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)

        print("-" * 20)
        print(f"Начало обучения на {self.epochs} эпох...")

        # 5. Цикл обучения
        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()  # Перевод модели в режим обучения (важно для Dropout/BatchNorm)
            running_loss = 0.0
            running_corrects = 0

            # Проход по батчам (пакетам картинок)
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Обнуляем градиенты (накопленные с предыдущего шага)
                optimizer.zero_grad()

                # Forward pass (Прямое распространение): картинка -> сеть -> предсказание
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)  # Получаем индексы классов с макс. вероятностью
                    loss = criterion(outputs, labels)  # Считаем ошибку

                    # Backward pass (Обратное распространение): считаем градиенты ошибки по весам
                    loss.backward()
                    # Optimizer step: обновляем веса (w = w - learning_rate * gradient)
                    optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)

            print(f'Epoch {epoch + 1}/{self.epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        time_elapsed = time.time() - start_time
        print(f'Обучение завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с')

    def predict(self, image_path: str) -> str:
        """Предсказание класса для одного изображения"""
        if self.model is None:
            raise RuntimeError("Модель не обучена и не загружена!")

        self.model.eval()  # Режим оценки (выключает dropout)

        # Загрузка и препроцессинг
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.data_transforms['val'](img)

        # Добавляем размерность батча (было [3, 224, 224], стало [1, 3, 224, 224])
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():  # Отключаем расчет градиентов для экономии памяти
            outputs = self.model(img_tensor)
            _, preds = torch.max(outputs, 1)
            class_idx = preds.item()

        return self.class_names[class_idx]

    def save(self, filepath: str) -> None:
        """Сохранение весов модели и метаданных"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Сохраняем не весь объект модели, а state_dict (словарь весов),
        # это считается best practice в PyTorch.
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'model_architecture': self.model_architecture
        }

        torch.save(checkpoint, filepath)
        print(f"Модель сохранена: {filepath}")

    def load(self, filepath: str) -> None:
        """Загрузка модели"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")

        # map_location='cpu' позволяет загрузить модель, сохраненную на GPU, на обычный CPU
        checkpoint = torch.load(filepath, map_location=self.device)

        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint['class_to_idx']
        self.model_architecture = checkpoint['model_architecture']

        # Снова создаем "каркас" модели и вставляем туда веса
        self._initialize_model(len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Нейросеть загружена из {filepath}")

    def get_params(self) -> Dict[str, Any]:
        return {
            'algorithm': 'neural',
            'architecture': self.model_architecture,
            'epochs': self.epochs,
            'device': str(self.device)
        }
