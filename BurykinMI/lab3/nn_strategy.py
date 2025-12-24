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
        # Вызывается DataLoader'ом для получения одной картинки
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]

        label = self.class_to_idx[label_str]

        try:
            # .convert('RGB') критически важен:
            # 1. Если картинка Ч/Б (1 канал) -> делает 3 канала (R=G=B).
            # 2. Если PNG (4 канала RGBA) -> убирает прозрачность.
            # Сеть ResNet ожидает строго 3 канала на входе.
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка чтения {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            # Применяем аугментацию и нормализацию.
            # На выходе: Тензор размера [3, 224, 224] (Channels, Height, Width)
            image = self.transform(image)

        return image, label


class NeuralNetworkStrategy(ClassificationStrategy):
    """Стратегия классификации на основе нейронной сети ResNet18"""

    def __init__(self, model_architecture: str = 'resnet', epochs: int = 10, batch_size: int = 16):
        self.model_architecture = model_architecture
        self.epochs = epochs
        self.batch_size = batch_size

        # Автоматический выбор GPU (cuda) для ускорения вычислений
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство для вычислений: {self.device}")

        self.model = None
        self.class_names = []
        self.class_to_idx = {}

        # Трансформации данных (Pipeline препроцессинга)
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),  # Приводим к единому размеру
                transforms.RandomCrop(224),  # Аугментация: вырезаем случайную часть (борьба с переобучением)
                transforms.RandomHorizontalFlip(),  # Аугментация: зеркальное отражение
                transforms.ToTensor(),  # PIL Image (0-255) -> Torch Tensor (0.0-1.0)
                # Нормализация (стандартизация): (pixel - mean) / std
                # Используются средние значения ImageNet. Сеть училась на них, нам нужно соответствовать.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def _initialize_model(self, num_classes: int):
        """Инициализация модели и Transfer Learning"""
        print("Загрузка предобученной модели ResNet18...")

        try:
            from torchvision.models import ResNet18_Weights
            # Загружаем веса, обученные на 1.2 млн изображений ImageNet
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            self.model = models.resnet18(pretrained=True)

        # ЗАМОРОЗКА ВЕСОВ (Feature Extractor):
        # Отключаем градиенты для всех слоев.
        # "Туловище" сети останется неизменным, оно уже умеет выделять признаки.
        for param in self.model.parameters():
            param.requires_grad = False

        # ЗАМЕНА ПОСЛЕДНЕГО СЛОЯ (Head):
        # self.model.fc - это полносвязный слой (Fully Connected).
        # В ResNet18 он выдает 1000 классов. Меняем его на слой с num_classes (3) выходами.
        # У нового слоя requires_grad=True по умолчанию -> он БУДЕТ обучаться.
        num_ftrs = self.model.fc.in_features  # Обычно 512 признаков на входе fc
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model = self.model.to(self.device)

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Цикл обучения"""
        self.class_names = sorted(list(set(train_labels)))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        print(f"Классы для обучения: {self.class_to_idx}")

        self._initialize_model(len(self.class_names))

        dataset = LandmarksDataset(train_data, train_labels, self.class_to_idx, self.data_transforms['train'])
        # DataLoader собирает картинки в пакеты (батчи).
        # Размер батча: [16, 3, 224, 224] (Batch, Channels, Height, Width)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Функция потерь CrossEntropyLoss (подходит для многоклассовой классификации)
        criterion = nn.CrossEntropyLoss()

        # Оптимизатор SGD меняет веса только последнего слоя (model.fc), т.к. остальные заморожены
        optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)

        print("-" * 20)
        print(f"Начало обучения на {self.epochs} эпох...")

        for epoch in range(self.epochs):
            self.model.train()  # Включаем режим обучения (важно для BatchNorm и Dropout)
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                # Перенос данных на GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 1. Обнуляем накопленные градиенты
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # 2. Forward pass: прогон картинок через сеть
                    # inputs shape: [16, 3, 224, 224] -> outputs shape: [16, 3]
                    outputs = self.model(inputs)

                    _, preds = torch.max(outputs, 1)  # Индекс класса с макс. вероятностью
                    loss = criterion(outputs, labels)  # Вычисление ошибки

                    # 3. Backward pass: вычисление градиентов ошибки (идет от конца к началу)
                    # Градиент доходит до fc слоя и останавливается (дальше заморожено)
                    loss.backward()

                    # 4. Optimizer step: обновление весов (w = w - learning_rate * grad)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(f'Epoch {epoch + 1}/{self.epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def predict(self, image_path: str) -> str:
        """Инференс (предсказание)"""
        if self.model is None:
            raise RuntimeError("Модель не обучена!")

        self.model.eval()  # Режим оценки (отключает dropout, фиксирует batchnorm)

        img = Image.open(image_path).convert('RGB')
        img_tensor = self.data_transforms['val'](img)

        # Добавляем размерность батча (unsqueeze): [3, 224, 224] -> [1, 3, 224, 224]
        # Сеть всегда ожидает 4D тензор на входе.
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():  # Отключаем расчет градиентов (экономит память и время)
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
