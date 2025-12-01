import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Проверка наличия CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


class ImageDataset(Dataset):
    """Кастомный класс датасета для PyTorch."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Открытие и конвертация в RGB
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # Обработка ошибок загрузки изображения
            return torch.zeros(3, 224, 224), 0


class TransferLearningClassifier:
    """
    Классификатор на основе предобученной нейронной сети (Transfer Learning).
    """

    def __init__(self, num_classes: int, model_name: str = 'resnet18'):
        self.num_classes = num_classes
        self.model_name = model_name
        # Путь для сохранения/загрузки fine-tuned весов
        self.model_path = 'models/nn_model_weights.pth'
        # Путь для загрузки базовых весов ResNet-18
        self.base_weights_path = 'models/resnet18-f37072fd.pth'

        self.model = self._initialize_model()

        # Стандартные преобразования
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _initialize_model(self):
        """Загрузка предобученной модели и замена финального слоя."""

        # 1. Загрузка структуры модели ResNet-18 без весов
        model = models.resnet18(weights=None)

        # 2. Загрузка базовых весов с диска
        if os.path.exists(self.base_weights_path):
            print(f"Загрузка базовых весов ResNet-18 из локального файла: {self.base_weights_path}")
            # Загружаем state_dict и применяем к модели
            state_dict = torch.load(self.base_weights_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"Ошибка: Базовые веса ResNet-18 не найдены по пути: {self.base_weights_path}. Пожалуйста, скачайте их вручную и поместите в models/.")

        # 3. Заморозка всех слоев
        for param in model.parameters():
            param.requires_grad = False

        # 4. Замена финального полносвязного слоя для нашей задачи
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        return model.to(device)

    def _load_weights(self):
        """Загружает fine-tuned веса модели."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл весов НС не найден: {self.model_path}. Запустите в режиме 'train'.")

        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        print(f"Fine-tuned веса НС успешно загружены из: {self.model_path}")

    def train(self, train_paths: List[str], train_labels: List[int], num_epochs: int = 10, batch_size: int = 8,
              learning_rate: float = 0.001):
        """Обучение финального слоя сети."""

        print("--- Обучение нейросетевого классификатора (Transfer Learning) ---")
        train_dataset = ImageDataset(train_paths, train_labels, transform=self.data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                if inputs.shape[0] == 0: continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            print(f"Эпоха {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Обучение завершено. Веса сохранены в: {self.model_path}")

    def predict(self, test_paths: List[str], batch_size: int = 8) -> List[int]:
        """Предсказание метки классов на тестовой выборке."""

        if self.model.training:
            self._load_weights()

        test_dataset = ImageDataset(test_paths, [0] * len(test_paths), transform=self.data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                if inputs.shape[0] == 0: continue

                inputs = inputs.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return predictions
