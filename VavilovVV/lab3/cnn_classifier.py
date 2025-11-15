import logging
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple

from base_classifier import BaseClassifier

log = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"Не удалось прочитать (возможно, битый файл): {img_path}")
                # Возвращаем первое изображение, чтобы не сломать батч
                return self.__getitem__(0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            if self.transform:
                img_tensor = self.transform(img_pil)

            return img_tensor, label

        except Exception as e:
            log.error(f"Ошибка при загрузке {img_path}: {e}")
            # Возвращаем первое изображение, чтобы не сломать батч
            return self.__getitem__(0)


class CnnClassifier(BaseClassifier):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.nn_model_path = self.model_save_dir / "nn_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Используется устройство: {self.device}")

        self.model = None

    def prepare_transforms(self, train: bool = True) -> transforms.Compose:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img_size = self.args.img_size

        if train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _init_nn_model(self):
        self.model = mobilenet_v2(pretrained=self.args.pretrained)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(self.device)
        log.info(f"Архитектура MobileNetV2 инициализирована (pretrained={self.args.pretrained}).")

    def train(self):
        log.info("---Начало обучения NN ---")
        if self.model is None:
            self._init_nn_model()

        train_paths, train_labels = self.load_data(self.train_file)
        if not train_paths:
            return

        train_transform = self.prepare_transforms(train=True)
        train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        num_epochs = self.args.epochs
        log.info(f"Старт обучения NN на {num_epochs} эпох...")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            log.info(f"Эпоха {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        log.info("Модель NN обучена.")
        self.save_model()

    def save_model(self):
        if self.model is None:
            log.error("NN модель не инициализирована. Сохранение отменено.")
            return

        torch.save(self.model.state_dict(), self.nn_model_path)
        log.info(f"Модель NN сохранена: {self.nn_model_path}")

    def load_model(self):
        if not self.nn_model_path.exists():
            raise FileNotFoundError(f"Файл модели NN не найден: {self.nn_model_path}")

        if self.model is None:
            self._init_nn_model()

        self.model.load_state_dict(
            torch.load(self.nn_model_path, map_location=self.device)
        )
        self.model.to(self.device)
        log.info(f"Модель NN загружена из {self.nn_model_path}")

    def test(self):
        log.info("--- Начало тестирования NN ---")
        if self.model is None:
            log.warning("NN модель не загружена. Попытка загрузки...")
            try:
                self.load_model()
            except FileNotFoundError as e:
                log.error(f"Ошибка загрузки: {e}")
                log.error("Сначала обучите модель в режиме 'train'.")
                return

        test_paths, test_labels = self.load_data(self.test_file)
        if not test_paths:
            return

        test_transform = self.prepare_transforms(train=False)
        test_dataset = ImageDataset(test_paths, test_labels, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.classes,
            zero_division=0
        )
        log.info(f"--- NN Результаты Теста ---\n"
                 f"Accuracy: {acc:.4f}\n"
                 f"{report}\n"

                 f"---------------------------")
