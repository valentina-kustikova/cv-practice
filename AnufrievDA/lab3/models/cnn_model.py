import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List
import os
import torchvision.transforms as transforms

class LandmarkDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = Image.new('RGB', (224, 224))
            
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CNNClassifier:
    def __init__(self, model_name: str = 'resnet18', num_classes: int = 3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes

        print(f"Инициализация модели '{model_name}'...")
        self.model = self._load_pretrained_model()
        self.model.to(self.device)

    def _load_pretrained_model(self):
        # Пытаемся загрузить локально, если нет - качаем
        local_path = os.path.join("models", f"{self.model_name}.pth")
        
        if self.model_name == 'resnet18':
            if os.path.exists(local_path):
                model = models.resnet18(weights=None)
                state_dict = torch.load(local_path, map_location='cpu')
                if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
            else:
                model = models.resnet18(weights='DEFAULT')
                
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model

    def fit(self, train_paths: List[str], train_labels: List[int],
            val_paths: List[str] = None, val_labels: List[int] = None,
            batch_size: int = 16, epochs: int = 10, lr: float = 1e-4):
        
        train_dataset = LandmarkDataset(train_paths, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 для Windows безопаснее

        val_loader = None
        if val_paths and val_labels:
            val_dataset = LandmarkDataset(val_paths, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        print(f"Начало обучения {self.model_name} (epochs={epochs})...")
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

            if val_loader:
                val_acc = self._evaluate_loader(val_loader)
                print(f"Val Acc: {val_acc:.4f}")
            
            scheduler.step()

    def _evaluate_loader(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0

    def predict(self, image_paths: List[str], batch_size: int = 32) -> List[int]:
        dataset = LandmarkDataset(image_paths, [0]*len(image_paths))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, pred = torch.max(outputs, 1)
                preds.extend(pred.cpu().numpy().tolist())
        return preds

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, path)
        print(f"Модель сохранена в: {path}")

    @classmethod
    def load(cls, path: str, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location=device)
        obj = cls(model_name=ckpt['model_name'], num_classes=ckpt['num_classes'], device=device)
        obj.model.load_state_dict(ckpt['model_state_dict'])
        return obj