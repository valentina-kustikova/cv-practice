import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from typing import List

class LandmarkDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        if self.transform is None:
            # Стандартные трансформации для ImageNet моделей
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Открываем через PIL (RGB)
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Возвращаем черный квадрат в случае ошибки
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CNNClassifier:
    def __init__(self, model_name: str = 'resnet18', num_classes: int = 3, device=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = self._load_pretrained_model()
        self.model = self.model.to(self.device)

    def _load_pretrained_model(self) -> nn.Module:
        # Путь к локальным весам (как в примере)
        local_path = os.path.join("models", f"{self.model_name}.pth")
        
        weights = None
        # Если файла нет, PyTorch скачает сам (weights='DEFAULT')
        # Если файл есть, загружаем его
        
        if self.model_name == 'resnet18':
            if os.path.exists(local_path):
                print(f"Loading local weights from {local_path}")
                model = models.resnet18(weights=None)
                model.load_state_dict(torch.load(local_path))
            else:
                print("Downloading pretrained weights (ResNet18)...")
                model = models.resnet18(weights='DEFAULT')
                
            # Заменяем последний слой
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            
        elif self.model_name == 'mobilenet_v2':
            if os.path.exists(local_path):
                print(f"Loading local weights from {local_path}")
                model = models.mobilenet_v2(weights=None)
                model.load_state_dict(torch.load(local_path))
            else:
                print("Downloading pretrained weights (MobileNetV2)...")
                model = models.mobilenet_v2(weights='DEFAULT')
            
            # Заменяем классификатор
            # У MobileNetV2 классификатор это Sequential, меняем последний слой [1]
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        return model

    def fit(self, train_paths, train_labels, val_paths=None, val_labels=None, batch_size=16, epochs=10, lr=1e-4):
        train_dataset = LandmarkDataset(train_paths, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        # Обучаем только последние слои или всю сеть? Обычно для лабы лучше всю, но с маленьким LR
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Starting training CNN ({self.model_name}) for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    def predict(self, image_paths: List[str], batch_size: int = 32) -> List[int]:
        self.model.eval()
        dataset = LandmarkDataset(image_paths, [0]*len(image_paths)) # Лейблы не важны
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                
        return all_preds