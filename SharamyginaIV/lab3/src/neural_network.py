import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import os
import pickle
from .utils import extract_class_name


class LandmarkDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_encoder = {}
        self.reverse_label_encoder = {}

        labels = []
        for img_path in image_paths:
            class_name = extract_class_name(img_path)
            if class_name not in self.label_encoder:
                idx = len(self.label_encoder)
                self.label_encoder[class_name] = idx
                self.reverse_label_encoder[idx] = class_name
            labels.append(self.label_encoder[class_name])

        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class NeuralNetworkClassifier:
    def __init__(self, num_classes=3, epochs=20, batch_size=32, learning_rate=0.001):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._build_model()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = self.model.to(self.device)

    def _build_model(self):
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

        return model

    def train(self, train_paths):
        train_dataset = LandmarkDataset(train_paths, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    def evaluate(self, test_paths):
        test_dataset = LandmarkDataset(test_paths, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100. * correct / total
        return accuracy / 100.0  # Return as decimal

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        self.num_classes = checkpoint['num_classes']
        self.epochs = checkpoint['epochs']
        self.batch_size = checkpoint['batch_size']
        self.learning_rate = checkpoint['learning_rate']

        print(f"Model loaded from {filepath}")
