# src/cnn_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class LandmarkDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels.astype(np.int64)  # ← ВОТ ЭТО ГЛАВНОЕ!
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # ← и тут long!
        return img, label

class TransferLearningClassifier:
    def __init__(self, num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights='IMAGENET1K_V1')  # обновлённый синтаксис
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def train(self, train_images, train_labels, epochs=15, batch_size=32):
        dataset = LandmarkDataset(train_images, train_labels, self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")
    
    def predict(self, test_images):
        dataset = LandmarkDataset(test_images, np.zeros(len(test_images)), self.transform)
        loader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                
        return np.array(predictions), np.array(probabilities)
    
    def save(self, path="cnn_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Модель сохранена → {path}")