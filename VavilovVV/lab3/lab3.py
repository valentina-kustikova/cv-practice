import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageClassifier:
    def __init__(self, data_dir, train_file, test_file, mode='train', algorithm='bow', params=None):
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.mode = mode
        self.algorithm = algorithm
        self.params = params or {}
        self.classes = ['01_NizhnyNovgorodKremlin', '04_ArkhangelskCathedral', '08_PalaceOfLabor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.detector = None
        self.classifier = None
        self.kmeans = None
        self.scaler = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\\')
                    if len(parts) >= 2:
                        class_name = parts[1]
                        img_path = os.path.join(self.data_dir, line.replace('\\', os.sep))
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                images.append(img)
                                labels.append(self.class_to_idx.get(class_name, -1))
        return images, np.array(labels)

    def extract_features_bow(self, images):
        descriptors = []
        self.detector = cv2.SIFT_create()  # Can be changed to ORB, etc.
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)
            if des is not None:
                descriptors.extend(des)
        return np.array(descriptors)

    def build_vocabulary(self, descriptors, k=100):
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        self.kmeans.fit(descriptors)
        return self.kmeans

    def get_bow_histograms(self, images):
        histograms = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)
            if des is not None:
                hist = np.zeros(self.params.get('k', 100))
                labels = self.kmeans.predict(des)
                for label in labels:
                    hist[label] += 1
                histograms.append(hist)
            else:
                histograms.append(np.zeros(self.params.get('k', 100)))
        return np.array(histograms)

    def train_bow(self):
        train_images, train_labels = self.load_data(self.train_file)
        descriptors = self.extract_features_bow(train_images)
        self.build_vocabulary(descriptors, k=self.params.get('k', 100))

        train_hist = self.get_bow_histograms(train_images)
        self.scaler = StandardScaler().fit(train_hist)
        train_hist = self.scaler.transform(train_hist)

        self.classifier = SVC(kernel='linear', random_state=42)
        self.classifier.fit(train_hist, train_labels)
        print("BoW model trained.")

    def test_bow(self):
        test_images, test_labels = self.load_data(self.test_file)
        test_hist = self.get_bow_histograms(test_images)
        test_hist = self.scaler.transform(test_hist)
        predictions = self.classifier.predict(test_hist)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(classification_report(test_labels, predictions, target_names=self.classes))

    def prepare_transforms(self, train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def train_nn(self):
        train_images, train_labels = self.load_data(self.train_file)
        train_transform = self.prepare_transforms(train=True)
        train_dataset = ImageDataset(train_images, train_labels, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.params.get('batch_size', 32), shuffle=True)

        self.model = mobilenet_v2(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params.get('lr', 0.001))

        for epoch in range(self.params.get('epochs', 10)):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        print("Neural network model trained.")

    def test_nn(self):
        test_images, test_labels = self.load_data(self.test_file)
        test_transform = self.prepare_transforms(train=False)
        test_dataset = ImageDataset(test_images, test_labels, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.params.get('batch_size', 32), shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy}")

    def run(self):
        if self.algorithm == 'bow':
            if 'train' in self.mode:
                self.train_bow()
            if 'test' in self.mode:
                self.test_bow()
        elif self.algorithm == 'nn':
            if 'train' in self.mode:
                self.train_nn()
            if 'test' in self.mode:
                self.test_nn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification App")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--train_file', type=str, required=True, help='Train split file')
    parser.add_argument('--test_file', type=str, required=True, help='Test split file')
    parser.add_argument('--mode', type=str, default='train,test', help='Mode: train, test or train,test')
    parser.add_argument('--algorithm', type=str, default='bow', choices=['bow', 'nn'], help='Algorithm: bow or nn')
    parser.add_argument('--params', type=str, default='{}', help='Parameters as JSON string')

    args = parser.parse_args()
    import json

    params = json.loads(args.params)

    classifier = ImageClassifier(args.data_dir, args.train_file, args.test_file, args.mode, args.algorithm, params)
    classifier.run()