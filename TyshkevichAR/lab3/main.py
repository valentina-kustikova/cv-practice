import os
import argparse
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

def read_split(split_file, data_root):

    paths, labels = [], []

    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            relpath = line.strip().replace("\\", "/")
            if not relpath:
                continue

            imgpath = os.path.join(data_root, relpath)
            if not os.path.exists(imgpath):
                print(f"[WARN] File not found: {imgpath}")
                continue

            parts = relpath.split('/')
            label = "unknown"
            if len(parts) >= 2:
                folder = parts[-2]
                label = ''.join([c for c in folder if not c.isdigit() and c not in ['_', '-', '.']])
                label = label.lower()

            paths.append(imgpath)
            labels.append(label)

    print(f"[INFO] Loaded {len(paths)} images from {split_file}")
    print(f"[INFO] Found classes: {sorted(set(labels))}")
    return paths, labels

# Мешок слов
class BoWClassifier:
    def __init__(self, k=100, detector_name='SIFT', svm_C=1.0):
        self.k = k
        self.detector_name = detector_name
        self.svm_C = svm_C
        self.kmeans = None
        self.scaler = StandardScaler()
        self.clf = None
        self.detector = self._create_detector(detector_name)

    def _create_detector(self, name):
        name = name.upper()
        if name == 'SIFT':
            try:
                return cv2.SIFT_create()
            except:
                print("[WARN] SIFT not available, using ORB.")
                return cv2.ORB_create(1000)
        return cv2.ORB_create(1000)

    def _extract_descriptors(self, image_paths):
        all_descs = []
        per_image_descs = []
        for p in tqdm(image_paths, desc="Extracting features"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Cannot read image: {p}")
                per_image_descs.append(None)
                continue
            kps, desc = self.detector.detectAndCompute(img, None)
            if desc is None:
                per_image_descs.append(None)
            else:
                per_image_descs.append(desc)
                all_descs.append(desc)
        if len(all_descs) == 0:
            return [], per_image_descs
        stacked = np.vstack(all_descs)
        return stacked, per_image_descs

    def fit(self, image_paths, labels):
        print("Extracting descriptors for training set...")
        stacked, per_image_descs = self._extract_descriptors(image_paths)
        if len(stacked) == 0:
            raise RuntimeError("No descriptors found in training images.")

        print(f"Clustering descriptors with k={self.k} ...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            random_state=0,
            batch_size=1000,
            verbose=1
        )
        kmeans.fit(stacked)
        self.kmeans = kmeans

        hists = []
        for desc in per_image_descs:
            if desc is None:
                hist = np.zeros(self.k)
            else:
                words = kmeans.predict(desc)
                hist, _ = np.histogram(words, bins=np.arange(self.k + 1))
            hists.append(hist)

        X = np.array(hists)
        X_scaled = self.scaler.fit_transform(X)
        self.clf = SVC(kernel='linear', C=self.svm_C)
        print("Training SVM classifier...")
        self.clf.fit(X_scaled, labels)
        print("BoW training finished.")

    def predict(self, image_paths):
        _, per_image_descs = self._extract_descriptors(image_paths)
        if self.kmeans is None:
            raise RuntimeError("Model not trained.")
        hists = []
        for desc in per_image_descs:
            if desc is None:
                hist = np.zeros(self.k)
            else:
                words = self.kmeans.predict(desc)
                hist, _ = np.histogram(words, bins=np.arange(self.k + 1))
            hists.append(hist)
        X = np.array(hists)
        X_scaled = self.scaler.transform(X)
        preds = self.clf.predict(X_scaled)
        return preds

    def save(self, path):
        joblib.dump({
            'k': self.k,
            'detector_name': self.detector_name,
            'svm_C': self.svm_C,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'clf': self.clf
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.k = data['k']
        self.detector_name = data['detector_name']
        self.svm_C = data['svm_C']
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']
        self.clf = data['clf']
        self.detector = self._create_detector(self.detector_name)

# Нейросетевой классификатор
class ImageFilesDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, label_to_idx=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform
        if label_to_idx is None:
            unique = sorted(list(set(labels)))
            self.label_to_idx = {l: i for i, l in enumerate(unique)}
        else:
            self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_to_idx[self.labels[idx]]
        return img, label


class CNNClassifier:
    def __init__(self, num_classes, device='cpu'):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        return model

    def train(self, train_paths, train_labels, epochs=10, batch_size=16, lr=1e-4):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        unique_labels = sorted(list(set(train_labels)))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        dataset = ImageFilesDataset(train_paths, train_labels, transform=transform_train, label_to_idx=label_to_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}

    def predict(self, image_paths, batch_size=16):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFilesDataset(image_paths, [list(self.label_to_idx.keys())[0]] * len(image_paths), transform=transform, label_to_idx=self.label_to_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, p = torch.max(outputs, 1)
                preds.extend(p.cpu().numpy())
        return [self.idx_to_label[i] for i in preds]

def evaluate(preds, gt):
    print("Accuracy:", accuracy_score(gt, preds))
    print(classification_report(gt, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(gt, preds))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--mode', default='train,test')
    parser.add_argument('--algo', choices=['bow', 'cnn'], default='bow')
    parser.add_argument('--bow_k', type=int, default=100)
    parser.add_argument('--detector', default='SIFT')
    parser.add_argument('--svm_C', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_out', default=None)
    parser.add_argument('--model_in', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_paths, train_labels = read_split(os.path.join(args.data_dir, args.train_file), args.data_dir)
    test_paths, test_labels = read_split(os.path.join(args.data_dir, args.test_file), args.data_dir)
    modes = [m.strip() for m in args.mode.split(',')]

    # Установка имени файла модели по типу алгоритма
    if args.algo == 'bow':
        model_path_out = args.model_out or 'model_bow.bz2'
        model_path_in = args.model_in or 'model_bow.bz2'
    else:
        model_path_out = args.model_out or 'model_cnn.pth'
        model_path_in = args.model_in or 'model_cnn.pth'

    if args.algo == 'bow':
        model = BoWClassifier(k=args.bow_k, detector_name=args.detector, svm_C=args.svm_C)
        if 'train' in modes:
            model.fit(train_paths, train_labels)
            model.save(model_path_out)
            print(f"[INFO] BoW model saved to {model_path_out}")
        if 'test' in modes:
            if model.kmeans is None and os.path.exists(model_path_in):
                model.load(model_path_in)
            preds = model.predict(test_paths)
            evaluate(preds, test_labels)

    elif args.algo == 'cnn':
        classes = sorted(list(set(train_labels)))
        num_classes = len(classes)
        device = args.device if torch.cuda.is_available() else 'cpu'
        model = CNNClassifier(num_classes=num_classes, device=device)
        if 'train' in modes:
            model.train(train_paths, train_labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            torch.save({'model_state': model.model.state_dict(), 'label_to_idx': model.label_to_idx}, model_path_out)
            print(f"[INFO] CNN model saved to {model_path_out}")
        if 'test' in modes:
            if os.path.exists(model_path_in):
                data = torch.load(model_path_in, map_location=device)
                model.model.load_state_dict(data['model_state'])
                model.label_to_idx = data['label_to_idx']
                model.idx_to_label = {v: k for k, v in model.label_to_idx.items()}
            preds = model.predict(test_paths, batch_size=args.batch_size)
            evaluate(preds, test_labels)


if __name__ == "__main__":
    main()
