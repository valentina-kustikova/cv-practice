import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LandmarkClassifier:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.svm = None
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()
        
    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors
    
    def build_vocabulary(self, image_paths):
        all_descriptors = []
        
        print("Извлечение признаков для построения словаря...")
        for i, image_path in enumerate(image_paths):
            descriptors = self.extract_features(image_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(image_paths)} изображений")
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {all_descriptors.shape[0]}")

        print("Кластеризация дескрипторов...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        
        return self.kmeans
    
    def image_to_histogram(self, image_path):
        descriptors = self.extract_features(image_path)
        if descriptors is None:
            return np.zeros(self.n_clusters)

        labels = self.kmeans.predict(descriptors)

        histogram, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))

        histogram = histogram.astype(float)
        if np.sum(histogram) > 0:
            histogram /= np.sum(histogram)
            
        return histogram
    
    def detect_label_from_path(self, file_path):
        if '01_NizhnyNovgorodKremlin' in file_path:
            return 'Нижегородский Кремль'
        elif '08_PalaceOfLabor' in file_path:
            return 'Дворец труда'
        elif '04_ArkhangelskCathedral' in file_path:
            return 'Архангельский собор'
        else:
            print(f"Не удалось определить метку для файла: {file_path}")
            return None
    
    def load_data(self, file_list_path, images_dir="."):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        image_paths = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            full_path = os.path.join(images_dir, line)
            
            if os.path.exists(full_path):
                label = self.detect_label_from_path(full_path)
                if label is not None:
                    image_paths.append(full_path)
                    labels.append(label)
                else:
                    print(f"Пропущен файл (не определена метка): {full_path}")
            else:
                print(f"Файл не найден: {full_path}")
        
        print(f"Загружено {len(image_paths)} изображений")
        print(f"Распределение по классам:")
        for label_name in ['Нижегородский Кремль', 'Дворец труда', 'Архангельский собор']:
            count = labels.count(label_name)
            print(f"  {label_name}: {count} изображений")
        
        return image_paths, labels
    
    def train(self, train_file, images_dir="."):
        print("Загрузка обучающих данных...")
        train_paths, train_labels = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        print("Построение словаря визуальных слов...")
        self.build_vocabulary(train_paths)
        
        print("Создание признаков для обучающей выборки...")
        train_features = []
        valid_labels = []
        
        for i, (path, label) in enumerate(zip(train_paths, train_labels)):
            histogram = self.image_to_histogram(path)
            train_features.append(histogram)
            valid_labels.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(train_paths)} обучающих изображений")
        
        train_features = np.array(train_features)
        
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        print("Обучение SVM классификатора...")
        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.svm.fit(train_features_scaled, valid_labels)
        
        train_predictions = self.svm.predict(train_features_scaled)
        train_accuracy = accuracy_score(valid_labels, train_predictions)
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        
        return train_accuracy
    
    def test(self, test_file, images_dir="."):
        print("Загрузка тестовых данных...")
        test_paths, test_labels = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        print("Создание признаков для тестовой выборки...")
        test_features = []
        valid_test_labels = []
        
        for i, (path, label) in enumerate(zip(test_paths, test_labels)):
            histogram = self.image_to_histogram(path)
            test_features.append(histogram)
            valid_test_labels.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(test_paths)} тестовых изображений")
        
        test_features = np.array(test_features)
        
        test_features_scaled = self.scaler.transform(test_features)
        
        test_predictions = self.svm.predict(test_features_scaled)
        test_accuracy = accuracy_score(valid_test_labels, test_predictions)
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        
        print("\nДетальный отчет по классификации:")
        print(classification_report(valid_test_labels, test_predictions))
        
        return test_accuracy

def main():
    train_file = "train_test_split/train.txt"
    test_file = "train_test_split/test.txt"
    images_dir = "."
    
    classifier = LandmarkClassifier(n_clusters=100)

    train_accuracy = classifier.train(train_file, images_dir)
    
    test_accuracy = classifier.test(test_file, images_dir)
    
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"TPR на обучающей выборке: {train_accuracy:.4f}")
    print(f"TPR на тестовой выборке: {test_accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
