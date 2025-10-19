import cv2
import numpy as np
import os
import argparse
import joblib
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

class LandmarkClassifier:
    def __init__(self, n_clusters=100, algorithm='bow', image_size=(224, 224)):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.image_size = image_size
        self.kmeans = None
        self.svm = None
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()
        self.model = None
        
        self.class_names = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_label = {i: name for i, name in enumerate(self.class_names)}
        
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
        file_path_lower = file_path.lower()
        
        if any(keyword in file_path_lower for keyword in ['kremlin', '01_nizhnynovgorodkremlin']):
            return 'Нижегородский Кремль'
        elif any(keyword in file_path_lower for keyword in ['palace', '08_palaceoflabor']):
            return 'Дворец труда'
        elif any(keyword in file_path_lower for keyword in ['arkhangelsk', '04_arkhangelskcathedral']):
            return 'Архангельский собор'
        else:
            print(f"Не удалось определить метку для файла: {file_path}")
            return None
    
    def load_data(self, file_list_path, images_dir="."):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        image_paths = []
        labels = []
        label_ids = []
        
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
                    label_ids.append(self.label_to_id[label])
                else:
                    print(f"Пропущен файл (не определена метка): {full_path}")
            else:
                print(f"Файл не найден: {full_path}")
        
        print(f"Загружено {len(image_paths)} изображений")
        print(f"Распределение по классам:")
        for label_name in self.class_names:
            count = labels.count(label_name)
            print(f"  {label_name}: {count} изображений")
        
        return image_paths, labels, label_ids
    
    def create_cnn_model(self):
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=(self.image_size[0], self.image_size[1], 3))
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def preprocess_image_cnn(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype('float32') / 255.0
        
        return image
    
    def train_bow(self, train_paths, train_labels):
        print("Построение словаря визуальных слов...")
        self.build_vocabulary(train_paths)
        
        print("Создание признаков для обучающей выборки...")
        train_features = []
        
        for i, path in enumerate(train_paths):
            histogram = self.image_to_histogram(path)
            train_features.append(histogram)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(train_paths)} обучающих изображений")
        
        train_features = np.array(train_features)
        
        train_features_scaled = self.scaler.fit_transform(train_features)

        print("Обучение SVM классификатора...")
        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.svm.fit(train_features_scaled, train_labels)

        train_predictions = self.svm.predict(train_features_scaled)
        train_accuracy = recall_score(train_labels, train_predictions, average="weighted")
        
        return train_accuracy
    
    def train_cnn(self, train_paths, train_labels, label_ids):
        print("Подготовка данных для CNN...")
        
        X_train = []
        y_train = []
        
        for i, (path, label_id) in enumerate(zip(train_paths, label_ids)):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_train.append(image)
                y_train.append(label_id)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(train_paths)} обучающих изображений")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print("Создание CNN модели...")
        self.model = self.create_cnn_model()
        
        print("Обучение CNN...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=5
        )
        
        train_predictions = np.argmax(self.model.predict(X_train), axis=1)
        train_accuracy = recall_score(y_train, train_predictions, average="weighted")
        
        return train_accuracy
    
    def train(self, train_file, images_dir="."):
        print("Загрузка обучающих данных...")
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        if self.algorithm == 'bow':
            train_accuracy = self.train_bow(train_paths, train_labels)
        elif self.algorithm == 'cnn':
            train_accuracy = self.train_cnn(train_paths, train_labels, label_ids)
        else:
            print(f"Неизвестный алгоритм: {self.algorithm}")
            return 0
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        return train_accuracy
    
    def test_bow(self, test_paths, test_labels):
        print("Создание признаков для тестовой выборки...")
        test_features = []
        
        for i, path in enumerate(test_paths):
            histogram = self.image_to_histogram(path)
            test_features.append(histogram)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(test_paths)} тестовых изображений")
        
        test_features = np.array(test_features)
        
        test_features_scaled = self.scaler.transform(test_features)
        
        test_predictions = self.svm.predict(test_features_scaled)
        test_accuracy = recall_score(test_labels, test_predictions, average="weighted")
        
        return test_accuracy, test_predictions
    
    def test_cnn(self, test_paths, test_labels, label_ids):
        print("Подготовка тестовых данных для CNN...")
        X_test = []
        y_test = []
        
        for i, (path, label_id) in enumerate(zip(test_paths, label_ids)):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_test.append(image)
                y_test.append(label_id)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(test_paths)} тестовых изображений")
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        test_predictions_proba = self.model.predict(X_test)
        test_predictions = np.argmax(test_predictions_proba, axis=1)
        test_accuracy = recall_score(y_test, test_predictions, average="weighted")
        
        test_predictions_labels = [self.class_names[pred] for pred in test_predictions]
        
        return test_accuracy, test_predictions_labels
    
    def test(self, test_file, images_dir="."):
        print("Загрузка тестовых данных...")
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        if self.algorithm == 'bow':
            test_accuracy, test_predictions = self.test_bow(test_paths, test_labels)
        elif self.algorithm == 'cnn':
            test_accuracy, test_predictions = self.test_cnn(test_paths, test_labels, label_ids)
        else:
            print(f"Неизвестный алгоритм: {self.algorithm}")
            return 0
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        
        print("\nДетальный отчет по классификации:")
        print(classification_report(test_labels, test_predictions))
        
        return test_accuracy
    
    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        if self.algorithm == 'bow':
            joblib.dump(self.kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
            joblib.dump(self.svm, os.path.join(model_dir, 'svm_model.pkl'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
            
            metadata = {
                'algorithm': 'bow',
                'n_clusters': self.n_clusters,
                'class_names': self.class_names
            }
            
        elif self.algorithm == 'cnn':
            self.model.save(os.path.join(model_dir, 'cnn_model.h5'))
            
            metadata = {
                'algorithm': 'cnn',
                'image_size': self.image_size,
                'class_names': self.class_names
            }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Модель сохранена в директории: {model_dir}")
    
    def load_model(self, model_dir="models"):
        with open(os.path.join(model_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.algorithm = metadata['algorithm']
        self.class_names = metadata['class_names']
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_label = {i: name for i, name in enumerate(self.class_names)}
        
        if self.algorithm == 'bow':
            self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
            self.svm = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.n_clusters = metadata['n_clusters']
            
        elif self.algorithm == 'cnn':
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
            self.image_size = tuple(metadata['image_size'])
        
        print(f"Модель загружена из директории: {model_dir}")
        print(f"Алгоритм: {self.algorithm}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Классификация достопримечательностей Нижнего Новгорода')
    parser.add_argument('--data_dir', type=str, default='.', help='Путь к директории с изображениями')
    parser.add_argument('--train_file', type=str, default='train_test_split/train.txt', help='Путь к файлу train.txt')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt', help='Путь к файлу test.txt')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='Алгоритм классификации: bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--clusters', type=int, default=100, help='Количество кластеров для метода мешок слов')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                       help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование)')
    parser.add_argument('--model_dir', type=str, default='models', help='Директория для сохранения/загрузки моделей')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both'] and not args.train_file:
        parser.error("Для режима обучения требуется указать --train_file")
    
    if args.mode in ['test', 'both'] and not args.test_file:
        parser.error("Для режима тестирования требуется указать --test_file")
    
    classifier = LandmarkClassifier(n_clusters=args.clusters, algorithm=args.algorithm)
    
    print(f"Режим работы: {args.mode}")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Директория с данными: {args.data_dir}")
    
    train_accuracy = 0
    test_accuracy = 0
    
    if args.mode in ['train', 'both']:
        print(f"Обучающая выборка: {args.train_file}")
        print("\n" + "="*50)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("="*50)
        
        train_accuracy = classifier.train(args.train_file, args.data_dir)
        
        classifier.save_model(args.model_dir)
    
    if args.mode in ['test', 'both']:
        print(f"Тестовая выборка: {args.test_file}")
        print("\n" + "="*50)
        print("НАЧАЛО ТЕСТИРОВАНИЯ")
        print("="*50)
        
        if args.mode == 'test':
            if not os.path.exists(args.model_dir):
                print(f"Ошибка: директория с моделью {args.model_dir} не существует!")
                return
            classifier.load_model(args.model_dir)
        
        test_accuracy = classifier.test(args.test_file, args.data_dir)
    
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Режим работы: {args.mode}")
    
    if args.mode in ['train', 'both']:
        print(f"TPR на обучающей выборке: {train_accuracy:.4f}")
    
    if args.mode in ['test', 'both']:
        print(f"TPR на тестовой выборке: {test_accuracy:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()
