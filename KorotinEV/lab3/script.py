import cv2
import numpy as np
import os
import argparse
import joblib
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

class BaseClassifier:
    def __init__(self, algorithm, image_size=(224, 224), class_names=None):
        self.algorithm = algorithm
        self.image_size = image_size
        
        if class_names is None:
            self.class_names = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
        else:
            self.class_names = class_names
            
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
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

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            'algorithm': self.algorithm,
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
        self.image_size = tuple(metadata['image_size'])
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Модель загружена из директории: {model_dir}")
        print(f"Алгоритм: {self.algorithm}")
        return True

    def train(self, train_file, images_dir="."):
        raise NotImplementedError("Метод train должен быть реализован в дочернем классе")

    def test(self, test_file, images_dir="."):
        raise NotImplementedError("Метод test должен быть реализован в дочернем классе")


class BOWClassifier(BaseClassifier):
    def __init__(self, n_clusters=100, image_size=(224, 224), class_names=None):
        super().__init__('bow', image_size, class_names)
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
    
    def build_vocabulary(self, all_descriptors):
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        return self.kmeans
    
    def descr_to_histogram(self, descriptors):
        labels = self.kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
        histogram = histogram.astype(float)
        if np.sum(histogram) > 0:
            histogram /= np.sum(histogram)
        return histogram

    def train(self, train_file, images_dir="."):
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        all_descriptors = []
        
        for image_path in train_paths:
            descriptors = self.extract_features(image_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        self.build_vocabulary(all_descriptors)
 
        train_features = []
        for descriptors in all_descriptors:
            histogram = self.descr_to_histogram(descriptors)
            train_features.append(histogram)
        
        train_features = np.array(train_features)     
        train_features_scaled = self.scaler.fit_transform(train_features)

        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.svm.fit(train_features_scaled, label_ids)

        train_predictions = self.svm.predict(train_features_scaled)
        train_accuracy = accuracy_score(label_ids, train_predictions)
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        return train_accuracy

    def test(self, test_file, images_dir="."):
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        test_features = []
        
        for path in test_paths:
            descriptors = self.extract_features(path)
            if descriptors is not None:
                histogram = self.descr_to_histogram(descriptors)
                test_features.append(histogram)
        
        test_features = np.array(test_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        test_predictions = self.svm.predict(test_features_scaled)
        test_accuracy = accuracy_score(label_ids, test_predictions)
        
        test_predictions_labels = [self.class_names[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print("\nДетальный отчет по классификации:")
        print(classification_report(test_labels, test_predictions_labels))
        
        return test_accuracy

    def save_model(self, model_dir="models"):
        super().save_model(model_dir)
        joblib.dump(self.kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
        joblib.dump(self.svm, os.path.join(model_dir, 'svm_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))

    def load_model(self, model_dir="models"):
        super().load_model(model_dir)
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        self.svm = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.n_clusters = self.kmeans.n_clusters
        return True


class CNNClassifier(BaseClassifier):
    def __init__(self, image_size=(224, 224), batch_size=16, epochs=5, learning_rate=0.001, 
                 dropout_rate=0.5, class_names=None):
        super().__init__('cnn', image_size, class_names)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        
    def create_cnn_model(self):
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=(self.image_size[0], self.image_size[1], 3))
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='categorical_crossentropy',
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

    def train(self, train_file, images_dir="."):
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        X_train = []
        y_train = []
        
        for (path, label_id) in zip(train_paths, label_ids):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_train.append(image)
                y_train.append(label_id)
        
        X_train = np.array(X_train)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.class_names))
        
        self.model = self.create_cnn_model()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        
        train_predictions = np.argmax(self.model.predict(X_train), axis=1)
        train_true_labels = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        return train_accuracy

    def test(self, test_file, images_dir="."):
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        X_test = []
        y_test = []
        
        for (path, label_id) in zip(test_paths, label_ids):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_test.append(image)
                y_test.append(label_id)
        
        X_test = np.array(X_test)
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(self.class_names))
        
        test_predictions_proba = self.model.predict(X_test)
        test_predictions = np.argmax(test_predictions_proba, axis=1)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        test_predictions_labels = [self.class_names[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print("\nДетальный отчет по классификации:")
        print(classification_report(test_labels, test_predictions_labels))
        
        return test_accuracy

    def save_model(self, model_dir="models"):
        super().save_model(model_dir)
        self.model.save(os.path.join(model_dir, 'cnn_model.h5'))

    def load_model(self, model_dir="models"):
        super().load_model(model_dir)
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
        return True

def create_classifier(algorithm, **kwargs):
    if algorithm == 'bow':
        return BOWClassifier(
            n_clusters=kwargs.get('n_clusters', 100),
            image_size=kwargs.get('image_size', (224, 224)),
            class_names=kwargs.get('class_names')
        )
    elif algorithm == 'cnn':
        return CNNClassifier(
            image_size=kwargs.get('image_size', (224, 224)),
            batch_size=kwargs.get('batch_size', 16),
            epochs=kwargs.get('epochs', 5),
            learning_rate=kwargs.get('learning_rate', 0.001),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            class_names=kwargs.get('class_names')
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")


def parser():
    parser = argparse.ArgumentParser(description='Классификация достопримечательностей Нижнего Новгорода')
    parser.add_argument('--data_dir', type=str, default='./NNClassification', help='Путь к директории с изображениями')
    parser.add_argument('--train_file', type=str, default='train_test_split/train.txt', help='Путь к файлу train.txt')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt', help='Путь к файлу test.txt')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='Алгоритм классификации: bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--clusters', type=int, default=100, help='Количество кластеров для метода мешок слов')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                       help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование)')
    parser.add_argument('--model_dir', type=str, default='models', help='Директория для сохранения/загрузки моделей')
    
    # CNN параметры
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча для CNN')
    parser.add_argument('--epochs', type=int, default=5, help='Количество эпох для CNN')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Скорость обучения для CNN')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate для CNN')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], 
                       help='Размер изображения (ширина высота)')

    return parser.parse_args()

def main():
    args = parser()
    
    if args.mode in ['train', 'both'] and not args.train_file:
        parser.error("Для режима обучения требуется указать --train_file")
    
    if args.mode in ['test', 'both'] and not args.test_file:
        parser.error("Для режима тестирования требуется указать --test_file")
    
    classifier_kwargs = {
        'n_clusters': args.clusters,
        'image_size': tuple(args.image_size),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate
    }
    
    classifier = create_classifier(args.algorithm, **classifier_kwargs)

    print(f"Режим работы: {args.mode}")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Директория с данными: {args.data_dir}")
    
    train_accuracy = 0
    test_accuracy = 0
    
    if args.mode in ['train', 'both']:
        train_accuracy = classifier.train(args.train_file, args.data_dir)
        classifier.save_model(args.model_dir)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            if not os.path.exists(args.model_dir):
                print(f"Ошибка: директория с моделью {args.model_dir} не существует!")
                return
            load_model(args.model_dir)
        
        test_accuracy = classifier.test(args.test_file, args.data_dir)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Режим работы: {args.mode}")
    
    if args.mode in ['train', 'both']:
        print(f"Accuracy на обучающей выборке: {train_accuracy:.4f}")
    
    if args.mode in ['test', 'both']:
        print(f"Accuracy на тестовой выборке: {test_accuracy:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()
