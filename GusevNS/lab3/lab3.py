import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Опциональный импорт TensorFlow (только для нейросетевого классификатора)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class ImageClassifier:
    """Базовый класс для классификатора изображений"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.class_names = ['01_NizhnyNovgorodKremlin', '04_ArkhangelskCathedral', '08_PalaceOfLabor', '77_airhockey']
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        
    def load_dataset(self, file_list):
        """Загрузка датасета из файла со списком изображений"""
        images, labels = [], []
        with open(file_list, 'r') as f:
            for line in f:
                img_path = self.data_path / line.strip()
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # Определение класса из пути
                class_name = img_path.parent.name
                if class_name in self.class_map:
                    images.append(img)
                    labels.append(self.class_map[class_name])
        return images, np.array(labels)

class BagOfWordsClassifier(ImageClassifier):
    """Классификатор на основе алгоритма Bag of Words"""
    def __init__(self, data_path, n_clusters=100, descriptor_type='sift'):
        super().__init__(data_path)
        self.n_clusters = n_clusters
        self.descriptor_type = descriptor_type
        self.kmeans = None
        self.svm = None
        
        # Выбор дескриптора
        if descriptor_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif descriptor_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
        else:
            raise ValueError("Поддерживаются только 'sift' и 'orb'")
    
    def extract_descriptors(self, images):
        """Извлечение дескрипторов из изображений"""
        all_descriptors = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detectAndCompute(gray, None)
            if desc is not None:
                all_descriptors.append(desc)
        return all_descriptors
    
    def build_histogram(self, descriptors):
        """Построение гистограммы визуальных слов"""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)
        
        if self.descriptor_type == 'orb':
            # ORB дескрипторы - бинарные, используем Hamming расстояние
            descriptors = descriptors.astype(np.float32)
        
        predictions = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.n_clusters)
        for pred in predictions:
            histogram[pred] += 1
        # Нормализация
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        return histogram
    
    def train(self, train_images, train_labels):
        """Обучение классификатора"""
        print("Извлечение дескрипторов для обучения...")
        all_descriptors = self.extract_descriptors(train_images)
        
        # Объединение всех дескрипторов для кластеризации
        desc_stack = np.vstack([d for d in all_descriptors if d is not None])
        if self.descriptor_type == 'orb':
            desc_stack = desc_stack.astype(np.float32)
        
        print(f"Кластеризация {len(desc_stack)} дескрипторов в {self.n_clusters} кластеров...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(desc_stack)
        
        # Построение гистограмм для обучающих изображений
        print("Построение гистограмм визуальных слов...")
        train_features = []
        for desc in all_descriptors:
            if desc is not None:
                hist = self.build_histogram(desc)
                train_features.append(hist)
        
        train_features = np.array(train_features)
        
        # Обучение SVM
        print("Обучение SVM классификатора...")
        self.svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        self.svm.fit(train_features, train_labels)
        print("Обучение завершено!")
    
    def predict(self, test_images):
        """Предсказание классов"""
        test_descriptors = self.extract_descriptors(test_images)
        predictions = []
        for desc in test_descriptors:
            if desc is not None:
                hist = self.build_histogram(desc)
                pred = self.svm.predict([hist])[0]
                predictions.append(pred)
            else:
                predictions.append(0)  # По умолчанию первый класс
        return np.array(predictions)
    
    def save(self, path):
        """Сохранение модели"""
        with open(path, 'wb') as f:
            pickle.dump({'kmeans': self.kmeans, 'svm': self.svm, 
                        'n_clusters': self.n_clusters, 'descriptor_type': self.descriptor_type}, f)
    
    def load(self, path):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.svm = data['svm']
            self.n_clusters = data['n_clusters']
            self.descriptor_type = data['descriptor_type']
            
            # Пересоздание детектора с правильным типом из сохраненной модели
            if self.descriptor_type == 'sift':
                self.detector = cv2.SIFT_create()
            elif self.descriptor_type == 'orb':
                self.detector = cv2.ORB_create(nfeatures=2000)

class NeuralNetworkClassifier(ImageClassifier):
    """Классификатор на основе нейронной сети с transfer learning"""
    def __init__(self, data_path, img_size=224, epochs=20, batch_size=16):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow не установлен. Установите: pip install tensorflow")
        super().__init__(data_path)
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def preprocess_images(self, images, labels=None):
        """Предобработка изображений для нейронной сети"""
        processed = []
        for img in images:
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = img_to_array(img_rgb) / 255.0
            processed.append(img_array)
        X = np.array(processed)
        if labels is not None:
            y = to_categorical(labels, num_classes=len(self.class_names))
            return X, y
        return X
    
    def build_model(self):
        """Построение модели на основе MobileNetV2"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(self.img_size, self.img_size, 3))
        
        # Заморозка базовой модели
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Добавление собственных слоев
        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Делаем одномерный вектор из огромной карты признаков (уменьшает размерность)
        x = Dense(128, activation='relu')(x) # Комбинирует большое кол-во чисел в 128 чисел для дальнейшей классификации
        x = Dropout(0.5)(x) # Предотвращает переобучение
        predictions = Dense(len(self.class_names), activation='softmax')(x) # Превращает 128 чисел в вероятности для каждого класса
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    def train(self, train_images, train_labels):
        """Обучение нейронной сети"""
        print("Предобработка изображений...")
        X_train, y_train = self.preprocess_images(train_images, train_labels)
        
        print("Построение модели...")
        self.build_model()
        
        print("Обучение нейронной сети...")
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                      validation_split=0.2, verbose=1)
        print("Обучение завершено!")
    
    def predict(self, test_images):
        """Предсказание классов"""
        X_test = self.preprocess_images(test_images)
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)
    
    def save(self, path):
        """Сохранение модели"""
        self.model.save(path)
    
    def load(self, path):
        """Загрузка модели"""
        self.model = tf.keras.models.load_model(path)

def evaluate(y_true, y_pred, class_names):
    """Оценка качества классификации"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nТочность (Accuracy): {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Классификация достопримечательностей Нижнего Новгорода')
    parser.add_argument('--data_path', type=str, default='Data', help='Путь к директории с данными')
    parser.add_argument('--train_file', type=str, default='Data/train.txt', help='Файл с обучающей выборкой')
    parser.add_argument('--test_file', type=str, default='Data/test.txt', help='Файл с тестовой выборкой')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', help='Режим работы')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'nn', 'both'], default='both', help='Алгоритм классификации')
    parser.add_argument('--bow_clusters', type=int, default=100, help='Количество кластеров для BoW')
    parser.add_argument('--bow_descriptor', type=str, choices=['sift', 'orb'], default='sift', help='Тип дескриптора для BoW')
    parser.add_argument('--nn_epochs', type=int, default=20, help='Количество эпох для нейросети')
    parser.add_argument('--nn_batch', type=int, default=16, help='Размер батча для нейросети')
    parser.add_argument('--model_path', type=str, default='models', help='Путь для сохранения моделей')
    
    args = parser.parse_args()
    
    # Создание директории для моделей
    os.makedirs(args.model_path, exist_ok=True)
    
    algorithms = ['bow', 'nn'] if args.algorithm == 'both' else [args.algorithm]
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Алгоритм: {'Bag of Words' if algo == 'bow' else 'Neural Network'}")
        print(f"{'='*60}")
        
        # Создание классификатора
        if algo == 'bow':
            classifier = BagOfWordsClassifier(args.data_path, args.bow_clusters, args.bow_descriptor)
            # Имя файла зависит от типа дескриптора
            if args.bow_descriptor == 'sift':
                model_path = f"{args.model_path}/bow_model.pkl"
            else:
                model_path = f"{args.model_path}/bow_model_{args.bow_descriptor}.pkl"
        else:
            classifier = NeuralNetworkClassifier(args.data_path, epochs=args.nn_epochs, batch_size=args.nn_batch)
            model_path = f"{args.model_path}/nn_model.h5"
        
        # Обучение
        if args.mode in ['train', 'both']:
            print("\nЗагрузка обучающей выборки...")
            train_images, train_labels = classifier.load_dataset(args.train_file)
            print(f"Загружено {len(train_images)} обучающих изображений")
            
            classifier.train(train_images, train_labels)
            classifier.save(model_path)
            print(f"Модель сохранена в {model_path}")
        
        # Тестирование
        if args.mode in ['test', 'both']:
            if args.mode == 'test':
                print("\nЗагрузка модели...")
                classifier.load(model_path)
            
            print("\nЗагрузка тестовой выборки...")
            test_images, test_labels = classifier.load_dataset(args.test_file)
            print(f"Загружено {len(test_images)} тестовых изображений")
            
            print("\nКлассификация тестовой выборки...")
            predictions = classifier.predict(test_images)
            
            evaluate(test_labels, predictions, classifier.class_names)

if __name__ == '__main__':
    main()

