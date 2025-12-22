import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.cluster.vq import vq
import matplotlib.pyplot as plt
from .base_classifier import BaseClassifier

class BOWClassifier(BaseClassifier):
    """Классификатор на основе алгоритма 'Мешок слов'."""
    
    def __init__(self, n_clusters=200, image_size=(224, 224), 
                 detector_type='sift', class_names=None):
        super().__init__('bow', image_size, class_names)
        self.n_clusters = n_clusters
        self.detector_type = detector_type
        self.kmeans = None
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Инициализация детектора/дескриптора
        if detector_type.lower() == 'sift':
            self.detector = cv2.SIFT_create()
        elif detector_type.lower() == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif detector_type.lower() == 'akaze':
            self.detector = cv2.AKAZE_create()
        else:
            print(f"Детектор {detector_type} не поддерживается, используется SIFT")
            self.detector = cv2.SIFT_create()
    
    def extract_features(self, image_path, max_features=200):
        """Извлекает локальные признаки из изображения."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
        
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение ключевых точек и вычисление дескрипторов
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            # Если не найдено ключевых точек, возвращаем пустой массив
            return np.array([]).reshape(0, 128) if self.detector_type == 'sift' else np.array([]).reshape(0, 32)
        
        # Ограничиваем количество дескрипторов для ускорения обработки
        if len(descriptors) > max_features:
            indices = np.random.choice(len(descriptors), max_features, replace=False)
            descriptors = descriptors[indices]
        
        return descriptors
    
    def build_vocabulary(self, all_descriptors, sample_size=100000):
        """Строит словарь визуальных слов с помощью K-means."""
        if len(all_descriptors) == 0:
            print("Ошибка: нет дескрипторов для построения словаря!")
            return None
        
        # Объединяем все дескрипторы
        all_descriptors = np.vstack(all_descriptors)
        
        # Если дескрипторов слишком много, берем выборку
        if len(all_descriptors) > sample_size:
            indices = np.random.choice(len(all_descriptors), sample_size, replace=False)
            all_descriptors = all_descriptors[indices]
        
        print(f"Построение словаря из {len(all_descriptors)} дескрипторов...")
        
        # Используем MiniBatchKMeans для больших наборов данных
        if len(all_descriptors) > 10000:
            self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, 
                                         batch_size=1000, 
                                         random_state=42)
        else:
            self.kmeans = KMeans(n_clusters=self.n_clusters, 
                                random_state=42, 
                                n_init=10)
        
        self.kmeans.fit(all_descriptors)
        print(f"Словарь из {self.n_clusters} визуальных слов построен.")
        return self.kmeans
    
    def image_to_histogram(self, descriptors):
        """Преобразует дескрипторы изображения в гистограмму визуальных слов."""
        if descriptors is None or len(descriptors) == 0:
            # Если нет дескрипторов, возвращаем нулевую гистограмму
            return np.zeros(self.n_clusters)
        
        # Предсказываем метки кластеров для каждого дескриптора
        labels = self.kmeans.predict(descriptors)
        
        # Строим гистограмму
        histogram, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
        
        # Нормализуем гистограмму
        histogram = histogram.astype(np.float32)
        if np.sum(histogram) > 0:
            histogram /= np.sum(histogram)
        
        return histogram
    
    def train(self, train_file, images_dir=".", **kwargs):
        """Обучает модель BOW на предоставленных данных."""
        from sklearn.metrics import accuracy_score
        
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ 'МЕШОК СЛОВ'")
        print(f"Детектор: {self.detector_type}, Кластеров: {self.n_clusters}")
        print("=" * 60)
        
        # Загрузка данных
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        # 1. Извлечение признаков
        print("\n1. Извлечение локальных признаков...")
        all_descriptors = []
        valid_indices = []
        
        for i, image_path in enumerate(train_paths):
            descriptors = self.extract_features(image_path)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
                valid_indices.append(i)
        
        if len(all_descriptors) == 0:
            print("Ошибка: не удалось извлечь признаки ни из одного изображения!")
            return 0
        
        print(f"Извлечено дескрипторов: {sum(len(d) for d in all_descriptors)}")
        
        # 2. Построение словаря визуальных слов
        print("\n2. Построение словаря визуальных слов...")
        self.build_vocabulary(all_descriptors)
        
        # 3. Преобразование изображений в гистограммы
        print("\n3. Преобразование изображений в гистограммы...")
        train_features = []
        train_labels_filtered = []
        train_ids_filtered = []
        
        for idx in valid_indices:
            descriptors = self.extract_features(train_paths[idx])
            histogram = self.image_to_histogram(descriptors)
            train_features.append(histogram)
            train_labels_filtered.append(train_labels[idx])
            train_ids_filtered.append(label_ids[idx])
        
        train_features = np.array(train_features)
        
        # 4. Масштабирование признаков
        print("\n4. Масштабирование признаков...")
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # 5. Обучение классификатора
        print("\n5. Обучение классификатора...")
        
        # Можно выбрать разные классификаторы
        classifier_type = kwargs.get('classifier_type', 'svm')
        
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', 
                                 probability=True, 
                                 random_state=42,
                                 C=1.0)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100,
                                                    random_state=42,
                                                    max_depth=10)
        else:
            print(f"Классификатор {classifier_type} не поддерживается, используется SVM")
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        
        self.classifier.fit(train_features_scaled, train_ids_filtered)
        
        # 6. Оценка на обучающей выборке
        print("\n6. Оценка качества на обучающей выборке...")
        train_predictions = self.classifier.predict(train_features_scaled)
        train_accuracy = accuracy_score(train_ids_filtered, train_predictions)
        
        train_predictions_labels = [self.id_to_label[pred] for pred in train_predictions]
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        print("\nОтчет по классификации (обучающая выборка):")
        print(classification_report(train_labels_filtered, train_predictions_labels))
        
        # Визуализация матрицы ошибок
        if kwargs.get('plot_confusion', False):
            self.plot_confusion_matrix(train_ids_filtered, train_predictions, 
                                      "Матрица ошибок (обучающая выборка)")
        
        return train_accuracy
    
    def test(self, test_file, images_dir=".", **kwargs):
        """Тестирует модель на тестовой выборке."""
        from sklearn.metrics import accuracy_score, classification_report
        
        print("=" * 60)
        print("ТЕСТИРОВАНИЕ МОДЕЛИ 'МЕШОК СЛОВ'")
        print("=" * 60)
        
        # Загрузка данных
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        # Преобразование тестовых изображений в гистограммы
        test_features = []
        valid_indices = []
        
        for i, path in enumerate(test_paths):
            descriptors = self.extract_features(path)
            if descriptors is not None and len(descriptors) > 0:
                histogram = self.image_to_histogram(descriptors)
                test_features.append(histogram)
                valid_indices.append(i)
        
        if len(test_features) == 0:
            print("Ошибка: не удалось извлечь признаки ни из одного тестового изображения!")
            return 0
        
        test_features = np.array(test_features)
        
        # Масштабирование признаков
        test_features_scaled = self.scaler.transform(test_features)
        
        # Предсказание
        test_predictions = self.classifier.predict(test_features_scaled)
        
        # Фильтрация меток для валидных изображений
        test_labels_filtered = [test_labels[i] for i in valid_indices]
        test_ids_filtered = [label_ids[i] for i in valid_indices]
        
        # Оценка качества
        test_accuracy = accuracy_score(test_ids_filtered, test_predictions)
        
        test_predictions_labels = [self.id_to_label[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print("\nДетальный отчет по классификации (тестовая выборка):")
        print(classification_report(test_labels_filtered, test_predictions_labels))
        
        # Визуализация матрицы ошибок
        if kwargs.get('plot_confusion', False):
            self.plot_confusion_matrix(test_ids_filtered, test_predictions, 
                                      "Матрица ошибок (тестовая выборка)")
        
        # Вывод примера предсказаний
        print("\nПримеры предсказаний:")
        for i in range(min(5, len(test_paths))):
            if i in valid_indices:
                idx = valid_indices.index(i)
                true_label = test_labels[i]
                pred_label = test_predictions_labels[idx]
                print(f"  {os.path.basename(test_paths[i])}: Истина - {true_label}, Предсказание - {pred_label}")
        
        return test_accuracy
    
    def predict_single(self, image_path):
        """Предсказывает класс для одного изображения."""
        if self.kmeans is None or self.classifier is None:
            print("Ошибка: модель не обучена!")
            return None, 0
        
        descriptors = self.extract_features(image_path)
        if descriptors is None or len(descriptors) == 0:
            print("Ошибка: не удалось извлечь признаки из изображения")
            return None, 0
        
        histogram = self.image_to_histogram(descriptors)
        histogram_scaled = self.scaler.transform([histogram])
        
        # Получаем предсказание и вероятности
        prediction_id = self.classifier.predict(histogram_scaled)[0]
        probabilities = self.classifier.predict_proba(histogram_scaled)[0]
        
        prediction_label = self.id_to_label[prediction_id]
        confidence = probabilities[prediction_id]
        
        return prediction_label, confidence
    
    def visualize_features(self, image_path, output_path=None):
        """Визуализирует ключевые точки на изображении."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение ключевых точек
        keypoints = self.detector.detect(gray, None)
        
        # Рисуем ключевые точки
        if self.detector_type == 'sift':
            image_with_keypoints = cv2.drawKeypoints(
                image, keypoints, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0)
            )
        else:
            image_with_keypoints = cv2.drawKeypoints(
                image, keypoints, None, 
                color=(0, 255, 0)
            )
        
        print(f"Найдено {len(keypoints)} ключевых точек на изображении {os.path.basename(image_path)}")
        
        # Сохраняем или показываем изображение
        if output_path:
            cv2.imwrite(output_path, image_with_keypoints)
            print(f"Изображение с ключевыми точками сохранено в: {output_path}")
            return image_with_keypoints
        else:
            # Показываем изображение
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
            plt.title(f'Ключевые точки ({self.detector_type.upper()})')
            plt.axis('off')
            plt.show()
            
            # Также показываем оригинальное изображение для сравнения
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Оригинальное изображение')
            plt.axis('off')
            plt.show()
            
        return image_with_keypoints
    
    def save_model(self, model_dir="models"):
        """Сохраняет модель в указанную директорию."""
        model_dir = os.path.join(model_dir, 'bow_model')
        os.makedirs(model_dir, exist_ok=True)
        
        super().save_model(model_dir)
        
        joblib.dump(self.kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
        joblib.dump(self.classifier, os.path.join(model_dir, 'classifier_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Сохраняем параметры детектора
        detector_params = {
            'detector_type': self.detector_type,
            'n_clusters': self.n_clusters
        }
        
        with open(os.path.join(model_dir, 'detector_params.json'), 'w') as f:
            json.dump(detector_params, f, indent=2)
        
        print(f"Модель BOW сохранена в директории: {model_dir}")
    
    def load_model(self, model_dir="models"):
        """Загружает модель из указанной директории."""
        model_dir = os.path.join(model_dir, 'bow_model')
        
        if not super().load_model(model_dir):
            return False
        
        try:
            self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
            self.classifier = joblib.load(os.path.join(model_dir, 'classifier_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            
            # Загружаем параметры детектора
            with open(os.path.join(model_dir, 'detector_params.json'), 'r') as f:
                detector_params = json.load(f)
            
            self.detector_type = detector_params['detector_type']
            self.n_clusters = detector_params['n_clusters']
            
            # Инициализируем детектор
            if self.detector_type.lower() == 'sift':
                self.detector = cv2.SIFT_create()
            elif self.detector_type.lower() == 'orb':
                self.detector = cv2.ORB_create(nfeatures=1000)
            elif self.detector_type.lower() == 'akaze':
                self.detector = cv2.AKAZE_create()
            
            print(f"Модель BOW загружена. Детектор: {self.detector_type}, Кластеров: {self.n_clusters}")
            return True
            
        except FileNotFoundError as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
