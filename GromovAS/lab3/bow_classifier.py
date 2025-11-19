import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import MiniBatchKMeans
import os


class BOWClassifier:
    def __init__(self, vocab_size=1000, detector='SIFT', classifier_type='svm'):
        self.vocab_size = vocab_size
        self.detector_type = detector
        self.classifier_type = classifier_type
        self.detector = None
        self.kmeans = None
        self.classifier = None
        self.classes = None

    def _init_detector(self):
        # Инициализация детектора и дескриптора
        if self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif self.detector_type == 'ORB':
            self.detector = cv2.ORB_create()
        elif self.detector_type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Неподдерживаемый детектор: {self.detector_type}")

    def extract_features(self, image):
        # Извлечение особенностей из изображения
        if self.detector is None:
            self._init_detector()

        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return descriptors

    def build_vocabulary(self, images, labels):
        # Построение визуального словаря
        print("Построение словаря...")
        all_descriptors = []

        for image, label in zip(images, labels):
            descriptors = self.extract_features(image)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # Объединение всех дескрипторов
        all_descriptors = np.vstack(all_descriptors)

        # Кластеризация для создания словаря
        self.kmeans = MiniBatchKMeans(n_clusters=self.vocab_size,
                                      random_state=42, batch_size=1000)
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")

    def image_to_bow(self, image):
        # Преобразование изображения в вектор bow
        if self.kmeans is None:
            raise ValueError("Словарь не построен!")

        descriptors = self.extract_features(image)
        if descriptors is None:
            return np.zeros(self.vocab_size)

        # Назначение дескрипторов кластерам
        visual_words = self.kmeans.predict(descriptors)

        # Построение гистограммы
        bow_vector, _ = np.histogram(visual_words,
                                     bins=self.vocab_size,
                                     range=(0, self.vocab_size - 1))
        return bow_vector

    def train(self, train_data, model_path):
        # Обучение классификатора
        images, labels = zip(*train_data)
        self.classes = list(set(labels))

        # Построение словаря
        self.build_vocabulary(images, labels)

        # Преобразование тренировочных данных в bow
        print("Преобразование тренировочных данных...")
        X_train = []
        y_train = []

        for image, label in train_data:
            bow_vector = self.image_to_bow(image)
            X_train.append(bow_vector)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Обучение классификатора
        if self.classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        elif self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        print("Обучение классификатора...")
        self.classifier.fit(X_train, y_train)

        # Сохранение модели
        self.save_model(model_path)
        print("Модель сохранена!")

    def test(self, test_data, model_path):
        # Тестирование классификатора
        if not os.path.exists(model_path):
            raise ValueError(f"Модель не найдена: {model_path}")

        self.load_model(model_path)

        images, labels = zip(*test_data)

        # Преобразование тестовых данных
        X_test = []
        y_test = []

        for image, label in test_data:
            bow_vector = self.image_to_bow(image)
            X_test.append(bow_vector)
            y_test.append(label)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Предсказание
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        return accuracy, report

    def save_model(self, path):
        # Сохранение модели
        model_data = {
            'vocab_size': self.vocab_size,
            'detector_type': self.detector_type,
            'classifier_type': self.classifier_type,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'classes': self.classes
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path):
        # Загрузка модели
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.vocab_size = model_data['vocab_size']
        self.detector_type = model_data['detector_type']
        self.classifier_type = model_data['classifier_type']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.classes = model_data['classes']
        self._init_detector()


def visualize_training_samples(self, train_data, num_samples=3, output_dir='./bow_visualization'):
    # Визуализация ключевых точек для примеров обучения
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    print(f"Визуализация {num_samples} примеров обучения...")

    samples = train_data[:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    if num_samples == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, (image, label) in enumerate(samples):
        # Оригинальное изображение
        axes[0][i].imshow(image)
        axes[0][i].set_title(f'Original: {label}')
        axes[0][i].axis('off')

        # Ключевые точки
        keypoints = self.detector.detect(image)
        image_with_kp = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        axes[1][i].imshow(image_with_kp)
        axes[1][i].set_title(f'Keypoints: {len(keypoints)}')
        axes[1][i].axis('off')

        # Сохранение отдельного изображения
        kp_path = os.path.join(output_dir, f'sample_{i}_{label}.jpg')
        plt.imsave(kp_path, image_with_kp)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'training_samples_comparison.jpg')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Визуализация сохранена в: {output_dir}")