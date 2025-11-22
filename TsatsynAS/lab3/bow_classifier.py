import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class BOWClassifier:
    def __init__(self, vocab_size=100, detector='SIFT'):
        self.vocab_size = vocab_size
        self.detector_name = detector
        self.detector = self._create_detector()
        self.bow_extractor = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.vocabulary = None

    def _create_detector(self):
        try:
            if self.detector_name == 'SIFT':
                return cv2.SIFT_create()
            elif self.detector_name == 'ORB':
                return cv2.ORB_create(nfeatures=1000)
            elif self.detector_name == 'AKAZE':
                return cv2.AKAZE_create()
            else:
                return cv2.SIFT_create()
        except Exception as e:
            print(f"Ошибка создания детектора {self.detector_name}: {e}")
            return cv2.SIFT_create()

    def _create_bow_extractor(self, descriptors_list):
        """Создание настоящего BOW экстрактора OpenCV"""
        print("🎯 Создание BOW словаря с OpenCV...")

        # Создаем BOW trainer
        bow_trainer = cv2.BOWKMeansTrainer(self.vocab_size)

        # Добавляем все дескрипторы в trainer
        total_descriptors = 0
        for descriptors in descriptors_list:
            if descriptors is not None:
                bow_trainer.add(descriptors.astype(np.float32))
                total_descriptors += len(descriptors)

        print(f"📊 Всего дескрипторов для кластеризации: {total_descriptors}")

        # Кластеризация с помощью OpenCV
        print("🔍 Кластеризация дескрипторов OpenCV...")
        self.vocabulary = bow_trainer.cluster()
        print(f"✅ Словарь создан: {self.vocabulary.shape}")

        # Создаем BOW экстрактор
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(
            self.detector,
            cv2.BFMatcher(cv2.NORM_L2)
        )
        self.bow_extractor.setVocabulary(self.vocabulary)

        return self.bow_extractor

    def train(self, images, labels, model_path='bow_model.pkl'):
        if len(images) == 0:
            raise ValueError("Нет изображений для обучения")

        print(f"🎯 Начало обучения BOW на {len(images)} изображениях...")
        print(f"🔧 Детектор: {self.detector_name}, Словарь: {self.vocab_size}")

        # 1. Извлечение дескрипторов OpenCV
        print("1. Извлечение дескрипторов OpenCV...")
        descriptors_list = []
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            descriptors_list.append(descriptors)

            if (i + 1) % 20 == 0:
                kp_count = len(keypoints) if keypoints else 0
                print(f"   Обработано {i + 1}/{len(images)} (точек: {kp_count})")

        # 2. Создание BOW словаря OpenCV
        print("2. Создание BOW словаря OpenCV...")
        self._create_bow_extractor(descriptors_list)

        # 3. Извлечение BOW признаков с OpenCV
        print("3. Извлечение BOW признаков OpenCV...")
        bow_features = []
        valid_indices = []

        for i, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Используем настоящий BOW экстрактор OpenCV
            keypoints = self.detector.detect(gray, None)
            if len(keypoints) > 0:
                bow_descriptor = self.bow_extractor.compute(gray, keypoints)
                if bow_descriptor is not None:
                    bow_features.append(bow_descriptor.flatten())
                    valid_indices.append(i)
                else:
                    bow_features.append(np.zeros(self.vocab_size))
            else:
                bow_features.append(np.zeros(self.vocab_size))

            if (i + 1) % 20 == 0:
                print(f"   Извлечены признаки для {i + 1}/{len(images)}")

        bow_features = np.array(bow_features)
        filtered_labels = [labels[i] for i in range(len(images))]

        print(f"📊 Размерность BOW признаков: {bow_features.shape}")

        # 4. Обучение классификатора
        print("4. Обучение классификатора...")
        bow_features = self.scaler.fit_transform(bow_features)

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.classifier.fit(bow_features, filtered_labels)

        self.is_trained = True
        self.save(model_path)
        print(f"💾 BOW модель сохранена в {model_path}")

    def predict(self, images):
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        print("🔍 Извлечение BOW признаков для предсказания...")
        features = []

        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            keypoints = self.detector.detect(gray, None)
            if len(keypoints) > 0:
                bow_descriptor = self.bow_extractor.compute(gray, keypoints)
                if bow_descriptor is not None:
                    features.append(bow_descriptor.flatten())
                else:
                    features.append(np.zeros(self.vocab_size))
            else:
                features.append(np.zeros(self.vocab_size))

        features = self.scaler.transform(np.array(features))
        return self.classifier.predict(features)

    def save(self, path):
        # Сохраняем только сериализуемые объекты
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'detector_name': self.detector_name,
                'vocabulary': self.vocabulary,
                'classifier': self.classifier,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.vocab_size = data['vocab_size']
        self.detector_name = data['detector_name']
        self.detector = self._create_detector()
        self.vocabulary = data['vocabulary']

        # Воссоздаем BOW экстрактор при загрузке
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(
            self.detector,
            cv2.BFMatcher(cv2.NORM_L2)
        )
        self.bow_extractor.setVocabulary(self.vocabulary)

        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']