import logging
import argparse
import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import List
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from base_classifier import BaseClassifier

log = logging.getLogger()


class BowClassifier(BaseClassifier):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.bow_model_path = self.model_save_dir / "bow_model.joblib"

        self.detector = None
        self.classifier = None
        self.kmeans = None
        self.scaler = None

    def _init_detector(self):
        detector_type = self.args.detector.lower()
        if detector_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Неподдерживаемый тип детектора: {detector_type}")
        log.info(f"Используется {detector_type.upper()} детектор.")

    def extract_features_bow(self, image_paths: List[str]) -> np.ndarray:
        if self.detector is None:
            self._init_detector()

        descriptors = []
        log.info(f"Извлечение дескрипторов из {len(image_paths)} изображений...")
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)
            if des is not None:
                descriptors.extend(des)

        if not descriptors:
            log.error("Дескрипторы не найдены! Проверьте изображения.")
            return np.array([])

        return np.array(descriptors)

    def build_vocabulary(self, descriptors: np.ndarray):
        k = self.args.k
        log.info(f"Построение словаря (k={k}) из {len(descriptors)} дескрипторов...")

        self.kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            batch_size=1000
        )
        self.kmeans.fit(descriptors)
        log.info("Словарь построен.")

    def get_bow_histograms(self, image_paths: List[str]) -> np.ndarray:
        if self.detector is None:
            self._init_detector()
        if self.kmeans is None:
            raise ValueError("Словарь KMeans (vocabulary) не построен.")

        k = self.kmeans.n_clusters
        histograms = []
        log.info(f"Создание BoW гистограмм для {len(image_paths)} изображений...")

        for img_path in image_paths:
            img = cv2.imread(img_path)
            hist = np.zeros(k)
            if img is None:
                histograms.append(hist)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)

            if des is not None:
                labels = self.kmeans.predict(des)
                for label in labels:
                    hist[label] += 1
            histograms.append(hist)

        return np.array(histograms)

    def train(self):
        log.info("---Начало обучения BoW ---")
        train_paths, train_labels = self.load_data(self.train_file)
        if not train_paths:
            return

        descriptors = self.extract_features_bow(train_paths)
        if descriptors.size == 0:
            return

        self.build_vocabulary(descriptors)

        train_hist = self.get_bow_histograms(train_paths)
        self.scaler = StandardScaler().fit(train_hist)
        train_hist = self.scaler.transform(train_hist)

        log.info(f"Обучение SVM (kernel={self.args.svm_kernel}, C={self.args.svm_c})...")
        self.classifier = SVC(
            kernel=self.args.svm_kernel,
            C=self.args.svm_c,
            random_state=42,
            probability=True
        )
        self.classifier.fit(train_hist, train_labels)
        log.info("Модель BoW обучена.")
        self.save_model()

    def save_model(self):
        if not all([self.kmeans, self.scaler, self.classifier]):
            log.error("Модель BoW не обучена полностью. Сохранение отменено.")
            return

        model_pack = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'svm': self.classifier,
            'detector_type': self.args.detector  # Сохраняем тип детектора
        }
        joblib.dump(model_pack, self.bow_model_path)
        log.info(f"Модель BoW сохранена: {self.bow_model_path}")

    def load_model(self):
        if not self.bow_model_path.exists():
            raise FileNotFoundError(f"Файл модели BoW не найден: {self.bow_model_path}")

        model_pack = joblib.load(self.bow_model_path)
        self.kmeans = model_pack['kmeans']
        self.scaler = model_pack['scaler']
        self.classifier = model_pack['svm']

        # Загружаем тип детектора, с которым модель была обучена
        self.args.detector = model_pack.get('detector_type', self.args.detector)

        self._init_detector()
        log.info(f"Модель BoW загружена из {self.bow_model_path}")

    def test(self):
        log.info("---Начало тестирования BoW ---")
        if not all([self.kmeans, self.scaler, self.classifier]):
            log.warning("BoW модель не загружена. Попытка загрузки...")
            try:
                self.load_model()
            except FileNotFoundError as e:
                log.error(f"Ошибка загрузки: {e}")
                log.error("Сначала обучите модель в режиме 'train'.")
                return

        test_paths, test_labels = self.load_data(self.test_file)
        if not test_paths:
            return

        test_hist = self.get_bow_histograms(test_paths)
        test_hist = self.scaler.transform(test_hist)

        predictions = self.classifier.predict(test_hist)

        acc = accuracy_score(test_labels, predictions)
        report = classification_report(
            test_labels,
            predictions,
            target_names=self.classes,
            zero_division=0
        )
        log.info(f"---BoW Результаты Теста ---\n"
                 f"Accuracy: {acc:.4f}\n"
                 f"{report}\n"
                 f"-----------------------------")

    def visualize_keypoints(self):
        log.info("--- Визуализация ключевых точек ---")
        if self.detector is None:
            self._init_detector()

        image_paths, labels = self.load_data(self.test_file)
        if not image_paths:
            log.error("Нет изображений для визуализации.")
            return

        sample_images = {}
        for img_path, label in zip(image_paths, labels):
            if label not in sample_images:
                sample_images[label] = img_path
            if len(sample_images) == len(self.classes):
                break

        vis_dir = Path("keypoint_visualizations")
        vis_dir.mkdir(exist_ok=True)
        log.info(f"Сохранение визуализаций в ./{vis_dir}")

        for label, img_path in sample_images.items():
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)

            img_with_kp = cv2.drawKeypoints(
                img, kp, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            class_name = self.classes[label]
            save_path = vis_dir / f"{class_name}_{self.args.detector}_keypoints.jpg"
            cv2.imwrite(str(save_path), img_with_kp)

            log.info(f"Сохранено: {save_path}")
