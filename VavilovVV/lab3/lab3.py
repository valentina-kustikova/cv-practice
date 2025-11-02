import os
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

class ImageDataset(Dataset):

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"Не удалось прочитать (возможно, битый файл): {img_path}")
                return self.__getitem__(0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            if self.transform:
                img_tensor = self.transform(img_pil)

            return img_tensor, label

        except Exception as e:
            log.error(f"Ошибка при загрузке {img_path}: {e}")
            return self.__getitem__(0)


class ImageClassifier:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.train_file = Path(args.train_file)
        self.test_file = Path(args.test_file)

        self.classes = args.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.model_save_dir = Path(args.model_dir)
        self.model_save_dir.mkdir(exist_ok=True)

        self.bow_model_path = self.model_save_dir / "bow_model.joblib"
        self.nn_model_path = self.model_save_dir / "nn_model.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Используется устройство: {self.device}")

        self.detector = None
        self.classifier = None
        self.kmeans = None
        self.scaler = None
        self.model = None

    def load_data(self, file_path: Path) -> Tuple[List[str], List[int]]:

        image_paths = []
        labels = []

        if not file_path.exists():
            log.error(f"Файл не найден: {file_path}")
            return image_paths, labels

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip().replace('\\', '/')
                if not line:
                    continue

                parts = line.split('/')
                if len(parts) < 2:
                    log.warning(f"Строка {line_num}: Неверный путь: {line}")
                    continue

                # parts[1] - имя класса
                class_name = parts[1]

                img_path = self.data_dir / line

                if not img_path.exists():
                    log.warning(f"Строка {line_num}: Файл не найден: {img_path}")
                    continue

                if class_name not in self.class_to_idx:
                    log.warning(f"Строка {line_num}: Неизвестный класс '{class_name}'.")
                    continue

                label = self.class_to_idx[class_name]
                image_paths.append(str(img_path))
                labels.append(label)

        log.info(f"Загружено {len(image_paths)} ссылок на изображения из {file_path}")
        return image_paths, labels

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

    def train_bow(self):
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
        self.save_bow_model()

    def save_bow_model(self):
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

    def load_bow_model(self):
        if not self.bow_model_path.exists():
            raise FileNotFoundError(f"Файл модели BoW не найден: {self.bow_model_path}")

        model_pack = joblib.load(self.bow_model_path)
        self.kmeans = model_pack['kmeans']
        self.scaler = model_pack['scaler']
        self.classifier = model_pack['svm']

        self.args.detector = model_pack.get('detector_type', self.args.detector)

        self._init_detector()
        log.info(f"Модель BoW загружена из {self.bow_model_path}")

    def test_bow(self):
        log.info("---Начало тестирования BoW ---")
        if not all([self.kmeans, self.scaler, self.classifier]):
            log.warning("BoW модель не загружена. Попытка загрузки...")
            self.load_bow_model()

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

    def prepare_transforms(self, train: bool = True) -> transforms.Compose:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img_size = self.args.img_size

        if train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _init_nn_model(self):
        self.model = mobilenet_v2(pretrained=self.args.pretrained)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(self.device)
        log.info(f"Архитектура MobileNetV2 инициализирована (pretrained={self.args.pretrained}).")

    def train_nn(self):
        log.info("---Начало обучения NN ---")
        self._init_nn_model()

        train_paths, train_labels = self.load_data(self.train_file)
        if not train_paths:
            return

        train_transform = self.prepare_transforms(train=True)
        train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        num_epochs = self.args.epochs
        log.info(f"Старт обучения NN на {num_epochs} эпох...")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            log.info(f"Эпоха {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        log.info("Модель NN обучена.")
        self.save_nn_model()

    def save_nn_model(self):
        if self.model is None:
            log.error("NN модель не инициализирована. Сохранение отменено.")
            return

        torch.save(self.model.state_dict(), self.nn_model_path)
        log.info(f"Модель NN сохранена: {self.nn_model_path}")

    def load_nn_model(self):
        if not self.nn_model_path.exists():
            raise FileNotFoundError(f"Файл модели NN не найден: {self.nn_model_path}")

        self._init_nn_model()
        self.model.load_state_dict(
            torch.load(self.nn_model_path, map_location=self.device)
        )
        self.model.to(self.device)
        log.info(f"Модель NN загружена из {self.nn_model_path}")

    def test_nn(self):
        log.info("--- Начало тестирования NN ---")
        if self.model is None:
            log.warning("NN модель не загружена. Попытка загрузки...")
            self.load_nn_model()

        test_paths, test_labels = self.load_data(self.test_file)
        if not test_paths:
            return

        test_transform = self.prepare_transforms(train=False)
        test_dataset = ImageDataset(test_paths, test_labels, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.classes,
            zero_division=0
        )
        log.info(f"--- NN Результаты Теста ---\n"
                 f"Accuracy: {acc:.4f}\n"
                 f"{report}\n"
                 f"---------------------------")

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

    def run(self):

        modes = self.args.mode.split(',')

        try:
            if 'train' in modes:
                if self.args.algorithm == 'bow':
                    self.train_bow()
                elif self.args.algorithm == 'nn':
                    self.train_nn()

            if 'test' in modes:
                if self.args.algorithm == 'bow':
                    self.test_bow()
                elif self.args.algorithm == 'nn':
                    self.test_nn()

            if 'visualize' in modes:
                self.visualize_keypoints()

        except FileNotFoundError as e:
            log.error(f"Ошибка: {e}")
            log.error("Файл модели не найден. Запустите 'train' режим для создания модели.")
        except Exception as e:
            log.exception(f"Произошла непредвиденная ошибка: {e}")

def main():
    parser = argparse.ArgumentParser(description="Классификатор изображений (BoW/NN)")

    parser.add_argument("--data-dir", type=str, required=True,
                        help="Корневая папка с данными (где лежат train.txt/test.txt и папки с фото)")
    parser.add_argument("--train-file", type=str, required=True,
                        help="Полный путь к train.txt")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Полный путь к test.txt")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Папка для сохранения моделей")

    # --- Режимы работы ---
    parser.add_argument("--mode", type=str, required=True,
                        help="Режимы работы, через запятую (e.g., 'train', 'test', 'train,test', 'visualize')")
    parser.add_argument("--algorithm", type=str, required=True, choices=['bow', 'nn'],
                        help="Алгоритм для использования ('bow' или 'nn')")

    # --- Общие параметры ---
    parser.add_argument("--classes", nargs='+',
                        default=['01_NizhnyNovgorodKremlin', '04_ArkhangelskCathedral', '08_PalaceOfLabor'],
                        help="Список имен классов")

    # --- BoW параметры ---
    parser.add_argument("--k", type=int, default=500,
                        help="[BoW] Размер словаря (кол-во кластеров KMeans)")
    parser.add_argument("--detector", type=str, default="sift", choices=['sift', 'orb'],
                        help="[BoW] Детектор ключевых точек")
    parser.add_argument("--svm-kernel", type=str, default="linear",
                        help="[BoW] Ядро SVM (linear, rbf, poly)")
    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="[BoW] Параметр регуляризации SVM")

    # --- NN параметры ---
    parser.add_argument("--epochs", type=int, default=20,
                        help="[NN] Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="[NN] Скорость обучения (learning rate)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[NN] Размер батча")
    parser.add_argument("--img-size", type=int, default=224,
                        help="[NN] Размер изображения (img_size x img_size)")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="[NN] Не использовать предобученные веса ImageNet")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2,
                        help="[NN] Количество потоков для загрузки данных")

    args = parser.parse_args()

    args.train_file = str(Path(args.data_dir) / args.train_file)
    args.test_file = str(Path(args.data_dir) / args.test_file)

    classifier = ImageClassifier(args)
    classifier.run()

if __name__ == "__main__":
    main()
