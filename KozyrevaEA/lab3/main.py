import argparse
import os
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bovw_classification.log"),
        logging.StreamHandler()
    ]
)


class ImageLoader:
    """Класс для загрузки и предобработки изображений из указанной директории"""

    def __init__(self, img_size: tuple[int, int] = (256, 256)) -> None:
        """Инициализация загрузчика изображений"""

        self.img_size: tuple[int, int] = img_size


    def load_images(self, dataset_path: str):
        """Загружает изображения из указанной директории, преобразует их в градации серого и изменяет размер"""

        images: list = []
        labels: list[int] = []

        for img_name in os.listdir(dataset_path):
            img_path: str = os.path.join(dataset_path, img_name)

            if img_name.startswith("cat."):
                label: int = 1
            elif img_name.startswith("dog."):
                label: int = 0
            else:
                logging.warning(f"Файл {img_name} не соответствует формату. Пропуск.")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(label)
            else:
                logging.warning(f"Не удалось загрузить изображение: {img_path}")

        return np.array(images), np.array(labels)


class FeatureExtractor:
    """Класс для извлечения ключевых точек и дескрипторов из изображений"""

    def __init__(self, feature_type: str = 'sift') -> None:
        """Инициализация экстрактора признаков"""

        if feature_type == 'sift':
            self.extractor: cv2.SIFT = cv2.SIFT_create()
        elif feature_type == 'orb':
            self.extractor: cv2.ORB = cv2.ORB_create()
        else:
            raise ValueError("Unsupported feature type. Choose 'sift' or 'orb'.")
    

    def extract_features(self, images: list[np.ndarray]):
        """Извлекает ключевые точки и дескрипторы из списка изображений"""

        keypoints_list: list = []  # список ключевых точек
        descriptors_list: list = []  # список дескрипторов
        
        for img in images:
            keypoints, descriptors = self.extractor.detectAndCompute(img, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors if descriptors is not None else None)
        
        return keypoints_list, descriptors_list


class KMeansTrainer:
    """Класс для обучения модели KMeans и построения гистограмм признаков"""

    def __init__(self, num_clusters: int) -> None:
        """Инициализация KMeansTrainer"""

        self.num_clusters: int = num_clusters
        self.kmeans = None


    def train(self, descriptors_list: list) -> None:
        """Обучает модель KMeans на переданных дескрипторах"""

        logging.info("Обучение KMeans")
        valid_descriptors = [desc for desc in descriptors_list if desc is not None]
        if not valid_descriptors:
            raise ValueError("Нет доступных дескрипторов для обучения KMeans.")
        
        all_descriptors = np.vstack(valid_descriptors)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)


    def build_histogram(self, descriptors) -> np.ndarray:
        """Строит гистограмму признаков на основе обученной модели KMeans"""

        if self.kmeans is None:
            raise ValueError("Модель KMeans не обучена.")
        
        histogram = np.zeros(self.num_clusters)
        if descriptors is not None:
            predictions = self.kmeans.predict(descriptors)
            for pred in predictions:
                histogram[pred] += 1
        
        return histogram

class Classifier:
    """Класс для обучения и предсказания с использованием различных моделей классификации."""

    def __init__(self, classifier_type = 'svm') -> None:
        """Инициализация классификатора"""

        self.scaler: StandardScaler = StandardScaler()
        self.model = None

        if classifier_type == 'svm':
            self.model = SVC(kernel='rbf', C=1, random_state=42)
        elif classifier_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif classifier_type == 'log_reg':
            self.model = LogisticRegression(max_iter=100, C=1, random_state=42)
        else:
            raise ValueError("Unsupported classifier type. Choose 'svm', 'random_forest', or 'log_reg'.")


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Обучает модель классификации"""

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Выполняет предсказание классов"""

        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

class Visualizer:
    @staticmethod
    def plot_accuracy_histogram(true_labels, predicted_labels, class_labels):
        confusion_mat = confusion_matrix(true_labels, predicted_labels)
        class_accuracies = np.diag(confusion_mat) / confusion_mat.sum(axis=1)

        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, class_accuracies, color='#88CFFD', edgecolor='black')
        plt.xlabel('Классы')
        plt.ylabel('Точность')
        plt.title('Точность классификации по классам')
        plt.ylim(0, 1)

        for index, accuracy in enumerate(class_accuracies):
            plt.text(index, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontsize=12, color='black')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_keypoints(image, keypoints):
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title("Keypoints")
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Матрица ошибок')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')

        plt.ylabel('Истина')
        plt.xlabel('Предсказание')
        plt.show()


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', '--train',
                        help='Path to train data',
                        type=str,
                        dest='train_path',
                        required=True)
    parser.add_argument('-test', '--test',
                        help='Path to test data',
                        type=str,
                        dest='test_path',
                        required=True)
    parser.add_argument('-cc', '--clusters',
                        help='Count of clusters',
                        type=int,
                        dest='count_clusters',
                        required=True)
    parser.add_argument('-alg', '--algorithm',
                        help='Feature extraction algorithm (sift or orb)',
                        type=str,
                        dest='feature_type',
                        choices=['sift', 'orb'],
                        default='sift')
    parser.add_argument('-clf', '--classifier',
                        help='Classifier to use (svm or random_forest or log_reg)',
                        type=str,
                        dest='classifier_type',
                        choices=['svm', 'random_forest', 'log_reg'],
                        default='svm')

    args = parser.parse_args()
    return args


def evaluate_classifier(y_true, y_pred, classifier_name):
    logging.info(f"Evaluating {classifier_name}...")
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Dog", "Cat"], output_dict=True)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]

    logging.info(f"{classifier_name} Metrics:")
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Precision (weighted avg): {precision:.4f}")
    logging.info(f"Recall (weighted avg): {recall:.4f}")
    logging.info(f"F1-score (weighted avg): {f1_score:.4f}")

    print(f"{classifier_name} Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (weighted avg): {precision:.4f}")
    print(f"Recall (weighted avg): {recall:.4f}")
    print(f"F1-score (weighted avg): {f1_score:.4f}")


def main():
    args = arg_parse()

    logging.info("Загрузка тренировочных данных...")
    image_loader = ImageLoader()
    X_train, y_train = image_loader.load_images(args.train_path)
    X_test, y_test = image_loader.load_images(args.test_path)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    logging.info("Извлечение признаков из тренировочных данных...")
    feature_extractor = FeatureExtractor(feature_type=args.feature_type)
    train_keypoints, train_descriptors = feature_extractor.extract_features(X_train)

    # Визуализация ключевых точек
    indices = np.random.randint(0, len(X_test) - 1, size=3)
    for idx in indices:
        Visualizer.visualize_keypoints(X_train[idx], train_keypoints[idx])

    logging.info("Обучение модели KMeans...")
    kmeans_trainer = KMeansTrainer(num_clusters=args.count_clusters)
    kmeans_trainer.train(train_descriptors)

    logging.info("Создание гистограмм для тренировочных данных...")
    X_train_bow = np.array([kmeans_trainer.build_histogram(desc) for desc in train_descriptors])

    logging.info("Извлечение признаков из тестовых данных...")
    _, test_descriptors = feature_extractor.extract_features(X_test)
    X_test_bow = np.array([kmeans_trainer.build_histogram(desc) for desc in test_descriptors])
    
    logging.info("Обучение классификатора...")
    classifier = Classifier(classifier_type=args.classifier_type)
    classifier.train(X_train_bow, y_train)


    logging.info("Предсказание на тестовых данных...")
    y_pred = classifier.predict(X_test_bow)

  
    logging.info("Вывод отчета о классификации...")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Dog", "Cat"]))


    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")


    logging.info("Визуализация точности по классам...")
    Visualizer.plot_accuracy_histogram(y_test, y_pred, ['Dogs', 'Cats'])


    logging.info("Визуализация матрицы ошибок...")
    Visualizer.plot_confusion_matrix(y_test, y_pred, class_names=["Dog", "Cat"])


if __name__ == '__main__':
    sys.exit(main() or 0)