import argparse
import json
import os
import pickle

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC


class LandmarkClassifier:
    """Классификатор достопримечательностей Нижнего Новгорода"""

    def __init__(self, algorithm='bow', detector_type='sift', n_clusters=300):
        """
        Инициализация классификатора

        Args:
            algorithm: 'bow' или 'neural' (пока только bow реализован)
            detector_type: 'sift', 'orb', 'akaze'
            n_clusters: количество кластеров для словаря визуальных слов
        """
        self.algorithm = algorithm
        self.detector_type = detector_type
        self.n_clusters = n_clusters

        # Инициализация детектора и дескриптора
        self.detector = self._init_detector()

        # Модели (заполняются при обучении)
        self.kmeans = None
        self.classifier = None
        self.class_names = None

        # Маппинг названий папок на классы
        self.folder_to_class = {
            '01_NizhnyNovgorodKremlin': 'kremlin',
            '04_ArkhangelskCathedral': 'cathedral',
            '08_PalaceOfLabor': 'palace'
        }

    def _init_detector(self):
        """Инициализация детектора признаков"""
        if self.detector_type == 'sift':
            return cv2.SIFT_create()
        elif self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Неизвестный детектор: {self.detector_type}")

    def extract_features(self, image_path):
        """Извлечение признаков из изображения"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None

        # Преобразуем в градации серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Детектируем ключевые точки и вычисляем дескрипторы
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return descriptors

    def load_dataset(self, data_dir, split_file):
        """
        Загрузка датасета

        Args:
            data_dir: путь к директории с данными (где лежат ExtDataset, NNSUDataset, train.txt)
            split_file: файл с перечнем файлов для тренировочной выборки

        Returns:
            train_data: список путей к тренировочным изображениям
            train_labels: список меток классов
            test_data: список путей к тестовым изображениям
            test_labels: список меток классов
        """
        # Читаем файл разбиения
        with open(split_file, 'r', encoding='utf-8') as f:
            train_files = [line.strip() for line in f if line.strip()]

        train_files_set = set(train_files)

        # Папки с датасетами
        dataset_folders = ['ExtDataset', 'NNSUDataset']

        # Собираем все изображения
        all_images = []

        for dataset_folder in dataset_folders:
            dataset_path = os.path.join(data_dir, dataset_folder)

            if not os.path.exists(dataset_path):
                print(f"Предупреждение: директория {dataset_path} не найдена")
                continue

            # Проходим по папкам с классами
            for class_folder in os.listdir(dataset_path):
                class_folder_path = os.path.join(dataset_path, class_folder)

                if not os.path.isdir(class_folder_path):
                    continue

                # Определяем класс по названию папки
                if class_folder not in self.folder_to_class:
                    print(f"Предупреждение: неизвестная папка {class_folder}, пропускаем")
                    continue

                class_name = self.folder_to_class[class_folder]

                # Собираем изображения из этой папки
                for img_file in os.listdir(class_folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # Полный путь к файлу
                        full_path = os.path.join(class_folder_path, img_file)

                        # Относительный путь для сравнения с train.txt
                        # Формат: ExtDataset/01_NizhnyNovgorodKremlin/img.jpg
                        rel_path = os.path.join(dataset_folder, class_folder, img_file)

                        all_images.append((full_path, class_name, rel_path))

        # Разделяем на train и test
        train_data, train_labels = [], []
        test_data, test_labels = [], []

        for full_path, class_name, rel_path in all_images:
            if rel_path in train_files_set:
                train_data.append(full_path)
                train_labels.append(class_name)
            else:
                test_data.append(full_path)
                test_labels.append(class_name)

        self.class_names = sorted(list(set(train_labels + test_labels)))

        print(f"Загружено {len(train_data)} тренировочных и {len(test_data)} тестовых изображений")
        print(f"Классы: {self.class_names}")
        print(f"Распределение тренировочных данных:")
        for class_name in self.class_names:
            count = train_labels.count(class_name)
            print(f"  {class_name}: {count}")
        print(f"Распределение тестовых данных:")
        for class_name in self.class_names:
            count = test_labels.count(class_name)
            print(f"  {class_name}: {count}")

        return train_data, train_labels, test_data, test_labels

    def build_vocabulary(self, image_paths):
        """
        Построение словаря визуальных слов (Bag of Words)

        Args:
            image_paths: список путей к изображениям
        """
        print("Извлечение дескрипторов из тренировочных изображений...")
        all_descriptors = []

        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(image_paths)} изображений")

            descriptors = self.extract_features(img_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # Объединяем все дескрипторы
        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {len(all_descriptors)}")

        # Кластеризация для построения словаря
        print(f"Построение словаря из {self.n_clusters} визуальных слов...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42,
                                      batch_size=1000, verbose=1)
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")

    def get_bow_features(self, image_path):
        """
        Получение BoW-представления изображения

        Args:
            image_path: путь к изображению

        Returns:
            histogram: гистограмма визуальных слов
        """
        descriptors = self.extract_features(image_path)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # Предсказываем метки кластеров для каждого дескриптора
        labels = self.kmeans.predict(descriptors)

        # Строим гистограмму
        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1

        # Нормализация
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)

        return histogram

    def train(self, train_data, train_labels):
        """
        Обучение классификатора

        Args:
            train_data: список путей к тренировочным изображениям
            train_labels: список меток классов
        """
        if self.algorithm == 'bow':
            # Построение словаря
            self.build_vocabulary(train_data)

            # Получение BoW-признаков для всех изображений
            print("Извлечение BoW-признаков...")
            X_train = []
            for i, img_path in enumerate(train_data):
                if i % 10 == 0:
                    print(f"Обработано {i}/{len(train_data)} изображений")
                bow_features = self.get_bow_features(img_path)
                X_train.append(bow_features)

            X_train = np.array(X_train)

            # Обучение SVM классификатора
            print("Обучение SVM классификатора...")
            self.classifier = SVC(kernel='rbf', C=10, gamma='scale', verbose=True)
            self.classifier.fit(X_train, train_labels)
            print("Обучение завершено!")

        elif self.algorithm == 'neural':
            # TODO: Реализация нейросетевого подхода
            raise NotImplementedError("Нейросетевой подход пока не реализован")

    def predict(self, image_path):
        """
        Предсказание класса для изображения

        Args:
            image_path: путь к изображению

        Returns:
            predicted_class: предсказанный класс
        """
        if self.algorithm == 'bow':
            bow_features = self.get_bow_features(image_path)
            prediction = self.classifier.predict([bow_features])[0]
            return prediction
        elif self.algorithm == 'neural':
            raise NotImplementedError("Нейросетевой подход пока не реализован")

    def test(self, test_data, test_labels, results_path=None):
        """
        Тестирование классификатора

        Args:
            test_data: список путей к тестовым изображениям
            test_labels: список меток классов
            results_path: путь для сохранения результатов (если None, не сохраняет)

        Returns:
            accuracy: точность классификации
        """
        print("Тестирование классификатора...")
        predictions = []

        for i, img_path in enumerate(test_data):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(test_data)} изображений")
            pred = self.predict(img_path)
            predictions.append(pred)

        # Вычисление метрик
        accuracy = accuracy_score(test_labels, predictions)
        report_dict = classification_report(test_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(test_labels, predictions)

        print("\n" + "=" * 50)
        print(f"ACCURACY: {accuracy:.4f}")
        print("=" * 50)
        print("\nОтчёт классификации:")
        print(classification_report(test_labels, predictions))
        print("\nМатрица ошибок:")
        print(conf_matrix)

        # Сохранение результатов
        if results_path:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            # Формируем читабельную матрицу ошибок
            conf_matrix_dict = {}
            for i, true_class in enumerate(sorted(set(test_labels))):
                conf_matrix_dict[true_class] = {}
                for j, pred_class in enumerate(sorted(set(test_labels))):
                    conf_matrix_dict[true_class][pred_class] = int(conf_matrix[i][j])

            # Формируем структурированные результаты
            results = {
                'summary': {
                    'accuracy': round(float(accuracy), 4),
                    'algorithm': self.algorithm,
                    'detector': self.detector_type,
                    'n_clusters': self.n_clusters,
                    'total_test_samples': len(test_labels)
                },
                'per_class_metrics': {},
                'confusion_matrix': conf_matrix_dict
            }

            # Добавляем метрики по каждому классу
            for class_name in sorted(set(test_labels)):
                if class_name in report_dict:
                    results['per_class_metrics'][class_name] = {
                        'precision': round(report_dict[class_name]['precision'], 4),
                        'recall': round(report_dict[class_name]['recall'], 4),
                        'f1-score': round(report_dict[class_name]['f1-score'], 4),
                        'support': int(report_dict[class_name]['support'])
                    }

            # Добавляем общие метрики
            results['overall_metrics'] = {
                'macro_avg': {
                    'precision': round(report_dict['macro avg']['precision'], 4),
                    'recall': round(report_dict['macro avg']['recall'], 4),
                    'f1-score': round(report_dict['macro avg']['f1-score'], 4)
                },
                'weighted_avg': {
                    'precision': round(report_dict['weighted avg']['precision'], 4),
                    'recall': round(report_dict['weighted avg']['recall'], 4),
                    'f1-score': round(report_dict['weighted avg']['f1-score'], 4)
                }
            }

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"\nРезультаты сохранены в {results_path}")

        return accuracy

    def save_model(self, filepath):
        """Сохранение модели"""
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'algorithm': self.algorithm,
            'detector_type': self.detector_type,
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'class_names': self.class_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath):
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.algorithm = model_data['algorithm']
        self.detector_type = model_data['detector_type']
        self.n_clusters = model_data['n_clusters']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.class_names = model_data['class_names']

        # Переинициализируем детектор
        self.detector = self._init_detector()
        print(f"Модель загружена из {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Классификатор достопримечательностей Нижнего Новгорода')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Путь до директории с данными (где лежат ExtDataset, NNSUDataset, train.txt)')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Файл разбиения на тренировочную и тестовую выборки (по умолчанию data_dir/train.txt)')
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test'],
                        help='Режим работы: train, test или train_test')
    parser.add_argument('--algorithm', type=str, default='bow',
                        choices=['bow', 'neural'],
                        help='Алгоритм: bow (мешок слов) или neural (нейросеть)')
    parser.add_argument('--detector', type=str, default='sift',
                        choices=['sift', 'orb', 'akaze'],
                        help='Тип детектора признаков')
    parser.add_argument('--n_clusters', type=int, default=300,
                        help='Количество кластеров для BoW')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Директория для сохранения моделей')

    args = parser.parse_args()

    # Если split_file не указан, используем train.txt из data_dir
    if args.split_file is None:
        args.split_file = os.path.join(args.data_dir, 'train.txt')

    # Определяем пути для сохранения в зависимости от алгоритма
    if args.algorithm == 'bow':
        model_subdir = os.path.join(args.models_dir, 'bow')
    else:  # neural
        model_subdir = os.path.join(args.models_dir, 'nn')

    # Создаем имя модели с параметрами
    model_name = f"{args.detector}_clusters{args.n_clusters}.pkl"
    model_path = os.path.join(model_subdir, model_name)
    results_path = os.path.join(model_subdir, f"{args.detector}_clusters{args.n_clusters}_results.json")

    # Создание классификатора
    clf = LandmarkClassifier(
        algorithm=args.algorithm,
        detector_type=args.detector,
        n_clusters=args.n_clusters
    )

    # Загрузка данных
    train_data, train_labels, test_data, test_labels = clf.load_dataset(
        args.data_dir, args.split_file
    )

    # Обучение
    if args.mode in ['train', 'train_test']:
        clf.train(train_data, train_labels)
        clf.save_model(model_path)

    # Тестирование
    if args.mode in ['test', 'train_test']:
        if args.mode == 'test':
            clf.load_model(model_path)
        accuracy = clf.test(test_data, test_labels, results_path)

        print(f"\n{'=' * 50}")
        print(f"Модель сохранена: {model_path}")
        print(f"Результаты сохранены: {results_path}")
        print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
