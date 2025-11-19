import argparse
import os
from typing import List

from abstract import ClassificationStrategy
from bow_strategy import BagOfWordsStrategy
from dataset_loader import DatasetLoader
from evaluator import ModelEvaluator
from feature_extractor import OpenCVFeatureExtractor
from nn_strategy import NeuralNetworkStrategy


# ============================================================================
# Главный модуль приложения. Содержит:
# - LandmarkClassifier: фасад для удобного использования стратегий классификации
# - StrategyFactory: фабрика для создания конкретных стратегий (паттерн Factory)
# - main(): обработка аргументов командной строки, координация обучения/тестирования
# Поддерживает режимы train, test, train_test для алгоритмов bow и neural.
# ============================================================================

# ============================================================================
# Фасад классификатора
# ============================================================================

class LandmarkClassifier:
    """Фасад для классификации достопримечательностей"""

    def __init__(self, strategy: ClassificationStrategy):
        """
        Args:
            strategy: Стратегия классификации
        """
        self.strategy = strategy

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучить модель"""
        self.strategy.train(train_data, train_labels)

    def predict(self, image_path: str) -> str:
        """Предсказать класс изображения"""
        return self.strategy.predict(image_path)

    def evaluate(self, test_data: List[str], test_labels: List[str],
                 results_path: str = None) -> float:
        """Оценить модель"""
        return ModelEvaluator.evaluate(
            self.strategy, test_data, test_labels, results_path
        )

    def save_model(self, filepath: str) -> None:
        """Сохранить модель"""
        self.strategy.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Загрузить модель"""
        self.strategy.load(filepath)


# ============================================================================
# Фабрика стратегий
# ============================================================================

class StrategyFactory:
    """Фабрика для создания стратегий классификации"""

    @staticmethod
    def create_strategy(algorithm: str, **kwargs) -> ClassificationStrategy:
        """
        Создать стратегию классификации

        Args:
            algorithm: Тип алгоритма ('bow' или 'neural')
            **kwargs: Дополнительные параметры

        Returns:
            Стратегия классификации
        """
        if algorithm == 'bow':
            detector_type = kwargs.get('detector_type', 'sift')
            n_clusters = kwargs.get('n_clusters', 300)

            feature_extractor = OpenCVFeatureExtractor(detector_type)
            return BagOfWordsStrategy(feature_extractor, n_clusters)

        elif algorithm == 'neural':
            model_architecture = kwargs.get('model_architecture', 'resnet')
            return NeuralNetworkStrategy(model_architecture)

        else:
            raise ValueError(f"Неизвестный алгоритм: {algorithm}")


# ============================================================================
# Main функция
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Классификатор достопримечательностей Нижнего Новгорода'
    )

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Путь до директории с данными')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Файл разбиения (по умолчанию data_dir/train.txt)')
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test'],
                        help='Режим работы')
    parser.add_argument('--algorithm', type=str, default='bow',
                        choices=['bow', 'neural'],
                        help='Алгоритм классификации')
    parser.add_argument('--detector', type=str, default='sift',
                        choices=['sift', 'orb', 'akaze'],
                        help='Тип детектора признаков (для BoW)')
    parser.add_argument('--n_clusters', type=int, default=300,
                        help='Количество кластеров (для BoW)')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Директория для сохранения моделей')

    args = parser.parse_args()

    # Определение путей
    if args.split_file is None:
        args.split_file = os.path.join(args.data_dir, 'train.txt')

    model_subdir = os.path.join(args.models_dir, args.algorithm)

    if args.algorithm == 'bow':
        model_name = f"{args.detector}_clusters{args.n_clusters}.pkl"
        results_name = f"{args.detector}_clusters{args.n_clusters}_results.json"
    else:
        model_name = "neural_model.pkl"
        results_name = "neural_results.json"

    model_path = os.path.join(model_subdir, model_name)
    results_path = os.path.join(model_subdir, results_name)

    # Загрузка данных
    dataset_loader = DatasetLoader(args.data_dir, args.split_file)
    train_data, train_labels, test_data, test_labels = dataset_loader.load()

    # Создание стратегии и классификатора
    strategy = StrategyFactory.create_strategy(
        args.algorithm,
        detector_type=args.detector,
        n_clusters=args.n_clusters
    )

    classifier = LandmarkClassifier(strategy)

    # Обучение
    if args.mode in ['train', 'train_test']:
        classifier.train(train_data, train_labels)
        classifier.save_model(model_path)

    # Тестирование
    if args.mode in ['test', 'train_test']:
        if args.mode == 'test':
            classifier.load_model(model_path)

        accuracy = classifier.evaluate(test_data, test_labels, results_path)

        print(f"\n{'=' * 50}")
        print(f"Модель сохранена: {model_path}")
        print(f"Результаты сохранены: {results_path}")
        print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
