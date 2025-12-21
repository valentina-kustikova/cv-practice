import json
import os
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from abstract import ClassificationStrategy


# ============================================================================
# Класс ModelEvaluator для оценки качества обученной модели.
# Вычисляет метрики: accuracy, precision, recall, F1-score, confusion matrix.
# Выводит результаты в консоль и сохраняет подробный отчет в JSON-файл.
# Использует sklearn.metrics для расчета метрик классификации.
# ============================================================================

class ModelEvaluator:
    """Класс для оценки качества модели"""

    @staticmethod
    def evaluate(
            strategy: ClassificationStrategy,
            test_data: List[str],
            test_labels: List[str],
            results_path: str = None
    ) -> float:
        """
        Оценить модель на тестовой выборке

        Args:
            strategy: Стратегия классификации
            test_data: Тестовые данные
            test_labels: Тестовые метки
            results_path: Путь для сохранения результатов

        Returns:
            accuracy: Точность классификации
        """
        print("Тестирование классификатора...")
        predictions = []

        for i, img_path in enumerate(test_data):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(test_data)} изображений")
            pred = strategy.predict(img_path)
            predictions.append(pred)

        accuracy = accuracy_score(test_labels, predictions)
        report_dict = classification_report(test_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(test_labels, predictions)

        ModelEvaluator._print_metrics(accuracy, test_labels, predictions, conf_matrix)

        if results_path:
            ModelEvaluator._save_results(
                accuracy, report_dict, conf_matrix,
                test_labels, strategy.get_params(), results_path
            )

        return accuracy

    @staticmethod
    def _print_metrics(accuracy: float, test_labels: List[str],
                       predictions: List[str], conf_matrix: np.ndarray) -> None:
        """Вывести метрики в консоль"""
        print("\n" + "=" * 50)
        print(f"ACCURACY: {accuracy:.4f}")
        print("=" * 50)
        print("\nОтчёт классификации:")
        print(classification_report(test_labels, predictions))
        print("\nМатрица ошибок:")
        print(conf_matrix)

    @staticmethod
    def _save_results(
            accuracy: float,
            report_dict: Dict,
            conf_matrix: np.ndarray,
            test_labels: List[str],
            model_params: Dict[str, Any],
            results_path: str
    ) -> None:
        """Сохранить результаты в JSON"""
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        conf_matrix_dict = {}
        for i, true_class in enumerate(sorted(set(test_labels))):
            conf_matrix_dict[true_class] = {}
            for j, pred_class in enumerate(sorted(set(test_labels))):
                conf_matrix_dict[true_class][pred_class] = int(conf_matrix[i][j])

        results = {
            'summary': {
                'accuracy': round(float(accuracy), 4),
                'total_test_samples': len(test_labels),
                **model_params
            },
            'per_class_metrics': {},
            'confusion_matrix': conf_matrix_dict
        }

        for class_name in sorted(set(test_labels)):
            if class_name in report_dict:
                results['per_class_metrics'][class_name] = {
                    'precision': round(report_dict[class_name]['precision'], 4),
                    'recall': round(report_dict[class_name]['recall'], 4),
                    'f1-score': round(report_dict[class_name]['f1-score'], 4),
                    'support': int(report_dict[class_name]['support'])
                }

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
