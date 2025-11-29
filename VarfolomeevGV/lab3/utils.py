"""
Вспомогательные функции для визуализации и оценки результатов.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI для совместимости
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Dict, List, Optional
import os


def print_metrics(results: Dict, class_names: Optional[List[str]] = None):
    """
    Вывод метрик классификации.
    
    Args:
        results: Словарь с метриками (accuracy, precision, recall, f1_score, confusion_matrix)
        class_names: Список имен классов
    """
    print("\n" + "="*50)
    print("МЕТРИКИ КЛАССИФИКАЦИИ")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("="*50)
    
    if class_names and 'true_labels' in results and 'predictions' in results:
        print("\nДетальная классификация по классам:")
        print(classification_report(
            results['true_labels'],
            results['predictions'],
            target_names=class_names,
            zero_division=0
        ))


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[str] = None, title: str = "Confusion Matrix"):
    """
    Визуализация матрицы ошибок.
    
    Args:
        cm: Матрица ошибок
        class_names: Список имен классов
        save_path: Путь для сохранения изображения (опционально)
        title: Заголовок графика
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Матрица ошибок сохранена в {save_path}")
    
    plt.close()


def save_results(results: Dict, filepath: str, algorithm_name: str = ""):
    """
    Сохранение результатов в текстовый файл.
    
    Args:
        results: Словарь с метриками
        filepath: Путь к файлу для сохранения
        algorithm_name: Название алгоритма
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Результаты классификации")
        if algorithm_name:
            f.write(f" - {algorithm_name}\n")
        else:
            f.write("\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n\n")
        
        if 'confusion_matrix' in results:
            f.write("Матрица ошибок:\n")
            cm = results['confusion_matrix']
            for row in cm:
                f.write(" ".join([str(x) for x in row]) + "\n")
    
    print(f"Результаты сохранены в {filepath}")


def plot_training_history(history, save_path: Optional[str] = None):
    """
    Визуализация истории обучения нейронной сети.
    
    Args:
        history: История обучения из Keras
        save_path: Путь для сохранения изображения (опционально)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # График потерь
    axes[1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График истории обучения сохранен в {save_path}")
    
    plt.close()


def compare_algorithms(results_bow: Dict, results_nn: Dict, 
                      save_path: Optional[str] = None):
    """
    Сравнение результатов двух алгоритмов.
    
    Args:
        results_bow: Результаты алгоритма "мешок слов"
        results_nn: Результаты нейросетевого классификатора
        save_path: Путь для сохранения изображения (опционально)
    """
    algorithms = ['Мешок слов', 'Нейронная сеть']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bow_values = [results_bow[m] for m in metrics]
    nn_values = [results_nn[m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bow_values, width, label='Мешок слов', alpha=0.8)
    bars2 = ax.bar(x + width/2, nn_values, width, label='Нейронная сеть', alpha=0.8)
    
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение алгоритмов классификации')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения сохранен в {save_path}")
    
    plt.close()

