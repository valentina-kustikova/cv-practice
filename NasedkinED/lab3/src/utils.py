import os
from glob import glob
from typing import List, Tuple, Dict


def load_data_paths(data_dir: str, train_file: str) -> Tuple[
    List[str], List[int], List[str], List[int], Dict[str, int]]:
    """
    Загружает пути к изображениям и соответствующие метки классов,
    разделяя их на train и test согласно train.txt, с учетом особенностей путей.
    """

    # 0. Определяем классы (названия папок второго уровня вложенности)
    class_to_label = {}
    label_counter = 0

    # Ищем папки датасетов (ExtDataset, NNSUDataset)
    dataset_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if
                    os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

    for dataset_path in dataset_dirs:
        # Ищем папки классов внутри папок датасетов
        for class_name in os.listdir(dataset_path):
            full_class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(full_class_path) and not class_name.startswith('.') and class_name not in class_to_label:
                class_to_label[class_name] = label_counter
                label_counter += 1

    print(f"DEBUG: Найдено {len(class_to_label)} классов: {list(class_to_label.keys())}")

    train_paths = set()

    # 1. Чтение train.txt для получения путей тренировочной выборки
    train_file_path = os.path.join(data_dir, train_file)
    try:
        with open(train_file_path, 'r') as f:
            train_lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл разбиения не найден по пути: {train_file_path}")
        return [], [], [], [], class_to_label

    # Формирование полных и стандартизированных путей тренировочной выборки
    for line in train_lines:
        line = line.strip()
        if not line:
            continue

        # Затем разбиение и сборка пути через os.path.join
        path_parts = line.replace('\\', '/').split('/')

        # Убедимся, что путь начинается с data_dir
        full_path = os.path.abspath(os.path.join(data_dir, *path_parts))

        # Нормализация пути для точного сравнения с glob
        normalized_path = os.path.normpath(full_path)
        train_paths.add(normalized_path)

    train_paths_list = list(train_paths)  # Для сохранения порядка

    # 2. Получение всех путей изображений (используем рекурсивный поиск)
    all_paths = []
    for p in glob(os.path.join(data_dir, '**', '*.*'), recursive=True):
        normalized_path = os.path.normpath(os.path.abspath(p))
        class_name = os.path.basename(os.path.dirname(normalized_path))

        # Фильтруем по известным классам и исключаем служебные файлы
        if class_name in class_to_label and os.path.isfile(p):
            all_paths.append(normalized_path)

    # 3. Формирование тестовой выборки
    test_paths_list = [p for p in all_paths if p not in train_paths]

    # 4. Формирование меток для train и test
    train_labels = [class_to_label[os.path.basename(os.path.dirname(p))] for p in train_paths_list if
                    os.path.basename(os.path.dirname(p)) in class_to_label]
    test_labels = [class_to_label[os.path.basename(os.path.dirname(p))] for p in test_paths_list if
                   os.path.basename(os.path.dirname(p)) in class_to_label]

    return train_paths_list, train_labels, test_paths_list, test_labels, class_to_label


def print_metrics(y_true: List[int], y_pred: List[int], class_labels: Dict[str, int]) -> float:
    """Выводит метрики качества классификации и возвращает точность."""
    from sklearn.metrics import accuracy_score, classification_report

    # Обратное преобразование для красивого вывода
    target_names = [name for name, label in sorted(class_labels.items(), key=lambda item: item[1])]

    # Сопоставление предсказаний и истинных меток
    min_len = min(len(y_true), len(y_pred))
    y_true_matched = y_true[:min_len]
    y_pred_matched = y_pred[:min_len]

    if not y_true_matched:
        print("Недостаточно данных для расчета метрик.")
        return 0.0

    accuracy = accuracy_score(y_true_matched, y_pred_matched)
    print(f"\nТочность классификации (Accuracy): {accuracy:.4f}")

    print("\nОтчет по классификации (Classification Report):")
    print(classification_report(y_true_matched, y_pred_matched, target_names=target_names, zero_division=0))
    return accuracy
