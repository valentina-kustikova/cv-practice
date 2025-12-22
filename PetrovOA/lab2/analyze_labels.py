"""
Скрипт для анализа файла разметки labels.txt
Выводит список всех уникальных классов объектов
"""

def analyze_labels(labels_path: str) -> dict:
    """
    Читает файл разметки и возвращает статистику по классам.
    
    Формат файла: <frame_id> <class_name> <x1> <y1> <x2> <y2>
    """
    classes_count = {}
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_name = parts[1]
                classes_count[class_name] = classes_count.get(class_name, 0) + 1
    
    return classes_count


if __name__ == "__main__":
    labels_path = "labels.txt"
    
    classes = analyze_labels(labels_path)
    
    print("=" * 40)
    print("Анализ файла разметки labels.txt")
    print("=" * 40)
    print(f"\nВсего уникальных классов: {len(classes)}")
    print("\nСписок классов и количество объектов:")
    print("-" * 40)
    
    for class_name, count in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count} объектов")
    
    print("-" * 40)
    print(f"Всего объектов: {sum(classes.values())}")
