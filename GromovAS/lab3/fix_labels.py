import os


def add_labels_to_split(input_file, output_file):
    # Добавляет метки классов к файлам разбиения
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            line = line.strip()
            if line:
                # Определяем класс по пути
                if '01_NizhnyNovgorodKremlin' in line:
                    label = 'kremlin'
                elif '04_ArkhangelskCathedral' in line:
                    label = 'cathedral'
                elif '08_PalaceOfLabor' in line:
                    label = 'palace'
                else:
                    print(f"Неизвестный класс для: {line}")
                    continue

                # Записываем строку с меткой
                f_out.write(f"{line} {label}\n")

    print(f"Обработан {input_file} -> {output_file}")
    print(f"Добавлено {len(lines)} записей")


def main():
    # Создаем исправленные версии файлов
    add_labels_to_split('data/train.txt', 'data/train_fixed.txt')
    add_labels_to_split('data/test.txt', 'data/test_fixed.txt')

    print("\nФайлы исправлены!")
    print("Используйте:")
    print(
        "python main.py --data_dir ./data --train_split ./data/train_fixed.txt --test_split ./data/test_fixed.txt --mode both --algorithm bow")


if __name__ == "__main__":
    main()