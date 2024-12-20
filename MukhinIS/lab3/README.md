# Классификация изображений с помощью бибилиотеки OpenCV.

Структура лабораторной работы:
1. Для обработки изображений используется программный интерфейс `Data`, внутри которого:
    * `__init__` - конструктор класса, который принимает путь до датасета, размер тренировочной и тестовой выборок,
    после чего происходит перемешивание путей до изображений;
    * `load_images` - метод класса, который загружает и сохраняет в списки изображения для тестовой и тренировочной
    выборок;
2. Для обработки детекторов используется программный интерфейс `Model`, внутри которого:
    * `__init__` - конструктор класса, принимающий количество кластеров, после чего создает детектор `SIFT`;
    * `extract_features` - метод, который применяет детектор на выборке и выделяет из них дескрипторы ключевых точек;
