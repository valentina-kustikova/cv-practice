# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV
Целью данной работы является разработка классификатора изображений, который будет классифицировать изображения кошек и собак с использованием метода "мешок визуальных слов" (BoVW) и алгоритма Support Vector Machine (SVM). Метод BoVW представляет изображения в виде гистограмм, отражающих распределение визуальных слов, которые извлекаются из изображений с помощью метода SIFT (Scale-Invariant Feature Transform). Затем эти признаки используются для обучения классификатора SVM, который выполняет предсказание категорий изображений.

## Методы
В процессе разработки использовались следующие ключевые методы и алгоритмы:
### 1)  Извлечение признаков с использованием SIFT.
Для извлечения признаков изображений используется метод SIFT. Этот метод позволяет находить ключевые точки на изображении. Для каждой ключевой точки вычисляется дескриптор, который описывает локальную структуру изображения в окрестности этой точки. В коде используется функция cv2.SIFT_create(), которая создает объект SIFT для извлечения ключевых точек и дескрипторов.
### 2) Кластеризация с использованием KMeans.
После извлечения дескрипторов для каждого изображения применяется кластеризация KMeans, которая группирует все дескрипторы в фиксированное количество кластеров (визуальных слов). Кластеризация позволяет создать компактное представление каждого изображения. Изображение представляется как гистограмма, в которой учитывается количество дескрипторов, попавших в каждый кластер. Это преобразование изображений в векторы фиксированной длины — ключевая часть метода BoVW.
### 3) Построение векторов признаков.
После того как изображения были преобразованы в визуальные слова с помощью KMeans, для каждого изображения строится вектор признаков. Этот вектор представляет собой гистограмму, которая показывает, как часто встречаются различные визуальные слова в изображении. Вектор признаков является входом для классификатора.
### 4) Классификация с использованием SVM.
Для классификации изображений используется метод SVM. После того как векторы признаков для обучающих и тестовых данных построены, они нормализуются с помощью StandardScaler, чтобы улучшить обучение модели. Модель обучается на обучающих данных и затем тестируется на тестовых, используя точность классификации как метрику. 
### 5) Оценка качества классификации.
Для оценки качества работы классификатора используются следующие метрики:
- Точность классификации (accuracy) на обучающих и тестовых данных.
- Отчёт о классификации (precision, recall, F1-score) для каждого класса (кошки и собаки).
- Матрица ошибок (confusion matrix), показывающая, как часто классификатор ошибается в предсказаниях.
- График точности по классам, отображающий точность классификации для каждого из классов.
### 6) Визуализация
Для лучшего понимания работы алгоритма реализованы несколько методов визуализации:
- Визуализация ключевых точек SIFT, где показываются найденные ключевые точки на изображениях.
- Гистограмма визуальных слов, показывающая распределение кластеров по данным.
- Комбинированная гистограмма, которая суммирует все гистограммы для изображений кошек и собак, позволяя анализировать, как распределяются визуальные слова для разных классов.

## Классы
### 1) BoVWFeatureExtractor 
Класс для извлечения признаков из изображений с использованием SIFT и построения гистограмм визуальных слов. Он включает методы для:
- Загрузки изображений из папки.
- Извлечения SIFT дескрипторов.
- Обучения модели KMeans.
- Построения гистограмм визуальных слов.
- Визуализации ключевых точек и комбинированных гистограмм.
### 2)BoVWClassifier 
Класс для классификации изображений с использованием алгоритма SVM. Он включает методы для:
- Извлечения признаков с использованием BoVWFeatureExtractor.
- Нормализации признаков.
- Обучения и тестирования модели SVM.
- Оценки качества классификации и визуализации результатов (графики точности, матрица ошибок).

## Результаты
После выполнения программы выводятся следующие результаты:
- Точность классификации на обучающих и тестовых данных.
- Отчёт о классификации для каждого класса.
- Матрица ошибок, показывающая точность предсказаний.
- Графики, отображающие точность классификации и визуализирующие результаты на изображениях.
