import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import pickle


class NNClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_indices = None
        self.is_trained = False

    def _create_model(self):
        """Создание модели с использованием transfer learning"""
        # Базовая модель MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Замораживаем базовые слои
        base_model.trainable = False

        # Добавляем свои слои
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _preprocess_with_opencv(self, images):
        """Предобработка изображений с использованием OpenCV"""
        processed_images = []

        for img in images:
            # Основная предобработка OpenCV
            img_resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

            # Конвертация BGR to RGB (OpenCV загружает как BGR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Нормализация для нейросети
            img_normalized = img_rgb.astype('float32') / 255.0

            processed_images.append(img_normalized)

        return np.array(processed_images)

    def _augment_with_opencv(self, image):
        """Аугментация данных с использованием OpenCV (опционально)"""
        augmented = []

        # Оригинальное изображение
        augmented.append(image)

        # 1. Горизонтальное отражение (OpenCV)
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)

        # 2. Небольшое размытие (OpenCV)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        augmented.append(blurred)

        return augmented

    def train(self, images, labels, model_path='nn_model.h5', epochs=30, use_augmentation=True):
        if len(images) == 0:
            raise ValueError("Нет изображений для обучения")

        print("🔧 Использование OpenCV для предобработки изображений...")

        # Преобразование меток
        unique_labels = list(set(labels))
        self.class_indices = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        print(f"Классы: {self.class_indices}")

        # Аугментация с OpenCV (опционально)
        if use_augmentation and len(images) < 200:  # Аугментируем только если мало данных
            augmented_images = []
            augmented_labels = []
            for img, label in zip(images, labels):
                aug_imgs = self._augment_with_opencv(img)
                augmented_images.extend(aug_imgs)
                augmented_labels.extend([label] * len(aug_imgs))

            images = images + augmented_images
            labels = labels + augmented_labels
            print(f"После аугментации OpenCV: {len(images)} изображений")

        y_numeric = np.array([self.class_indices[label] for label in labels])

        # Предобработка с OpenCV
        X_processed = self._preprocess_with_opencv(images)

        print(f"Форма данных после OpenCV обработки: {X_processed.shape}")

        # Создание модели
        self.model = self._create_model()

        print("🎯 Обучение нейронной сети...")

        # Data augmentation в TensorFlow
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        history = self.model.fit(
            datagen.flow(X_processed, y_numeric, batch_size=16, subset='training'),
            epochs=epochs,
            validation_data=datagen.flow(X_processed, y_numeric, batch_size=16, subset='validation'),
            verbose=1
        )

        # Fine-tuning
        print("🔧 Fine-tuning...")
        self.model.layers[0].trainable = True
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_fine = self.model.fit(
            datagen.flow(X_processed, y_numeric, batch_size=16, subset='training'),
            epochs=10,
            validation_data=datagen.flow(X_processed, y_numeric, batch_size=16, subset='validation'),
            verbose=1
        )

        self.is_trained = True
        self.save(model_path)
        print(f"💾 Модель сохранена в {model_path}")

        return history

    def predict(self, images):
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        # Предобработка с OpenCV
        X_processed = self._preprocess_with_opencv(images)

        predictions = self.model.predict(X_processed, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)

        # Обратное преобразование индексов в метки
        index_to_class = {v: k for k, v in self.class_indices.items()}
        return [index_to_class[idx] for idx in predicted_indices]

    def demonstrate_opencv_usage(self, image):
        """Демонстрация использования OpenCV"""
        print("🎯 Демонстрация OpenCV в нейросетевом классификаторе:")
        print(f"   - Загрузка изображения: {image.shape}")

        # Покажем различные операции OpenCV
        resized = cv2.resize(image, (224, 224))
        print(f"   - Изменение размера: {resized.shape}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"   - Конвертация в grayscale: {gray.shape}")

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        print(f"   - Размытие Гаусса: {blurred.shape}")

        print("✅ OpenCV используется для предобработки изображений!")

    def save(self, path):
        self.model.save(path)
        with open(path.replace('.h5', '_classes.pkl'), 'wb') as f:
            pickle.dump(self.class_indices, f)

    def load(self, path):
        self.model = keras.models.load_model(path)
        with open(path.replace('.h5', '_classes.pkl'), 'rb') as f:
            self.class_indices = pickle.load(f)
        self.is_trained = True