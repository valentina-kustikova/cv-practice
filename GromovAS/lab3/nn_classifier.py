import tensorflow as tf
import numpy as np
import pickle
import os
import cv2
from sklearn.metrics import classification_report, accuracy_score


class NNClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, epochs=50, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes = None
        self.label_encoder = {}
        self.label_decoder = {}

    def _build_model(self):
        """Построение модели с transfer learning"""
        # Используем tf.keras напрямую
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_images(self, images, labels):
        """Предобработка изображений"""
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}

        X = []
        y = []

        for image, label in zip(images, labels):
            resized_image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            normalized_image = resized_image.astype('float32') / 255.0
            X.append(normalized_image)
            y.append(self.label_encoder[label])

        return np.array(X), np.array(y)

    def create_data_generator(self, X, y):
        """Создание генератора данных с аугментацией"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        return datagen.flow(X, y, batch_size=self.batch_size)

    def train(self, train_data, model_path):
        """Обучение нейронной сети"""
        images, labels = zip(*train_data)
        X_train, y_train = self.preprocess_images(images, labels)
        self.num_classes = len(self.label_encoder)

        self.model = self._build_model()
        train_generator = self.create_data_generator(X_train, y_train)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        print("Обучение нейронной сети...")
        history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            steps_per_epoch=len(X_train) // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.save_model(model_path)
        print("Модель сохранена!")

    def test(self, test_data, model_path):
        """Тестирование нейронной сети"""
        if not os.path.exists(model_path):
            raise ValueError(f"Модель не найдена: {model_path}")

        self.load_model(model_path)

        images, labels = zip(*test_data)
        X_test, y_test = self.preprocess_images(images, labels)

        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)

        y_test_decoded = [self.label_decoder[label] for label in y_test]
        predictions_decoded = [self.label_decoder[label] for label in predicted_classes]

        accuracy = accuracy_score(y_test_decoded, predictions_decoded)
        report = classification_report(y_test_decoded, predictions_decoded)

        return accuracy, report

    def save_model(self, path):
        """Сохранение модели"""
        model_path = path.replace('.pkl', '_keras.h5')
        self.model.save(model_path)

        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'classes': list(self.label_encoder.keys())
        }

        with open(path, 'wb') as f:
            pickle.dump(model_info, f)

    def load_model(self, path):
        """Загрузка модели"""
        model_path = path.replace('.pkl', '_keras.h5')
        self.model = tf.keras.models.load_model(model_path)

        with open(path, 'rb') as f:
            model_info = pickle.load(f)

        self.input_shape = model_info['input_shape']
        self.num_classes = model_info['num_classes']
        self.label_encoder = model_info['label_encoder']
        self.label_decoder = model_info['label_decoder']