"""
Реализация сверточной нейронной сети с Transfer Learning
Использует предобученные модели (VGG16, MobileNetV2, ResNet50)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from base_classifier import BaseClassifier

class CNNClassifier(BaseClassifier):
    """Классификатор на основе сверточной нейронной сети с Transfer Learning."""
    
    def __init__(self, image_size=(224, 224), batch_size=16, epochs=20, 
                 learning_rate=0.001, dropout_rate=0.5, base_model_name='vgg16'):
        super().__init__('cnn', image_size)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.base_model_name = base_model_name
        self.model = None
        self.history = None
    
    def create_cnn_model(self):
        """Создает модель CNN на основе предобученной сети."""
        print(f"Создание модели на основе {self.base_model_name.upper()}...")
        
        input_shape = (self.image_size[0], self.image_size[1], 3)
        
        # Используем VGG16 как базовую модель
        weights_path = './weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if os.path.exists(weights_path):
            base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
            base_model.load_weights(weights_path)
        else:
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Замораживаем базовые слои
        base_model.trainable = False
        
        # Добавляем собственные слои
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate / 2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Компиляция модели
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def preprocess_image(self, image_path):
        """Предобработка изображения для CNN."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype('float32') / 255.0
        return image
    
    def prepare_dataset(self, image_paths, label_ids):
        """Подготавливает датасет для обучения/тестирования."""
        X, y = [], []
        
        for path, label_id in zip(image_paths, label_ids):
            image = self.preprocess_image(path)
            if image is not None:
                X.append(image)
                y.append(label_id)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.class_names))
        return X, y
    
    def train(self, train_file, images_dir):
        """Обучение CNN модели."""
        print("=" * 60)
        print("ОБУЧЕНИЕ СВЕРТОЧНОЙ НЕЙРОННОЙ СЕТИ")
        print(f"Базовая модель: {self.base_model_name.upper()}")
        print("=" * 60)
        
        # Загрузка данных
        train_paths, train_labels, train_ids = self.load_data(train_file, images_dir)
        
        # Подготовка данных
        X_train, y_train = self.prepare_dataset(train_paths, train_ids)
        print(f"Размер обучающей выборки: {X_train.shape}")
        
        # Создание модели
        self.model = self.create_cnn_model()
        
        # Callbacks для улучшения обучения
        callbacks = [
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Обучение модели
        print("\nНачало обучения...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка на обучающей выборке
        print("\nОценка качества...")
        train_predictions = np.argmax(self.model.predict(X_train), axis=1)
        train_predictions_labels = [self.id_to_label[pred] for pred in train_predictions]
        
        accuracy = self.evaluate(train_labels, train_predictions_labels, "обучающей выборке")

        self.plot_confusion_matrix(train_ids, train_predictions, 
                                  "Матрица ошибок CNN (обучающая выборка)")

        return accuracy
    
    def test(self, test_file, images_dir):
        """Тестирование CNN модели."""
        print("=" * 60)
        print("ТЕСТИРОВАНИЕ СВЕРТОЧНОЙ НЕЙРОННОЙ СЕТИ")
        print("=" * 60)
        
        # Загрузка данных
        test_paths, test_labels, test_ids = self.load_data(test_file, images_dir)
        
        # Подготовка данных
        X_test, y_test = self.prepare_dataset(test_paths, test_ids)
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        # Предсказание
        print("\nВыполнение предсказаний...")
        test_predictions_proba = self.model.predict(X_test, verbose=0)
        test_predictions = np.argmax(test_predictions_proba, axis=1)
        test_predictions_labels = [self.id_to_label[pred] for pred in test_predictions]
        
        # Оценка качества
        accuracy = self.evaluate(test_labels, test_predictions_labels, "тестовой выборке")
        
        # Матрица ошибок
        self.plot_confusion_matrix(test_ids, test_predictions, 
                                  "Матрица ошибок CNN (тестовая выборка)")
        
        return accuracy
    
    def predict_single(self, image_path):
        """Предсказание класса для одного изображения."""
        image = self.preprocess_image(image_path)
        image_batch = np.expand_dims(image, axis=0)
        
        prediction_proba = self.model.predict(image_batch, verbose=0)[0]
        prediction_id = np.argmax(prediction_proba)
        
        prediction_label = self.id_to_label[prediction_id]
        confidence = prediction_proba[prediction_id]
        
        return prediction_label, confidence
    
    def plot_training_history(self):
        """Визуализация истории обучения."""
        if self.history is None:
            print("История обучения отсутствует")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # График точности
        axes[0].plot(self.history.history['accuracy'], label='Точность')
        axes[0].set_title('Точность модели')
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)
        
        # График потерь
        axes[1].plot(self.history.history['loss'], label='Потери')
        axes[1].set_title('Потери модели')
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Потери')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_dir):
        """Сохраняет модель CNN."""
        model_dir = os.path.join(model_dir, 'cnn_model')
        super().save_model(model_dir)
        
        self.model.save(os.path.join(model_dir, 'cnn_model.h5'))
        
        if self.history is not None:
            np.save(os.path.join(model_dir, 'training_history.npy'), self.history.history)
    
    def load_model(self, model_dir):
        """Загружает модель CNN."""
        model_dir = os.path.join(model_dir, 'cnn_model')
        super().load_model(model_dir)
        
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
        
        history_path = os.path.join(model_dir, 'training_history.npy')
        if os.path.exists(history_path):
            history_dict = np.load(history_path, allow_pickle=True).item()
            self.history = type('History', (), {'history': history_dict})()
        
        return True
