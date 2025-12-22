import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from .base_classifier import BaseClassifier

class CNNClassifier(BaseClassifier):
    """Классификатор на основе сверточной нейронной сети с Transfer Learning."""
    
    def __init__(self, image_size=(224, 224), batch_size=16, epochs=20, 
                 learning_rate=0.001, dropout_rate=0.5, 
                 base_model_name='vgg16', class_names=None):
        super().__init__('cnn', image_size, class_names)
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
        
        # Выбор базовой модели
        input_shape = (self.image_size[0], self.image_size[1], 3)
        
        if self.base_model_name.lower() == 'vgg16':
            # Попытка загрузить локальные веса
            weights_path = os.path.expanduser('./weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
            if os.path.exists(weights_path):
                print("Загрузка весов VGG16 из локального файла...")
                base_model = VGG16(weights=None, 
                                  include_top=False, 
                                  input_shape=input_shape)
                base_model.load_weights(weights_path)
            else:
                print("Загрузка предобученных весов VGG16 из интернета...")
                base_model = VGG16(weights='imagenet', 
                                  include_top=False, 
                                  input_shape=input_shape)
                
        elif self.base_model_name.lower() == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet',
                                    include_top=False,
                                    input_shape=input_shape)
        elif self.base_model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet',
                                 include_top=False,
                                 input_shape=input_shape)
        else:
            print(f"Базовая модель {self.base_model_name} не поддерживается, используется VGG16")
            base_model = VGG16(weights='imagenet', 
                              include_top=False, 
                              input_shape=input_shape)
        
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
            layers.Dropout(self.dropout_rate/2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Компиляция модели
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.summary()
        return model
    
    def create_augmentation_generator(self):
        """Создает генератор для аугментации данных."""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def preprocess_image_cnn(self, image_path):
        """Предобрабатывает изображение для CNN."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
            
        # Конвертация в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменение размера
        image = cv2.resize(image, self.image_size)
        
        # Нормализация
        image = image.astype('float32') / 255.0
        
        return image
    
    def prepare_dataset(self, image_paths, label_ids, augmentation=False):
        """Подготавливает датасет для обучения/тестирования."""
        X = []
        y = []
        
        for path, label_id in zip(image_paths, label_ids):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X.append(image)
                y.append(label_id)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.class_names))
        
        return X, y
    
    def train(self, train_file, images_dir=".", **kwargs):
        """Обучает CNN модель на предоставленных данных."""
        from sklearn.metrics import accuracy_score
        
        print("=" * 60)
        print("ОБУЧЕНИЕ СВЕРТОЧНОЙ НЕЙРОННОЙ СЕТИ")
        print(f"Базовая модель: {self.base_model_name.upper()}")
        print(f"Размер изображения: {self.image_size}")
        print("=" * 60)
        
        # Загрузка данных
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        # Подготовка данных
        print("\nПодготовка данных для обучения...")
        X_train, y_train = self.prepare_dataset(train_paths, label_ids)
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        
        # Создание модели
        self.model = self.create_cnn_model()
        
        # Callbacks
        callbacks = []
        
        early_stopping = EarlyStopping(
            monitor='val_loss' if kwargs.get('validation_split', 0) > 0 else 'loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if kwargs.get('validation_split', 0) > 0 else 'loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Сохранение лучшей модели
        if kwargs.get('save_best', False):
            checkpoint = ModelCheckpoint(
                'best_cnn_model.h5',
                monitor='val_accuracy' if kwargs.get('validation_split', 0) > 0 else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Обучение модели
        print("\nНачало обучения...")
        
        validation_split = kwargs.get('validation_split', 0.2)
        
        if validation_split > 0:
            # Используем валидационное разделение
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Без валидационного разделения
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        # Оценка на обучающей выборке
        print("\nОценка качества на обучающей выборке...")
        train_predictions = np.argmax(self.model.predict(X_train), axis=1)
        train_true_labels = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        train_predictions_labels = [self.id_to_label[pred] for pred in train_predictions]
        train_true_labels_names = [self.id_to_label[label] for label in train_true_labels]
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        
        from sklearn.metrics import classification_report
        print("\nОтчет по классификации (обучающая выборка):")
        print(classification_report(train_true_labels_names, train_predictions_labels))
        
        # Визуализация истории обучения
        if self.history is not None and kwargs.get('plot_history', False):
            self.plot_training_history()
        
        return train_accuracy
    
    def test(self, test_file, images_dir=".", **kwargs):
        """Тестирует CNN модель на тестовой выборке."""
        from sklearn.metrics import accuracy_score, classification_report
        
        print("=" * 60)
        print("ТЕСТИРОВАНИЕ СВЕРТОЧНОЙ НЕЙРОННОЙ СЕТИ")
        print("=" * 60)
        
        # Загрузка данных
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        # Подготовка данных
        X_test, y_test = self.prepare_dataset(test_paths, label_ids)
        
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        # Предсказание
        print("\nВыполнение предсказаний...")
        test_predictions_proba = self.model.predict(X_test, verbose=0)
        test_predictions = np.argmax(test_predictions_proba, axis=1)
        
        # Оценка качества
        test_accuracy = accuracy_score(label_ids, test_predictions)
        
        test_predictions_labels = [self.id_to_label[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print("\nДетальный отчет по классификации (тестовая выборка):")
        print(classification_report(test_labels, test_predictions_labels))
        
        # Визуализация матрицы ошибок
        if kwargs.get('plot_confusion', False):
            self.plot_confusion_matrix(label_ids, test_predictions, 
                                      "Матрица ошибок CNN (тестовая выборка)")
        
        # Вывод примера предсказаний с уверенностью
        print("\nПримеры предсказаний с уверенностью:")
        for i in range(min(5, len(test_paths))):
            true_label = test_labels[i]
            pred_label = test_predictions_labels[i]
            confidence = test_predictions_proba[i][test_predictions[i]]
            print(f"  {os.path.basename(test_paths[i])}:")
            print(f"    Истина: {true_label}, Предсказание: {pred_label} (уверенность: {confidence:.2%})")
        
        return test_accuracy
    
    def predict_single(self, image_path):
        """Предсказывает класс для одного изображения."""
        if self.model is None:
            print("Ошибка: модель не обучена!")
            return None, 0
        
        image = self.preprocess_image_cnn(image_path)
        if image is None:
            return None, 0
        
        # Добавляем размерность батча
        image_batch = np.expand_dims(image, axis=0)
        
        # Предсказание
        prediction_proba = self.model.predict(image_batch, verbose=0)[0]
        prediction_id = np.argmax(prediction_proba)
        
        prediction_label = self.id_to_label[prediction_id]
        confidence = prediction_proba[prediction_id]
        
        return prediction_label, confidence
    
    def plot_training_history(self):
        """Визуализирует историю обучения модели."""
        if self.history is None:
            print("История обучения отсутствует")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # График точности
        axes[0].plot(self.history.history['accuracy'], label='Точность (обучение)')
        if 'val_accuracy' in self.history.history:
            axes[0].plot(self.history.history['val_accuracy'], label='Точность (валидация)')
        axes[0].set_title('Точность модели')
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)
        
        # График потерь
        axes[1].plot(self.history.history['loss'], label='Потери (обучение)')
        if 'val_loss' in self.history.history:
            axes[1].plot(self.history.history['val_loss'], label='Потери (валидация)')
        axes[1].set_title('Потери модели')
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Потери')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_dir="models"):
        """Сохраняет модель в указанную директорию."""
        model_dir = os.path.join(model_dir, 'cnn_model')
        os.makedirs(model_dir, exist_ok=True)
        
        super().save_model(model_dir)
        
        # Сохраняем модель Keras
        model_path = os.path.join(model_dir, 'cnn_model.h5')
        self.model.save(model_path)
        
        # Сохраняем историю обучения
        if self.history is not None:
            history_path = os.path.join(model_dir, 'training_history.npy')
            np.save(history_path, self.history.history)
        
        print(f"Модель CNN сохранена в директории: {model_dir}")
    
    def load_model(self, model_dir="models"):
        """Загружает модель из указанной директории."""
        model_dir = os.path.join(model_dir, 'cnn_model')
        
        if not super().load_model(model_dir):
            return False
        
        try:
            # Загружаем модель Keras
            model_path = os.path.join(model_dir, 'cnn_model.h5')
            self.model = tf.keras.models.load_model(model_path)
            
            # Загружаем историю обучения
            history_path = os.path.join(model_dir, 'training_history.npy')
            if os.path.exists(history_path):
                history_dict = np.load(history_path, allow_pickle=True).item()
                self.history = type('History', (), {'history': history_dict})()
            
            print(f"Модель CNN загружена. Базовая модель: {self.base_model_name}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки модели CNN: {e}")
            return False
