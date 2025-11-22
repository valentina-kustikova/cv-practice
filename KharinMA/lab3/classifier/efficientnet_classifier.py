import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from .base_classifier import BaseClassifier

class EfficientNetClassifier(BaseClassifier):
    def __init__(self, model_dir='efficientnet_model', image_size=224,
                 learning_rate=0.001, batch_size=16, epochs=20):
        super().__init__(model_dir)
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []

    def create_model(self, n_classes):
        print("\n=== Создание модели EfficientNetB0 ===")
        # Load EfficientNetB0 with ImageNet weights, exclude top layers
        # EfficientNet expects inputs in range [0, 255]
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.image_size, self.image_size, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(n_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print(f"✓ Модель создана. Backbone заморожен.")

    def _preprocess_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        # Use decode_image to handle both JPEG and PNG
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        # Ensure shape is known for resize (needed in some TF versions/contexts)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.image_size, self.image_size])
        # EfficientNet expects [0, 255]
        return image, label

    def _preprocess_image_test(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image

    def train(self, train_paths, train_labels):
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ EFFICIENTNET КЛАССИФИКАТОРА")
        print("="*60)

        self.class_names = sorted(list(set(train_labels)))
        encoded_labels = self.label_encoder.fit_transform(train_labels)
        
        if self.model is None:
            self.create_model(len(self.class_names))

        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((train_paths, encoded_labels))
        
        # Shuffle and split
        dataset_size = len(train_paths)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        # Apply preprocessing and batching
        train_dataset = train_dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = val_dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"Train size: {train_size}, Val size: {val_size}")
        
        # Train
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs
        )
        
        self.save_model()
        print("Обучение завершено")

    def test(self, test_paths, test_labels=None):
        print("\n" + "="*60)
        print("ТЕСТИРОВАНИЕ EFFICIENTNET КЛАССИФИКАТОРА")
        print("="*60)
        
        if self.model is None:
            self.load_model()
            
        # Create dataset for testing
        dataset = tf.data.Dataset.from_tensor_slices(test_paths)
        dataset = dataset.map(self._preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        
        predictions_probs = self.model.predict(dataset)
        predicted_indices = np.argmax(predictions_probs, axis=1)
        predictions = self.label_encoder.inverse_transform(predicted_indices)
        
        accuracy = None
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predictions)
            print(f"\nТочность классификации: {accuracy:.3f}")
            print("\nОтчет по классификации:")
            print(self.evaluate(test_labels, predictions, self.class_names))
            
        return predictions, accuracy

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        model_path = os.path.join(self.model_dir, 'efficientnet_model.keras')
        self.model.save(model_path)
        
        metadata = {
            'class_names': self.class_names,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ Модель сохранена: {model_path}")

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'efficientnet_model.keras')
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
             raise ValueError("Модель не найдена")
             
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        self.class_names = metadata['class_names']
        self.image_size = metadata['image_size']
        self.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        
        self.model = load_model(model_path)
        print(f"✓ Модель загружена успешно")
