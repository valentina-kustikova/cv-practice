import os
import cv2
import numpy as np
import argparse
import pickle
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min


class ImageClassifier:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    def load_images(self, directory):
        images = []
        labels = []
        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                if 'cat' in filename:
                    labels.append(0)
                elif 'dog' in filename:
                    labels.append(1)
        return images, labels

    # Извлечение дескрипторов SIFT
    def extract_sift_features(self, images):
        sift = cv2.SIFT_create()
        descriptors_list = []
        for img in images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
            else:
                descriptors_list.append(np.array([]))
        return descriptors_list

    # Создание визуального словаря
    def create_visual_dictionary(self, descriptors_list):
        all_descriptors = np.vstack(descriptors_list)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    # Преобразование изображений в гистограммы признаков
    def compute_bow_histograms(self, descriptors_list):
        histograms = []
        for descriptors in descriptors_list:
            if descriptors.size > 0:
                labels = self.kmeans.predict(descriptors)
                hist, _ = np.histogram(labels, bins=range(self.n_clusters + 1), density=True)
                histograms.append(hist)
            else:
                histograms.append(np.zeros(self.n_clusters))
        return np.array(histograms)

    def save_model(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump((self.kmeans, self.scaler, self.rf_classifier, self.n_clusters), f)
        print(f"Модель сохранена в {model_file}")

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            self.kmeans, self.scaler, self.rf_classifier, self.n_clusters = pickle.load(f)

    # Обучение модели
    def train(self, train_dir, model_file):
        train_images, train_labels = self.load_images(train_dir)
        descriptors_list = self.extract_sift_features(train_images)
        self.create_visual_dictionary(descriptors_list)
        train_histograms = self.compute_bow_histograms(descriptors_list)

        train_histograms_scaled = self.scaler.fit_transform(train_histograms)

        X_train, X_val, y_train, y_val = train_test_split(train_histograms_scaled, train_labels, test_size=0.2, random_state=42)

        self.rf_classifier.fit(X_train, y_train)

        val_accuracy = self.rf_classifier.score(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        self.save_model(model_file)

    # Тестирование модели
    def test(self, test_dir, model_file, labels_file):
        self.load_model(model_file)

        test_images, _ = self.load_images(test_dir)
        true_labels = self.load_true_labels(labels_file)

        test_descriptors_list = self.extract_sift_features(test_images)
        test_histograms = self.compute_bow_histograms(test_descriptors_list)
        test_histograms_scaled = self.scaler.transform(test_histograms)

        probabilities = self.rf_classifier.predict_proba(test_histograms_scaled)
        predictions = np.argmax(probabilities, axis=1)

        accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        print(f"Test accuracy: {accuracy:.4f}")

        self.plot_confusion_matrix(true_labels, predictions)
        self.plot_roc_curve(true_labels, probabilities)
        self.plot_probability_histogram(probabilities)
        self.print_most_confident_incorrect_predictions(true_labels, predictions, probabilities)
        self.plot_classification_results(true_labels, predictions)

    # Загрузка правильных меток
    def load_true_labels(self, labels_file):
        true_labels = []
        with open(labels_file, 'r') as f:
            for line in f:
                _, label = line.strip().split()
                true_labels.append(int(label))
        return true_labels

    # Матрица ошибок
    def plot_confusion_matrix(self, true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
        disp.plot(cmap='viridis')
        plt.show()

    # ROC-кривая
    def plot_roc_curve(self, true_labels, probabilities):
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # Гистограмма вероятностей
    def plot_probability_histogram(self, probabilities):
        plt.hist(probabilities[:, 1], bins=20, alpha=0.7, label='Dog')
        plt.hist(probabilities[:, 0], bins=20, alpha=0.7, label='Cat')
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Probabilities")
        plt.legend()
        plt.show()

    # Вывод самых уверенных ошибок
    def print_most_confident_incorrect_predictions(self, true_labels, predictions, probabilities):
        confidence = np.max(probabilities, axis=1)
        incorrect_indices = np.where(np.array(true_labels) != predictions)[0]
        confident_errors = sorted(incorrect_indices, key=lambda i: confidence[i], reverse=True)

        print("\nMost confident incorrect predictions:")
        for i in confident_errors[:5]:  # Покажем топ-5
            print(f"Image {i + 1}: True Label = {true_labels[i]}, Predicted = {predictions[i]}, Confidence = {confidence[i]:.4f}")

    # Гистограмма правильных и неправильных классификаций
    def plot_classification_results(self, true_labels, predictions):
        cat_correct = np.sum((predictions == 0) & (np.array(true_labels) == 0))
        dog_correct = np.sum((predictions == 1) & (np.array(true_labels) == 1))
        incorrect = np.sum(np.array(true_labels) != np.array(predictions))

        labels = ['Correct Cats', 'Correct Dogs', 'Incorrect']
        counts = [cat_correct, dog_correct, incorrect]

        plt.bar(labels, counts, color=['blue', 'green', 'red'])
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Classification Results: Correct and Incorrect Predictions")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification using Random Forest and Bag of Words.")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or test")

    train_parser = subparsers.add_parser("train", help="Train the random forest model")
    train_parser.add_argument("--train_dir", type=str, required=True, help="Directory with training images (cats and dogs)")
    train_parser.add_argument("--model_file", type=str, required=True, help="File to save the trained model")
    train_parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for visual dictionary")

    test_parser = subparsers.add_parser("test", help="Test the random forest model")
    test_parser.add_argument("--test_dir", type=str, required=True, help="Directory with test images")
    test_parser.add_argument("--model_file", type=str, required=True, help="File with the trained model")
    test_parser.add_argument("--labels_file", type=str, required=True, help="File with true labels for test images")

    args = parser.parse_args()

    if args.mode == "train":
        classifier = ImageClassifier(n_clusters=args.n_clusters)
        classifier.train(args.train_dir, args.model_file)
    elif args.mode == "test":
        classifier = ImageClassifier(n_clusters=100)
        classifier.test(args.test_dir, args.model_file, args.labels_file)
    else:
        parser.print_help()

