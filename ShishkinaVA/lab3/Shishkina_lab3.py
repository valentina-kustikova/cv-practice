import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import argparse
from sklearn.utils import shuffle

class ImageLoader:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def load_images_from_folder(self, folder, label):
        images = []
        labels = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = self.denoise_image(img)
                img = cv2.resize(img, self.image_size)
                images.append(img)
                labels.append(label)
        return images, labels
    def denoise_image(self, img):
        # Применение медианной фильтрации для удаления шумов
        denoised_img = cv2.medianBlur(img, 3)
        return denoised_img

class FeatureExtractor:
    def __init__(self, k=100, descriptor='sift'):
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        if descriptor == 'sift':
            self.detector = cv2.SIFT_create()
        elif descriptor == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    def extract_features(self, images):
        descriptors_list = []
        keypoints_counts = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                keypoints_counts.append(len(keypoints))
        self.average_keypoints = np.mean(keypoints_counts) if keypoints_counts else 0
        return descriptors_list

    def build_histogram(self, descriptors):
        histogram = np.zeros(self.k)
        if descriptors is not None:
            predictions = self.kmeans.predict(descriptors)
            for pred in predictions:
                histogram[pred] += 1
        return histogram

    def build_feature_vectors(self, images):
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            histogram = self.build_histogram(descriptors)
            features.append(histogram)
        return np.array(features)

    def fit_kmeans(self, descriptors_list):
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None:
                all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors)


class ModelTrainer:
    def __init__(self, model=SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=42)):
        self.model = model
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report, y_pred, y_prob

class Visualizer:
    @staticmethod
    def plot_sample_predictions(images, y_true, y_pred, n=100):
        plt.figure(figsize=(18, 9))
        for i in range(n):
            plt.subplot(10, 10, i + 1)
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
            plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_features(images, detector):
        for i, img in enumerate(images[:5]):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, _ = detector.detectAndCompute(gray, None)
            img_with_keypoints = cv2.drawKeypoints(img, keypoints, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            plt.title('Keypoints')
            plt.axis('off')
            plt.show()

    @staticmethod
    def plot_classification_histogram(y_true, y_pred, class_names, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        correct_counts = np.diag(cm)
        total_counts = np.sum(cm, axis=1)
        accuracy_per_class = correct_counts / total_counts
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, accuracy_per_class, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title(f'Class detection accuracy ({dataset_name} sample)')
        plt.ylim(0, 1)
        for i, acc in enumerate(accuracy_per_class):
            plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=12)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix ({dataset_name} sample)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Добавление значений в ячейки матрицы
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def plot_combined_histogram(cats_descriptors, dogs_descriptors, kmeans, k, dataset_name):
        all_cats_descriptors = np.vstack([desc for desc in cats_descriptors if desc is not None])
        all_dogs_descriptors = np.vstack([desc for desc in dogs_descriptors if desc is not None])

        cats_predictions = kmeans.predict(all_cats_descriptors)
        dogs_predictions = kmeans.predict(all_dogs_descriptors)

        cats_histogram = np.bincount(cats_predictions, minlength=k)
        dogs_histogram = np.bincount(dogs_predictions, minlength=k)

        plt.figure(figsize=(14, 7))

        # Построение гистограммы для кошек
        plt.subplot(1, 2, 1)
        plt.bar(range(k), cats_histogram, color='skyblue')
        plt.xlabel('Cluster Index')
        plt.ylabel('Number of Descriptors')
        plt.title(f'Histogram of Descriptors Distribution for Cats ({dataset_name} sample)')

        # Построение гистограммы для собак
        plt.subplot(1, 2, 2)
        plt.bar(range(k), dogs_histogram, color='orange')
        plt.xlabel('Cluster Index')
        plt.ylabel('Number of Descriptors')
        plt.title(f'Histogram of Descriptors Distribution for Dogs ({dataset_name} sample)')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_classification_results(y_true, y_pred, class_names, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        correct_counts = np.diag(cm)
        incorrect_counts = cm.sum(axis=1) - correct_counts

        plt.figure(figsize=(12, 6))

        # Построение гистограммы правильно классифицированных изображений
        plt.subplot(1, 2, 1)
        plt.bar(class_names, correct_counts, color='green')
        plt.xlabel('Classes')
        plt.ylabel('Number of correctly classified images')
        plt.title(f'Correctly classified images ({dataset_name} sample)')

        # Построение гистограммы неправильно классифицированных изображений
        plt.subplot(1, 2, 2)
        plt.bar(class_names, incorrect_counts, color='red')
        plt.xlabel('Classes')
        plt.ylabel('Number of misclassified images')
        plt.title(f'Incorrectly classified images ({dataset_name} sample)')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_prob, dataset_name):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic ({dataset_name} sample)')
        plt.legend(loc="lower right")
        plt.show()

class ArgumentParser:
    @staticmethod
    def cli_argument_parser():
        parser = argparse.ArgumentParser(description="Image classification using various algorithms and Bag of Words with SIFT and ORB.")

        parser.add_argument('-m', '--mode',
                            help='Mode (\'train_test\')',
                            type=str,
                            dest='mode',
                            default='train_test')
        parser.add_argument('-td', '--train_dir',
                            help='Directory with training images (cats and dogs)',
                            type=str,
                            dest='train_dir')
        parser.add_argument('-tsd', '--test_dir',
                            help='Directory with test images',
                            type=str,
                            dest='test_dir')
        parser.add_argument('-nc', '--n_clusters',
                            help='Number of clusters for visual dictionary',
                            type=int,
                            dest='n_clusters',
                            default=100)
        parser.add_argument('-a', '--algorithm',
                            help='Algorithm to use for training and testing',
                            type=str,
                            choices=['knn', 'svc', 'rf', 'gb'],
                            dest='algorithm',
                            default='gb')
        parser.add_argument('-d', '--descriptor',
                            help='Descriptor to use (sift or orb)',
                            type=str,
                            choices=['sift', 'orb'],
                            dest='descriptor',
                            default='sift')

        args = parser.parse_args()
        return args

def main(args):
    image_loader = ImageLoader()
    feature_extractor = FeatureExtractor(k=args.n_clusters, descriptor=args.descriptor)

    if args.algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    elif args.algorithm == 'svc':
        model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=42)
    elif args.algorithm == 'rf':
        model = RandomForestClassifier(n_estimators=8, class_weight='balanced', min_samples_split=4, min_samples_leaf=3, max_depth=3, max_features='log2', random_state=42)
    elif args.algorithm == 'gb':
        model = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, subsample=1, random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    model_trainer = ModelTrainer(model=model)
    visualizer = Visualizer()

    print("Загрузка изображений")
    cats_train, y_cats_train = image_loader.load_images_from_folder(os.path.join(args.train_dir, 'cats'), 0)
    dogs_train, y_dogs_train = image_loader.load_images_from_folder(os.path.join(args.train_dir, 'dogs'), 1)
    cats_test, y_cats_test = image_loader.load_images_from_folder(os.path.join(args.test_dir, 'cats'), 0)
    dogs_test, y_dogs_test = image_loader.load_images_from_folder(os.path.join(args.test_dir, 'dogs'), 1)

    X_train = cats_train + dogs_train
    y_train = y_cats_train + y_dogs_train
    X_test = cats_test + dogs_test
    y_test = y_cats_test + y_dogs_test

    print(f"Количество тренировочных изображений: {len(X_train)}")
    print(f"Количество тестовых изображений: {len(X_test)}")

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    train_descriptors = feature_extractor.extract_features(X_train)
    feature_extractor.fit_kmeans(train_descriptors)

    print(f"Среднее количество точек, обнаруженных на тренировочных изображениях: {feature_extractor.average_keypoints}")

    cats_descriptors = feature_extractor.extract_features(cats_train)
    dogs_descriptors = feature_extractor.extract_features(dogs_train)
    visualizer.plot_combined_histogram(cats_descriptors, dogs_descriptors, feature_extractor.kmeans, feature_extractor.k, 'Train')

    X_train_features = feature_extractor.build_feature_vectors(X_train)
    X_test_features = feature_extractor.build_feature_vectors(X_test)

    model_trainer.train(X_train_features, y_train)

    train_accuracy, train_report, y_train_pred, y_train_prob = model_trainer.evaluate(X_train_features, y_train)
    test_accuracy, test_report, y_test_pred, y_test_prob = model_trainer.evaluate(X_test_features, y_test)

    print("Train Accuracy:", train_accuracy)
    print("Train Classification Report:")
    print(train_report)
    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:")
    print(test_report)

    class_names = ['Cats', 'Dogs']
    # visualizer.plot_classification_histogram(y_train, y_train_pred, class_names, 'Train')
    # visualizer.plot_classification_histogram(y_test, y_test_pred, class_names, 'Test')

    visualizer.plot_classification_results(y_train, y_train_pred, class_names, 'Train')
    visualizer.plot_classification_results(y_test, y_test_pred, class_names, 'Test')

    visualizer.plot_roc_curve(y_train, y_train_prob, 'Train')
    visualizer.plot_roc_curve(y_test, y_test_prob, 'Test')

    visualizer.visualize_features(X_train, feature_extractor.detector)

    visualizer.plot_confusion_matrix(y_train, y_train_pred, class_names, 'Train')
    visualizer.plot_confusion_matrix(y_test, y_test_pred, class_names, 'Test')

if __name__ == "__main__":
    args = ArgumentParser.cli_argument_parser()
    main(args)
