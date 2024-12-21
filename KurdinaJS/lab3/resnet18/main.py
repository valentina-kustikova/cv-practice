import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import cv2


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', '--train',
                        help='Path to train data',
                        type=str,
                        dest='train_path')
    parser.add_argument('-test', '--test',
                        help='Path to test data',
                        type=str,
                        dest='test_path')
    parser.add_argument('-cc', '--clusters',
                        help='Count of clusters',
                        type=int,
                        dest='count_clusters',
                        default=100)

    args = parser.parse_args()
    return args


def load_images(dataset_path, img_size=(256, 256)):
    images = []
    labels = []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        if img_name.startswith("cat"):
            label = 1
        elif img_name.startswith("dog"):
            label = 0
        else:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


def extract_sift_features(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def extract_combined_features(images):
    model = models.resnet18(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    global_features = []
    local_features = []

    with torch.no_grad():
        for img in images:
            img_tensor = transform(img).unsqueeze(0)
            output = model(img_tensor)
            global_features.append(output.squeeze().cpu().numpy())

            keypoints, descriptors = extract_sift_features(img)
            local_features.append((keypoints, descriptors))

    return np.array(global_features), local_features


def new_histogram(features, kmeans_model):
    clusters = kmeans_model.predict(features)
    histogram = np.zeros(kmeans_model.n_clusters)
    for cluster in clusters:
        histogram[cluster] += 1
    return histogram


def combine_features(global_features, local_features, kmeans_model):
    combined_features = []
    for global_vec, (keypoints, local_desc) in zip(global_features, local_features):
        if local_desc is not None:
            local_histogram = new_histogram(local_desc, kmeans_model)
            combined = np.concatenate([global_vec, local_histogram])
            combined_features.append(combined)
        else:
            combined_features.append(global_vec)
    return np.array(combined_features)


def visualize_keypoints(image, keypoints):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_keypoints)
    plt.title("SIFT Keypoints")
    plt.show()


def plot_accuracy_histogram(true_labels, predicted_labels, class_labels):
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    class_accuracies = np.diag(confusion_mat) / confusion_mat.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_accuracies, color='#88CFFD', edgecolor='black')
    plt.xlabel('Классы')
    plt.ylabel('Точность')
    plt.title('Точность классификации по классам')
    plt.ylim(0, 1)

    for index, accuracy in enumerate(class_accuracies):
        plt.text(index, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontsize=12, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    args = cli_argument_parser()

    X_train, y_train = load_images(args.train_path)
    X_test, y_test = load_images(args.test_path)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    X_train_global, X_train_local = extract_combined_features(X_train)
    X_test_global, X_test_local = extract_combined_features(X_test)

    all_descriptors = np.vstack([desc for _, desc in X_train_local if desc is not None])
    kmeans = KMeans(n_clusters=args.count_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    X_train_combined = combine_features(X_train_global, X_train_local, kmeans)
    X_test_combined = combine_features(X_test_global, X_test_local, kmeans)

    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_test_combined = scaler.transform(X_test_combined)

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_combined, y_train)

    y_pred = model.predict(X_test_combined)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Dog", "Cat"]))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    plot_accuracy_histogram(y_test, y_pred, ['Dogs', 'Cats'])

    indices = np.random.randint(len(X_test)-1, size=3)
    for idx in indices:
        keypoints, _ = X_test_local[idx]
        if keypoints:
            visualize_keypoints(X_test[idx], keypoints)

    indices = np.random.randint(len(X_test)-1, size=10)
    for idx in indices:
        plt.imshow(cv2.cvtColor(X_test[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {'Dog' if y_pred[idx] == 0 else 'Cat'}, Actual: {'Dog' if y_test[idx] == 0 else 'Cat'}")
        plt.show()


if __name__ == '__main__':
    sys.exit(main() or 0)
