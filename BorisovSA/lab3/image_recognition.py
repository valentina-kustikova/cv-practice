import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import argparse
import random


def load_images_from_folder(folder, label, image_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image {img_path} not found!")
            continue
        img = cv2.resize(img, image_size)
        images.append((img, label))
    return images

def build_visual_vocab(descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

def compute_bow_features(images, vocab, detector):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    bow_extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
    bow_extractor.setVocabulary(vocab)
    features = []
    for img, _ in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp = detector.detect(gray, None)
        hist = bow_extractor.compute(gray, kp)
        if hist is not None:
            features.append(hist.flatten())
        else:
            features.append(np.zeros(len(vocab)))
    return np.array(features)

def train_and_evaluate(train_data, test_data, vocab):
    sift = cv2.SIFT_create()

    train_labels = [label for _, label in train_data]
    train_features = compute_bow_features(train_data, vocab, sift)

    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_features_scaled = scaler.fit_transform(train_features)

    model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=42)
    model.fit(train_features_scaled, train_labels_encoded)

    test_labels = [label for _, label in test_data]
    test_features = compute_bow_features(test_data, vocab, sift)

    test_labels_encoded = label_encoder.transform(test_labels)
    test_features_scaled = scaler.transform(test_features)

    train_predictions = model.predict(train_features_scaled)
    train_accuracy = accuracy_score(train_labels_encoded, train_predictions)
    train_report = classification_report(train_labels_encoded, train_predictions, target_names=["cat", "dog"])

    test_predictions = model.predict(test_features_scaled)
    test_accuracy = accuracy_score(test_labels_encoded, test_predictions)
    test_report = classification_report(test_labels_encoded, test_predictions, target_names=["cat", "dog"])

    return train_accuracy, train_report, test_accuracy, test_report, test_predictions, test_labels_encoded

def keypoints_stat(images, detector):
    keypoints_counts = []
    for i, (img, _) in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray, None)
        keypoints_counts.append(len(keypoints))

    print("\nStatistics of keypoints detected for all images:")
    print(f"Min keypoints: {min(keypoints_counts)}")
    print(f"Average keypoints: {np.mean(keypoints_counts).astype(int)}")
    print(f"Max keypoints: {max(keypoints_counts)}")

def visualize_random_keypoints(images, detector, num_samples=2, label="unknown"):
    random_images = random.sample(images, min(num_samples, len(images)))

    for img, _ in random_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray, None)

        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title(f"Keypoints for {label}")
        plt.axis("off")
        plt.show()

def visualize_clusters(descriptors, kmeans, num_clusters):
    predictions = kmeans.predict(descriptors)

    cluster_counts = np.bincount(predictions, minlength=num_clusters)

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_clusters), cluster_counts, color='skyblue')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Descriptors')
    plt.title('Distribution of Descriptors Across Clusters')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title(title)
    plt.show()

def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors = []
    for img, _ in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        if des is not None:
            descriptors.append(des)
    return np.vstack(descriptors) if descriptors else np.array([])

def argument_parser():
    parser = argparse.ArgumentParser(description="Image classification using Bag of Visual Words and SVM.")
    parser.add_argument("--train_dir", required=True, help="Path to the training dataset directory.")
    parser.add_argument("--test_dir", required=True, help="Path to the testing dataset directory.")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for visual vocabulary.")
    return parser.parse_args()

def main():
    args = argument_parser()

    train_cats = load_images_from_folder(os.path.join(args.train_dir, "cats"), "cat")
    train_dogs = load_images_from_folder(os.path.join(args.train_dir, "dogs"), "dog")
    test_cats = load_images_from_folder(os.path.join(args.test_dir, "cats"), "cat")
    test_dogs = load_images_from_folder(os.path.join(args.test_dir, "dogs"), "dog")

    train_data = train_cats + train_dogs
    test_data = test_cats + test_dogs

    if not train_data or not test_data:
        print("Error: Not enough images found in the dataset.")
        return

    descriptors = extract_sift_descriptors(train_data)

    if descriptors.size == 0:
        print("Error: No descriptors found in the training set.")
        return

    kmeans = build_visual_vocab(descriptors, args.clusters)

    print("Visualizing the distribution of descriptors across clusters...")
    visualize_clusters(descriptors, kmeans, args.clusters)

    print("Visualizing keypoints on random images...")
    keypoints_stat(train_data, cv2.SIFT_create())
    visualize_random_keypoints(train_cats, cv2.SIFT_create(), num_samples=2, label="cat")
    visualize_random_keypoints(train_dogs, cv2.SIFT_create(), num_samples=2, label="dog")

    train_accuracy, train_report, test_accuracy, test_report, test_predictions, test_labels_encoded = train_and_evaluate(train_data, test_data, kmeans.cluster_centers_)

    print(f"Accuracy on train: {train_accuracy:.3f}")
    print("Classification report on train:")
    print(train_report)

    print(f"Accuracy on test: {test_accuracy:.3f}")
    print("Classification report on test:")
    print(test_report)

    plot_confusion_matrix(test_labels_encoded, test_predictions, classes=["cat", "dog"], title="Confusion Matrix")

if __name__ == "__main__":
    main()
