
import cv2
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_images_from_dir(dir, label, image_size=(256, 256)):
    images = []
    labels = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

def load_ds(dir):
    cats, y_cats = load_images_from_dir(os.path.join(dir, 'cats'), 0)
    dogs, y_dogs = load_images_from_dir(os.path.join(dir, 'dogs'), 1)

    X = cats + dogs
    y = y_cats + y_dogs

    X, y = shuffle(X, y, random_state=42)
    
    return (X, y)


def extract_sift_features(images, sift):
    # Извлечение дескрипторов для всех изображений
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

def build_histogram(descriptors, kmeans):
    histogram = np.zeros(kmeans.n_clusters)
    if descriptors is not None:
        predictions = kmeans.predict(descriptors)
        for pred in predictions:
            histogram[pred] += 1
    return histogram

def build_feature_vectors(images, sift, kmeans):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        histogram = build_histogram(descriptors, kmeans)
        features.append(histogram)
    return np.array(features)

def train(X_train, y_train, k, random_state=42):
    sift = cv2.SIFT_create()
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', random_state=random_state)
    #clf = NuSVC(random_state=random_state)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=random_state)

    print('Extracting features with SIFT')
    train_descriptors = extract_sift_features(X_train, sift)
    all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])
    print('Clasterisation with k-means')
    kmeans.fit(all_descriptors)
    X_train_features = build_feature_vectors(X_train, sift, kmeans)
    X_train_features = scaler.fit_transform(X_train_features)
    print('Training classification model')
    clf.fit(X_train_features, y_train)
    return sift, kmeans, scaler, clf

def evaluate(X_test, y_test, sift, kmeans, scaler, clf):
    X_test_features = build_feature_vectors(X_test, sift, kmeans)
    X_test_features = scaler.transform(X_test_features)
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    return (y_pred, accuracy)

def visualization(images, labels, sift, number):
    samples = images[10:number+10]
    samples_y = labels[10:number+10]
    classes = {0: 'Cat', 1: 'Dog'}

    prediction, acc = evaluate(X_test, y_test, sift, kmeans, scaler, clf)

    for i, img in enumerate(samples):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift_keypoints, _ = sift.detectAndCompute(gray, None)
        img_with_sift_keypoints = cv2.drawKeypoints(img, sift_keypoints, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(img_with_sift_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('SIFT Keypoints')
        plt.suptitle('label: {0}\npredicted: {1}'.format(classes[samples_y[i]], 
                                                         classes[prediction[i]]))
        plt.axis('off')
        plt.show()        

def plot_classification_histogram(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    correct_counts = np.diag(cm)
    total_counts = np.sum(cm, axis=1)
    accuracy_per_class = correct_counts / total_counts
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracy_per_class, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Classification performance')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracy_per_class):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=12)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", help="Path to source dataset directory", required=True)
    parser.add_argument("-k", "--clusters", help="Number of clusters", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    k = int(args.clusters)

    print("Loading dataset")
    X_train, y_train = load_ds(os.path.join(args.data_dir, 'train'))
    X_test, y_test = load_ds(os.path.join(args.data_dir, 'test'))

    sift, kmeans, scaler, clf = train(X_train, y_train, k)
    
    print("Evaluating")
    y_test_pred, accuracy = evaluate(X_test, y_test, sift, kmeans, scaler, clf)
    
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    class_names = ['Cats', 'Dogs']
    plot_classification_histogram(y_test, y_test_pred, class_names)

    print('Visualizing')
    visualization(X_test, y_test, sift, 6)


