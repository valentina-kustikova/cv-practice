import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import argparse
import joblib
from tensorflow.data import Dataset
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

def load_images_from_split(file, mode = None):
    images, labels= [], []
    class_dirs = {} 

    with open(file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                class_dir = os.path.basename(os.path.dirname(path))
                if class_dir not in class_dirs:
                    class_dirs[class_dir] = len(class_dirs)

    with open(file, 'r') as f:
        for line in f:
            path = line.strip()
            path = "data\\" + path
            if path:
                img = cv2.imread(path)
                if img is not None:
                    if mode == "NN": img = cv2.resize(img,(224,224))
                    class_dir = os.path.basename(os.path.dirname(path))
                    images.append(img)
                    labels.append(class_dirs[class_dir])

    return images, np.array(labels)

def sift_descr(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

def bag_of_words(descriptors_list, k=500):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
    #kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

def bow_histograms(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
        histograms.append(hist)
    return np.array(histograms)

def train_bow_model(hists, labels):
    X = hists
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    clf = SVC(kernel = 'linear', probability = True, random_state = 42)
    #clf = clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_scaled, labels)
    return clf, scaler

def test_bow_model(clf, scaler, hists, labels=None):
    X = scaler.transform(hists)
    preds = clf.predict(X)
    if labels is not None:
        acc = accuracy_score(labels, preds)
        tpr = recall_score(labels, preds, average= "macro")
        return preds, acc, tpr
    return preds

def save_model(clf, scaler, bag,  output_path):
    joblib.dump({'clf': clf, 'scaler': scaler, 'bag': bag}, output_path)

def my_load_model(model_path):
    data = joblib.load(model_path)
    return data['clf'], data['scaler'], data['bag']


def train_nn(train_images, train_labels):
    data_train = Dataset.from_tensor_slices((train_images, train_labels))
    data_train = data_train.shuffle(buffer_size=len(train_images)).batch(32)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    num_classes = len(np.unique(train_labels))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_train, epochs=5)
    model.save('model.keras')

def test_nn(model, test_images, test_labels):
    data_test = Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    preds = np.argmax(model.predict(data_test),axis=1)
    if test_labels is not None:
        acc = accuracy_score(test_labels, preds)
        tpr = recall_score(test_labels, preds, average= "macro")
        return acc, tpr, preds
    return preds

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', help='Path to train split file', type=str)
    parser.add_argument('-e', '--test_file', help='Path to test split file', type=str)
    parser.add_argument('-m', '--method', help='Method to apply',
                        choices=['BOW', 'NN'],
                        required=True)
    parser.add_argument('-j', '--type', help='Type of job: train or test',
                        choices=['train', 'test'],
                        required=True)
    parser.add_argument('--clf', help='Path to classifier for test mode', type=str, default='model.joblib')
    parser.add_argument('--model', help='Path to NN model for test mode', type=str, default='model.keras')
    parser.add_argument('--output', help='Path to save classifier for train mode', type=str, default='model.joblib')

    return parser.parse_args()

def main():
    args = cli_argument_parser()

    if args.method == "BOW":
        if args.type == "train":
            train_images, train_labels = load_images_from_split(args.train_file)
            descriptors = sift_descr(train_images)
            bag = bag_of_words(descriptors)
            hists = bow_histograms(descriptors, bag)
            clf, scaler = train_bow_model(hists, train_labels)
            save_model(clf, scaler, bag, args.output)
            print(f"\nDone!")
        if args.type == "test":
            test_images, test_labels = load_images_from_split(args.test_file)
            if args.clf:
                clf, scaler, bag = my_load_model(args.clf)
            else:
                raise ValueError("Classifier path (--clf) is required for test mode")
            descriptors = sift_descr(test_images)
            hists = bow_histograms(descriptors, bag)
            preds, acc, tpr = test_bow_model(clf, scaler, hists, test_labels)
            print("Accuracy: ", acc)
            print("TPR: ", tpr)
    if args.method == "NN":
        if args.type == "train":
            train_images, train_labels = load_images_from_split(args.train_file, mode = "NN")
            train_nn(train_images,train_labels)
            print("\nDone!")
        if args.type == "test":
            test_images, test_labels = load_images_from_split(args.test_file, mode = "NN")
            model = load_model(args.model)
            acc,tpr,preds = test_nn(model,test_images,test_labels)
            print("Accuracy: ", acc)
            print("TPR: ", tpr)
main()