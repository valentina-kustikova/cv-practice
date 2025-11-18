import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_dataset
from bow_classifier import BoWClassifier
from nn_classifier import NNClassifier
from detectors import SIFTDetector, ORBDetector, AKAZEDetector


def visualize_keypoints(detector_type="sift"):
    print("═" * 90)
    print(f"   ВИЗУАЛИЗАЦИЯ {detector_type.upper()}".center(90))
    print("═" * 90)

    from detectors import SIFTDetector, ORBDetector, AKAZEDetector
    detector_map = {
        "sift": SIFTDetector(),
        "orb": ORBDetector(),
        "akaze": AKAZEDetector()
    }
    detector = detector_map[detector_type]

    train_data, test_data, _ = load_dataset("./data", "train.txt")
    all_images = train_data + test_data
    np.random.shuffle(all_images)
    selected = all_images[:4]

    plt.figure(figsize=(14, 8))

    for i, (path, _) in enumerate(selected):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, _ = detector.detect_and_compute(gray)

        if not kp:
            kp = []

        kp = sorted(kp, key=lambda x: -x.response)[:500]

        # Ресайз изображения
        h, w = img.shape[:2]
        scale = 400 / h
        small = cv2.resize(img, (int(w * scale), 400), interpolation=cv2.INTER_LANCZOS4)

        # Масштабируем точки
        kp_scaled = []
        for p in kp:
            x = p.pt[0] * scale
            y = p.pt[1] * scale
            size = p.size * scale * 1.4  # ещё чуть крупнее — идеально видно
            kp_scaled.append(cv2.KeyPoint(x=x, y=y, size=size,
                                          angle=p.angle, response=p.response,
                                          octave=p.octave, class_id=p.class_id))

        # Классический OpenCV-стиль — самый красивый!
        vis = cv2.drawKeypoints(
            small.copy(),
            kp_scaled,
            None,
            color=detector.color,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        plt.subplot(2, 4, 2*i + 1)
        plt.title(os.path.basename(path), fontsize=11)
        plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 4, 2*i + 2)
        color_name = 'red' if detector_type == 'sift' else 'orange' if detector_type == 'orb' else 'green'
        plt.title(f"{detector.name} — 500 точек", fontsize=12, fontweight='bold', color=color_name)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.suptitle(f"Детектор ключевых точек: {detector.name}", fontsize=20, fontweight='bold',
                 color='red' if detector_type=='sift' else 'darkorange' if detector_type=='orb' else 'green')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Визуализация завершена!")


def visualize_gradcam():
    print("═" * 70)
    print("   GRAD-CAM ВИЗУАЛИЗАЦИЯ".center(70))
    print("═" * 70)

    if not os.path.exists("models/nn_model.keras"):
        print("Ошибка: Сначала обучите нейросеть!")
        return

    _, test_data, class_names = load_dataset("./data", "train.txt")
    np.random.shuffle(test_data)
    test_data = test_data[:4]

    model = tf.keras.models.load_model("models/nn_model.keras")
    last_conv = model.get_layer("top_conv")
    grad_model = tf.keras.models.Model(model.inputs, [last_conv.output, model.output])

    plt.figure(figsize=(16, 9))
    for i, (path, true_label) in enumerate(test_data):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img_orig = img.numpy().astype(np.uint8)
        img_resized = tf.image.resize(img, (224, 224))
        img_input = tf.expand_dims(img_resized, 0)
        img_input = tf.keras.applications.efficientnet.preprocess_input(img_input)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_input)
            pred_idx = tf.argmax(preds[0])
            loss = preds[:, pred_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap.numpy(), (img_orig.shape[1], img_orig.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_orig, 0.7, heatmap, 0.3, 0)

        max_w = 380
        h, w = img_orig.shape[:2]
        if w > max_w:
            scale = max_w / w
            new_h = int(h * scale)
            img_orig = cv2.resize(img_orig, (max_w, new_h))
            superimposed = cv2.resize(superimposed, (max_w, new_h))

        plt.subplot(2, 4, i + 1)
        plt.title(f"Правда: {class_names[true_label]}", fontsize=11)
        plt.imshow(img_orig[..., ::-1])
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        conf = tf.nn.softmax(preds[0])[pred_idx].numpy()
        plt.title(f"Предсказано: {class_names[pred_idx]}\nУверенность: {conf:.1%}", fontsize=11)
        plt.imshow(superimposed[..., ::-1])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Лабораторная №3 — Классификация достопримечательностей")
    parser.add_argument("--data", type=str, default="./data", help="Путь к папке с данными")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--algo", choices=["bow", "nn", "both"], default="both")
    parser.add_argument("--detector", choices=["sift", "orb", "akaze"], default="sift", help="Детектор для BoW")
    parser.add_argument("--visualize", choices=["sift", "orb", "akaze", "gradcam"], help="Только визуализация")

    args = parser.parse_args()

    if args.visualize:
        if args.visualize in ["sift", "orb", "akaze"]:
            visualize_keypoints(args.visualize)
        elif args.visualize == "gradcam":
            visualize_gradcam()
    else:
        train_data, test_data, class_names = load_dataset(args.data, "train.txt")
        os.makedirs("models", exist_ok=True)

        if args.algo in ["bow", "both"]:
            detector = args.detector
            if args.mode in ["train", "both"]:
                bow = BoWClassifier(k=600, detector_type=detector)  # ← правильный аргумент!
                bow.fit(train_data)
                bow.save(f"models/bow_{detector}_model.pkl")
            if args.mode in ["test", "both"]:
                bow = BoWClassifier.load(f"models/bow_{detector}_model.pkl")
                bow.predict(test_data)

        if args.algo in ["nn", "both"]:
            nn = NNClassifier()
            if args.mode in ["train", "both"]:
                val_data = test_data[:len(test_data)//3]
                nn.fit(train_data, val_data)
            if args.mode in ["test", "both"]:
                nn.predict(test_data)

        print("\nВсё выполнено успешно!")