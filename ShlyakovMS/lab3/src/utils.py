# src/utils.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_bow_beautifully(test_images, test_labels, pred_bow, bow_model, top_k=5, samples_per_class=1):

    os.makedirs("bow_visualizations", exist_ok=True)
    class_names = ['Кремль', 'Дворец труда', 'Архангельский собор']
    num_classes = len(class_names)

    selected_indices = []
    for cls in range(num_classes):
        candidates = np.where((test_labels == cls) & (pred_bow == cls))[0]
        if len(candidates) == 0:
            candidates = np.where(test_labels == cls)[0]
        if len(candidates) > 0:
            selected_indices.append(np.random.choice(candidates))
        else:
            selected_indices.append(None)

    valid_indices = [i for i in selected_indices if i is not None]
    if len(valid_indices) == 0:
        print(" Нет изображений для визуализации.")
        return

    test_sample = test_images[valid_indices]
    sample_labels = test_labels[valid_indices]
    sample_preds = pred_bow[valid_indices]

    desc_list, kp_list = bow_model.extract_sift(test_sample)
    histograms = bow_model.create_histogram(desc_list)

    n_vis = len(valid_indices)
    fig, axes = plt.subplots(1, n_vis, figsize=(6 * n_vis, 6))
    if n_vis == 1:
        axes = [axes]
    fig.suptitle("BoW: Топ-визуальные слова (по одному на класс)", fontsize=16, weight='bold', y=0.95)

    colors_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    colors_rgb = [tuple(c[::-1]) for c in colors_bgr[:top_k]]

    for idx, (img, kps, desc, hist, true_label, pred_label) in enumerate(
        zip(test_sample, kp_list, desc_list, histograms, sample_labels, sample_preds)
    ):
        if len(kps) == 0:
            axes[idx].imshow(img)
            axes[idx].set_title(f"{class_names[true_label]} | Нет точек", color='red')
            axes[idx].axis('off')
            continue

        top_indices = np.argsort(hist)[-top_k:][::-1]
        top_words = [int(i) for i in top_indices]

        word_kps = {w: [] for w in top_words}
        for kp, des in zip(kps, desc):
            word_id = bow_model.kmeans.predict(des.reshape(1, -1))[0]
            if word_id in top_words:
                word_kps[word_id].append(kp.pt)

        overlay = img.copy()
        for word_id, color_rgb in zip(top_words, colors_rgb):
            pts = np.array(word_kps[word_id])
            if len(pts) == 0:
                continue
            for (x, y) in pts:
                cv2.circle(overlay, (int(x), int(y)), 6, color_rgb[::-1], 1)
                cv2.circle(overlay, (int(x), int(y)), 5, color_rgb[::-1], -1)

        alpha = 0.7
        img_vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        ax = axes[idx]
        ax.imshow(img_vis)
        ax.axis('off')

        pred_name = class_names[pred_label]
        true_name = class_names[true_label]
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"{pred_name} | {true_name}", fontsize=12, pad=10,
                     backgroundcolor='white', alpha=0.8, color=color)

        legend_text = ""
        for i, word_id in enumerate(top_words):
            freq = hist[word_id]
            legend_text += f"  • Слово {word_id}: {freq:.1%}\n"
        ax.text(0.02, 0.98, legend_text.strip(), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', color='black',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                linespacing=1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(top=0.88)

    # Сохранение
    save_path = "bow_visualizations/bow_by_class.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f" Визуализация по классам сохранена: {save_path}")