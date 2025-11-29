import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(true_labels, predictions, class_names):
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=class_names, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    
    return accuracy, report, cm

def print_metrics(accuracy, report, cm, class_names):
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def visualize_predictions(images, true_labels, predictions, class_names, num_samples=5):
    if len(images) == 0:
        print("No images to visualize")
        return
        
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[predictions[idx]]
        color = 'green' if true_labels[idx] == predictions[idx] else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()