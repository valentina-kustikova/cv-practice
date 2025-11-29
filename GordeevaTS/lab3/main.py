import argparse
import os
import sys
import json
from datetime import datetime
from data_loader import DataLoader
from bow_classifier import BOWClassifier
from nn_classifier import NNClassifier
from utils import evaluate_classifier, print_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Landmark Classification App')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to data directory')
    parser.add_argument('--train_split', type=str, required=True,
                       help='File with train split')
    parser.add_argument('--test_split', type=str, required=True,
                       help='File with test split')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], 
                       default='both', help='Operation mode')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'nn'], 
                       required=True, help='Algorithm to use')
    parser.add_argument('--model_path', type=str, default='models',
                       help='Path to save/load models')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to save/load')
    
    # BOW parameters
    parser.add_argument('--detector', type=str, default='SIFT',
                       choices=['SIFT', 'ORB', 'AKAZE'], help='Feature detector')
    parser.add_argument('--vocab_size', type=int, default=50,
                       help='Vocabulary size for BOW')
    
    # NN parameters
    parser.add_argument('--classifier', type=str, default='RandomForest',
                       choices=['RandomForest', 'SVM', 'KNN'],
                       help='Classifier for neural features')
    
    return parser.parse_args()

def create_model_directory(base_path, algorithm, detector=None, vocab_size=None, classifier_type=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if algorithm == 'bow':
        model_dir = f"bow_{detector}_vocab{vocab_size}_{timestamp}"
    else:
        model_dir = f"nn_{classifier_type}_{timestamp}"
    
    full_path = os.path.join(base_path, model_dir)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, model_dir

def save_model_info(model_path, algorithm, args, accuracy=None):
    info = {
        'algorithm': algorithm,
        'created_at': datetime.now().isoformat(),
        'parameters': {
            'detector': getattr(args, 'detector', None),
            'vocab_size': getattr(args, 'vocab_size', None),
            'classifier': getattr(args, 'classifier', None),
        },
        'data_info': {
            'train_split': args.train_split,
            'test_split': args.test_split,
            'data_path': args.data_path
        },
        'performance': {
            'accuracy': accuracy
        }
    }
    
    info_path = os.path.join(model_path, 'model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"Model info saved to: {info_path}")

def load_model_info(model_path):
    info_path = os.path.join(model_path, 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def find_latest_model(model_path, algorithm):
    if not os.path.exists(model_path):
        return None
    
    model_dirs = [d for d in os.listdir(model_path) 
                 if os.path.isdir(os.path.join(model_path, d)) and d.startswith(algorithm)]
    
    if not model_dirs:
        return None
    
    model_dirs.sort(key=lambda x: os.path.getctime(os.path.join(model_path, x)), reverse=True)
    return os.path.join(model_path, model_dirs[0])

def main():
    args = parse_arguments()
    os.makedirs(args.model_path, exist_ok=True)

    print("Loading data...")
    data_loader = DataLoader(args.data_path, args.train_split, args.test_split)
    train_images, train_labels, test_images, test_labels = data_loader.load_data()
    class_names = data_loader.get_class_names()
    
    if args.algorithm == 'bow':
        classifier = BOWClassifier(
            detector_type=args.detector,
            vocab_size=args.vocab_size
        )
    else:
        classifier = NNClassifier(
            classifier_type=args.classifier,
            num_classes=len(class_names)
        )
    
    if args.mode == 'test' and not args.model_name:
        model_path = find_latest_model(args.model_path, args.algorithm)
        if model_path is None:
            print(f"No trained {args.algorithm} model found in {args.model_path}")
            return
        specific_model_path = model_path
    elif args.model_name:
        specific_model_path = os.path.join(args.model_path, args.model_name)
    else:
        specific_model_path, model_dir = create_model_directory(
            args.model_path, args.algorithm, args.detector, args.vocab_size, args.classifier
        )
        print(f"Model will be saved to: {specific_model_path}")
    
    if args.mode in ['train', 'both']:
        print("Training classifier...")
        classifier.train(train_images, train_labels)
        
        model_file_path = os.path.join(specific_model_path, 'model')
        classifier.save(model_file_path)
        print(f"Model saved to {model_file_path}")
        
        print("Testing on training data for accuracy...")
        predictions = classifier.predict(train_images)
        accuracy, _, _ = evaluate_classifier(train_labels, predictions, class_names)
        save_model_info(specific_model_path, args.algorithm, args, accuracy)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test' or (args.mode == 'both' and args.model_name):
            model_file_path = os.path.join(specific_model_path, 'model')
            print(f"Loading model from {model_file_path}")
            classifier.load(model_file_path)
            
            model_info = load_model_info(specific_model_path)
            if model_info:
                print(f"\nModel info:")
                print(f"Algorithm: {model_info['algorithm']}")
                print(f"Created: {model_info['created_at']}")
                if model_info['performance']['accuracy']:
                    print(f"Training accuracy: {model_info['performance']['accuracy']:.4f}")
        
        print("Testing classifier...")
        predictions = classifier.predict(test_images)
        
        accuracy, report, cm = evaluate_classifier(test_labels, predictions, class_names)
        print_metrics(accuracy, report, cm, class_names)
        
        if args.mode == 'both':
            model_info = load_model_info(specific_model_path)
            if model_info:
                model_info['performance']['test_accuracy'] = accuracy
                info_path = os.path.join(specific_model_path, 'model_info.json')
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)

def list_models():
    model_path = 'models'
    if not os.path.exists(model_path):
        print("No models directory found")
        return
    
    print("Available models:")
    for model_dir in os.listdir(model_path):
        full_path = os.path.join(model_path, model_dir)
        if os.path.isdir(full_path):
            info = load_model_info(full_path)
            if info:
                print(f"- {model_dir}")
                print(f"  Algorithm: {info['algorithm']}")
                print(f"  Created: {info['created_at']}")
                if info['performance']['accuracy']:
                    print(f"  Training Accuracy: {info['performance']['accuracy']:.4f}")
            else:
                print(f"- {model_dir} (no info file)")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        list_models()
    else:
        main()