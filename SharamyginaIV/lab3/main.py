import argparse
import os
import sys
from src.bag_of_words import BagOfWordsClassifier
from src.neural_network import NeuralNetworkClassifier
from src.utils import load_image_paths, load_split_file


def main():
    parser = argparse.ArgumentParser(description='Image Classification for Nizhny Novgorod Landmarks')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--train_split', type=str, required=True,
                        help='File containing training split')
    parser.add_argument('--test_split', type=str, required=True,
                        help='File containing test split')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train_test'],
                        default='train_test', help='Mode of operation')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'neural_network'],
                        required=True, help='Algorithm to use')
    parser.add_argument('--bow_k', type=int, default=100,
                        help='Size of vocabulary for Bag of Words')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for neural network training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for neural network training')
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to save/load the model')

    args = parser.parse_args()

    # Store the actual parameters used for the run
    actual_params = {
        'data_dir': args.data_dir,
        'train_split': args.train_split,
        'test_split': args.test_split,
        'mode': args.mode,
        'algorithm': args.algorithm,
        'model_path': args.model_path
    }

    # Load image paths from split files
    train_paths = load_split_file(args.train_split, args.data_dir)
    test_paths = load_split_file(args.test_split, args.data_dir)

    print(f"Loaded {len(train_paths)} training images and {len(test_paths)} test images")

    if args.algorithm == 'bow':
        classifier = BagOfWordsClassifier(vocab_size=args.bow_k)
        # Add BOW-specific parameters
        actual_params.update({
            'bow_k': args.bow_k,
            'epochs': None,  # Not applicable for BOW
            'batch_size': None  # Not applicable for BOW
        })
    else:
        classifier = NeuralNetworkClassifier(
            num_classes=3,  # 3 landmarks: Кремль, Дворец труда, Архангельский собор
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        # Add NN-specific parameters
        actual_params.update({
            'bow_k': None,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        })

    if 'train' in args.mode:
        print("Starting training...")
        classifier.train(train_paths)
        classifier.save_model(args.model_path)
        print("Training completed.")

    if 'test' in args.mode:
        if 'train' not in args.mode:
            # Load model if not trained in this run
            classifier.load_model(args.model_path)

        print("Starting testing...")
        accuracy = classifier.evaluate(test_paths)
        print(f"Test accuracy: {accuracy:.4f}")

        # Save results with actual parameters used
        with open('results.txt', 'w') as f:
            f.write(f"Algorithm: {args.algorithm}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Parameters: {actual_params}\n")


if __name__ == '__main__':
    main()