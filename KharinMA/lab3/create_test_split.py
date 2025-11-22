import os
import argparse

def create_test_split(data_dir, train_file, test_file):
    # Read train files
    train_images = set()
    with open(train_file, 'r') as f:
        for line in f:
            # Normalize path separators
            path = line.strip().replace('\\', os.sep).replace('/', os.sep)
            train_images.add(path)

    print(f"Loaded {len(train_images)} training images.")

    test_images = []
    # Scan data_dir
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                # Normalize
                rel_path = rel_path.replace('\\', os.sep).replace('/', os.sep)
                
                # Check if it's in train_images
                # We need to be careful about path normalization matching exactly
                # Let's try to match normalized paths
                if rel_path not in train_images:
                    test_images.append(rel_path)

    print(f"Found {len(test_images)} test images.")
    
    with open(test_file, 'w') as f:
        for img in test_images:
            f.write(img + '\n')
    
    print(f"Test split saved to {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--test_file', required=True)
    args = parser.parse_args()
    
    create_test_split(args.data_dir, args.train_file, args.test_file)
