import os


def extract_class_name(image_path):
    # Get the parent directory name which should be the class name
    parent_dir = os.path.basename(os.path.dirname(image_path))
    return parent_dir


def load_split_file(split_file, data_dir):
    image_paths = []

    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Construct full path
                full_path = os.path.join(data_dir, line)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    print(f"Warning: Image not found: {full_path}")

    return image_paths


def load_image_paths(data_dir):
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def create_class_mapping(data_dir):
    class_names = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)

    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class
