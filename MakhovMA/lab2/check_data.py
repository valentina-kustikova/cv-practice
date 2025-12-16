import os
import glob


def check_data_structure(images_path, annotations_path):
    print("=== CHECKING DATA STRUCTURE ===")

    # Проверяем изображения
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))

    print(f"Found {len(image_files)} images")
    if image_files:
        print(f"First 5 images: {[os.path.basename(f) for f in image_files[:5]]}")

    # Проверяем аннотации
    annotation_files = glob.glob(os.path.join(annotations_path, "*.txt"))
    print(f"Found {len(annotation_files)} annotation files")
    if annotation_files:
        print(f"First 5 annotations: {[os.path.basename(f) for f in annotation_files[:5]]}")

    # Проверяем соответствие имен
    if image_files and annotation_files:
        image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
        annotation_names = {os.path.splitext(os.path.basename(f))[0] for f in annotation_files}

        common_names = image_names & annotation_names
        print(f"Common files (with both image and annotation): {len(common_names)}")

        if common_names:
            print(f"First 5 common names: {list(common_names)[:5]}")

        # Проверяем содержимое аннотаций
        if annotation_files:
            sample_annotation = annotation_files[0]
            print(f"\nSample annotation file: {os.path.basename(sample_annotation)}")
            with open(sample_annotation, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    print(f"Number of objects: {len(lines)}")
                    print(f"First 3 lines: {lines[:3]}")
                else:
                    print("Annotation file is empty")


if __name__ == "__main__":
    check_data_structure("./images", "./annotations")